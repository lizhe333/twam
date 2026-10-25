from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from ..cloud import complete_checkpoints, model_config
from .data import ContactDataset, MixedSequenceSampler, collate_contact
from ..paths import atomic_json, sha256
from ..metrics import Metrics
from ..timing import TrainingTiming
from ..preprocessing import PromptTokenizer
from ..sequence import SequenceTrainingModel, prepare_batch


def optimizer_for(policy, run, *, device=None):
    new, pretrained = [], []
    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        (new if name.startswith("contact.") or any(part in name for part in (".world_", ".wam_")) else pretrained).append(param)
    groups = []
    if new:
        groups.append({"params": new, "lr": run["lr_new"], "name": "new"})
    if pretrained:
        groups.append({"params": pretrained, "lr": run["lr_policy"], "name": "policy"})
    if device is None:
        device = next((param.device for param in policy.parameters()), torch.device("cpu"))
    else:
        device = torch.device(device)
    optimizer_options = {"betas": (.9, .95), "eps": 1e-8, "weight_decay": .01}
    if run.get("fused_adamw", False) and device.type == "cuda" and torch.cuda.is_available():
        # The formal policy is still on CPU here. Accelerator.prepare moves the
        # same Parameter objects to its CUDA device before the first optimizer
        # step, so the fused AdamW option must be selected from the target
        # device passed by train(), rather than the current parameter device.
        optimizer_options.update(fused=True, foreach=None)
    return torch.optim.AdamW(groups, **optimizer_options)


def training_contract(run):
    """Inputs that must remain identical for an exact optimizer/sampler resume."""
    defaults = {"task_balanced": False, "policy_action_mask": False, "stage2_original_steps": 0,
                "data_mix": "1:1", "seed": 42, "fused_adamw": False}
    values = {key: run.get(key, defaults.get(key)) for key in
              ("num_gpus", "global_batch", "micro_batch", "accumulation_steps", "seed", "data_mix",
               "task_balanced", "policy_action_mask", "stage2_original_steps", "stage1_steps", "stage2_steps",
               "lr_new", "lr_policy", "grad_clip", "fused_adamw")}
    values.update(manifest_sha256=sha256(run["manifest"]), stats_sha256=sha256(run["stats"]))
    return values


def check_resume_contract(run, metadata):
    saved = metadata.get("training_contract")
    if saved is not None:
        saved = dict(saved)
        # Older checkpoints predate the fused optimizer option and therefore
        # used the non-fused AdamW path.
        saved.setdefault("fused_adamw", False)
        current = training_contract(run)
        changed = [key for key in set(saved) | set(current) if saved.get(key) != current.get(key)]
        if changed:
            raise ValueError(f"Exact-resume data/sampling configuration changed: {sorted(changed)}; start a new run")


def validation(wrapper, dataset, tokenizer, device, batches, micro_batch, autocast):
    was_training = wrapper.training
    wrapper.eval()
    metrics = Metrics()
    if getattr(dataset, "tasks", None) is not None:
        groups = {(task, source): np.flatnonzero((np.asarray(dataset.tasks) == task) &
                                                (np.asarray(dataset.sources) == source))
                  for task in sorted(set(dataset.tasks)) for source in ("original", "recovery")}
    else:
        groups = {source: np.flatnonzero(np.asarray(dataset.sources) == source) for source in ("original", "recovery")}
    groups = {key: value for key, value in groups.items() if len(value)}
    indices = []
    # Fixed, source-balanced observations: validation can be compared across steps.
    for values in groups.values():
        if len(values):
            positions = np.linspace(0, len(values) - 1, max(1, batches * micro_batch // len(groups)), dtype=int)
            indices.extend(values[positions].tolist())
    with torch.no_grad():
        for start in range(0, len(indices), micro_batch):
            raw = collate_contact([dataset[(idx, 12345 + idx)] for idx in indices[start:start + micro_batch]])
            batch = prepare_batch(raw, tokenizer, device, include_language=wrapper.policy.config.stage != "stage1")
            with autocast():
                result = wrapper(batch)
            metrics.add(result, raw["source"], raw.get("task"))
    wrapper.train(was_training)
    return metrics.result()


def export_model(policy, path, run):
    path = Path(path)
    policy.save_pretrained(path)
    shutil.copy2(run["stats"], path / "quantiles.json")
    # Export can be moved independently of the training host. No base checkpoint
    # is needed for ContactWAM.from_pretrained; all required tensors are saved.
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(run["tokenizer_path"], local_files_only=True).save_pretrained(path / "tokenizer")
    atomic_json(path / "deployment.json", {"camera": "head", "action": "absolute_first8_next_frame",
                "physical_steps_per_action": 2, "encoder_reference": policy.config.encoder_reference,
                "tactile_offsets": [-6, -4, -2, 0], "task": run["task"]})


def save_checkpoint(accelerator, wrapper, run, stage, step, draws):
    path = Path(run["output"]) / stage / "checkpoints" / f"step_{step:08d}"
    accelerator.wait_for_everyone()
    accelerator.save_state(str(path / "training"))
    accelerator.wait_for_everyone()  # All rank RNG/state files must exist before the completion marker.
    if accelerator.is_main_process:
        policy = accelerator.unwrap_model(wrapper).policy
        export_model(policy, path / "pretrained_model", run)
        metadata = {"stage": stage, "step": step, "draws_per_rank": draws,
                    "num_gpus": accelerator.num_processes, "complete": True,
                    "training_contract": training_contract(run)}
        atomic_json(path / "complete.json", metadata)
        atomic_json(Path(run["output"]) / "run_state.json", {**metadata, "checkpoint": str(path)})
    accelerator.wait_for_everyone()
    return path


def train(run, stage, steps, resume=None):
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from accelerate.utils import set_seed
    from ..model import ContactWAM
    from ..weights import initialize

    if run.get("device", "cuda") != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal train entry requires CUDA; use debug for the bounded developer-machine checks")
    # This entry is intended for the manually submitted A800 job. It is never
    # invoked by prepare/debug/serve/eval.
    accelerator = Accelerator(gradient_accumulation_steps=run["accumulation_steps"],
                              mixed_precision=run["precision"],
                              step_scheduler_with_optimizer=False,
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    if accelerator.num_processes != run["num_gpus"]:
        raise ValueError("Launched process count differs from the run configuration")
    set_seed(run["seed"])
    cfg = model_config(run, stage)
    start, draws = 0, 0
    if resume:
        resume = Path(resume)
        meta = json.loads((resume / "complete.json").read_text())
        if meta["stage"] != stage:
            raise ValueError("Resume checkpoint stage does not match")
        if meta["num_gpus"] != accelerator.num_processes:
            raise ValueError("Exact optimizer/RNG resume requires the original process count")
        check_resume_contract(run, meta)
        policy = ContactWAM.from_pretrained(resume / "pretrained_model", stage=stage)
        # Detect accidental action/timing changes on exact resume.
        if policy.config.to_dict() != cfg.to_dict():
            raise ValueError("Model configuration changed; start a new run to change k or mode")
        start, draws = meta["step"], meta["draws_per_rank"]
        report = None
    else:
        stage1 = complete_checkpoints(run["output"], "stage1")
        handoff = stage1[-1][1] / "pretrained_model" if stage == "stage2" and cfg.use_world and stage1 else None
        if stage == "stage2" and cfg.use_world and run.get("stage1_checkpoint"):
            handoff = Path(run["stage1_checkpoint"])
            # A shared WM must see exactly the same physical action normalization
            # in its task policies. Never reinterpret per-task normalized values.
            saved_stats = handoff / "quantiles.json"
            if not saved_stats.is_file() or json.loads(saved_stats.read_text()) != json.loads(Path(run["stats"]).read_text()):
                raise ValueError("Stage-1 and policy normalization statistics differ")
        policy, report = initialize(cfg, world_init=run["world_init"], stage1_checkpoint=handoff)
    wrapper = SequenceTrainingModel(policy)
    optimizer = optimizer_for(policy, run, device=accelerator.device)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .5 * (1 + math.cos(math.pi * min(step, steps) / max(1, steps))))
    wrapper, optimizer, scheduler = accelerator.prepare(wrapper, optimizer, scheduler)
    if resume:
        accelerator.load_state(str(resume / "training"))
    else:
        set_seed(run["seed"] + accelerator.process_index)
    tokenizer = PromptTokenizer(run["tokenizer_path"])
    dataset_args = dict(manifest=run["manifest"], data_root=run["data_root"], cache_root=run["tactile_cache"],
                        stats=run["stats"], config=cfg, seed=run["seed"],
                        policy_action_mask=run.get("policy_action_mask", False))
    dataset = ContactDataset(**dataset_args)
    val = (ContactDataset(**dataset_args, split="val")
           if accelerator.is_main_process and run["val_every"] > 0 else None)
    ratio = tuple(float(v) for v in run["data_mix"].split(":"))
    if len(ratio) != 2:
        raise ValueError("--data-mix expects original:recovery")
    micro, accum = run["micro_batch"], run["accumulation_steps"]
    sampler = MixedSequenceSampler(dataset, draws=(steps - start) * micro * accum, ratio=ratio,
                                   seed=run["seed"], rank=accelerator.process_index, start_draw=draws,
                                   task_balanced=run.get("task_balanced", False),
                                   original_only_draws=(run.get("stage2_original_steps", 0) * micro * accum
                                                        if stage == "stage2" else 0))
    loader = DataLoader(dataset, batch_size=micro, sampler=sampler, collate_fn=collate_contact,
                        num_workers=run.get("workers", 2), pin_memory=True,
                        generator=torch.Generator().manual_seed(run["seed"] + accelerator.process_index))
    writer = None
    log_dir = Path(run["output"]) / "logs"
    if accelerator.is_main_process:
        from torch.utils.tensorboard import SummaryWriter
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir / "tensorboard"))
        if report is not None:
            atomic_json(Path(run["output"]) / f"initialization_{stage}.json", report)
            atomic_json(Path(run["output"]) / "initialization_report.json", report)
    wrapper.train()
    optimizer.zero_grad(set_to_none=True)
    interval_metrics, interval_time, interval_sequences = Metrics(), time.monotonic(), 0
    step = start
    timing = TrainingTiming(enabled=accelerator.is_main_process)
    data_requested_at = time.monotonic()
    for raw in loader:
        timing.add_wall("data_wait", time.monotonic() - data_requested_at)
        timing.micro_batches += 1
        with timing.wall("prepare"):
            batch = prepare_batch(raw, tokenizer, accelerator.device, include_language=stage != "stage1")
        with accelerator.accumulate(wrapper):
            with timing.cuda("forward"), accelerator.autocast():
                result = wrapper(batch)
            if not torch.isfinite(result["loss"]):
                raise FloatingPointError(f"Nonfinite loss at {stage} step {step}")
            with timing.cuda("backward"):
                accelerator.backward(result["loss"])
            grad_norm = None
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(wrapper.parameters(), run["grad_clip"])
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("Nonfinite gradient norm")
            with timing.cuda("optimizer"):
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        draws += len(raw["source"])
        interval_sequences += len(raw["source"])
        if accelerator.is_main_process:
            with timing.wall("metrics_host"):
                interval_metrics.add(result, raw["source"], raw.get("task"))
        if not accelerator.sync_gradients:
            data_requested_at = time.monotonic()
            continue
        step += 1
        accelerator.unwrap_model(wrapper).policy.optimizer_step.fill_(step)
        if accelerator.is_main_process and (step % 10 == 0 or step == start + 1 or step == steps):
            metrics = {"train/" + k: v for k, v in interval_metrics.result().items()}
            metrics.update(grad_norm=float(grad_norm), sequences_per_second=interval_sequences * accelerator.num_processes / (time.monotonic() - interval_time),
                           gpu_memory_gib=torch.cuda.max_memory_allocated() / 2**30)
            metrics.update(timing.result_and_reset())
            metrics["recovery_sampling_enabled"] = float(stage != "stage2" or step > run.get("stage2_original_steps", 0))
            for group in optimizer.param_groups:
                metrics[f"lr/{group['name']}"] = group["lr"]
            write_metrics(log_dir, writer, stage, step, metrics)
            interval_metrics, interval_time, interval_sequences = Metrics(), time.monotonic(), 0
        if run["val_every"] > 0 and (step % run["val_every"] == 0 or step == steps):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                values = validation(accelerator.unwrap_model(wrapper), val, tokenizer, accelerator.device,
                                    run.get("val_batches", 8), micro, accelerator.autocast)
                write_metrics(log_dir, writer, stage, step, {"val/" + k: v for k, v in values.items()})
            accelerator.wait_for_everyone()
        if step % run["save_every"] == 0 or step == steps:
            save_checkpoint(accelerator, wrapper, run, stage, step, draws)
        data_requested_at = time.monotonic()
    if stage == "stage2" and accelerator.is_main_process:
        export_model(accelerator.unwrap_model(wrapper).policy, Path(run["output"]) / "export/final", run)
    if writer:
        writer.close()
    accelerator.wait_for_everyone()
    accelerator.end_training()


def write_metrics(log_dir, writer, stage, step, metrics):
    row = {"stage": stage, "step": step, "time": time.time(), **metrics}
    with (log_dir / "metrics.jsonl").open("a") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")
    with (log_dir / "train.log").open("a") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")
    for key, value in metrics.items():
        writer.add_scalar(f"{stage}/{key}", value, step)
    print(json.dumps(row), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run-config", required=True)
    p.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--resume")
    args = p.parse_args(argv)
    train(yaml.safe_load(Path(args.run_config).read_text()), args.stage, args.steps, args.resume)


if __name__ == "__main__":
    main()
