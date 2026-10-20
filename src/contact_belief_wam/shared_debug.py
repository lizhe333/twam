"""Bounded full-model CUDA checks for the shared-teacher training pipeline."""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from .cloud import model_config
from .data import ContactDataset, collate_contact
from .paths import atomic_json, sha256
from .preprocessing import PromptTokenizer
from .sequence import SequenceTrainingModel, observation_at, prepare_batch
from .shared_pipeline import stage_run
from .trainer import export_model, optimizer_for, validation


def debug_optimizer(policy, run, device):
    ordinary = optimizer_for(policy, run)
    if device == "cuda":
        return ordinary
    from .debug_optimizer import CPUOffloadedAdamW
    groups = [{**group, "params": list(group["params"])} for group in ordinary.param_groups]
    # AdamW's generated groups include foreach/fused=None. The CPU implementation
    # uses one tensor at a time to avoid an optimizer-sized temporary allocation.
    for group in groups:
        group.update(foreach=False, fused=False)
    del ordinary
    return CPUOffloadedAdamW(groups)


def choose_sample(dataset, task, source):
    cfg = dataset.config
    for index, (rid, start, stop, frame) in enumerate(dataset.index):
        row = dataset.rows[rid]
        if row["task"] != task or row["source"] != source or frame < start + 4 * cfg.k or frame + cfg.k >= stop:
            continue
        if not (dataset.cache_root / (row["episode_id"] + ".npz")).is_file():
            continue
        # Prefer recovery windows that actually span a masked forced action.
        if source == "recovery":
            sample = dataset[(index, 0)]
            if not (sample["action_mask"] & ~sample["policy_action_mask"]).any():
                continue
            if sample["policy_action_mask"][-1].sum() == 0:
                continue
            return sample
        return dataset[(index, 0)]
    raise ValueError(f"No cached, valid debug window for {task}/{source}")


def state_signature(state):
    """Summarize optimizer tensors on CPU without allocating a second state copy."""
    values = []
    def visit(value):
        if torch.is_tensor(value):
            flat = value.detach().cpu().reshape(-1)
            values.append([str(value.dtype), list(value.shape), float(flat.sum(dtype=torch.float64)),
                           float(flat.abs().sum(dtype=torch.float64))])
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
    visit(state)
    return values


def debug_worker(run, stage, updates, device, optimizer_device, threads):
    from .model import ContactWAM
    from .weights import initialize
    if not 1 <= updates <= 2:
        raise ValueError("Debug is limited to 1–2 updates per model")
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This requested 4090 debug requires visible CUDA")
    torch.set_num_threads(threads)
    torch.manual_seed(run["seed"])
    np.random.seed(run["seed"])
    random.seed(run["seed"])
    output = Path(run["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "debug_report.json").exists():
        raise FileExistsError(f"Debug already has a report: {output}")
    report = {"complete": False, "formal_training": False, "stage": stage, "task": run["task"],
              "device": torch.cuda.get_device_name(), "optimizer_device": optimizer_device,
              "updates": [], "shared_stats_sha256": sha256(run["stats"])}
    started = time.monotonic()
    try:
        cfg = model_config(run, stage)
        tokenizer = PromptTokenizer(run["tokenizer_path"])
        data_args = dict(manifest=run["manifest"], data_root=run["data_root"], cache_root=run["tactile_cache"],
                         stats=run["stats"], config=cfg, policy_action_mask=True)
        dataset = ContactDataset(**data_args)
        samples = ([choose_sample(dataset, task, "original") for task in run["shared_tasks"]]
                   if stage == "stage1" else [choose_sample(dataset, run["task"], "original")])
        recovery = [choose_sample(dataset, task, "recovery") for task in
                    (run["policy_tasks"] if stage == "stage1" else [run["task"]])]
        if stage == "stage1":
            # Two insertion tasks have two examples each, so task-preserving
            # action ranking is exercised alongside all eight task forwards.
            samples += recovery
        policy, initialization = initialize(cfg, world_init=run["world_init"],
                                             stage1_checkpoint=run.get("stage1_checkpoint"))
        report["initialization"] = initialization
        if stage == "stage2":
            previous_stats = Path(run["stage1_checkpoint"]) / "quantiles.json"
            if sha256(previous_stats) != report["shared_stats_sha256"]:
                raise AssertionError("Stage 1 and task policy statistics differ")
        policy.to(device).train()
        wrapper = SequenceTrainingModel(policy)
        optimizer = debug_optimizer(policy, run, optimizer_device)
        horizon = run["stage1_steps"] if stage == "stage1" else run["stage2_steps"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .5 * (1 + math.cos(math.pi * step / horizon)))
        frozen = {name: p._version for name, p in policy.named_parameters() if not p.requires_grad}
        report["trainable_parameters"] = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        report["frozen_parameters"] = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
        autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.dtype == "bfloat16")
        torch.cuda.reset_peak_memory_stats()
        for update in range(updates):
            raw = collate_contact(samples if update == 0 else recovery)
            batch = prepare_batch(raw, tokenizer, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                result = wrapper(batch)
            if not torch.isfinite(result["loss"]):
                raise FloatingPointError("Nonfinite loss")
            result["loss"].backward()
            missing, gradients = [], {}
            for name, parameter in policy.named_parameters():
                if not parameter.requires_grad:
                    if parameter.grad is not None:
                        raise AssertionError(f"Frozen gradient: {name}")
                    continue
                if parameter.grad is None:
                    if name.startswith("contact."):
                        missing.append(name)
                    continue
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"Nonfinite gradient {name}")
                key = name.split(".")[1] if name.startswith("contact.") else "policy_or_world"
                gradients[key] = gradients.get(key, 0.) + float(parameter.grad.norm(dtype=torch.float32))
            if missing:
                raise AssertionError(f"Missing active contact gradients: {missing}")
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), run["grad_clip"], foreach=False)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient norm")
            optimizer.step()
            scheduler.step()
            policy.optimizer_step.add_(1)
            if any(p._version != frozen[name] for name, p in policy.named_parameters() if name in frozen):
                raise AssertionError("Frozen parameter changed")
            trace = {"update": update + 1, "loss": float(result["loss"].detach()), "grad_norm": float(norm),
                     "tasks": raw["task"], "sources": raw["source"], "gradient_norm_by_module": gradients,
                     "masked_forced_action_targets": int((raw["action_mask"] & ~raw["policy_action_mask"]).sum()),
                     "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                     "frozen_parameters_unchanged": True}
            report["updates"].append(trace)
            print(json.dumps({"event": "debug_update", **trace}), flush=True)
            optimizer.zero_grad(set_to_none=True)
            del result
        # Fixed-noise deployment output, then a full model/optimizer/scheduler/RNG round trip.
        policy.eval()
        obs = observation_at(batch, batch["state"].shape[1] - 1)
        noise = torch.randn(len(raw["source"]), cfg.chunk_size, 32, device=device)
        def predict(model):
            with torch.no_grad(), autocast():
                if stage == "stage2":
                    prediction = model.predict_action_chunk(obs, noise=noise)
                    return torch.cat([prediction.actions.flatten(1), prediction.belief.flatten(1),
                                      prediction.expected_contact.flatten(1)], 1).cpu()
                state = model.history_step(obs, batch["actions"][:, -1])
                return torch.cat([v.flatten(1) for v in state if v is not None], 1).cpu()
        before = predict(policy)
        checkpoint = output / "pretrained_model"
        export_model(policy, checkpoint, run)
        signature = state_signature(optimizer.state_dict()["state"])
        state_path = output / "debug_training.pt"
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "step": int(policy.optimizer_step), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": np.random.get_state(),
                    "python_rng": random.getstate()}, state_path)
        expected_rng = torch.rand(4)
        expected_cuda_rng = torch.rand(4, device=device).cpu()
        del optimizer, scheduler, wrapper, policy
        gc.collect()
        torch.cuda.empty_cache()
        restored = ContactWAM.from_pretrained(checkpoint, stage=stage, device=device).eval()
        restored_optimizer = debug_optimizer(restored, run, optimizer_device)
        restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda step: .5 * (1 + math.cos(math.pi * step / horizon)))
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        restored_optimizer.load_state_dict(state["optimizer"])
        restored_scheduler.load_state_dict(state["scheduler"])
        if state_signature(restored_optimizer.state_dict()["state"]) != signature:
            raise AssertionError("Optimizer state mismatch")
        if int(restored.optimizer_step) != state["step"] or restored_scheduler.last_epoch != updates:
            raise AssertionError("Step/scheduler mismatch")
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        torch.testing.assert_close(torch.rand(4), expected_rng, rtol=0, atol=0)
        torch.testing.assert_close(torch.rand(4, device=device).cpu(), expected_cuda_rng, rtol=0, atol=0)
        after = predict(restored)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        evaluation_split = "val" if run.get("val_every", 0) > 0 else "train"
        val = ContactDataset(**data_args, split=evaluation_split)
        # Full shared validation is run by the cloud trainer. Debug validates
        # each insertion task with a cached original and recovery window.
        selected_rows = [r for r in val.rows if (val.cache_root / (r["episode_id"] + ".npz")).is_file()]
        selected_ids = {r["episode_id"] for r in selected_rows}
        valid_indices = [i for i, (rid, *_rest) in enumerate(val.index) if val.rows[rid]["episode_id"] in selected_ids]
        metrics = {}
        for source in ("original", "recovery"):
            index = next(i for i in valid_indices if val.sources[i] == source)
            validation_raw = collate_contact([val[(index, 0)]])
            with torch.no_grad(), autocast():
                value = SequenceTrainingModel(restored)(prepare_batch(validation_raw, tokenizer, device))
            metrics[source] = {key: float(v["sum"].sum() / v["count"].sum().clamp_min(1)) for key, v in value["metrics"].items()}
        report.update(complete=True, checkpoint=str(checkpoint), optimizer_restored=True, scheduler_restored=True,
                      rng_restored=True, checkpoint_max_abs_difference=float((before - after).abs().max()),
                      evaluation_split=evaluation_split, validation=metrics, seconds=time.monotonic() - started)
        return report
    except Exception:
        import traceback
        report["error"] = traceback.format_exc()
        raise
    finally:
        atomic_json(output / "debug_report.json", report)


def run_debug_pipeline(run, *, updates=2, device="cuda", optimizer_device="cpu", threads=8):
    output = Path(run["output"])
    if (output / "pipeline_config.yaml").exists():
        raise ValueError("Debug must use a separate --output, never a formal training directory")
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    teacher = output / "shared_wm/pretrained_model"
    cases = [("stage1", None)] + [("stage2", task) for task in run["policy_tasks"]]
    for stage, task in cases:
        case = stage_run(run, stage, task, teacher, output)
        # This memory/compute tradeoff belongs to the 24GB developer debug only.
        # Formal A800 configs remain untouched and disable checkpointing.
        case["gradient_checkpointing"] = True
        directory = Path(case["output"])
        directory.mkdir(parents=True, exist_ok=True)
        config = directory / "debug_run.yaml"
        # Resume a bounded debug with its original inputs even if the formal
        # recipe changes meanwhile. Never mix statistics across its teacher/tasks.
        if config.exists():
            case = yaml.safe_load(config.read_text())
        else:
            config.write_text(yaml.safe_dump(case, sort_keys=False))
        report_file = directory / "debug_report.json"
        if report_file.exists() and json.loads(report_file.read_text()).get("complete"):
            results[task or "shared_wm"] = json.loads(report_file.read_text())
            continue
        if report_file.exists():
            raise RuntimeError(f"Prior incomplete debug at {directory}; inspect it before repeating updates")
        command = [sys.executable, "-u", "-m", "contact_belief_wam.shared_debug", "--worker-config", str(config),
                   "--stage", stage, "--updates", str(updates), "--device", device,
                   "--optimizer-device", optimizer_device, "--threads", str(threads)]
        print(json.dumps({"event": "debug_start", "stage": stage, "task": task, "command": command}), flush=True)
        with (directory / "debug.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        results[task or "shared_wm"] = json.loads(report_file.read_text())
        atomic_json(output / "debug_summary.json", {"formal_training": False, "complete": False, "models": results})
    summary = {"formal_training": False, "complete": True, "models": results}
    atomic_json(output / "debug_summary.json", summary)
    print(json.dumps({"event": "debug_complete", "report": str(output / "debug_summary.json")}), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker-config", required=True)
    p.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    p.add_argument("--updates", type=int, choices=[1, 2], default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--optimizer-device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    debug_worker(yaml.safe_load(Path(args.worker_config).read_text()), args.stage, args.updates,
                 args.device, args.optimizer_device, args.threads)


if __name__ == "__main__":
    main()
