"""Distributed trainer for shared contact extensions and task policies.

The LR schedule and FP32 EMA helpers in this file follow the local WAVE-0
reference in ``wave0-univtac-fast/src/wave0_univtac/training.py``. The aligned
data contract and ContactWAM objective remain implemented by this project.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
import yaml

from .metrics import Metrics
from .ema import EMA, ShardedEMA
from . import optimizer_shards
from .paths import PROJECT, atomic_json, read_jsonl, sha256
from .timing import TrainingTiming


def lr_at(step: int, peak: float, final: float, warmup: int, steps: int) -> float:
    """Linear warmup followed by cosine decay, matching the WAVE-0 reference."""
    if step < warmup:
        start = peak / (warmup + 1)
        return start + (peak - start) * step / max(1, warmup)
    fraction = min(1.0, (step - warmup) / max(1, steps - warmup))
    return final + (peak - final) * 0.5 * (1 + math.cos(math.pi * fraction))


def parameter_group(name: str) -> str:
    """The WAVE-0 learned future/world path uses the new-parameter LR."""
    if name.startswith("contact.updater.") or ".contact.updater." in name:
        return "belief"
    if name.startswith("contact.") or ".contact." in name:
        return "new"
    if any(part in name for part in (".wam_", ".world_")):
        return "new"
    return "policy"


def make_optimizer(policy, run: dict, stage: str, device=None):
    cfg = run[stage]
    if stage in {"tactile", "stage1"}:
        trainable = [(name, p) for name, p in policy.named_parameters()
                     if p.requires_grad and (".world_" in name or name.startswith("world_")
                                             or name.startswith("contact."))]
        if "lr_world" in cfg:
            groups = []
            for group, prefix, peak in (("world", False, cfg["lr_world"]),
                                         ("contact", True, cfg["lr"])):
                params = [p for name, p in trainable if name.startswith("contact.") == prefix]
                if params:
                    groups.append({"params": params, "lr": float(peak), "peak": float(peak), "name": group})
        else:
            groups = [{"params": [p for _, p in trainable],
                       "lr": float(cfg["lr"]), "peak": float(cfg["lr"]), "name": "world"}]
    else:
        values = {"policy": [], "new": [], "belief": []}
        for name, parameter in policy.named_parameters():
            if parameter.requires_grad:
                values[parameter_group(name)].append(parameter)
        groups = []
        for name in ("policy", "new", "belief"):
            if values[name]:
                peak = float(cfg.get("lr_belief", cfg["lr"]) if name == "belief"
                             else cfg["lr_new"] if name == "new" else cfg["lr"])
                groups.append({"params": values[name], "lr": peak, "peak": peak, "name": name})
    if not groups or any(not group["params"] for group in groups):
        raise ValueError(f"No trainable parameters for {stage}")
    target_device = torch.device(device or next(policy.parameters()).device)
    options = {"betas": (0.9, 0.95), "eps": 1e-8,
               "weight_decay": float(cfg["weight_decay"])}
    if bool(run.get("fused_adamw", True)) and target_device.type == "cuda":
        options.update(fused=True, foreach=None)
    else:
        options["foreach"] = False
    if stage == "stage2" and cfg.get("shard_optimizer_state", False):
        return optimizer_shards.make_sharded_adamw(groups, **options)
    return torch.optim.AdamW(groups, **options)


def set_lr(optimizer, cfg: dict, stage: str, step: int) -> None:
    for group in optimizer.param_groups:
        if stage in {"tactile", "stage1"}:
            final = float(cfg["final_lr"]) * (float(group["peak"]) / float(cfg["lr"])
                                             if "lr_world" in cfg else 1.0)
        else:
            # Keep the new-parameter LR at the same ratio to the policy LR.
            final = float(cfg["final_lr"]) * float(group["peak"]) / float(cfg["lr"])
        group["lr"] = lr_at(step, float(group["peak"]), final,
                            int(cfg["warmup"]), int(cfg["steps"]))



def _cuda_memory(device):
    free, total = torch.cuda.mem_get_info(device)
    return {"allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
            "reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "free_gib": free / 2**30, "total_gib": total / 2**30}


def _memory_event(output, stage, rank, device, phase, step, **details):
    row = {"event": "cuda_memory", "stage": stage, "rank": rank,
           "phase": phase, "step": step, "time": time.time(),
           **_cuda_memory(device), **details}
    with (Path(output) / f"memory.rank{rank:02d}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(row, allow_nan=False), flush=True)


@contextmanager
def _memory_phase(output, stage, rank, device, phase, step, *, trace):
    """Trace the first updates and identify the phase of any later OOM."""
    try:
        yield
    except torch.OutOfMemoryError:
        _memory_event(output, stage, rank, device, phase, step, oom=True)
        raise
    else:
        if trace:
            _memory_event(output, stage, rank, device, phase, step)


def rng_state():
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"] is not None:
        # Flush queued manual_seed callbacks before restoring a checkpoint in
        # a fresh process. Otherwise lazy CUDA initialization can overwrite
        # this state with an earlier queued seed on the first CUDA operation.
        torch.cuda.init()
        torch.cuda.set_rng_state(state["cuda"])


def _json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def training_contract(run: dict, stage: str, world_size: int) -> dict:
    """The data, sampler, optimizer and model values required for exact resume."""
    cfg = run[stage]
    manifest = run["shared_manifest"] if stage in {"tactile", "stage1"} else run["task_manifest"]
    stats = run["wm_stats"] if stage in {"tactile", "stage1"} else run["task_stats"]
    task_stats = (run.get("task_stats_by_task", run.get("task_stats"))
                  if stage in {"tactile", "stage1"} else run.get("task_stats"))
    task_stats_hash = ({task: sha256(path) for task, path in sorted(task_stats.items())}
                       if isinstance(task_stats, dict) else sha256(task_stats))
    contract = {
        "schema_version": 1, "stage": stage, "task": run["task"],
        "variant": run["variant"], "mode": run["mode"], "world_size": int(world_size),
        "global_batch": int(cfg["global_batch"]), "micro_batch": int(cfg["micro_batch"]),
        "accumulation_steps": int(cfg["global_batch"] // (cfg["micro_batch"] * world_size)),
        "seed": int(run["seed"]), "data_mix": run.get("data_mix", "1:1"),
        "expert_only": bool(run.get("expert_only", False)),
        "steps": int(cfg["steps"]), "lr": float(cfg["lr"]),
        "lr_new": float(cfg.get("lr_new", cfg["lr"])), "warmup": int(cfg["warmup"]),
        "lr_belief": float(cfg.get("lr_belief", cfg["lr"])),
        "final_lr": float(cfg["final_lr"]), "weight_decay": float(cfg["weight_decay"]),
        "grad_clip": float(run["grad_clip"]), "fused_adamw": bool(run.get("fused_adamw", True)),
        "gradient_checkpointing": bool(run.get("gradient_checkpointing", True)),
        "manifest_sha256": sha256(manifest), "task_stats_sha256": task_stats_hash,
        "wm_stats_sha256": sha256(run["wm_stats"]),
        "tokenizer_sha256": sha256(_tokenizer_file(run["tokenizer_path"])),
    }
    if stage in {"tactile", "stage1"}:
        if "lr_world" in cfg:
            contract["lr_world"] = float(cfg["lr_world"])
        contract["episode_fingerprint"] = run.get("episode_fingerprint")
        contract["validate_every"] = int(cfg["validate_every"])
        if contract["validate_every"] < 0:
            raise ValueError("validate_every must be nonnegative")
        if contract["validate_every"]:
            contract.update(selection_metric=cfg.get("selection_metric", "tactile_future" if stage == "tactile" else "world_total"),
                            validation_samples_per_task=int(cfg["validation_samples_per_task"]),
                            early_stop_patience=int(cfg["early_stop_patience"]),
                            early_stop_min_steps=int(cfg["early_stop_min_steps"]))
        else:
            contract["checkpoint_selection"] = "final"
    else:
        contract["episode_fingerprint"] = run.get("policy_fingerprints", {}).get(run["task"])
        contract["stage1_checkpoint"] = run.get("stage1_checkpoint_identity")
        contract["shard_optimizer_state"] = bool(cfg.get("shard_optimizer_state", False))
        contract["ema_device"] = cfg.get("ema_device", "cuda")
        if not run.get("ema_enabled", True):
            contract["ema_enabled"] = False
        if cfg.get("shard_ema", False):
            contract["shard_ema"] = True
        if run.get("batch_stage2_policy", False):
            contract["batch_stage2_policy"] = True
    if run.get("finetune_from"):
        contract.update(finetune_from=run["finetune_from"],
                        finetune_use_ema=run["finetune_use_ema"],
                        policy_supervised_only=run.get("policy_supervised_only", False))
        if run.get("finetune_stage2_only"):
            contract["finetune_stage2_only"] = True
    return contract


def _tokenizer_file(path) -> Path:
    path = Path(path)
    if path.is_dir():
        candidates = (path / "tokenizer.model", path / "paligemma_tokenizer.model")
        path = next((value for value in candidates if value.is_file()), candidates[0])
    return path


def latest_checkpoint(output, stage):
    paths = []
    root = Path(output) / stage / "checkpoints"
    for marker in root.glob("step_*/complete.json"):
        try:
            metadata = json.loads(marker.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        paths.append((int(metadata["step"]), marker.parent))
    return max(paths, default=(0, None), key=lambda value: value[0])[1]


def _model_state(policy):
    from safetensors.torch import load_file
    return load_file(str(Path(policy) / "model.safetensors"), device="cpu")


def _export(policy, path, run, *, ema=None):
    from .paths import atomic_json
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(path)
    task_stats = run.get("task_stats")
    if isinstance(task_stats, dict):
        task_stats = task_stats.get(run["task"])
    if not task_stats:
        task_stats = run["wm_stats"]
    shutil.copy2(task_stats, path / "norm_stats.json")
    shutil.copy2(run["wm_stats"], path / "wm_norm_stats.json")
    tokenizer = _tokenizer_file(run["tokenizer_path"])
    if not tokenizer.is_file():
        raise FileNotFoundError(f"Tokenizer model is missing: {tokenizer}")
    shutil.copy2(tokenizer, path / "tokenizer.model")
    atomic_json(path / "deployment.json", {
        "task": run["task"], "variant": run["variant"], "mode": run["mode"],
        "aligned": True, "camera": policy.config.camera_mode,
        "simulator_action": "absolute_joint",
        "model_action": "first7_chunk_relative_absolute_gripper",
        "action_chunk_length": 50, "executed_action_prefix": 10,
        "normalization": "task_and_shared_world_quantiles", "use_ema": ema is not None,
        "ema_file": "ema.safetensors" if ema is not None else None,
    })
    if ema is not None and not ema.is_sharded:
        ema.save(path / "ema.safetensors")


def _atomic_checkpoint(policy, optimizer, ema, run, stage, step, contract, identity,
                       world_size, rank, rng_states, draws, selection):
    output = Path(run["output"])
    path = output / stage / "checkpoints" / f"step_{step:08d}"
    pending = path.with_name(path.name + ".partial")
    sharded = optimizer_shards.is_sharded(optimizer)
    optimizer_format = optimizer_shards.FORMAT if sharded else optimizer_shards.DENSE_FORMAT
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Will not overwrite complete checkpoint {path}")
        if pending.exists():
            pending.rename(pending.with_name(pending.name + f".abandoned-{time.time_ns()}"))
        pending.mkdir(parents=True)
    if sharded:
        dist.barrier()
        optimizer_shards.save_local_state(optimizer, policy, pending, rank, world_size)
        # No complete marker or rename until ALL optimizer shards exist.
        dist.barrier()
    if ema is not None and ema.is_sharded:
        ema.save_collective(pending / "pretrained_model" / "ema.safetensors")
    if rank == 0:
        _export(policy, pending / "pretrained_model", run, ema=ema)
        torch.save({"optimizer": None if sharded else optimizer.state_dict(),
                    "optimizer_format": optimizer_format, "rng_by_rank": rng_states,
                    "step": int(step), "draws_per_rank": int(draws),
                    "world_size": int(world_size), "selection": selection}, pending / "training.pt")
        atomic_json(pending / "training_contract.json", contract)
        atomic_json(pending / "run_identity.json", identity)
        atomic_json(pending / "complete.json", {
            "stage": stage, "step": int(step), "draws_per_rank": int(draws),
            "world_size": int(world_size), "identity": identity["fingerprint"],
            "optimizer_format": optimizer_format,
            "optimizer_shards_sha256": {
                optimizer_shards.shard_name(index): sha256(pending / optimizer_shards.shard_name(index))
                for index in range(world_size)} if sharded else {},
            "training_contract": contract, "selected_checkpoint": selection.get("best_checkpoint")
            if stage in {"tactile", "stage1"} else None,
            "model_sha256": sha256(pending / "pretrained_model" / "model.safetensors"),
            "training_sha256": sha256(pending / "training.pt"),
            "ema_sha256": (sha256(pending / "pretrained_model" / "ema.safetensors")
                           if ema is not None else None),
        })
        pending.rename(path)
    if world_size > 1:
        dist.barrier()
    return path


def _prune_checkpoints(output, stage, keep=2, protected=None):
    root = Path(output) / stage / "checkpoints"
    candidates = sorted((path.parent for path in root.glob("step_*/complete.json")),
                        key=lambda path: path.name)
    keep_paths = set(candidates[-keep:])
    if protected:
        keep_paths.add(Path(protected))
    for path in candidates:
        if path not in keep_paths:
            shutil.rmtree(path)


def _source_identity(run, stage, contract):
    recipe = {key: value for key, value in run.items()
              if key not in {"output", "workflow_root", "stage1_checkpoint", "stage1_checkpoint_identity", "tactile_checkpoint"}}
    source_hashes = {}
    for path in sorted((PROJECT / "src/contact_belief_wam").glob("*.py")):
        source_hashes[str(path.relative_to(PROJECT))] = sha256(path)
    reference_root = PROJECT / "vendor/aligned_reference"
    source_manifest_path = reference_root / "SOURCE.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    for relative_path, expected_hash in source_manifest["files_sha256"].items():
        reference_path = reference_root / relative_path
        actual_hash = sha256(reference_path)
        if actual_hash != expected_hash:
            raise ValueError(f"Pinned aligned reference changed: {reference_path}")
        source_hashes[str(reference_path.relative_to(PROJECT))] = actual_hash
    source_hashes[str(source_manifest_path.relative_to(PROJECT))] = sha256(source_manifest_path)
    for relative_path in (
        "vendor/SOURCE.json",
        "vendor/wave0/src/lerobot/policies/wave0/modeling_wave0.py",
        "vendor/transformers/src/transformers/models/gemma/modeling_gemma.py",
        "vendor/transformers/src/transformers/models/paligemma/modeling_paligemma.py",
    ):
        path = PROJECT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned modeling source: {path}")
        source_hashes[relative_path] = sha256(path)
    value = {"stage": stage, "recipe": recipe, "contract": contract, "source_hashes": source_hashes}
    if stage == "stage2" and run.get("stage1_checkpoint"):
        teacher = Path(run["stage1_checkpoint"])
        value["teacher_sha256"] = sha256(teacher / "pretrained_model" / "model.safetensors")
        value["teacher_complete"] = json.loads((teacher / "complete.json").read_text())
    from .weights import checkpoint_file
    value["pi05_base_sha256"] = sha256(checkpoint_file(run["pi05_base"]))
    if run.get("finetune_from"):
        source = Path(run["finetune_from"])
        value["finetune_source"] = {name: sha256(source / name) for name in
                                    ("config.json", "model.safetensors", "ema.safetensors")
                                    if name != "ema.safetensors" or run["finetune_use_ema"]}
    if stage == "stage1" and run.get("tactile_checkpoint"):
        value["tactile_teacher_sha256"] = sha256(Path(run["tactile_checkpoint"]) / "pretrained_model/model.safetensors")
    if stage in {"tactile", "stage1"} or not run.get("stage1_checkpoint"):
        value["world_init_sha256"] = sha256(checkpoint_file(run["world_init"]))
    value["fingerprint"] = _json_digest(value)
    return value


def _validate_resume(path, stage, contract, identity, world_size, *, rank=None):
    path = Path(path)
    marker = path / "complete.json"
    if not marker.is_file():
        raise ValueError(f"Incomplete aligned checkpoint: {path}")
    meta = json.loads(marker.read_text())
    if meta.get("stage") != stage:
        raise ValueError("Resume checkpoint stage differs")
    if meta.get("world_size") != world_size:
        raise ValueError("Exact resume requires the same number of ranks")
    saved_contract = meta.get("training_contract")
    if saved_contract != contract:
        changed = sorted(key for key in set(saved_contract or {}) | set(contract)
                         if (saved_contract or {}).get(key) != contract.get(key))
        raise ValueError(f"Exact-resume contract changed: {changed}")
    saved_identity = json.loads((path / "run_identity.json").read_text())
    if saved_identity != identity:
        raise ValueError("Recipe or source identity changed; use a new output directory")
    if sha256(path / "pretrained_model" / "model.safetensors") != meta.get("model_sha256"):
        raise ValueError("Checkpoint model integrity check failed")
    training_state = path / "training.pt"
    if not training_state.is_file() or sha256(training_state) != meta.get("training_sha256"):
        raise ValueError("Checkpoint training-state integrity check failed")
    sharded = bool(contract.get("shard_optimizer_state", False))
    expected_format = optimizer_shards.FORMAT if sharded else optimizer_shards.DENSE_FORMAT
    if meta.get("optimizer_format", optimizer_shards.DENSE_FORMAT) != expected_format:
        raise ValueError("Checkpoint optimizer format differs")
    if sharded:
        shards = meta.get("optimizer_shards_sha256", {})
        expected_names = {optimizer_shards.shard_name(index) for index in range(world_size)}
        if set(shards) != expected_names:
            raise ValueError("Checkpoint optimizer shard manifest is incomplete")
        # Each training rank verifies its own large shard. Standalone audits
        # (rank=None) verify every shard without multiplying read traffic.
        names = expected_names if rank is None else {optimizer_shards.shard_name(rank)}
        for name in names:
            shard = path / name
            if not shard.is_file() or sha256(shard) != shards[name]:
                raise ValueError(f"Checkpoint optimizer shard integrity check failed: {name}")
    if stage == "stage2" and contract.get("ema_enabled", True):
        ema_path = path / "pretrained_model" / "ema.safetensors"
        if not ema_path.is_file() or sha256(ema_path) != meta.get("ema_sha256"):
            raise ValueError("Checkpoint EMA integrity check failed")
    elif stage == "stage2" and meta.get("ema_sha256") is not None:
        raise ValueError("Disabled EMA checkpoint unexpectedly contains EMA weights")
    return meta


def _metric_total(result, key):
    metric = result.get("metrics", {}).get(key)
    if not isinstance(metric, dict) or "sum" not in metric or "count" not in metric:
        raise ValueError(f"The model result has no per-sample `{key}` metric")
    return metric["sum"].sum(), metric["count"].sum()


def _fixed_validation_sampler(dataset, tasks, per_task, world_size, rank, batch_size, seed=1729):
    """Yield deterministic, task-balanced validation indices with same-task negatives."""
    dataset_tasks = list(dataset.tasks)
    selected = {}
    for task_index, task in enumerate(tasks):
        indices = np.flatnonzero(np.asarray(dataset_tasks) == task)
        if len(indices) < per_task:
            raise ValueError(f"Validation task {task} has {len(indices)} samples; needs {per_task}")
        order = np.random.default_rng(seed + task_index).permutation(indices)[:per_task]
        selected[task] = order[rank * (per_task // world_size):(rank + 1) * (per_task // world_size)]
    per_rank = per_task // world_size
    per_micro_task = batch_size // len(tasks)
    if per_task % world_size or per_rank % per_micro_task or batch_size % len(tasks):
        raise ValueError("Validation sample count must partition across ranks and balanced microbatches")
    batches = []
    for offset in range(0, per_rank, per_micro_task):
        batch = []
        for task in tasks:
            chunk = selected[task][offset:offset + per_micro_task]
            batch.extend((int(index), int(seed + offset + task_index))
                         for task_index, index in enumerate(chunk))
        batches.append(batch)
    return batches


def validate_prediction_stage(wrapper, dataset, prepare_batch, run, device, world_size, rank, *, stage="stage1"):
    from .data import collate_contact

    eval_model = wrapper.module if isinstance(wrapper, DistributedDataParallel) else wrapper
    was_training = eval_model.training
    previous_rng = rng_state()
    eval_model.eval()
    totals = Metrics()
    sampler_batches = _fixed_validation_sampler(
        dataset, run["tasks"], int(run[stage]["validation_samples_per_task"]),
        world_size, rank, int(run[stage]["micro_batch"]))
    loader = DataLoader(dataset, batch_sampler=sampler_batches, collate_fn=collate_contact,
                        num_workers=0, pin_memory=device.type == "cuda")
    torch.manual_seed(1729 + rank)
    try:
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16,
                                                     enabled=device.type == "cuda"):
            for raw in loader:
                batch = prepare_batch(raw, device)
                result = eval_model(batch)
                labels = raw["task"]
                if torch.is_tensor(labels):
                    labels = labels.tolist()
                labels = [value[0] if isinstance(value, (tuple, list)) else value
                          for value in labels]
                totals.add(result, raw["source"], labels)
        rank_values = [totals.values]
        if world_size > 1:
            rank_values = [None] * world_size
            dist.all_gather_object(rank_values, totals.values)
        combined = {}
        for values in rank_values:
            for key, pair in values.items():
                total = combined.setdefault(key, [0.0, 0.0])
                total[0] += pair[0]
                total[1] += pair[1]
        means = {key: total / count for key, (total, count) in combined.items() if count > 0}
        metric_names = ("loss", "world_total", "contact_total", "visual_initial", "visual_delta", "rank",
                        "tactile_future", "visual_corrected", "tactile_persistence",
                        "tactile_gain", "tactile_wrong_action", "tactile_action_gap")
        per_task = {
            task: {metric: means[f"tasks/{task}/{metric}"] for metric in metric_names
                   if f"tasks/{task}/{metric}" in means}
            for task in run["tasks"]
        }
        per_source = {
            source: {metric: means[f"{source}/{metric}"] for metric in metric_names
                     if f"{source}/{metric}" in means}
            for source in ("original", "recovery")
            if any(f"{source}/{metric}" in means for metric in metric_names)
        }
        per_task_source = {
            task: {
                source: {metric: means[f"tasks/{task}/{source}/{metric}"]
                         for metric in metric_names
                         if f"tasks/{task}/{source}/{metric}" in means}
                for source in ("original", "recovery")
                if any(f"tasks/{task}/{source}/{metric}" in means for metric in metric_names)
            }
            for task in run["tasks"]
        }
        pooled = {metric: means[f"all/{metric}"] for metric in metric_names
                  if f"all/{metric}" in means}
        # The fixed validation design draws the same number of windows per
        # task. Macro-averaging keeps episode length and recurrent-history
        # counts from changing each task's contribution to checkpoint choice.
        aggregate = {
            metric: sum(per_task[task][metric] for task in run["tasks"] if metric in per_task[task])
            / sum(metric in per_task[task] for task in run["tasks"])
            for metric in metric_names
            if any(metric in per_task[task] for task in run["tasks"])
        }
        selection_metric = run[stage].get("selection_metric", "tactile_future" if stage == "tactile" else "world_total")
        return {"loss": aggregate[selection_metric], "selection_metric": selection_metric, "metrics": aggregate,
                "per_task": per_task, "per_source": per_source,
                "per_task_source": per_task_source, "pooled_metrics": pooled,
                "samples_per_task": int(run[stage]["validation_samples_per_task"]),
                "validation_episode_ids": run.get("validation_episode_ids", [])}
    finally:
        eval_model.train(was_training)
        restore_rng(previous_rng)


def _move_batch(raw, device):
    from .data import prepare_aligned_batch
    return prepare_aligned_batch(raw, device)


def _load_dataset(run, stage, validation=False):
    from .data import AlignedDataset
    from .config import ContactConfig
    if stage in {"tactile", "stage1"}:
        manifest = run["shared_manifest"]
        task_stats = run.get("task_stats_by_task", run["wm_stats"])
        wm_stats = run["wm_stats"]
        split = "val" if validation else "train"
    else:
        manifest = run["task_manifest"]
        task_stats, wm_stats = run["task_stats"], run["wm_stats"]
        # Stage 2 deliberately trains on all original episodes, including the
        # Stage-1 held-out set, plus Recovery only in mixed-data recipes.
        split = "train" if run.get("finetune_from") else "all_train"
    if run.get("expert_only"):
        rows = read_jsonl(manifest)
        if any(row.get("source") != "original" for row in rows):
            raise ValueError(f"expert_only requires an Original-only manifest: {manifest}")
        manifest = rows
    dataset_config = dict(run["model_config"])
    dataset_config["stage"] = stage
    return AlignedDataset(manifest, run["data_root"], run["tactile_cache"], task_stats,
                          wm_stats, ContactConfig(**dataset_config), split=split,
                          seed=run["seed"], policy_action_mask=True,
                          policy_supervised_only=stage == "stage2" and run.get("policy_supervised_only", False),
                          tokenizer_path=run["tokenizer_path"])


def _load_model(run, stage, resume_path=None):
    from .model import ContactWAM
    from .weights import initialize
    cfg_values = dict(run["model_config"])
    cfg_values.update(stage=stage, mode=run["mode"], aligned=True,
                      policy_norm_stats=run.get("policy_norm_stats"),
                      world_norm_stats=run.get("world_norm_stats"))
    from .config import ContactConfig
    config = ContactConfig(**cfg_values)
    if resume_path:
        policy = ContactWAM.from_pretrained(Path(resume_path) / "pretrained_model", stage=stage)
        from .performance_restart import is_transfer
        if stage == "stage2" and is_transfer(run, resume_path):
            # The flag changes batching, never parameters or the recurrent objective.
            policy.config.batch_stage2_policy = config.batch_stage2_policy
        if policy.config.to_dict() != config.to_dict():
            raise ValueError("Resume model config differs from the resolved aligned run")
        return policy, None
    if run.get("finetune_from"):
        from .finetune_weights import initialize_finetune
        return initialize_finetune(config, run["finetune_from"],
                                   use_ema=run["finetune_use_ema"],
                                   stage1_checkpoint=run.get("stage1_checkpoint"),
                                   stage2_only=run.get("finetune_stage2_only", False))
    if stage in {"tactile", "stage1"}:
        tactile_checkpoint = run.get("tactile_checkpoint") if stage == "stage1" else None
        policy, report = initialize(config, world_init=run["world_init"],
                                    tactile_checkpoint=Path(tactile_checkpoint) / "pretrained_model" if tactile_checkpoint else None)
    else:
        checkpoint = run.get("stage1_checkpoint")
        if config.use_belief and not checkpoint:
            raise ValueError("Contact policy training requires the shared Stage-1 checkpoint")
        policy, report = initialize(config, world_init=run["world_init"],
                                    stage1_checkpoint=Path(checkpoint) / "pretrained_model" if checkpoint else None)
    return policy, report


def make_ema(policy, run, stage, *, device, rank=0, initialize=True):
    """Disabled EMA allocates no shadow model and needs no EMA file on resume."""
    if stage != "stage2" or not run.get("ema_enabled", True):
        return None
    setting = run[stage].get("ema_device", "cuda")
    if setting not in {"cuda", "cpu"}:
        raise ValueError("ema_device must be cuda or cpu")
    ema_device = device if setting == "cuda" else torch.device("cpu")
    sharded = run[stage].get("shard_ema", False)
    if rank != 0 and not sharded:
        return None
    return (ShardedEMA if sharded else EMA)(policy, run.get("ema_decay", .99),
                                           device=ema_device, initialize=initialize)


def train(run: dict, stage: str, *, resume=None, device="cuda", threads=4):
    """Run one aligned distributed training stage under torchrun."""
    from .data import AlignedTaskBalancedSampler
    from .data import collate_contact

    if stage not in {"tactile", "stage1", "stage2"}:
        raise ValueError(f"Unknown stage: {stage}")
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal aligned training requires CUDA")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != int(run[stage]["num_gpus"]):
        raise ValueError(f"Expected {run[stage]['num_gpus']} ranks for {stage}, received {world_size}")
    torch.set_num_threads(threads)
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    output = Path(run["output"])
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(run["seed"]) + rank)
    np.random.seed(int(run["seed"]) + rank)
    random.seed(int(run["seed"]) + rank)
    if run[stage]["global_batch"] % (run[stage]["micro_batch"] * world_size):
        raise ValueError("Global batch must be divisible by micro batch × world size")
    accumulation = run[stage]["global_batch"] // (run[stage]["micro_batch"] * world_size)
    contract = training_contract(run, stage, world_size)
    identity = _source_identity(run, stage, contract) if rank == 0 else None
    if world_size > 1:
        identity_holder = [identity]
        dist.broadcast_object_list(identity_holder, src=0)
        identity = identity_holder[0]
    checkpoint = Path(resume) if resume else None
    transfer = None
    if checkpoint:
        from .performance_restart import is_transfer, validate_transfer
        if stage == "stage2" and is_transfer(run, checkpoint):
            transfer = validate_transfer(checkpoint, run, identity, world_size, rank=rank)
        else:
            _validate_resume(checkpoint, stage, contract, identity, world_size, rank=rank)
    policy, initialization = _load_model(run, stage, checkpoint)
    policy.to(dev)
    optimizer = make_optimizer(policy, run, stage, device=dev)
    ema = make_ema(policy, run, stage, device=dev, rank=rank, initialize=checkpoint is None)
    start, draws, selection = 0, 0, {
        "best_loss": None, "best_checkpoint": None, "stale": 0, "stopped_early": False,
    }
    if checkpoint:
        state = torch.load(checkpoint / "training.pt", map_location="cpu", weights_only=False)
        if int(state["world_size"]) != world_size:
            raise ValueError("Exact resume requires the original process count")
        optimizer_shards.restore_optimizer(optimizer, policy, checkpoint, state, rank, world_size)
        if ema:
            ema.load(checkpoint / "pretrained_model" / "ema.safetensors", policy)
        start, draws = int(state["step"]), int(state["draws_per_rank"])
        selection = dict(state.get("selection", selection))
        rank_states = state["rng_by_rank"]
        if len(rank_states) != world_size:
            raise ValueError("Checkpoint does not contain RNG state for every rank")
        restore_rng(rank_states[rank])
        del state
    from .sequence import SequenceTrainingModel
    sequence_model = SequenceTrainingModel(policy)
    if world_size > 1:
        wrapped = DistributedDataParallel(sequence_model, device_ids=[local_rank],
                                          find_unused_parameters=True,
                                          gradient_as_bucket_view=True,
                                          broadcast_buffers=False)
    else:
        wrapped = sequence_model
    local_parameters = sum(p.numel() for group in optimizer_shards.local_optimizer(optimizer).param_groups
                           for p in group["params"])
    _memory_event(output, stage, rank, dev, "initialized", start,
                  optimizer_state_sharded=optimizer_shards.is_sharded(optimizer),
                  estimated_local_fp32_moments_gib=local_parameters * 8 / 2**30,
                  ema_device=str(ema.device) if ema else None,
                  ema_gib=sum(value.numel() * value.element_size() for value in ema.values.values()) / 2**30
                  if ema else 0.0)
    dataset = _load_dataset(run, stage)
    validate_every = int(run[stage]["validate_every"]) if stage in {"tactile", "stage1"} else 0
    # Every rank participates when enabled; all-training runs have no held-out dataset.
    val_dataset = _load_dataset(run, stage, validation=True) if validate_every > 0 else None
    total_draws = (int(run[stage]["steps"]) - start) * int(run[stage]["micro_batch"]) * accumulation
    sampler = AlignedTaskBalancedSampler(
        dataset, draws=total_draws, seed=int(run["seed"]), rank=rank, start_draw=draws,
        world_size=world_size, micro_batch=int(run[stage]["micro_batch"]),
        task_balanced=(stage in {"tactile", "stage1"}),
        source_ratio=tuple(float(v) for v in run[stage].get("data_mix", run.get("data_mix", "1:1")).split(":")))
    loader = DataLoader(dataset, batch_size=int(run[stage]["micro_batch"]), sampler=sampler,
                        collate_fn=collate_contact, num_workers=int(run.get("workers", 2)),
                        pin_memory=True, generator=torch.Generator().manual_seed(int(run["seed"]) + rank))
    wrapped.train()
    optimizer.zero_grad(set_to_none=True)
    metrics = Metrics()
    timing = TrainingTiming(enabled=rank == 0)
    interval_started = time.monotonic()
    interval_sequences = 0
    last_data_requested = time.monotonic()
    updates_per_log = int(run.get("log_every", 10))
    step = start
    path = checkpoint
    result_path = output / "result.json"
    if rank == 0 and initialization is not None:
        atomic_json(output / f"initialization_{stage}.json", initialization)
    if rank == 0 and transfer is not None:
        atomic_json(output / "performance_restart.json", transfer)
        print(json.dumps({"event": "performance_restart", **transfer}), flush=True)
    iterator = iter(loader)
    while step < int(run[stage]["steps"]) and not selection["stopped_early"]:
        micro_batches = []
        for _ in range(accumulation):
            try:
                micro_batches.append(next(iterator))
            except StopIteration as exc:
                raise RuntimeError("Aligned sampler ended before the configured optimizer steps") from exc
        timing.add_wall("data_wait", time.monotonic() - last_data_requested)
        timing.micro_batches += accumulation
        for accumulation_index in range(accumulation):
            raw = micro_batches[accumulation_index]
            with timing.wall("prepare"):
                batch = _move_batch(raw, dev)
            if stage == "stage2":
                policy.optimizer_step.fill_(step)
                policy._optimizer_step_value = step
            sync = nullcontext() if world_size == 1 or accumulation_index == accumulation - 1 else wrapped.no_sync()
            with sync:
                with _memory_phase(output, stage, rank, dev, "forward", step + 1, trace=step < start + 2), \
                        timing.cuda("forward"), torch.autocast("cuda", dtype=torch.bfloat16):
                    result = wrapped(batch)
                if not torch.isfinite(result["loss"]):
                    raise FloatingPointError(f"Nonfinite {stage} loss at step {step}")
                with _memory_phase(output, stage, rank, dev, "backward", step + 1, trace=step < start + 2), \
                        timing.cuda("backward"):
                    (result["loss"] / accumulation).backward()
            if rank == 0:
                metrics.add(result, raw["source"], raw.get("task"))
            interval_sequences += len(raw["source"])
            del result, batch
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float(run["grad_clip"]))
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Nonfinite gradient norm")
        set_lr(optimizer, run[stage], stage, step)
        with _memory_phase(output, stage, rank, dev, "optimizer", step + 1, trace=step < start + 2), \
                timing.cuda("optimizer"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        step += 1
        draws += int(run[stage]["micro_batch"]) * accumulation
        if stage == "stage2":
            policy.optimizer_step.fill_(step)
            policy._optimizer_step_value = step
            if ema:
                with timing.wall("ema"), \
                        _memory_phase(output, stage, rank, dev, "ema", step, trace=step <= start + 2):
                    ema.update(policy)
        log_due = step <= start + 2 or step % updates_per_log == 0 or step == int(run[stage]["steps"])
        if log_due:
            _memory_event(output, stage, rank, dev, "update_complete", step)
        if rank == 0 and log_due:
            elapsed = max(time.monotonic() - interval_started, 1e-9)
            row = {"stage": stage, "task": run["task"], "variant": run["variant"], "step": step,
                   "train": metrics.result(), "grad_norm": float(grad_norm),
                   "sequences_per_second": interval_sequences * world_size / elapsed,
                   "global_sequences_per_second": interval_sequences * world_size / elapsed,
                   "global_batch": int(run[stage]["global_batch"]),
                   "timing": timing.result_and_reset(),
                   "cuda_memory": _cuda_memory(dev),
                   "lr": {group["name"]: float(group["lr"]) for group in optimizer.param_groups},
                   "time": time.time()}
            with (output / "train.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
            metrics, interval_started, interval_sequences = Metrics(), time.monotonic(), 0
        validate_due = validate_every > 0 and step % validate_every == 0
        if validate_due:
            validation = validate_prediction_stage(wrapped, val_dataset, _move_batch, run, dev, world_size, rank, stage=stage)
            validation["step"] = step
            if rank == 0:
                with (output / "validation.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(validation, allow_nan=False) + "\n")
                print(json.dumps({"event": "validation", **validation}, allow_nan=False), flush=True)
            if selection["best_loss"] is None or validation["loss"] < selection["best_loss"]:
                selection.update(best_loss=validation["loss"], stale=0,
                                 best_checkpoint=str(output / stage / "checkpoints" / f"step_{step:08d}"))
            else:
                selection["stale"] += 1
            selection["stopped_early"] = bool(
                step >= int(run[stage]["early_stop_min_steps"])
                and selection["stale"] >= int(run[stage]["early_stop_patience"]))
        elif stage in {"tactile", "stage1"} and validate_every == 0 and step == int(run[stage]["steps"]):
            # Persist the final teacher in both the checkpoint and result so
            # resuming after its save can still hand it to every task policy.
            selection.update(method="final", best_checkpoint=str(output / stage / "checkpoints" / f"step_{step:08d}"))
        save_due = validate_due \
            or step % int(run.get("save_every", 1000)) == 0 \
            or step == int(run[stage]["steps"]) or selection["stopped_early"]
        if save_due:
            current_rng = rng_state()
            rng_states = [None] * world_size
            if world_size > 1:
                dist.all_gather_object(rng_states, current_rng)
            else:
                rng_states[0] = current_rng
            path = _atomic_checkpoint(policy, optimizer, ema, run, stage, step, contract,
                                      identity, world_size, rank, rng_states, draws, selection)
            if rank == 0:
                _prune_checkpoints(output, stage, int(run.get("keep_checkpoints", 2)),
                                   selection.get("best_checkpoint") if stage in {"tactile", "stage1"} else None)
        if selection["stopped_early"]:
            break
        last_data_requested = time.monotonic()
    if stage in {"tactile", "stage1"} and not selection["best_checkpoint"]:
        raise RuntimeError("Stage 1 ended without a selected checkpoint")
    if stage == "stage2":
        if ema is not None and ema.is_sharded:
            ema.save_collective(output / "export" / "final" / "ema.safetensors")
        if rank == 0:
            _export(policy, output / "export" / "final", run, ema=ema)
    if rank == 0:
        result = {"complete": True, "formal_training": True, "stage": stage,
                  "task": run["task"], "variant": run["variant"], "step": step,
                  "identity": identity["fingerprint"], "world_size": world_size,
                  "global_batch": int(run[stage]["global_batch"]),
                  "micro_batch": int(run[stage]["micro_batch"]), "accumulation_steps": accumulation,
                  "checkpoint": str(path) if path is not None else None,
                  "selected_checkpoint": selection.get("best_checkpoint") if stage in {"tactile", "stage1"} else None,
                  "selection": selection if stage in {"tactile", "stage1"} else None,
                  "export": str(output / "export" / "final") if stage == "stage2" else None}
        atomic_json(result_path, result)
    if val_dataset is not None:
        close = getattr(val_dataset, "close", None)
        if close:
            close()
    close = getattr(dataset, "close", None)
    if close:
        close()
    if world_size > 1:
        dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--stage", choices=("tactile", "stage1", "stage2"), required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    run = yaml.safe_load(Path(args.run_config).read_text())
    train(run, args.stage, resume=args.resume, device=args.device, threads=args.threads)


if __name__ == "__main__":
    main()
