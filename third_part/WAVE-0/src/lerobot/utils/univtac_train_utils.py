from __future__ import annotations

from contextlib import nullcontext
import fcntl
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
from torch.utils.data import DataLoader
from safetensors.torch import load_file, save_file

from lerobot.datasets.univtac_dataset import UniVTACDataset, ReplaySampler
from lerobot.policies.wave0.loading_wave0 import create_policy, group_name, gradient_report
from lerobot.utils.univtac_runtime import atomic_json, digest, object_digest, project_root
from lerobot.optim.optimizer_shards import (FORMAT as SHARD_FORMAT, is_sharded, local_optimizer,
                               make_sharded_adamw, save_local_state, load_local_state)


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def profile_due(enabled, step, log_every, limit):
    return enabled and (step % log_every == 0 or step == limit)

def assert_finite(value, message):
    finite = torch.isfinite(value).all()
    if value.device.type == "cuda":
        # Avoid a host read on every CUDA update while still surfacing invalid
        # values through the current stream's asynchronous assertion.
        torch._assert_async(finite, message)
    elif not finite:
        raise FloatingPointError(message)

def identity(run, debug):
    cfg = {k:v for k,v in run.items() if k not in ("workspace", "base", "world_init", "tokenizer", "baseline", "output",
                                                 "workflow_root", "wm_norm", "stage1_checkpoint")}
    root = project_root()
    hashes = {str(p.relative_to(root)): digest(p) for p in sorted((root/"src").rglob("*.py"))}
    for task in run.get("tasks", [run["task"]]):
        for p in sorted((root/"assets/univtac/baseline"/task).glob("*.json")):
            hashes["baseline/" + str(p.relative_to(root/"assets/univtac/baseline"))] = digest(p)
    if run.get("wm_norm"):
        hashes["shared_wm_normalization"] = digest(run["wm_norm"])
    if run.get("stage1_checkpoint"):
        cfg["shared_teacher"] = json.loads((Path(run["stage1_checkpoint"])/"complete.json").read_text())
    hashes["tokenizer"] = digest(run["tokenizer"])
    # Explicit source receipts bind initialization without rehashing 14 GB at every resume.
    for key, receipt in (("base", "conversion.json"), ("world_init", "DOWNLOAD_SOURCE.json")):
        if run.get(key):
            p = Path(run[key])/receipt
            hashes[key+"_receipt"] = digest(p) if p.exists() else None
            hashes[key+"_bytes"] = (Path(run[key])/"model.safetensors").stat().st_size
    result = {"recipe": cfg, "hashes": hashes, "debug": debug}
    result["fingerprint"] = object_digest(result)
    return result

def latest(output, stage):
    paths = sorted((Path(output)/stage/"checkpoints").glob("step_*/complete.json"))
    return paths[-1].parent if paths else None

def deployment_contract(run):
    if run["task"]=="shared_wm":
        return {}
    path = Path(run["baseline"])/"deployment_contract.json"
    if path.exists():
        contract = json.loads(path.read_text())
        comparison = contract.get("baseline_camera_comparison", {})
        comparison["target_mode"] = run.get("camera_mode", "head_only")
        comparison["matched"] = comparison.get("source_mode") == comparison["target_mode"]
        contract["baseline_camera_comparison"] = comparison
        return contract
    return {"prompt": json.loads((Path(run["baseline"])/"instruction.json").read_text())["seen"][0],
            "policy_step_budget": 317 if run["task"]=="insert_hole" else 600}

def check_shared_teacher(run, debug):
    if not run.get("stage1_checkpoint"):
        raise ValueError("Shared Stage 2 requires --stage1-checkpoint from the completed shared WM run")
    path = Path(run["stage1_checkpoint"]).resolve()
    meta = json.loads((path/"complete.json").read_text())
    saved = json.loads((path/"run_identity.json").read_text())
    if meta["stage"]!="stage1" or saved["recipe"]["task"]!="shared_wm" or saved["debug"]!=debug:
        raise ValueError("Teacher must be a shared Stage-1 checkpoint of the same debug/formal mode")
    if saved["recipe"]["tasks"] != run["tasks"] or digest(path/"wm_norm_stats.json") != digest(run["wm_norm"]):
        raise ValueError("Teacher tasks or world normalization differ")
    if digest(path/"pretrained_model/model.safetensors") != meta["model_sha256"]:
        raise ValueError("Teacher weight integrity check failed")
    if not debug:
        result = json.loads((path.parents[1]/"result.json").read_text())
        if not result["complete"] or result["identity"]!=saved["fingerprint"] or Path(result["selected_checkpoint"]).resolve()!=path:
            raise ValueError("Shared WM selection has not completed")
    return path

def prune_checkpoints(run, stage, selected=None):
    keep = run.get("keep_checkpoints", 0)
    if not keep:
        return
    paths = sorted((Path(run["output"])/stage/"checkpoints").glob("step_*/complete.json"))
    for p in paths[:-keep]:
        if selected is None or p.parent != Path(selected):
            shutil.rmtree(p.parent)

def validate_world(policy, run, device, world, rank, debug=False):
    """Fixed, task-balanced held-out episodes; no train transforms or RNG drift."""
    from lerobot.datasets.univtac_shared import SharedDataset
    from torch.utils.data._utils.collate import default_collate
    ds = SharedDataset(run, validation=True)
    count = 2 if debug else run["stage1"]["validation_samples_per_task"]
    # All tasks contribute equally; every physical task batch has same-task negatives.
    local = max(2, count//world)
    if not debug and (count % world or count//world < 2):
        raise ValueError("Validation requires >=2 samples/task/rank and an exact partition")
    totals = torch.zeros(len(ds.datasets), dtype=torch.float64, device=device)
    previous, was_training = rng_state(), policy.training
    policy.eval()
    torch.manual_seed(1729+rank)
    try:
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
            for task, child in enumerate(ds.datasets):
                order = np.random.default_rng(1729+task).permutation(len(child))
                if local*world > len(order):
                    raise ValueError("Validation sample budget exceeds available held-out frames")
                indices = order[rank*local:(rank+1)*local]
                for start in range(0, local, min(local, run["stage1"]["micro_batch"])):
                    chosen = indices[start:start+run["stage1"]["micro_batch"]]
                    raw = default_collate([child[int(i)] for i in chosen])
                    batch = {k:v.to(device) if torch.is_tensor(v) else v for k,v in raw.items()}
                    loss, _ = policy(batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite held-out WM loss")
                    totals[task] += loss.double()*len(chosen)/local
        if world > 1:
            dist.all_reduce(totals)
            totals /= world
        return {"loss": float(totals.mean()), "per_task": dict(zip(run["tasks"], totals.cpu().tolist())),
                "samples_per_task": local*world, "source_episode_ids": run["wm_validation_episode_ids"]}
    finally:
        ds.close()
        policy.train(was_training)
        restore_rng(previous)

def rng_state():
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}

def restore_rng(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])

def tensor_witness(tensor):
    value = tensor.detach().float().flatten()
    # Every parameter contributes a small deterministic witness, without duplicating the model.
    return value[:8].cpu().tolist()

def witnesses(policy, optimizer, ema):
    return {"parameters": {n:tensor_witness(p) for n,p in policy.named_parameters() if p.requires_grad},
            "optimizer": {str(i):{k:tensor_witness(v) for k,v in state.items() if torch.is_tensor(v)}
                          for i,state in enumerate(local_optimizer(optimizer).state.values())},
            "ema": {n:tensor_witness(v) for n,v in ema.values.items()} if ema else None}

def save_checkpoint(policy, optimizer, ema, run, stage, step, run_identity, world, rank, audit):
    state = rng_state()
    states = [None]*world
    if world > 1:
        dist.all_gather_object(states, state)
    else:
        states[0] = state
    path = Path(run["output"])/stage/"checkpoints"/f"step_{step:08d}"
    pending = path.with_name(path.name+".partial")
    if rank == 0:
        if path.exists():
            raise FileExistsError(f"Will not overwrite checkpoint: {path}")
        if pending.exists():
            # Preserve failed writes for inspection while allowing one-command resume.
            pending.rename(pending.with_name(pending.name+f".abandoned-{time.time_ns()}"))
        pending.mkdir(parents=True)
    if is_sharded(optimizer):
        dist.barrier()
        save_local_state(optimizer, policy, pending, rank, world)
        dist.barrier()
    if rank == 0:
        policy.save_pretrained(pending/"pretrained_model")
        torch.save({"optimizer": None if is_sharded(optimizer) else optimizer.state_dict(),
                    "optimizer_format": SHARD_FORMAT if is_sharded(optimizer) else "full",
                    "rng": states, "step": step, "world_size": world}, pending/"training.pt")
        if ema:
            ema.save(pending/"ema.safetensors")
        norm = Path(run["wm_norm"]) if run["task"]=="shared_wm" else Path(run["baseline"])/"norm_stats.json"
        shutil.copy2(norm, pending/"norm_stats.json")
        if run.get("shared_wm"):
            shutil.copy2(run["wm_norm"], pending/"wm_norm_stats.json")
        shutil.copy2(run["tokenizer"], pending/"tokenizer.model")
        atomic_json(pending/"run_identity.json", run_identity)
        atomic_json(pending/"deployment.json", {
            "task": run["task"], "camera": run.get("camera_mode", "head_only"), "chunk_size": 50, "execute_steps": 10,
            "strategy_hz": 60, "physics_hz": 120, "physical_steps_per_action": 2,
            "action": "first7_delta_from_chunk_observation_plus_absolute_gripper",
            "normalization": "baseline_q01_q99_range_plus_1e-6", "tokenized_state_dim": 8,
            "future_offset_frames": 10, "official_inclusive_action_horizon": 9,
            "policy_step_budget": deployment_contract(run).get("policy_step_budget"),
            "evaluation_seeds": [1000000, 1000049], "use_ema": ema is not None,
            "prompt": deployment_contract(run).get("prompt"),
            "shared_world_normalization": bool(run.get("shared_wm")),
            "baseline_camera_comparison": deployment_contract(run).get("baseline_camera_comparison"),
            "formal_training": not run_identity["debug"]})
        if run_identity["debug"]:
            atomic_json(pending/"witnesses.json", witnesses(policy, optimizer, ema))
        atomic_json(pending/"audit.json", audit)
        meta = {"step": step, "stage": stage, "world_size": world, "identity": run_identity["fingerprint"]}
        if is_sharded(optimizer):
            meta["optimizer_format"] = SHARD_FORMAT
            meta["optimizer_shards"] = [f"optimizer.rank{r:02d}.pt" for r in range(world)]
        if stage=="stage1" and run.get("shared_wm"):
            meta["model_sha256"] = digest(pending/"pretrained_model/model.safetensors")
        atomic_json(pending/"complete.json", meta)
        pending.rename(path)
    if world > 1:
        dist.barrier()
    return path

def probe(policy, batch, device, seed=123, cpu_offload=False):
    previous = rng_state()
    was_training = policy.training
    policy.eval()
    torch.manual_seed(seed)
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, cache_enabled=not cpu_offload):
        if cpu_offload:
            from lerobot.utils.debug_offload import offloaded_forward
            loss, metrics = offloaded_forward(policy, batch, device, execution_dtype=torch.bfloat16)
        else:
            loss, metrics = policy(batch)
    policy.train(was_training)
    restore_rng(previous)
    return {"loss": float(loss), "metrics": metrics}


def compatible_legacy_identity(saved, current, receipt=None):
    """Accept only recorded source relocations, with recipe/assets unchanged.

    This does not rewrite a historical checkpoint or ignore an unknown code
    revision. New integrated runs continue to require exact identity equality.
    """
    if saved == current:
        return True
    expected = object_digest({k: v for k, v in saved.items() if k != "fingerprint"})
    if saved.get("fingerprint") != expected:
        return False
    if saved.get("recipe") != current.get("recipe") or saved.get("debug") != current.get("debug"):
        return False
    def code(hashes):
        return {k: v for k, v in hashes.items() if k.startswith(("src/", "vendor/"))}
    def assets(hashes):
        return {k: v for k, v in hashes.items() if not k.startswith(("src/", "vendor/"))}
    if assets(saved["hashes"]) != assets(current["hashes"]):
        return False
    if receipt is None:
        receipt = json.loads((project_root()/"assets/univtac/provenance/legacy_sources.json").read_text())
    allowed = [*receipt["source_variants"].values(), *receipt["checkpoint_source_sets"]]
    return code(saved["hashes"]) in allowed


def resume_identity(current, saved, migration_path):
    if saved == current:
        return current
    if not compatible_legacy_identity(saved, current):
        raise ValueError("Recipe/code/data identity changed; unrecognized legacy source or incompatible inputs")
    migration_path = Path(migration_path)
    record = {"legacy_fingerprint": saved["fingerprint"], "integrated_identity": current}
    if migration_path.exists() and json.loads(migration_path.read_text()) != record:
        raise ValueError("Integrated source changed since checkpoint migration")
    return saved
