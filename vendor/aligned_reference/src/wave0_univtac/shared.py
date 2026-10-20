"""Shared-WM data contract; policy and world normalization stay independent."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Sampler

from .config import TASKS
from .data import UniVTACDataset, audit_data
from .runtime import project_root, atomic_json, digest, object_digest


def task_run(run, task):
    return {**run, "task": task, "baseline": str(project_root()/"baseline"/task)}


def norm_contract(run):
    return {"version": 1, "tasks": list(TASKS),
            "validation_episode_ids": sorted(run["wm_validation_episode_ids"]),
            "training_episode_ids": sorted(set(range(100))-set(run["wm_validation_episode_ids"])),
            "quantiles": "task-balanced weighted empirical CDF, midpoint interpolation",
            "state": "current 8D joints at valid t+10 frames",
            "actions": "ten q[t+1:t+11] targets; first7 relative to q[t], gripper absolute",
            "epsilon": 1e-6, "clip": False,
            "segments_sha256": {t: digest(project_root()/"baseline"/t/"segments.json") for t in TASKS}}


def balanced_statistics(parts):
    """Each task contributes 1/N mass, regardless of its trajectory length."""
    values = np.concatenate(parts).astype(np.float64)
    weights = np.concatenate([np.full(len(v), 1/(len(parts)*len(v))) for v in parts])
    mean = np.sum(values*weights[:, None], axis=0)
    std = np.sqrt(np.sum((values-mean)**2*weights[:, None], axis=0))
    qs = []
    for dim in range(values.shape[1]):
        order = np.argsort(values[:, dim], kind="stable")
        w = weights[order]
        qs.append(np.interp([.01, .99], np.cumsum(w)-w/2, values[order, dim]))
    q = np.asarray(qs)
    return {"mean": mean.tolist(), "std": std.tolist(), "q01": q[:, 0].tolist(), "q99": q[:, 1].tolist()}


def prepare_shared(run, verify_hashes=False):
    reports = {t: audit_data(task_run(run, t), verify_hashes) for t in TASKS}
    contract = norm_contract(run)
    path = Path(run["wm_norm"])
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("contract") != contract or saved.get("fingerprint") != object_digest(
                {k: v for k, v in saved.items() if k != "fingerprint"}):
            raise ValueError("Shared normalization contract changed or file was modified; use a new output")
    else:
        states, actions, counts = [], [], {}
        for task in TASKS:
            segments = json.loads((project_root()/"baseline"/task/"segments.json").read_text())
            task_states, task_actions = [], []
            for s in segments:
                if s["source_episode_id"] not in contract["training_episode_ids"]:
                    continue
                with h5py.File(Path(run["workspace"])/s["source_path"], "r") as f:
                    q = np.asarray(f["embodiment/joint"][s["start_frame"]:s["stop_frame"], :8], np.float32)
                n = len(q)-10
                if n <= 0:
                    continue
                state = q[:n]
                action = q[np.arange(n)[:, None]+np.arange(1, 11)[None, :]].copy()
                action[..., :7] -= state[:, None, :7]
                task_states.append(state)
                task_actions.append(action.reshape(-1, 8))
            states.append(np.concatenate(task_states))
            actions.append(np.concatenate(task_actions))
            counts[task] = {"states": len(states[-1]), "actions": len(actions[-1])}
        saved = {"contract": contract, "counts": counts,
                 "norm_stats": {"state": balanced_statistics(states), "actions": balanced_statistics(actions)}}
        saved["fingerprint"] = object_digest(saved)
        atomic_json(path, saved)
    report = {"tasks": reports, "wm_norm_sha256": digest(path), "wm_norm": str(path),
              "wm_training_episodes": 720, "wm_validation_episodes": 80,
              "policy_training_episodes_per_task": 100, "normalization_contract": contract}
    atomic_json(Path(run["workflow_root"])/"artifacts/data_report.json", report)
    return report


class SharedDataset(ConcatDataset):
    def __init__(self, run, validation=False):
        contract = norm_contract(run)
        saved = json.loads(Path(run["wm_norm"]).read_text())
        if saved["contract"] != contract:
            raise ValueError("Prepare the shared WM normalization before training")
        ids = contract["validation_episode_ids" if validation else "training_episode_ids"]
        super().__init__([UniVTACDataset(task_run(run, t), episode_ids=ids, stats=run["wm_norm"],
                                       valid_future_only=True) for t in TASKS])
        self.sizes = [len(d) for d in self.datasets]
        if not all(self.sizes):
            raise ValueError("Every task needs valid future examples in both splits")

    def close(self):
        for ds in self.datasets:
            ds.close()


class BalancedReplaySampler(Sampler):
    """Equal tasks in EVERY physical batch, same-task negatives and exact resume.

    Per-task epoch permutations are partitioned across ranks, then microbatches.
    Different ranks do not repeat samples until that task's epoch wraps.
    """
    def __init__(self, sizes, seed, global_batch, world_size, rank, micro_batch, start_step, stop_step):
        self.sizes, self.seed, self.batch = sizes, seed, global_batch
        self.world, self.rank, self.micro = world_size, rank, micro_batch
        self.start, self.stop = start_step, stop_step
        self.tasks = len(sizes)
        if micro_batch % (2*self.tasks) or global_batch % (world_size*micro_batch):
            raise ValueError("WM batches require >=2 samples/task/microbatch and integral accumulation")

    def __iter__(self):
        per_task_micro = self.micro//self.tasks
        per_task_global = self.batch//self.tasks
        local = per_task_global//self.world
        offsets = np.cumsum([0]+list(self.sizes[:-1]))
        cache = {}
        for step in range(self.start, self.stop):
            for micro in range(self.batch//(self.world*self.micro)):
                for task, size in enumerate(self.sizes):
                    for j in range(per_task_micro):
                        pos = step*per_task_global+self.rank*local+micro*per_task_micro+j
                        epoch, index = divmod(pos, size)
                        if task not in cache or cache[task][0] != epoch:
                            order = np.random.default_rng(np.random.SeedSequence([self.seed, task, epoch])).permutation(size)
                            cache[task] = (epoch, order)
                        yield int(offsets[task]+cache[task][1][index])

    def __len__(self):
        return (self.stop-self.start)*self.batch//self.world


def bridge_tensor(value, source, target):
    """Invert task quantiles and apply WM quantiles; preserve padded dimensions."""
    lo = value.new_tensor(source["q01"])
    span = value.new_tensor(source["q99"])-lo+1e-6
    dst_lo = value.new_tensor(target["q01"])
    dst_span = value.new_tensor(target["q99"])-dst_lo+1e-6
    first = ((value[..., :8]+1)*.5*span+lo-dst_lo)/dst_span*2-1
    return torch.cat([first, value[..., 8:]], dim=-1)


def install_world_normalization(policy, task_stats, wm_stats):
    """Hooks preserve official parameter names and affect ONLY WM inputs.

    The policy prompt/flow target keeps the original task statistics. The frozen
    teacher and inference state encoder see the shared Stage-1 coordinate system.
    No learnable parameters or mutable batch context are introduced.
    """
    if getattr(policy, "_world_norm_handles", None):
        raise ValueError("World normalization already installed")
    source = json.loads(Path(task_stats).read_text())["norm_stats"]
    target = json.loads(Path(wm_stats).read_text())["norm_stats"]

    def state_hook(module, args):
        return (bridge_tensor(args[0], source["state"], target["state"]), *args[1:])

    def action_hook(module, args):
        actions, state, *rest = args
        return (bridge_tensor(actions, source["actions"], target["actions"]),
                bridge_tensor(state, source["state"], target["state"]) if state is not None else None, *rest)

    handles = [policy.model.world_state_encoder.register_forward_pre_hook(state_hook)]
    if policy.model.world_action_encoder is not None:
        handles.append(policy.model.world_action_encoder.register_forward_pre_hook(action_hook))
    policy._world_norm_handles = handles
