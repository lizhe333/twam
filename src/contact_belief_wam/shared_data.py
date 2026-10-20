"""Preparation helpers for the shared eight-task Contact-Belief dataset."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .data import build_manifest
from .paths import atomic_json, read_jsonl, write_jsonl

SHARED_TASKS = (
    "insert_hole",
    "insert_HDMI",
    "insert_tube",
    "pull_out_key",
    "grasp_classify",
    "lift_bottle",
    "lift_can",
    "put_bottle_in_shelf",
)


def _shared_quantiles(rows, data_root, scope="shared_tasks"):
    states, actions = [], []
    for row in rows:
        if row["split"] != "train":
            continue
        with h5py.File(Path(data_root) / row["path"], "r") as handle:
            joints = np.asarray(handle["embodiment/joint"][:, :8], dtype=np.float32)
        for start, stop in row["segments"]:
            if stop - start > 1:
                # The command represented at observation t is the absolute joint
                # state of the following saved frame, q[t+1].
                states.append(joints[start:stop - 1])
                actions.append(joints[start + 1:stop])
    if not states:
        raise ValueError("Shared training manifest has no adjacent state/action frames")
    stats = {
        name: {
            "q01": np.quantile(np.concatenate(values, axis=0), .01, axis=0).tolist(),
            "q99": np.quantile(np.concatenate(values, axis=0), .99, axis=0).tolist(),
        }
        for name, values in (("state", states), ("action", actions))
    }
    stats["metadata"] = {
        "scope": scope,
        "tasks": sorted({row["task"] for row in rows}),
        "action": "next_frame_absolute_first8",
        "split": "train",
        "episodes": [row["episode_id"] for row in rows if row["split"] == "train"],
    }
    return stats


def prepare_shared_data(data_root, original_root=None, recovery_root=None, tasks=SHARED_TASKS, seed=42,
                        preserve_existing_splits=True, train_on_all_data=False):
    """Build per-task and merged manifests plus train-only shared joint quantiles.

    Existing per-task train/validation assignments are retained for matching
    episode IDs. New tasks use the normal deterministic manifest split.
    """
    data_root = Path(data_root).resolve()
    tasks = tuple(tasks)
    if not tasks:
        raise ValueError("At least one task is required")
    if not train_on_all_data:
        summary = build_manifest(
            data_root=data_root,
            original_root=original_root,
            recovery_root=recovery_root,
            tasks=tasks,
            seed=seed,
            preserve_existing_splits=preserve_existing_splits,
        )
        rows = [row for task in tasks for row in read_jsonl(data_root / "manifests" / f"{task}.jsonl")]
        write_jsonl(data_root / "manifests" / "shared.jsonl", rows)
        stats_path = data_root / "stats" / "shared.quantiles.json"
        stats = _shared_quantiles(rows, data_root)
        # Keep the configured order for reference while quantile metadata lists the
        # same tasks in a stable order.
        stats["metadata"]["tasks"] = list(tasks)
        atomic_json(stats_path, stats)
        manifest_path = data_root / "manifests" / "shared.jsonl"
    else:
        # Existing per-task artifacts are authoritative for old experiments. Only
        # materialize a task if it has no manifest yet; never regenerate an
        # existing manifest or task-level quantile file in all-train mode.
        missing = [task for task in tasks
                   if not (data_root / "manifests" / f"{task}.jsonl").is_file()]
        if missing:
            build_manifest(
                data_root=data_root,
                original_root=original_root,
                recovery_root=recovery_root,
                tasks=missing,
                seed=seed,
                preserve_existing_splits=preserve_existing_splits,
            )

        original_rows = {
            task: read_jsonl(data_root / "manifests" / f"{task}.jsonl")
            for task in tasks
        }
        task_summary = {
            task: {source: sum(row.get("source") == source for row in original_rows[task])
                   for source in ("original", "recovery")}
            for task in tasks
        }
        # Copy rows before changing split labels so the prior manifests remain
        # intact for validation and single-task experiments.
        rows = [{**row, "split": "train"}
                for task in tasks for row in original_rows[task]]
        all_train_root = data_root / "manifests" / "all_train"
        for task in tasks:
            task_rows = [{**row, "split": "train"} for row in original_rows[task]]
            write_jsonl(all_train_root / f"{task}.jsonl", task_rows)
        manifest_path = all_train_root / "shared.jsonl"
        write_jsonl(manifest_path, rows)
        stats = _shared_quantiles(rows, data_root, scope="shared_tasks_all_train")
        stats["metadata"]["tasks"] = list(tasks)
        stats["metadata"]["split"] = "all_train"
        stats_path = data_root / "stats" / "shared.all_train.quantiles.json"
        atomic_json(stats_path, stats)
        summary = task_summary

    return {
        "tasks": summary,
        "shared": {
            "episodes": len(rows),
            "train_episodes": sum(row["split"] == "train" for row in rows),
            "val_episodes": sum(row["split"] == "val" for row in rows),
            "manifest": str(manifest_path),
            "stats": str(stats_path),
        },
    }
