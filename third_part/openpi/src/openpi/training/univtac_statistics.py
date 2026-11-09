"""Per-task quantile statistics over every valid training frame and its 50 targets."""
from __future__ import annotations
from openpi.shared.univtac_runtime import configure, root, atomic_json
configure()
import json
import h5py
import numpy as np
from openpi.shared import normalize
from openpi.shared.univtac_constants import HORIZON, TASKS
from openpi.training.univtac_data import scan, select_split


def compute(task: str, split: str, manifest: dict | None = None) -> dict:
    if split not in ("full", "debug_train"):
        raise ValueError("Validation must reuse debug_train statistics, never fit its own")
    manifest = manifest or scan(tasks=task)
    if manifest["split"] != split:
        manifest = select_split(manifest, split)
    state_stats, action_stats = normalize.RunningStats(), normalize.RunningStats()
    pairs = 0
    selected = [segment for segment in manifest["segments"] if segment["task"] == task]
    for segment in selected:
        with h5py.File(root() / segment["source_path"], "r") as f:
            joints = np.asarray(f["embodiment/joint"][segment["start_frame"]:segment["stop_frame"], :8], dtype=np.float64)
        states, targets = joints[:-1], joints[1:]
        indices = np.minimum(np.arange(len(targets))[:, None] + np.arange(HORIZON)[None, :], len(targets) - 1)
        chunks = targets[indices].copy()
        chunks[..., :7] -= states[:, None, :7]
        state_stats.update(states)
        action_stats.update(chunks)
        pairs += len(states)
    destination = root() / "assets/pi05_univtac" / split / task
    stats = {"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()}
    normalize.save(destination, stats)
    report = {"task": task, "split": split, "state_count": pairs, "action_count": pairs * HORIZON,
              "horizon": HORIZON, "segment_count": len(selected),
              "normalization": "official openpi RunningStats, q01/q99", "tail": "repeat final valid target",
              "source_episode_ids": sorted({s["source_episode_id"] for s in selected}),
              "dimension": 8, "delta_dimensions": list(range(7)), "gripper": "absolute metres"}
    atomic_json(destination / "manifest.json", report)
    return report


def all_statistics(tasks=TASKS):
    manifest = scan(tasks=tasks)
    return [compute(task, split, manifest) for split in ("debug_train", "full") for task in tasks]


if __name__ == "__main__":
    import argparse
    from openpi.shared.univtac_constants import task_names
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="all")
    parser.add_argument("--split", choices=("full", "debug_train"), default="full")
    args = parser.parse_args()
    tasks = task_names(args.task)
    manifest = scan(tasks=tasks)
    print(json.dumps([compute(task, args.split, manifest) for task in tasks], indent=2))
