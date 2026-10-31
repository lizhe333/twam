#!/usr/bin/env python3
"""Write a local recipe referencing an existing tactile workspace; never launch jobs."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def configuration(workspace):
    workspace = Path(workspace).expanduser().resolve()
    text = (ROOT / "configs/aligned_4xa800.yaml").read_text()
    # Preserve the original data, cache, normalization, and weight locations.
    # New checkpoints and derived alignment files belong to this checkout.
    raw = {key: value.replace("/zhe_li/tactile-wam", str(workspace)) if isinstance(value, str) else value
           for key, value in yaml.safe_load(text).items()}
    raw.update(
        workspace=str(workspace),
        output=str(ROOT / "runs/contact-head-wrist-two-stage-4gpu"),
        aligned_manifest_dir=str(ROOT / "data/manifests/aligned"),
        shared_manifest=str(ROOT / "data/manifests/aligned/shared.jsonl"),
        policy_manifest_dir=str(ROOT / "data/manifests/aligned/policy_all_train"),
        task_stats_dir=str(ROOT / "data/stats/aligned/tasks"),
        wm_stats=str(ROOT / "data/stats/aligned/wm_norm_stats.json"),
        reference_root=str(ROOT / "vendor/aligned_reference"),
        baseline_dir=str(ROOT / "vendor/aligned_reference/baseline"),
    )
    return raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "configs/workspace.local.yaml")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        yaml.safe_dump(configuration(args.workspace), out, sort_keys=False)
    print(f"Wrote {args.output.resolve()}; review asset paths before running prepare/debug/train.")
