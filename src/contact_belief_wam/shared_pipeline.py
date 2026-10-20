"""Eight-task shared visuotactile WM, followed by independent task policies."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import yaml

from . import cloud
from .paths import PROJECT, WORKSPACE, atomic_json, configure, read_jsonl


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "train", "debug"])
    p.add_argument("--config", default=str(PROJECT / "configs/shared_8xa800_bs64.yaml"))
    p.add_argument("--output")
    p.add_argument("--num-gpus", type=int)
    p.add_argument("--micro-batch", type=int)
    p.add_argument("--global-batch", type=int)
    p.add_argument("--k", type=int)
    p.add_argument("--data-mix")
    p.add_argument("--stage1-steps", type=int)
    p.add_argument("--stage2-steps", type=int)
    p.add_argument("--stage2-original-steps", type=int)
    p.add_argument("--resume", choices=["none", "auto"], default="none")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16, help="Frozen feature extraction batch")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--manifest-only", action="store_true")
    p.add_argument("--updates", type=int, choices=[1, 2], default=2, help="Bounded debug updates per model")
    p.add_argument("--optimizer-device", choices=["cpu", "cuda"],
                   help="Debug only: CPU AdamW state keeps the complete policy within 4090 memory")
    return p


def resolve(args):
    raw = yaml.safe_load(Path(args.config).read_text())
    cli = ["--run-config", args.config, "--manifest", raw["shared_manifest"],
           "--stats", raw["shared_stats"], "--task", "shared_wm", "--resume", args.resume]
    for key in ("output", "num_gpus", "micro_batch", "global_batch", "k", "data_mix", "stage1_steps", "stage2_steps"):
        if getattr(args, key) is not None:
            cli += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    # Changing GPU count or micro batch alone retains GA=1 for this recipe.
    if args.global_batch is None and (args.num_gpus is not None or args.micro_batch is not None):
        cli += ["--global-batch", str((args.num_gpus or raw["num_gpus"]) * (args.micro_batch or raw["micro_batch"]))]
    run, _ = cloud.resolve(cli)
    if run.get("train_on_all_data", False):
        run["val_every"] = 0
    if args.stage2_original_steps is not None:
        run["stage2_original_steps"] = args.stage2_original_steps
    if not 0 <= run["stage2_original_steps"] <= run["stage2_steps"]:
        raise ValueError("stage2_original_steps must be within Stage 2")
    if run["stage1_steps"] == "auto":
        run["stage1_steps"] = run["stage1_scratch_steps"] if run["world_init"] == "none" else run["stage1_pretrained_steps"]
    if run["stage1_steps"] < 1 or run["stage2_steps"] < 1:
        raise ValueError("Shared training needs positive steps for both stages")
    if not run.get("shared_tasks") or not run.get("policy_tasks"):
        raise ValueError("Specify shared_tasks and policy_tasks")
    if not set(run["policy_tasks"]) <= set(run["shared_tasks"]):
        raise ValueError("Policy tasks must be represented in shared training data")
    return run


def stage_run(run, stage, task=None, checkpoint=None, output=None):
    root = Path(output or run["output"])
    shared = stage == "stage1"
    return {**run, "task": "shared_wm" if shared else task,
            "manifest": run["shared_manifest"] if shared else str(Path(run.get("policy_manifest_dir", Path(run["data_root"]) / "manifests")) / f"{task}.jsonl"),
            "stats": run["shared_stats"], "task_balanced": shared,
            "policy_action_mask": True,
            "stage1_checkpoint": None if shared else str(checkpoint),
            "output": str(root / "shared_wm" if shared else root / "tasks" / task)}


def check_data(run):
    rows = read_jsonl(run["shared_manifest"])
    if set(r["task"] for r in rows) != set(run["shared_tasks"]):
        raise ValueError("Shared manifest task set differs from shared_tasks")
    for key in ("shared_stats", "pi05_base", "tokenizer_path"):
        if not Path(run[key]).exists():
            raise FileNotFoundError(f"{key}: {run[key]}")
    missing = [r["episode_id"] for r in rows
               if not (Path(run["tactile_cache"]) / (r["episode_id"] + ".npz")).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} tactile caches, e.g. {missing[:3]}; run shared_train.sh prepare")
    return rows


def execute_stage(run, stage, steps, *, dry_run=False):
    directory = Path(run["output"])
    candidates = cloud.complete_checkpoints(directory, stage)
    resume = candidates[-1][1] if run["resume"] == "auto" and candidates else None
    if resume and not dry_run:
        from .trainer import check_resume_contract
        check_resume_contract(run, json.loads((resume / "complete.json").read_text()))
    if resume and candidates[-1][0] >= steps:
        if stage == "stage2" and not dry_run:
            cloud.restore_final_export(run, directory)
        print(json.dumps({"event": "stage_already_complete", "task": run["task"], "stage": stage}), flush=True)
        return
    command = cloud.training_command(run, directory, stage, steps, resume)
    print(json.dumps({"event": "shared_stage", "task": run["task"], "stage": stage,
                      "micro_batch": run["micro_batch"], "global_batch": run["global_batch"],
                      "GA": run["accumulation_steps"], "command": command}), flush=True)
    if dry_run:
        return
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "run_config.yaml").exists() and run["resume"] == "none":
        raise FileExistsError(f"Run exists: {directory}; use --resume auto")
    (directory / "run_config.yaml").write_text(yaml.safe_dump(run, sort_keys=False))
    shutil.copy2(run["manifest"], directory / "data_manifest.jsonl")
    shutil.copy2(run["stats"], directory / "quantiles.json")
    subprocess.run(command, check=True, env=os.environ.copy())


def train_shared(run, *, dry_run=False):
    output = Path(run["output"])
    if not dry_run:
        check_data(run)
        output.mkdir(parents=True, exist_ok=True)
        if (output / "pipeline_config.yaml").exists() and run["resume"] == "none":
            raise FileExistsError(f"Pipeline exists: {output}; use --resume auto or a new output")
        (output / "pipeline_config.yaml").write_text(yaml.safe_dump(run, sort_keys=False))
    wm_run = stage_run(run, "stage1")
    state = {"formal_training": True, "stage1": "running", "tasks": {}}
    if not dry_run:
        atomic_json(output / "pipeline_state.json", state)
    execute_stage(wm_run, "stage1", run["stage1_steps"], dry_run=dry_run)
    if dry_run:
        teacher = Path(wm_run["output"]) / "stage1/checkpoints" / f"step_{run['stage1_steps']:08d}" / "pretrained_model"
    else:
        checkpoints = cloud.complete_checkpoints(wm_run["output"], "stage1")
        if not checkpoints or checkpoints[-1][0] < run["stage1_steps"]:
            raise RuntimeError("Stage 1 exited without its completed checkpoint")
        teacher = checkpoints[-1][1] / "pretrained_model"
        state.update(stage1="complete", shared_checkpoint=str(teacher))
        atomic_json(output / "pipeline_state.json", state)
    for task in run["policy_tasks"]:
        policy_run = stage_run(run, "stage2", task, teacher)
        execute_stage(policy_run, "stage2", run["stage2_steps"], dry_run=dry_run)
        if not dry_run:
            cloud.restore_final_export(policy_run, Path(policy_run["output"]), stage2_ran=True)
            state["tasks"][task] = {"status": "complete", "export": str(Path(policy_run["output"]) / "export/final")}
            atomic_json(output / "pipeline_state.json", state)
    if not dry_run:
        state["complete"] = True
        atomic_json(output / "pipeline_state.json", state)


def main(argv=None):
    configure()
    args = parser().parse_args(argv)
    if args.dry_run and args.command != "train":
        raise ValueError("--dry-run only prints formal train commands; it cannot be combined with prepare/debug")
    if args.optimizer_device is not None and args.command != "debug":
        raise ValueError("--optimizer-device is a debug-only option; formal training uses GPU AdamW")
    if args.command == "debug" and not args.output:
        args.output = str(PROJECT / "outputs/debug_shared_4090")
    run = resolve(args)
    if args.command == "prepare":
        from .shared_data import prepare_shared_data
        print(json.dumps(prepare_shared_data(run["data_root"], tasks=run["shared_tasks"],
                                            train_on_all_data=run.get("train_on_all_data", False)), indent=2), flush=True)
        if not args.manifest_only:
            from .tactile import AnyTouchEncoder, cache_episodes
            encoder = AnyTouchEncoder(WORKSPACE / "AnyTouch2", device=args.device, threads=args.threads)
            cache_episodes(read_jsonl(run["shared_manifest"]), run["data_root"], run["tactile_cache"], encoder, args.batch_size)
            from .feature_cache import prepare_mmap_cache
            report = prepare_mmap_cache(read_jsonl(run["shared_manifest"]), run["tactile_cache"])
            atomic_json(Path(run["data_root"]) / "mmap_cache_prepare.json", report)
            print(json.dumps({k: v for k, v in report.items() if k != "episode_results"}), flush=True)
    elif args.command == "debug":
        from .shared_debug import run_debug_pipeline
        return run_debug_pipeline(run, updates=args.updates, device=args.device,
                                  optimizer_device=args.optimizer_device or "cpu", threads=args.threads)
    else:
        train_shared(run, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
