from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from lerobot.utils.univtac_runtime import configure, atomic_json


def main():
    root = configure()
    p = argparse.ArgumentParser(description="WAVE-0 UniVTAC training and inference tools")
    p.add_argument("command", choices=["prepare", "plan", "train", "debug", "infer-smoke"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--workspace", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--stage", choices=["stage1", "stage2"])
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--verify-hashes", action="store_true")
    p.add_argument("--debug-updates", type=int, choices=[1,2])
    p.add_argument("--debug-cpu-offload", action="store_true",
                   help="single-GPU debug only: keep Stage-2 master weights, gradients and Adam on CPU")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--task")
    p.add_argument("--stage1-checkpoint", type=Path)
    args = p.parse_args()
    from lerobot.configs.univtac import load_config
    run = load_config(args.config, args.workspace, args.output, args.task, args.stage1_checkpoint)
    if args.command=="plan":
        launcher = "train_all.sh" if run.get("shared_wm") else "train.sh"
        result = {"run": run, "launch": f"scripts/train/univtac/{launcher} --config {shlex.quote(str(args.config))} --output OUTPUT [--resume]",
                  "formal_training_started": False}
    elif args.command=="prepare":
        import torch
        torch.set_num_threads(args.threads)
        from lerobot.datasets.univtac_dataset import audit_data
        if run.get("shared_wm"):
            from lerobot.datasets.univtac_shared import prepare_shared
            result = prepare_shared(run, args.verify_hashes)
        else:
            result = audit_data(run, args.verify_hashes)
        atomic_json(root/"reports"/f"data-{run['task']}.json", result)
    elif args.command=="train":
        if args.stage is None:
            p.error("train requires --stage")
        from lerobot.scripts.lerobot_train import train_univtac as train
        result = train(run, args.stage, debug=args.debug_updates is not None,
                       updates=args.debug_updates or 1, resume=args.resume,
                       device=args.device, threads=args.threads, verify_only=args.verify_only,
                       debug_cpu_offload=args.debug_cpu_offload)
    elif args.command=="debug":
        if run.get("shared_wm"):
            p.error("Use scripts/debug/univtac/debug_shared.py for bounded shared-WM and per-task checks")
        if args.output is None:
            p.error("debug requires a fresh explicit --output")
        if args.output.exists():
            p.error("Use a new debug output; partial debug runs can be resumed with train --debug-updates")
        records = []
        for stage in ("stage1", "stage2"):
            common = [sys.executable, "-m", "lerobot.scripts.lerobot_train", "--univtac", "train", "--config", str(args.config.resolve()),
                      "--workspace", run["workspace"], "--output", run["output"], "--stage", stage,
                      "--device", args.device, "--threads", str(args.threads)]
            if args.debug_cpu_offload:
                common += ["--debug-cpu-offload"]
            # Two true updates, across a fresh process restart, using the same
            # production optimizer/checkpoint path. No frozen-policy shortcut.
            for count in (1,2):
                command = common+["--debug-updates", str(count)]+(["--resume"] if count==2 else [])
                subprocess.run(command, check=True)
            subprocess.run(common+["--debug-updates", "2", "--resume", "--verify-only"], check=True)
            records.append(json.loads((Path(run["output"])/stage/"restore_report.json").read_text()))
        subprocess.run([sys.executable, "-m", "lerobot.scripts.lerobot_train", "--univtac", "infer-smoke", "--config", str(args.config.resolve()),
                        "--workspace", run["workspace"], "--output", run["output"], "--device", args.device,
                        "--threads", str(args.threads)], check=True)
        result = {"task":run["task"], "passed":True,"formal_training":False,
                  "updates_per_stage":2, "restore": records,
                  "inference":json.loads((Path(run["output"])/"inference_report.json").read_text())}
        atomic_json(Path(run["output"])/"debug_report.json", result)
    else:
        import numpy as np
        import torch
        from lerobot.policies.wave0.inference_wave0 import Policy
        from lerobot.scripts.lerobot_train import latest
        from lerobot.datasets.univtac_dataset import UniVTACDataset, decode
        torch.set_num_threads(args.threads)
        checkpoint = args.checkpoint or latest(run["output"], "stage2")
        if checkpoint is None:
            p.error("No Stage-2 checkpoint")
        ds = UniVTACDataset(run)
        source = ds._file(ds.segments[0])
        image = decode(source["observation/head/rgb"][0])
        wrist = decode(source["observation/wrist/rgb"][0]) if run.get("camera_mode") == "head_wrist" else None
        state = np.asarray(source["embodiment/joint"][0,:8], np.float32)
        policy = Policy(checkpoint, args.device)
        noise = torch.randn((1,50,32), generator=torch.Generator().manual_seed(123)).to(args.device)
        first = policy.infer(image, state, noise=noise, wrist_rgb=wrist)
        second = policy.infer(image, state, noise=noise, wrist_rgb=wrist)
        if not np.array_equal(first, second):
            raise AssertionError("Fixed-noise inference is not repeatable")
        result = {"task":run["task"], "shape":list(first.shape), "finite":bool(np.isfinite(first).all()),
                  "fixed_noise_max_abs_difference":float(np.abs(first-second).max()), "ema_loaded":policy.use_ema,
                  "action_min":first.min(0).tolist(), "action_max":first.max(0).tolist(),
                  "simulator_rollout":False}
        atomic_json(Path(run["output"])/"inference_report.json", result)
        ds.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__=="__main__":
    main()
