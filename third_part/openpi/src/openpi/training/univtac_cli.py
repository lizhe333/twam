"""UniVTAC commands integrated into the OpenPI training and serving entries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openpi.shared.univtac_runtime import configure


def main():
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "train"):
        p = commands.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--task", default="all")
        p.add_argument("--mode", choices=("full", "lora", "overfit", "smoke"), default="full")
        p.add_argument("--run", required=True)
        p.add_argument("--resume", action="store_true")
        p.add_argument("--steps", type=int)
    for name in ("serve", "infer"):
        p = commands.add_parser(name)
        p.add_argument("--task", required=True)
        p.add_argument("--config", type=Path)
        p.add_argument("--model-mode", choices=("base", "full", "lora", "overfit", "smoke"), default="full")
        p.add_argument("--checkpoint", type=Path)
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    from openpi.shared.univtac_constants import task_names
    tasks = task_names(args.task)
    if args.command in ("plan", "train"):
        from openpi.training.univtac_config import make_config
        configs = [make_config(task, args.mode, run=args.run, resume=args.resume,
                               steps=args.steps, recipe=args.config) for task in tasks]
        if args.command == "plan":
            print(json.dumps({"training_started": False, "tasks": [
                {"task": task, "camera_mode": c.data.camera_mode,
                 "batch_size": c.batch_size, "fsdp_devices": c.fsdp_devices,
                 "steps": c.num_train_steps, "ema_decay": c.ema_decay,
                 "dataset": c.data.repo_id, "checkpoint_dir": str(c.checkpoint_dir)}
                for task, c in zip(tasks, configs)]}, indent=2))
        elif len(tasks) == 1:
            from openpi.training.univtac_trainer import run
            run(tasks[0], args.mode, args.run, resume=args.resume, steps=args.steps, recipe=args.config)
        else:
            if args.steps is not None or args.mode == "overfit":
                parser.error("--steps and overfit require a single task; set num_train_steps in the full recipe for a queue")
            from openpi.training.univtac_queue import run_queue
            run_queue(tasks, args.mode, args.run, resume=args.resume, recipe=args.config.resolve())
    elif args.command == "serve":
        from openpi.serving.univtac import serve
        serve(args)
    else:
        if len(tasks) != 1 or args.checkpoint is None:
            parser.error("infer requires one task and --checkpoint")
        import numpy as np
        from openpi.serving.univtac import TaskPolicy
        from openpi.training import data_loader
        from openpi.training.univtac_config import make_config
        policy = TaskPolicy(tasks, checkpoint=args.checkpoint, mode=args.model_mode, recipe=args.config)
        config = make_config(tasks[0], args.model_mode, recipe=args.config)
        data = config.data.create(config.assets_dirs, config.model)
        dataset = data_loader.create_torch_dataset(data, config.model.action_horizon, config.model)
        result = policy.infer({**dataset[0], "task": tasks[0]}, noise=np.zeros((50, 32), np.float32))
        actions = np.asarray(result["actions"])
        if actions.shape != (50, 8) or not np.isfinite(actions).all():
            raise ValueError("Invalid restored policy output")
        print(json.dumps({"shape": list(actions.shape), "finite": True}))


if __name__ == "__main__":
    main()
