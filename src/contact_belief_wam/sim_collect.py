from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import importlib
import json
import os
import signal
from pathlib import Path
import sys
import time
import traceback

from .config import RecoveryConfig
from .paths import WORKSPACE, atomic_json


def main(argv=None):
    os.chdir(WORKSPACE / "UniVTAC")
    sys.path.insert(0, str(WORKSPACE / "UniVTAC"))
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("insert_hole", "insert_HDMI"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--start-seed", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    import yaml
    cfg_values = yaml.safe_load(args.config.read_text()) if args.config else {}
    cfg = RecoveryConfig(**(cfg_values or {}))
    if args.episodes is not None:
        cfg.successful_episodes_per_task = args.episodes
    if args.start_seed is not None:
        cfg.seed = args.start_seed
    if args.max_attempts is not None:
        cfg.max_attempts = args.max_attempts
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.headless, args.enable_cameras, args.livestream = True, True, 0
    launcher = AppLauncher(args)
    app = launcher.app
    # Isaac installs its own handlers. A requested cancellation must leave the
    # collector's try/finally so an old process cannot keep consuming the GPU.
    def cancel_collection(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, cancel_collection)
    task = None
    try:
        from .recovery import RecoveryController, plot_recovery
        module = importlib.import_module(f"envs.{args.task}")
        task_cfg = module.TaskCfg()
        source = yaml.safe_load((WORKSPACE / "UniVTAC/task_config/clean.yml").read_text())
        task_cfg.save_dir = args.output
        task_cfg.decimation = 1
        task_cfg.save_frequency = 2
        task_cfg.video_frequency = 2
        task_cfg.render_frequency = 0
        task_cfg.tactile_sensor_type = "gsmini"
        task_cfg.obs_data_type = source["observations"]
        task_cfg.random_texture = source.get("random_texture", False)
        task_cfg.max_save_frames = 5000
        task_cfg.reset_time_limit = 300
        task_cfg.sim.device = args.device or "cuda:0"
        task = module.Task(task_cfg, mode="collect")
        signal.signal(signal.SIGTERM, cancel_collection)
        prompt = json.loads((WORKSPACE / "UniVTAC/instructions" / f"{args.task}.json").read_text())["seen"][0]
        records = []
        report_path = args.output / "collection.json"
        if report_path.exists():
            records = json.loads(report_path.read_text())["attempts"]
        for record in records:
            if record["success"] and not (args.output / "hdf5" / f"{record['seed']}.hdf5").is_file():
                record.update(success=False, error="Recorded HDF5 is missing")
        successes = sum(r["success"] and not r.get("excluded_from_training", False) for r in records)
        seen = {r["seed"] for r in records}
        atomic_json(args.output / "recovery_config.json", asdict(cfg))
        for seed in range(cfg.seed, cfg.seed + cfg.max_attempts):
            if successes >= cfg.successful_episodes_per_task:
                break
            if seed in seen:
                continue
            controller = None
            begin = time.monotonic()
            record = {"seed": seed, "success": False, "controller_version": "local-recovery-v6"}
            try:
                task.reset(seed=seed, instructions=[prompt])
                signal.signal(signal.SIGTERM, cancel_collection)
                controller = RecoveryController(task, args.task, cfg)
                success = controller.run()
                record.update(blockages=controller.blockages)
                if success and controller.blockages >= 1:
                    task.save_to_hdf5()
                    detail = args.output / "diagnostics" / f"{seed}.json"
                    atomic_json(detail, controller.report())
                    if successes < 3:
                        plot_recovery(detail)
                    record["success"] = True
                    successes += 1
                else:
                    record["success"] = False
                    atomic_json(args.output / "debug" / f"{seed}.json", controller.report())
            except Exception as exc:
                record.update(success=False, error=str(exc), traceback=traceback.format_exc())
                if controller:
                    atomic_json(args.output / "debug" / f"{seed}.json", controller.report())
            finally:
                if controller:
                    controller.close()
                task.clean_cache(result="success" if record["success"] else "fail")
                # Failed attempts are diagnostics, not expert training episodes.
                if not record["success"]:
                    saved = args.output / "hdf5" / f"{seed}.hdf5"
                    if saved.exists():
                        quarantine = args.output / "debug" / f"{seed}.incomplete.hdf5"
                        quarantine.parent.mkdir(parents=True, exist_ok=True)
                        saved.replace(quarantine)
                    for video in (args.output / "video").glob(f"{seed}*.mp4"):
                        if len(records) > 2:
                            video.unlink()
                record["elapsed_seconds"] = time.monotonic() - begin
                records.append(record)
                atomic_json(report_path, {"task": args.task, "requested": cfg.successful_episodes_per_task,
                                         "successful": successes, "attempts": records})
                print(json.dumps({"task": args.task, "seed": seed, "success": record["success"],
                                  "collected": successes, "error": record.get("error")}), flush=True)
        if successes < cfg.successful_episodes_per_task:
            raise RuntimeError(f"Collection incomplete: {successes}/{cfg.successful_episodes_per_task}; see {report_path}")
    finally:
        if task is not None:
            task.close()
        app.close()


if __name__ == "__main__":
    main()
