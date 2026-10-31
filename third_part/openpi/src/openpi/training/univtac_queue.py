"""One fail-fast, resumable task queue shared by local debug and A800 training."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import subprocess
import sys
import time
import uuid

from openpi.shared.univtac_runtime import root, atomic_json


def now():
    return datetime.now(timezone.utc).isoformat()


def configuration(task, mode, run_name, recipe=None):
    from openpi.training.univtac_config import make_config
    return make_config(task, mode, run=run_name, recipe=recipe)


def invoke(module, arguments, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        subprocess.run([sys.executable, "-m", module, *map(str, arguments)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def _validate_queue_report(report, tasks, mode):
    if not isinstance(report, dict):
        raise RuntimeError("Queue resume refused: prior report is not an object")
    if (report.get("task_order") != list(tasks) or report.get("mode") != mode):
        raise RuntimeError("Queue resume refused: task order or mode changed")
    rows = report.get("tasks")
    if (not isinstance(rows, list) or len(rows) != len(tasks)
            or any(not isinstance(row, dict) for row in rows)
            or [row.get("task") for row in rows] != list(tasks)):
        raise RuntimeError("Queue resume refused: prior task rows do not match the task order")


def run_queue(tasks, mode, run_name, *, resume=False, recipe=None):
    if recipe is None:
        raise ValueError("A queue requires an explicit experiment config path")
    tasks = tuple(tasks)
    if mode not in ("lora", "smoke", "full") or not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Expected unique tasks and a lora/smoke/full queue mode")
    configs = {task: configuration(task, mode, run_name, recipe) for task in tasks}
    directory = root() / "outputs/queues" / mode / run_name
    if directory.exists() and not resume:
        raise FileExistsError(f"Queue exists: {directory}; use a new --run or explicit --resume")
    path = directory / "report.json"
    if resume and not path.is_file():
        raise RuntimeError(f"Queue resume refused: matching prior queue report is required: {path}")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"This queue is already running: {directory}") from exc
        allowed_incomplete = set()
        if path.exists():
            report = json.loads(path.read_text())
            _validate_queue_report(report, tasks, mode)
            if not resume:
                raise FileExistsError(f"Queue exists: {directory}; use a new --run or explicit --resume")
        else:
            if resume:
                raise RuntimeError(f"Queue resume refused: matching prior queue report is required: {path}")
            report = {"mode": mode, "run": run_name, "task_order": list(tasks),
                      "created_at": now(), "tasks": [{"task": task, "status": "pending"} for task in tasks]}
        # Check every input/output before launching the first task. A checkpoint
        # directory without a committed step may be archived only when the
        # prior queue report proves that this exact task was already launched.
        for index, task in enumerate(tasks):
            config = configs[task]
            if not config.checkpoint_dir.exists():
                continue
            saved = any(p.is_dir() and p.name.isdigit() for p in config.checkpoint_dir.iterdir())
            if saved:
                if not resume:
                    raise FileExistsError(config.checkpoint_dir)
                continue
            if not resume:
                raise FileExistsError(config.checkpoint_dir)
            row = report["tasks"][index]
            if (row.get("status") not in ("running", "failed")
                    or row.get("checkpoint_dir") != str(config.checkpoint_dir)):
                raise RuntimeError(f"Queue resume refused: incomplete task directory is not recorded for {task}")
            allowed_incomplete.add(task)
        report.update(status="running", passed=False, returncode=None, invocation_started_at=now())
        atomic_json(path, report)
        target = configs[tasks[0]].num_train_steps
        active = None
        try:
            for index, task in enumerate(tasks):
                active = report["tasks"][index]
                config = configs[task]
                # Each task initializes from base in its own fresh process.
                # No previous task checkpoint is ever passed as a weight source.
                begin = time.monotonic()
                active.update(status="running", started_at=now(), initialization="official_pi05_base",
                              checkpoint_dir=str(config.checkpoint_dir), target_steps=target)
                if config.checkpoint_dir.exists() and not any(
                    p.is_dir() and p.name.isdigit() for p in config.checkpoint_dir.iterdir()
                ):
                    if task not in allowed_incomplete:
                        raise RuntimeError("Refusing to restart an unverified incomplete task")
                    archived = config.checkpoint_dir.with_name(f"{config.checkpoint_dir.name}.incomplete.{uuid.uuid4().hex[:8]}")
                    config.checkpoint_dir.rename(archived)
                    active.setdefault("preserved_incomplete_runs", []).append(str(archived))
                report["active_task"] = task
                atomic_json(path, report)
                print(f"[{index + 1}/{len(tasks)}] {task}: {mode} training started", flush=True)
                marker = config.checkpoint_dir / "final.json"
                complete = json.loads(marker.read_text()) if marker.exists() else {}
                completed = complete.get("passed") and complete.get("step", 0) >= target
                if not completed:
                    stages = (10, 20) if mode == "lora" else (50, 100) if mode == "smoke" else (target,)
                    for stop in stages:
                        saved = [int(p.name) for p in config.checkpoint_dir.iterdir()
                                 if p.is_dir() and p.name.isdigit()] if config.checkpoint_dir.exists() else []
                        latest = max(saved, default=0)
                        if latest >= stop and stop != target:
                            continue
                        arguments = ["--task", task, "--mode", mode, "--run", run_name, "--steps", stop]
                        if saved:
                            arguments.append("--resume")
                        arguments += ["--config", str(recipe)]
                        invoke("openpi.training.univtac_trainer", arguments, directory / "logs" / f"{task}-train-{stop}.log")
                trained = json.loads((config.checkpoint_dir / "training_report.json").read_text())
                if not trained.get("passed") or trained.get("end_step", 0) < target:
                    raise RuntimeError(f"Training acceptance failed for {task}")
                if mode in ("lora", "smoke") and not trained.get("resume_verified"):
                    raise RuntimeError(f"Independent-process restore not verified for {task}")
                if mode == "lora":
                    invoke("openpi.training.univtac_cli", ["infer", "--task", task, "--checkpoint",
                           config.checkpoint_dir / str(trained["end_step"]), "--model-mode", mode],
                           directory / "logs" / f"{task}-inference.log")
                active.update(status="passed", finished_at=now(), elapsed_seconds=time.monotonic() - begin,
                              end_step=trained["end_step"], resume_verified=trained["resume_verified"],
                              checkpoint=str(config.checkpoint_dir / str(trained["end_step"])), reused_complete_run=bool(completed))
                atomic_json(path, report)
                print(f"[{index + 1}/{len(tasks)}] {task}: passed; next task starts automatically", flush=True)
            report.update(status="completed", passed=True, returncode=0, active_task=None, finished_at=now())
            atomic_json(path, report)
            print(f"Queue completed: {len(tasks)}/{len(tasks)}, exit=0, report={path}", flush=True)
            return report
        except BaseException as exc:
            if active is not None:
                active.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=now())
            report.update(status="failed", passed=False, returncode=getattr(exc, "returncode", 1), finished_at=now())
            atomic_json(path, report)
            raise
