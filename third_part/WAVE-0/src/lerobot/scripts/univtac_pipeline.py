"""Ordered, resumable launcher for the shared WAVE-0 training workflow.

This module only coordinates child training processes.  Each child remains the
source of truth for checkpoint identity and resume validation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from typing import Callable, Iterable, Mapping, Sequence
import uuid


STAGE2_WAVES = (
    ("insert_hole", "insert_HDMI", "insert_tube", "pull_out_key"),
    ("grasp_classify", "lift_bottle", "lift_can", "put_bottle_in_shelf"),
)
GPU_COUNT = 8
STAGE1_TASK = "shared_wm"
STATUS_FILENAME = "pipeline_status.json"
_PENDING_SIGNAL: int | None = None


class PipelineInterrupted(Exception):
    """Raised by the top-level signal handler to start process cleanup."""

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"pipeline interrupted by signal {signum}")


def _project_root() -> Path:
    from lerobot.utils.univtac_runtime import project_root
    return project_root()


def _resolve_config(config: str | os.PathLike[str]) -> Path:
    path = Path(config).expanduser()
    if not path.is_absolute():
        project_candidate = _project_root() / path
        if project_candidate.exists():
            path = project_candidate
    return path.resolve()


def _physical_gpus() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    values = [str(index) for index in range(GPU_COUNT)] if raw is None else raw.split(",")
    devices = [item.strip() for item in values]
    if len(devices) != GPU_COUNT or any(not item for item in devices):
        raise ValueError(
            "The workflow requires exactly eight non-empty CUDA_VISIBLE_DEVICES entries"
        )
    if len(set(devices)) != GPU_COUNT:
        raise ValueError("CUDA_VISIBLE_DEVICES must contain eight unique GPU IDs")
    return devices


def _load_run(config: str | os.PathLike[str], workspace=None, output=None) -> dict:
    from lerobot.configs.univtac import load_config

    config_path = _resolve_config(config)
    return load_config(config_path, workspace=workspace, output=output)


def plan(config, workspace=None, output=None, stage1_checkpoint=None) -> dict:
    """Return the complete task and physical-GPU mapping without CUDA checks."""
    config_path = _resolve_config(config)
    run = _load_run(config_path, workspace=workspace, output=output)
    if run.get("shared_wm") is not True or run.get("task") != STAGE1_TASK:
        raise ValueError("The pipeline requires the shared_wm_8xa800 recipe without a task override")
    for stage_name, expected_gpus in (("stage1", 8), ("stage2", 2)):
        configured_gpus = run.get(stage_name, {}).get("num_gpus", run.get("num_gpus"))
        if configured_gpus != expected_gpus:
            raise ValueError(
                f"The shared recipe requires {stage_name} to use {expected_gpus} GPUs; "
                f"found {configured_gpus!r}"
            )
    default_output = run.get("workflow_root", run["output"])
    output_root = Path(output if output is not None else default_output).expanduser().resolve()
    workspace_path = Path(workspace or run["workspace"]).expanduser().resolve()
    devices = _physical_gpus()

    stage1 = {
        "stage": "stage1",
        "task": STAGE1_TASK,
        "output": str(output_root / STAGE1_TASK),
        "gpus": devices,
        "nproc_per_node": GPU_COUNT,
        "concurrency": 1,
    }
    external_stage1_checkpoint = (
        str(Path(stage1_checkpoint).expanduser().resolve()) if stage1_checkpoint is not None else None
    )
    if external_stage1_checkpoint is not None:
        stage1["mode"] = "reuse_external_checkpoint"
        stage1["checkpoint"] = external_stage1_checkpoint
    else:
        stage1["mode"] = "train"
    waves = []
    for wave_index, tasks in enumerate(STAGE2_WAVES, start=1):
        jobs = []
        for task_index, task in enumerate(tasks):
            assigned = devices[task_index * 2 : task_index * 2 + 2]
            jobs.append(
                {
                    "stage": "stage2",
                    "task": task,
                    "output": str(output_root / "tasks" / task),
                    "gpus": assigned,
                    "nproc_per_node": 2,
                    "concurrency": 1,
                }
            )
        waves.append({"wave": wave_index, "jobs": jobs})

    return {
        "config": str(config_path),
        "workspace": str(workspace_path),
        "output": str(output_root),
        "stage1_checkpoint": external_stage1_checkpoint,
        "physical_gpus": devices,
        "stage1": stage1,
        "waves": waves,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@contextmanager
def workflow_lock(path: str | os.PathLike[str]):
    """Hold an exclusive, non-blocking lock for one pipeline invocation."""
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another workflow holds {lock_path}") from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _build_training_command(
    *,
    config: str,
    workspace: str,
    output: str,
    stage: str,
    task: str | None,
    threads: int,
    resume: bool,
    stage1_checkpoint: str | None = None,
    rendezvous_id: str | None = None,
) -> list[str]:
    if stage not in ("stage1", "stage2"):
        raise ValueError(f"Unknown training stage: {stage}")
    nproc = GPU_COUNT if stage == "stage1" else 2
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint",
        "localhost:0",
        "--rdzv_id",
        rendezvous_id or str(uuid.uuid4()),
        "--nnodes",
        "1",
        "--nproc_per_node",
        str(nproc),
        "-m",
        "lerobot.scripts.lerobot_train", "--univtac",
        "train",
        "--config",
        config,
        "--workspace",
        workspace,
        "--output",
        output,
        "--stage",
        stage,
        "--device",
        "cuda",
        "--threads",
        str(threads),
    ]
    if stage == "stage2":
        if not task:
            raise ValueError("Stage 2 training requires a task")
        if not stage1_checkpoint:
            raise ValueError("Stage 2 training requires the selected Stage 1 checkpoint")
        command.extend(["--task", task, "--stage1-checkpoint", stage1_checkpoint])
    if resume:
        command.append("--resume")
    return command


def _job_environment(gpus: Sequence[str], base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    if not env.get("OMP_NUM_THREADS"):
        env["OMP_NUM_THREADS"] = "4"
    return env


def _pipeline_environment(threads: int) -> dict[str, str]:
    project = _project_root()
    env = dict(os.environ)
    env["WAVE0_ROOT"] = str(project)
    pythonpath = [str(project / "src")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if not env.get("OMP_NUM_THREADS"):
        env["OMP_NUM_THREADS"] = str(threads)
    return env


def _validate_cuda(devices: Sequence[str]) -> None:
    """Check eight visible logical GPUs with at least 75 GiB CUDA memory."""
    import torch

    count = torch.cuda.device_count()
    if count != GPU_COUNT:
        raise RuntimeError(f"Expected 8 visible GPUs for {list(devices)}, found {count}")
    # An 80 GB A800 reports roughly 79 GiB before driver reservations. Keep
    # some headroom for otherwise valid 80 GB devices.
    minimum_bytes = 75 * 1024**3
    memories = [int(torch.cuda.get_device_properties(index).total_memory) for index in range(count)]
    too_small = [index for index, memory in enumerate(memories) if memory < minimum_bytes]
    if too_small:
        details = ", ".join(f"logical GPU {i}: {memories[i] / 1024**3:.1f} GiB" for i in too_small)
        raise RuntimeError(f"Every workflow GPU must have at least 75 GiB of CUDA memory; {details}")


def _mark_pending_cancelled(status: dict, job_ids: Iterable[str], reason: str) -> None:
    for job_id in job_ids:
        record = status["jobs"][job_id]
        if record["state"] == "pending":
            record["state"] = "cancelled"
            record["error"] = reason
            record["finished_at"] = _utc_now()


def _signal_group_id(process_group_id: int, signum: int) -> None:
    try:
        os.killpg(process_group_id, signum)
    except ProcessLookupError:
        pass


def _signal_group(process: subprocess.Popen, signum: int) -> None:
    _signal_group_id(process.pid, signum)


def _group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_active(
    active: dict[str, subprocess.Popen],
    status: dict,
    validate: Callable[[dict], None] | None,
    persist: Callable[[], None],
    grace_seconds: float,
    reason: str,
    owned_groups: Iterable[int] = (),
) -> None:
    # A torchrun leader can exit while a worker remains in its process group.
    # Keep group IDs for the whole wave and clean all of them on failure.
    groups = set(owned_groups)
    groups.update(process.pid for process in active.values())
    for process_group_id in groups:
        _signal_group_id(process_group_id, signal.SIGTERM)

    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline:
        leaders_done = all(process.poll() is not None for process in active.values())
        groups_alive = any(_group_exists(process_group_id) for process_group_id in groups)
        if leaders_done and not groups_alive:
            break
        time.sleep(0.025)

    # Reap stragglers even when their process-group leader already exited.
    # These IDs were created by this wave and are used only during short cleanup.
    for process_group_id in groups:
        _signal_group_id(process_group_id, signal.SIGKILL)

    for job_id, process in active.items():
        try:
            return_code = process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _signal_group(process, signal.SIGKILL)
            return_code = process.wait()
        record = status["jobs"][job_id]
        record["exit_code"] = return_code
        record["finished_at"] = _utc_now()
        if return_code == 0 and validate is not None:
            try:
                validate(record)
            except Exception as exc:
                record["state"] = "failed"
                record["error"] = str(exc)
            else:
                record["state"] = "done"
        elif return_code not in (0, None) and return_code > 0:
            record["state"] = "failed"
            record["error"] = f"child exited with status {return_code}"
        else:
            record["state"] = "cancelled"
            record["error"] = reason
    persist()


def _raise_if_interrupted() -> None:
    if _PENDING_SIGNAL is not None:
        raise PipelineInterrupted(_PENDING_SIGNAL)


def run_wave(
    job_ids: Sequence[str],
    *,
    status: dict,
    persist: Callable[[], None],
    spawn: Callable[[dict], subprocess.Popen],
    validate: Callable[[dict], None] | None = None,
    max_concurrent: int = 4,
    poll_interval: float = 0.05,
    terminate_grace: float = 10.0,
) -> bool:
    """Run one wave, stopping all owned process groups after the first failure."""
    if max_concurrent < 1:
        raise ValueError("max_concurrent must be positive")
    records = [status["jobs"][job_id] for job_id in job_ids]
    claimed: set[str] = set()
    for record in records:
        overlap = claimed.intersection(record.get("gpus", []))
        if overlap:
            raise ValueError(f"A wave assigns GPUs to multiple jobs: {sorted(overlap)}")
        claimed.update(record.get("gpus", []))

    active: dict[str, subprocess.Popen] = {}
    owned_groups: set[int] = set()
    next_index = 0
    failure_reason: str | None = None

    def start_available() -> None:
        nonlocal next_index, failure_reason
        while failure_reason is None and len(active) < max_concurrent and next_index < len(job_ids):
            job_id = job_ids[next_index]
            next_index += 1
            record = status["jobs"][job_id]
            record["state"] = "running"
            record["started_at"] = _utc_now()
            persist()
            try:
                process = spawn(record)
            except Exception as exc:
                record["state"] = "failed"
                record["error"] = f"could not start child: {exc}"
                record["finished_at"] = _utc_now()
                failure_reason = record["error"]
                persist()
                return
            record["pid"] = process.pid
            active[job_id] = process
            owned_groups.add(process.pid)
            persist()

    try:
        start_available()
        _raise_if_interrupted()
        if failure_reason is not None:
            _stop_active(
                active,
                status,
                validate,
                persist,
                terminate_grace,
                failure_reason,
                owned_groups,
            )
            active.clear()
            _mark_pending_cancelled(status, job_ids[next_index:], failure_reason)
            persist()
            return False
        while active:
            _raise_if_interrupted()
            for job_id, process in list(active.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                del active[job_id]
                record = status["jobs"][job_id]
                record["exit_code"] = int(return_code)
                record["finished_at"] = _utc_now()
                if return_code != 0:
                    record["state"] = "failed"
                    record["error"] = f"child exited with status {return_code}"
                    failure_reason = record["error"]
                else:
                    try:
                        if validate is not None:
                            validate(record)
                    except Exception as exc:
                        record["state"] = "failed"
                        record["error"] = str(exc)
                        failure_reason = record["error"]
                    else:
                        record["state"] = "done"
                persist()
                if failure_reason is not None:
                    break

            if failure_reason is not None:
                _stop_active(
                    active,
                    status,
                    validate,
                    persist,
                    terminate_grace,
                    failure_reason,
                    owned_groups,
                )
                active.clear()
                _mark_pending_cancelled(status, job_ids[next_index:], failure_reason)
                persist()
                return False

            start_available()
            if failure_reason is not None:
                _stop_active(
                    active,
                    status,
                    validate,
                    persist,
                    terminate_grace,
                    failure_reason,
                    owned_groups,
                )
                active.clear()
                _mark_pending_cancelled(status, job_ids[next_index:], failure_reason)
                persist()
                return False
            if active:
                time.sleep(poll_interval)

        _raise_if_interrupted()
        return failure_reason is None
    except BaseException as exc:
        reason = str(exc) or type(exc).__name__
        try:
            _stop_active(
                active,
                status,
                validate,
                lambda: None,
                terminate_grace,
                reason,
                owned_groups,
            )
        except BaseException:
            # Keep the original exception after a best-effort cleanup.
            pass
        active.clear()
        for job_id in job_ids:
            record = status["jobs"][job_id]
            if record["state"] == "running":
                record["state"] = "cancelled"
                record["error"] = reason
                record["finished_at"] = _utc_now()
        _mark_pending_cancelled(status, job_ids[next_index:], reason)
        try:
            persist()
        except BaseException:
            pass
        raise


def _make_job_record(job_id: str, spec: Mapping) -> dict:
    return {
        "id": job_id,
        "stage": spec["stage"],
        "task": spec["task"],
        "wave": spec.get("wave"),
        "output": spec["output"],
        "gpus": list(spec["gpus"]),
        "command": None,
        "log": None,
        "state": "pending",
        "exit_code": None,
    }


def _command_for_spec(
    spec: Mapping,
    *,
    schedule: Mapping,
    threads: int,
    resume: bool,
    stage1_checkpoint: str | None = None,
) -> list[str]:
    return _build_training_command(
        config=schedule["config"],
        workspace=schedule["workspace"],
        # The shared recipe expands this pipeline root to ROOT/shared_wm for
        # Stage 1 and ROOT/tasks/TASK when --task is supplied for Stage 2.
        output=schedule["output"],
        stage=spec["stage"],
        task=None if spec["stage"] == "stage1" else spec["task"],
        threads=threads,
        resume=resume,
        stage1_checkpoint=stage1_checkpoint,
    )


def _spawn_logged(record: dict, env: Mapping[str, str]) -> subprocess.Popen:
    log_path = Path(record["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_utc_now()}] {shlex.join(record['command'])}\n")
        log.flush()
        return subprocess.Popen(
            record["command"],
            cwd=_project_root(),
            env=_job_environment(record["gpus"], env),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def _run_preflight(
    record: dict,
    *,
    status: dict,
    persist: Callable[[], None],
    env: Mapping[str, str],
) -> int:
    log_path = Path(record["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record["state"] = "running"
    record["started_at"] = _utc_now()
    persist()
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{_utc_now()}] {shlex.join(record['command'])}\n")
            log.flush()
            process = subprocess.Popen(
                record["command"],
                cwd=_project_root(),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        record["pid"] = process.pid
        persist()
        while process.poll() is None:
            _raise_if_interrupted()
            time.sleep(0.05)
        _raise_if_interrupted()
        record["exit_code"] = int(process.returncode)
        record["state"] = "done" if process.returncode == 0 else "failed"
        record["finished_at"] = _utc_now()
        if process.returncode:
            record["error"] = f"preflight exited with status {process.returncode}"
        persist()
        return int(process.returncode)
    except PipelineInterrupted as exc:
        if "process" in locals() and process.poll() is None:
            _signal_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _signal_group(process, signal.SIGKILL)
                process.wait()
        record["state"] = "cancelled"
        record["exit_code"] = process.poll() if "process" in locals() else None
        record["error"] = str(exc)
        record["finished_at"] = _utc_now()
        persist()
        raise
    except Exception as exc:
        record["state"] = "failed"
        record["error"] = f"could not run preflight: {exc}"
        record["finished_at"] = _utc_now()
        persist()
        return 127


def _preflight_commands(schedule: Mapping) -> list[dict]:
    common = ["--config", schedule["config"], "--workspace", schedule["workspace"], "--output", schedule["output"]]
    project = _project_root()
    return [
        {
            "name": "verify_assets",
            "command": [sys.executable, str(project / "scripts/debug/univtac/verify_assets.py"), *common],
            "log": str(Path(schedule["output"]) / "logs" / "preflight_verify_assets.log"),
            "state": "pending",
            "exit_code": None,
        },
        {
            "name": "prepare",
            "command": [
                sys.executable,
                "-m",
                "lerobot.scripts.lerobot_train", "--univtac",
                "prepare",
                "--verify-hashes",
                *common,
            ],
            "log": str(Path(schedule["output"]) / "logs" / "preflight_prepare.log"),
            "state": "pending",
            "exit_code": None,
        },
    ]


def _initial_status(schedule: Mapping, resume: bool, old_status: Mapping | None) -> dict:
    history = list(old_status.get("history", [])) if old_status else []
    if old_status:
        history.append(
            {
                "attempt": old_status.get("attempt"),
                "state": old_status.get("state"),
                "started_at": old_status.get("started_at"),
                "finished_at": old_status.get("finished_at"),
                "jobs": old_status.get("jobs", {}),
                "preflight": old_status.get("preflight", []),
                "error": old_status.get("error"),
            }
        )
    jobs: dict[str, dict] = {"stage1": _make_job_record("stage1", schedule["stage1"])}
    for wave in schedule["waves"]:
        for spec in wave["jobs"]:
            job_id = spec["task"]
            jobs[job_id] = _make_job_record(job_id, {**spec, "wave": wave["wave"]})
    return {
        "schema_version": 1,
        "run_id": old_status.get("run_id") if old_status else str(uuid.uuid4()),
        "attempt": int(old_status.get("attempt", 0)) + 1 if old_status else 1,
        "state": "running",
        "resume": bool(resume),
        "config": schedule["config"],
        "workspace": schedule["workspace"],
        "output": schedule["output"],
        "plan": schedule,
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "finished_at": None,
        "preflight": _preflight_commands(schedule),
        "jobs": jobs,
        "stage1_checkpoint": schedule.get("stage1_checkpoint"),
        "history": history,
    }


def _cancel_pending(status: dict, reason: str) -> None:
    for record in status["jobs"].values():
        if record["state"] == "pending":
            record["state"] = "cancelled"
            record["error"] = reason
            record["finished_at"] = _utc_now()
    for record in status["preflight"]:
        if record["state"] == "pending":
            record["state"] = "cancelled"
            record["error"] = reason
            record["finished_at"] = _utc_now()


def _validate_stage1_result(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"Stage 1 result is missing: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("complete") is not True:
        raise RuntimeError("Stage 1 result does not report complete=true")
    if not result.get("identity"):
        raise RuntimeError("Stage 1 result is missing its training identity")
    if result.get("formal_training") is not True:
        raise RuntimeError("Stage 1 result does not report formal_training=true")
    selected = result.get("selected_checkpoint")
    if not isinstance(selected, str) or not Path(selected).is_absolute():
        raise RuntimeError("Stage 1 result must contain an absolute selected_checkpoint")
    checkpoint = Path(selected)
    if not checkpoint.is_dir():
        raise RuntimeError(f"Selected Stage 1 checkpoint directory is missing: {checkpoint}")
    complete_path = checkpoint / "complete.json"
    identity_path = checkpoint / "run_identity.json"
    if not complete_path.is_file() or not identity_path.is_file():
        raise RuntimeError(f"Selected Stage 1 checkpoint is incomplete: {checkpoint}")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    checkpoint_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    fingerprint = result["identity"]
    if complete.get("stage") != "stage1" or complete.get("identity") != fingerprint:
        raise RuntimeError("Selected Stage 1 checkpoint metadata does not match the result identity")
    if (
        checkpoint_identity.get("fingerprint") != fingerprint
        or checkpoint_identity.get("recipe", {}).get("task") != STAGE1_TASK
        or checkpoint_identity.get("debug") is not False
    ):
        raise RuntimeError("Selected Stage 1 checkpoint run identity does not match formal shared training")
    return str(checkpoint.resolve())


def _validate_external_stage1_checkpoint(path: str | os.PathLike[str], schedule: Mapping) -> str:
    """Validate a reused formal shared-WM checkpoint against prepared run assets."""
    checkpoint = Path(path).expanduser().resolve()
    result_path = checkpoint.parent.parent / "result.json"
    selected = _validate_stage1_result(result_path)
    if selected != str(checkpoint):
        raise RuntimeError(
            "External Stage 1 checkpoint is not the formally selected shared WM checkpoint"
        )

    # Reuse the training-side checks for task-set, normalization, and weight
    # hashes. The prepare preflight runs first so this checks the current run's
    # freshly verified normalization file.
    from lerobot.scripts.lerobot_train import check_shared_teacher

    run = _load_run(schedule["config"], workspace=schedule["workspace"], output=schedule["output"])
    run["stage1_checkpoint"] = str(checkpoint)
    verified = check_shared_teacher(run, debug=False)
    if Path(verified).resolve() != checkpoint:
        raise RuntimeError("Shared WM teacher validation returned a different checkpoint")
    return str(checkpoint)


def _validate_stage2_result(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"Stage 2 result is missing: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("complete") is not True:
        raise RuntimeError(f"Stage 2 result does not report complete=true: {path}")
    if result.get("formal_training") is not True:
        raise RuntimeError(f"Stage 2 result does not report formal_training=true: {path}")
    if not result.get("identity"):
        raise RuntimeError(f"Stage 2 result is missing its training identity: {path}")


def _signal_handlers():
    @contextmanager
    def installed():
        global _PENDING_SIGNAL
        previous = {}
        _PENDING_SIGNAL = None

        def handle(signum, frame):
            global _PENDING_SIGNAL
            _PENDING_SIGNAL = signum

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, handle)
        try:
            yield
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            _PENDING_SIGNAL = None

    return installed()


def main(
    config,
    workspace=None,
    output=None,
    resume=False,
    plan_only=False,
    threads=4,
    stage1_checkpoint=None,
):
    """Run the full workflow, returning its durable status document."""
    if threads < 1:
        raise ValueError("threads must be positive")
    schedule = plan(
        config,
        workspace=workspace,
        output=output,
        stage1_checkpoint=stage1_checkpoint,
    )
    if plan_only:
        return schedule

    output_root = Path(schedule["output"])
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / STATUS_FILENAME
    lock_path = output_root / ".pipeline.lock"

    with workflow_lock(lock_path):
        old_status = None
        if status_path.exists():
            if not resume:
                raise FileExistsError(
                    f"Workflow status already exists at {status_path}; pass --resume to continue"
                )
            old_status = json.loads(status_path.read_text(encoding="utf-8"))
            if old_status.get("plan") != schedule:
                raise RuntimeError("Cannot resume because the saved pipeline plan differs from this invocation")

        status = _initial_status(schedule, resume, old_status)

        def persist():
            status["updated_at"] = _utc_now()
            _atomic_json(status_path, status)

        env = _pipeline_environment(threads)
        persist()

        def failed(error: Exception | str, state: str = "failed") -> dict:
            status["state"] = state
            status["error"] = str(error)
            status["finished_at"] = _utc_now()
            _cancel_pending(status, str(error))
            persist()
            return status

        try:
            with _signal_handlers():
                for preflight in status["preflight"]:
                    print(
                        f"[wave0-pipeline] preflight {preflight['name']} log={preflight['log']}",
                        flush=True,
                    )
                    code = _run_preflight(preflight, status=status, persist=persist, env=env)
                    if code != 0:
                        return failed(preflight.get("error", f"{preflight['name']} failed"))

                _validate_cuda(schedule["physical_gpus"])

                stage1_record = status["jobs"]["stage1"]
                if schedule.get("stage1_checkpoint"):
                    try:
                        selected_checkpoint = _validate_external_stage1_checkpoint(
                            schedule["stage1_checkpoint"], schedule
                        )
                    except Exception as exc:
                        stage1_record["state"] = "failed"
                        stage1_record["error"] = str(exc)
                        persist()
                        return failed(exc)
                    stage1_result = Path(selected_checkpoint).parent.parent / "result.json"
                    stage1_record.update(
                        state="done",
                        source="external",
                        started_at=_utc_now(),
                        result=str(stage1_result),
                        selected_checkpoint=selected_checkpoint,
                        finished_at=_utc_now(),
                    )
                    persist()
                    print(
                        f"[wave0-pipeline] reusing external stage1 checkpoint={selected_checkpoint}",
                        flush=True,
                    )
                else:
                    stage1_record["command"] = _command_for_spec(
                        schedule["stage1"], schedule=schedule, threads=threads, resume=resume
                    )
                    stage1_record["log"] = str(output_root / "logs" / "stage1.log")
                    persist()
                    print(
                        f"[wave0-pipeline] stage1 on 8 GPUs log={stage1_record['log']}",
                        flush=True,
                    )

                    if not run_wave(
                        ["stage1"],
                        status=status,
                        persist=persist,
                        spawn=lambda record: _spawn_logged(record, env),
                        max_concurrent=1,
                    ):
                        return failed(status["jobs"]["stage1"].get("error", "Stage 1 failed"))

                    stage1_result = output_root / "shared_wm" / "stage1" / "result.json"
                    try:
                        selected_checkpoint = _validate_stage1_result(stage1_result)
                    except Exception as exc:
                        stage1_record["state"] = "failed"
                        stage1_record["error"] = str(exc)
                        persist()
                        return failed(exc)
                    stage1_record["result"] = str(stage1_result)
                    stage1_record["selected_checkpoint"] = selected_checkpoint
                    persist()

                wave_ids = []
                for wave in schedule["waves"]:
                    current_ids = []
                    for spec in wave["jobs"]:
                        job_id = spec["task"]
                        record = status["jobs"][job_id]
                        record["command"] = _command_for_spec(
                            spec,
                            schedule=schedule,
                            threads=threads,
                            resume=resume,
                            stage1_checkpoint=selected_checkpoint,
                        )
                        record["log"] = str(output_root / "logs" / f"{job_id}.log")
                        record["result"] = str(output_root / "tasks" / job_id / "stage2" / "result.json")
                        current_ids.append(job_id)
                    wave_ids.append(current_ids)
                persist()

                for index, job_ids in enumerate(wave_ids):
                    def validate_stage2(record: dict):
                        _validate_stage2_result(Path(record["result"]))

                    print(
                        f"[wave0-pipeline] stage2 wave {index + 1}/2 tasks={','.join(job_ids)}",
                        flush=True,
                    )
                    for job_id in job_ids:
                        record = status["jobs"][job_id]
                        print(
                            f"[wave0-pipeline] {job_id} GPUs={','.join(record['gpus'])} log={record['log']}",
                            flush=True,
                        )

                    if not run_wave(
                        job_ids,
                        status=status,
                        persist=persist,
                        spawn=lambda record: _spawn_logged(record, env),
                        validate=validate_stage2,
                        max_concurrent=4,
                    ):
                        return failed(
                            next(
                                (
                                    status["jobs"][job_id].get("error")
                                    for job_id in job_ids
                                    if status["jobs"][job_id]["state"] == "failed"
                                ),
                                f"Stage 2 wave {index + 1} failed",
                            )
                        )

                status["state"] = "done"
                status["finished_at"] = _utc_now()
                persist()
                return status
        except PipelineInterrupted as exc:
            return failed(exc, state="cancelled")
        except Exception as exc:
            return failed(exc)


def cli_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ordered shared WAVE-0 training workflow")
    parser.add_argument("--config", required=True, help="shared_wm_8xa800.yaml")
    parser.add_argument("--workspace", help="dataset and model asset workspace")
    parser.add_argument("--output", required=True, help="pipeline output root")
    parser.add_argument("--resume", action="store_true", help="resume all child runs with their identity checks")
    parser.add_argument(
        "--stage1-checkpoint",
        help="reuse a completed, formally selected shared WM checkpoint instead of training Stage 1",
    )
    parser.add_argument("--plan", action="store_true", help="print the GPU/task plan without CUDA checks")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        result = main(
            args.config,
            workspace=args.workspace,
            output=args.output,
            resume=args.resume,
            plan_only=args.plan,
            threads=args.threads,
            stage1_checkpoint=args.stage1_checkpoint,
        )
    except Exception as exc:
        print(f"pipeline: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.plan:
        return 0
    if result["state"] == "done":
        return 0
    if result["state"] == "cancelled":
        return 130
    return 1


if __name__ == "__main__":
    raise SystemExit(cli_main())
