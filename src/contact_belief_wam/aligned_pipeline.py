"""Prepare, inspect, and manually launch the aligned WAVE-0 training recipe."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import fields
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
import uuid

import yaml

from .paths import PROJECT, atomic_json, configure


TASKS = ("insert_hole", "insert_HDMI", "insert_tube", "pull_out_key",
         "grasp_classify", "lift_bottle", "lift_can", "put_bottle_in_shelf")
POLICY_TASKS = ("insert_hole", "insert_HDMI")
GPU_COUNT = 8
STATUS_FILENAME = "aligned_pipeline_status.json"


def _resolve_path(value, base=PROJECT):
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def _read_norm(path):
    value = json.loads(Path(path).read_text())
    return value.get("norm_stats", value)


def _model_values(raw, mode):
    from .config import ContactConfig
    available = {field.name for field in fields(ContactConfig)}
    values = {key: value for key, value in raw.items() if key in available}
    values.update(mode=mode, aligned=True, stage="stage1", dtype="float32",
                  gradient_checkpointing=True, k=int(raw.get("k", 10)),
                  attention_backend=raw.get("attention_backend", "sdpa"))
    if mode == "wave0":
        values.update(belief_warmup_min=0, belief_warmup_max=0, belief_bptt_steps=1)
    return values


def resolve_config(config="configs/aligned_8xa800.yaml", *, variant="full", output=None):
    """Resolve portable paths and the explicit output namespace for a variant."""
    if variant not in {"full", "wave0_recovery"}:
        raise ValueError("variant must be full or wave0_recovery")
    config_path = Path(config).expanduser()
    if not config_path.is_absolute():
        candidate = PROJECT / config_path
        if candidate.exists():
            config_path = candidate
    config_path = config_path.resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw.get("tasks") != list(TASKS):
        raise ValueError("Aligned Stage 1 must use all eight canonical tasks in order")
    if raw.get("policy_tasks") != list(POLICY_TASKS):
        raise ValueError("Aligned Stage 2 must train insert_hole and insert_HDMI")
    raw["config"] = str(config_path)
    raw["variant"] = variant
    raw["mode"] = "full" if variant == "full" else "wave0"
    raw["aligned"] = True
    for key in ("data_root", "original_root", "recovery_root", "tactile_cache", "pi05_base",
                "tokenizer_path", "world_init", "shared_manifest", "policy_manifest_dir",
                "task_stats_dir", "wm_stats"):
        if raw.get(key):
            raw[key] = _resolve_path(raw[key])
    raw["workflow_root"] = str(Path(output or raw["output"]).expanduser().resolve() / variant)
    raw["output"] = raw["workflow_root"]
    raw["workspace"] = str(Path(raw.get("workspace", PROJECT.parent)).expanduser().resolve())
    raw["tasks"] = list(TASKS)
    raw["policy_tasks"] = list(POLICY_TASKS)
    raw.setdefault("workers", 4)
    raw.setdefault("seed", 42)
    raw.setdefault("precision", "bf16")
    raw.setdefault("ema_decay", 0.99)
    raw.setdefault("grad_clip", 1.0)
    raw.setdefault("fused_adamw", True)
    raw.setdefault("gradient_checkpointing", True)
    raw.setdefault("data_mix", "1:1")
    raw.setdefault("log_every", 10)
    raw.setdefault("save_every", 1000)
    raw.setdefault("keep_checkpoints", 2)
    if raw["precision"] != "bf16":
        raise ValueError("Aligned formal training uses BF16 autocast with FP32 parameters")
    for name, expected in (("stage1", 8), ("stage2", 2)):
        stage = raw.get(name, {})
        if stage.get("num_gpus") != expected:
            raise ValueError(f"{name} must use {expected} GPUs")
        divisor = stage["num_gpus"] * stage["micro_batch"]
        if stage["global_batch"] % divisor:
            raise ValueError(f"{name}: global_batch must divide num_gpus × micro_batch")
        stage["accumulation_steps"] = stage["global_batch"] // divisor
        if stage["accumulation_steps"] != 1:
            raise ValueError(f"{name}: this recipe requires gradient accumulation 1")
    raw["model_config"] = _model_values(raw, raw["mode"])
    raw["shared_manifest"] = raw.get("shared_manifest")
    raw["policy_manifests"] = {}
    raw["task_stats_by_task"] = {}
    raw["wm_norm_stats"] = None
    raw["policy_norm_stats"] = None
    raw["stage1_checkpoint"] = None
    raw["stage1_checkpoint_identity"] = None
    return raw


def _physical_gpus():
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = [str(index) for index in range(GPU_COUNT)] if raw is None else [item.strip() for item in raw.split(",")]
    if len(devices) != GPU_COUNT or any(not item for item in devices):
        raise ValueError("The aligned workflow requires exactly eight non-empty CUDA_VISIBLE_DEVICES entries")
    if len(set(devices)) != GPU_COUNT:
        raise ValueError("CUDA_VISIBLE_DEVICES must contain eight unique GPU IDs")
    return devices


def plan(run):
    devices = _physical_gpus()
    stage1 = {"stage": "stage1", "task": "shared_wm",
              "output": str(Path(run["output"]) / "shared_wm"), "gpus": devices,
              "nproc_per_node": 8, "concurrency": 1}
    stage2 = [{"stage": "stage2", "task": task,
               "output": str(Path(run["output"]) / "tasks" / task),
               "gpus": devices[index * 2:index * 2 + 2], "nproc_per_node": 2,
               "concurrency": 2} for index, task in enumerate(POLICY_TASKS)]
    return {"config": run["config"], "variant": run["variant"], "mode": run["mode"],
            "workspace": run["workspace"], "output": run["output"],
            "physical_gpus": devices, "stage1": stage1, "stage2": stage2}


def stage_run(run, stage, task=None, checkpoint=None):
    result = dict(run)
    if stage == "stage1":
        result.update(task="shared_wm", output=str(Path(run["output"]) / "shared_wm"),
                      task_manifest=None, task_stats=run["task_stats_by_task"],
                      task_stats_by_task=run["task_stats_by_task"],
                      policy_norm_stats=run["wm_norm_stats"],
                      world_norm_stats=run["wm_norm_stats"],
                      validation_episode_ids=run.get("validation_episode_ids", []))
    elif stage == "stage2":
        if task not in POLICY_TASKS:
            raise ValueError(f"Unknown aligned Stage-2 task: {task}")
        result.update(task=task, output=str(Path(run["output"]) / "tasks" / task),
                      task_manifest=run["policy_manifests"][task],
                      task_stats=run["task_stats_by_task"][task],
                      stage1_checkpoint=str(Path(checkpoint).resolve()) if checkpoint else None,
                      stage1_checkpoint_identity=run.get("stage1_checkpoint_identity"),
                      policy_norm_stats=_read_norm(run["task_stats_by_task"][task]),
                      world_norm_stats=run["wm_norm_stats"])
    else:
        raise ValueError(f"Unknown stage: {stage}")
    result["model_config"] = _model_values({**result, **{
        "policy_norm_stats": result["policy_norm_stats"],
        "world_norm_stats": result["world_norm_stats"]}}, run["mode"])
    result["model_config"].update(policy_norm_stats=result["policy_norm_stats"],
                                  world_norm_stats=result["world_norm_stats"])
    return result


def _atomic_yaml(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        yaml.safe_dump(value, handle, sort_keys=False)
        temporary = Path(handle.name)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _pipeline_env(threads):
    env = dict(os.environ)
    source_paths = [str(PROJECT / "src"), str(PROJECT / "vendor/wave0/src")]
    if env.get("PYTHONPATH"):
        source_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(source_paths)
    env.setdefault("OMP_NUM_THREADS", str(threads))
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def _training_command(run_file, stage, *, stage1_checkpoint=None, resume=None, threads=4):
    # Task and teacher paths are already part of the stage-specific run file.
    nproc = 8 if stage == "stage1" else 2
    command = [sys.executable, "-m", "torch.distributed.run", "--rdzv_backend", "c10d",
               "--rdzv_endpoint", "localhost:0", "--rdzv_id", str(uuid.uuid4()),
               "--nnodes", "1", "--nproc_per_node", str(nproc),
               "-m", "contact_belief_wam.aligned_trainer", "--run-config", str(run_file),
               "--stage", stage, "--device", "cuda", "--threads", str(threads)]
    if resume:
        command += ["--resume", str(resume)]
    return command


def _job_env(gpus, env):
    value = dict(env)
    value["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    return value


def _spawn(record, env):
    log = Path(record["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{_utc_now()}] {shlex.join(record['command'])}\n")
        handle.flush()
        return subprocess.Popen(record["command"], cwd=PROJECT, env=_job_env(record["gpus"], env),
                                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)


def _stop_processes(processes, grace_seconds=10):
    groups = [process.pid for process in processes.values()]
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and any(process.poll() is None for process in processes.values()):
        time.sleep(0.1)
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in processes.values():
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


@contextmanager
def workflow_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another aligned workflow holds {path}") from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _run_jobs(records, env, status, persist, *, concurrent=1):
    pending = list(records)
    active = {}
    failed = None
    try:
        while pending or active:
            while pending and len(active) < concurrent:
                record = pending.pop(0)
                record.update(state="running", started_at=_utc_now())
                persist()
                try:
                    process = _spawn(record, env)
                except Exception as exc:
                    record.update(state="failed", error=f"Could not start child: {exc}", finished_at=_utc_now())
                    persist()
                    failed = record["error"]
                    break
                record["pid"] = process.pid
                active[record["task"]] = process
                persist()
            if failed:
                break
            for task, process in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                record = next(row for row in records if row["task"] == task)
                record.update(exit_code=int(code), finished_at=_utc_now(),
                              state="done" if code == 0 else "failed")
                if code:
                    record["error"] = f"Child exited with status {code}"
                    failed = record["error"]
                persist()
                del active[task]
                if failed:
                    break
            if failed:
                break
            if active:
                time.sleep(0.1)
    except BaseException as exc:
        _stop_processes(active)
        for record in records:
            if record["state"] == "running":
                record.update(state="cancelled", error=str(exc), finished_at=_utc_now())
        persist()
        raise
    if failed:
        _stop_processes(active)
        for record in records:
            if record["state"] == "running":
                record.update(state="cancelled", error=failed, finished_at=_utc_now())
        persist()
        raise RuntimeError(failed)


def _latest_resume(output, stage):
    from .aligned_trainer import latest_checkpoint
    return latest_checkpoint(output, stage)


def _verify_stage1(path, expected_identity=None):
    path = Path(path)
    meta = json.loads((path / "complete.json").read_text())
    if meta.get("stage") != "stage1" or not (path / "pretrained_model/model.safetensors").is_file():
        raise RuntimeError(f"Stage-1 checkpoint is invalid: {path}")
    from .paths import sha256
    if sha256(path / "pretrained_model/model.safetensors") != meta.get("model_sha256"):
        raise RuntimeError("Selected Stage-1 checkpoint failed its model hash check")
    saved_identity = json.loads((path / "run_identity.json").read_text())
    if expected_identity is not None and saved_identity != expected_identity:
        raise RuntimeError("Selected Stage-1 checkpoint recipe/data identity differs from this run")
    result = json.loads((path.parents[2] / "result.json").read_text())
    if result.get("complete") is not True or Path(result.get("selected_checkpoint", "")).resolve() != path.resolve():
        raise RuntimeError("Stage 1 did not finish with this selected checkpoint")
    return path


def _prepare(run):
    from .aligned_data import prepare_aligned_data
    report = prepare_aligned_data(run, data_root=run["data_root"],
                                  original_root=run["original_root"],
                                  recovery_root=run["recovery_root"])
    run.update(shared_manifest=report["shared_manifest"],
               policy_manifests=report["policy_manifests"],
               task_stats_by_task=report["task_stats"], wm_stats=report["wm_stats"],
               data_root=report["data_root"], tactile_cache=report["cache_root"],
               tokenizer_path=report["tokenizer_path"],
               episode_fingerprint=report["episode_fingerprint"],
               policy_fingerprints=report["policy_fingerprints"])
    run["validation_episode_ids"] = report.get("validation_episode_ids", [])
    run["wm_norm_stats"] = _read_norm(run["wm_stats"])
    run["policy_norm_stats"] = run["wm_norm_stats"]
    run["model_config"]["tokenizer_path"] = run["tokenizer_path"]
    return report


def _require_prepared(run):
    required = [run["shared_manifest"], run["wm_stats"]]
    required += list(run.get("task_stats_by_task", {}).values())
    required += list(run.get("policy_manifests", {}).values())
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Aligned data is not prepared; run `aligned_train.sh prepare` first. Missing: "
                                + ", ".join(missing[:6]))
    from .weights import checkpoint_file
    checkpoint_file(run["pi05_base"])
    checkpoint_file(run["world_init"])


def _find_resume(run, stage, resume):
    if resume == "none":
        return None
    if resume != "auto":
        path = Path(resume).expanduser().resolve()
        if not (path / "complete.json").is_file():
            raise ValueError(f"Incomplete aligned checkpoint: {path}")
        return path
    return _latest_resume(run["output"], stage)


def main(config="configs/aligned_8xa800.yaml", *, command="train", variant="full",
         output=None, resume="none", plan_only=False, device="cuda", threads=4,
         updates=2, optimizer_device="cpu", micro_batch=2):
    configure()
    run = resolve_config(config, variant=variant, output=output)
    if command not in {"prepare", "debug", "train"}:
        raise ValueError(f"Unknown aligned pipeline command: {command}")
    if threads < 1:
        raise ValueError("threads must be positive")
    if plan_only:
        return plan(run)
    if command == "prepare":
        return _prepare(run)
    if command == "debug":
        if not output:
            raise ValueError("Debug requires an explicit isolated --output directory")
        if updates not in {1, 2} or micro_batch not in {1, 2}:
            raise ValueError("Debug updates and micro-batch must each be 1 or 2")
        from .aligned_debug import run_debug_pipeline
        prepared = _prepare(run)
        return run_debug_pipeline(run, updates=updates, device=device, threads=threads,
                                  optimizer_device=optimizer_device, micro_batch=micro_batch,
                                  data_report=prepared)
    schedule = plan(run)
    if resume not in {"auto", "none"}:
        raise ValueError("Formal pipeline resume accepts only auto or none")
    _prepare(run)
    _require_prepared(run)
    if not Path(run["world_init"]).is_absolute():
        raise ValueError("Stage 1 requires a fresh official world-model initialization checkpoint")
    workflow = Path(run["output"])
    status_path = workflow / STATUS_FILENAME
    workflow.mkdir(parents=True, exist_ok=True)
    lock = workflow_lock(workflow / ".aligned_pipeline.lock")
    lock.__enter__()
    if status_path.exists() and resume == "none":
        lock.__exit__(None, None, None)
        raise FileExistsError(f"Aligned workflow exists at {workflow}; use --resume auto or another output")
    status = {"schema_version": 1, "variant": variant, "mode": run["mode"],
              "output": str(workflow), "state": "running", "resume": resume,
              "started_at": _utc_now(), "stage1": {}, "stage2": {}}

    def persist():
        status["updated_at"] = _utc_now()
        _atomic_json(status_path, status)

    persist()
    env = _pipeline_env(threads)
    try:
        stage1_run = stage_run(run, "stage1")
        from .aligned_trainer import _source_identity, training_contract
        expected_stage1_identity = _source_identity(
            stage1_run, "stage1", training_contract(stage1_run, "stage1", 8))
        stage1_dir = Path(stage1_run["output"])
        stage1_dir.mkdir(parents=True, exist_ok=True)
        stage1_config = stage1_dir / "run_config.yaml"
        _atomic_yaml(stage1_config, stage1_run)
        existing_result = stage1_dir / "result.json"
        if resume == "auto" and existing_result.is_file():
            value = json.loads(existing_result.read_text())
            selected = value.get("selected_checkpoint")
            checkpoint = _verify_stage1(selected, expected_stage1_identity) \
                if value.get("complete") and selected else None
        else:
            checkpoint = None
        if checkpoint is None:
            stage1_resume = _find_resume(stage1_run, "stage1", resume)
            record = {"task": "stage1", "stage": "stage1", "gpus": schedule["stage1"]["gpus"],
                      "log": str(workflow / "logs/stage1.log"), "state": "pending"}
            record["command"] = _training_command(stage1_config, "stage1", resume=stage1_resume,
                                                   threads=threads)
            status["stage1"] = record
            persist()
            _run_jobs([record], env, status["stage1"], persist)
            result_path = stage1_dir / "result.json"
            value = json.loads(result_path.read_text())
            checkpoint = _verify_stage1(value["selected_checkpoint"], expected_stage1_identity)
        status["selected_stage1_checkpoint"] = str(checkpoint)
        status["stage1_result"] = str(stage1_dir / "result.json")
        persist()

        records = []
        for task, device_spec in zip(POLICY_TASKS, schedule["stage2"]):
            task_run = stage_run(run, "stage2", task, checkpoint)
            task_run["stage1_checkpoint_identity"] = json.loads((checkpoint / "complete.json").read_text()).get("identity")
            task_run["model_config"].update(policy_norm_stats=task_run["policy_norm_stats"],
                                            world_norm_stats=task_run["world_norm_stats"])
            task_dir = Path(task_run["output"])
            task_dir.mkdir(parents=True, exist_ok=True)
            config_path = task_dir / "run_config.yaml"
            _atomic_yaml(config_path, task_run)
            task_resume = _find_resume(task_run, "stage2", resume)
            record = {"task": task, "stage": "stage2", "gpus": device_spec["gpus"],
                      "log": str(workflow / "logs" / f"{task}.log"),
                      "result": str(task_dir / "result.json"), "state": "pending"}
            record["command"] = _training_command(config_path, "stage2", stage1_checkpoint=checkpoint,
                                                   resume=task_resume, threads=threads)
            records.append(record)
        status["stage2"] = {record["task"]: record for record in records}
        persist()
        _run_jobs(records, env, status["stage2"], persist, concurrent=2)
        for record in records:
            result_path = Path(record["result"])
            result_value = json.loads(result_path.read_text())
            if result_value.get("complete") is not True or not result_value.get("identity"):
                raise RuntimeError(f"Stage 2 result is incomplete: {result_path}")
        status.update(state="done", finished_at=_utc_now())
        persist()
        return status
    except BaseException as exc:
        status.update(state="cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=str(exc), finished_at=_utc_now())
        persist()
        raise
    finally:
        lock.__exit__(None, None, None)


def cli_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "debug", "train"))
    parser.add_argument("--config", default=str(PROJECT / "configs/aligned_8xa800.yaml"))
    parser.add_argument("--variant", choices=("full", "wave0_recovery"), default="full")
    parser.add_argument("--output")
    parser.add_argument("--resume", choices=("auto", "none"), default="none")
    parser.add_argument("--plan", action="store_true", help="Print physical GPU/task placement without CUDA checks")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--updates", choices=(1, 2), type=int, default=2,
                        help="Debug only: optimizer updates per stage")
    parser.add_argument("--micro-batch", choices=(1, 2), type=int, default=2,
                        help="Debug only: per-rank micro batch")
    parser.add_argument("--optimizer-device", choices=("cpu", "cuda"), default="cpu",
                        help="Debug only: optimizer state placement")
    args = parser.parse_args(argv)
    try:
        result = main(args.config, command=args.command, variant=args.variant, output=args.output,
                      resume=args.resume, plan_only=args.plan, device=args.device, threads=args.threads,
                      updates=args.updates, optimizer_device=args.optimizer_device,
                      micro_batch=args.micro_batch)
    except Exception as exc:
        print(f"aligned pipeline: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
