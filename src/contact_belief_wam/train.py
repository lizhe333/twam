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
DEFAULT_STAGE1_GPUS = 4
STATUS_FILENAME = "train_status.json"


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
        values.update(independent_tactile=False, belief_warmup_min=0, belief_warmup_max=0, belief_bptt_steps=1)
    return values


def resolve_config(config="configs/aligned_4xa800.yaml", *, variant="full", output=None):
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
    finetune = bool(raw.get("finetune_from"))
    if type(raw.get("finetune_stage2_only", False)) is not bool:
        raise ValueError("finetune_stage2_only must be a boolean")
    if raw.get("finetune_stage2_only") and not finetune:
        raise ValueError("Stage-2-only fine-tuning requires finetune_from")
    if raw.get("finetune_prepared_from") and not finetune:
        raise ValueError("Reusing frozen fine-tuning data requires finetune_from")
    if finetune:
        if raw.get("tasks") != ["insert_HDMI"] or variant != "full":
            raise ValueError("This fine-tuning recipe is full TWAM on insert_HDMI only")
        if raw.get("expert_only") or raw.get("stage2_restart_from"):
            raise ValueError("Recovery fine-tuning uses mixed data and a new optimizer, not an expert-only restart")
        if raw.get("camera_mode") != "head_only":
            raise ValueError("HDMI fine-tuning must retain head_only inputs")
        if type(raw.get("finetune_use_ema")) is not bool:
            raise ValueError("finetune_use_ema must explicitly select raw or EMA source weights")
        if raw.get("stage1_freeze_world") is not False:
            raise ValueError("HDMI Stage 1 must update the visual WM")
    elif raw.get("tasks") != list(TASKS):
        raise ValueError("Aligned Stage 1 must use all eight canonical tasks in order")
    policy_tasks = raw.get("policy_tasks", raw["tasks"] if finetune else list(POLICY_TASKS))
    if not policy_tasks or len(set(policy_tasks)) != len(policy_tasks) or any(task not in TASKS for task in policy_tasks):
        raise ValueError("policy_tasks must be a nonempty, unique subset of the eight tasks")
    raw["config"] = str(config_path)
    raw["variant"] = variant
    raw["mode"] = "full" if variant == "full" else "wave0"
    raw["aligned"] = True
    for key in ("data_root", "original_root", "recovery_root", "tactile_cache", "pi05_base",
                "tokenizer_path", "world_init", "aligned_manifest_dir", "shared_manifest", "policy_manifest_dir",
                "task_stats_dir", "wm_stats", "reference_wm_stats", "reference_root", "stage2_restart_from"):
        if raw.get(key):
            raw[key] = _resolve_path(raw[key])
    if finetune:
        for key in ("finetune_from", "finetune_data_dir", "original_manifest", "recovery_manifest", "recovery_cache"):
            raw[key] = _resolve_path(raw[key])
        if raw.get("finetune_prepared_from"):
            raw["finetune_prepared_from"] = _resolve_path(raw["finetune_prepared_from"])
            if raw["finetune_prepared_from"] == raw["finetune_data_dir"]:
                raise ValueError("Continuation data metadata needs a separate finetune_data_dir")
        if Path(raw["finetune_from"]).is_relative_to(Path(output or raw["output"]).expanduser().resolve()):
            raise ValueError("Fine-tuning output must not contain its source checkpoint")
        if policy_tasks != raw["tasks"]:
            raise ValueError("Fine-tuning Stage 2 must use the same HDMI task")
    raw["workflow_root"] = str(Path(output or raw["output"]).expanduser().resolve() / variant)
    raw["output"] = raw["workflow_root"]
    raw["workspace"] = str(Path(raw.get("workspace", PROJECT.parent)).expanduser().resolve())
    raw["tasks"] = list(raw["tasks"]) if finetune else list(TASKS)
    raw["policy_tasks"] = list(policy_tasks)
    raw.setdefault("workers", 4)
    raw.setdefault("seed", 42)
    raw.setdefault("precision", "bf16")
    raw.setdefault("ema_decay", 0.99)
    raw.setdefault("grad_clip", 1.0)
    raw.setdefault("fused_adamw", True)
    raw.setdefault("gradient_checkpointing", True)
    raw.setdefault("expert_only", False)
    if type(raw["expert_only"]) is not bool:
        raise ValueError("expert_only must be a boolean")
    raw.setdefault("data_mix", "1:0" if raw["expert_only"] else "1:1")
    if raw["expert_only"] and raw["data_mix"] != "1:0":
        raise ValueError("expert_only requires data_mix: '1:0' (Original only)")
    raw.setdefault("log_every", 10)
    raw.setdefault("save_every", 1000)
    raw.setdefault("keep_checkpoints", 2)
    if raw["precision"] != "bf16":
        raise ValueError("Aligned formal training uses BF16 autocast with FP32 parameters")
    for name, allowed in (("stage1", (4, 8)), ("stage2", (2, 4))):
        stage = raw.get(name, {})
        if stage.get("num_gpus") not in allowed:
            raise ValueError(f"{name} must use {' or '.join(map(str, allowed))} GPUs")
        for key in ("num_gpus", "micro_batch", "global_batch"):
            if type(stage.get(key)) is not int or stage[key] < 1:
                raise ValueError(f"{name}: {key} must be a positive integer")
        divisor = stage["num_gpus"] * stage["micro_batch"]
        if stage["global_batch"] % divisor:
            raise ValueError(f"{name}: global_batch must be divisible by num_gpus × micro_batch")
        stage["accumulation_steps"] = stage["global_batch"] // divisor
    validate_every = raw["stage1"].get("validate_every", 500)
    if type(validate_every) is not int or validate_every < 0:
        raise ValueError("stage1.validate_every must be a nonnegative integer; 0 disables validation")
    raw["stage1"]["validate_every"] = validate_every
    if validate_every == 0:
        raw.setdefault("wm_validation_episode_ids", [])
        if raw["wm_validation_episode_ids"]:
            raise ValueError("Disabled Stage-1 validation requires wm_validation_episode_ids: []")
    if type(raw["stage2"].get("shard_optimizer_state", False)) is not bool:
        raise ValueError("stage2.shard_optimizer_state must be a boolean")
    if raw["stage2"].get("ema_device", "cuda") not in {"cuda", "cpu"}:
        raise ValueError("stage2.ema_device must be cuda or cpu")
    for value, name in ((raw.get("batch_stage2_policy", False), "batch_stage2_policy"),
                        (raw.get("ema_enabled", True), "ema_enabled"),
                        (raw.get("stage2_restart_rebatch", False), "stage2_restart_rebatch"),
                        (raw["stage2"].get("shard_ema", False), "stage2.shard_ema")):
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if raw["stage2"].get("shard_ema") and raw["stage2"].get("ema_device", "cuda") != "cuda":
        raise ValueError("Sharded EMA stays on GPU; use ema_device: cuda")
    if not raw.get("ema_enabled", True) and raw["stage2"].get("shard_ema"):
        raise ValueError("Disabled EMA requires stage2.shard_ema: false")
    if raw.get("batch_stage2_policy") and (raw["mode"] != "full" or not raw.get("independent_tactile")):
        raise ValueError("Batched Stage-2 policy requires full independent tactile dynamics")
    if raw.get("stage2_restart_from") == raw["workflow_root"]:
        raise ValueError("Performance restart requires a new output directory")
    if raw.get("stage2_restart_rebatch") and (not raw.get("stage2_restart_from") or not raw["expert_only"]):
        raise ValueError("Rebatch restart requires stage2_restart_from and expert_only")
    if raw["mode"] == "wave0":
        raw["stage1"]["selection_metric"] = "world_total"
    raw["model_config"] = _model_values(raw, raw["mode"])
    raw["shared_manifest"] = raw.get("shared_manifest")
    raw["policy_manifests"] = {}
    raw["task_stats_by_task"] = {}
    raw["wm_norm_stats"] = None
    raw["policy_norm_stats"] = None
    raw["stage1_checkpoint"] = None
    raw["stage1_checkpoint_identity"] = None
    return raw


def _physical_gpus(count=DEFAULT_STAGE1_GPUS):
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = [str(index) for index in range(count)] if raw is None else [item.strip() for item in raw.split(",")]
    if not devices or any(not item for item in devices):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain non-empty GPU IDs")
    if len(set(devices)) != len(devices):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain unique GPU IDs")
    return devices


def plan(run, *, stop_after=None):
    if stop_after not in {None, "stage1"}:
        raise ValueError("The two-stage workflow supports only --stop-after stage1")
    reuse_world = run["mode"] == "wave0" and run.get("stage1_freeze_world", False)
    reuse_stage1 = run.get("stage1_init_checkpoint")
    stage2_only = run.get("finetune_stage2_only", False)
    if stage2_only and (stop_after is not None or reuse_stage1):
        raise ValueError("Stage-2-only continuation cannot run or overlay Stage 1")
    if reuse_stage1 and (run["mode"] != "full" or stop_after is not None):
        raise ValueError("A Stage-1 initialization requires full mode and training Stage 2")
    tasks = run.get("policy_tasks", POLICY_TASKS)
    stage1_gpus = 0 if reuse_world or reuse_stage1 or stage2_only else int(run.get("stage1", {}).get("num_gpus", DEFAULT_STAGE1_GPUS))
    policy_gpus = int(run.get("stage2", {}).get("num_gpus", 2))
    stage2_gpus = 0 if stop_after == "stage1" else min(2, len(tasks)) * policy_gpus
    required = max(stage1_gpus, stage2_gpus)
    devices = _physical_gpus(required) if required else []
    if len(devices) < required:
        raise ValueError(f"This workflow needs {required} visible GPUs; received {len(devices)}")
    devices = devices[:required]
    stage1 = {"stage": "stage1", "task": "shared_wm",
              "output": str(Path(run["output"]) / "shared_wm"), "gpus": devices[:stage1_gpus],
              "nproc_per_node": stage1_gpus, "concurrency": 1}
    if stage2_only:
        stage1.update(gpus=[], nproc_per_node=0, reuse_full_checkpoint=run["finetune_from"])
    elif reuse_world:
        stage1.update(gpus=[], nproc_per_node=0, reuse_world_init=run["world_init"])
    elif reuse_stage1:
        stage1.update(gpus=[], nproc_per_node=0, reuse_checkpoint=reuse_stage1)
    stage2 = [{"stage": "stage2", "task": task,
               "output": str(Path(run["output"]) / "tasks" / task),
               "gpus": devices[(index % 2) * policy_gpus:(index % 2 + 1) * policy_gpus],
               "nproc_per_node": policy_gpus,
               "concurrency": min(2, len(tasks))} for index, task in enumerate(tasks)]
    result = {"config": run["config"], "variant": run["variant"], "mode": run["mode"],
            "workspace": run["workspace"], "output": run["output"],
            "physical_gpus": devices, "stage1": stage1, "stage2": stage2}
    if stop_after == "stage1":
        result.pop("stage2")
    return result


def stage_run(run, stage, task=None, checkpoint=None):
    result = dict(run)
    if stage in {"tactile", "stage1"}:
        name = "tactile_wm" if stage == "tactile" else "shared_wm"
        result.update(task=name, output=str(Path(run["output"]) / name),
                      task_manifest=None, task_stats=run["task_stats_by_task"],
                      task_stats_by_task=run["task_stats_by_task"],
                      policy_norm_stats=run["wm_norm_stats"],
                      world_norm_stats=run["wm_norm_stats"],
                      validation_episode_ids=run.get("validation_episode_ids", []))
    elif stage == "stage2":
        if task not in run.get("policy_tasks", POLICY_TASKS):
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
    if stage == "tactile":
        result["model_config"].update(stage="tactile", belief_warmup_min=0,
                                      belief_warmup_max=0, belief_bptt_steps=1)
    if run.get("finetune_from"):
        # This also pins the different source mixtures in the resume contract.
        result["data_mix"] = run[stage].get("data_mix", run["data_mix"])
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


def _training_command(run_file, stage, *, stage1_checkpoint=None, resume=None, threads=4, nproc=None):
    # Task and teacher paths are already part of the stage-specific run file.
    nproc = nproc or (8 if stage == "stage1" else 2)
    command = [sys.executable, "-m", "torch.distributed.run", "--rdzv_backend", "c10d",
               "--rdzv_endpoint", "localhost:0", "--rdzv_id", str(uuid.uuid4()),
               "--nnodes", "1", "--nproc_per_node", str(nproc),
               "-m", "contact_belief_wam.trainer", "--run-config", str(run_file),
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
                occupied = {gpu for row in records if row["task"] in active for gpu in row["gpus"]}
                available = next((index for index, row in enumerate(pending)
                                  if not occupied.intersection(row["gpus"])), None)
                if available is None:
                    break
                record = pending.pop(available)
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
    from .trainer import latest_checkpoint
    return latest_checkpoint(output, stage)


def _verify_prediction_checkpoint(path, expected_identity=None, *, stage="stage1"):
    path = Path(path)
    meta = json.loads((path / "complete.json").read_text())
    if meta.get("stage") != stage or not (path / "pretrained_model/model.safetensors").is_file():
        raise RuntimeError(f"{stage} checkpoint is invalid: {path}")
    from .paths import sha256
    if sha256(path / "pretrained_model/model.safetensors") != meta.get("model_sha256"):
        raise RuntimeError(f"Selected {stage} checkpoint failed its model hash check")
    saved_identity = json.loads((path / "run_identity.json").read_text())
    if expected_identity is not None and saved_identity != expected_identity:
        raise RuntimeError(f"Selected {stage} checkpoint recipe/data identity differs from this run")
    result = json.loads((path.parents[2] / "result.json").read_text())
    if result.get("complete") is not True or Path(result.get("selected_checkpoint", "")).resolve() != path.resolve():
        raise RuntimeError(f"{stage} did not finish with this selected checkpoint")
    return path


def _verify_stage1_initialization(path, run):
    """Transfer a completed teacher while allowing a new Stage-2 batch/GPU recipe."""
    path = _verify_prediction_checkpoint(Path(path).resolve())
    from .config import ContactConfig
    from .paths import sha256
    from .weights import checkpoint_file
    saved = ContactConfig.load(path / "pretrained_model/config.json")
    expected = ContactConfig(**{**run["model_config"], "world_norm_stats": run["wm_norm_stats"]})
    if saved.stage != "stage1" or saved.mode != "full":
        raise ValueError("Stage-2 initialization requires a full Contact Stage-1 checkpoint")
    for key in ("aligned", "mode", "independent_tactile", "camera_mode", "k", "chunk_size",
                "action_dim", "padded_dim", "width", "heads", "ffn", "belief_tokens", "tactile_tokens",
                "world_width", "world_depth", "world_heads", "world_ffn", "vision_width", "vision_pool",
                "expert_width", "tokenizer_max_length", "encoder_reference", "world_norm_stats",
                "belief_to_latent_queries", "belief_to_world_state", "belief_to_future_corrector",
                "belief_to_contact_predictor", "belief_to_expert"):
        if getattr(saved, key) != getattr(expected, key):
            raise ValueError(f"Stage-1 initialization differs in {key}")
    identity = json.loads((path / "run_identity.json").read_text())
    meta = json.loads((path / "complete.json").read_text())
    if identity.get("stage") != "stage1" or not identity.get("fingerprint") \
            or identity["fingerprint"] != meta.get("identity"):
        raise ValueError("Stage-1 initialization has inconsistent provenance")
    if run.get("expert_only") and identity.get("contract", {}).get("expert_only") is not True:
        raise ValueError("expert_only requires an expert-only Stage-1 checkpoint; "
                         "retrain Stage 1 without --stage1-checkpoint to exclude Recovery training")
    if sha256(checkpoint_file(run["pi05_base"])) != identity.get("pi05_base_sha256"):
        raise ValueError("Stage-1 initialization uses a different pi05_base / frozen visual encoder")
    return path


def _prepare(run):
    if run.get("finetune_from"):
        from .finetune_data import prepare_finetune_data
        report = prepare_finetune_data(run)
    else:
        from .data import prepare_data
        report = prepare_data(run, data_root=run["data_root"],
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
    if run.get("finetune_from"):
        from .finetune_data import verify_source
        verify_source(run)
    if run.get("stage1_freeze_world"):
        from .config import ContactConfig
        from .weights import validate_world_contract
        values = {**run["model_config"], "world_norm_stats": run["wm_norm_stats"]}
        validate_world_contract(ContactConfig(**values), run["world_init"])


def _find_resume(run, stage, resume):
    if resume == "none":
        return None
    if resume != "auto":
        path = Path(resume).expanduser().resolve()
        if not (path / "complete.json").is_file():
            raise ValueError(f"Incomplete aligned checkpoint: {path}")
        return path
    checkpoint = _latest_resume(run["output"], stage)
    if checkpoint is None and stage == "stage2" and run.get("stage2_restart_from"):
        from .performance_restart import source_directory
        source = source_directory(run)
        if not Path(run["stage2_restart_from"]).is_dir():
            raise FileNotFoundError(f"Performance restart source does not exist: {run['stage2_restart_from']}")
        checkpoint = _latest_resume(source, stage)
        if checkpoint is None and ((source / "train.jsonl").is_file() or (source / "result.json").is_file()):
            raise ValueError(f"Source task has training progress but no complete checkpoint: {source}. "
                             "Wait for checkpoint saving to finish before stopping the source job.")
    return checkpoint


def _run_prediction_stage(run, stage, placement, workflow, env, status, persist, resume, threads):
    """Launch, resume and validate a shared prediction stage."""
    from .trainer import _source_identity, training_contract
    stage_config = stage_run(run, stage)
    identity = _source_identity(stage_config, stage, training_contract(
        stage_config, stage, int(run[stage]["num_gpus"])))
    directory = Path(stage_config["output"])
    directory.mkdir(parents=True, exist_ok=True)
    config_path, result_path = directory / "run_config.yaml", directory / "result.json"
    _atomic_yaml(config_path, stage_config)
    checkpoint = None
    if resume == "auto" and result_path.is_file():
        value = json.loads(result_path.read_text())
        if value.get("complete") and value.get("selected_checkpoint"):
            checkpoint = _verify_prediction_checkpoint(value["selected_checkpoint"], identity, stage=stage)
    if checkpoint is None:
        record = {"task": stage_config["task"], "stage": stage, "gpus": placement["gpus"],
                  "log": str(workflow / "logs" / f"{stage}.log"), "state": "pending"}
        record["command"] = _training_command(config_path, stage, threads=threads,
            resume=_find_resume(stage_config, stage, resume), nproc=int(run[stage]["num_gpus"]))
        status[stage] = record
        persist()
        _run_jobs([record], env, status[stage], persist)
        value = json.loads(result_path.read_text())
        checkpoint = _verify_prediction_checkpoint(value["selected_checkpoint"], identity, stage=stage)
    status[f"selected_{stage}_checkpoint"] = str(checkpoint)
    status[f"{stage}_result"] = str(result_path)
    persist()
    return checkpoint


def main(config="configs/aligned_4xa800.yaml", *, command="train", variant="full",
         output=None, resume="none", plan_only=False, device="cuda", threads=4,
         updates=2, optimizer_device="cpu", micro_batch=2, stop_after=None, stage1_checkpoint=None):
    configure()
    run = resolve_config(config, variant=variant, output=output)
    if stage1_checkpoint is not None:
        if run.get("finetune_from"):
            raise ValueError("HDMI two-stage fine-tuning adapts Stage 1; do not pass --stage1-checkpoint")
        if command != "train" or variant != "full" or stop_after is not None:
            raise ValueError("--stage1-checkpoint requires full training without --stop-after")
        run["stage1_init_checkpoint"] = _resolve_path(stage1_checkpoint)
    if stop_after not in {None, "stage1"}:
        raise ValueError("The two-stage workflow supports only --stop-after stage1")
    if command not in {"prepare", "debug", "train"}:
        raise ValueError(f"Unknown aligned pipeline command: {command}")
    if threads < 1:
        raise ValueError("threads must be positive")
    if plan_only:
        return plan(run, stop_after=stop_after)
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
    schedule = plan(run, stop_after=stop_after)
    if resume not in {"auto", "none"}:
        raise ValueError("Formal pipeline resume accepts only auto or none")
    _prepare(run)
    _require_prepared(run)
    teacher = (_verify_stage1_initialization(run["stage1_init_checkpoint"], run)
               if run.get("stage1_init_checkpoint") else None)
    if not Path(run["world_init"]).is_absolute():
        raise ValueError("world_init must identify the pretrained shared world model")
    workflow = Path(run["output"])
    status_path = workflow / STATUS_FILENAME
    workflow.mkdir(parents=True, exist_ok=True)
    lock = workflow_lock(workflow / ".train.lock")
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
        checkpoint = None
        if schedule["stage1"].get("reuse_full_checkpoint"):
            status["stage1"] = {"state": "reused_from_full_policy", "checkpoint": run["finetune_from"]}
            persist()
        elif teacher is not None:
            checkpoint = teacher
            status["stage1"] = {"state": "reused", "checkpoint": str(checkpoint)}
            status["selected_stage1_checkpoint"] = str(checkpoint)
            status["stage1_result"] = str(checkpoint.parents[2] / "result.json")
            persist()
        elif schedule["stage1"].get("reuse_world_init"):
            status["stage1"] = {"state": "reused", "world_init": run["world_init"]}
        else:
            checkpoint = _run_prediction_stage(run, "stage1", schedule["stage1"], workflow, env,
                                               status, persist, resume, threads)
        if stop_after == "stage1":
            status.update(state="stage1_complete", finished_at=_utc_now())
            persist()
            return status

        records = []
        for device_spec in schedule["stage2"]:
            task = device_spec["task"]
            task_run = stage_run(run, "stage2", task, checkpoint)
            task_run["stage1_checkpoint_identity"] = (json.loads((checkpoint / "complete.json").read_text()).get("identity")
                                                       if checkpoint else None)
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
                                                   resume=task_resume, threads=threads,
                                                   nproc=device_spec["nproc_per_node"])
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
    parser.add_argument("command", choices=("prepare", "debug", "train"), nargs="?", default="train")
    parser.add_argument("--stop-after", choices=("stage1",),
                        help="Finish and validate this stage, then stop; --resume auto continues later")
    parser.add_argument("--config", default=str(PROJECT / "configs/aligned_4xa800.yaml"))
    parser.add_argument("--variant", choices=("full", "wave0_recovery"), default="full")
    parser.add_argument("--output")
    parser.add_argument("--resume", choices=("auto", "none"), default="none")
    parser.add_argument("--stage1-checkpoint", help="Reuse a completed Contact Stage-1 checkpoint and train only Stage 2")
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
                      micro_batch=args.micro_batch, stop_after=args.stop_after,
                      stage1_checkpoint=args.stage1_checkpoint)
    except Exception as exc:
        print(f"aligned pipeline: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
