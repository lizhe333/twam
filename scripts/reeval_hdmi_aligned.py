#!/usr/bin/env python3
"""Queue one HDMI re-evaluation with JPEG tactile input and measured-q history.

Reuses the existing checkpoint validation, GPU lock, native evaluator, retries
and result publishing. All training sources, checkpoints and old results remain
unchanged. A completed run writes comparison.json beside its manifest.json.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import watch_eval_10k as watch


PROFILE = "jpeg-measured-history-v1"
SUITE = "tactile-head-only-expert-mb16-003-hdmi-aligned-inputs-v1"


def inventory(root):
    return {str(p.relative_to(root)): watch.digest(p)
            for p in sorted(root.rglob("*")) if p.is_file()
            and not p.is_symlink() and "__pycache__" not in p.parts
            and p.name != "runtime_manifest.json"}


def verify_runtime(runtime):
    record = json.loads((runtime / "runtime_manifest.json").read_text())
    if record["profile"] != PROFILE:
        raise ValueError("Another inference alignment profile is pinned")
    for name, expected in record["files"].items():
        if watch.digest(runtime / name) != expected:
            raise ValueError(f"Pinned inference source changed: {name}")
    return record


def pin_runtime(root, suite, batch, *, use_ema=True):
    runtime = suite / "runtime"
    if runtime.exists():
        record = verify_runtime(runtime)
        if record.get("use_ema", True) != use_ema:
            raise ValueError("Pinned runtime uses another EMA selection")
        return runtime, record
    project = root / "twam"
    temporary = runtime.with_name("runtime.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
        for folder in ("src", "inference", "vendor/wave0"):
            shutil.copytree(project / folder, temporary / folder, ignore=ignore)
        (temporary / "scripts").mkdir()
        for name in ("pyproject.toml", "scripts/cloud_train.sh"):
            shutil.copy2(project / name, temporary / name)
        (temporary / "vendor/transformers").symlink_to(
            (project / "vendor/transformers").resolve(), target_is_directory=True)
        shutil.copy2(root / "UniVTAC/policy/TactileWave0Wam/deploy_policy.py",
                     temporary / "native_policy_original.py")
        if not use_ema:
            bridge = temporary / "native_policy_original.py"
            source = bridge.read_text()
            if source.count("use_ema=True") != 3 or source.count("'--backbone-precision', 'bfloat16']") != 1:
                raise ValueError("Native bridge changed; cannot select ordinary weights safely")
            source = source.replace("use_ema=True", "use_ema=False").replace(
                "'--backbone-precision', 'bfloat16']", "'--backbone-precision', 'bfloat16', '--no-ema']")
            bridge.write_text(source)
        shutil.copy2(root / "UniVTAC/scripts/eval_policy.py", temporary / "native_eval.py")
        files = inventory(temporary)
        record = dict(profile=PROFILE, created_at=watch.now(), files=files,
                      source_project=str(project),
                      fingerprint=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
        if not use_ema:
            record["use_ema"] = False
        batch.write(temporary / "runtime_manifest.json", record)
        temporary.rename(runtime)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return runtime, record


def pin_reference(args, job, batch):
    """Keep the original result and per-seed outcomes; never overwrite them."""
    destination = args.suite / "comparison_reference.json"
    manifest = batch.read(args.previous_suite / "manifest.json")
    matches = [j for j in manifest["jobs"] if j["task"] == "insert_HDMI"]
    if len(matches) != 1:
        raise ValueError("Expected one previous HDMI result")
    previous = matches[0]
    for key in ("model_sha256", "ema_sha256", "training_identity", "step", "camera"):
        if previous[key] != job[key]:
            raise ValueError(f"Re-evaluation must use the same checkpoint: {key}")
    directory = args.previous_suite / "results" / previous["id"]
    result = batch.read(directory / "result.json")
    metadata = batch.read(directory / "metadata.json")
    seeds = {str(i) for i in range(1000000, 1000000 + args.episodes)}
    outcomes = {k: v["result"] for k, v in metadata.items() if k in seeds}
    if (result["status"] != "completed" or result["completed_episodes"] != args.episodes
            or result["evaluation_protocol"] != batch.EVAL_PROTOCOL or set(outcomes) != seeds
            or sum(v == "success" for v in outcomes.values()) != result["successes"]):
        raise ValueError("Previous result does not match the fixed-seed protocol")
    record = dict(result_file=str(directory / "result.json"), result=result, outcomes=outcomes,
                  model_sha256=job["model_sha256"], ema_sha256=job["ema_sha256"],
                  training_identity=job["training_identity"])
    if destination.exists() and batch.read(destination) != record:
        raise ValueError("Pinned comparison reference has changed")
    batch.write(destination, record)


def launch(job, root, suite, gpu, episodes, port, model_python, *, runner):
    runtime = suite / "runtime"
    record = verify_runtime(runtime)
    job["attempts"] += 1
    work = suite / "work" / job["id"] / f'attempt{job["attempts"]:02d}'
    work.mkdir(parents=True)
    deploy = work / "deploy.yml"
    runner.batch.write(deploy, dict(policy_name="TactileWave0WamAligned", seed=0, instruction_type="seen",
        pi05_host="127.0.0.1", pi05_port=port, pi05_python=str(model_python),
        pi05_startup_timeout=1800, pi05_request_timeout=1800,
        pi05_physics_steps_per_action=2, pi05_expert_preflight=True,
        pi05_eval_protocol=runner.batch.EVAL_PROTOCOL))
    env = runner.batch.environment(root, work, gpu)
    env.update(TACTILE_WAVE0_CHECKPOINT=job["checkpoint"], CBWAM_ROOT=str(runtime),
               PYTHONDONTWRITEBYTECODE="1")
    env.pop("PI05_TRACE_DIR", None)
    command = ["bash", str(root / "pi05-univtac/scripts/eval-worker.sh"),
               str(runtime / "inference/run_native.py"), job["task"], "clean", str(deploy),
               "--headless", "--livestream", "0", "--device", "cuda:0",
               "--total_num", str(episodes), "--start_seed", "1000000", "--run_name", job["id"]]
    logfile = work / "eval.log"
    with logfile.open("w") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
    job.update(state="running", log=str(logfile), pid=child.pid,
               input_alignment=PROFILE, runtime_fingerprint=record["fingerprint"])
    print(f'{watch.now()} START aligned HDMI port={port}', flush=True)
    return dict(process=child, log=logfile, work=work, start=time.monotonic(), warm=False,
                progress_time=time.monotonic(), progress="", job=job)


def idle_gpu_guard(claim, idle_seconds):
    """Require sustained idleness as well as the existing no-compute-process gate."""
    since, uuid = None, None
    def guarded(args, runner):
        nonlocal since, uuid
        lock, info, reason = claim(args, runner)
        if lock is None:
            since = uuid = None
            return lock, info, reason
        if "4090" not in info["name"]:
            lock.close()
            return None, None, f'Refusing a non-4090 GPU: {info["name"]}'
        current = time.monotonic()
        if since is None or uuid != info["uuid"]:
            since, uuid = current, info["uuid"]
        if current - since < idle_seconds:
            lock.close()
            return None, None, f"Waiting for {idle_seconds:g} seconds of uninterrupted GPU idleness"
        return lock, info, None
    return guarded


def compare(suite, manifest, batch):
    reference = batch.read(suite / "comparison_reference.json")
    job = manifest["jobs"][0]
    if job["state"] != "completed":
        return
    result = job["result"]
    metadata = batch.read(Path(result["native_result_dir"]) / "metadata.json")
    outcomes = {key: metadata[key]["result"] for key in reference["outcomes"]}
    for key in outcomes:
        row = metadata[key]
        execution = row.get("policy_execution", {})
        if (row["result"] != "error" or execution) and execution.get("input_alignment") != PROFILE:
            raise ValueError(f"Seed {key} did not use aligned inference")
    gained = sorted(int(k) for k, v in outcomes.items()
                    if v == "success" and reference["outcomes"][k] != "success")
    lost = sorted(int(k) for k, v in outcomes.items()
                  if v != "success" and reference["outcomes"][k] == "success")
    before, after = reference["result"]["successes"], result["successes"]
    episodes = result["completed_episodes"]
    report = dict(status="completed", input_alignment=PROFILE, task="insert_HDMI",
        episodes=episodes, previous_successes=before, aligned_successes=after,
        previous_success_rate=before / episodes, aligned_success_rate=after / episodes,
        change_percentage_points=100 * (after - before) / episodes,
        gained_success_seeds=gained, lost_success_seeds=lost,
        previous_result=reference["result_file"],
        aligned_result=str(Path(result["native_result_dir"]) / "result.json"),
        ema_sha256=reference["ema_sha256"], runtime_fingerprint=job["runtime_fingerprint"],
        note="Same checkpoint and scene seeds; action-sampling noise is not paired with the previous run.")
    batch.write(suite / "comparison.json", report)
    print(f'{watch.now()} COMPARE HDMI {before}/{episodes} -> {after}/{episodes} '
          f'({report["change_percentage_points"]:+g} percentage points)', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=watch.ROOT)
    parser.add_argument("--run-root", type=Path, default=watch.PROJECT / "runs" / watch.RUN / "full")
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--previous-suite", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--idle-seconds", type=float, default=60)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if args.idle_seconds < 0 or args.poll_seconds <= 0 or not args.gpu or "," in args.gpu:
        parser.error("Expected one GPU and positive timing limits")
    args.root, args.run_root = args.root.resolve(), args.run_root.resolve()
    args.suite = (args.suite or args.root / "outputs/eval_suites" / SUITE).resolve()
    args.previous_suite = (args.previous_suite or args.root / "outputs/eval_suites" / watch.SUITE).resolve()
    if args.suite == args.previous_suite or args.suite.is_relative_to(args.run_root):
        parser.error("Use a separate output suite; preserve training and previous results")
    args.episodes, args.max_attempts = 50, 2
    args.timeout, args.stall_timeout = 43200, 1800
    args.scan_only = args.prepare_only
    args.model_python = args.root / "contact-belief-wam/.venv/bin/python"
    if not os.access(args.model_python, os.X_OK):
        parser.error(f"Missing model environment: {args.model_python}")
    os.environ["TWAM_ROOT"] = str(args.root)
    os.environ.setdefault("PI05_SIM_GRAPHICS_LIBRARY_PATH", str(args.root / ".cache/sim-graphics-runtime/lib"))
    os.environ["PATH"] = str(args.model_python.parent) + os.pathsep + os.environ.get("PATH", "")
    args.suite.mkdir(parents=True, exist_ok=True)
    with (args.suite / "queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Queue already running: {args.suite}", flush=True)
            return 0
        runner = watch.load_runner(args.root)
        watch.TASKS = ("insert_HDMI",)
        runtime, record = pin_runtime(args.root, args.suite, runner.batch)
        manifest = watch.prepare(args, runner)
        if manifest.get("input_alignment", PROFILE) != PROFILE:
            raise ValueError("Queue has a different input alignment")
        manifest.update(input_alignment=PROFILE, runtime=str(runtime), runtime_fingerprint=record["fingerprint"])
        watch.refresh(args, manifest, runner)
        if len(manifest["jobs"]) != 1:
            raise ValueError("The requested HDMI 10k checkpoint is not ready")
        pin_reference(args, manifest["jobs"][0], runner.batch)
        runner.batch.save_summary(args.suite, manifest)
        runner.launch = lambda *values: launch(*values, runner=runner)
        watch.claim_gpu = idle_gpu_guard(watch.claim_gpu, args.idle_seconds)
        if not args.scan_only:
            (args.suite / "daemon.pid").write_text(f"{os.getpid()}\n")
        print(f'{watch.now()} HDMI aligned re-evaluation; GPU={args.gpu}; seeds=1000000..1000049', flush=True)
        code = watch.run_monitor(args, manifest, runner)
        if not args.scan_only:
            compare(args.suite, manifest, runner.batch)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
