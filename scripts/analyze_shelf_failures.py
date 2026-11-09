#!/usr/bin/env python3
"""Numerical shelf diagnostics: audit, two fresh-process action replays, report.

No policy model is imported. Original evaluations are read-only. All recovered
object trajectories are replay inferences, even when joint histories match.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import html
import importlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

import numpy as np

ROOT = Path(os.environ.get("SHELF_DIAGNOSTIC_ROOT", Path(__file__).resolve().parents[2])).resolve()
TASK = "put_bottle_in_shelf"
SUITE = ROOT / "outputs/eval_suites/twam-single-view"
DEFAULT_INPUT = SUITE / "results/put_bottle_in_shelf"
DEFAULT_OUTPUT = ROOT / "outputs/diagnostics/put_bottle_in_shelf-step10000-failure-analysis"
LIMITS = np.array([.02, .10, .02])
COS_LIMIT = .965
ANGLE_LIMIT = math.acos(COS_LIMIT)
ARM_TOL = 1e-3
GRIP_TOL = 1e-4


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


def relative_metrics(delta, dot):
    delta = np.asarray(delta, dtype=np.float64)
    if delta.shape != (3,) or not np.isfinite(delta).all() or not np.isfinite(dot):
        raise ValueError("Non-finite or invalid object pose")
    tilt = math.acos(float(np.clip(dot, -1, 1)))
    passes = np.r_[np.abs(delta) < LIMITS, dot > COS_LIMIT]
    return dict(delta_m=delta.tolist(), tilt_deg=math.degrees(tilt), dot=float(dot),
                passes=passes.tolist(), success=bool(passes.all()),
                score=float(max(*np.abs(delta) / LIMITS, tilt / ANGLE_LIMIT)),
                excess_mm=(np.maximum(np.abs(delta) - LIMITS, 0) * 1000).tolist())


def matrix_metrics(bottle, target):
    """Testable coordinate conversion; replay itself uses the native Pose.rebase."""
    relative = np.linalg.inv(target) @ bottle
    return relative_metrics(relative[:3, 3], relative[2, 2])


def infer_termination(row):
    if row.get("failure_kind"):
        return row["failure_kind"]
    if row["result"] == "success":
        return "success"
    execution = row["policy_execution"]
    if execution["physical_action_calls"] >= execution["physics_call_limit"]:
        return "action_budget_exhausted"
    return "early_stop_or_unrecorded_termination"


def validate_episode(seed, row):
    errors = []
    actions = row.get("tactile_feedback", {}).get("actions", [])
    execution = row.get("policy_execution", {})
    calls = execution.get("physical_action_calls", -1)
    initial_step = row.get("cost_step", 0) - calls
    previous = initial_step
    advances = []
    for i, item in enumerate(actions):
        if item.get("step") != i:
            errors.append(f"action {i}: non-contiguous strategy index")
        for key in ("action", "measured_joint"):
            array = np.asarray(item.get(key, []), dtype=float)
            if array.shape != (8,) or not np.isfinite(array).all():
                errors.append(f"action {i}: invalid {key}")
        physical = item.get("physical_step", -1)
        advance = physical - previous
        allowed_single = i == len(actions) - 1 and row.get("result") == "success"
        if advance != 2 and not (advance == 1 and allowed_single):
            errors.append(f"action {i}: unexpected physical advance {advance}")
        advances.append(advance)
        previous = physical
    if not actions or sum(advances) != calls or previous != row.get("cost_step"):
        errors.append("saved actions do not account for physical calls/cost_step")
    if execution.get("physical_steps_per_action") != 2:
        errors.append("unsupported physical_steps_per_action")
    return dict(seed=int(seed), original_result=row.get("result"),
                termination=infer_termination(row), failure_reason=row.get("failure_reason"),
                actions=len(actions), physical_calls=calls, initial_step=initial_step,
                final_step=row.get("cost_step"), physics_call_limit=execution.get("physics_call_limit"),
                errors=errors)


def audit(input_dir, output):
    if input_dir == output or input_dir in output.parents:
        raise ValueError("Diagnostic output must be outside the original evaluation directory")
    metadata = read_json(input_dir / "metadata.json")
    result = read_json(input_dir / "result.json")
    suite = input_dir.parent.parent
    runtime = suite / "runtime"
    manifest = read_json(runtime / "runtime_manifest.json")
    errors = []
    if result.get("task") != TASK:
        errors.append("input is not the put_bottle_in_shelf task")
    mismatches = []
    for base, files in ((runtime, manifest["files"]), (ROOT, manifest["external_sources"])):
        for name, expected in files.items():
            path = base / name
            if not path.is_file() or digest(path) != expected:
                mismatches.append(str(path))
    fingerprint = hashlib.sha256(json.dumps(dict(files=manifest["files"], external=manifest["external_sources"]), sort_keys=True).encode()).hexdigest()
    if fingerprint != manifest["fingerprint"]:
        errors.append("runtime manifest fingerprint is invalid")
    if mismatches:
        errors.append("runtime/environment source hashes differ")
    episodes = [validate_episode(str(seed), metadata[str(seed)]) for seed in range(1000000, 1000050) if str(seed) in metadata]
    if len(episodes) != 50:
        errors.append("expected all 50 seeds 1000000..1000049")
    counts = Counter(row["original_result"] for row in episodes)
    if counts["success"] != result["successes"] or counts["failed"] != result["failures"]:
        errors.append("result.json counts disagree with metadata")
    log = (input_dir / "log.log").read_text()
    for row in episodes:
        seed = row["seed"]
        match = re.search(rf"Seed {seed} (success|failed) after[^\n]*\nsteps:\s*(\d+)\s*, actions:\s*(\d+)", log)
        if row["termination"] == "invalid_policy_output":
            if not re.search(rf"Seed {seed} Policy output rejected", log):
                row["errors"].append("policy rejection missing from log")
        elif not match or (match[1], int(match[2]), int(match[3])) != (row["original_result"], row["final_step"], row["physical_calls"]):
            row["errors"].append("log outcome/steps/actions disagree")
    original_files = {str(path.relative_to(input_dir)): digest(path) for path in input_dir.rglob("*") if path.is_file()}
    record = dict(formal_result=False, created_at=now(), input=str(input_dir),
                  runtime=str(runtime), runtime_fingerprint=fingerprint,
                  source_mismatches=mismatches, errors=errors, counts=dict(counts),
                  original_result=result, original_files_sha256=original_files,
                  simulator="Isaac Sim 4.5", expected_gpu="NVIDIA GeForce RTX 4090",
                  expected_physics_dt=1 / 120, episodes=episodes,
                  original_object_trajectory_available=False,
                  evidence_note="Object poses are absent from episode metadata; frame-5 dump is not an episode trajectory.")
    record["valid"] = not errors and not any(row["errors"] for row in episodes)
    write_json(output / "audit.json", record)
    return record


def execute_saved_actions(task, row, to_tensor, capture, native_check):
    """Match the native evaluator/Adapter's stepping and observation ordering."""
    actions = row["tactile_feedback"]["actions"]
    checks = []
    context = dict(action_index=-1, substep=-1)
    original_check = task.check_success

    def traced_check():
        result = bool(native_check())
        capture(context["action_index"], context["substep"], result, "check")
        return result

    task.check_success = traced_check
    try:
        capture(-1, -1, bool(native_check()), "initial")
        reason = "saved_actions_exhausted"
        for i, item in enumerate(actions):
            task._get_observations()  # Outer native evaluator observation.
            value = to_tensor(item["action"])
            before = task.step_count
            succeeded = False
            for substep in range(2):
                context.update(action_index=i, substep=substep)
                okay, succeeded = task.take_action(value, action_type="qpos", force=True)
                if not okay or not task.plan_success:
                    raise RuntimeError("Replay action rejected by simulator")
                if succeeded:
                    break
            advanced = task.step_count - before
            if advanced != 2 and not (advanced == 1 and succeeded):
                raise RuntimeError(f"Unexpected physical advance: {advanced}")
            obs = task._get_observations()  # Adapter.execute observation.
            joint = obs["embodiment"]["joint"]
            measured = joint.detach().cpu().numpy()[:8] if hasattr(joint, "detach") else np.asarray(joint)[:8]
            error = np.abs(measured - np.asarray(item["measured_joint"]))
            checks.append(dict(action_index=i, physical_step=int(task.step_count),
                               expected_physical_step=item["physical_step"],
                               arm_error_rad=float(error[:7].max()), grip_error_m=float(error[7]),
                               physical_step_matches=task.step_count == item["physical_step"]))
            if succeeded:
                reason = "success"
                break
            if task.check_early_stop():
                reason = "early_stop"
                break
        else:
            reason = infer_termination(row)
            if reason == "invalid_policy_output":
                # Native eval observes once before requesting the rejected next
                # action. There is no saved/reconstructible command to execute.
                task._get_observations()
        return checks, reason
    finally:
        task.check_success = original_check


def summarize_trace(trace):
    checks = np.flatnonzero(trace["kind"] == "check")
    if not len(checks):
        return None
    best = int(checks[np.argmin(trace["score"][checks])])
    final = int(checks[-1])

    def at(i):
        passes = trace["passes"][i]
        return dict(physical_step=int(trace["physical_step"][i]),
                    action_index=int(trace["action_index"][i]),
                    delta_mm=(trace["delta_m"][i] * 1000).tolist(),
                    excess_mm=(np.maximum(np.abs(trace["delta_m"][i]) - LIMITS, 0) * 1000).tolist(),
                    tilt_deg=float(trace["tilt_deg"][i]), dot=float(trace["dot"][i]),
                    tilt_excess_deg=max(0., float(trace["tilt_deg"][i]) - math.degrees(ANGLE_LIMIT)),
                    position_pass=bool(passes[:3].all()), tilt_pass=bool(passes[3]),
                    failed_conditions=[label for label, value in zip(("X", "Y", "Z", "tilt"), passes) if not value],
                    score=float(trace["score"][i]))
    native = trace["native_success"][checks]
    calculated = trace["passes"][checks].all(axis=1)
    first = np.flatnonzero(native)
    return dict(final=at(final), best=at(best), ever_success=bool(native.any()),
                initial_physical_step=int(trace["physical_step"][0]),
                first_success_physical_step=int(trace["physical_step"][checks[first[0]]]) if len(first) else None,
                native_predicate_matches=bool(np.array_equal(native, calculated)))


def compare_runs(first, second):
    keys = ("physical_step", "action_index", "substep", "kind")
    aligned = all(np.array_equal(first[key], second[key]) for key in keys)
    outcome_changes = bool(first["native_success"].any()) != bool(second["native_success"].any())
    if not aligned:
        return dict(aligned=False, unstable=True, outcome_changes=outcome_changes,
                    reason="Different trace lengths or physical/check timing")
    changed = not np.array_equal(first["passes"], second["passes"])
    delta = (first["delta_m"] - second["delta_m"]) * 1000
    return dict(aligned=True, unstable=bool(changed or outcome_changes),
                outcome_changes=outcome_changes, per_condition_changes=bool(changed),
                max_relative_position_difference_mm=float(np.linalg.norm(delta, axis=1).max()),
                max_world_position_difference_mm=float(np.linalg.norm(first["bottle_pose"][:, :3] - second["bottle_pose"][:, :3], axis=1).max() * 1000),
                max_tilt_difference_deg=float(np.abs(first["tilt_deg"] - second["tilt_deg"]).max()))


def require_completed_run(status):
    """Kit can exit with code zero while unwinding a Python exception."""
    episodes = status.get("episodes", [])
    if (status.get("state") != "completed"
            or [row["seed"] for row in episodes] != list(range(1000000, 1000050))
            or any(row.get("error") for row in episodes)):
        raise RuntimeError(f"Incomplete diagnostic run: state={status.get('state')}, episodes={len(episodes)}; {status.get('error', '')}")


def worker(input_dir, output, run_id):
    """Called only in a fresh, simulator-configured Python process."""
    import faulthandler
    app = None
    task = None
    run_dir = output / f"run{run_id:02d}"
    status = dict(formal_result=False, run=run_id, state="initializing", pid=os.getpid(), started_at=now(), episodes=[])
    write_json(run_dir / "status.json", status)
    try:
        identity = read_json(output / "analysis_identity.json")
        if digest(__file__) != identity["script_sha256"]:
            raise RuntimeError("Frozen analysis script changed after preparation")
        reference_runtime = Path(read_json(output / "audit.json")["runtime"])
        for name, expected in read_json(reference_runtime / "runtime_manifest.json")["external_sources"].items():
            if digest(ROOT / name) != expected:
                raise RuntimeError(f"Simulator source changed before this round: {name}")
        import torch
        from isaaclab.app import AppLauncher
        args = argparse.Namespace(headless=True, livestream=0, enable_cameras=True, device="cuda:0")
        app = AppLauncher(args).app
        import yaml
        os.chdir(ROOT / "UniVTAC")
        sys.path.insert(0, str(ROOT / "UniVTAC"))
        module = importlib.import_module(f"envs.{TASK}")
        config = yaml.safe_load((ROOT / "UniVTAC/task_config/clean.yml").read_text())
        cfg = module.TaskCfg()
        cfg.save_dir = str(run_dir / "native")
        for name in ("decimation", "save_frequency", "video_frequency"):
            setattr(cfg, name, config.get(name, getattr(cfg, name)))
        cfg.obs_data_type = config.get("observations", {})
        cfg.random_texture = config.get("random_texture", False)
        cfg.scene.num_envs = 1
        cfg.sim.device = "cuda:0"
        task = module.Task(cfg, mode="eval")
        write_json(run_dir / "environment.json", dict(formal_result=False, physics_dt=float(task.physics_dt),
                   analysis_script_sha256=digest(__file__),
                   decimation=int(cfg.decimation), uipc_dt=float(cfg.uipc_sim.dt),
                   observations=cfg.obs_data_type, video_frequency=cfg.video_frequency,
                   scene_queries_enabled=bool(cfg.sim.enable_scene_query_support)))
        if not math.isclose(float(task.physics_dt), 1 / 120) or cfg.decimation != 1:
            raise RuntimeError("Physics timing differs from original evaluation")
        attachments = {name: len(sensor.attachment.attachment_points_idx) for name, sensor in task._tactile_manager.tactiles.items()}
        if not cfg.sim.enable_scene_query_support or not all(attachments.values()):
            raise RuntimeError(f"Invalid tactile physics initialization: {attachments}")
        # Same preflight seed and eval_test mode as native validate_expert_baseline.
        task.mode = "eval_test"
        faulthandler.dump_traceback_later(90, repeat=True)
        task.reset(seed=0)
        if not task.plan_success:
            raise RuntimeError("Expert pre_move failed")
        task.play_once()
        faulthandler.cancel_dump_traceback_later()
        expert = bool(task.plan_success) and bool(task.check_success())
        write_json(run_dir / "expert_preflight.json", dict(seed=0, success=expert, attachments=attachments, formal_result=False))
        task.clean_cache(result="diagnostic_expert_success" if expert else "diagnostic_expert_failed")
        if not expert:
            raise RuntimeError("Expert preflight failed; replay cannot be trusted")
        task.mode = "eval"
        metadata = read_json(input_dir / "metadata.json")
        instructions = read_json(ROOT / f"UniVTAC/instructions/{TASK}.json")["seen"]
        for seed in range(1000000, 1000050):
            status.update(state="replaying", current_seed=seed, updated_at=now())
            write_json(run_dir / "status.json", status)
            row = metadata[str(seed)]
            samples = []
            result = dict(seed=seed, run=run_id, formal_result=False, evidence="replay_inference", original_result=row["result"])
            try:
                task.reset(seed=seed, instructions=instructions)
                if not task.plan_success:
                    raise RuntimeError("Native pre_move failed")
                task.cfg.step_lim = row["policy_execution"]["physics_call_limit"]
                task.mean_steps = task.cfg.step_lim
                native_check = task.check_success

                def capture(index, substep, native, kind):
                    bottle = task.bottle.get_pose()
                    relative = bottle.rebase(task.place_target)
                    metric = relative_metrics(relative.p, relative.to_transformation_matrix()[2, 2])
                    joint = task._robot_manager.get_observations(["joint"])["joint"]
                    samples.append(dict(physical_step=int(task.step_count), action_index=index,
                                        substep=substep, kind=kind, physical_calls=int(task.take_action_cnt),
                                        eval_success_before_latch=bool(task.eval_success), plan_success=bool(task.plan_success),
                                        bottle_pose=bottle.tolist(), target_pose=task.place_target.tolist(),
                                        shelf_pose=task.shelf.get_pose().tolist(), relative_pose=relative.tolist(),
                                        measured_joint=joint.detach().cpu().numpy()[:8].tolist(),
                                        native_success=native, **metric))

                checks, reason = execute_saved_actions(task, row,
                    lambda values: torch.as_tensor(values, dtype=torch.float32, device=task._robot_manager.device), capture, native_check)
                result.update(termination=reason, checks=checks, complete_saved_sequence=len(checks) == len(row["tactile_feedback"]["actions"]))
                max_arm = max((x["arm_error_rad"] for x in checks), default=0)
                max_grip = max((x["grip_error_m"] for x in checks), default=0)
                count_match = task.take_action_cnt == row["policy_execution"]["physical_action_calls"]
                step_match = all(x["physical_step_matches"] for x in checks) and task.step_count == row["cost_step"]
                outcome_match = bool(task.eval_success) == (row["result"] == "success")
                result.update(max_arm_error_rad=max_arm, max_grip_error_m=max_grip,
                              physical_calls_match=count_match, physical_steps_match=step_match,
                              original_outcome_matches=outcome_match,
                              reference_consistent=bool(max_arm <= ARM_TOL and max_grip <= GRIP_TOL and count_match and step_match and outcome_match))
            except Exception:
                result.update(error=traceback.format_exc(), reference_consistent=False)
            if samples:
                trace = {key: np.asarray([sample[key] for sample in samples]) for key in samples[0]}
                trace_dir = run_dir / "traces"
                trace_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(trace_dir / f"{seed}.npz", **trace)
                result.update(summarize_trace(trace) or {})
                if not result.get("native_predicate_matches", True):
                    result["reference_consistent"] = False
            write_json(run_dir / "episodes" / f"{seed}.json", result)
            task.clean_cache(result="diagnostic")
            status["episodes"].append(dict(seed=seed, error=result.get("error"), consistent=result.get("reference_consistent")))
            write_json(run_dir / "status.json", status)
            print(json.dumps(dict(event="episode_completed", run=run_id, seed=seed, consistent=result.get("reference_consistent"))), flush=True)
            if result.get("error"):
                raise RuntimeError("Episode error: stopping to preserve reset-order evidence")
        status.update(state="completed", finished_at=now())
    except Exception:
        status.update(state="failed", error=traceback.format_exc(), finished_at=now())
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        write_json(run_dir / "status.json", status)
        if task is not None:
            task.close()
        if app is not None:
            app.close()


def replay(input_dir, output, gpu):
    import fcntl
    record = audit(input_dir, output)
    if not record["valid"]:
        raise RuntimeError("Input audit failed; inspect audit.json")
    gpu_info = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=uuid,name,memory.free", "--format=csv,noheader,nounits"], text=True, timeout=15).strip().split(", ")
    if len(gpu_info) != 3 or "4090" not in gpu_info[1]:
        raise RuntimeError(f"Expected original RTX 4090 GPU, got {gpu_info}")
    lock_path = input_dir.parent.parent.parent / f".{gpu_info[0]}.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        processes = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
        if processes:
            raise RuntimeError(f"GPU is in use by existing jobs: {processes}")
        write_json(output / "gpu.json", dict(uuid=gpu_info[0], name=gpu_info[1], free_mib=int(gpu_info[2]), formal_result=False))
        snapshot = output / "analysis_snapshot.py"
        if snapshot.exists():
            raise RuntimeError("Refusing to overwrite an existing analysis snapshot")
        shutil.copyfile(__file__, snapshot)
        write_json(output / "analysis_identity.json", dict(formal_result=False,
                   script_sha256=digest(snapshot), script=str(snapshot),
                   arm_tolerance_rad=ARM_TOL, gripper_tolerance_m=GRIP_TOL,
                   position_limits_m=LIMITS.tolist(), tilt_cosine_limit=COS_LIMIT))
        for run_id in (1, 2):
            if digest(snapshot) != read_json(output / "analysis_identity.json")["script_sha256"]:
                raise RuntimeError("Frozen analysis script changed between rounds")
            run_dir = output / f"run{run_id:02d}"
            if (run_dir / "status.json").exists():
                raise RuntimeError(f"Refusing to overwrite an existing replay: {run_dir}")
            run_dir.mkdir(parents=True)
            env = dict(os.environ, TWAM_ROOT=str(ROOT), SIM_PREFIX=str(ROOT / ".conda/univtac-sim"),
                       SHELF_DIAGNOSTIC_ROOT=str(ROOT),
                       CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE="1",
                       OMNI_WORKSPACE_ROOT=str(output / "kit"),
                       PI05_MODEL_LD_LIBRARY_PATH="", PYTHONUNBUFFERED="1",
                       PI05_SIM_GRAPHICS_LIBRARY_PATH=str(ROOT / ".cache/sim-graphics-runtime/lib"),
                       PI05_EVAL_OUTPUT_ROOT=str(run_dir / "native"))
            for key in ("PYTHONHOME", "PYTHONPATH", "PI05_SIM_NATIVE_ROOT", "LD_LIBRARY_PATH"):
                env.pop(key, None)
            env["PATH"] = str(ROOT / ".conda/univtac-sim/bin") + os.pathsep + env.get("PATH", "")
            # eval-worker sources exactly the original simulator environment.
            command = ["bash", str(ROOT / "pi05-univtac/scripts/eval-worker.sh"), str(snapshot),
                       "_worker", "--input", str(input_dir), "--output", str(output), "--run-id", str(run_id)]
            print(json.dumps(dict(event="run_started", run=run_id, command=command)), flush=True)
            with (run_dir / "eval.log").open("w") as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                write_json(output / "processes.json", dict(formal_result=False, replay_parent_pid=os.getpid(),
                           worker_pid=child.pid, run=run_id))
                previous_progress = None
                while child.poll() is None:
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        pass
                    state_path = run_dir / "status.json"
                    if state_path.exists():
                        state = read_json(state_path)
                        progress = (state["state"], len(state["episodes"]))
                        if progress != previous_progress:
                            report(input_dir, output)
                            previous_progress = progress
                if child.returncode:
                    raise subprocess.CalledProcessError(child.returncode, command)
                require_completed_run(read_json(run_dir / "status.json"))
                for seed in range(1000000, 1000050):
                    result = read_json(run_dir / "episodes" / f"{seed}.json")
                    if not result.get("native_predicate_matches") or not (run_dir / "traces" / f"{seed}.npz").is_file():
                        raise RuntimeError(f"Missing or invalid numeric trace for seed {seed}")
            report(input_dir, output)
    current = {str(path.relative_to(input_dir)): digest(path) for path in input_dir.rglob("*") if path.is_file()}
    if current != record["original_files_sha256"]:
        raise RuntimeError("Original evaluation files changed while diagnostics were running")
    write_json(output / "completed.json", dict(formal_result=False, completed_at=now(), original_files_unchanged=True))
    # Only our newly generated diagnostic videos/cache are disposable.
    for run_id in (1, 2):
        video_dir = output / f"run{run_id:02d}" / "native/video"
        if video_dir.is_dir():
            for video in video_dir.glob("*.mp4"):
                video.unlink()
        native = output / f"run{run_id:02d}" / "native"
        for name in (".cache", "scene"):
            if (native / name).is_dir():
                shutil.rmtree(native / name)


def load_trace(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def episode_figure(seed, traces):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), layout="constrained")
    for run, trace in traces.items():
        steps = trace["physical_step"]
        for i, label in enumerate(("X", "Y", "Z")):
            axes[0, i].plot(steps, trace["delta_m"][:, i] * 1000, label=f"run {run}", alpha=.8)
            axes[0, i].axhspan(-LIMITS[i] * 1000, LIMITS[i] * 1000, color="green", alpha=.06)
            axes[0, i].set(title=f"{label} error (mm)", xlabel="physical step")
        axes[1, 0].plot(steps, trace["tilt_deg"], label=f"run {run}")
        for ax, j in ((axes[1, 1], 1), (axes[1, 2], 2)):
            ax.plot(trace["delta_m"][:, 0] * 1000, trace["delta_m"][:, j] * 1000, alpha=.7)
            ax.scatter(trace["delta_m"][-1, 0] * 1000, trace["delta_m"][-1, j] * 1000, marker="x")
    for i in range(3):
        for sign in (-1, 1):
            axes[0, i].axhline(sign * LIMITS[i] * 1000, color="red", linestyle="--", linewidth=.8)
    axes[1, 0].axhline(math.degrees(ANGLE_LIMIT), color="red", linestyle="--")
    axes[1, 0].set(title="Tilt (degrees)", xlabel="physical step")
    for ax, j, label in ((axes[1, 1], 1, "Y"), (axes[1, 2], 2, "Z")):
        ax.add_patch(Rectangle((-20, -LIMITS[j] * 1000), 40, LIMITS[j] * 2000, fill=False, edgecolor="red", linestyle="--"))
        ax.set(xlabel="X error (mm)", ylabel=f"{label} error (mm)", title="Target frame; cross = final")
    axes[0, 0].legend()
    fig.suptitle(f"Seed {seed} — replay inference")
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", metadata={"Date": None})
    plt.close(fig)
    svg = buffer.getvalue()
    return svg[svg.index("<svg"):]


def report(input_dir, output):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _report_unlocked(input_dir, output)


def _report_unlocked(input_dir, output):
    audit_record = read_json(output / "audit.json")
    rows, details, consistency = [], [], []
    reason_counts = Counter()
    for original in audit_record["episodes"]:
        seed = original["seed"]
        traces, results = {}, {}
        for run in (1, 2):
            path = output / f"run{run:02d}/episodes/{seed}.json"
            if path.exists():
                results[run] = read_json(path)
            path = output / f"run{run:02d}/traces/{seed}.npz"
            if path.exists():
                traces[run] = load_trace(path)
        cross = compare_runs(traces[1], traces[2]) if len(traces) == 2 else None
        reference_ok = len(results) == 2 and all(r.get("reference_consistent") for r in results.values())
        credible = reference_ok and cross is not None and not cross["unstable"]
        confidence = "replay_inference_consistent" if credible else ("pending_replay" if not results else "replay_inference_unconfirmed")
        flat = dict(seed=seed, original_result=original["original_result"], original_termination=original["termination"],
                    original_failure_reason=original["failure_reason"], evidence=confidence,
                    two_run_stable=None if cross is None else not cross["unstable"])
        for key in ("actions", "physical_calls", "initial_step", "final_step"):
            flat[f"original_{key}"] = original.get(key)
        comparisons = dict(seed=seed, evidence=confidence, cross_run=cross,
                           runs={str(k): {key: value for key, value in r.items() if key != "checks"} for k, r in results.items()})
        consistency.append(comparisons)
        for run, result in results.items():
            if "initial_physical_step" in result and original.get("initial_step") is not None:
                flat[f"run{run}_initial_step_offset"] = result["initial_physical_step"] - original["initial_step"]
            for moment in ("final", "best"):
                sample = result.get(moment)
                if not sample:
                    continue
                prefix = f"run{run}_{moment}"
                for i, axis in enumerate("xyz"):
                    flat[f"{prefix}_d{axis}_mm"] = sample["delta_mm"][i]
                    flat[f"{prefix}_{axis}_excess_mm"] = sample["excess_mm"][i]
                for key in ("physical_step", "tilt_deg", "dot", "score", "position_pass", "tilt_pass"):
                    flat[f"{prefix}_{key}"] = sample[key]
                flat[f"{prefix}_tilt_excess_deg"] = max(0., sample["tilt_deg"] - math.degrees(ANGLE_LIMIT))
                flat[f"{prefix}_failed_conditions"] = "+".join(sample["failed_conditions"])
            for key in ("ever_success", "reference_consistent", "max_arm_error_rad", "max_grip_error_m", "termination"):
                flat[f"run{run}_{key}"] = result.get(key)
        finals = [r["final"] for r in results.values() if "final" in r]
        if len(finals) == 2:
            for i, axis in enumerate("xyz"):
                values = [f["delta_mm"][i] for f in finals]
                flat[f"final_d{axis}_mm_min"], flat[f"final_d{axis}_mm_max"] = min(values), max(values)
            flat["final_tilt_deg_min"] = min(f["tilt_deg"] for f in finals)
            flat["final_tilt_deg_max"] = max(f["tilt_deg"] for f in finals)
        rows.append(flat)
        if original["original_result"] == "failed":
            if credible:
                for condition in finals[0]["failed_conditions"]:
                    reason_counts[condition] += 1
            narrative = [f"原记录终止原因：{original['termination']}。证据：{confidence}。"]
            if original["failure_reason"]:
                narrative.append(original["failure_reason"])
            if not results:
                narrative.append("尚无回放位姿，无法量化瓶子误差。")
            for run, result in results.items():
                if "final" in result:
                    f, b = result["final"], result["best"]
                    narrative.append(f"回放 {run}：最终 Δxyz={np.round(f['delta_mm'], 3).tolist()} mm，倾角 {f['tilt_deg']:.3f}°；超限项 {','.join(f['failed_conditions']) or '无'}。最接近时刻 step={b['physical_step']}，score={b['score']:.5f}；全程达标={result['ever_success']}。")
                if result.get("ever_success"):
                    narrative.append("原记录失败但回放达标：须调查复现和检查时序，不能据此宣布原评测误判。")
                if result.get("error"):
                    narrative.append(result["error"])
            svg = episode_figure(seed, traces) if traces else ""
            details.append(f"<details><summary>Seed {seed} · {html.escape(confidence)}</summary><p>{'<br>'.join(html.escape(n) for n in narrative)}</p>{svg}<pre>{html.escape(json.dumps(comparisons, ensure_ascii=False, indent=2))}</pre></details>")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "consistency.json", dict(formal_result=False, episodes=consistency))
    statuses = {str(run): read_json(path) if (path := output / f"run{run:02d}/status.json").exists() else {"state": "not_started"} for run in (1, 2)}
    confirmed = sum(row["evidence"] == "replay_inference_consistent" for row in rows)
    summary = dict(formal_result=False, created_at=now(), original_counts=audit_record["counts"],
                   replay_consistent_episodes=confirmed, run_status=statuses,
                   confirmed_failure_conditions=dict(reason_counts), audit_valid=audit_record["valid"],
                   note="All recovered object poses are replay inferences, not original object-state measurements.")
    summary["original_failed_replay_success_seeds"] = {
        str(run): [row["seed"] for row in rows if row["original_result"] == "failed" and row.get(f"run{run}_ever_success")]
        for run in (1, 2)}
    write_json(output / "summary.json", summary)
    table_rows = []
    for row in rows:
        values = [row["seed"], row["original_result"], row["original_termination"], row["evidence"]]
        values += [" / ".join(f"{row[f'run{run}_final_d{axis}_mm']:.3f}" if f"run{run}_final_d{axis}_mm" in row else "待回放" for run in (1, 2)) for axis in "xyz"]
        table_rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in values) + "</tr>")
    document = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Shelf failure diagnostics</title>
<style>body{{font:15px system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#1b2734}}table{{border-collapse:collapse;font-size:12px}}td,th{{border:1px solid #ccd4dc;padding:6px}}details{{margin:16px 0;border:1px solid #ccd4dc;padding:12px}}summary{{cursor:pointer;font-weight:600}}svg{{max-width:100%;height:auto}}pre{{white-space:pre-wrap;font-size:11px}}.notice{{background:#fff4d1;padding:16px}}</style>
<h1>put_bottle_in_shelf 数值失败诊断</h1><p>原结果：27 / 50 成功，23 次失败。formal_result=false</p>
<p class="notice">全部物体位姿均为动作回放推断。关节吻合不能证明原始接触轨迹一致。未完成或不一致的回放不提供原轨迹的精确位姿结论。</p>
<p>当前完成两轮且与原关节/计数/结果一致：{confirmed}/50。运行状态：{html.escape(', '.join(f'run {k}: {v["state"]}' for k,v in statuses.items()))}。</p>
<p>原生标准：|Δx| &lt; 20 mm，|Δy| &lt; 100 mm，|Δz| &lt; 20 mm，cos(倾角) &gt; 0.965（{math.degrees(ANGLE_LIMIT):.5f}°）。所有比较为严格不等式；目标采用原 place_target。</p>
<p>可信回放失败项计数（允许重叠）：{html.escape(json.dumps(dict(reason_counts), ensure_ascii=False))}。未确认的样本不纳入此统计。</p>
<h2>全部原始 episode；误差列为 run 1 / run 2</h2><table><tr><th>seed</th><th>原结果</th><th>原终止原因</th><th>证据</th><th>Δx mm</th><th>Δy mm</th><th>Δz mm</th></tr>{''.join(table_rows)}</table>
<h2>23 次失败的逐例证据</h2>{''.join(details)}<p>输入：{html.escape(str(input_dir))}</p></html>"""
    (output / "report.html").write_text(document)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "replay", "report", "_worker"))
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--run-id", type=int, choices=(1, 2))
    args = parser.parse_args()
    input_dir, output = args.input.resolve(), args.output.resolve()
    if output == input_dir or input_dir in output.parents:
        parser.error("Output cannot be inside original evaluation")
    if args.stage == "_worker":
        if args.run_id is None:
            parser.error("_worker requires --run-id")
        worker(input_dir, output, args.run_id)
    elif args.stage == "audit":
        record = audit(input_dir, output)
        print(json.dumps(dict(valid=record["valid"], counts=record["counts"], errors=record["errors"])))
        return 0 if record["valid"] else 1
    elif args.stage == "report":
        if not (output / "audit.json").exists():
            audit(input_dir, output)
        print(json.dumps(report(input_dir, output), ensure_ascii=False))
    else:
        try:
            replay(input_dir, output, args.gpu)
        except Exception:
            write_json(output / "replay_error.json", dict(formal_result=False, error=traceback.format_exc(), at=now()))
            if (output / "audit.json").exists():
                report(input_dir, output)
            raise
        report(input_dir, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
