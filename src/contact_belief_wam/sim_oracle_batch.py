"""One native WAVE-0 rollout and optional same-instance privileged recovery.

Launch through scripts/sim.sh. Quotas belong to oracle_batch_campaign; this
worker reserves recovery only after a completed natural failure. No training.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import traceback
from types import SimpleNamespace

import numpy as np

from .paths import WORKSPACE, atomic_json
from .sim_takeover import Runner, TeacherStopped, make_trace
from .oracle_batch_detector import PrivilegedStallDetector, surface_distance


def batch_trace(root, checkpoint, context, task_name):
    trace = make_trace(root, checkpoint, context)
    trace.task_name = task_name
    return trace


class BatchRunner(Runner):
    def __init__(self, task, trace, config, axis, context):
        self.task, self.trace, self.cfg, self.trace_context = task, trace, config, context
        self.metadata, self.teacher_events = [], []
        self.success = False
        self.detector = PrivilegedStallDetector(axis, **getattr(config, "detector", {}))
        self.trace.append_observation(task, task._get_observations())
        self.topology = {name: trace.file[f"{name}_triangles"][:] for name in ("prism", "slot")}
        self.last_distance = None
        self.observe_detector()

    def observe_detector(self):
        row = self.row()
        self.last_distance = surface_distance(row, self.topology)
        return self.detector.observe(touch=np.stack([row[f"{s}_rgb_marker"] for s in ("left", "right")]),
            object_position=row["prism_pose_world"][:3], observed_fk=row["observed_fk_world"],
            command_fk=self.trace.file["actions/command_q_fk_ee_world"][-1] if self.step else None,
            distance=self.last_distance)

    def contact_distance(self):
        if self.last_distance < .0005:
            return float(self.last_distance)
        from .oracle_batch_geometry import surface_clearance
        return float(surface_clearance(self.row(), self.topology))


def fork_prefix(baseline, root, checkpoint, task_name, step, detector):
    """Copy a bounded executed prefix from the CLOSED natural HDF5 trace."""
    import h5py
    context = dict(source="oracle", phase="takeover", decision_step=step, policy_eligible=False)
    trace = batch_trace(root, checkpoint, context, task_name)
    trace._open(baseline.task)
    with h5py.File(baseline.trace.path, "r") as original:
        for group, count in (("observations", step+1), ("actions", step)):
            for key, source in original[group].items():
                target = trace.file[group].create_dataset(key, shape=(count,)+source.shape[1:],
                    dtype=source.dtype, maxshape=(None,)+source.shape[1:], chunks=source.chunks)
                for offset in range(0, count, 8):
                    target[offset:offset+8] = source[offset:min(count, offset+8)]
                for k, v in source.attrs.items():
                    target.attrs[k] = v
        for name in ("prism_triangles", "slot_triangles"):
            original.copy(original[name], trace.file, name=name)
        for key, value in original.attrs.items():
            trace.file.attrs[key] = value
        trace.latest_observation = {k: v[step] for k, v in original["observations"].items()}
    trace.file.attrs.update(terminal_status="running", terminal_reason="", action_rows=step, observation_rows=step+1)
    trace.action_count, trace.observation_count = step, step+1
    trace.fk_checked = int(trace.file["actions/fk_valid"][:].sum())
    trace.calibration_valid = bool(step and trace.file["actions/calibration_valid"][:].all())
    trace.file.attrs.update(calibration_checked_actions=trace.fk_checked, calibration_valid=trace.calibration_valid)
    trace.file.flush()
    branch = BatchRunner.__new__(BatchRunner)
    branch.task, branch.trace, branch.cfg = baseline.task, trace, baseline.cfg
    branch.detector, branch.metadata = deepcopy(detector), deepcopy(baseline.metadata[:step])
    branch.teacher_events, branch.success, branch.trace_context = [], False, context
    branch.topology = baseline.topology
    branch.last_distance = surface_distance(trace.latest_observation, branch.topology)
    return branch


def mark_labels(branch, events):
    flags = np.zeros(branch.step, dtype=bool)
    if branch.success:
        for event in events:
            if event.get("bc_eligible"):
                flags[event["step"]:event["stop"]] = True
    branch.trace.file["actions/policy_eligible"][:] = flags
    for row, eligible in zip(branch.metadata, flags):
        row["policy_eligible"] = bool(eligible)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--campaign-root", type=Path, required=True)
    p.add_argument("--task", choices=("insert_HDMI", "insert_hole"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    os.chdir(WORKSPACE/"UniVTAC")
    sys.path[:0] = [str(WORKSPACE/"UniVTAC"), str(WORKSPACE/"pi05-univtac/src")]
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(argv)
    args.output = args.output.resolve()
    config = json.loads(args.config.read_text())
    task_config = config["tasks"][args.task]
    checkpoint = Path(task_config["checkpoint"]).resolve()
    cfg = SimpleNamespace(budget=int(task_config["budget"]), execute_steps=10, detector=config.get("detector", {}))
    if cfg.budget != {"insert_HDMI": 600, "insert_hole": 317}[args.task]:
        raise ValueError("Native deployment budget mismatch")
    args.headless, args.enable_cameras, args.livestream = True, True, 0
    app = AppLauncher(args).app
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    task = policy = baseline = branch = teacher = None
    result = dict(task=args.task, seed=args.seed, budget=cfg.budget, checkpoint=str(checkpoint),
        status="error", demonstrator="privileged_oracle", training_enabled=False,
        natural=None, recovery=None, student_uses_ground_truth=False)
    try:
        import yaml
        from scipy.spatial.transform import Rotation
        from pi05_univtac.constants import prompt
        from pi05_univtac.contact_restore import capture_task_state, restore_task_state
        from pi05_univtac.contact_recovery import RecoveryPlanError
        from policy.Wave0Wam.deploy_policy import Policy, RecordedController
        from .oracle_batch_campaign import BatchLedger
        from .oracle_batch_teacher import BatchOracleTeacher
        from .takeover import compare_replay
        from .oracle_recovery import physical_equivalence
        ledger = BatchLedger(args.campaign_root)
        # Worker cannot be used to silently evade parent reservation accounting.
        reservations = [r for r in ledger.snapshot()["attempts"] if r["task"] == args.task and r["seed"] == args.seed]
        if len(reservations) != 1 or reservations[0]["status"] != "running":
            raise RuntimeError("Worker requires one live natural-rollout reservation")
        reservation = reservations[0]
        if Path(reservation["directory"]).resolve() != args.output:
            raise RuntimeError("Worker output does not match its reservation")
        result["collection_provenance"] = dict(
            mode="fresh_wave0_rollout_same_instance_recovery",
            source_revision=reservation.get("source_revision"),
            revisited_missed_candidate=bool(reservation.get("prior_attempts")),
            previous_results=[str(Path(old["directory"]) / "result.json")
                              for old in reservation.get("prior_attempts", [])])
        module = importlib.import_module(f"envs.{args.task}")
        native_cfg = module.TaskCfg()
        clean = yaml.safe_load((WORKSPACE/"UniVTAC/task_config/clean.yml").read_text())
        native_cfg.save_dir, native_cfg.decimation = args.output/"native", 1
        native_cfg.save_frequency, native_cfg.video_frequency, native_cfg.render_frequency = 2, 2, 0
        native_cfg.tactile_sensor_type, native_cfg.obs_data_type = "gsmini", clean["observations"]
        native_cfg.random_texture = clean.get("random_texture", False)
        native_cfg.step_lim, native_cfg.reset_time_limit = 2*cfg.budget, 300
        native_cfg.sim.device = args.device or "cuda:0"
        task = module.Task(native_cfg, mode="eval")
        task.reset(seed=args.seed, instructions=[prompt(args.task)])
        signal.signal(signal.SIGTERM, interrupt)
        if not task.plan_success:
            # Native initialization failure is a consumed rollout, never a recovery.
            result.update(status="completed", natural=dict(success=False, status="initialization_failed", strategy_actions=0, takeover_step=None))
            ledger.record_natural(args.task, args.seed, result["natural"])
        else:
            pose = np.asarray(task.target_pose.tolist() if args.task == "insert_hole" else task.slot.get_pose().tolist())
            axis = Rotation.from_quat(pose[[4, 5, 6, 3]]).apply([0., 0., -1.])
            context = dict(source="wave0", phase="natural", decision_step=0, policy_eligible=False)
            baseline = BatchRunner(task, batch_trace(args.output/"natural/trace", checkpoint, context, args.task), cfg, axis, context)
            os.environ.update(WAVE0_CHECKPOINT=str(checkpoint), PI05_SERVER_LOG=str(args.output/"policy-server.log"))
            os.environ.pop("PI05_TRACE_DIR", None)
            policy = Policy(dict(task_name=args.task, pi05_eval_protocol="bounded-actions-fixed-seeds-v2",
                pi05_python=config["model_python"], pi05_port=config["port"],
                wave0_device=config["policy_device"], wave0_cpu_threads=2))
            metadata = policy.client.metadata
            if metadata.get("camera_mode") != "head_wrist" or metadata.get("use_ema") is not True:
                raise RuntimeError("Selected dual-view EMA WAVE-0 input contract was not satisfied")
            if Path(metadata["checkpoint"]).resolve() != checkpoint:
                raise RuntimeError("WAVE-0 service checkpoint mismatch")
            result["inference_metadata"] = metadata
            controller = RecordedController(policy.client, args.task, arm_clip_tolerance=.02)
            snapshot = saved_detector = None
            snapshot_step = None
            natural_reason = "budget_exhausted"
            while baseline.step < cfg.budget and not baseline.success:
                if task.check_early_stop():
                    natural_reason = "native_early_stop"
                    break
                action = controller.next_action(task, task._get_observations())
                baseline.execute(action, raw=controller.last_raw_target, replan=controller.last_replan,
                    source="wave0", phase="natural", decision_step=baseline.step//10*10)
                candidate = baseline.detector.first_confirmed
                if candidate is None:
                    snapshot = saved_detector = snapshot_step = None
                elif candidate != snapshot_step and baseline.step < cfg.budget and not baseline.success:
                    # Read-only sensors are already primed by the post-action recorder.
                    # Snapshot writes native UIPC state without rendering or stepping.
                    snapshot = capture_task_state(task)
                    saved_detector, snapshot_step = deepcopy(baseline.detector), candidate
                    atomic_json(args.output/"snapshot.json", dict(step=candidate, uipc_frame=snapshot.uipc_frame,
                        capability=snapshot.capability.as_dict(), selection="first_persistent_native_object_stall"))
            if baseline.success:
                baseline.trace.file["actions/policy_eligible"][:] = True
                natural_reason = "native_success"
            baseline.trace.finish(task, "success" if baseline.success else "failed", natural_reason)
            natural = dict(success=bool(baseline.success), status="success" if baseline.success else "failed",
                reason=natural_reason, strategy_actions=baseline.step, trajectory=str(baseline.trace.path),
                takeover_step=snapshot_step, detector=baseline.detector.report())
            result["natural"] = natural
            ledger.record_natural(args.task, args.seed, natural)
            atomic_json(args.output/"natural/result.json", dict(task=args.task, seed=args.seed, **natural))
            policy.close(); policy = None
            if baseline.success:
                result["recovery_status"] = "natural_success_no_intervention"
            elif snapshot is None:
                result["recovery_status"] = "no_persistent_contact_candidate"
            else:
                ledger.reserve_recovery(args.task, args.seed)
                recovery = result["recovery"] = dict(success=False, status="failed", takeover_step=snapshot_step,
                    events=[], strategy_actions=snapshot_step, timing=None)
                restore_task_state(task, snapshot)
                branch = fork_prefix(baseline, args.output/"recovery/trace", checkpoint, args.task, snapshot_step, saved_detector)
                expected = deepcopy(branch.row())
                restored = branch.trace._observation_defaults(task, task._get_observations())
                physical = physical_equivalence(compare_replay([expected], [restored], [], []), [expected], [restored])
                physical["scope"] = "same_instance_physical_restore_for_imitation"
                recovery["physical_restore"] = physical
                recovery["trajectory"] = str(branch.trace.path)
                atomic_json(args.output/"physical_restore.json", physical)
                if not physical["passed"]:
                    recovery["reason"] = "physical_restore_mismatch"
                else:
                    # The prefix remains the actual natural branch input at this
                    # boundary; successor observations come from real recovery.
                    branch.trace.latest_observation = expected
                    branch.last_distance = surface_distance(expected, branch.topology)
                    candidate = saved_detector.candidate
                    tc, tt = candidate["t_contact"], candidate["t_tactile"]
                    recovery["timing"] = dict(t_contact=tc, t_tactile=tt, t_decision=snapshot_step,
                        t_recovery=snapshot_step, passed=tc is not None and tt is not None and tc <= tt <= snapshot_step,
                        tactile_window_indices=candidate["input_tactile_indices"])
                    teacher = BatchOracleTeacher(branch, args.task, task_config.get("teacher"))
                    recovery["teacher_config"] = teacher.config
                    recovery["events"] = teacher.events
                    try:
                        teacher.run()
                    except (TeacherStopped, RecoveryPlanError) as exc:
                        recovery["reason"] = str(exc)
                    finally:
                        recovery.update(success=bool(branch.success), status="success" if branch.success else "failed",
                                        events=teacher.events, strategy_actions=branch.step)
                mark_labels(branch, recovery["events"])
                branch.trace.finish(task, recovery["status"], recovery.get("reason", ""))
                atomic_json(args.output/"recovery/result.json", dict(task=args.task, seed=args.seed, **recovery))
            result["status"] = "completed"
    except BaseException as exc:
        result.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                      error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        if policy is not None:
            policy.close()
        for runner in (baseline, branch):
            if runner is not None and runner.trace.file is not None:
                runner.trace.finish(task, result["status"], result.get("error", ""))
        atomic_json(args.output/"result.json", result)
        if task is not None:
            task.close()
        app.close()
    print(json.dumps({k: result.get(k) for k in ("task", "seed", "status", "error")}), flush=True)
    return 1 if result["status"] in ("error", "interrupted") else 0


if __name__ == "__main__":
    raise SystemExit(main())
