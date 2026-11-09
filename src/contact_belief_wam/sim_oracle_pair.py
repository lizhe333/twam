"""One same-seed live WAVE-0 failure and same-instance oracle recovery pair.

Reinitializing CuRobo can change the grasp slightly. This path creates a NEW
paired reference, explicitly distinct from the old saved trace. It preserves
all simulator/sensor state at the selected boundary and validates restoration.
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

import numpy as np

from .paths import WORKSPACE, atomic_json
from .sim_takeover import Runner, TeacherStopped, make_trace
from .takeover import TakeoverConfig, compare_replay, finalize_supervision


def fork_prefix(runner, root, checkpoint):
    """Copy the actually executed shared prefix without stepping or resetting history."""
    context = dict(runner.trace_context)
    trace = make_trace(root, checkpoint, context)
    trace._open(runner.task)
    original = runner.trace.file
    for group in ("observations", "actions"):
        del trace.file[group]
        original.copy(original[group], trace.file, name=group)
    for name in ("prism_triangles", "slot_triangles"):
        original.copy(original[name], trace.file, name=name)
    for key, value in original.attrs.items():
        trace.file.attrs[key] = value
    trace.action_count, trace.observation_count = runner.step, runner.step+1
    trace.fk_checked = runner.trace.fk_checked
    trace.calibration_valid = runner.trace.calibration_valid
    trace.latest_observation = deepcopy(runner.row())
    trace.file.flush()
    branch = Runner.__new__(Runner)
    branch.task, branch.trace, branch.cfg = runner.task, trace, runner.cfg
    branch.detector, branch.metadata = deepcopy(runner.detector), deepcopy(runner.metadata)
    branch.teacher_events, branch.success, branch.trace_context = [], False, context
    return branch


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True, help="Historical trace, provenance only; not claimed identical")
    p.add_argument("--oracle-config", type=Path, required=True)
    p.add_argument("--takeover-step", type=int, default=40)
    p.add_argument("--seed", type=int, default=3000001)
    p.add_argument("--port", type=int, default=18891)
    p.add_argument("--policy-device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--physical-diagnostic", action="store_true")
    p.add_argument("--model-python", default=str(WORKSPACE/"contact-belief-wam/.venv/bin/python"))
    os.chdir(WORKSPACE/"UniVTAC")
    sys.path[:0] = [str(WORKSPACE/"UniVTAC"), str(WORKSPACE/"pi05-univtac/src")]
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(argv)
    args.output, args.checkpoint = args.output.resolve(), args.checkpoint.resolve()
    args.headless, args.enable_cameras, args.livestream = True, True, 0
    app = AppLauncher(args).app
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    task = policy = baseline = branch = probe = None
    result = dict(kind="oracle", seed=args.seed, status="error", success=False, qualified=False,
        controller="privileged_oracle", ground_truth_used_online=True, training_enabled=False,
        excluded_from_training=True, user_approved=False, reference_recreated_live=True,
        historical_reference=str(args.reference), historical_trace_strict_replay_passed=False,
        takeover_step=args.takeover_step, takeover_selection="offline_selected_boundary_of_same_seed_reproduction")
    result["physical_diagnostic_requested"] = args.physical_diagnostic
    cfg = TakeoverConfig()
    try:
        import h5py
        import yaml
        from pi05_univtac.constants import prompt
        from pi05_univtac.contact_restore import capture_task_state, restore_task_state
        from pi05_univtac.contact_recovery import RecoveryPlanError
        from policy.Wave0Wam.deploy_policy import Policy, RecordedController
        module = importlib.import_module("envs.insert_HDMI")
        task_cfg = module.TaskCfg()
        clean = yaml.safe_load((WORKSPACE/"UniVTAC/task_config/clean.yml").read_text())
        task_cfg.save_dir, task_cfg.decimation = args.output/"native", 1
        task_cfg.save_frequency, task_cfg.video_frequency, task_cfg.render_frequency = 2, 2, 0
        task_cfg.tactile_sensor_type, task_cfg.obs_data_type = "gsmini", clean["observations"]
        task_cfg.random_texture = clean.get("random_texture", False)
        task_cfg.step_lim, task_cfg.reset_time_limit = 1200, 300
        task_cfg.sim.device = args.device or "cuda:0"
        task = module.Task(task_cfg, mode="eval")
        task.reset(seed=args.seed, instructions=[prompt("insert_HDMI")])
        signal.signal(signal.SIGTERM, interrupt)
        if not task.plan_success:
            raise RuntimeError("native_pre_move_failed")
        context = dict(source="wave0", phase="natural", decision_step=0, policy_eligible=False)
        baseline = Runner(task, make_trace(args.output/"paired_baseline/trace", args.checkpoint, context), cfg)
        baseline.trace_context = context
        os.environ.update(WAVE0_CHECKPOINT=str(args.checkpoint), PI05_SERVER_LOG=str(args.output/"policy-server.log"))
        os.environ.pop("PI05_TRACE_DIR", None)
        policy = Policy(dict(task_name="insert_HDMI", pi05_eval_protocol="bounded-actions-fixed-seeds-v2",
                             pi05_python=args.model_python, pi05_port=args.port,
                             wave0_device=args.policy_device, wave0_cpu_threads=16 if args.policy_device == "cpu" else 2))
        if policy.client.metadata.get("camera_mode") != "head_wrist":
            raise RuntimeError("Live reproduction requires the selected dual-view WAVE-0 service")
        result["inference_metadata"] = policy.client.metadata
        controller = RecordedController(policy.client, "insert_HDMI", arm_clip_tolerance=.02)
        snapshot = None
        while baseline.step < cfg.budget and not baseline.success:
            if baseline.step == args.takeover_step:
                snapshot = capture_task_state(task)
                branch = fork_prefix(baseline, args.output/"trace", args.checkpoint)
                atomic_json(args.output/"snapshot.json", dict(step=baseline.step,
                    physics_step=task.step_count, uipc_frame=snapshot.uipc_frame,
                    capability=snapshot.capability.as_dict(), prefix_copied_without_physics=True))
            action = controller.next_action(task, task._get_observations())
            baseline.execute(action, raw=controller.last_raw_target, replan=controller.last_replan,
                             source="wave0", phase="natural", decision_step=baseline.step//10*10)
        outcome = "success" if baseline.success else "failed"
        baseline.trace.finish(task, outcome)
        result["reference"] = str(baseline.trace.path)
        atomic_json(args.output/"paired_baseline/result.json", dict(seed=args.seed, kind="natural_reproduction",
            status=outcome, success=baseline.success, strategy_actions=baseline.step,
            trajectory=str(baseline.trace.path), action_metadata=baseline.metadata,
            detector=baseline.detector.report(), training_enabled=False))
        policy.close(); policy = None
        if baseline.success:
            raise RuntimeError("same_seed_live_reproduction_succeeded_no_failure_to_rescue")
        if snapshot is None or branch is None:
            raise RuntimeError("takeover_snapshot_missing")

        restore_task_state(task, snapshot)
        context = dict(source="validation", phase="restore_probe", decision_step=args.takeover_step, policy_eligible=False)
        probe = Runner(task, make_trace(args.output/"restore_probe/trace", args.checkpoint, context), cfg)
        probe.trace_context = context
        with h5py.File(baseline.trace.path, "r") as reference:
            a = args.takeover_step
            # Ten real actions verify dynamics/sensors, not just restored cached images.
            for t in range(a, a+10):
                probe.execute(reference["actions/clipped_actual_target"][t], source="validation",
                              phase="restore_probe", decision_step=a)
            expected = [{k:d[t] for k,d in reference["observations"].items()} for t in range(a,a+11)]
            actual = [probe.row(i) for i in range(11)]
            check = compare_replay(expected, actual, reference["actions/clipped_actual_target"][a:a+10],
                                   probe.trace.file["actions/clipped_actual_target"][:])
            check.update(method="same_instance_physx_uipc_snapshot", shared_prefix_actions=a,
                exact_shared_prefix=True, forward_probe_actions=10, reference_recreated_live=True,
                historical_trace_strict_replay_passed=False)
            atomic_json(args.output/"replay_check.json", check)
            result["replay"] = check
            from .oracle_recovery import physical_equivalence
            physical = physical_equivalence(check, expected, actual)
            result["physical_replay"] = physical
            atomic_json(args.output/"physical_replay_check.json", physical)
            probe.trace.finish(task, "validation_passed" if check["passed"] else "validation_failed")
            if not check["passed"] and not (args.physical_diagnostic and physical["passed"]):
                raise RuntimeError(f"same_instance_restore_probe_failed: {check['violations'][:5]}")
            restore_task_state(task, snapshot)
            restored = branch.trace._observation_defaults(task, task._get_observations())
            recheck = compare_replay([expected[0]], [restored], [], [])
            atomic_json(args.output/"takeover_restore_check.json", recheck)
            physical_recheck = physical_equivalence(recheck, [expected[0]], [restored])
            result["physical_takeover_restore"] = physical_recheck
            atomic_json(args.output/"physical_takeover_restore_check.json", physical_recheck)
            if not recheck["passed"] and not (args.physical_diagnostic and physical_recheck["passed"]):
                raise RuntimeError("second_restore_at_takeover_failed")
        from .oracle_recovery import OracleTeacher
        teacher = OracleTeacher(branch, json.loads(args.oracle_config.read_text()))
        result["oracle"] = dict(config=teacher.config, events=teacher.events)
        try:
            teacher.run()
        except (TeacherStopped, RecoveryPlanError) as exc:
            result["teacher_stop_reason"] = str(exc)
        result.update(status="success" if branch.success else "failed", success=branch.success)
    except BaseException as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        if policy is not None:
            policy.close()
        if branch is not None:
            result.update(strategy_actions=branch.step, detector=branch.detector.report(),
                action_metadata=branch.metadata, teacher_events=branch.teacher_events,
                policy_supervision_ranges=finalize_supervision(branch.metadata),
                context_range=[0,branch.step+1], wm_context_continuous=True,
                trajectory=str(branch.trace.path))
            if branch.trace.file is not None:
                branch.trace.file["actions/policy_eligible"][:] = False
                branch.trace.finish(task, result["status"], reason=result.get("error", ""))
        for remaining in (baseline,probe):
            if remaining is not None and remaining.trace.file is not None:
                remaining.trace.finish(task, "error", reason=result.get("error", "interrupted"))
        atomic_json(args.output/"result.json", result)
        if task is not None:
            task.close()
        app.close()
    print(json.dumps({k:result.get(k) for k in ("seed","status","success","strategy_actions","error")}), flush=True)
    return 1 if result["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
