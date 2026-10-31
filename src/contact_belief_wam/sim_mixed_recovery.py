"""One scene seed: live WAVE-0, bounded physical perturbations, optional Oracle."""
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
from .mixed_recovery import MixedLedger, rank_trials, stage_for_distance
from .sim_oracle_batch import BatchRunner, batch_trace
from .sim_hole_preview import finish, restore_branch
from .sim_takeover import TeacherStopped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--task", choices=("insert_HDMI", "insert_hole"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    os.chdir(WORKSPACE / "UniVTAC")
    sys.path[:0] = [str(WORKSPACE/"UniVTAC"), str(WORKSPACE/"pi05-univtac/src")]
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    tc = config["tasks"][args.task]
    output, checkpoint = args.output.resolve(), Path(tc["checkpoint"])
    ledger = MixedLedger(args.campaign_root)
    reservation = next(a for a in ledger.snapshot()["attempts"] if a["task"] == args.task and a["seed"] == args.seed)
    if reservation["status"] != "running" or Path(reservation["directory"]) != output:
        raise ValueError("Worker requires its durable live seed reservation")
    if (output / "result.json").exists():
        raise ValueError("Refusing to repeat a consumed seed")
    output.mkdir(parents=True, exist_ok=True)
    args.headless, args.enable_cameras, args.livestream = True, True, 0
    app = AppLauncher(args).app
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    task = policy = None
    runners = []
    result = dict(task=args.task, seed=args.seed, checkpoint=str(checkpoint), budget=tc["budget"],
                  status="running", training_enabled=False, branches=[], skipped=[],
                  provenance="live_wave0_physical_perturbation_same_instance_recovery")
    try:
        import yaml
        from pi05_univtac.constants import prompt
        from pi05_univtac.contact_restore import capture_task_state
        from pi05_univtac.contact_recovery import RecoveryPlanError
        from policy.Wave0Wam.deploy_policy import Policy, RecordedController
        from .oracle_batch_teacher import BatchOracleTeacher, rotation, ee_for_object_pose, relative_pose
        from .hole_preview_teacher import InContactHoleOracle
        module = importlib.import_module(f"envs.{args.task}")
        native = module.TaskCfg()
        clean = yaml.safe_load((WORKSPACE/"UniVTAC/task_config/clean.yml").read_text())
        native.save_dir, native.decimation = output/"native", 1
        native.save_frequency, native.video_frequency, native.render_frequency = 2, 2, 0
        native.tactile_sensor_type, native.obs_data_type = "gsmini", clean["observations"]
        native.random_texture = clean.get("random_texture", False)
        native.step_lim, native.reset_time_limit = 2*tc["budget"], 300
        native.sim.device = args.device or "cuda:0"
        task = module.Task(native, mode="eval")
        task.reset(seed=args.seed, instructions=[prompt(args.task)])
        if not task.plan_success:
            result.update(status="completed", reason="native_initialization_failed")
        else:
            reference = np.asarray(task.target_pose.tolist() if args.task == "insert_hole" else task.slot.get_pose().tolist())
            cfg = SimpleNamespace(budget=tc["budget"], execute_steps=10, detector=config["detector"])
            axis = rotation(reference).apply([0, 0, -1])
            result["native_target_pose"] = reference.tolist()
            result["target_axis_world"] = axis.tolist()
            context = dict(source="wave0", phase="normal", decision_step=0, policy_eligible=False)
            baseline = BatchRunner(task, batch_trace(output/"normal/trace", checkpoint, context, args.task), cfg, axis, context)
            runners.append(baseline)
            initial_grasp = relative_pose(baseline.row()["ee_pose_world"], baseline.row()["prism_pose_world"])
            os.environ.update(WAVE0_CHECKPOINT=str(checkpoint), PI05_SERVER_LOG=str(output/"policy-server.log"))
            os.environ.pop("PI05_TRACE_DIR", None)
            policy = Policy(dict(task_name=args.task, pi05_eval_protocol="bounded-actions-fixed-seeds-v2",
                pi05_python=config["model_python"], pi05_port=config["port"], wave0_device=config["policy_device"],
                wave0_cpu_threads=2))
            metadata = policy.client.metadata
            if (metadata.get("camera_mode") != "head_wrist" or metadata.get("use_ema") is not True
                    or Path(metadata["checkpoint"]).resolve() != checkpoint.resolve()):
                raise ValueError("Selected dual-view 10k EMA inference contract was not satisfied")
            result["inference_metadata"] = metadata
            snapshots = {}
            need = ledger.needs(args.task)

            def stopped():
                return (ledger.root/"STOP_AFTER_CURRENT").exists()

            def save(runner, kind, directory, *, branch_id=None, **kw):
                details = finish(runner, kind, directory, seed=args.seed, checkpoint=str(checkpoint),
                    native_target_pose=reference.tolist(), target_axis_world=axis.tolist(),
                    same_instance_prefix=True, provenance=result["provenance"], **kw)
                path = directory/"result.json"
                ledger.record_result(args.task, args.seed, path, branch_id)
                result["branches"].append(str(path))
                atomic_json(output/"progress.json", result)
                return details

            def rollout(runner, phase, *, collect_stages=False, earliest_candidate=0):
                # New inference context after each physical perturbation; no stale action chunk.
                controller = RecordedController(policy.client, args.task, arm_clip_tolerance=.02)
                saved = None
                while runner.step < cfg.budget and not runner.success:
                    if collect_stages and runner.step >= 6 and runner.step % 10 == 0 and runner.detector.t_contact is None:
                        stage = stage_for_distance(runner.last_distance)
                        grasp = relative_pose(runner.row()["ee_pose_world"], runner.row()["prism_pose_world"])
                        stable = (np.linalg.norm(grasp[:3]-initial_grasp[:3]) <= .002 and
                                  np.rad2deg((rotation(grasp)*rotation(initial_grasp).inv()).magnitude()) <= 3.)
                        if stage and stage not in snapshots and stable and not task.check_early_stop():
                            snapshots[stage] = (capture_task_state(task), deepcopy(runner.detector), runner.step)
                    action = controller.next_action(task, task._get_observations())
                    runner.execute(action, raw=controller.last_raw_target, replan=controller.last_replan,
                        source="wave0", phase=phase, decision_step=runner.step//10*10)
                    candidate = runner.detector.first_confirmed
                    if candidate is None:
                        saved = None
                    elif candidate >= earliest_candidate and (saved is None or saved[2] != candidate) and not runner.success:
                        if runner.step != candidate:
                            raise ValueError("A stall snapshot must be captured at its own decision observation")
                        saved = (capture_task_state(task), deepcopy(runner.detector), candidate)
                    if task.check_early_stop():
                        break
                return saved

            oracle_attempted = False
            accepted_self = accepted_oracle = False

            def recover(parent, snapshot, failed, *, perturb=None, trial=None):
                nonlocal oracle_attempted, accepted_oracle
                if not need["oracle_recovery"] or oracle_attempted or snapshot is None or stopped():
                    return
                bid = ledger.reserve_branch(args.task, args.seed, "oracle")
                if bid is None:
                    return
                oracle_attempted = True
                directory = output/bid
                branch, physical = restore_branch(parent, snapshot[0], snapshot[1], snapshot[2], directory, checkpoint)
                runners.append(branch)
                cls = InContactHoleOracle if args.task == "insert_hole" else BatchOracleTeacher
                teacher = cls(branch, args.task, tc["teacher"])
                reason = ""
                try:
                    teacher.run()
                except (TeacherStopped, RecoveryPlanError) as exc:
                    reason = str(exc)
                candidate = snapshot[1].candidate
                timing = dict(t_contact=candidate["t_contact"], t_tactile=candidate["t_tactile"],
                    t_decision=snapshot[2], t_recovery=snapshot[2], tactile_window_indices=candidate["input_tactile_indices"])
                entry = save(branch, "oracle_recovery", directory, branch_id=bid, perturb=perturb,
                    takeover=snapshot[2], events=teacher.events, timing=timing, physical_restore=physical,
                    reason=reason, natural_reference=failed["trajectory"], trial=trial,
                    teacher_config=teacher.config, teacher_revision=tc["teacher_revision"],
                    collection_origin="controlled_robot_perturbation" if perturb else "natural_failure")
                accepted_oracle = bool(entry["success"])

            saved = rollout(baseline, "normal", collect_stages=True)
            normal = save(baseline, "normal", output/"normal", collection_origin="unperturbed")
            if not baseline.success:
                recover(baseline, saved, normal)
                if saved is None:
                    result["reason"] = "natural_failure_without_persistent_contact_evidence"
            elif not stopped() and (need["self_corrected"] or need["oracle_recovery"]):
                tried = 0
                while tried < tc["max_perturb_per_seed"]:
                    if stopped() or ((accepted_self or not need["self_corrected"]) and (oracle_attempted or not need["oracle_recovery"])):
                        break
                    state = ledger.snapshot()
                    used = [b.get("trial") for b in MixedLedger._attempt(state, args.task, args.seed)["branches"]]
                    ranked = rank_trials(config, state, args.task, args.seed)
                    available = [t for t in ranked if t["stage"] in snapshots and t not in used]
                    if not available:
                        result["skipped"].append(dict(reason="no_eligible_precontact_snapshot", stages=list(snapshots)))
                        break
                    trial = available[0]
                    bid = ledger.reserve_branch(args.task, args.seed, "perturb", trial=trial)
                    if bid is None:
                        break
                    tried += 1
                    snapshot, detector, step = snapshots[trial["stage"]]
                    directory = output/bid
                    branch, physical = restore_branch(baseline, snapshot, detector, step, directory, checkpoint)
                    runners.append(branch)
                    before = branch.row()["prism_pose_world"].copy()
                    desired = before.copy()
                    desired[:3] += rotation(reference).apply(np.r_[np.asarray(trial["offset_mm"])/1000, 0.])
                    target = ee_for_object_pose(branch.row()["ee_pose_world"], before, desired)
                    error = ""
                    try:
                        branch.move(target, phase="perturb", source="perturb", eligible=False, duration=10)
                    except (TeacherStopped, RecoveryPlanError) as exc:
                        error = str(exc)
                    perturb = dict(start=step, stop=branch.step, offset_target_frame_mm=trial["offset_mm"],
                        actual_object_delta_world_mm=((branch.row()["prism_pose_world"][:3]-before[:3])*1000).tolist(),
                        before_object_pose=before.tolist(), after_object_pose=branch.row()["prism_pose_world"].tolist(),
                        mechanism="recorded_robot_IK_commands", bc_eligible=False, stage=trial["stage"],
                        direction=trial["direction"], amplitude_mm=trial["amplitude_mm"])
                    if error or branch.step-step != 10 or branch.success:
                        # A success caused by the intervention itself is never a WAVE-0 correction.
                        entry = save(branch, "self_corrected", directory, branch_id=bid, perturb=perturb,
                            trial=trial, physical_restore=physical, reason=error or "perturbation_not_completed_before_success",
                            eligible_for_export=False, collection_origin="controlled_robot_perturbation")
                        continue
                    # Only subsequent WAVE-0 insertion commands can establish
                    # a stalled attempt. Preserve touch/reference/history while
                    # discarding a candidate caused by the intervention itself.
                    branch.detector.first_confirmed = None
                    branch.detector.candidate = None
                    saved = rollout(branch, "wave0_after_perturb", earliest_candidate=perturb["stop"]+10)
                    entry = save(branch, "self_corrected", directory, branch_id=bid, perturb=perturb,
                        trial=trial, physical_restore=physical, collection_origin="controlled_robot_perturbation")
                    if branch.success:
                        from .mixed_recovery_export import self_correction_check
                        check = self_correction_check(entry, args.task)
                        # Keep every attempt; only a measured correction satisfies the target.
                        atomic_json(directory/"self_correction.json", check)
                        accepted_self |= check["passed"]
                    else:
                        recover(branch, saved, entry, perturb=perturb, trial=trial)
            result.update(status="completed", accepted=dict(normal=bool(baseline.success),
                self_corrected=accepted_self, oracle_recovery=accepted_oracle), snapshot_stages=list(snapshots))
    except BaseException as exc:
        result.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                      error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        if policy is not None:
            policy.close()
        for runner in runners:
            if runner.trace.file is not None:
                runner.trace.finish(task, result["status"], result.get("error", ""))
        atomic_json(output/"result.json", result)
        if task is not None:
            task.close()
        app.close()
    return int(result["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
