"""One paired insert_hole preview, with bounded robot-command perturbations.

This entry point neither enrolls data nor trains. It preserves the unmodified
dual-view WAVE-0 inference path and the complete physical action history.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import sys
import traceback
from types import SimpleNamespace

import numpy as np

from .paths import WORKSPACE, atomic_json
from .sim_oracle_batch import BatchRunner, batch_trace, fork_prefix
from .sim_takeover import TeacherStopped
from .hole_preview import label_ranges, validate_preview_labels


def finish(runner, kind, destination, *, perturb=None, takeover=None, events=(), **extra):
    sources = runner.trace.file["actions/source"].asstr()[:]
    phases = runner.trace.file["actions/phase"].asstr()[:]
    decisions = runner.trace.file["actions/decision_step"][:]
    start, stop = (perturb["start"], perturb["stop"]) if perturb else (None, None)
    ranges, flags = label_ranges(sources, phases, decisions, success=runner.success,
        kind=kind, perturb_start=start, perturb_stop=stop, takeover=takeover, teacher_events=events)
    runner.trace.file["actions/policy_eligible"][:] = flags
    check = validate_preview_labels(sources, ranges, start, stop)
    result = dict(kind=kind, success=bool(runner.success), trajectory=str(runner.trace.path),
        strategy_actions=runner.step, observations=runner.step+1, budget=runner.cfg.budget,
        task=runner.trace.task_name,
        perturbation=perturb, takeover_step=takeover, events=list(events),
        policy_supervision_ranges=ranges, label_check=check, detector=runner.detector.report(),
        training_enabled=False, demonstrator="privileged_oracle" if takeover is not None else "wave0",
        student_uses_ground_truth=False, dynamics_valid="all_actual_robot_commands",
        **extra)
    runner.trace.finish(runner.task, "success" if runner.success else "failed", extra.get("reason", ""))
    atomic_json(destination / "result.json", result)
    print(json.dumps(dict(event="branch_finished", kind=kind, success=result["success"],
                          actions=result["strategy_actions"], path=str(destination)), ensure_ascii=False), flush=True)
    return result


def restore_branch(baseline, snapshot, detector, step, directory, checkpoint):
    from pi05_univtac.contact_restore import restore_task_state
    from .oracle_recovery import physical_equivalence
    from .takeover import compare_replay
    restore_task_state(baseline.task, snapshot)
    branch = fork_prefix(baseline, directory / "trace", checkpoint, baseline.trace.task_name, step, detector)
    expected = deepcopy(branch.row())
    actual = branch.trace._observation_defaults(branch.task, branch.task._get_observations())
    checked = physical_equivalence(compare_replay([expected], [actual], [], []), [expected], [actual])
    atomic_json(directory / "physical_restore.json", checked)
    if not checked["passed"]:
        raise RuntimeError(f"Physical restoration failed: {checked['violations']}")
    # Keep the recorded prefix observation, including its genuine touch history.
    branch.trace.latest_observation = expected
    return branch, checked


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    os.chdir(WORKSPACE / "UniVTAC")
    sys.path[:0] = [str(WORKSPACE/"UniVTAC"), str(WORKSPACE/"pi05-univtac/src")]
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(argv)
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise RuntimeError("Preview output already has a result; refusing to overwrite")
    if config.get("training_enabled") is not False or len(config["trials"]) > 6:
        raise ValueError("Preview is capped at six perturbation trials, with training disabled")
    checkpoint = Path(config["checkpoint"]).resolve()
    args.headless, args.enable_cameras, args.livestream = True, True, 0
    app = AppLauncher(args).app
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    task = policy = None
    runners = []
    result = dict(status="running", task="insert_hole", seed=config["seed"],
                  checkpoint=str(checkpoint), training_enabled=False, branches=[])
    try:
        import yaml
        from envs.insert_hole import Task, TaskCfg
        from pi05_univtac.constants import prompt
        from pi05_univtac.contact_restore import capture_task_state
        from pi05_univtac.contact_recovery import RecoveryPlanError
        from policy.Wave0Wam.deploy_policy import Policy, RecordedController
        from .oracle_batch_teacher import BatchOracleTeacher, rotation, ee_for_object_pose
        native = TaskCfg()
        clean = yaml.safe_load((WORKSPACE / "UniVTAC/task_config/clean.yml").read_text())
        native.save_dir, native.decimation = output / "native", 1
        native.save_frequency, native.video_frequency, native.render_frequency = 2, 2, 0
        native.tactile_sensor_type, native.obs_data_type = "gsmini", clean["observations"]
        native.random_texture = clean.get("random_texture", False)
        native.step_lim, native.reset_time_limit = 634, 300
        native.sim.device = args.device or "cuda:0"
        task = Task(native, mode="eval")
        task.reset(seed=int(config["seed"]), instructions=[prompt("insert_hole")])
        if not task.plan_success:
            raise RuntimeError("Native initialization failed")
        axis = rotation(task.target_pose.tolist()).apply([0., 0., -1.])
        cfg = SimpleNamespace(budget=317, execute_steps=10, detector=config.get("detector", {}))
        context = dict(source="wave0", phase="normal", decision_step=0, policy_eligible=False)
        baseline = BatchRunner(task, batch_trace(output / "normal/trace", checkpoint, context, "insert_hole"), cfg, axis, context)
        runners.append(baseline)
        os.environ.update(WAVE0_CHECKPOINT=str(checkpoint), PI05_SERVER_LOG=str(output / "policy-server.log"))
        os.environ.pop("PI05_TRACE_DIR", None)
        policy = Policy(dict(task_name="insert_hole", pi05_eval_protocol="bounded-actions-fixed-seeds-v2",
            pi05_python=config["model_python"], pi05_port=config["port"], wave0_device="cuda", wave0_cpu_threads=2))
        metadata = policy.client.metadata
        if metadata.get("camera_mode") != "head_wrist" or metadata.get("use_ema") is not True or Path(metadata["checkpoint"]).resolve() != checkpoint:
            raise RuntimeError("Dual-view 10k EMA contract not satisfied")
        result["inference_metadata"] = metadata
        controller = RecordedController(policy.client, "insert_hole", arm_clip_tolerance=.02)
        snapshots = {}
        required_steps = {int(t["step"]) for t in config["trials"]}
        while baseline.step < cfg.budget and not baseline.success:
            if baseline.step in required_steps and baseline.step not in snapshots:
                snapshots[baseline.step] = (capture_task_state(task), deepcopy(baseline.detector))
            action = controller.next_action(task, task._get_observations())
            baseline.execute(action, raw=controller.last_raw_target, replan=controller.last_replan,
                source="wave0", phase="normal", decision_step=baseline.step//10*10)
            if task.check_early_stop():
                break
        normal = finish(baseline, "normal", output / "normal")
        result["branches"].append(normal)
        atomic_json(output / "progress.json", result)
        if not baseline.success:
            raise RuntimeError("The paired unperturbed branch did not succeed")
        accepted_self = accepted_oracle = False
        for trial in config["trials"]:
            if accepted_self and accepted_oracle:
                break
            if trial["purpose"] == "self_corrected" and accepted_self:
                continue
            if trial["purpose"] == "oracle_recovery" and accepted_oracle:
                continue
            directory = output / "trials" / trial["id"]
            step = int(trial["step"])
            if step not in snapshots:
                raise RuntimeError("Perturbation boundary was not reached by normal execution")
            branch, physical = restore_branch(baseline, *snapshots[step], step, directory, checkpoint)
            runners.append(branch)
            row = branch.row()
            before = np.asarray(row["prism_pose_world"]).copy()
            desired = before.copy()
            offset = np.asarray(trial["offset_mm"], float)
            if offset.shape != (2,) or np.linalg.norm(offset) > 5.:
                raise ValueError("Perturbation exceeds the 5 mm local preview bound")
            desired[:3] += rotation(task.target_pose.tolist()).apply(np.r_[offset/1000, 0.])
            target = ee_for_object_pose(row["ee_pose_world"], before, desired)
            print(f"PREVIEW perturb {trial['id']} step={step} offset_mm={offset.tolist()}", flush=True)
            branch.move(target, phase="perturb", source="perturb", eligible=False, duration=10)
            perturb = dict(start=step, stop=branch.step, offset_target_frame_mm=offset.tolist(),
                           actual_object_delta_world_mm=((branch.row()["prism_pose_world"][:3]-before[:3])*1000).tolist(),
                           mechanism="recorded_robot_IK_commands", bc_eligible=False,
                           before_object_pose=before.tolist(), after_object_pose=branch.row()["prism_pose_world"].tolist())
            # Fresh inference at the post-perturbation observation, never reuse
            # the pre-perturbation open-loop chunk or block WAVE-0 corrections.
            controller = RecordedController(policy.client, "insert_hole", arm_clip_tolerance=.02)
            saved = saved_detector = None
            saved_step = None
            while branch.step < cfg.budget and not branch.success:
                action = controller.next_action(task, task._get_observations())
                branch.execute(action, raw=controller.last_raw_target, replan=controller.last_replan,
                    source="wave0", phase="wave0_after_perturb", decision_step=branch.step//10*10)
                candidate = branch.detector.first_confirmed
                if candidate is None:
                    saved = saved_detector = saved_step = None
                elif candidate >= perturb["stop"] + 10 and candidate != saved_step and not branch.success:
                    saved, saved_detector, saved_step = capture_task_state(task), deepcopy(branch.detector), candidate
                if task.check_early_stop():
                    break
            rollout = finish(branch, "self_corrected", directory, perturb=perturb,
                             physical_restore=physical, trial=trial)
            result["branches"].append(rollout)
            if branch.success:
                accepted_self = True
            elif saved is not None and not accepted_oracle:
                recovery_dir = directory / "recovery"
                recovered, checked = restore_branch(branch, saved, saved_detector, saved_step, recovery_dir, checkpoint)
                runners.append(recovered)
                teacher = BatchOracleTeacher(recovered, "insert_hole", config.get("teacher"))
                reason = ""
                try:
                    teacher.run()
                except (TeacherStopped, RecoveryPlanError) as exc:
                    reason = str(exc)
                candidate = saved_detector.candidate
                timing = dict(t_contact=candidate["t_contact"], t_tactile=candidate["t_tactile"],
                              t_decision=saved_step, t_recovery=saved_step,
                              tactile_window_indices=candidate["input_tactile_indices"])
                recovered_result = finish(recovered, "oracle_recovery", recovery_dir, perturb=perturb,
                    takeover=saved_step, events=teacher.events, timing=timing, physical_restore=checked,
                    reason=reason, perturbed_reference=rollout["trajectory"], teacher_config=teacher.config, trial=trial)
                result["branches"].append(recovered_result)
                accepted_oracle = bool(recovered.success)
            atomic_json(output / "progress.json", result)
        result.update(status="completed", accepted=dict(normal=bool(baseline.success),
                      self_corrected=accepted_self, oracle_recovery=accepted_oracle))
    except BaseException as exc:
        result.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                      error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        if policy is not None:
            policy.close()
        for runner in runners:
            if runner.trace.file is not None:
                runner.trace.finish(task, result["status"], result.get("error", ""))
        atomic_json(output / "result.json", result)
        if task is not None:
            task.close()
        app.close()
    return int(result["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
