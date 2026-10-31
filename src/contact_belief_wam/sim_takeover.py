"""One native HDMI rollout or verified replay/takeover (run with sim.sh).

WAVE-0 inference and action execution are imported from the validated deployment.
Privileged scene data is confined to the recorder and offline acceptance report,
except for explicitly requested ``oracle`` diagnostic branches (never BC data).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import traceback

import numpy as np

from .paths import WORKSPACE, atomic_json
from .takeover import (BlockageDetector, TakeoverConfig, VERSION, compare_replay,
                       feedback_decision, relative_grasp_pose, finalize_supervision)


class TeacherStopped(RuntimeError):
    """An expected bounded-teacher failure, not an infrastructure error."""


def make_trace(root, checkpoint, context):
    from pi05_univtac.contact_trace import ContactTrace, curobo_fk_ee_world

    class TakeoverTrace(ContactTrace):
        def _append_dataset(self, group, key, value, *, compression=True):
            # Pilot storage is plentiful. Lossless, uncompressed chunks avoid
            # spending simulator wall time repeatedly compressing four cameras.
            return super()._append_dataset(group, key, value, compression=False)

        def _observation_defaults(self, task, observation):
            row = super()._observation_defaults(task, observation)
            row["object_in_gripper_pose_world"] = relative_grasp_pose(row)
            row["observed_fk_world"] = curobo_fk_ee_world(task, row["joint_q"][:7], row["root_pose_world"])
            # Native UIPC surface geometry is audit-only, never a Teacher input.
            for name in ("prism", "slot"):
                actor = getattr(task, name)
                row[f"{name}_vertices_world"] = np.asarray(actor.vertices, np.float32).copy()
                if f"{name}_triangles" not in self.file:
                    sim = actor._uipc_sim
                    surface = sim.sio.simplicial_surface(2)
                    triangles = np.asarray(surface.triangles().topo().view()).reshape(-1, 3)
                    lo, hi = sim._surf_vertex_offsets[actor.obj_id-1:actor.obj_id+1]
                    selected = triangles[((triangles >= lo) & (triangles < hi)).all(1)] - lo
                    if not len(selected):
                        raise RuntimeError(f"Missing native surface topology for {name}")
                    self.file.create_dataset(f"{name}_triangles", data=selected.astype(np.int32))
            self.latest_observation = row
            return row

        def append_action(self, *args, **kwargs):
            # Append custom columns before the base recorder increments N.
            self._open(args[0])
            for key, value in context.items():
                # h5py cannot assign NumPy unicode into an existing fixed-byte
                # column. Encode on EVERY row, not only dataset creation.
                encoded = np.asarray(value.encode("utf-8"), dtype="S64") if isinstance(value, str) else np.asarray(value)
                self._append_dataset(self.file["actions"], key, encoded)
            super().append_action(*args, **kwargs)

    return TakeoverTrace(root, checkpoint, "insert_HDMI", 0)


class Runner:
    def __init__(self, task, trace, config):
        self.task, self.trace, self.cfg = task, trace, config
        self.detector = BlockageDetector(config)
        self.metadata, self.teacher_events = [], []
        self.success = False
        self.trace.append_observation(task, task._get_observations())
        self.observe_detector()

    @property
    def step(self):
        return self.trace.action_count

    def row(self, index=-1):
        if index == -1 and hasattr(self.trace, "latest_observation"):
            return self.trace.latest_observation
        return {k: v[index] for k, v in self.trace.file["observations"].items()}

    def observe_detector(self):
        row = self.row()
        touch = np.stack([row[f"{side}_rgb_marker"] for side in ("left", "right")])
        command_fk = self.trace.file["actions/command_q_fk_ee_world"][-1] if self.step else None
        return self.detector.observe(touch, row["observed_fk_world"], command_fk)

    def execute(self, action, *, source, phase, decision_step, eligible=False, raw=None, replan=False):
        from pi05_univtac.sim_policy import execute_action
        from .sim_eval import robot_action_bounds
        if self.step >= self.cfg.budget:
            raise TeacherStopped("strategy_action_budget_exhausted")
        action = np.asarray(action, np.float32)
        bounds = robot_action_bounds(self.task)
        if action.shape != (8,) or not np.isfinite(action).all() or np.any(action < bounds[:, 0]-1e-6) or np.any(action > bounds[:, 1]+1e-6):
            raise ValueError("Invalid actuator target; Teacher never clips an infeasible waypoint")
        observation = self.task._get_observations()
        begin, calls = self.task.step_count, self.task.take_action_cnt
        metadata = dict(source=source, phase=phase, decision_step=int(decision_step), policy_eligible=bool(eligible))
        self.trace_context.update(metadata)
        error = ""
        try:
            self.success, _ = execute_action(self.task, action)
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            self.trace.append_action(self.task, observation, raw_target=action if raw is None else raw,
                clipped_target=action, chunk_index=self.step // self.cfg.execute_steps,
                replan=replan, physics_step_begin=begin, action_count_begin=calls,
                success=not error, error=error)
            self.metadata.append(metadata)
        self.observe_detector()
        if self.step % self.cfg.execute_steps == 0:
            atomic_json(self.trace.root / "progress.json", dict(strategy_actions=self.step,
                first_confirmed=self.detector.first_confirmed, success=self.success,
                phase=phase, physical_step=int(self.task.step_count)))
        return self.success

    def move(self, target, *, phase, eligible, duration=10, gripper=None, source="teacher"):
        from pi05_univtac.contact_recovery import _solve_one_waypoint
        start = self.row()["joint_q"].copy()
        solver = None
        if source in {"oracle", "perturb"}:
            if not hasattr(self, "_oracle_ik_solver"):
                from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
                base = self.task._robot_manager.planner.motion_gen.ik_solver
                # Pre-move's captured CUDA graph has a different problem shape.
                # A separate solver leaves that planner and its graph untouched.
                self._oracle_ik_solver = IKSolver(IKSolverConfig.load_from_robot_config(
                    base.robot_config, tensor_args=base.tensor_args,
                    world_coll_checker=base.world_coll_checker, num_seeds=1,
                    position_threshold=base.position_threshold,
                    rotation_threshold=base.rotation_threshold,
                    use_cuda_graph=False, high_precision=True))
            solver = self._oracle_ik_solver
        q, diagnostic = _solve_one_waypoint(self.task._robot_manager, target, start[:7], label=phase,
                                            ik_solver=solver)
        decision = self.step
        self.teacher_events.append(dict(step=decision, phase=phase, ik=diagnostic))
        for fraction in np.linspace(0, 1, duration + 1)[1:]:
            action = start.copy()
            action[:7] = start[:7] + fraction * (q - start[:7])
            if gripper is not None:
                action[7] = start[7] + fraction * (gripper - start[7])
            if self.execute(action, source=source, phase=phase, decision_step=decision, eligible=eligible):
                break

    def run_teacher(self, seed):
        from scipy.spatial.transform import Rotation
        cfg = self.cfg
        origin = self.row()["ee_pose_world"].copy()
        contact_z = float(origin[2])
        rotation0 = Rotation.from_quat(origin[[4, 5, 6, 3]])
        unloaded = origin.copy()
        unloaded[2] += cfg.retreat_mm / 1000
        self.move(unloaded, phase="unload", eligible=True, duration=20)
        # No signed GT error enters this order. Use a private generator.
        directions = [(kind, axis, sign) for kind, axes in (("translation", (0, 1)), ("rotation", (0, 1, 2)))
                      for axis in axes for sign in (-1, 1)]
        offset = int(np.random.default_rng(seed).integers(len(directions)))
        directions = directions[offset:] + directions[:offset]
        candidate_index, repeated, probes, direction_known = 0, False, 0, False
        stable = self.row()["ee_pose_world"].copy()
        grip_reference = float(self.row()["joint_q"][7])
        while not self.success and self.step < cfg.budget and probes < cfg.probe_limit:
            kind, axis, sign = directions[candidate_index % len(directions)]
            candidate = stable.copy()
            if not repeated:
                if kind == "translation":
                    candidate[axis] += sign * cfg.translation_mm / 1000
                else:
                    vec = np.zeros(3); vec[axis] = np.deg2rad(sign * cfg.rotation_deg)
                    rot = Rotation.from_rotvec(vec) * Rotation.from_quat(candidate[[4, 5, 6, 3]])
                    quat = rot.as_quat(); candidate[3:] = np.r_[quat[3], quat[:3]]
            angle = np.rad2deg((Rotation.from_quat(candidate[[4, 5, 6, 3]]) * rotation0.inv()).magnitude())
            if np.linalg.norm(candidate[:2]-origin[:2]) > cfg.lateral_cap_mm/1000 + 1e-9 or angle > cfg.rotation_cap_deg + 1e-6:
                candidate_index += 1; direction_known = False; repeated = False
                if candidate_index >= len(directions):
                    break
                continue
            self.move(candidate, phase="feedback_adjust" if direction_known else "exploration",
                      eligible=direction_known)
            if self.success:
                break
            # Unloading creates free space. Returning through that space is not
            # evidence that a lateral correction improved the contact. Revisit
            # the observed blocked depth before scoring the final .5 mm probe.
            approach = self.row()["ee_pose_world"].copy()
            approach[2] = contact_z
            self.move(approach, phase="feedback_approach" if direction_known else "exploration_approach",
                      eligible=direction_known, duration=20)
            if self.success:
                break
            before = self.row()
            score_before = self.detector.scores[-1]
            probe = before["ee_pose_world"].copy(); probe[2] -= cfg.translation_mm/1000
            decision = self.step
            probe_start = self.step
            self.move(probe, phase="probe", eligible=False)
            probes += 1
            after = self.row()
            progress = float(before["observed_fk_world"][2]-after["observed_fk_world"][2])*1000
            grip_stable = abs(float(after["joint_q"][7])-grip_reference) <= .0002
            choice, reason = feedback_decision(requested_mm=cfg.translation_mm, achieved_mm=progress,
                touch_before=score_before, touch_after=self.detector.scores[-1],
                threshold=self.detector.threshold, grip_stable=grip_stable, repeated=repeated, config=cfg)
            self.teacher_events.append(dict(step=self.step, observation_decision_step=self.step,
                probe_start=probe_start, probe_number=probes, direction=[kind, axis, sign],
                progress_mm=progress, tactile_before=score_before, tactile_after=self.detector.scores[-1],
                choice=choice, reason=reason))
            if self.success:
                break
            if choice == "abort":
                raise TeacherStopped(reason)
            if choice == "continue":
                # Insertion continuation is labeled only AFTER feedback arrived.
                direction_known, repeated = True, False
                for _ in range(4):
                    p = self.row()["ee_pose_world"].copy(); p[2] -= cfg.translation_mm/1000
                    begin = self.row()["observed_fk_world"].copy()
                    self.move(p, phase="feedback_insert", eligible=True)
                    if self.success:
                        break
                    achieved = float(begin[2]-self.row()["observed_fk_world"][2])*1000
                    if achieved < cfg.translation_mm * cfg.blocked_progress_ratio:
                        # Failed continuation is not BC-positive even if later rescued.
                        for row in self.metadata[-10:]: row["policy_eligible"] = False
                        contact_z = float(self.row()["observed_fk_world"][2])
                        break
                # Observable insertion travel is a proxy, not an oracle. The
                # native success test independently decides whether this finish
                # actually inserted the object. Never align to object/slot GT.
                travel = (origin[2] - self.row()["observed_fk_world"][2]) * 1000
                if not self.success and travel >= 3. and achieved >= cfg.translation_mm * cfg.improving_progress_ratio:
                    p = self.row()["ee_pose_world"].copy()
                    self.move(p, phase="release", eligible=True, duration=20,
                              gripper=float(self.task._robot_manager.gripper_max_qpos))
                    if not self.success:
                        p[2] += .02
                        self.move(p, phase="finish_lift", eligible=True, duration=20)
                    break
                if self.success:
                    break
                stable = self.row()["ee_pose_world"].copy()
                stable[2] += cfg.retreat_mm/1000
                self.move(stable, phase="feedback_unload", eligible=True, duration=20)
            else:
                self.move(stable, phase="feedback_withdraw", eligible=True)
                if choice == "repeat":
                    repeated = True
                    # Repeat at the same candidate, not at a new unseen offset.
                    stable = candidate.copy()
                else:
                    candidate_index += 1; repeated = False; direction_known = False
                    if candidate_index >= len(directions):
                        break
        return dict(probes=probes, events=self.teacher_events, success=self.success)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--kind", choices=("natural", "recovery", "oracle"), required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--takeover-step", type=int)
    parser.add_argument("--oracle-config", type=Path)
    parser.add_argument("--port", type=int, default=18891)
    parser.add_argument("--model-python", default=str(WORKSPACE/"contact-belief-wam/.venv/bin/python"))
    # All CLI validation possible without Isaac is performed by the parent.
    os.chdir(WORKSPACE / "UniVTAC")
    sys.path[:0] = [str(WORKSPACE/"UniVTAC"), str(WORKSPACE/"pi05-univtac/src")]
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    args.output, args.checkpoint = args.output.resolve(), args.checkpoint.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.headless, args.enable_cameras, args.livestream = True, True, 0
    app = AppLauncher(args).app
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    task = policy = trace = runner = None
    result = dict(version=VERSION, seed=args.seed, kind=args.kind, status="error", qualified=False)
    if args.kind == "oracle":
        result.update(controller="privileged_oracle", ground_truth_used_online=True,
                      training_enabled=False, excluded_from_training=True, user_approved=False,
                      takeover_selection="offline_ground_truth_diagnosis")
    cfg = TakeoverConfig()
    try:
        import h5py
        import yaml
        module = importlib.import_module("envs.insert_HDMI")
        task_cfg = module.TaskCfg()
        clean = yaml.safe_load((WORKSPACE/"UniVTAC/task_config/clean.yml").read_text())
        task_cfg.save_dir, task_cfg.decimation = args.output / "native", 1
        task_cfg.save_frequency, task_cfg.video_frequency, task_cfg.render_frequency = 2, 2, 0
        task_cfg.tactile_sensor_type = "gsmini"
        task_cfg.obs_data_type = clean["observations"]
        task_cfg.random_texture = clean.get("random_texture", False)
        task_cfg.step_lim, task_cfg.reset_time_limit = 1200, 300
        task_cfg.sim.device = args.device or "cuda:0"
        task = module.Task(task_cfg, mode="eval")
        from pi05_univtac.constants import prompt
        task.reset(seed=args.seed, instructions=[prompt("insert_HDMI")])
        signal.signal(signal.SIGTERM, interrupt)
        if not task.plan_success:
            raise RuntimeError("native_pre_move_failed")
        context = dict(source="wave0", phase="natural", decision_step=0, policy_eligible=False)
        trace = make_trace(args.output / "trace", args.checkpoint, context)
        runner = Runner(task, trace, cfg)
        runner.trace_context = context
        if args.kind == "natural":
            from policy.Wave0Wam.deploy_policy import Policy, RecordedController
            os.environ.update(WAVE0_CHECKPOINT=str(args.checkpoint), PI05_SERVER_LOG=str(args.output/"policy-server.log"))
            os.environ.pop("PI05_TRACE_DIR", None)
            policy = Policy(dict(task_name="insert_HDMI", pi05_eval_protocol="bounded-actions-fixed-seeds-v2",
                pi05_python=args.model_python, pi05_port=args.port))
            if policy.client.metadata.get("camera_mode") != "head_wrist":
                raise RuntimeError("Pilot requires a genuinely dual-view inference service")
            controller = RecordedController(policy.client, "insert_HDMI", arm_clip_tolerance=.02)
            result["inference_metadata"] = policy.client.metadata
            while runner.step < cfg.budget and not runner.success:
                observation = task._get_observations()
                action = controller.next_action(task, observation)
                runner.execute(action, raw=controller.last_raw_target, replan=controller.last_replan,
                    source="wave0", phase="natural", decision_step=(runner.step//10)*10)
                if task.check_early_stop():
                    break
        else:
            if args.reference is None or args.takeover_step is None or args.takeover_step <= 0 or args.takeover_step % 10:
                raise ValueError("Recovery requires a reference and a decision-boundary takeover index")
            with h5py.File(args.reference, "r") as reference:
                if int(reference.attrs["seed"]) != args.seed or args.takeover_step >= len(reference["actions/clipped_actual_target"]):
                    raise ValueError("Reference seed/index does not match this branch")
                if reference.attrs["terminal_status"] == "success":
                    raise ValueError("Successful WAVE-0 rollouts cannot be failure-recovery references")
                comparisons, replay_maxima = [], {}
                def verify(t):
                    a = {k: d[t] for k, d in reference["observations"].items()}
                    b = runner.row()
                    reference_actions = [] if not t else [reference["actions/clipped_actual_target"][t-1]]
                    actual_actions = [] if not t else [trace.file["actions/clipped_actual_target"][t-1]]
                    check = compare_replay([a], [b], reference_actions, actual_actions)
                    for key, value in check.get("metrics", {}).items():
                        if isinstance(value, (float, int)):
                            replay_maxima[key] = max(replay_maxima.get(key, 0), value)
                    if not check["passed"]:
                        atomic_json(args.output/"replay_check.json", dict(passed=False, step=t, **{k:v for k,v in check.items() if k!='passed'}))
                        raise RuntimeError(f"replay_mismatch_at_observation_{t}: {check['violations'][:3]}")
                    comparisons.append(t)
                verify(0)
                for t in range(args.takeover_step):
                    action = reference["actions/clipped_actual_target"][t]
                    runner.execute(action, source="replay", phase="baseline_prefix", decision_step=(t//10)*10)
                    verify(t+1)
                    if runner.success:
                        raise RuntimeError("replay_succeeded_before_takeover")
                check = dict(passed=True, observations_checked=len(comparisons), includes_relative_grasp=True,
                             full_prefix=True, policy_and_tactile_history_continuous=True, maximum_errors=replay_maxima)
                atomic_json(args.output/"replay_check.json", check)
                result["replay"] = check
            if args.kind == "recovery" and runner.detector.first_confirmed != args.takeover_step:
                raise RuntimeError("replay_did_not_reproduce_first_causal_blockage")
            from pi05_univtac.contact_recovery import RecoveryPlanError
            try:
                if args.kind == "oracle":
                    from .oracle_recovery import OracleTeacher
                    teacher = OracleTeacher(runner, json.loads(args.oracle_config.read_text()) if args.oracle_config else {})
                    # Keep the list reference, so failed branches retain every decision.
                    result["oracle"] = dict(config=teacher.config, events=teacher.events)
                    teacher.run()
                else:
                    result["teacher"] = runner.run_teacher(args.seed)
            except (TeacherStopped, RecoveryPlanError) as exc:
                result["teacher_stop_reason"] = str(exc)
        result.update(status="success" if runner.success else "failed", success=runner.success)
    except BaseException as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        if runner is not None:
            result.update(strategy_actions=runner.step, detector=runner.detector.report(),
                          action_metadata=runner.metadata, teacher_events=runner.teacher_events,
                          takeover_step=args.takeover_step, reference=str(args.reference) if args.reference else None,
                          policy_supervision_ranges=finalize_supervision(runner.metadata,
                              normal_success=runner.success and args.kind=="natural",
                              recovery_success=runner.success and args.kind=="recovery"),
                          context_range=[0, runner.step+1], wm_context_continuous=True)
        if trace is not None:
            if runner is not None and trace.file is not None and runner.metadata:
                trace.file["actions/policy_eligible"][:] = [r["policy_eligible"] for r in runner.metadata]
            trace.finish(task, result["status"], reason=result.get("error", ""))
            result["trajectory"] = str(trace.path)
        if policy is not None:
            policy.close()
        atomic_json(args.output / "result.json", result)
        if task is not None:
            task.close()
        app.close()
    print(json.dumps({k:v for k,v in result.items() if k in ("seed","kind","status","error","strategy_actions")}), flush=True)
    return 1 if result["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
