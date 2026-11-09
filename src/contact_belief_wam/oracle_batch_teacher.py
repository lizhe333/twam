"""Bounded privileged recovery teachers for the two-task oracle campaign.

The controller reads simulator poses, but moves only the robot through Runner's
ordinary IK/control interface. Commands are recorded as ineligible until the
whole branch succeeds; the resulting events supply candidate BC intervals.
All poses use world coordinates and [x, y, z, qw, qx, qy, qz].
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .sim_takeover import TeacherStopped


COMMON = dict(retreat_mm=2., translation_mm=.5, max_cycles=3,
              min_progress_mm=.04, xy_tolerance_mm=.15,
              max_grasp_translation_mm=40.)
DEFAULTS = {
    "insert_HDMI": dict(COMMON, lateral_cap_mm=3., retreat_actions=20,
                        lateral_actions=10, insert_actions=10,
                        max_insert_steps=20, rotation_cap_deg=3.,
                        rotation_step_deg=1., rotation_actions=2,
                        rotation_tolerance_deg=.2, unload_cap_mm=2.,
                        clearance_mm=.5),
    "insert_hole": dict(COMMON, lateral_cap_mm=5., retreat_actions=10,
                        lateral_actions=5, insert_actions=2,
                        max_insert_steps=200, rotation_cap_deg=35.,
                        rotation_step_deg=1., rotation_actions=2,
                        rotation_tolerance_deg=.2, unload_cap_mm=6.,
                        clearance_mm=.5),
}


def pose_array(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Expected a finite world pose with 7 components")
    if abs(np.linalg.norm(pose[3:]) - 1.) > 1e-3:
        raise ValueError("Expected a unit world-pose quaternion")
    return pose.copy()


def rotation(pose):
    return Rotation.from_quat(np.asarray(pose)[[4, 5, 6, 3]])


def with_rotation(pose, orientation):
    result = pose_array(pose)
    q = orientation.as_quat()
    result[3:] = q[[3, 0, 1, 2]]
    return result


def ee_for_object_pose(ee_pose, object_pose, desired_object_pose):
    """Apply O_new * inverse(O_now) to E_now, preserving the measured grasp."""
    ee, obj, desired = map(pose_array, (ee_pose, object_pose, desired_object_pose))
    delta = rotation(desired) * rotation(obj).inv()
    result = with_rotation(ee, delta * rotation(ee))
    result[:3] = desired[:3] + delta.apply(ee[:3] - obj[:3])
    return result


def relative_pose(reference, pose):
    reference, pose = map(pose_array, (reference, pose))
    out = with_rotation(pose, rotation(reference).inv() * rotation(pose))
    out[:3] = rotation(reference).inv().apply(pose[:3] - reference[:3])
    return out


class BatchOracleTeacher:
    """Ground-truth-assisted robot controller; raw actions remain quarantined.

    ``events`` contains exclusive [step, stop) action intervals. A successful
    worker may promote only intervals whose ``bc_eligible`` is true. Failed
    branches clear that flag globally, including previously useful movements.
    """

    def __init__(self, runner, task_name, config=None):
        if task_name not in DEFAULTS:
            raise ValueError(f"Unsupported oracle recovery task: {task_name}")
        if set(config or {}) - DEFAULTS[task_name].keys():
            raise ValueError("Unknown batch oracle configuration key")
        self.config = dict(DEFAULTS[task_name], **(config or {}))
        for key, value in self.config.items():
            if not np.isfinite(value) or value < 0 or (value == 0 and key != "rotation_cap_deg"):
                raise ValueError(f"Invalid oracle configuration: {key}")
            if key.endswith("actions") or key in ("max_cycles", "max_insert_steps"):
                if int(value) != value:
                    raise ValueError(f"Expected integer action/count setting: {key}")
        self.runner, self.task_name = runner, task_name
        self.events = []
        self.origin = pose_array(runner.row()["ee_pose_world"])
        self.object_origin = pose_array(runner.row()["prism_pose_world"])
        self.grasp_origin = self._grasp(runner.row())
        self.lateral_correction = np.zeros(2, dtype=float)
        self.rotation_travel_deg = 0.
        self.branch_success = False
        if task_name == "insert_hole":
            # Native target_pose is an actor/world pose, not robot-root-relative.
            self.target = pose_array(runner.task.target_pose)
            if not callable(getattr(runner, "contact_distance", None)):
                raise ValueError("insert_hole requires native contact_distance() in meters")
        else:
            self.target = pose_array(runner.row()["slot_pose_world"])

    def _grasp(self, row):
        frame = row.get("gripper_pose_world", row["ee_pose_world"])
        return relative_pose(frame, row["prism_pose_world"])

    def metrics(self, row=None):
        row = self.runner.row() if row is None else row
        obj = pose_array(row["prism_pose_world"])
        ref = self.target if self.task_name == "insert_hole" else pose_array(row["slot_pose_world"])
        rel = relative_pose(ref, obj)
        return dict(object_in_target_mm=(rel[:3] * 1000).tolist(),
                    object_in_target_rotvec_deg=np.rad2deg(rotation(rel).as_rotvec()).tolist(),
                    ee_pose_world=pose_array(row["ee_pose_world"]).tolist(),
                    object_in_gripper_pose=self._grasp(row).tolist())

    def _check_state(self):
        row = self.runner.row()
        grasp = self._grasp(row)
        slip = np.linalg.norm(grasp[:3] - self.grasp_origin[:3]) * 1000
        if slip > self.config["max_grasp_translation_mm"]:
            raise TeacherStopped("oracle_object_dropped_or_grasp_lost")
        native_stop = getattr(self.runner.task, "check_early_stop", None)
        if callable(native_stop) and native_stop():
            raise TeacherStopped("oracle_native_early_stop")

    def _contact_distance(self):
        value = float(self.runner.contact_distance())
        if not np.isfinite(value) or value < 0:
            raise ValueError("Native contact distance must be finite and nonnegative meters")
        return value

    def _move(self, target, phase, reason, duration, *, movement_axis=None,
              desired_object=None, lateral_correction=None, rotation_delta_deg=0.):
        runner, cfg = self.runner, self.config
        if runner.success:
            return None
        self._check_state()
        target = pose_array(target)
        duration = int(duration)
        if runner.step + duration > runner.cfg.budget:
            raise TeacherStopped("oracle_strategy_action_budget_exhausted")
        if self.task_name == "insert_HDMI":
            offset = np.linalg.norm(target[:2] - self.origin[:2]) * 1000
            angle = np.rad2deg((rotation(target) * rotation(self.origin).inv()).magnitude())
            if offset > cfg["lateral_cap_mm"] + 1e-4 or angle > cfg["rotation_cap_deg"] + 1e-4:
                raise TeacherStopped("oracle_local_workspace_limit")
        else:
            if lateral_correction is not None and np.linalg.norm(lateral_correction) > cfg["lateral_cap_mm"] + 1e-6:
                raise TeacherStopped("oracle_local_workspace_limit")
            if self.rotation_travel_deg + rotation_delta_deg > cfg["rotation_cap_deg"] + 1e-6:
                raise TeacherStopped("oracle_cumulative_rotation_limit")
        before_row = {k: np.asarray(v).copy() for k, v in runner.row().items()}
        start = int(runner.step)
        event = dict(step=start, stop=start, decision_step=start, actual_actions=0,
                     phase=phase, reason=reason, before=self.metrics(before_row),
                     after=None, privileged_feedback=True, demonstrator="privileged_oracle",
                     requested_actions=duration, target_ee_pose_world=target.tolist(),
                     bc_eligible=False, candidate_bc_eligible=False)
        self.events.append(event)
        completed = False
        try:
            # Eligibility is finalized only after true object feedback and native
            # terminal success. Never alter the gripper or use simulator resets.
            runner.move(target, phase=phase, eligible=False, duration=duration,
                        gripper=None, source="oracle")
            completed = True
        finally:
            event.update(stop=int(runner.step), actual_actions=int(runner.step)-start,
                         after=self.metrics(), success=bool(runner.success))
            if not completed:
                event["rejection_reason"] = "movement_failed_or_interrupted"
        self._check_state()
        if event["actual_actions"] <= 0:
            raise TeacherStopped("oracle_no_actions_executed")
        if event["actual_actions"] != duration and not runner.success:
            raise TeacherStopped("oracle_incomplete_waypoint")
        after_row = runner.row()
        actual_delta = np.asarray(after_row["prism_pose_world"][:3]) - before_row["prism_pose_world"][:3]
        valid = True
        if movement_axis is not None:
            progress = float(np.dot(actual_delta, movement_axis) * 1000)
            event["object_progress_mm"] = progress
            valid = progress >= cfg["min_progress_mm"]
            if not valid:
                event["rejection_reason"] = "insufficient_actual_object_progress"
        elif desired_object is not None:
            desired_object = pose_array(desired_object)
            if phase == "oracle_align_rotation":
                prior = (rotation(desired_object) * rotation(before_row["prism_pose_world"]).inv()).magnitude()
                after = (rotation(desired_object) * rotation(after_row["prism_pose_world"]).inv()).magnitude()
                actual_rotation = np.rad2deg((rotation(after_row["prism_pose_world"]) * rotation(before_row["prism_pose_world"]).inv()).magnitude())
                event["rotation_progress_deg"] = float(np.rad2deg(prior-after))
                valid = event["rotation_progress_deg"] > .01
                # Charge commanded travel even if actual motion is smaller.
                self.rotation_travel_deg += rotation_delta_deg
                event["actual_rotation_deg"] = float(actual_rotation)
                event["cumulative_rotation_deg"] = self.rotation_travel_deg
            else:
                prior = np.linalg.norm(desired_object[:3] - before_row["prism_pose_world"][:3])
                after = np.linalg.norm(desired_object[:3] - after_row["prism_pose_world"][:3])
                event["correction_progress_mm"] = float((prior-after)*1000)
                valid = event["correction_progress_mm"] >= .02
            if not valid:
                event["rejection_reason"] = "correction_did_not_improve_object_pose"
        if lateral_correction is not None:
            self.lateral_correction = np.asarray(lateral_correction).copy()
        event["candidate_bc_eligible"] = bool(valid)
        event["bc_eligible"] = bool(valid)
        return event

    def unload(self):
        cfg = self.config
        total = 0.
        while total + cfg["retreat_mm"] <= cfg["unload_cap_mm"] + 1e-8:
            row = self.runner.row()
            obj = pose_array(row["prism_pose_world"])
            reference = obj if self.task_name == "insert_hole" else pose_array(row["slot_pose_world"])
            axis = rotation(reference).apply([0, 0, 1])
            desired = obj.copy()
            desired[:3] += axis * cfg["retreat_mm"] / 1000
            target = ee_for_object_pose(row["ee_pose_world"], obj, desired)
            event = self._move(target, "oracle_unload", "unload_along_current_tube_axis" if self.task_name == "insert_hole" else "unload_in_slot_axis",
                               cfg["retreat_actions"], movement_axis=axis)
            total += cfg["retreat_mm"]
            if self.runner.success:
                return
            if not event["candidate_bc_eligible"]:
                raise TeacherStopped("oracle_unload_made_no_object_progress")
            if self.task_name == "insert_HDMI":
                return
            distance = self._contact_distance()
            event["contact_distance_m"] = distance
            event["cycle_unload_mm"] = total
            if distance >= cfg["clearance_mm"] / 1000:
                return
        raise TeacherStopped("oracle_cannot_clear_contact_within_unload_limit")

    def orient(self):
        if self.task_name != "insert_hole":
            return
        cfg = self.config
        while not self.runner.success:
            if self._contact_distance() < cfg["clearance_mm"] / 1000:
                raise TeacherStopped("oracle_rotation_requires_unloaded_contact")
            row = self.runner.row()
            obj = pose_array(row["prism_pose_world"])
            correction = (rotation(self.target) * rotation(obj).inv()).as_rotvec()
            error_deg = float(np.rad2deg(np.linalg.norm(correction)))
            if error_deg <= cfg["rotation_tolerance_deg"]:
                return
            angle = min(error_deg, cfg["rotation_step_deg"])
            delta = Rotation.from_rotvec(correction * (angle / error_deg))
            desired = with_rotation(obj, delta * rotation(obj))
            target = ee_for_object_pose(row["ee_pose_world"], obj, desired)
            event = self._move(target, "oracle_align_rotation", "current_true_object_target_orientation",
                               cfg["rotation_actions"], desired_object=desired,
                               rotation_delta_deg=angle)
            if event is not None and not event["candidate_bc_eligible"]:
                raise TeacherStopped("oracle_rotation_did_not_improve_orientation")

    def align(self):
        cfg = self.config
        # Finite even with a slipping object; the branch's cap is never reset.
        limit = 12 if self.task_name == "insert_HDMI" else int(np.ceil(cfg["lateral_cap_mm"] / cfg["translation_mm"])) + 2
        for _ in range(limit):
            if self.runner.success:
                return
            row = self.runner.row()
            obj = pose_array(row["prism_pose_world"])
            ref = self.target if self.task_name == "insert_hole" else pose_array(row["slot_pose_world"])
            relative = relative_pose(ref, obj)
            delta = -relative[:2] * 1000
            if np.linalg.norm(delta) <= cfg["xy_tolerance_mm"]:
                return
            delta *= min(1., cfg["translation_mm"] / np.linalg.norm(delta))
            desired = obj.copy()
            desired[:3] += rotation(ref).apply(np.r_[delta / 1000, 0])
            target = ee_for_object_pose(row["ee_pose_world"], obj, desired)
            limited = False
            cumulative = None
            if self.task_name == "insert_HDMI":
                offset = target[:2] - self.origin[:2]
                cap = cfg["lateral_cap_mm"] / 1000 - 1e-6
                if np.linalg.norm(offset) > cap:
                    limited = True
                    target[:2] = self.origin[:2] + offset * cap / np.linalg.norm(offset)
                    desired[:3] = obj[:3] + target[:3] - np.asarray(row["ee_pose_world"][:3])
            else:
                cumulative = self.lateral_correction + delta
                if np.linalg.norm(cumulative) > cfg["lateral_cap_mm"]:
                    limited = True
                    cumulative *= cfg["lateral_cap_mm"] / np.linalg.norm(cumulative)
                    delta = cumulative - self.lateral_correction
                    desired[:3] = obj[:3] + rotation(ref).apply(np.r_[delta / 1000, 0])
                    target = ee_for_object_pose(row["ee_pose_world"], obj, desired)
            if np.linalg.norm(target[:3] - row["ee_pose_world"][:3]) < .00002:
                return
            self._move(target, "oracle_align_xy", "current_true_object_target_lateral_error",
                       cfg["lateral_actions"], desired_object=desired,
                       lateral_correction=cumulative)
            if limited:
                return

    def insert(self):
        row = self.runner.row()
        obj = pose_array(row["prism_pose_world"])
        ref = self.target if self.task_name == "insert_hole" else pose_array(row["slot_pose_world"])
        axis = rotation(ref).apply([0, 0, -1])
        desired = obj.copy()
        desired[:3] += axis * self.config["translation_mm"] / 1000
        target = ee_for_object_pose(row["ee_pose_world"], obj, desired)
        return self._move(target, "oracle_insert", "test_actual_object_progress_along_target_axis",
                          self.config["insert_actions"], movement_axis=axis)

    def run(self):
        try:
            for cycle in range(int(self.config["max_cycles"])):
                if self.runner.success:
                    break
                self.unload()
                self.orient()
                self.align()
                blocked = 0
                for _ in range(int(self.config["max_insert_steps"])):
                    if self.runner.success:
                        break
                    event = self.insert()
                    blocked = blocked + 1 if event["object_progress_mm"] < self.config["min_progress_mm"] else 0
                    event.update(cycle=cycle, consecutive_stalls=blocked,
                                 feedback_reason="unload_next" if blocked >= 2 else "continue_insertion")
                    if blocked >= 2:
                        break
                if self.runner.success:
                    break
            if not self.runner.success:
                raise TeacherStopped("oracle_bounded_recovery_did_not_succeed")
            self.branch_success = True
        except BaseException:
            for event in self.events:
                event["bc_eligible"] = False
                event["branch_rejection_reason"] = "recovery_branch_did_not_succeed"
            raise
        return dict(success=True, strategy_actions=int(self.runner.step), events=self.events)
