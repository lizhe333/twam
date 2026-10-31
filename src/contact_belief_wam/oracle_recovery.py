"""Explicitly privileged, bounded HDMI recovery diagnostic; no policy training.

All feedback uses the current real simulator state. No object teleportation,
teacher resets, trajectory splicing, or execution-budget bypass is allowed.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .paths import atomic_json
from .sim_takeover import TeacherStopped


DEFAULTS = dict(retreat_mm=2., translation_mm=.5, lateral_cap_mm=3.,
                rotation_cap_deg=3., align_rotation=False, max_cycles=3,
                waypoint_actions=10, retreat_actions=20, max_insert_steps=20,
                min_progress_mm=.04, xy_tolerance_mm=.15)


def physical_equivalence(strict, expected, actual):
    """Separate oracle-only physical audit; never changes strict replay status.

    Rendering/depth buffers are not inputs to this privileged controller. Keep
    every pose, joint, grasp, marker and action limit, and additionally check the
    actual native object surfaces. This does not certify observation equivalence.
    """
    excluded = tuple(f"{side}_{kind}." for side, kinds in (
        ("head", ("rgb",)), ("wrist", ("rgb",)),
        ("left", ("rgb_marker", "depth")), ("right", ("rgb_marker", "depth"))) for kind in kinds)
    violations = [v for v in strict["violations"] if not v.startswith(excluded)]
    surface_errors = {}
    for key in ("prism_vertices_world", "slot_vertices_world"):
        maximum = 0.
        for a, b in zip(expected, actual):
            if key not in a or key not in b:
                violations.append(f"Missing physical surface: {key}")
                break
            x, y = np.asarray(a[key]), np.asarray(b[key])
            if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
                violations.append(f"Invalid physical surface: {key}")
                break
            maximum = max(maximum, float(np.linalg.norm(x-y, axis=-1).max()))
        surface_errors[key] = maximum
        if maximum > strict["tolerances"]["position_m"]:
            violations.append(f"Physical surface mismatch: {key}={maximum}")
    return dict(passed=not violations, violations=violations, surface_max_error_m=surface_errors,
        strict_observation_equivalence_passed=strict["passed"],
        scope="privileged_physical_diagnostic_only", certifies_observation_equivalence=False,
        excluded_render_metrics=list(excluded))


def rotation(pose):
    return Rotation.from_quat(np.asarray(pose)[[4, 5, 6, 3]])


def state_metrics(row):
    object_pose, slot = row["prism_pose_world"], row["slot_pose_world"]
    relative = rotation(slot).inv().apply(object_pose[:3] - slot[:3])
    return dict(object_in_slot_mm=(relative * 1000).tolist(),
                object_in_slot_rotvec_deg=np.rad2deg((rotation(slot).inv() * rotation(object_pose)).as_rotvec()).tolist(),
                ee_pose_world=row["ee_pose_world"].tolist(), joint_q=row["joint_q"].tolist(),
                object_in_gripper_pose=row["object_in_gripper_pose_world"].tolist())


class OracleTeacher:
    def __init__(self, runner, config=None):
        if set(config or {}) - DEFAULTS.keys():
            raise ValueError("Unknown oracle configuration key")
        self.config = dict(DEFAULTS, **(config or {}))
        self.runner = runner
        self.events = []
        self.origin = runner.row()["ee_pose_world"].copy()

    def record(self, phase, reason, before, **extra):
        event = dict(step=self.runner.step, phase=phase, reason=reason,
                     privileged_feedback=True, before=before, **extra)
        self.events.append(event)
        atomic_json(self.runner.trace.root.parent / "oracle_progress.json", dict(
            strategy_actions=self.runner.step, success=self.runner.success, events=self.events))
        print(f"ORACLE step={self.runner.step} phase={phase} reason={reason}", flush=True)
        return event

    def move(self, target, phase, reason, *, duration=None, gripper=None):
        runner, cfg = self.runner, self.config
        if runner.success:
            return
        count = int(duration or cfg["waypoint_actions"])
        if runner.step + count > runner.cfg.budget:
            raise TeacherStopped("oracle_strategy_action_budget_exhausted")
        lateral = np.linalg.norm(target[:2] - self.origin[:2]) * 1000
        angle = np.rad2deg((rotation(target) * rotation(self.origin).inv()).magnitude())
        if lateral > cfg["lateral_cap_mm"] + 1e-4 or angle > cfg["rotation_cap_deg"] + 1e-4:
            raise TeacherStopped("oracle_local_workspace_limit")
        event = self.record(phase, reason, state_metrics(runner.row()), target=target.tolist(),
                            requested_actions=count)
        start = runner.step
        runner.move(target, phase=phase, eligible=False, duration=count, gripper=gripper, source="oracle")
        event.update(stop=runner.step, actual_actions=runner.step-start, after=state_metrics(runner.row()), success=runner.success)
        atomic_json(runner.trace.root.parent / "oracle_progress.json", dict(
            strategy_actions=runner.step, success=runner.success, events=self.events))

    def unload(self):
        row = self.runner.row()
        target = row["ee_pose_world"].copy()
        target[:3] += rotation(row["slot_pose_world"]).apply([0, 0, self.config["retreat_mm"] / 1000])
        self.move(target, "oracle_unload", "real_object_stall_unload_before_correction",
                  duration=self.config["retreat_actions"])

    def align(self):
        cfg = self.config
        for _ in range(12):
            row = self.runner.row()
            before = state_metrics(row)
            delta = -np.asarray(before["object_in_slot_mm"][:2])
            if np.linalg.norm(delta) <= cfg["xy_tolerance_mm"]:
                return
            delta *= min(1., cfg["translation_mm"] / np.linalg.norm(delta))
            target = row["ee_pose_world"].copy()
            target[:3] += rotation(row["slot_pose_world"]).apply(np.r_[delta / 1000, 0])
            offset = target[:2] - self.origin[:2]
            cap = cfg["lateral_cap_mm"] / 1000 - 1e-6
            limited = np.linalg.norm(offset) > cap
            if limited:
                target[:2] = self.origin[:2] + offset * cap / np.linalg.norm(offset)
            if np.linalg.norm(target[:3] - row["ee_pose_world"][:3]) < .00002:
                self.record("oracle_alignment_limit", "retain_bounded_residual_and_test_insertion", before)
                return
            self.move(target, "oracle_align_xy", "current_true_object_slot_lateral_error")
            if self.runner.success or limited:
                return

    def orient(self):
        """Rotate about the held object's current center, using the measured grasp."""
        row = self.runner.row()
        ee, obj, slot = (row[k] for k in ("ee_pose_world", "prism_pose_world", "slot_pose_world"))
        correction = (rotation(slot) * rotation(obj).inv()).as_rotvec()
        norm = np.linalg.norm(correction)
        if norm < np.deg2rad(.2):
            return
        correction *= min(1., np.deg2rad(1.) / norm)
        delta = Rotation.from_rotvec(correction)
        target = ee.copy()
        target[:3] = obj[:3] + delta.apply(ee[:3] - obj[:3])
        quat = (delta * rotation(ee)).as_quat()
        target[3:] = np.r_[quat[3], quat[:3]]
        self.move(target, "oracle_align_rotation", "current_true_object_slot_rotation_error")

    def run(self):
        runner, cfg = self.runner, self.config
        self.record("oracle_takeover", "offline_selected_early_natural_failure", state_metrics(runner.row()))
        for cycle in range(cfg["max_cycles"]):
            self.unload()
            if cfg["align_rotation"]:
                self.orient()
            self.align()
            blocked = 0
            for _ in range(cfg["max_insert_steps"]):
                if runner.success:
                    return
                row = runner.row()
                previous = state_metrics(row)
                target = row["ee_pose_world"].copy()
                target[:3] += rotation(row["slot_pose_world"]).apply([0, 0, -cfg["translation_mm"] / 1000])
                self.move(target, "oracle_insert", "test_true_object_progress_after_alignment")
                current = state_metrics(runner.row())
                progress = previous["object_in_slot_mm"][2] - current["object_in_slot_mm"][2]
                blocked = blocked + 1 if progress < cfg["min_progress_mm"] else 0
                self.record("oracle_feedback", "unload_next" if blocked >= 2 else "continue_insertion",
                            current, cycle=cycle, object_progress_mm=progress, consecutive_stalls=blocked)
                if runner.success:
                    return
                if blocked >= 2:
                    break
            if runner.success:
                return
        raise TeacherStopped("oracle_bounded_recovery_did_not_succeed")
