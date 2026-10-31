"""Geometry and execution-contract checks; no simulator or GPU required."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from contact_belief_wam.oracle_batch_teacher import (
    BatchOracleTeacher, ee_for_object_pose, relative_pose, rotation, with_rotation,
)
from contact_belief_wam.sim_takeover import TeacherStopped


def pose(position=(0., 0., 0.), pitch=0.):
    q = Rotation.from_euler("y", pitch, degrees=True).as_quat()
    return np.r_[position, q[[3, 0, 1, 2]]]


class Runner:
    """Rigid-grasp stand-in, with controllable contact and insertion stalls."""

    def __init__(self, task="insert_HDMI", pitch=0., offset_mm=2., step=40,
                 budget=None, success_depth_mm=2., stalled_inserts=0):
        target = pose((.6, 0., .1), pitch)
        obj = pose((.6 + offset_mm/1000, 0., .104), 0.)
        if task == "insert_HDMI":
            target = pose((.6, 0., .002))
            obj = pose((.6 + offset_mm/1000, 0., .004))
        ee = pose(obj[:3] + np.array([.01, -.005, .1]), 8.)
        self.state = dict(ee_pose_world=ee, prism_pose_world=obj,
                          slot_pose_world=pose(target[:3]), gripper_pose_world=ee.copy(),
                          joint_q=np.r_[np.zeros(7), .012])
        self.task = SimpleNamespace(target_pose=target, check_early_stop=lambda: False)
        self.cfg = SimpleNamespace(budget=budget or (600 if task == "insert_HDMI" else 317))
        self.step, self.success = step, False
        self.calls = []
        self.stalled_inserts = stalled_inserts
        self.success_depth_mm = success_depth_mm
        self.contact = .001
        self.contact_history = []

    def row(self):
        return {k: v.copy() for k, v in self.state.items()}

    def contact_distance(self):
        if self.contact_history:
            return self.contact_history.pop(0)
        return self.contact

    def move(self, target, **kwargs):
        assert kwargs["gripper"] is None
        assert kwargs["eligible"] is False
        assert kwargs["source"] == "oracle"
        self.calls.append(dict(target=target.copy(), **kwargs))
        before = self.state["ee_pose_world"]
        obj = self.state["prism_pose_world"]
        delta = rotation(target) * rotation(before).inv()
        new_obj = with_rotation(obj, delta * rotation(obj))
        new_obj[:3] = target[:3] + delta.apply(obj[:3] - before[:3])
        if kwargs["phase"] == "oracle_insert" and self.stalled_inserts:
            self.stalled_inserts -= 1
        else:
            self.state["prism_pose_world"] = new_obj
        self.state["ee_pose_world"] = target.copy()
        self.state["gripper_pose_world"] = target.copy()
        self.step += kwargs["duration"]
        if kwargs["phase"] == "oracle_insert":
            rel = relative_pose(self.task.target_pose, self.state["prism_pose_world"])
            self.success = bool(rel[2] * 1000 <= -self.success_depth_mm)


@pytest.mark.parametrize("pitch", [-30., 30.])
def test_object_rotation_preserves_actual_grasp_for_both_hole_tilts(pitch):
    runner = Runner("insert_hole", pitch=pitch)
    teacher = BatchOracleTeacher(runner, "insert_hole")
    before = runner.row()
    grasp = relative_pose(before["ee_pose_world"], before["prism_pose_world"])
    teacher.unload()
    unloaded = runner.row()["prism_pose_world"][:3].copy()
    teacher.orient()
    after = runner.row()
    assert np.allclose(after["prism_pose_world"][:3], unloaded, atol=1e-12)
    assert np.allclose(relative_pose(after["ee_pose_world"], after["prism_pose_world"]), grasp, atol=1e-12)
    assert (rotation(after["prism_pose_world"]) * rotation(teacher.target).inv()).magnitude() < 1e-10
    assert teacher.rotation_travel_deg == pytest.approx(30.)
    # Rotating a 100 mm lever legitimately moves the wrist farther than 5 mm.
    assert np.linalg.norm(after["ee_pose_world"][:2] - before["ee_pose_world"][:2]) > .04
    assert len([e for e in teacher.events if e["phase"] == "oracle_align_rotation"]) == 30
    assert all(e["actual_actions"] == 2 for e in teacher.events[1:])


@pytest.mark.parametrize("pitch", [-30., 30.])
def test_hole_correction_and_insertion_follow_target_frame_not_world_xy(pitch):
    runner = Runner("insert_hole", pitch=pitch, offset_mm=0.)
    obj = pose((.6, 0., .1), pitch)
    obj[:3] += rotation(runner.task.target_pose).apply([.001, .0005, .004])
    runner.state["prism_pose_world"] = obj
    runner.state["ee_pose_world"] = pose(obj[:3] + [0., 0., .1], pitch)
    runner.state["gripper_pose_world"] = runner.state["ee_pose_world"].copy()
    teacher = BatchOracleTeacher(runner, "insert_hole")
    teacher.align()
    rel = relative_pose(teacher.target, runner.row()["prism_pose_world"])
    assert np.linalg.norm(rel[:2]) <= .00015
    before = runner.row()["prism_pose_world"][:3]
    event = teacher.insert()
    displacement = runner.row()["prism_pose_world"][:3] - before
    assert np.allclose(displacement, rotation(teacher.target).apply([0, 0, -.0005]))
    assert abs(displacement[0]) == pytest.approx(.00025)
    assert event["object_progress_mm"] == pytest.approx(.5)
    assert event["actual_actions"] == 2


def test_hdmi_preserves_existing_translation_only_waypoints():
    runner = Runner(offset_mm=2.5, success_depth_mm=1.)
    initial_rotation = rotation(runner.row()["ee_pose_world"])
    teacher = BatchOracleTeacher(runner, "insert_HDMI")
    result = teacher.run()
    assert result["success"]
    assert teacher.events[0]["phase"] == "oracle_unload"
    assert teacher.events[0]["actual_actions"] == 20
    corrections = [e for e in teacher.events if e["phase"] == "oracle_align_xy"]
    assert len(corrections) == 5
    assert all(e["actual_actions"] == 10 for e in corrections)
    for call in runner.calls:
        assert (rotation(call["target"]) * initial_rotation.inv()).magnitude() < 1e-10
        assert np.linalg.norm(call["target"][:2] - teacher.origin[:2]) <= .003
    assert all(e["stop"]-e["step"] == e["actual_actions"] for e in teacher.events)
    assert all(e["decision_step"] == e["step"] for e in teacher.events)
    assert all(a["stop"] == b["step"] for a, b in zip(teacher.events, teacher.events[1:]))


def test_relative_grasp_transform_not_nominal_robot_or_object_orientation():
    ee = pose((.3, -.05, .2), 17.)
    obj = pose((.27, -.02, .12), -13.)
    desired = pose((.28, -.01, .125), 30.)
    target = ee_for_object_pose(ee, obj, desired)
    assert np.allclose(relative_pose(target, desired), relative_pose(ee, obj), atol=1e-12)


def test_budget_refuses_partial_waypoint_and_quarantines_failed_branch():
    runner = Runner(step=580)
    teacher = BatchOracleTeacher(runner, "insert_HDMI")
    with pytest.raises(TeacherStopped, match="budget"):
        teacher.run()
    assert runner.step == 600
    assert len(runner.calls) == 1
    assert not teacher.branch_success
    assert not any(e["bc_eligible"] for e in teacher.events)


def test_hdmi_alignment_saturates_at_three_mm_and_does_not_teleport():
    runner = Runner(offset_mm=20.)
    teacher = BatchOracleTeacher(runner, "insert_HDMI")
    teacher.align()
    assert np.linalg.norm(runner.row()["ee_pose_world"][:2] - teacher.origin[:2]) == pytest.approx(.002999)
    assert runner.row()["prism_pose_world"][0] > .615
    assert len(runner.calls) == 6


def test_hole_five_mm_cap_accumulates_across_correction_calls():
    runner = Runner("insert_hole", pitch=0., offset_mm=20.)
    teacher = BatchOracleTeacher(runner, "insert_hole")
    teacher.align()
    calls = len(runner.calls)
    teacher.align()
    assert np.linalg.norm(teacher.lateral_correction) == pytest.approx(5.)
    assert len(runner.calls) == calls == 10


def test_hole_rotation_cap_counts_whole_branch_and_requires_clearance():
    runner = Runner("insert_hole", pitch=36.)
    teacher = BatchOracleTeacher(runner, "insert_hole")
    with pytest.raises(TeacherStopped, match="cumulative_rotation_limit"):
        teacher.orient()
    assert teacher.rotation_travel_deg == pytest.approx(35.)
    assert runner.step == 110
    runner.contact = .0001
    with pytest.raises(TeacherStopped, match="requires_unloaded"):
        teacher.orient()


def test_hole_unload_retries_at_most_six_mm_before_any_rotation():
    runner = Runner("insert_hole", pitch=30.)
    runner.contact = .0001
    teacher = BatchOracleTeacher(runner, "insert_hole")
    with pytest.raises(TeacherStopped, match="unload_limit"):
        teacher.run()
    assert len(runner.calls) == 3
    assert all(c["phase"] == "oracle_unload" for c in runner.calls)
    assert all(c["duration"] == 10 for c in runner.calls)
    assert not any(e["bc_eligible"] for e in teacher.events)


def test_hole_contact_backend_is_required_and_invalid_distance_is_rejected():
    runner = Runner("insert_hole")
    runner.contact_distance = None
    with pytest.raises(ValueError, match="contact_distance"):
        BatchOracleTeacher(runner, "insert_hole")
    runner.contact_distance = lambda: float("nan")
    teacher = BatchOracleTeacher(runner, "insert_hole")
    with pytest.raises(ValueError, match="contact distance"):
        teacher.unload()


def test_stalled_insert_windows_are_excluded_even_when_branch_recovers():
    runner = Runner(offset_mm=0., success_depth_mm=0., stalled_inserts=2)
    teacher = BatchOracleTeacher(runner, "insert_HDMI")
    teacher.run()
    failed = [e for e in teacher.events if e["phase"] == "oracle_insert" and e["object_progress_mm"] < .04]
    assert len(failed) == 2
    assert not any(e["bc_eligible"] for e in failed)
    assert failed[-1]["feedback_reason"] == "unload_next"
    assert len([e for e in teacher.events if e["phase"] == "oracle_unload"]) == 2
    assert any(e["bc_eligible"] for e in teacher.events if e["phase"] == "oracle_insert")


def test_native_early_stop_prevents_robot_motion():
    runner = Runner()
    runner.task.check_early_stop = lambda: True
    teacher = BatchOracleTeacher(runner, "insert_HDMI")
    with pytest.raises(TeacherStopped, match="native_early_stop"):
        teacher.run()
    assert runner.calls == []


@pytest.mark.parametrize("pitch", [-30., 30.])
def test_hole_full_teacher_reaches_native_success_within_total_budget(pitch):
    runner = Runner("insert_hole", pitch=pitch)
    teacher = BatchOracleTeacher(runner, "insert_hole")
    teacher.run()
    assert teacher.branch_success
    assert runner.success and runner.step <= 317
    assert sum(event["actual_actions"] for event in teacher.events) == runner.step - 40
    assert all(event["bc_eligible"] for event in teacher.events)


def test_three_cycles_are_the_limit_even_when_more_budget_remains():
    runner = Runner(offset_mm=0., stalled_inserts=100)
    teacher = BatchOracleTeacher(runner, "insert_HDMI")
    with pytest.raises(TeacherStopped, match="bounded_recovery_did_not_succeed"):
        teacher.run()
    assert len([e for e in teacher.events if e["phase"] == "oracle_unload"]) == 3
    assert len([e for e in teacher.events if e["phase"] == "oracle_insert"]) == 6
    assert runner.step < runner.cfg.budget
    assert not any(e["bc_eligible"] for e in teacher.events)


def test_hole_budget_includes_already_executed_natural_prefix():
    runner = Runner("insert_hole", step=310)
    teacher = BatchOracleTeacher(runner, "insert_hole")
    with pytest.raises(TeacherStopped, match="budget"):
        teacher.run()
    assert runner.step == 310 and runner.calls == []


def test_ik_failure_keeps_interval_and_excludes_all_branch_labels():
    runner = Runner()
    teacher = BatchOracleTeacher(runner, "insert_HDMI")
    def fail(*args, **kwargs):
        runner.step += 1
        raise RuntimeError("IK failure")
    runner.move = fail
    with pytest.raises(RuntimeError, match="IK failure"):
        teacher.run()
    assert teacher.events[0]["actual_actions"] == 1
    assert not teacher.events[0]["bc_eligible"]
    assert teacher.events[0]["rejection_reason"] == "movement_failed_or_interrupted"
