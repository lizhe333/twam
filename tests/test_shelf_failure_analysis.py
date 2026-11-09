"""Tests for geometric boundaries and the actual action-replay state machine."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "shelf_diagnostics", Path(__file__).parents[1] / "scripts/analyze_shelf_failures.py")
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


@pytest.mark.parametrize("axis,limit", list(enumerate([.02, .1, .02])))
def test_position_limits_are_strict_in_both_directions(axis, limit):
    for sign in [-1, 1]:
        p = np.zeros(3)
        p[axis] = sign * limit
        assert not diagnostic.relative_metrics(p, 1)["success"]
        p[axis] = sign * np.nextafter(limit, 0)
        assert diagnostic.relative_metrics(p, 1)["success"]


def test_tilt_uses_exact_cosine_not_rounded_fifteen_degrees():
    assert not diagnostic.relative_metrics([0, 0, 0], .965)["success"]
    assert diagnostic.relative_metrics([0, 0, 0], np.nextafter(.965, 1))["success"]
    assert diagnostic.relative_metrics([0, 0, 0], math.cos(math.radians(15.1)))["success"]


def test_rotated_translated_target_is_used_as_coordinate_frame():
    target = np.eye(4)
    target[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    target[:3, 3] = [1, 2, 3]
    relative = np.eye(4)
    relative[:3, 3] = [.019, .099, .019]
    bottle = target @ relative
    result = diagnostic.matrix_metrics(bottle, target)
    np.testing.assert_allclose(result["delta_m"], relative[:3, 3])
    assert result["success"]


def trace(points, native=None):
    metrics = [diagnostic.relative_metrics(p, 1) for p in points]
    return dict(kind=np.array(["check"] * len(points)),
                physical_step=np.arange(len(points)), action_index=np.arange(len(points)),
                substep=np.zeros(len(points)),
                delta_m=np.asarray(points), score=np.array([m["score"] for m in metrics]),
                passes=np.array([m["passes"] for m in metrics]),
                native_success=np.array([m["success"] for m in metrics] if native is None else native),
                tilt_deg=np.zeros(len(points)), dot=np.ones(len(points)),
                bottle_pose=np.column_stack([points, np.ones((len(points), 4))]))


def test_closest_state_never_combines_minima_from_different_steps():
    t = trace([[0, .2, 0], [.04, 0, 0]])
    result = diagnostic.summarize_trace(t)
    assert not result["ever_success"]
    assert result["best"]["physical_step"] == 0  # Same score: earliest wins.
    assert result["best"]["delta_mm"] == [0, 200, 0]


def test_any_condition_change_between_replays_is_unstable():
    a = trace([[.019, .2, 0]])
    b = trace([[.021, .2, 0]])
    comparison = diagnostic.compare_runs(a, b)
    assert not comparison["outcome_changes"]
    assert comparison["unstable"]
    assert comparison["max_relative_position_difference_mm"] == pytest.approx(2)


class FakeTask:
    def __init__(self, success_call=None):
        self.step_count = 100
        self.take_action_cnt = 0
        self.plan_success = True
        self.eval_success = False
        self.success_call = success_call
        self.events = []

    def check_success(self):
        return self.take_action_cnt == self.success_call

    def take_action(self, value, **kwargs):
        self.step_count += 1
        self.take_action_cnt += 1
        self.events.append("physical")
        self.eval_success = self.check_success()
        return True, self.eval_success

    def _get_observations(self):
        self.events.append("observe")
        return {"embodiment": {"joint": np.zeros(8)}}

    def check_early_stop(self):
        self.events.append("early_stop")
        return False


def episode(count=2, success=False, rejected=False):
    calls = count * 2 - int(success)
    row = dict(result="success" if success else "failed", cost_step=100 + calls,
               policy_execution=dict(physical_action_calls=calls, physics_call_limit=1230,
                                     physical_steps_per_action=2),
               tactile_feedback=dict(actions=[dict(step=i, physical_step=100 + min((i + 1) * 2, calls),
                                                   action=[0.] * 8, measured_joint=[0.] * 8)
                                               for i in range(count)]))
    if rejected:
        row.update(failure_kind="invalid_policy_output", failure_reason="outside limits")
    return row


def test_success_on_first_physical_call_stops_without_second_call():
    task = FakeTask(success_call=1)
    samples = []
    checks, reason = diagnostic.execute_saved_actions(task, episode(1, success=True), lambda x: x,
        lambda *args: samples.append(args), task.check_success)
    assert reason == "success"
    assert task.take_action_cnt == 1
    assert task.events == ["observe", "physical", "observe"]
    assert checks[0]["physical_step_matches"]
    assert samples[-1] == (0, 0, True, "check")


def test_rejected_policy_output_does_not_invent_an_action():
    task = FakeTask()
    row = episode(rejected=True)
    checks, reason = diagnostic.execute_saved_actions(task, row, lambda x: x,
                                                      lambda *args: None, task.check_success)
    assert reason == "invalid_policy_output"
    assert len(checks) == 2 and task.take_action_cnt == 4
    assert task.events == ["observe", "physical", "physical", "observe", "early_stop"] * 2 + ["observe"]
    assert diagnostic.validate_episode("1000047", row)["errors"] == []


def test_audit_rejects_missing_measured_joint_or_physical_step_gap():
    row = episode()
    row["tactile_feedback"]["actions"][0]["physical_step"] += 1
    del row["tactile_feedback"]["actions"][1]["measured_joint"]
    errors = diagnostic.validate_episode("1000003", row)["errors"]
    assert any("physical advance" in error for error in errors)
    assert any("measured_joint" in error for error in errors)


def test_predicate_mismatch_is_visible():
    result = diagnostic.summarize_trace(trace([[0, 0, 0]], native=[False]))
    assert not result["native_predicate_matches"]


def test_report_leaves_missing_object_metrics_empty(tmp_path):
    original = dict(seed=1000003, original_result="failed", termination="action_budget_exhausted", failure_reason=None)
    diagnostic.write_json(tmp_path / "audit.json", dict(episodes=[original], counts={"failed": 1}, valid=True))
    summary = diagnostic.report(tmp_path / "original", tmp_path)
    assert summary["replay_consistent_episodes"] == 0
    document = (tmp_path / "report.html").read_text()
    assert "尚无回放位姿" in document
    assert "pending_replay" in (tmp_path / "episodes.csv").read_text()


def test_zero_exit_code_cannot_hide_incomplete_or_failed_simulation():
    with pytest.raises(RuntimeError, match="Incomplete"):
        diagnostic.require_completed_run(dict(state="failed", episodes=[], error="reset timeout"))
    rows = [dict(seed=seed, error=None, consistent=False) for seed in range(1000000, 1000050)]
    diagnostic.require_completed_run(dict(state="completed", episodes=rows))
    with pytest.raises(RuntimeError, match="Incomplete"):
        diagnostic.require_completed_run(dict(state="completed", episodes=rows[:-1]))
