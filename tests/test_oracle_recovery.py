"""Oracle diagnostics remain bounded, explicit, and excluded from BC."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from contact_belief_wam.oracle_recovery import OracleTeacher, state_metrics
from contact_belief_wam.sim_takeover import TeacherStopped
from contact_belief_wam.takeover import PilotLedger, TakeoverConfig


def row():
    return dict(ee_pose_world=np.array([.5, 0, .15, 1., 0, 0, 0]),
                prism_pose_world=np.array([.501, .002, .013, 1., 0, 0, 0]),
                slot_pose_world=np.array([.5, 0, 0, 1., 0, 0, 0]),
                object_in_gripper_pose_world=np.array([0., 0, -.13, 1, 0, 0, 0]),
                joint_q=np.zeros(8))


def test_oracle_geometry_is_explicitly_in_slot_frame():
    value = row()
    np.testing.assert_allclose(state_metrics(value)["object_in_slot_mm"], [1, 2, 13])
    value["slot_pose_world"][3:] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    np.testing.assert_allclose(state_metrics(value)["object_in_slot_mm"], [2, -1, 13], atol=1e-10)


def test_oracle_retries_share_quota_and_success_stops_collection(tmp_path):
    cfg = replace(TakeoverConfig(), recovery_limit=2)
    ledger = PilotLedger(tmp_path, cfg, {})
    natural = ledger.reserve("natural", 3000001)
    ledger.finish(natural, dict(status="failed"))
    ledger.reserve_oracle(3000001)
    ledger.close()
    ledger = PilotLedger(tmp_path, cfg, {})
    assert ledger.counts()["recovery"] == 1
    assert ledger.state["attempts"][-1]["status"] == "interrupted"
    attempt = ledger.reserve_oracle(3000001)
    ledger.finish(attempt, dict(status="success", success=True, qualified=False))
    assert ledger.counts()["recovery"] == 2
    assert ledger.counts()["recovery_success"] == 0
    with pytest.raises(RuntimeError, match="succeeded"):
        ledger.reserve_oracle(3000001)
    with pytest.raises(RuntimeError, match="quota"):
        ledger.reserve("recovery", 3000002)
    ledger.close()


def test_oracle_command_refuses_budget_and_local_limits_before_motion(tmp_path):
    calls = []
    runner = SimpleNamespace(row=row, step=595, cfg=TakeoverConfig(), success=False,
                             move=lambda *a, **kw: calls.append((a, kw)), trace=SimpleNamespace(root=tmp_path/"trace"))
    oracle = OracleTeacher(runner)
    target = row()["ee_pose_world"].copy()
    with pytest.raises(TeacherStopped, match="budget"):
        oracle.move(target, "oracle_test", "test")
    runner.step = 40
    target[0] += .004
    with pytest.raises(TeacherStopped, match="workspace"):
        oracle.move(target, "oracle_test", "test")
    assert not calls


def test_oracle_actions_are_never_bc_positive(tmp_path):
    calls = []
    runner = SimpleNamespace(row=row, step=40, cfg=TakeoverConfig(), success=False,
                             trace=SimpleNamespace(root=tmp_path/"trace"))
    def move(target, **kwargs):
        calls.append(kwargs)
        runner.step += kwargs["duration"]
    runner.move = move
    oracle = OracleTeacher(runner)
    oracle.move(row()["ee_pose_world"], "oracle_test", "ground_truth")
    assert calls[0]["source"] == "oracle" and calls[0]["eligible"] is False
    assert oracle.events[0]["actual_actions"] == 10
    assert oracle.events[0]["privileged_feedback"] is True


def test_fork_preserves_prefix_and_keeps_branch_history_independent(tmp_path, monkeypatch):
    import h5py
    from contact_belief_wam import sim_oracle_pair
    from contact_belief_wam.sim_takeover import make_trace
    task = SimpleNamespace(cfg=SimpleNamespace(seed=3000001), step_count=80, take_action_cnt=80)
    context = dict(source="wave0", phase="natural", decision_step=30, policy_eligible=False)
    original = make_trace(tmp_path/"baseline", "checkpoint", context)
    original._open(task)
    for group, key, value in (
        ("observations", "joint_q", np.zeros((41,8))),
        ("observations", "left_rgb_marker", np.zeros((41,2,2,3), np.uint8)),
        ("actions", "clipped_actual_target", np.arange(320).reshape(40,8)),
        ("actions", "policy_eligible", np.zeros(40, bool))):
        original.file[group].create_dataset(key, data=value, maxshape=(None,*value.shape[1:]), chunks=True)
    for name in ("prism_triangles", "slot_triangles"):
        original.file.create_dataset(name, data=np.zeros((1,3),int))
    original.action_count, original.observation_count = 40, 41
    original.latest_observation = row()
    runner = SimpleNamespace(trace=original, task=task, cfg=TakeoverConfig(), step=40,
        trace_context=context, row=lambda:original.latest_observation,
        detector=SimpleNamespace(scores=list(range(41))),
        metadata=[dict(source="wave0", decision_step=t//10*10, policy_eligible=False) for t in range(40)])
    branch = sim_oracle_pair.fork_prefix(runner, tmp_path/"branch", "checkpoint")
    assert branch.step == 40 and branch.trace.observation_count == 41
    np.testing.assert_array_equal(branch.trace.file['actions/clipped_actual_target'][:], original.file['actions/clipped_actual_target'][:])
    branch.detector.scores.append(100)
    branch.metadata[0]["source"] = "changed"
    assert len(runner.detector.scores) == 41 and runner.metadata[0]["source"] == "wave0"
    assert task.step_count == task.take_action_cnt == 80
    branch.trace.close(); original.close()


def test_physical_diagnostic_keeps_strict_failure_and_rejects_state_errors():
    from copy import deepcopy
    from contact_belief_wam.oracle_recovery import physical_equivalence
    surfaces = {k:np.zeros((2,3)) for k in ('prism_vertices_world','slot_vertices_world')}
    strict = dict(passed=False, violations=['head_rgb.mae_normalized=0.01 exceeds 0.004',
        'left_depth.rmse_mm=0.04 exceeds 0.02'], tolerances={'position_m':5e-5})
    out = physical_equivalence(strict, [surfaces], [deepcopy(surfaces)])
    assert out['passed'] and not out['certifies_observation_equivalence']
    assert not strict['passed'] and len(strict['violations']) == 2
    for failure in ('object_in_gripper_pose_world.position_m=0.001 exceeds 0.00005',
                    'left_marker.rms_px=1 exceeds 0.1', 'next_action_abs=1 exceeds 0.00001'):
        invalid = dict(strict, violations=strict['violations']+[failure])
        assert not physical_equivalence(invalid, [surfaces], [surfaces])['passed']
    moved = deepcopy(surfaces)
    moved['prism_vertices_world'][0,0] = .0001
    assert not physical_equivalence(strict, [surfaces], [moved])['passed']
