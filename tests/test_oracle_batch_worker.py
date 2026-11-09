"""Native worker prefix and label bookkeeping without starting Isaac Sim."""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from contact_belief_wam import sim_oracle_batch as worker
from contact_belief_wam.paths import sha256
from contact_belief_wam.takeover import tactile_indices


class FakeTrace:
    def __init__(self, root, checkpoint, context, task):
        self.path = root / "trajectory.h5"
        self.checkpoint, self.context, self.task_name = checkpoint, context, task

    def _open(self, task):
        self.path.parent.mkdir(parents=True)
        self.file = h5py.File(self.path, "x")
        self.file.create_group("actions")
        self.file.create_group("observations")
        self.file.attrs.update(task=self.task_name, checkpoint=str(self.checkpoint))


@pytest.mark.parametrize("task", ["insert_HDMI", "insert_hole"])
@pytest.mark.parametrize("takeover", [0, 4, 10])
def test_prefix_has_n_commands_n_plus_one_observations_and_continuous_history(tmp_path, monkeypatch, task, takeover):
    n, original = 14, tmp_path / "natural.h5"
    joints = np.arange((n + 1) * 8).reshape(n + 1, 8).astype(np.float32)
    commands = joints[:-1] + .125
    with h5py.File(original, "w") as f:
        f.attrs.update(task=task, checkpoint="checkpoint", terminal_status="failed",
                       terminal_reason="budget", action_rows=n, observation_rows=n + 1)
        obs, actions = f.create_group("observations"), f.create_group("actions")
        for key, data in (("joint_q", joints), ("physics_step", np.arange(n + 1) * 2),
                          ("action_count", np.arange(n + 1) * 2)):
            obs.create_dataset(key, data=data, chunks=True, maxshape=(None,) + data.shape[1:])
        obs["joint_q"].attrs["units"] = "rad"
        for key in ("head_rgb", "wrist_rgb", "left_rgb_marker", "right_rgb_marker"):
            obs.create_dataset(key, data=np.broadcast_to(np.arange(n + 1)[:, None, None, None],
                               (n + 1, 8, 8, 3)).astype(np.uint8), chunks=(1, 8, 8, 3), maxshape=(None, 8, 8, 3))
        actions.create_dataset("clipped_actual_target", data=commands, chunks=True, maxshape=(None, 8))
        actions.create_dataset("policy_eligible", data=np.ones(n, bool), chunks=True, maxshape=(None,))
        actions.create_dataset("fk_valid", data=np.ones(n, bool), chunks=True, maxshape=(None,))
        actions.create_dataset("calibration_valid", data=np.ones(n, bool), chunks=True, maxshape=(None,))
        actions.create_dataset("decision_step", data=np.repeat([0, 4, 8, 12], 4)[:n], chunks=True, maxshape=(None,))
        actions.create_dataset("source", data=np.asarray([b"wave0"] * n, dtype="S64"), chunks=True, maxshape=(None,))
        for name in ("prism", "slot"):
            f[name + "_triangles"] = np.array([[0, 1, 2]], np.int32)
    source_hash = sha256(original)
    detector = SimpleNamespace(candidate={"t_contact": 1, "t_tactile": 2}, history=[{"t": t} for t in range(takeover + 1)])
    baseline = SimpleNamespace(task=object(), trace=SimpleNamespace(path=original, fk_checked=n, calibration_valid=True),
                               cfg=SimpleNamespace(budget=317 if task == "insert_hole" else 600),
                               metadata=[dict(step=t, phase="wave0", policy_eligible=True) for t in range(n)],
                               topology={"prism": np.array([[0, 1, 2]]), "slot": np.array([[0, 1, 2]])})
    monkeypatch.setattr(worker, "batch_trace", FakeTrace)
    monkeypatch.setattr(worker, "surface_distance", lambda row, topology: .0003)
    branch = worker.fork_prefix(baseline, tmp_path / "branch", "checkpoint", task, takeover, detector)
    try:
        f = branch.trace.file
        assert branch.step == takeover and branch.trace.observation_count == takeover + 1
        assert f.attrs["action_rows"] == takeover and f.attrs["observation_rows"] == takeover + 1
        assert f.attrs["task"] == task and branch.trace.task_name == task
        assert f.attrs["terminal_status"] == "running" and not f.attrs["terminal_reason"]
        assert branch.trace.fk_checked == takeover
        assert branch.trace.calibration_valid == bool(takeover)
        np.testing.assert_array_equal(f["actions/clipped_actual_target"][:], commands[:takeover])
        np.testing.assert_array_equal(f["observations/joint_q"][:], joints[:takeover + 1])
        assert f["observations/joint_q"].attrs["units"] == "rad"
        np.testing.assert_array_equal(branch.row()["joint_q"], joints[takeover])
        for side in ("left", "right"):
            # No takeover-local reset: the model can still access the real four-frame history.
            actual = [f[f"observations/{side}_rgb_marker"][int(t), 0, 0, 0] for t in tactile_indices(takeover)]
            np.testing.assert_array_equal(actual, tactile_indices(takeover))
        assert len(branch.metadata) == takeover and not branch.success
        assert branch.trace_context == dict(source="oracle", phase="takeover", decision_step=takeover, policy_eligible=False)
        assert branch.teacher_events == [] and branch.last_distance == .0003
        branch.detector.history.append({"t": 999})
        assert len(detector.history) == takeover + 1
        if takeover:
            branch.metadata[-1]["phase"] = "changed"
            assert baseline.metadata[takeover - 1]["phase"] == "wave0"
    finally:
        branch.trace.file.close()
    assert sha256(original) == source_hash


@pytest.mark.parametrize("success", [False, True])
def test_failed_branch_has_no_positive_bc_flags_even_when_segments_were_useful(tmp_path, success):
    with h5py.File(tmp_path / "labels.h5", "w") as handle:
        handle["actions/policy_eligible"] = np.ones(12, bool)
        trace = SimpleNamespace(file=handle, action_count=12)
        branch = worker.BatchRunner.__new__(worker.BatchRunner)
        branch.trace, branch.success = trace, success
        branch.metadata = [dict(policy_eligible=True) for _ in range(12)]
        events = [dict(step=4, stop=8, bc_eligible=True), dict(step=8, stop=10, bc_eligible=False),
                  dict(step=10, stop=12, bc_eligible=True)]
        worker.mark_labels(branch, events)
        expected = np.zeros(12, bool)
        if success:
            expected[4:8] = expected[10:12] = True
        np.testing.assert_array_equal(handle["actions/policy_eligible"][:], expected)
        np.testing.assert_array_equal([r["policy_eligible"] for r in branch.metadata], expected)


def test_batch_trace_assigns_actual_task_before_open(monkeypatch, tmp_path):
    created = SimpleNamespace(task_name="insert_HDMI")
    monkeypatch.setattr(worker, "make_trace", lambda *args: created)
    result = worker.batch_trace(tmp_path, "checkpoint", {}, "insert_hole")
    assert result is created and result.task_name == "insert_hole"
