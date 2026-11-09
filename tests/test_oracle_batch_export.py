"""Campaign exports exercise the production loader without running simulation."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from contact_belief_wam import oracle_batch_export as export
from contact_belief_wam.paths import sha256
from contact_belief_wam.takeover_dataset import FORMAT


class Encoder:
    metadata = {"encoder": "test", "offsets": [-6, -4, -2, 0]}

    def __init__(self):
        self.windows = []

    def __call__(self, windows):
        self.windows.extend(windows.copy())
        return np.broadcast_to(windows[:, 0, -1, 0, 0, 0, None, None], (len(windows), 8, 512)).astype(np.float32)


def make_trace(path, task, checkpoint, *, success, recovery):
    n = 12
    q = np.arange((n + 1) * 8, dtype=np.float32).reshape(n + 1, 8) / 1000
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs.update(schema_version=FORMAT, task=task, checkpoint=str(checkpoint),
                       terminal_status="success" if success else "timeout", calibration_valid=True)
        f["observations/joint_q"] = q
        for key, clock in (("physics_step", np.arange(n + 1) * 2 + 300),
                           ("action_count", np.arange(n + 1) * 2)):
            f[f"observations/{key}"] = clock
            f[f"actions/{key}_begin"] = clock[:-1]
            f[f"actions/{key}_end"] = clock[1:]
        for i, key in enumerate(("head_rgb", "wrist_rgb", "left_rgb_marker", "right_rgb_marker")):
            f[f"observations/{key}"] = np.broadcast_to(np.arange(n + 1)[:, None, None, None] + i * 20,
                                                       (n + 1, 12, 16, 3)).astype(np.uint8)
        for key in ("joint_q", "head_rgb", "wrist_rgb", "left_rgb_marker", "right_rgb_marker", "observation"):
            f[f"observations/{key}_valid"] = np.ones(n + 1, bool)
        f["actions/clipped_actual_target"] = q[:-1] + .25
        f["actions/substeps_executed"] = np.full(n, 2)
        f["actions/action_success"] = np.ones(n, bool)
        f["actions/calibration_valid"] = np.ones(n, bool)
        f["actions/decision_step"] = np.array([0] * 4 + [4] * 4 + [8] * 2 + [10] * 2)
        phases = ["wave0"] * 4 + ["unload"] * 4 + ["insert"] * 4 if recovery else ["wave0"] * n
        f.create_dataset("actions/phase", data=phases, dtype=h5py.string_dtype())
    return path


@pytest.fixture
def make_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "_video", lambda trace, dest, reference=None: dest.write_bytes(b"test-video"))

    def make(task="insert_HDMI", seed=1, *, normal=False):
        directory = tmp_path / task / str(seed)
        checkpoint = directory / "checkpoint"
        checkpoint.mkdir(parents=True)
        (checkpoint / "deployment.json").write_text(json.dumps({"camera": "head_wrist", "prompt": task}))
        low, high = (-1., 1.) if task == "insert_HDMI" else (-2., 3.)
        stats = {key: {"q01": [low] * 8, "q99": [high] * 8} for key in ("state", "actions")}
        for name in ("norm_stats.json", "wm_norm_stats.json"):
            (checkpoint / name).write_text(json.dumps(stats))
        natural = make_trace(directory / "natural.h5", task, checkpoint, success=normal, recovery=False)
        recovery = make_trace(directory / "recovery.h5", task, checkpoint, success=True, recovery=True)
        result = dict(task=task, seed=seed, checkpoint=str(checkpoint), budget=export.TASK_BUDGETS[task],
                      natural=dict(success=normal, status="success" if normal else "timeout",
                                   trajectory=str(natural), strategy_actions=12),
                      recovery=dict(success=True, status="success", trajectory=str(recovery),
                                    takeover_step=4, strategy_actions=12, physical_restore={"passed": True},
                                    timing=dict(passed=True, t_contact=1, t_tactile=2, t_decision=4, t_recovery=4),
                                    events=[dict(step=4, stop=8, decision_step=4, phase="unload", bc_eligible=True),
                                            dict(step=8, stop=10, decision_step=8, phase="insert", bc_eligible=False, stalled=True),
                                            dict(step=10, stop=12, decision_step=10, phase="insert", bc_eligible=True)]))
        path = directory / "result.json"
        path.write_text(json.dumps(result))
        return path, result
    return make


def test_recovery_real_loader_boundaries_history_and_source_immutability(make_attempt, tmp_path):
    path, result = make_attempt()
    before = {str(p): sha256(p) for p in (path, Path(result["natural"]["trajectory"]), Path(result["recovery"]["trajectory"]))}
    encoder = Encoder()
    rows = export.export_attempt(path, tmp_path / "out", encoder=encoder)
    row = rows[0]
    assert row["episode_id"] == "insert_HDMI/recovery/1"
    assert row["kind"] == "recovery" and row["suitable_for_imitation"] and not row["training_enabled"]
    assert [(r["start"], r["stop"]) for r in row["policy_supervision_ranges"]] == [(4, 8), (10, 12)]
    # Actual windows retain pre-takeover history and never include future frames.
    np.testing.assert_array_equal(encoder.windows[4][0, :, 0, 0, 0], [40, 40, 42, 44])
    check = json.loads((Path(row["review_page"]).parent / "validation.json").read_text())
    assert check["loader"]["anchors_checked"] == 12
    assert check["loader"]["supervised_anchors"] == 6
    assert check["loader"]["excluded_future_poison_checks"] >= 4
    assert check["loader"]["history_continuous"]
    assert before == {p: sha256(p) for p in before}
    assert len(export.export_attempt(path, tmp_path / "out", encoder=encoder)) == 1
    assert len(encoder.windows) == 13  # Idempotent export does not re-encode.


def test_two_task_normalization_and_ids_do_not_collide(make_attempt, tmp_path):
    output = tmp_path / "out"
    rows = []
    for task in ("insert_HDMI", "insert_hole"):
        path, _ = make_attempt(task, seed=7, normal=True)
        rows.extend(export.export_attempt(path, output, encoder=Encoder()))
    assert len({r["episode_id"] for r in rows}) == 2
    assert len({sha256(r["task_stats"]) for r in rows}) == 2
    assert all(r["policy_supervision_ranges"] == [dict(start=0, stop=12, reason="normal_success")] for r in rows)
    summary = export.rebuild_index(output)
    assert summary["counts"]["insert_hole"] == {"normal": 1, "recovery": 0}
    assert len((output / "review_manifest.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("change,match", [
    (lambda r: r["recovery"]["timing"].update(t_tactile=5), "Contact evidence"),
    (lambda r: r["recovery"]["physical_restore"].update(passed=False), "Physical restoration"),
    (lambda r: r["recovery"]["events"][0].update(stop=10), "crosses a feedback"),
    (lambda r: r.update(budget=601), "budget changed"),
])
def test_invalid_recovery_never_enters_manifest(make_attempt, tmp_path, change, match):
    path, result = make_attempt()
    change(result)
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match=match):
        export.export_attempt(path, tmp_path / "out", encoder=Encoder())
    assert not (tmp_path / "out/review_manifest.jsonl").exists()


def test_failed_branches_are_not_policy_samples(make_attempt, tmp_path):
    path, result = make_attempt()
    result["recovery"]["success"] = False
    path.write_text(json.dumps(result))
    assert export.export_attempt(path, tmp_path / "out", encoder=Encoder()) == []


def test_legacy_cache_import_keeps_flags_and_drops_stalled_insert(make_attempt, tmp_path):
    path, result = make_attempt()
    events = result["recovery"]["events"]
    for event in events:
        event.pop("bc_eligible")
        if event["phase"] == "insert":
            event.update(before={"object_in_slot_mm": [0., 0., 2.]},
                         after={"object_in_slot_mm": [0., 0., 2. if event["step"] == 8 else 1.]})
    legacy = dict(kind="oracle", seed=1, success=True, training_enabled=False, excluded_from_training=True,
                  qualified=False, trajectory=result["recovery"]["trajectory"], reference=result["natural"]["trajectory"],
                  takeover_step=4, strategy_actions=12, physical_takeover_restore={"passed": True},
                  timing=result["recovery"]["timing"], oracle={"events": events}, encoder=Encoder.metadata)
    path.write_text(json.dumps(legacy))
    np.savez(path.parent / "tactile_audit.npz", tokens=np.ones((13, 8, 512), np.float32),
             window_indices=np.stack([export.tactile_indices(t) for t in range(13)]),
             encoder_metadata=json.dumps(Encoder.metadata))
    before = sha256(path)
    row = export.import_existing(path, tmp_path / "out")[0]
    assert row["imported_existing"] and row["source_cache_sha256"]
    assert sha256(path) == before
    assert [r["start"] for r in row["policy_supervision_ranges"]] == [4, 10]
    assert json.loads(path.read_text())["excluded_from_training"]


def test_changed_source_and_reset_windows_are_rejected(make_attempt, tmp_path):
    path, result = make_attempt(normal=True)
    output = tmp_path / "out"
    export.export_attempt(path, output, encoder=Encoder())
    result["extra"] = "changed"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Previously exported source changed"):
        export.export_attempt(path, output, encoder=Encoder())
    new_path, result = make_attempt(seed=2)
    row = dict(episode_id="insert_HDMI/recovery/2", frames=13, trace_sha256=sha256(result["recovery"]["trajectory"]),
               path=result["recovery"]["trajectory"], source_result=str(new_path))
    windows = np.stack([export.tactile_indices(t) for t in range(13)])
    windows[4] = 4
    cache = output / "bad.npz"
    np.savez(cache, tokens=np.ones((13, 8, 512)), window_indices=windows,
             encoder_metadata=json.dumps(Encoder.metadata))
    with pytest.raises(ValueError, match="future frames or history reset"):
        export._cache(row, output, None, cache)


def test_configured_normal_quota_is_durable(make_attempt, tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    (output / "effective_config.json").write_text(json.dumps({"tasks": {"insert_HDMI": {"normal_target": 1}}}))
    first, _ = make_attempt(seed=1, normal=True)
    second, _ = make_attempt(seed=2, normal=True)
    assert len(export.export_attempt(first, output, encoder=Encoder())) == 1
    assert export.export_attempt(second, output, encoder=Encoder()) == []
    assert len((output / "review_manifest.jsonl").read_text().splitlines()) == 1
