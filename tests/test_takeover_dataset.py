"""Exercise real AlignedDataset loading, causal history and forbidden labels."""
import json

import h5py
import numpy as np
import pytest

from contact_belief_wam.config import ContactConfig
from contact_belief_wam.data import AlignedDataset, AlignedProcessor, prepare_aligned_batch
from contact_belief_wam.sequence import observation_at, targets_at
from contact_belief_wam.paths import sha256
from contact_belief_wam.takeover import tactile_indices
from contact_belief_wam.takeover_dataset import FORMAT


@pytest.fixture
def case(tmp_path):
    n = 18
    q = np.arange((n + 1) * 8, dtype=np.float32).reshape(n + 1, 8) / 1000
    commands = q[:-1] + .25  # Deliberately not future measured joint states.
    path = tmp_path / "trajectory.h5"
    with h5py.File(path, "w") as f:
        f.attrs.update(schema_version=FORMAT, terminal_status="success", calibration_valid=True)
        f["observations/joint_q"] = q
        for key, clock in (("physics_step", np.arange(n + 1) * 2 + 300),
                           ("action_count", np.arange(n + 1) * 2)):
            f[f"observations/{key}"] = clock
            f[f"actions/{key}_begin"] = clock[:-1]
            f[f"actions/{key}_end"] = clock[1:]
        for i, key in enumerate(("head_rgb", "wrist_rgb", "left_rgb_marker", "right_rgb_marker")):
            f[f"observations/{key}"] = np.full((n + 1, 12, 16, 3), 30 + 40 * i, np.uint8)
        for key in ("joint_q", "head_rgb", "wrist_rgb", "left_rgb_marker", "right_rgb_marker", "observation"):
            f[f"observations/{key}_valid"] = np.ones(n + 1, bool)
        f["actions/clipped_actual_target"] = commands
        f["actions/substeps_executed"] = np.full(n, 2)
        f["actions/action_success"] = np.ones(n, bool)
        f["actions/calibration_valid"] = np.ones(n, bool)
        f["actions/decision_step"] = np.array([0] * 4 + [4] * 4 + [8] * 5 + [13] * 5)
    metadata = {"offsets": [-6, -4, -2, 0], "encoder": "test"}
    cache = tmp_path / "cache"
    cache.mkdir()
    np.savez(cache / "test.npz", tokens=np.broadcast_to(np.arange(n + 1)[:, None, None],
              (n + 1, 8, 512)).astype(np.float16), raw_step=np.arange(n + 1),
              window_indices=np.stack([tactile_indices(t) for t in range(n + 1)]),
              metadata=json.dumps(metadata), trace_sha256=sha256(path))
    row = dict(data_format=FORMAT, frames=n + 1, path=str(path), episode_id="test",
               task="insert_HDMI", source="recovery", split="review", prompt="insert HDMI",
               segments=[[0, n + 1]], context_range=[0, n + 1], training_enabled=False,
               actual_command_key="actions/clipped_actual_target", encoder_metadata=metadata,
               trace_sha256=sha256(path), policy_supervision_ranges=[
                   dict(start=8, stop=13, decision_step=8), dict(start=13, stop=18, decision_step=13)])
    stats = {key: {"q01": [-1.] * 8, "q99": [1.] * 8} for key in ("state", "actions")}
    def dataset(**kw):
        cfg = kw.pop("config", ContactConfig(aligned=True, stage="stage2", camera_mode="head_wrist",
                     k=4, chunk_size=8, belief_warmup_min=2, belief_warmup_max=2))
        return AlignedDataset([row], tmp_path, cache, {"insert_HDMI": stats}, stats, cfg, split="all",
                              allow_review=kw.pop("allow_review", True), **kw)
    return dataset, row, q, commands, cache, stats


def sample_at(dataset, frame):
    return dataset[next(i for i, entry in enumerate(dataset.index) if entry[3] == frame)]


def test_actual_command_labels_and_continuous_pre_takeover_history(case):
    build, row, joints, commands, _, stats = case
    ds = build()
    s = sample_at(ds, 8)
    np.testing.assert_array_equal(s["step"], [0, 4, 8])
    np.testing.assert_array_equal(s["sequence_mask"], [True] * 3)
    inverse = AlignedProcessor(stats, None, "")
    decoded = inverse.absolute(s["actions"][-1], joints[8])
    np.testing.assert_allclose(decoded[:5], commands[8:13], atol=1e-6)
    np.testing.assert_allclose(decoded[5:], np.repeat(commands[12:13], 3, axis=0), atol=1e-6)
    history = inverse.absolute(s["executed_actions"][-1], joints[4])
    np.testing.assert_allclose(history, commands[4:8], atol=1e-6)
    np.testing.assert_array_equal(s["tactile"][:, 0, 0], [0, 4, 8])
    assert s["executed_valid"][-1].all()
    assert not s["policy_action_mask"][:2].any()
    assert s["policy_action_mask"][-1].sum() == 5
    assert not np.allclose(decoded[0], joints[9])
    assert not np.array_equal(s["image"], s["wrist_image"])
    # Successful recovery does not turn the failed prefix into positive BC labels.
    assert not sample_at(ds, 7)["policy_action_mask"].any()


def test_excluded_future_values_cannot_enter_flow_input_but_wm_keeps_transition(case):
    build, row, joints, commands, _, _ = case
    ds = build()
    before = sample_at(ds, 8)
    ds._load(0)["commands"][13:] += 123
    after = sample_at(ds, 8)
    for key in ("actions", "image", "wrist_image", "tactile", "state", "executed_actions"):
        np.testing.assert_array_equal(before[key], after[key])
    assert not np.array_equal(before["world_actions"], after["world_actions"])
    # The exact noisy tensor sent to a flow expert is also independent of excluded labels.
    noise = np.random.default_rng(4).normal(size=before["actions"].shape)
    np.testing.assert_array_equal(.3 * noise + .7 * before["actions"], .3 * noise + .7 * after["actions"])


def test_boundary_masks_and_terminal_observation(case):
    build, row, joints, commands, _, _ = case
    ds = build()
    assert sample_at(ds, 12)["policy_action_mask"][-1].sum() == 1
    assert sample_at(ds, 13)["policy_action_mask"][-1].sum() == 5
    assert sample_at(ds, 17)["action_mask"][-1].sum() == 1
    assert sample_at(ds, 17)["policy_action_mask"][-1].sum() == 1
    assert sample_at(ds, 14)["future_mask"][-1]
    assert sample_at(ds, 14)["future_visual_frame_index"][-1] == 18
    assert not sample_at(ds, 15)["future_mask"][-1]
    assert len(ds) == 18  # terminal o[18] is a WM target, never an action anchor


def test_normal_success_retains_full_valid_supervision(case):
    build, row, joints, commands, _, _ = case
    row["policy_supervision_ranges"] = [dict(start=0, stop=18, reason="normal_success")]
    row["source"] = "original"
    s = sample_at(build(), 7)
    assert s["policy_action_mask"][-1].sum() == 8


def test_review_contract_cannot_accidentally_enter_training(case):
    build, row, *_ = case
    with pytest.raises(ValueError, match="training is disabled"):
        build(allow_review=False)
    with pytest.raises(ValueError, match="cannot be disabled"):
        build(policy_action_mask=False)
    row["segments"] = [[8, 19]]
    with pytest.raises(ValueError, match=r"complete N\+1"):
        build()


def test_missing_wrist_bad_clock_and_future_tactile_are_rejected(case):
    build, row, _, _, cache, _ = case
    with h5py.File(row["path"], "r+") as f:
        original = f["actions/physics_step_begin"][8]
        f["actions/physics_step_begin"][8] += 1
    with pytest.raises(ValueError, match="Discontinuous"):
        sample_at(build(), 8)
    with h5py.File(row["path"], "r+") as f:
        f["actions/physics_step_begin"][8] = original
        f["observations/wrist_rgb_valid"][8] = False
    with pytest.raises(ValueError, match="wrist"):
        sample_at(build(), 8)
    with h5py.File(row["path"], "r+") as f:
        f["observations/wrist_rgb_valid"][8] = True
    with np.load(cache / "test.npz") as f:
        values = {k: f[k] for k in f.files}
    values["window_indices"][8, -1] = 9
    np.savez(cache / "test.npz", **values)
    with pytest.raises(ValueError, match="causal"):
        sample_at(build(), 8)


def test_review_batch_reaches_real_model_input_without_ground_truth(case):
    import torch
    from torch.utils.data import DataLoader
    build, *_ = case
    batch = prepare_aligned_batch(next(iter(DataLoader(build(), batch_size=2))), "cpu")
    obs, target = observation_at(batch, -1), targets_at(batch, -1)
    assert obs.image.shape == obs.wrist_image.shape == (2, 3, 224, 224)
    assert obs.tactile.shape == (2, 8, 512)
    assert target.actions.shape == (2, 8, 32)
    assert not any(any(secret in key for secret in ("prism", "slot", "oracle", "truth")) for key in batch)
    assert torch.isfinite(target.actions).all()


def test_changed_raw_commands_and_cross_decision_labels_fail_closed(case):
    build, row, *_ = case
    row["policy_supervision_ranges"][0]["stop"] = 14
    row["policy_supervision_ranges"] = row["policy_supervision_ranges"][:1]
    with pytest.raises(ValueError, match="crosses a new feedback"):
        sample_at(build(), 8)
    row["policy_supervision_ranges"][0]["stop"] = 13
    with h5py.File(row["path"], "r+") as f:
        f["actions/clipped_actual_target"][12, 0] += .01
    with pytest.raises(ValueError, match="Raw trace changed"):
        sample_at(build(), 8)


def test_head_only_keeps_bc_boundaries_and_unfiltered_history(case):
    build, row, joints, commands, _, stats = case
    cfg = ContactConfig(aligned=True, stage="stage2", camera_mode="head_only", k=4, chunk_size=8,
                        belief_warmup_min=2, belief_warmup_max=2)
    ds = build(config=cfg, policy_supervised_only=True)
    assert [entry[3] for entry in ds.index] == list(range(8, 18))
    sample = sample_at(ds, 8)
    assert "wrist_image" not in sample and "future_wrist_image" not in sample
    np.testing.assert_array_equal(sample["step"], [0, 4, 8])
    np.testing.assert_array_equal(sample["tactile"][:, 0, 0], [0, 4, 8])
    decoded = AlignedProcessor(stats, None, "").absolute(sample["executed_actions"][-1], joints[4])
    np.testing.assert_allclose(decoded, commands[4:8], atol=1e-6)
    ds._load(0)["commands"][13:] += 100
    np.testing.assert_array_equal(sample["actions"], sample_at(ds, 8)["actions"])


def test_stage1_head_only_retains_unsupervised_dynamics(case):
    build, row, joints, commands, cache, stats = case
    cfg = ContactConfig(aligned=True, stage="stage1", camera_mode="head_only", k=4, chunk_size=8,
                        belief_warmup_min=0, belief_warmup_max=0)
    ds = build(config=cfg)
    sample = sample_at(ds, 4)
    assert not sample["policy_action_mask"].any()
    expected = AlignedProcessor(stats, None, "").absolute(sample["world_actions"][-1], joints[4])
    np.testing.assert_allclose(expected, commands[4:12], atol=1e-6)
    assert "wrist_image" not in sample


def test_prepared_model_cache_matches_raw_inputs_and_preserves_causal_history(case, tmp_path):
    from contact_belief_wam.finetune_data import prepare_model_cache
    build, row, _, _, cache, _ = case
    cfg = ContactConfig(aligned=True, stage="stage2", camera_mode="head_only", k=4, chunk_size=8,
                        belief_warmup_min=2, belief_warmup_max=2)
    reference = sample_at(build(config=cfg), 8)
    row["feature_cache_root"] = str(cache)
    path = tmp_path / "model_inputs"
    checksums = prepare_model_cache(row, path)
    row["model_input_cache"] = str(path)
    ds = build(config=cfg)
    cached = sample_at(ds, 8)
    assert isinstance(ds._load(0)["images"]["head"], np.memmap)
    for key in ("image", "future_image", "actions", "world_actions", "executed_actions", "tactile", "policy_action_mask"):
        np.testing.assert_array_equal(reference[key], cached[key])
    assert checksums == prepare_model_cache(row, path)
    pixels = np.load(path / "head.npy", mmap_mode="r+")
    pixels[0, 0, 0, 0] += 1
    pixels.flush()
    with pytest.raises(ValueError, match="cache changed"):
        prepare_model_cache(row, path)
