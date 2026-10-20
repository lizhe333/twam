import json

import h5py
import numpy as np
import pytest

from contact_belief_wam.config import ContactConfig
from contact_belief_wam.data import ContactDataset, MixedSequenceSampler, build_manifest, contiguous_segments
from contact_belief_wam.paths import read_jsonl, write_jsonl
from contact_belief_wam.preprocessing import Quantiles


def _make_dataset(tmp_path, episode_specs, config):
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    data_root.mkdir(parents=True)
    rows, episode_values = [], {}
    for spec in episode_specs:
        n = spec["frames"]
        declared_segments = spec["segments"]
        steps = np.empty(n, dtype=np.int64)
        for segment_number, (start, stop) in enumerate(declared_segments):
            steps[start:stop] = segment_number * 1000 + 2 * np.arange(stop - start)
        segments = contiguous_segments(steps)
        assert segments == declared_segments

        image_base = spec["image_base"]
        joint_base = spec["joint_base"]
        tactile_base = spec["tactile_base"]
        joints = np.empty((n, 9), dtype=np.float32)
        joints[:, :8] = joint_base + 10 * np.arange(n)[:, None] + np.arange(8)[None]
        joints[:, 8] = 9999.0  # Sentinel: state and action targets must use only the first 8 joints.
        rgb = np.stack([
            np.full((2, 2, 3), image_base + frame, dtype=np.uint8)
            for frame in range(n)
        ])

        relative_path = f"hdf5/{spec['source']}/insert_hole/{spec['name']}.hdf5"
        hdf5_path = data_root / relative_path
        hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(hdf5_path, "w") as handle:
            handle.create_dataset("step", data=steps)
            handle.require_group("embodiment").create_dataset("joint", data=joints)
            handle.require_group("observation/head").create_dataset("rgb", data=rgb)

        episode_id = f"{spec['source']}/insert_hole/{spec['name']}"
        row = {
            "episode_id": episode_id,
            "task": "insert_hole",
            "source": spec["source"],
            "path": relative_path,
            "split": "train",
            "frames": n,
            "segments": segments,
            "prompt": "insert the object",
        }
        rows.append(row)
        cache_path = cache_root / f"{episode_id}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tokens = np.broadcast_to(
            (tactile_base + np.arange(n, dtype=np.float32))[:, None, None],
            (n, 8, 512),
        ).astype(np.float16).copy()
        np.savez_compressed(cache_path, tokens=tokens, raw_step=steps)
        episode_values[episode_id] = {
            "image_base": image_base,
            "joints": joints,
            "source": spec["source"],
            "frames": n,
        }

    manifest = data_root / "manifest.jsonl"
    write_jsonl(manifest, rows)
    stats = {
        key: {"q01": [0.0] * 8, "q99": [10000.0] * 8}
        for key in ("state", "action")
    }
    dataset = ContactDataset(manifest, data_root, cache_root, stats, config, seed=17)
    return dataset, episode_values, stats


def _find_item(dataset, episode_id, frame):
    for index, (rid, _start, _stop, current) in enumerate(dataset.index):
        if dataset.rows[rid]["episode_id"] == episode_id and current == frame:
            return index
    raise AssertionError(f"No dataset item for {episode_id} at frame {frame}")


def _spec(source, name, frames, segments, image_base, joint_base, tactile_base):
    return {
        "source": source,
        "name": name,
        "frames": frames,
        "segments": segments,
        "image_base": image_base,
        "joint_base": joint_base,
        "tactile_base": tactile_base,
    }


def _write_manifest_hdf5(path, frames=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("step", data=2 * np.arange(frames, dtype=np.int64))
        handle.require_group("embodiment").create_dataset(
            "joint", data=np.arange(frames * 9, dtype=np.float32).reshape(frames, 9),
        )
        handle.require_group("observation/head").create_dataset(
            "rgb", data=np.zeros((frames, 2, 2, 3), dtype=np.uint8),
        )
        for side in ("left", "right"):
            handle.require_group(f"tactile/{side}_gsmini").create_dataset(
                "rgb_marker", data=np.zeros((frames, 2, 2, 3), dtype=np.uint8),
            )


def test_build_manifest_requires_eligible_collection_record_for_recovery(tmp_path, monkeypatch):
    import contact_belief_wam.data as data_module

    workspace = tmp_path / "workspace"
    instructions = workspace / "UniVTAC/instructions"
    instructions.mkdir(parents=True)
    (instructions / "insert_hole.json").write_text(json.dumps({"seen": ["insert the object"]}))
    monkeypatch.setattr(data_module, "WORKSPACE", workspace)

    original_root = tmp_path / "original"
    _write_manifest_hdf5(original_root / "insert_hole/hdf5/original_episode.hdf5")

    # With no collection manifest, recovery files are ignored while original data
    # continues to build and validate normally.
    missing_report_root = tmp_path / "recovery_missing_report"
    _write_manifest_hdf5(missing_report_root / "insert_hole/hdf5/no_manifest.hdf5")
    missing_report_output = tmp_path / "prepared_missing_report"
    summary = build_manifest(
        missing_report_output, original_root=original_root,
        recovery_root=missing_report_root, tasks=("insert_hole",),
    )
    assert summary["insert_hole"] == {"original": 1, "recovery": 0}
    rows = read_jsonl(missing_report_output / "manifests/insert_hole.jsonl")
    assert [row["episode_id"] for row in rows] == ["original/insert_hole/original_episode"]

    empty_report_root = tmp_path / "recovery_empty_report"
    _write_manifest_hdf5(empty_report_root / "insert_hole/hdf5/no_record.hdf5")
    (empty_report_root / "insert_hole/collection.json").write_text(json.dumps({"attempts": []}))
    empty_report_output = tmp_path / "prepared_empty_report"
    summary = build_manifest(
        empty_report_output, original_root=original_root,
        recovery_root=empty_report_root, tasks=("insert_hole",),
    )
    assert summary["insert_hole"] == {"original": 1, "recovery": 0}
    rows = read_jsonl(empty_report_output / "manifests/insert_hole.jsonl")
    assert [row["episode_id"] for row in rows] == ["original/insert_hole/original_episode"]

    # A present report still has to contain an explicitly eligible row for each
    # recovery HDF5. Missing records, failures, and excluded episodes are rejected.
    recovery_root = tmp_path / "recovery_with_report"
    recovery_dir = recovery_root / "insert_hole"
    for name in ("no_record", "failed", "excluded", "eligible"):
        _write_manifest_hdf5(recovery_dir / f"hdf5/{name}.hdf5")
    attempts = [
        {"seed": "failed", "success": False, "blockages": 1},
        {"seed": "excluded", "success": True, "blockages": 1, "excluded_from_training": True},
        {"seed": "eligible", "success": True, "blockages": 1, "excluded_from_training": False},
    ]
    (recovery_dir / "collection.json").write_text(json.dumps({"attempts": attempts}))

    output = tmp_path / "prepared_with_report"
    summary = build_manifest(
        output, original_root=original_root, recovery_root=recovery_root,
        tasks=("insert_hole",),
    )
    assert summary["insert_hole"] == {"original": 1, "recovery": 1}
    rows = read_jsonl(output / "manifests/insert_hole.jsonl")
    assert {(row["source"], row["episode_id"]) for row in rows} == {
        ("original", "original/insert_hole/original_episode"),
        ("recovery", "recovery/insert_hole/eligible"),
    }


@pytest.mark.parametrize("k", [5, 10])
def test_nine_joint_hdf5_produces_eight_dimensional_next_frame_targets_and_t_k_masks(tmp_path, k):
    segment_length = 2 * k + 2
    frames = segment_length * 2
    spec = _spec(
        "original", f"next-frame-k{k}", frames,
        [[0, segment_length], [segment_length, frames]],
        image_base=20, joint_base=100, tactile_base=30,
    )
    config = ContactConfig(
        k=k, chunk_size=2 * k, belief_warmup_min=0, belief_warmup_max=0,
    )
    dataset, episode_values, stats = _make_dataset(tmp_path, [spec], config)
    episode_id = "original/insert_hole/next-frame-k" + str(k)
    raw_joints = episode_values[episode_id]["joints"]
    quantiles = Quantiles(stats)

    current = 4
    sample = dataset[(_find_item(dataset, episode_id, current), 41)]
    assert sample["state"].shape == (1, 8)
    assert sample["actions"].shape == (1, 2 * k, 8)
    np.testing.assert_allclose(
        quantiles.unnormalize(sample["state"], "state")[0], raw_joints[current, :8], atol=2e-4,
    )
    np.testing.assert_allclose(
        quantiles.unnormalize(sample["actions"], "action")[0, 0], raw_joints[current + 1, :8], atol=2e-4,
    )
    assert not np.allclose(raw_joints[current + 1, :8], raw_joints[current + 1, :8] - raw_joints[current, :8])
    assert sample["action_mask"][0, 0]

    segment_stop = segment_length
    for current, expected_valid in ((segment_stop - k - 1, True), (segment_stop - k, False)):
        boundary_sample = dataset[(_find_item(dataset, episode_id, current), 3)]
        expected_target = min(current + k, segment_stop - 1)
        assert bool(boundary_sample["future_mask"][0]) is expected_valid
        assert boundary_sample["future_image"][0, 0, 0, 0] == 20 + expected_target
        assert boundary_sample["future_tactile"][0, 0, 0] == 30 + expected_target

        if not expected_valid:
            # The action chunk is clipped to the segment's final target and masked there;
            # it must never read the first frame of the following segment.
            assert int(boundary_sample["action_mask"][0].sum()) == k - 1
            raw_actions = quantiles.unnormalize(boundary_sample["actions"][0], "action")
            expected_padding = np.broadcast_to(raw_joints[segment_stop - 1, :8], raw_actions[k - 1:].shape)
            np.testing.assert_allclose(raw_actions[k - 1:], expected_padding, atol=2e-4)
            assert boundary_sample["future_image"][0, 0, 0, 0] != 20 + segment_stop


def test_random_zero_to_four_history_uses_left_padding_and_stays_in_its_segment(tmp_path):
    spec = _spec(
        "original", "history", 64, [[0, 30], [30, 64]],
        image_base=20, joint_base=100, tactile_base=30,
    )
    config = ContactConfig(k=2, chunk_size=8, belief_warmup_min=0, belief_warmup_max=4)
    dataset, _, _ = _make_dataset(tmp_path, [spec], config)
    episode_id = "original/insert_hole/history"
    start, current = 30, 46
    item = _find_item(dataset, episode_id, current)
    seen_warmups = set()

    for draw_seed in range(100):
        sample = dataset[(item, draw_seed)]
        valid_count = int(sample["sequence_mask"].sum())
        warmup = valid_count - 1
        seen_warmups.add(warmup)
        first_valid_frame = current - warmup * config.k
        expected_valid_steps = np.arange(first_valid_frame - start, current - start + 1, config.k)
        np.testing.assert_array_equal(sample["step"][-valid_count:], expected_valid_steps)
        np.testing.assert_array_equal(sample["step"][:-valid_count], np.repeat(first_valid_frame - start, 5 - valid_count))
        assert sample["sequence_mask"].tolist() == [False] * (5 - valid_count) + [True] * valid_count
        if valid_count < 5:
            # Invalid leading slots repeat the earliest real frame, rather than carrying
            # unrelated older observations into a sequence described as left-padded.
            assert np.all(sample["image"][:5 - valid_count, 0, 0, 0] == 20 + first_valid_frame)

    assert seen_warmups == {0, 1, 2, 3, 4}

    # Request four warmup steps at the first frame after a segment break. The available
    # history is zero, so all left padding must repeat frame 30 and never frame 29.
    boundary_config = ContactConfig(k=2, chunk_size=8, belief_warmup_min=4, belief_warmup_max=4)
    boundary_dataset, _, _ = _make_dataset(tmp_path / "boundary", [spec], boundary_config)
    boundary = boundary_dataset[(_find_item(boundary_dataset, "original/insert_hole/history", 31), 9)]
    assert boundary["sequence_mask"].tolist() == [False, False, False, False, True]
    np.testing.assert_array_equal(boundary["step"], [1, 1, 1, 1, 1])
    np.testing.assert_array_equal(boundary["image"][:, 0, 0, 0], np.full(5, 20 + 31))
    assert 20 + 29 not in boundary["image"][:, 0, 0, 0]


def test_mixed_sampler_is_one_to_one_reproducible_resumable_and_episode_local(tmp_path):
    specs = [
        _spec("original", "source-original", 64, [[0, 30], [30, 64]], 20, 100, 30),
        _spec("recovery", "source-recovery", 64, [[0, 30], [30, 64]], 120, 3000, 130),
    ]
    config = ContactConfig(k=2, chunk_size=8, belief_warmup_min=0, belief_warmup_max=4)
    dataset, episode_values, _ = _make_dataset(tmp_path, specs, config)

    complete = list(MixedSequenceSampler(dataset, draws=1200, seed=1234))
    resumed = list(MixedSequenceSampler(dataset, draws=1100, seed=1234, start_draw=100))
    assert resumed == complete[100:]
    source_counts = {source: 0 for source in ("original", "recovery")}
    for index, _draw_seed in complete:
        source_counts[dataset.sources[index]] += 1
    assert abs(source_counts["original"] / len(complete) - 0.5) < 0.06
    assert abs(source_counts["recovery"] / len(complete) - 0.5) < 0.06

    for draw in complete[::60]:
        index, draw_seed = draw
        sample = dataset[draw]
        rid = dataset.index[index][0]
        row = dataset.rows[rid]
        values = episode_values[row["episode_id"]]
        assert sample["episode_id"] == row["episode_id"]
        assert sample["source"] == row["source"] == dataset.sources[index]
        for key in ("image", "future_image"):
            markers = sample[key][..., 0]
            assert np.all(markers >= values["image_base"])
            assert np.all(markers < values["image_base"] + values["frames"])
        for key in ("tactile", "future_tactile"):
            markers = sample[key][..., 0]
            assert np.all(markers >= (130 if row["source"] == "recovery" else 30))
            assert np.all(markers < (130 if row["source"] == "recovery" else 30) + values["frames"])

        repeated = dataset[(index, draw_seed)]
        for key in ("image", "state", "actions", "sequence_mask", "future_mask"):
            np.testing.assert_array_equal(repeated[key], sample[key])
