import h5py
import numpy as np

from contact_belief_wam.config import ContactConfig
import contact_belief_wam.data as data_module
from contact_belief_wam.data import ContactDataset
from contact_belief_wam.feature_cache import prepare_mmap_cache
from contact_belief_wam.paths import write_jsonl


def _make_dataset(tmp_path):
    data_root, cache_root = tmp_path / "data", tmp_path / "cache"
    frames = 12
    steps = 2 * np.arange(frames, dtype=np.int64)
    joints = np.arange(frames * 9, dtype=np.float32).reshape(frames, 9)
    images = np.stack([
        np.full((3, 4, 3), frame, dtype=np.uint8) for frame in range(frames)
    ])
    path = data_root / "hdf5/original/insert_hole/episode.hdf5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("step", data=steps)
        handle.require_group("embodiment").create_dataset("joint", data=joints)
        handle.require_group("observation/head").create_dataset("rgb", data=images)

    episode_id = "original/insert_hole/episode"
    manifest = data_root / "manifests/train.jsonl"
    write_jsonl(manifest, [{
        "episode_id": episode_id,
        "task": "insert_hole",
        "source": "original",
        "path": "hdf5/original/insert_hole/episode.hdf5",
        "split": "train",
        "frames": frames,
        "segments": [[0, frames]],
        "prompt": "insert",
    }])
    cache_path = cache_root / f"{episode_id}.npz"
    cache_path.parent.mkdir(parents=True)
    tokens = np.arange(frames, dtype=np.float16)[:, None, None]
    np.savez(cache_path, tokens=np.broadcast_to(tokens, (frames, 8, 512)), raw_step=steps)
    stats = {key: {"q01": [0] * 8, "q99": [200] * 8} for key in ("state", "action")}
    config = ContactConfig(k=2, chunk_size=8, belief_warmup_min=0, belief_warmup_max=4)
    return ContactDataset(manifest, data_root, cache_root, stats, config), images, cache_root


def _legacy_sample(dataset, item, decode):
    """Reference the pre-deduplication __getitem__ behavior for exact parity."""
    index, draw_seed = item
    rng = np.random.default_rng(draw_seed)
    rid, start, stop, current = dataset.index[index]
    row = dataset.rows[rid]
    joints, encoded, tactile, raw_steps, atom_tags = dataset._load(rid)
    cfg = dataset.config
    warmup = int(rng.integers(cfg.belief_warmup_min, cfg.belief_warmup_max + 1))
    warmup = min(warmup, (current - start) // cfg.k)
    length = cfg.belief_warmup_max + 1
    frames = current - np.arange(length - 1, -1, -1) * cfg.k
    seq_valid = np.arange(length) >= length - warmup - 1
    frames = np.clip(frames, start, stop - 1)
    padding = length - warmup - 1
    if padding:
        frames[:padding] = frames[padding]
    action_indices = frames[:, None] + 1 + np.arange(cfg.chunk_size)[None]
    action_valid = (action_indices < stop) & seq_valid[:, None]
    future_indices = frames + cfg.k
    future_valid = (future_indices < stop) & seq_valid
    action_indices = np.minimum(action_indices, stop - 1)
    future_indices = np.minimum(future_indices, stop - 1)
    executed_indices = frames[:, None] - cfg.k + 1 + np.arange(cfg.k)[None]
    executed_valid = (executed_indices > start) & seq_valid[:, None]
    executed_indices = np.clip(executed_indices, start, stop - 1)
    image = np.stack([decode(encoded[i]) for i in frames])
    future_image = np.stack([decode(encoded[i]) for i in future_indices])
    policy_action_mask = action_valid.copy()
    if dataset.policy_action_mask:
        excluded = np.zeros_like(action_valid)
        if atom_tags is not None:
            excluded |= atom_tags[action_indices] == "perturb"
        target_steps = raw_steps[action_indices]
        for interval in row.get("policy_action_excluded_ranges", []):
            excluded |= ((target_steps > int(interval["start_exclusive"])) &
                         (target_steps <= int(interval["end_inclusive"])))
        policy_action_mask &= ~excluded
    return {
        "image": image,
        "future_image": future_image,
        "state": dataset.norm.normalize(joints[frames], "state"),
        "actions": dataset.norm.normalize(joints[action_indices], "action"),
        "executed_actions": dataset.norm.normalize(joints[executed_indices], "action"),
        "executed_valid": executed_valid,
        "tactile": tactile[frames].astype(np.float32),
        "future_tactile": tactile[future_indices].astype(np.float32),
        "action_mask": action_valid,
        "future_mask": future_valid,
        "sequence_mask": seq_valid,
        "policy_action_mask": policy_action_mask,
        "step": frames - start,
        "source": row["source"],
        "task": row["task"],
        "prompt": row["prompt"],
        "episode_id": row["episode_id"],
    }


def test_getitem_decodes_unique_frames_once_and_matches_legacy(tmp_path, monkeypatch):
    dataset, encoded_images, _cache_root = _make_dataset(tmp_path)
    # The dataset ends at observation frame n-2; use that item for both warmup
    # cases so the future frame also exercises end-of-episode clamping.
    item_index = next(i for i, (_rid, _start, _stop, frame) in enumerate(dataset.index)
                      if frame == len(encoded_images) - 2)
    seeds = {}
    for seed in range(1000):
        warmup = int(np.random.default_rng(seed).integers(0, 5))
        seeds.setdefault(warmup, seed)
        if 0 in seeds and 4 in seeds:
            break
    assert set(seeds) >= {0, 4}

    legacy_decode = data_module.decode_rgb
    for warmup in (0, 4):
        expected = _legacy_sample(dataset, (item_index, seeds[warmup]), legacy_decode)
        calls = []

        def counted_decode(encoded):
            calls.append(int(encoded[0, 0, 0]))
            return legacy_decode(encoded)

        monkeypatch.setattr(data_module, "decode_rgb", counted_decode)
        actual = dataset[(item_index, seeds[warmup])]
        monkeypatch.setattr(data_module, "decode_rgb", legacy_decode)

        assert actual.keys() == expected.keys()
        for key, expected_value in expected.items():
            if isinstance(expected_value, np.ndarray):
                assert actual[key].dtype == expected_value.dtype, key
                np.testing.assert_array_equal(actual[key], expected_value, err_msg=key)
            else:
                assert actual[key] == expected_value
        all_legacy_indices = np.concatenate((expected["step"],
                                             np.minimum(expected["step"] + dataset.config.k,
                                                        len(encoded_images) - 1)))
        assert len(calls) == len(np.unique(all_legacy_indices))
        assert len(calls) < 2 * len(expected["step"])


def test_getitem_tactile_mmap_matches_npz_fallback(tmp_path):
    dataset, _encoded_images, cache_root = _make_dataset(tmp_path)
    item_index = next(i for i, (_rid, _start, _stop, frame) in enumerate(dataset.index)
                      if frame == 7)
    sample_npz = dataset[(item_index, 123)]
    assert dataset._episodes[0][2].dtype == np.float16

    prepare_mmap_cache(dataset.rows, cache_root)
    dataset._episodes.clear()
    sample_mmap = dataset[(item_index, 123)]
    assert isinstance(dataset._episodes[0][2], np.memmap)
    assert sample_npz["tactile"].dtype == np.float32
    assert sample_npz["future_tactile"].dtype == np.float32
    for key, expected_value in sample_npz.items():
        if isinstance(expected_value, np.ndarray):
            np.testing.assert_array_equal(sample_mmap[key], expected_value, err_msg=key)
        else:
            assert sample_mmap[key] == expected_value
