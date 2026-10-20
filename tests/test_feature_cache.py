import json
import os
from pathlib import Path

import numpy as np
import pytest

from contact_belief_wam.feature_cache import load_tactile_tokens, prepare_mmap_cache


def _write_source(cache_root, episode_id="original/task/0", frames=3, value=0):
    tokens_path = cache_root / f"{episode_id}.npz"
    tokens_path.parent.mkdir(parents=True, exist_ok=True)
    tokens = (np.arange(frames * 8 * 512, dtype=np.int32).reshape(frames, 8, 512) % 97)
    tokens = (tokens.astype(np.float32) / 97 + value).astype(np.float16)
    steps = (100 + 2 * np.arange(frames)).astype(np.int64)
    np.savez_compressed(tokens_path, tokens=tokens, raw_step=steps)
    row = {"episode_id": episode_id, "frames": frames}
    return row, tokens_path, tokens, steps


def test_prepare_companion_is_float16_memmap_and_reused(tmp_path, monkeypatch):
    row, source_path, expected_tokens, steps = _write_source(tmp_path)

    first = prepare_mmap_cache([row], tmp_path)
    assert first["converted"] == 1
    assert first["reused"] == 0
    assert first["converted_frames"] == len(steps)

    base = tmp_path / row["episode_id"]
    tokens_path = Path(f"{base}.tokens.npy")
    metadata_path = Path(f"{base}.tokens.json")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["source_npz_size"] == source_path.stat().st_size
    assert metadata["source_npz_mtime_ns"] == source_path.stat().st_mtime_ns
    assert metadata["shape"] == list(expected_tokens.shape)
    assert metadata["dtype"] == "float16"
    assert metadata["raw_step"] == steps.tolist()

    original_load = np.load

    def reject_npz_load(path, *args, **kwargs):
        if str(path).endswith(".npz"):
            raise AssertionError("valid companion should not decompress the source NPZ")
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", reject_npz_load)
    loaded = load_tactile_tokens(tmp_path, row["episode_id"], steps)
    assert isinstance(loaded, np.memmap)
    assert loaded.dtype == np.float16
    np.testing.assert_array_equal(loaded, expected_tokens)

    monkeypatch.setattr(np, "load", original_load)
    tokens_mtime = tokens_path.stat().st_mtime_ns
    second = prepare_mmap_cache([row], tmp_path)
    assert second["converted"] == 0
    assert second["reused"] == 1
    assert tokens_path.stat().st_mtime_ns == tokens_mtime


def test_changed_source_npz_invalidates_companion_and_falls_back(tmp_path):
    row, source_path, _, steps = _write_source(tmp_path)
    prepare_mmap_cache([row], tmp_path)

    new_steps = steps + 1000
    new_tokens = np.full((len(steps), 8, 512), 0.75, dtype=np.float16)
    np.savez_compressed(source_path, tokens=new_tokens, raw_step=new_steps)
    stat = source_path.stat()
    os.utime(source_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    with pytest.warns(RuntimeWarning, match="source_npz_stat_mismatch"):
        loaded = load_tactile_tokens(tmp_path, row["episode_id"], new_steps)
    assert not isinstance(loaded, np.memmap)
    assert loaded.dtype == np.float16
    np.testing.assert_array_equal(loaded, new_tokens)


def test_mismatched_expected_steps_never_returns_stale_features(tmp_path):
    row, _, _, steps = _write_source(tmp_path)
    prepare_mmap_cache([row], tmp_path)
    incorrect_steps = steps.copy()
    incorrect_steps[-1] += 1

    with pytest.warns(RuntimeWarning, match="raw_step_mismatch"):
        with pytest.raises(ValueError, match="raw_step"):
            load_tactile_tokens(tmp_path, row["episode_id"], incorrect_steps)


def test_corrupt_companion_dtype_falls_back_to_source_npz(tmp_path):
    row, _, expected_tokens, steps = _write_source(tmp_path)
    prepare_mmap_cache([row], tmp_path)
    tokens_path = Path(f"{tmp_path / row['episode_id']}.tokens.npy")
    np.save(tokens_path, expected_tokens.astype(np.float32))

    with pytest.warns(RuntimeWarning, match="tokens_metadata_mismatch"):
        loaded = load_tactile_tokens(tmp_path, row["episode_id"], steps)
    assert not isinstance(loaded, np.memmap)
    assert loaded.dtype == np.float16
    np.testing.assert_array_equal(loaded, expected_tokens)


def test_missing_companion_uses_npz_without_warning(tmp_path):
    row, _, expected_tokens, steps = _write_source(tmp_path)
    loaded = load_tactile_tokens(tmp_path, row["episode_id"], steps)
    assert not isinstance(loaded, np.memmap)
    assert loaded.dtype == np.float16
    np.testing.assert_array_equal(loaded, expected_tokens)
