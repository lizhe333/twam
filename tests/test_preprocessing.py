import h5py
import numpy as np

from contact_belief_wam.preprocessing import Quantiles, TOUCH_OFFSETS, window_indices
from contact_belief_wam.tactile import cache_episodes


def test_four_frame_offsets_clamp_at_each_segment_start():
    assert TOUCH_OFFSETS == (-6, -4, -2, 0)
    np.testing.assert_array_equal(window_indices(2, segment_start=0), [0, 0, 0, 2])
    np.testing.assert_array_equal(window_indices(5, segment_start=5), [5, 5, 5, 5])
    np.testing.assert_array_equal(window_indices(12, segment_start=8), [8, 8, 10, 12])


def test_tactile_cache_uses_four_frames_without_crossing_segment_boundaries(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    hdf5_path = data_root / "episode.hdf5"
    steps = np.array([0, 2, 4, 100, 102], dtype=np.int64)
    with h5py.File(hdf5_path, "w") as handle:
        handle.create_dataset("step", data=steps)
        for side, bias in (("left", 0), ("right", 40)):
            group = handle.require_group(f"tactile/{side}_gsmini")
            images = np.stack([
                np.full((1, 1, 3), bias + frame, dtype=np.uint8)
                for frame in range(len(steps))
            ])
            group.create_dataset("rgb_marker", data=images)

    class RecordingEncoder:
        metadata = {"test_encoder": "recording"}

        def __init__(self):
            self.windows = []

        def __call__(self, batch):
            self.windows.append(batch.copy())
            return np.zeros((len(batch), 8, 512), dtype=np.float32)

    encoder = RecordingEncoder()
    row = {
        "episode_id": "original/insert_hole/segment-test",
        "path": "episode.hdf5",
        "frames": len(steps),
        "segments": [[0, 3], [3, 5]],
    }
    cache_root = tmp_path / "cache"
    cache_episodes([row], data_root, cache_root, encoder, batch_size=2)

    captured = np.concatenate(encoder.windows, axis=0)
    expected = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 2],
        [3, 3, 3, 3],
        [3, 3, 3, 4],
    ])
    np.testing.assert_array_equal(captured[:, 0, :, 0, 0, 0], expected)
    np.testing.assert_array_equal(captured[:, 1, :, 0, 0, 0], expected + 40)
    with np.load(cache_root / f"{row['episode_id']}.npz") as saved:
        assert saved["tokens"].shape == (5, 8, 512)
        np.testing.assert_array_equal(saved["raw_step"], steps)


def test_quantiles_keep_constant_dimensions_finite_and_round_trip():
    stats = {
        "action": {
            "q01": [5.0, 0.0, 1.0],
            "q99": [5.0, 10.0, 3.0],
        }
    }
    quantiles = Quantiles(stats)
    raw = np.array([[5.0, 0.0, 1.0], [5.0, 5.0, 2.0], [5.0, 10.0, 3.0]])
    normalized = quantiles.normalize(raw, "action")

    assert normalized.dtype == np.float32
    assert np.isfinite(normalized).all()
    np.testing.assert_array_equal(normalized[:, 0], [-1.0, -1.0, -1.0])
    np.testing.assert_allclose(normalized[:, 1:], [[-1, -1], [0, 0], [1, 1]])
    np.testing.assert_allclose(quantiles.unnormalize(normalized), raw, atol=1e-6)
