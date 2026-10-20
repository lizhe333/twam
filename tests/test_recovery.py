import numpy as np

from contact_belief_wam.config import RecoveryConfig
from contact_belief_wam.recovery import TactileBlockageDetector, insertion_progress


def test_tilted_hole_does_not_count_transverse_sliding_as_insertion():
    axis = np.array([0.5, 0, -np.sqrt(0.75)])
    normal = np.array([0, 0, 1])
    # A stalled part pivots sideways along the entrance while the hand advances.
    slip = np.array([0.001, 0, 0])
    assert float(slip @ axis) == 0.0005
    assert insertion_progress(slip, axis, normal) == 0
    assert np.isclose(insertion_progress(axis * 0.0005, axis, normal), 0.0005)


def test_vertical_insertion_progress_keeps_hdmi_convention():
    assert insertion_progress([0, 0, -0.0005], [0, 0, -1], [0, 0, 1]) == 0.0005


def _touch_image(value, height=2, width=3):
    return np.full((2, height, width, 3), value, dtype=np.uint8)


def test_detector_uses_floor_for_stable_baseline():
    cfg = RecoveryConfig(tactile_threshold_floor=0.05)
    baseline = np.stack([_touch_image(40) for _ in range(cfg.stable_window_steps)])
    detector = TactileBlockageDetector(cfg, baseline)

    assert detector.threshold == 0.05
    assert detector.score(_touch_image(40)) == 0.0
    assert detector.score(_touch_image(255)) > detector.threshold


def test_detector_subtracts_recent_baseline_mean():
    cfg = RecoveryConfig(tactile_threshold_floor=0.001, tactile_threshold_std_scale=2.0)
    baseline = np.stack([_touch_image(0), _touch_image(10), _touch_image(20)])
    detector = TactileBlockageDetector(cfg, baseline)
    raw = [detector.raw_score(frame) for frame in baseline]
    expected = max(cfg.tactile_threshold_floor, 2.0 * np.std(raw))

    assert np.isclose(detector.threshold, expected)
    assert np.isclose(detector.baseline_mean, np.mean(raw))
    assert detector.score(_touch_image(10)) == 0


def test_detector_requires_consecutive_tactile_change_and_low_progress():
    cfg = RecoveryConfig(
        tactile_threshold_floor=0.2,
        blocked_persistence_steps=3,
        progress_ratio_threshold=0.2,
    )
    baseline = np.stack([_touch_image(0) for _ in range(cfg.stable_window_steps)])
    detector = TactileBlockageDetector(cfg, baseline)
    high = _touch_image(255)
    low = _touch_image(0)

    detector.observe(high)
    detector.observe(high)
    assert not detector.blocked(actual_progress=0.0, requested_progress=0.001)

    detector.observe(low)  # A low reading breaks persistence in the sliding window.
    assert not detector.blocked(actual_progress=0.0, requested_progress=0.001)
    detector.observe(high)
    detector.observe(high)
    assert not detector.blocked(actual_progress=0.0, requested_progress=0.001)
    detector.observe(high)

    # Progress equal to the configured cutoff is not considered blocked; less is.
    assert not detector.blocked(actual_progress=0.0002, requested_progress=0.001)
    assert detector.blocked(actual_progress=0.00019, requested_progress=0.001)
