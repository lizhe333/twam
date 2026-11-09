import numpy as np

from contact_belief_wam.pixel_diagnostic import depth_difference, cosine_distance


def test_depth_rmse_does_not_hide_single_far_clip_pixel():
    original = np.full((240, 320), 28., np.float32)
    replay = original.copy()
    replay[104, 90] = 34.
    metric = depth_difference(original, replay)
    assert metric["rmse_mm"] > .02
    assert metric["outlier_count_over_0_01mm"] == 1
    assert metric["all_outliers_are_far_clip_switches"]
    assert metric["non_clip_rmse_mm"] == 0
    assert metric["far_clip_switches"][0]["x"] == 90


def test_non_clip_geometry_error_is_not_classified_as_clipping_noise():
    original = np.full((240, 320), 28., np.float32)
    replay = original + .05
    metric = depth_difference(original, replay)
    assert not metric["all_outliers_are_far_clip_switches"]
    assert metric["outlier_count_over_0_01mm"] == 240 * 320
    assert metric["non_clip_rmse_mm"] > .04


def test_cosine_comparison_keeps_temporal_axis_and_detects_changed_feature():
    a = np.ones((11, 8, 512))
    b = a.copy()
    b[2] *= -1
    distance = cosine_distance(a, b)
    assert distance.shape == (11,)
    np.testing.assert_allclose(distance[[0, 1, 3]], 0, atol=1e-12)
    np.testing.assert_allclose(distance[2], 2, atol=1e-12)
