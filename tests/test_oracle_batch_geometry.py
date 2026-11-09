import random

import numpy as np
import pytest

from contact_belief_wam.oracle_batch_geometry import clearance_distance, triangle_pair_distances
from contact_belief_wam.oracle_batch_detector import surface_distance


def query(a, b, **kwargs):
    a, b = np.asarray(a, float).reshape(-1, 3, 3), np.asarray(b, float).reshape(-1, 3, 3)
    row = dict(prism_vertices_world=a.reshape(-1, 3), slot_vertices_world=b.reshape(-1, 3))
    topology = {"prism": np.arange(a.size//3).reshape(-1, 3), "slot": np.arange(b.size//3).reshape(-1, 3)}
    return clearance_distance(row, topology, **kwargs)


def test_edge_face_intersection_missed_by_previous_proximity_proxy():
    a = np.array([[-.01, -.01, 0], [.01, -.01, 0], [0, .01, 0]])
    b = np.array([[0, -.02, -.01], [0, -.02, .01], [0, .02, .005]])
    row = dict(prism_vertices_world=a, slot_vertices_world=b)
    topology = dict(prism=np.array([[0, 1, 2]]), slot=np.array([[0, 1, 2]]))
    assert surface_distance(row, topology) > .001
    assert clearance_distance(row, topology) == 0.
    assert query(b, a) == 0.


def test_crossed_edges_have_small_separation_without_close_vertices():
    a = [[-.01, 0, 0], [.01, 0, 0], [.01, -.01, 0]]
    b = [[0, -.01, .0002], [0, .01, .0002], [-.01, .01, .01]]
    assert query(a, b) == pytest.approx(.0002)
    assert query(b, a) == pytest.approx(.0002)


@pytest.mark.parametrize("gap", [0., .0002, .0005, .001])
def test_parallel_faces_exact_within_band_and_saturated_above(gap):
    a = np.array([[-.01, -.01, 0], [.01, -.01, 0], [0, .01, 0]])
    b = a + [0, 0, gap]
    assert query(a, b) == pytest.approx(min(gap, .0005), abs=1e-12)


def test_degenerate_point_and_edge_meshes_do_not_create_false_clearance():
    a = [[0, 0, 0]]*3
    b = [[-.01, 0, 0], [.01, 0, 0], [.01, 0, 0]]
    assert query(a, b) == 0.
    assert query(a, np.asarray(b) + [0, .0001, 0]) == pytest.approx(.0001)


def test_coplanar_overlapping_faces_and_shared_edges():
    a = np.array([[0, 0, 0], [.01, 0, 0], [0, .01, 0]])
    assert query(a, a + [.001, .001, 0]) == 0.
    assert query(a, [[0, 0, 0], [.01, 0, 0], [0, -.01, 0]]) == 0.


def test_work_budget_exhaustion_is_conservatively_blocked():
    a = np.array([[-.01, -.01, 0], [.01, -.01, 0], [0, .01, 0]])
    b = np.stack([a + [0, 0, .0002], a + [0, 0, .0003]])
    assert query(a, b, max_candidate_pairs=1) == 0.


def test_spatially_rejected_faces_do_not_consume_pair_budget():
    a = np.array([[-.01, -.01, 0], [.01, -.01, 0], [0, .01, 0]])
    distant = np.stack([a + [1+i*.1, 0, 0] for i in range(100)])
    b = np.concatenate([(a + [0, 0, .0002])[None], distant])
    assert query(a, b, max_candidate_pairs=1) == pytest.approx(.0002)


def test_invalid_geometry_is_never_certified_clear():
    row = dict(prism_vertices_world=np.zeros((3, 3)), slot_vertices_world=np.zeros((3, 3)))
    bad = dict(prism=np.array([[0, 1, 5]]), slot=np.array([[0, 1, 2]]))
    with pytest.raises(ValueError, match="mesh"):
        clearance_distance(row, bad)
    row["prism_vertices_world"][0, 0] = np.nan
    with pytest.raises(ValueError, match="mesh"):
        clearance_distance(row, dict(prism=np.array([[0, 1, 2]]), slot=np.array([[0, 1, 2]])))


def test_batching_and_triangle_order_do_not_change_result_or_rng():
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=(13, 3, 3))*.001, rng.normal(size=(17, 3, 3))*.001
    state = random.getstate()
    numpy_state = np.random.get_state()
    first = query(a, b, pair_batch=1)
    second = query(a[::-1], b[::-1], pair_batch=8)
    assert first == pytest.approx(second, abs=1e-12)
    assert random.getstate() == state
    for before, after in zip(numpy_state, np.random.get_state()):
        np.testing.assert_equal(before, after)


def test_pair_distance_is_symmetric_and_translation_invariant():
    rng = np.random.default_rng(33)
    a, b = rng.normal(size=(32, 3, 3))*.01, rng.normal(size=(32, 3, 3))*.01
    distance = triangle_pair_distances(a, b)
    np.testing.assert_allclose(distance, triangle_pair_distances(b, a), atol=1e-12)
    np.testing.assert_allclose(distance, triangle_pair_distances(a+[.5, .2, -.1], b+[.5, .2, -.1]), atol=1e-12)
