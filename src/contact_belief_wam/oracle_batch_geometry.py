"""Native mesh clearance for oracle rotation, including edge-only contact.

This is intentionally separate from the cheap per-observation contact proxy.
The result is clipped to ``clearance_m``: returning that value certifies at
least that separation; smaller values are measured surface distances. A
bounded-work fallback returns zero, so uncertainty never permits rotation.
The helper is deterministic and neither steps physics nor changes RNG state.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _bounds_distance(a_min, a_max, b_min, b_max):
    gap = np.maximum(np.maximum(a_min - b_max, b_min - a_max), 0.)
    return np.linalg.norm(gap, axis=-1)


def _point_segment_sq(points, start, end):
    edge = end-start
    norm = np.sum(edge*edge, axis=-1)
    t = np.sum((points-start)*edge, axis=-1) / np.maximum(norm, 1e-30)
    delta = points-start-np.clip(t, 0., 1.)[..., None]*edge
    return np.sum(delta*delta, axis=-1)


def _point_triangle_sq(points, triangle):
    """Broadcastable paired point/triangle squared distances."""
    a, b, c = (triangle[..., i, :] for i in range(3))
    ab, ac, ap = b-a, c-a, points-a
    d00, d01, d11 = (np.sum(x*y, axis=-1) for x, y in ((ab, ab), (ab, ac), (ac, ac)))
    d20, d21 = np.sum(ap*ab, axis=-1), np.sum(ap*ac, axis=-1)
    denom = d00*d11-d01*d01
    v = (d11*d20-d01*d21) / np.maximum(denom, 1e-30)
    w = (d00*d21-d01*d20) / np.maximum(denom, 1e-30)
    inside = (denom > 1e-28) & (v >= 0) & (w >= 0) & (v+w <= 1)
    delta = ap-v[..., None]*ab-w[..., None]*ac
    best = np.where(inside, np.sum(delta*delta, axis=-1), np.inf)
    for x, y in ((a, b), (b, c), (c, a)):
        best = np.minimum(best, _point_segment_sq(points, x, y))
    return best


def _segment_segment_sq(a0, a1, b0, b1):
    """Interior line solution plus all four endpoint/segment boundaries."""
    u, v, w = a1-a0, b1-b0, a0-b0
    aa, bb, cc, dd, ee = (np.sum(x*y, axis=-1) for x, y in
                          ((u, u), (u, v), (v, v), (u, w), (v, w)))
    den = aa*cc-bb*bb
    safe = np.maximum(den, 1e-30)
    s, t = (bb*ee-cc*dd)/safe, (aa*ee-bb*dd)/safe
    interior = (den > 1e-14*aa*cc) & (aa > 1e-30) & (cc > 1e-30) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    delta = w+s[..., None]*u-t[..., None]*v
    best = np.where(interior, np.sum(delta*delta, axis=-1), np.inf)
    for point, begin, end in ((a0, b0, b1), (a1, b0, b1), (b0, a0, a1), (b1, a0, a1)):
        best = np.minimum(best, _point_segment_sq(point, begin, end))
    return best


def _segment_triangle_intersects(start, end, tri):
    """Paired Möller–Trumbore intersections; coplanar pairs use distances."""
    direction = end-start
    edge1, edge2 = tri[..., 1, :]-tri[..., 0, :], tri[..., 2, :]-tri[..., 0, :]
    h = np.cross(direction, edge2)
    det = np.sum(edge1*h, axis=-1)
    scale = np.linalg.norm(direction, axis=-1)*np.linalg.norm(edge1, axis=-1)*np.linalg.norm(edge2, axis=-1)
    nonsingular = np.abs(det) > np.maximum(1e-30, scale*1e-12)
    inv = np.divide(1., det, out=np.zeros_like(det), where=nonsingular)
    s = start-tri[..., 0, :]
    u = inv*np.sum(s*h, axis=-1)
    q = np.cross(s, edge1)
    v = inv*np.sum(direction*q, axis=-1)
    t = inv*np.sum(edge2*q, axis=-1)
    eps = 1e-10
    return nonsingular & (u >= -eps) & (v >= -eps) & (u+v <= 1+eps) & (t >= -eps) & (t <= 1+eps)


def triangle_pair_distances(first, second):
    """Exact unsigned distances for corresponding triangle pairs [N,3,3]."""
    a, b = np.broadcast_arrays(np.asarray(first, float), np.asarray(second, float))
    if a.ndim != 3 or a.shape[1:] != (3, 3):
        raise ValueError("Expected paired triangles [N,3,3]")
    best = np.full(len(a), np.inf)
    for i in range(3):
        best = np.minimum(best, _point_triangle_sq(a[:, i], b))
        best = np.minimum(best, _point_triangle_sq(b[:, i], a))
        crosses = (_segment_triangle_intersects(a[:, i], a[:, (i+1) % 3], b)
                   | _segment_triangle_intersects(b[:, i], b[:, (i+1) % 3], a))
        best[crosses] = 0.
        for j in range(3):
            best = np.minimum(best, _segment_segment_sq(a[:, i], a[:, (i+1) % 3],
                                                       b[:, j], b[:, (j+1) % 3]))
    return np.sqrt(np.maximum(best, 0.))


def _mesh(row, topology, name):
    vertices = np.asarray(row[f"{name}_vertices_world"], dtype=float)
    indices = np.asarray(topology[name])
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices)
            or not np.isfinite(vertices).all() or indices.ndim != 2 or indices.shape[1:] != (3,)
            or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
            or indices.min() < 0 or indices.max() >= len(vertices)):
        raise ValueError(f"Invalid native {name} clearance mesh")
    return vertices[indices]


def clearance_distance(row, topology, *, clearance_m=.0005, pair_batch=128,
                       max_candidate_pairs=1_000_000):
    """Return min(native triangle-surface separation, requested clearance).

    Mesh AABBs and a centroid/radius tree eliminate only provably distant
    pairs. Triangle checks include edge/face intersections and edge/edge
    distances, absent from the old vertex/triangle contact proxy. Degenerate
    triangles reduce to edges/points. Geometry that exhausts the work budget
    is conservatively treated as uncleared (zero); no optimistic fallback.
    """
    if not np.isfinite(clearance_m) or clearance_m <= 0 or pair_batch < 1 or max_candidate_pairs < 1:
        raise ValueError("Invalid native clearance query limits")
    a, b = (_mesh(row, topology, name) for name in ("prism", "slot"))
    alo, ahi, blo, bhi = a.min((0, 1)), a.max((0, 1)), b.min((0, 1)), b.max((0, 1))
    if _bounds_distance(alo, ahi, blo, bhi) >= clearance_m:
        return float(clearance_m)
    # Discard triangles outside the other entire mesh's clearance envelope.
    a = a[_bounds_distance(a.min(1), a.max(1), blo, bhi) <= clearance_m]
    b = b[_bounds_distance(b.min(1), b.max(1), alo, ahi) <= clearance_m]
    if not len(a) or not len(b):
        return float(clearance_m)
    if len(a) > len(b):
        a, b = b, a
    bc = b.mean(1)
    br = np.linalg.norm(b-bc[:, None, :], axis=-1).max(1)
    tree = cKDTree(bc)
    bmin, bmax, max_radius = b.min(1), b.max(1), br.max()
    best, checked = float(clearance_m), 0
    for tri in a:
        center = tri.mean(0)
        radius = np.linalg.norm(tri-center, axis=-1).max()
        # One query at a time bounds candidate-list storage by one mesh size.
        candidates = np.asarray(tree.query_ball_point(center, radius+max_radius+best+1e-12), dtype=np.int64)
        if not len(candidates):
            continue
        bounds = _bounds_distance(tri.min(0), tri.max(0), bmin[candidates], bmax[candidates])
        candidates = candidates[bounds <= best+1e-12]
        checked += len(candidates)
        if checked > max_candidate_pairs:
            return 0.
        for offset in range(0, len(candidates), pair_batch):
            selected = candidates[offset:offset+pair_batch]
            distances = triangle_pair_distances(tri[None], b[selected])
            if not np.isfinite(distances).all():
                return 0.
            best = min(best, float(distances.min()))
            if best <= 1e-12:
                return 0.
    return best


# Name used by the native worker; kept explicit to document the certificate.
surface_clearance = clearance_distance
