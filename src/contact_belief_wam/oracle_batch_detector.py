"""Causal privileged stall detection for collecting natural failure recoveries.

The signed progress is native object motion along the insertion axis, never FK
motion mislabeled as insertion. No physics, rendering, or random sampling occurs.
"""
from collections import deque

import numpy as np
from scipy.spatial import cKDTree

from .takeover import tactile_indices
from .takeover_report import point_triangle_distance


def surface_distance(row, topology):
    """Native surface proximity in metres (conservative for edge-only contact)."""
    vertices = [np.asarray(row[f"{name}_vertices_world"]) for name in ("prism", "slot")]
    if not all(len(x) and np.isfinite(x).all() for x in vertices):
        raise ValueError("Missing or nonfinite native contact geometry")
    lower = np.maximum(vertices[0].min(0)-vertices[1].max(0),
                       vertices[1].min(0)-vertices[0].max(0))
    aabb = float(np.linalg.norm(np.maximum(lower, 0)))
    if aabb > .003:
        return aabb
    nearest = float(cKDTree(vertices[1]).query(vertices[0], k=1)[0].min())
    if nearest <= .0005:
        return nearest
    distances = [nearest]
    for j, name in enumerate(("prism", "slot")):
        triangles = vertices[j][topology[name]]
        other = vertices[1-j]
        selected = ((triangles.max(1) >= other.min(0)-.003) &
                    (triangles.min(1) <= other.max(0)+.003)).all(1)
        triangles = triangles[selected]
        if not len(triangles):
            continue
        points = other[((other >= vertices[j].min(0)-.003) &
                        (other <= vertices[j].max(0)+.003)).all(1)]
        for offset in range(0, len(points), 32):
            distances.append(point_triangle_distance(points[offset:offset+32], triangles))
    return min(distances)


class PrivilegedStallDetector:
    def __init__(self, insertion_axis, *, execute_steps=10, min_command_mm=.2,
                 progress_ratio=.2, tactile_persistence_observations=3, resumed_progress_mm=.04):
        self.axis = np.asarray(insertion_axis, float)
        if self.axis.shape != (3,) or not np.isfinite(self.axis).all() or not np.isclose(np.linalg.norm(self.axis), 1):
            raise ValueError("Insertion axis must be a unit world vector")
        self.execute_steps = int(execute_steps)
        self.min_command_mm, self.progress_ratio = float(min_command_mm), float(progress_ratio)
        self.persistence = int(tactile_persistence_observations)
        self.resumed_progress_mm = float(resumed_progress_mm)
        self.events, self.scores, self.contact, self.positions, self.commands = [], [], [], [], []
        self.history = deque(maxlen=8)
        self.reference, self.threshold = None, None
        self.t_contact = self.t_tactile = self.first_confirmed = None
        self.candidate = None
        self.previous_fk = None

    def observe(self, *, touch, object_position, observed_fk, command_fk=None, distance):
        t = len(self.positions)
        touch = np.asarray(touch, np.float32) / 255.
        # Full image changes are reduced per side, exactly as the pilot audit.
        in_contact = float(distance) <= .0005
        if in_contact and self.t_contact is None:
            self.t_contact = t
            if len(self.history) >= 3:
                frames = np.stack(self.history)
                self.reference = frames.mean(0)
                calibration = np.abs(frames-self.reference).mean(axis=tuple(range(2, frames.ndim))).max(1)
                self.threshold = float(calibration.mean()+max(.01, 3*calibration.std()))
        score = 0. if self.reference is None else float(np.abs(touch-self.reference).reshape(2, -1).mean(1).max())
        if self.reference is not None and in_contact and score >= self.threshold and self.t_tactile is None:
            self.t_tactile = t
        self.scores.append(score)
        self.contact.append(in_contact)
        self.positions.append(float(np.dot(object_position, self.axis))*1000)
        if command_fk is not None and self.previous_fk is not None:
            self.commands.append(max(0., float(np.dot(np.asarray(command_fk)[:3]-self.previous_fk[:3], self.axis))*1000))
        elif t:
            raise ValueError("Missing issued command at a dynamics transition")
        self.previous_fk = np.asarray(observed_fk).copy()
        self.history.append(touch.copy())
        if t == 0 or t % self.execute_steps:
            return None
        start = t-self.execute_steps
        commanded = sum(self.commands[start:t])
        progress = self.positions[t]-self.positions[start]
        pushing = commanded >= self.min_command_mm
        sustained = (self.threshold is not None and
                     all(self.contact[-self.persistence:]) and
                     all(x >= self.threshold for x in self.scores[-self.persistence:]))
        stalled = pushing and progress < self.progress_ratio*commanded
        status = "unconfirmed" if stalled else "no_stall"
        # Entry and release must use comparable motion scales. In particular,
        # 0.04 mm of creep under a <0.2 mm command does not establish that a
        # confirmed contact stall has cleared. The old asymmetric release rule
        # discarded valid early snapshots while the connector remained stuck.
        # A real advance (including passive slip with no new inward command)
        # still clears the candidate, so an earlier resolved stall is not used
        # to explain an unrelated later failure.
        resume_required = max(self.min_command_mm, self.resumed_progress_mm,
                              self.progress_ratio*commanded)
        if progress >= resume_required:
            self.candidate = self.first_confirmed = None
            status = "progress_resumed"
        elif stalled and sustained and self.candidate is None:
            self.first_confirmed = t
            self.candidate = dict(step=t, t_contact=self.t_contact, t_tactile=self.t_tactile,
                                  t_decision=t, input_tactile_indices=tactile_indices(t).tolist())
            status = "confirmed"
        elif self.candidate is not None:
            status = "persistent_candidate"
        event = dict(step=t, status=status, commanded_mm=commanded, object_progress_mm=progress,
                     resume_required_mm=resume_required,
                     progress_ratio=progress/max(commanded, 1e-12), tactile_sustained=bool(sustained),
                     contact_distance_m=float(distance), tactile_score=score,
                     tactile_threshold=self.threshold, candidate_step=self.first_confirmed)
        self.events.append(event)
        return event

    def report(self):
        return dict(ground_truth_used_online=True, progress_source="native_object_insertion_axis",
                    selection_version="symmetric-progress-release-v2",
                    first_confirmed=self.first_confirmed, candidate=self.candidate,
                    scores=self.scores, contact=self.contact, events=self.events,
                    tactile_threshold=self.threshold, tactile_persistence_observations=self.persistence)
