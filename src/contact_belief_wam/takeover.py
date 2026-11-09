"""Causal, simulator-independent contracts for the HDMI takeover pilot.

Indices are observation indices: observation t precedes command t. Neither
the detector nor the feedback rule accepts object/slot ground-truth poses.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np

from .paths import atomic_json
from .preprocessing import TOUCH_OFFSETS

VERSION = "wave0-takeover-v1"


@dataclass(frozen=True)
class TakeoverConfig:
    seed_start: int = 3000000
    natural_limit: int = 20
    recovery_limit: int = 10
    recovery_target: int = 3
    normal_target: int = 2
    budget: int = 600
    execute_steps: int = 10
    physics_steps: int = 2
    baseline_observations: int = 8
    tactile_persistence_observations: int = 3
    tactile_floor: float = .01
    tactile_std_scale: float = 3.
    command_threshold_mm: float = .2
    blocked_progress_ratio: float = .2
    improving_progress_ratio: float = .5
    retreat_mm: float = 2.
    translation_mm: float = .5
    rotation_deg: float = 1.
    lateral_cap_mm: float = 3.
    rotation_cap_deg: float = 3.
    probe_limit: int = 12

    def __post_init__(self):
        if (self.execute_steps, self.physics_steps, self.budget) != (10, 2, 600):
            raise ValueError("Pilot execution contract is 10 / 2 / 600")
        if not (1 <= self.natural_limit <= 20 and 1 <= self.recovery_limit <= 10):
            raise ValueError("Pilot limits cannot exceed 20 natural / 10 takeover attempts")
        if (self.recovery_target, self.normal_target) != (3, 2):
            raise ValueError("Expansion needs a separately reviewed collection, not a pilot override")
        if self.seed_start < 3000000 or self.probe_limit > 12:
            raise ValueError("Invalid pilot seed range or probe limit")


class BlockageDetector:
    """Consume one observation after each actual strategy action, never futures."""
    def __init__(self, config=TakeoverConfig()):
        self.cfg = config
        self.baseline = []
        self.reference = None
        self.scores, self.positions, self.commands = [], [], []
        self.events = []
        self.first_confirmed = None
        self.candidate = None

    def observe(self, tactile, fk_position, command_fk=None):
        cfg = self.cfg
        rgb = np.asarray(tactile, np.float32)
        pos = np.asarray(fk_position, float)
        if rgb.ndim != 4 or rgb.shape[0] != 2 or rgb.shape[-1] != 3:
            raise ValueError("Expected both tactile RGB observations [2,H,W,3]")
        if pos.shape != (3,) or not np.isfinite(pos).all() or not np.isfinite(rgb).all():
            raise ValueError("Missing finite tactile/FK observation")
        t = len(self.positions)
        if t and command_fk is None:
            raise ValueError("Every post-action observation needs its executed command FK")
        if command_fk is not None:
            cmd = np.asarray(command_fk, float)
            if cmd.shape != (3,) or not np.isfinite(cmd).all():
                raise ValueError("Invalid executed command FK")
            self.commands.append(cmd)
        self.positions.append(pos.copy())
        if self.reference is None:
            self.baseline.append(rgb.copy())
            if len(self.baseline) == cfg.baseline_observations:
                self.reference = np.mean(self.baseline, axis=0)
                noise = [self._raw(x) for x in self.baseline]
                self.noise_mean = float(np.mean(noise))
                self.threshold = max(cfg.tactile_floor, cfg.tactile_std_scale * float(np.std(noise)))
                self.baseline.clear()
            score = 0.
        else:
            score = max(0., self._raw(rgb) - self.noise_mean)
        self.scores.append(score)
        if not t or t % cfg.execute_steps:
            return None
        start = t - cfg.execute_steps
        # HDMI's native insertion direction is world -Z. No object pose is used.
        inward_commands = (self.positions[start][2] - np.asarray(self.commands[start:t])[:, 2]) * 1000
        requested = max(0., float(inward_commands.max()))
        achieved = max(0., float(self.positions[start][2] - pos[2]) * 1000)
        ratio = achieved / max(requested, 1e-9)
        persistent_push = bool(np.all(inward_commands[-cfg.tactile_persistence_observations:] >= cfg.command_threshold_mm))
        stalled = persistent_push and ratio < cfg.blocked_progress_ratio
        tactile_present = self.reference is not None and min(self.scores[-cfg.tactile_persistence_observations:]) >= self.threshold
        event = dict(step=t, requested_mm=requested, observed_ee_progress_mm=achieved,
                     progress_ratio=ratio, tactile_score=score, tactile_threshold=getattr(self, "threshold", None),
                     tactile_sustained=tactile_present, persistent_push=persistent_push, status="normal")
        if stalled:
            self.candidate = t if self.candidate is None else self.candidate
            age = (t - self.candidate) // cfg.execute_steps + 1
            event.update(status="candidate", candidate_step=self.candidate, candidate_chunks=age)
            if tactile_present and age <= 2:
                event["status"] = "confirmed"
                if self.first_confirmed is None:
                    self.first_confirmed = t
            elif age >= 2:
                event["status"] = "unconfirmed"
        else:
            self.candidate = None
        self.events.append(event)
        return event

    def _raw(self, rgb):
        return float(np.abs(rgb - self.reference).mean(axis=(1, 2, 3)).max() / 255.)

    def report(self):
        return dict(first_confirmed=self.first_confirmed, events=self.events, scores=self.scores,
                    tactile_threshold=getattr(self, "threshold", None),
                    calibration_observations=self.cfg.baseline_observations,
                    progress_source="observed_joint_FK_world_minus_z_not_object_insertion")


def feedback_decision(*, requested_mm, achieved_mm, touch_before, touch_after,
                      threshold, grip_stable, repeated=False, config=TakeoverConfig()):
    """Identical observable feedback always produces the same next decision."""
    values = [requested_mm, achieved_mm, touch_before, touch_after, threshold]
    if not np.isfinite(values).all() or requested_mm <= 0 or not grip_stable:
        return "abort", "invalid_feedback_or_unstable_grasp"
    ratio = achieved_mm / requested_mm
    if ratio >= config.improving_progress_ratio:
        return "continue", "progress_improved_with_stable_grasp"
    if ratio < config.blocked_progress_ratio and touch_after >= threshold:
        return "withdraw", "persistent_stall_with_contact_evidence"
    if repeated:
        return "withdraw", "ambiguous_after_one_repeat"
    return "repeat", "insufficient_evidence_one_bounded_repeat"


def tactile_indices(step):
    return np.maximum(0, int(step) + np.asarray(TOUCH_OFFSETS)).astype(np.int64)


def policy_prefix(commands, anchor, ranges, horizon=50):
    """Remove forbidden future values BEFORE constructing flow inputs.

Padding repeats the last permitted target (or current action if wholly masked),
never the hidden future. Context/WM indices are deliberately not altered.
"""
    commands = np.asarray(commands, np.float32)
    if commands.ndim != 2 or commands.shape[1] != 8 or not 0 <= anchor < len(commands):
        raise ValueError("Invalid absolute command sequence or anchor")
    stop = anchor
    for row in ranges:
        if row["start"] <= anchor < row["stop"]:
            stop = min(row["stop"], anchor + horizon, len(commands))
            break
    count = stop - anchor
    # For a wholly masked anchor do not expose even its excluded command.
    padding = commands[stop - 1] if count else np.zeros(8, np.float32)
    result = np.repeat(padding[None], horizon, axis=0)
    result[:count] = commands[anchor:stop]
    return result, np.arange(horizon) < count


def supervision_ranges(action_metadata, *, normal_success=False, recovery_success=False):
    if normal_success:
        return [dict(start=0, stop=len(action_metadata), reason="normal_success")]
    if not recovery_success:
        return []
    ranges = []
    for i, row in enumerate(action_metadata):
        if not row.get("policy_eligible", False):
            continue
        # A new feedback decision always opens a new range, even if adjacent.
        key = row["decision_step"]
        if ranges and ranges[-1]["decision_step"] == key and ranges[-1]["stop"] == i:
            ranges[-1]["stop"] = i + 1
        else:
            ranges.append(dict(start=i, stop=i + 1, decision_step=key, reason=row["phase"]))
    return ranges


def finalize_supervision(action_metadata, *, normal_success=False, recovery_success=False):
    """Keep per-action flags and exported ranges consistent after the outcome."""
    for row in action_metadata:
        row["policy_eligible"] = bool(normal_success or (recovery_success and row.get("policy_eligible", False)))
    return supervision_ranges(action_metadata, normal_success=normal_success, recovery_success=recovery_success)


def relative_grasp_pose(row):
    from scipy.spatial.transform import Rotation
    grip, obj = [np.asarray(row[k], float) for k in ("gripper_pose_world", "prism_pose_world")]
    r = Rotation.from_quat(grip[[4, 5, 6, 3]])
    ro = Rotation.from_quat(obj[[4, 5, 6, 3]])
    quat = (r.inv() * ro).as_quat()
    return np.r_[r.inv().apply(obj[:3] - grip[:3]), quat[3], quat[:3]]


def compare_replay(reference, replay, reference_actions, replay_actions):
    from pi05_univtac.contact_restore import compare_observation_sequences
    def with_grasp(rows):
        return [dict(row, object_in_gripper_pose_world=relative_grasp_pose(row)) for row in rows]
    return compare_observation_sequences(with_grasp(reference), with_grasp(replay),
        reference_actions=reference_actions, restored_actions=replay_actions,
        require_contact_trace_fields=True)


def seed_overlap(roots, candidates):
    """Audit explicit seed fields and the repository's numeric episode filenames."""
    candidates = set(map(int, candidates))
    conflicts = []
    checked = 0
    hdf_checked = 0
    def walk(value):
        if isinstance(value, dict):
            for k, v in value.items():
                if k in {"path", "episode_id"} and isinstance(v, str) and Path(v).stem.isdigit():
                    seed = int(Path(v).stem)
                    if seed in candidates:
                        yield seed
                if k in {"seed", "start_seed", "seed_start"} and isinstance(v, int) and v in candidates:
                    yield v
                if k in {"seeds", "evaluation_seeds"} and isinstance(v, list):
                    if k == "evaluation_seeds" and len(v) == 2 and all(isinstance(x, int) for x in v):
                        yield from (s for s in candidates if v[0] <= s <= v[1])
                    else:
                        yield from (s for s in v if isinstance(s, int) and s in candidates)
                if k in {"start_seed", "seed_start"} and isinstance(v, int):
                    count = value.get("episodes", value.get("total_num", value.get("natural_limit", 1)))
                    if isinstance(count, int):
                        yield from (s for s in candidates if v <= s < v + count)
                yield from walk(v)
        elif isinstance(value, list):
            for row in value:
                yield from walk(row)
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        hdf_paths = ([root] if root.is_file() and root.suffix in {".h5", ".hdf5"}
                     else sorted(set(root.rglob("*.h5")) | set(root.rglob("*.hdf5"))) if root.is_dir() else [])
        for path in hdf_paths:
            hdf_checked += 1
            if path.stem.isdigit():
                seeds = [int(path.stem)]
            else:
                import h5py
                with h5py.File(path, "r") as handle:
                    seeds = [int(handle.attrs[key]) for key in ("seed", "episode_seed") if key in handle.attrs]
                if not seeds:
                    raise ValueError(f"No verifiable seed in existing HDF episode: {path}")
            found = sorted(set(seeds) & candidates)
            if found:
                conflicts.append(dict(path=str(path), seeds=found))
        if root.is_file() and root.suffix in {".h5", ".hdf5"}:
            continue
        paths = [root] if root.is_file() else sorted(set(root.rglob("*.jsonl")) | set(root.rglob("manifest.json")) | set(root.rglob("collection.json")))
        for path in paths:
            if path.stat().st_size > 100_000_000:
                raise ValueError(f"Manifest too large to audit safely: {path}")
            checked += 1
            rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip()) if path.suffix == ".jsonl" else [json.loads(path.read_text())]
            found = sorted({s for row in rows for s in walk(row)})
            if found:
                conflicts.append(dict(path=str(path), seeds=found))
    if not checked and not hdf_checked:
        raise ValueError("No existing dataset/evaluation manifests were audited")
    return dict(manifests_checked=checked, hdf_episodes_checked=hdf_checked, conflicts=conflicts)


class PilotLedger:
    """Exclusive, crash-safe attempt reservations. Interrupted attempts count."""
    def __init__(self, root, config, identity):
        self.root, self.cfg = Path(root), config
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / "pilot.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.path = self.root / "pilot.json"
        self.identity = dict(version=VERSION, config=asdict(config), source=identity)
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state["identity"] != self.identity:
                self.close()
                raise ValueError("Pilot identity/config changed; existing quotas and review cannot be reset")
            for row in self.state["attempts"]:
                if row["status"] == "running":
                    row.update(status="interrupted", qualified=False)
            if self.state.get("error"):
                self.state.setdefault("previous_errors", []).append(self.state.pop("error"))
        else:
            self.state = dict(identity=self.identity, status="ready", attempts=[])
        self.save()

    def counts(self):
        rows = self.state["attempts"]
        return {kind: sum(r["kind"] == kind for r in rows) for kind in ("natural", "recovery")} | {
            "normal_success": sum(r["kind"] == "natural" and r.get("qualified", False) for r in rows),
            "recovery_success": sum(r["kind"] == "recovery" and r.get("qualified", False) for r in rows)}

    def target_reached(self):
        c = self.counts()
        return c["normal_success"] >= self.cfg.normal_target and c["recovery_success"] >= self.cfg.recovery_target

    def reserve(self, kind, seed, **details):
        return self._reserve(kind, seed, details, oracle_retry=False)

    def reserve_oracle(self, seed, **details):
        """A requested diagnostic retry consumes the shared recovery allowance."""
        if not any(r["kind"] == "natural" and r["seed"] == seed and r["status"] == "failed"
                   for r in self.state["attempts"]):
            raise ValueError("Oracle requires an existing completed natural failure")
        if any(r.get("controller") == "privileged_oracle" and r["seed"] == seed and r.get("success")
               for r in self.state["attempts"]):
            raise RuntimeError("Oracle preview already succeeded; stop for user review")
        details.update(controller="privileged_oracle", diagnostic_only=True, excluded_from_training=True)
        return self._reserve("recovery", seed, details, oracle_retry=True)

    def _reserve(self, kind, seed, details, *, oracle_retry):
        if self.state["status"] == "awaiting_user_review" or self.target_reached():
            raise RuntimeError("Pilot stopped for user review; automatic expansion is disabled")
        limit = {"natural": self.cfg.natural_limit, "recovery": self.cfg.recovery_limit}[kind]
        if self.counts()[kind] >= limit:
            raise RuntimeError(f"{kind} attempt quota exhausted")
        if not oracle_retry and any(r["kind"] == kind and r["seed"] == seed for r in self.state["attempts"]):
            raise RuntimeError("Attempt already consumed, including interrupted attempts")
        if not self.cfg.seed_start <= seed < self.cfg.seed_start + self.cfg.natural_limit:
            raise ValueError("Seed outside pilot reservation")
        row = dict(kind=kind, seed=seed, status="running", qualified=False, **details)
        self.state["attempts"].append(row)
        self.state["status"] = "collecting"
        self.save()
        return row

    def finish(self, row, result):
        row.update(result)
        if row["status"] == "running":
            raise ValueError("Attempt needs a terminal status")
        self.save()

    def save(self):
        self.state["counts"] = self.counts()
        atomic_json(self.path, self.state)
        # atomic_json replaces a temporary file. Flush the result before the
        # simulator starts, so process restarts never refund a reserved attempt.
        with self.path.open("rb") as handle:
            import os
            os.fsync(handle.fileno())

    def pause_for_review(self, reason):
        self.state.update(status="awaiting_user_review", stop_reason=reason)
        self.save()

    def close(self):
        if not self.lock.closed:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()


def fingerprint(paths):
    result = {}
    for path in map(Path, paths):
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        result[str(path.resolve())] = digest.hexdigest()
    return result
