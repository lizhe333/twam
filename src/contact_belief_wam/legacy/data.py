from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..config import ContactConfig
from ..feature_cache import load_tactile_tokens
from ..paths import WORKSPACE, atomic_json, read_jsonl, write_jsonl
from ..preprocessing import Quantiles, decode_rgb
from ..tactile import tactile_keys

TASKS = ("insert_hole", "insert_HDMI")


def contiguous_segments(steps, expected_stride=2):
    steps = np.asarray(steps)
    boundaries = np.r_[0, np.flatnonzero(np.diff(steps) != expected_stride) + 1, len(steps)]
    return [[int(a), int(b)] for a, b in zip(boundaries[:-1], boundaries[1:])]


def _blocked_hold_ranges(report_path):
    """Return physical-step intervals covered by blocked-hold actions in a report."""
    report_path = Path(report_path)
    if not report_path.is_file():
        return []
    try:
        events = json.loads(report_path.read_text()).get("events", [])
    except (OSError, json.JSONDecodeError):
        return []
    ranges = []
    for index, event in enumerate(events):
        if event.get("event") != "blocked_hold" or event.get("step") is None:
            continue
        start = int(event["step"])
        end = next((int(later["step"]) for later in events[index + 1:]
                    if later.get("step") is not None and int(later["step"]) > start), None)
        if end is None:
            # A report normally records retreat immediately after the hold. Keep
            # a conservative fallback for a truncated report with strategy_steps.
            end = start + 2 * int(event.get("strategy_steps", 0))
        if end > start:
            ranges.append({"start_exclusive": start, "end_inclusive": end})
    return ranges


def build_manifest(data_root, original_root=None, recovery_root=None, tasks=TASKS, seed=42,
                   preserve_existing_splits=False):
    """Materialize portable relative paths; hardlinks preserve original HDF5 bytes."""
    data_root = Path(data_root).resolve()
    original_root = Path(original_root or WORKSPACE / "UniVTAC/data/isaac45")
    recovery_root = Path(recovery_root or data_root / "recovery")
    summaries = {}
    for task in tasks:
        prompt = json.loads((WORKSPACE / "UniVTAC/instructions" / f"{task}.json").read_text())["seen"][0]
        rows = []
        manifest_path = data_root / "manifests" / f"{task}.jsonl"
        previous_splits = {}
        if preserve_existing_splits and manifest_path.is_file():
            previous_splits = {row["episode_id"]: row["split"]
                               for row in read_jsonl(manifest_path) if row.get("split") in {"train", "val"}}
        recovery_report = recovery_root / task / "collection.json"
        recovery_metadata = {}
        if recovery_report.exists():
            recovery_metadata = {str(r["seed"]): r for r in json.loads(recovery_report.read_text())["attempts"]}
        for source, candidates in (
            ("original", sorted((original_root / task / "hdf5").glob("*.hdf5"))),
            ("recovery", sorted((recovery_root / task / "hdf5").glob("*.hdf5"))),
        ):
            if source == "recovery":
                # Recovery HDF5 files are not self-authenticating: only collection
                # records explicitly marked successful and physically blocked may
                # enter training. In particular, a missing/empty report must not
                # turn every file in the directory into a positive example.
                candidates = [
                    path for path in candidates
                    if (record := recovery_metadata.get(path.stem)) is not None
                    and record.get("success") is True
                    and record.get("blockages", 0) >= 1
                    and not record.get("excluded_from_training", False)
                ]
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(candidates))
            n_val = max(1, round(len(candidates) * .1)) if len(candidates) > 1 else 0
            val = set(order[:n_val].tolist())
            for idx, path in enumerate(candidates):
                relative = Path("hdf5") / source / task / path.name
                target = data_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    try:
                        os.link(path, target)
                    except OSError:
                        import shutil
                        shutil.copy2(path, target)
                with h5py.File(target, "r") as f:
                    n = len(f["step"])
                    if f["embodiment/joint"].shape != (n, 9):
                        raise ValueError(f"Unexpected raw joint layout in {path}")
                    tactile_keys(f)
                    if "observation/head/rgb" not in f:
                        raise ValueError(f"Missing head camera: {path}")
                    segments = contiguous_segments(f["step"][:])
                episode_id = f"{source}/{task}/{path.stem}"
                split = previous_splits.get(episode_id, "val" if idx in val else "train")
                row = {"episode_id": episode_id, "task": task, "source": source,
                             "path": str(relative), "split": split,
                             "frames": n, "segments": segments, "prompt": prompt, "physical_stride": 2}
                if source == "recovery":
                    row["policy_action_excluded_ranges"] = _blocked_hold_ranges(
                        recovery_root / task / "diagnostics" / f"{path.stem}.json")
                rows.append(row)
                if source == "recovery":
                    rows[-1]["recovery"] = recovery_metadata.get(path.stem, {})
        if not rows:
            raise ValueError(f"No data for {task}")
        write_jsonl(manifest_path, rows)
        states, actions = [], []
        for row in rows:
            if row["split"] != "train":
                continue
            with h5py.File(data_root / row["path"], "r") as f:
                joints = np.asarray(f["embodiment/joint"][:, :8], np.float32)
            for a, b in row["segments"]:
                if b - a > 1:
                    states.append(joints[a:b-1])
                    actions.append(joints[a+1:b])
        stats = {key: {"q01": np.quantile(np.concatenate(values), .01, axis=0).tolist(),
                       "q99": np.quantile(np.concatenate(values), .99, axis=0).tolist()}
                 for key, values in (("state", states), ("action", actions))}
        stats["metadata"] = {"task": task, "action": "next_frame_absolute_first8", "split": "train",
                             "episodes": [r["episode_id"] for r in rows if r["split"] == "train"]}
        atomic_json(data_root / "stats" / f"{task}.quantiles.json", stats)
        summaries[task] = {source: sum(r["source"] == source for r in rows) for source in ("original", "recovery")}
    atomic_json(data_root / "preparation.json", summaries)
    return summaries


class ContactDataset(Dataset):
    """Fixed-capacity, left-padded recurrent sequences; valid suffix length is sampled 1..5."""
    def __init__(self, manifest, data_root, cache_root, stats, config: ContactConfig, split="train", seed=42,
                 policy_action_mask=False):
        self.rows = [r for r in read_jsonl(manifest) if r["split"] == split]
        self.root, self.cache_root = Path(data_root), Path(cache_root)
        self.norm, self.config, self.split, self.seed = Quantiles(stats), config, split, seed
        self.policy_action_mask = bool(policy_action_mask)
        self.index, self.sources, self.tasks = [], [], []
        for rid, row in enumerate(self.rows):
            for start, stop in row["segments"]:
                for frame in range(start, stop - 1):
                    self.index.append((rid, start, stop, frame))
                    self.sources.append(row["source"])
                    self.tasks.append(row["task"])
        if not self.index:
            raise ValueError(f"Empty {split} dataset")
        self._episodes = OrderedDict()

    def __len__(self):
        return len(self.index)

    def _load(self, rid):
        if rid in self._episodes:
            self._episodes.move_to_end(rid)
            return self._episodes[rid]
        row = self.rows[rid]
        with h5py.File(self.root / row["path"], "r") as f:
            joints = np.asarray(f["embodiment/joint"][:, :8], np.float32)
            images = f["observation/head/rgb"][:]
            steps = f["step"][:]
            tags = None
            if "atom/tag" in f:
                raw_tags = f["atom/tag"][:]
                tags = np.char.decode(raw_tags, "utf-8") if raw_tags.dtype.kind == "S" else raw_tags.astype(str)
        tactile = load_tactile_tokens(self.cache_root, row["episode_id"], steps)
        value = (joints, images, tactile, steps, tags)
        self._episodes[rid] = value
        while len(self._episodes) > 4:
            self._episodes.popitem(last=False)
        return value

    def __getitem__(self, item):
        # Sampler supplies a draw seed so resumed/DDP training and validation remain reproducible.
        if isinstance(item, tuple):
            index, draw_seed = item
        else:
            index, draw_seed = int(item), self.seed + int(item)
        rng = np.random.default_rng(draw_seed)
        rid, start, stop, current = self.index[index]
        row = self.rows[rid]
        joints, encoded, tactile, raw_steps, atom_tags = self._load(rid)
        cfg = self.config
        warmup = int(rng.integers(cfg.belief_warmup_min, cfg.belief_warmup_max + 1))
        warmup = min(warmup, (current - start) // cfg.k)
        length = cfg.belief_warmup_max + 1
        frames = current - np.arange(length - 1, -1, -1) * cfg.k
        seq_valid = np.arange(length) >= length - warmup - 1
        frames = np.clip(frames, start, stop - 1)
        # The sequence has fixed capacity, so repeat its earliest valid observation
        # into the masked prefix instead of exposing unrelated older frames as padding.
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
        # Decode each distinct HDF5 frame once. Left padding and overlapping
        # observation/future windows commonly reference the same JPEG frame.
        # Restore the original sequence order after decoding the unique indices.
        frame_count = frames.size
        decode_indices = np.concatenate((frames.reshape(-1), future_indices.reshape(-1)))
        unique_indices, inverse = np.unique(decode_indices, return_inverse=True)
        decoded = np.stack([decode_rgb(encoded[i]) for i in unique_indices])
        image_shape = (*frames.shape, *decoded.shape[1:])
        future_image_shape = (*future_indices.shape, *decoded.shape[1:])
        image = decoded[inverse[:frame_count]].reshape(image_shape)
        future_image = decoded[inverse[frame_count:]].reshape(future_image_shape)
        # The action at t is the absolute joint state at t+1. Exclusion signals
        # are therefore read from each target frame, not the observation frame.
        policy_action_mask = action_valid.copy()
        if self.policy_action_mask:
            excluded = np.zeros_like(action_valid)
            if atom_tags is not None:
                excluded |= atom_tags[action_indices] == "perturb"
            target_steps = raw_steps[action_indices]
            for interval in row.get("policy_action_excluded_ranges", []):
                excluded |= ((target_steps > int(interval["start_exclusive"])) &
                             (target_steps <= int(interval["end_inclusive"])))
            policy_action_mask &= ~excluded
        return {"image": image, "future_image": future_image,
                "state": self.norm.normalize(joints[frames], "state"),
                "actions": self.norm.normalize(joints[action_indices], "action"),
                "executed_actions": self.norm.normalize(joints[executed_indices], "action"),
                "executed_valid": executed_valid, "tactile": tactile[frames].astype(np.float32),
                "future_tactile": tactile[future_indices].astype(np.float32), "action_mask": action_valid,
                "future_mask": future_valid, "sequence_mask": seq_valid,
                "policy_action_mask": policy_action_mask,
                "step": frames - start, "source": row["source"], "task": row["task"], "prompt": row["prompt"],
                "episode_id": row["episode_id"]}


class MixedSequenceSampler(Sampler):
    """Sample frames with reproducible source mixing and optional task balancing."""
    def __init__(self, dataset, draws, ratio=(1, 1), seed=42, rank=0, start_draw=0,
                 task_balanced=False, original_only_draws=0):
        self.groups = [np.flatnonzero(np.asarray(dataset.sources) == s) for s in ("original", "recovery")]
        self.ratio = np.asarray(ratio, dtype=float)
        if np.any(self.ratio < 0) or self.ratio.sum() <= 0:
            raise ValueError("Data mixing weights must be nonnegative and have a positive sum")
        for group, weight in zip(self.groups, self.ratio):
            if weight > 0 and len(group) == 0:
                if task_balanced and weight > 0 and len(group) == 0:
                    continue
                raise ValueError("Requested mixture includes an empty source; generate recovery data first")
        self.ratio /= self.ratio.sum()
        self.draws, self.seed, self.rank, self.start_draw = int(draws), int(seed), int(rank), int(start_draw)
        self.task_balanced = bool(task_balanced)
        self.original_only_draws = int(original_only_draws)
        if self.original_only_draws < 0:
            raise ValueError("original_only_draws must be nonnegative")
        self.task_groups = {}
        if self.task_balanced:
            tasks = np.asarray(dataset.tasks)
            sources = np.asarray(dataset.sources)
            for task in sorted(set(tasks.tolist())):
                self.task_groups[task] = {
                    source: np.flatnonzero((tasks == task) & (sources == source))
                    for source in ("original", "recovery")
                }
                if len(self.task_groups[task]["original"]) == 0:
                    raise ValueError(f"Task {task} has no original samples")

    def __iter__(self):
        for draw in range(self.start_draw, self.start_draw + self.draws):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.rank, draw]))
            if self.task_balanced:
                task = sorted(self.task_groups)[int(rng.integers(len(self.task_groups)))]
                task_groups = self.task_groups[task]
                if draw < self.original_only_draws or len(task_groups["recovery"]) == 0:
                    group = task_groups["original"]
                else:
                    available = np.asarray([len(task_groups[source]) > 0 for source in ("original", "recovery")])
                    probabilities = self.ratio * available
                    if probabilities.sum() <= 0:
                        group = task_groups["original"]
                    else:
                        probabilities /= probabilities.sum()
                        source_id = int(rng.choice(2, p=probabilities))
                        group = task_groups[("original", "recovery")[source_id]]
            elif draw < self.original_only_draws:
                group = self.groups[0]
            else:
                group = self.groups[rng.choice(2, p=self.ratio)]
            yield (int(rng.choice(group)), int(rng.integers(0, 2**32)))

    def __len__(self):
        return self.draws


def collate_contact(samples):
    result = {}
    for key in samples[0]:
        values = [s[key] for s in samples]
        result[key] = torch.from_numpy(np.stack(values)) if isinstance(values[0], np.ndarray) else values
    return result
