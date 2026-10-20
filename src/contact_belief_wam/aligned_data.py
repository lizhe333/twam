"""Data alignment with the reference WAVE-0 UniVTAC input contract.

The aligned adapter deliberately keeps the existing HDF5 files and AnyTouch
feature cache as the source of truth.  It only writes portable manifests and
copies the reference normalization assets into a dedicated aligned directory.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler

from .config import ContactConfig
from .feature_cache import load_tactile_tokens
from .paths import PROJECT, WORKSPACE, atomic_json, read_jsonl, write_jsonl


SHARED_TASKS = (
    "insert_hole",
    "insert_HDMI",
    "insert_tube",
    "pull_out_key",
    "grasp_classify",
    "lift_bottle",
    "lift_can",
    "put_bottle_in_shelf",
)
VALIDATION_EPISODE_IDS = (14, 20, 51, 60, 71, 74, 82, 86, 91, 92)
IMAGE = "observation.images.head"
FUTURE = "observation.world_model.future.h9.images.head"
TOKEN_LENGTH = 200


def resize(image):
    """Reference PIL bilinear resize-with-pad, including integer rounding."""
    image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    width, height = image.size
    ratio = max(width / 224, height / 224)
    resized_width, resized_height = int(width / ratio), int(height / ratio)
    result = Image.new(image.mode, (224, 224), 0)
    result.paste(
        image.resize((resized_width, resized_height), Image.Resampling.BILINEAR),
        ((224 - resized_width) // 2, (224 - resized_height) // 2),
    )
    return np.asarray(result).copy()


def decode(encoded):
    """Decode a UniVTAC JPEG and apply the exact reference resize."""
    import cv2

    image = cv2.imdecode(np.frombuffer(bytes(encoded), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid UniVTAC JPEG")
    # The UniVTAC writer stores numerical RGB arrays without a channel swap.
    return resize(image)


def _read_norm_stats(stats):
    if isinstance(stats, (str, os.PathLike)):
        value = json.loads(Path(stats).read_text())
    else:
        value = stats
    return value.get("norm_stats", value)


class AlignedProcessor:
    """Reference WAVE-0 normalization and SentencePiece tokenization."""

    def __init__(self, stats, tokenizer, prompt):
        self.stats = _read_norm_stats(stats)
        if tokenizer is None:
            self.sp = None
        else:
            try:
                import sentencepiece
            except ImportError as exc:
                raise ImportError(
                    "Aligned UniVTAC tokenization requires the sentencepiece package"
                ) from exc
            self.sp = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer))
        self.prompt = str(prompt).strip().replace("_", " ").replace("\n", " ")

    def normalize(self, value, key):
        """Match OpenPI's q01/q99 transform and epsilon placement exactly."""
        quantiles = self.stats[key]
        low = np.asarray(quantiles["q01"])
        high = np.asarray(quantiles["q99"])
        return (np.asarray(value) - low) / (high - low + 1e-6) * 2 - 1

    def tokens(self, normalized_state):
        if self.sp is None:
            raise RuntimeError("This AlignedProcessor was created without a tokenizer")
        bins = np.digitize(np.asarray(normalized_state), np.linspace(-1, 1, 257)[:-1]) - 1
        text = f"Task: {self.prompt}, State: {' '.join(map(str, bins))};\nAction: "
        ids = self.sp.encode(text, add_bos=True)
        if len(ids) > TOKEN_LENGTH:
            raise ValueError("Baseline task prompt exceeds 200 tokens")
        length = len(ids)
        padded = np.asarray(ids + [0] * (TOKEN_LENGTH - length), dtype=np.int64)
        return padded, np.arange(TOKEN_LENGTH) < length

    def observation(self, image, state):
        """Return the same keys and [0,1] image range as WAVE-0 Processor."""
        normalized_state = self.normalize(np.asarray(state, np.float32), "state")
        ids, mask = self.tokens(normalized_state)
        pixels = resize(image).transpose(2, 0, 1)
        return {
            IMAGE: torch.from_numpy(pixels).float() / 255.0,
            "observation.state": torch.from_numpy(normalized_state.astype(np.float32)),
            "observation.language.tokens": torch.from_numpy(ids),
            "observation.language.attention_mask": torch.from_numpy(mask),
        }

    def actions(self, absolute, state):
        """Normalize absolute targets relative to their chunk anchor and pad to 32."""
        delta = np.asarray(absolute, np.float32).copy()
        delta[..., :7] -= np.asarray(state, np.float32)[..., :7]
        result = self.normalize(delta, "actions").astype(np.float32)
        if result.ndim != 2 or result.shape[-1] != 8:
            raise ValueError(f"Expected [horizon,8] actions, got {result.shape}")
        return np.pad(result, ((0, 0), (0, 24)))

    def absolute(self, normalized, state):
        """Invert normalized actions and add the anchor back to arm joints."""
        value = np.asarray(normalized)[..., :8]
        quantiles = self.stats["actions"]
        low, high = np.asarray(quantiles["q01"]), np.asarray(quantiles["q99"])
        result = (value + 1) / 2 * (high - low + 1e-6) + low
        result[..., :7] += np.asarray(state)[..., :7]
        return result.astype(np.float32)


def _copy_reference_asset(source, destination):
    """Copy a reference asset once and reject silent changes to an aligned copy."""
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"Missing reference normalization asset: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_digest = _sha256(source)
    if destination.exists():
        if _sha256(destination) != source_digest:
            raise ValueError(f"Aligned normalization asset differs from its reference: {destination}")
        return destination
    # Copy through an adjacent temporary file so a failed copy is never visible.
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        if _sha256(temporary) != source_digest:
            raise IOError(f"Normalization copy failed verification: {source}")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if _sha256(destination) != source_digest:
                raise ValueError(f"Aligned normalization asset differs from its reference: {destination}")
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_number(row):
    try:
        return int(str(row["episode_id"]).rsplit("/", 1)[-1])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Cannot read source episode number from {row.get('episode_id')!r}") from exc


def _baseline_segments(baseline_dir, tasks):
    result = {}
    for task in tasks:
        path = Path(baseline_dir) / task / "segments.json"
        entries = json.loads(path.read_text())
        grouped = {}
        for entry in entries:
            if entry.get("task", task) != task:
                raise ValueError(f"Reference segment has the wrong task in {path}")
            episode_id = int(entry["source_episode_id"])
            grouped.setdefault(episode_id, []).append(
                (int(entry["start_frame"]), int(entry["stop_frame"]))
            )
        if set(grouped) != set(range(100)):
            raise ValueError(f"Reference Stage-1 segments must cover original IDs 0..99 for {task}")
        result[task] = {episode: sorted(segments) for episode, segments in grouped.items()}
    return result


def _fingerprint_rows(rows, data_root):
    records = []
    for row in rows:
        path = Path(data_root) / row["path"]
        try:
            stat = path.stat()
            size = int(stat.st_size)
        except OSError as exc:
            raise FileNotFoundError(f"Aligned manifest raw data is unavailable: {path}") from exc
        records.append({
            "episode_id": row["episode_id"],
            "task": row["task"],
            "source": row["source"],
            "path": row["path"],
            "size_bytes": size,
            "segments": row["segments"],
            "split": row["split"],
        })
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def prepare_aligned_data(run, *, data_root=None, original_root=None, recovery_root=None):
    """Create aligned Stage-1 and Stage-2 manifests and copy reference stats.

    ``original_root`` and ``recovery_root`` are accepted for configuration
    parity, but raw files are never copied or modified; the already materialized
    Contact-Belief manifest paths remain authoritative.
    """
    del original_root, recovery_root
    workspace = Path(run.get("workspace", WORKSPACE)).resolve()
    data_root = Path(data_root or run.get("data_root", PROJECT / "data")).resolve()
    tasks = tuple(run.get("tasks", run.get("shared_tasks", SHARED_TASKS)))
    if not tasks:
        raise ValueError("Aligned preparation needs at least one task")
    if any(task not in SHARED_TASKS for task in tasks):
        raise ValueError(f"Unsupported aligned tasks: {tasks}")

    reference_root = Path(run.get("reference_root", PROJECT.parent / "wave0-univtac-fast")).resolve()
    baseline_dir = Path(run.get("baseline_dir", reference_root / "baseline")).resolve()
    source_wm_stats = Path(
        run.get("reference_wm_stats", run.get("source_wm_stats", run.get("wm_norm")))
        or workspace / "outputs/wave0-shared-8task-mb32-002/artifacts/wm_norm_stats.json"
    ).resolve()
    aligned_manifest_dir = Path(
        run.get("aligned_manifest_dir", data_root / "manifests/aligned")
    ).resolve()
    policy_manifest_dir = Path(
        run.get("policy_manifest_dir", aligned_manifest_dir / "policy_all_train")
    ).resolve()
    task_stats_dir = Path(run.get("task_stats_dir", data_root / "stats/aligned/tasks")).resolve()
    wm_stats = Path(run.get("wm_stats", data_root / "stats/aligned/wm_norm_stats.json")).resolve()
    cache_root = Path(run.get("cache_root", run.get("tactile_cache", data_root / "cache/anytouch2"))).resolve()
    tokenizer_path = Path(
        run.get("tokenizer_path", workspace / "models/pi05_base/big_vision/paligemma_tokenizer.model")
    ).resolve()
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Missing reference SentencePiece tokenizer: {tokenizer_path}")

    validation_ids = tuple(int(value) for value in run.get("wm_validation_episode_ids", VALIDATION_EPISODE_IDS))
    if len(validation_ids) != 10 or len(set(validation_ids)) != 10 or any(value not in range(100) for value in validation_ids):
        raise ValueError("Stage-1 validation requires the fixed ten original episode IDs")
    reference_segments = _baseline_segments(baseline_dir, tasks)

    copied_task_stats = {}
    for task in tasks:
        copied_task_stats[task] = str(_copy_reference_asset(
            baseline_dir / task / "norm_stats.json", task_stats_dir / f"{task}.norm_stats.json"
        ))
    copied_wm_stats = str(_copy_reference_asset(source_wm_stats, wm_stats))

    shared_rows, policy_rows = [], {}
    for task in tasks:
        source_manifest = data_root / "manifests" / f"{task}.jsonl"
        if not source_manifest.is_file():
            raise FileNotFoundError(f"Missing prepared Contact-Belief manifest: {source_manifest}")
        original = {}
        recoveries = []
        prompt = json.loads((baseline_dir / task / "instruction.json").read_text())["seen"][0]
        for source_row in read_jsonl(source_manifest):
            row = dict(source_row)
            row["prompt"] = prompt
            if row.get("source") == "original":
                episode = _episode_number(row)
                if episode in original:
                    raise ValueError(f"Duplicate original episode {episode} for {task}")
                if episode not in reference_segments[task]:
                    raise ValueError(f"Original episode {episode} has no Stage-1 reference segment for {task}")
                # Keep the original reference segment boundaries exactly.
                row["segments"] = [list(segment) for segment in reference_segments[task][episode]]
                row["split"] = "val" if episode in validation_ids else "train"
                original[episode] = row
            elif row.get("source") == "recovery":
                row["split"] = "train"
                recoveries.append(row)
            else:
                raise ValueError(f"Unknown source in {source_manifest}: {row.get('source')!r}")
        if set(original) != set(range(100)):
            raise ValueError(f"Aligned original rows must cover 100 episodes for {task}; got {len(original)}")
        if task in {"insert_hole", "insert_HDMI"} and len(recoveries) != 50:
            raise ValueError(f"Aligned Stage-2 expects 50 recovery episodes for {task}, got {len(recoveries)}")
        ordered_original = [original[index] for index in range(100)]
        task_policy_rows = [dict(row, split="train") for row in ordered_original] + recoveries
        policy_rows[task] = task_policy_rows
        shared_rows.extend(ordered_original)
        shared_rows.extend(recoveries)

    shared_path = aligned_manifest_dir / "shared.jsonl"
    write_jsonl(shared_path, shared_rows)
    policy_manifests = {}
    for task, rows in policy_rows.items():
        path = policy_manifest_dir / f"{task}.jsonl"
        write_jsonl(path, rows)
        policy_manifests[task] = str(path)

    val_counts = {task: sum(row["task"] == task and row["split"] == "val" for row in shared_rows)
                  for task in tasks}
    if sum(val_counts.values()) != 10 * len(tasks):
        raise ValueError("Aligned validation manifest must contain only the fixed original holdouts")
    if sum(row["split"] == "val" and row["source"] == "recovery" for row in shared_rows):
        raise ValueError("Recovery episodes cannot appear in Stage-1 validation")

    episode_fingerprint = _fingerprint_rows(shared_rows, data_root)
    policy_fingerprints = {task: _fingerprint_rows(rows, data_root) for task, rows in policy_rows.items()}
    summary = {
        "tasks": {
            task: {
                "stage1_train_original": sum(row["task"] == task and row["source"] == "original" and row["split"] == "train" for row in shared_rows),
                "stage1_train_recovery": sum(row["task"] == task and row["source"] == "recovery" and row["split"] == "train" for row in shared_rows),
                "stage1_val_original": val_counts[task],
                "stage2_original": sum(row["source"] == "original" for row in policy_rows[task]),
                "stage2_recovery": sum(row["source"] == "recovery" for row in policy_rows[task]),
                "policy_manifest": policy_manifests[task],
                "policy_fingerprint": policy_fingerprints[task],
            }
            for task in tasks
        },
        "stage1_train_original": sum(row["source"] == "original" and row["split"] == "train" for row in shared_rows),
        "stage1_train_recovery": sum(row["source"] == "recovery" and row["split"] == "train" for row in shared_rows),
        "stage1_val_original": sum(row["source"] == "original" and row["split"] == "val" for row in shared_rows),
        "shared_manifest": str(shared_path),
        "episode_fingerprint": episode_fingerprint,
    }
    preparation_path = aligned_manifest_dir / "preparation.json"
    atomic_json(preparation_path, summary)
    return {
        "aligned_manifest_dir": str(aligned_manifest_dir),
        "shared_manifest": str(shared_path),
        "manifest": str(shared_path),
        "policy_manifest_dir": str(policy_manifest_dir),
        "policy_manifests": policy_manifests,
        "task_stats_dir": str(task_stats_dir),
        "task_stats": copied_task_stats,
        "wm_stats": copied_wm_stats,
        "cache_root": str(cache_root),
        "data_root": str(data_root),
        "tokenizer_path": str(tokenizer_path),
        "episode_fingerprint": episode_fingerprint,
        "policy_fingerprints": policy_fingerprints,
        "validation_episode_ids": list(validation_ids),
        "summary": summary,
        "preparation": str(preparation_path),
    }


class AlignedDataset(Dataset):
    """ContactDataset-compatible recurrent examples with WAVE-0 world targets."""

    def __init__(self, manifest, data_root, cache_root, task_stats, wm_stats, config: ContactConfig,
                 split="train", seed=42, policy_action_mask=True, *, tokenizer_path=None,
                 valid_future_only=None):
        self.rows = [row for row in (read_jsonl(manifest) if isinstance(manifest, (str, os.PathLike)) else manifest)
                     if split in {"all", "all_train"} or row["split"] == split]
        self.root, self.cache_root = Path(data_root), Path(cache_root)
        self.config, self.split, self.seed = config, split, int(seed)
        self.policy_action_mask = bool(policy_action_mask)
        self.valid_future_only = (config.stage == "stage1") if valid_future_only is None else bool(valid_future_only)
        if isinstance(task_stats, (str, os.PathLike)):
            task_stats = {task: task_stats for task in {row["task"] for row in self.rows}}
        self.task_stats = dict(task_stats)
        self.wm_norm = _read_norm_stats(wm_stats)
        self.tokenizer_path = tokenizer_path
        self.processors = {}
        self.index, self.sources, self.tasks = [], [], []
        for rid, row in enumerate(self.rows):
            task = row["task"]
            if task not in self.task_stats:
                raise ValueError(f"Missing reference task normalization for {task}")
            if task not in self.processors:
                self.processors[task] = AlignedProcessor(
                    self.task_stats[task], None if config.stage == "stage1" else tokenizer_path,
                    row["prompt"],
                )
            for start, stop in row["segments"]:
                final_anchor = stop - config.k if self.valid_future_only else stop - 1
                for frame in range(int(start), max(int(start), int(final_anchor))):
                    # ``range(start, stop-k)`` means frame+k remains inside the
                    # same segment, which avoids cross-boundary future leakage.
                    self.index.append((rid, int(start), int(stop), frame))
                    self.sources.append(row["source"])
                    self.tasks.append(task)
        if not self.index:
            raise ValueError(f"Empty {split} aligned dataset")
        self._episodes = OrderedDict()

    def __len__(self):
        return len(self.index)

    def _load(self, rid):
        if rid in self._episodes:
            self._episodes.move_to_end(rid)
            return self._episodes[rid]
        row = self.rows[rid]
        with h5py.File(self.root / row["path"], "r") as handle:
            joints = np.asarray(handle["embodiment/joint"][:, :8], np.float32)
            images = handle["observation/head/rgb"][:]
            steps = handle["step"][:]
            tags = None
            if "atom/tag" in handle:
                raw_tags = handle["atom/tag"][:]
                tags = np.char.decode(raw_tags, "utf-8") if raw_tags.dtype.kind == "S" else raw_tags.astype(str)
        tactile = load_tactile_tokens(self.cache_root, row["episode_id"], steps)
        episode = (joints, images, tactile, steps, tags)
        self._episodes[rid] = episode
        while len(self._episodes) > 4:
            self._episodes.popitem(last=False)
        return episode

    @staticmethod
    def _world_actions(absolute, anchor, stats):
        delta = np.asarray(absolute, np.float32).copy()
        delta[..., :7] -= np.asarray(anchor, np.float32)[..., :7]
        low = np.asarray(stats["actions"]["q01"])
        high = np.asarray(stats["actions"]["q99"])
        normalized = (delta - low) / (high - low + 1e-6) * 2 - 1
        if normalized.ndim != 2 or normalized.shape[-1] != 8:
            raise ValueError(f"Expected [horizon,8] world actions, got {normalized.shape}")
        return np.pad(normalized.astype(np.float32), ((0, 0), (0, 24)))

    def __getitem__(self, item):
        if isinstance(item, tuple):
            index, draw_seed = int(item[0]), int(item[1])
        else:
            index, draw_seed = int(item), self.seed + int(item)
        rng = np.random.default_rng(draw_seed)
        rid, start, stop, current = self.index[index]
        row = self.rows[rid]
        joints, encoded, tactile, raw_steps, atom_tags = self._load(rid)
        cfg = self.config

        warmup = int(rng.integers(cfg.belief_warmup_min, cfg.belief_warmup_max + 1))
        warmup = min(warmup, (current - start) // cfg.k)
        sequence_length = cfg.belief_warmup_max + 1
        frames = current - np.arange(sequence_length - 1, -1, -1) * cfg.k
        sequence_mask = np.arange(sequence_length) >= sequence_length - warmup - 1
        frames = np.clip(frames, start, stop - 1)
        padding = sequence_length - warmup - 1
        if padding:
            frames[:padding] = frames[padding]

        action_indices = frames[:, None] + 1 + np.arange(cfg.chunk_size)[None, :]
        action_indices = np.minimum(action_indices, stop - 1)
        future_indices = np.minimum(frames + cfg.k, stop - 1)
        future_mask = (frames + cfg.k < stop) & sequence_mask
        executed_indices = frames[:, None] - cfg.k + 1 + np.arange(cfg.k)[None, :]
        executed_valid = (executed_indices > start) & sequence_mask[:, None]
        executed_indices = np.clip(executed_indices, start, stop - 1)

        decode_indices = np.concatenate((frames.reshape(-1), future_indices.reshape(-1)))
        unique_indices, inverse = np.unique(decode_indices, return_inverse=True)
        decoded = np.stack([decode(encoded[frame]) for frame in unique_indices])
        sequence_frames = frames.size
        image = decoded[inverse[:sequence_frames]].reshape(sequence_length, 224, 224, 3)
        future_image = decoded[inverse[sequence_frames:]].reshape(sequence_length, 224, 224, 3)
        image = image.transpose(0, 3, 1, 2).astype(np.float32) / 127.5 - 1.0
        future_image = future_image.transpose(0, 3, 1, 2).astype(np.float32) / 127.5 - 1.0

        task_processor = self.processors[row["task"]]
        normalized_state = task_processor.normalize(joints[frames], "state")
        state = normalized_state.astype(np.float32)
        world_low = np.asarray(self.wm_norm["state"]["q01"])
        world_high = np.asarray(self.wm_norm["state"]["q99"])
        world_state = ((joints[frames] - world_low) / (world_high - world_low + 1e-6) * 2 - 1).astype(np.float32)
        actions, world_actions, tokens, token_mask = [], [], [], []
        for position, frame in enumerate(frames):
            target = joints[action_indices[position]]
            actions.append(task_processor.actions(target, joints[frame]))
            world_actions.append(self._world_actions(target, joints[frame], self.wm_norm))
            if task_processor.sp is None:
                ids, mask = np.zeros((1,), dtype=np.int64), np.zeros((1,), dtype=bool)
            else:
                # The reference bins the float64 quantile result before the
                # observation is cast to float32 for model input.
                ids, mask = task_processor.tokens(normalized_state[position])
            tokens.append(ids)
            token_mask.append(mask)
        actions = np.stack(actions)
        world_actions = np.stack(world_actions)
        tokens = np.stack(tokens)
        token_mask = np.stack(token_mask)

        # Historical executions span the prior interval [t-k+1, t] and must be
        # represented relative to that interval's anchor q[t-k].
        executed_anchor_indices = np.clip(frames - cfg.k, start, stop - 1)
        executed_anchor = joints[executed_anchor_indices]
        executed_target = joints[executed_indices]
        executed_actions = np.empty((*executed_indices.shape, 8), dtype=np.float32)
        for position in range(sequence_length):
            executed_actions[position] = self._world_actions(
                executed_target[position], executed_anchor[position], self.wm_norm
            )[:, :8]

        policy_action_mask = np.ones(action_indices.shape, dtype=bool)
        if self.policy_action_mask:
            excluded = np.zeros_like(policy_action_mask)
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
            "state": state,
            "world_state": world_state,
            "actions": actions,
            "world_actions": world_actions,
            "executed_actions": executed_actions,
            "executed_valid": executed_valid,
            "tactile": tactile[frames].astype(np.float32),
            "future_tactile": tactile[future_indices].astype(np.float32),
            "tokens": tokens,
            "token_mask": token_mask,
            "action_mask": np.ones(action_indices.shape, dtype=bool),
            "future_mask": future_mask,
            "sequence_mask": sequence_mask,
            "policy_action_mask": policy_action_mask,
            "step": frames - start,
            "sample_index": index,
            "source": row["source"],
            "task": row["task"],
            "prompt": row["prompt"],
            "episode_id": row["episode_id"],
        }

    def close(self):
        self._episodes.clear()

    def __getstate__(self):
        return {**self.__dict__, "_episodes": OrderedDict()}


def prepare_aligned_batch(batch, device):
    """Move preprocessed batch tensors to ``device`` without repeating work."""
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


class AlignedTaskBalancedSampler(Sampler):
    """Replayable sampling with exact per-microbatch task/source composition.

    In Stage 1, each local microbatch contains four samples per task. Insert
    tasks split those four evenly across original/recovery data; other tasks
    use original data. In Stage 2, ``task_balanced=False`` uses the configured
    source ratio across each local microbatch (1:1 by default).
    """

    def __init__(self, dataset, draws, *, seed=42, rank=0, start_draw=0,
                 micro_batch=32, task_balanced=True, source_ratio=(1, 1), world_size=1):
        self.dataset = dataset
        self.draws, self.seed = int(draws), int(seed)
        self.rank, self.world_size = int(rank), int(world_size)
        self.start_draw, self.micro_batch = int(start_draw), int(micro_batch)
        self.task_balanced = bool(task_balanced)
        self.source_ratio = np.asarray(source_ratio, dtype=np.float64)
        if self.draws < 0 or self.start_draw < 0:
            raise ValueError("draws and start_draw must be nonnegative")
        if self.micro_batch <= 0 or self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("Invalid microbatch or distributed rank configuration")
        if np.any(self.source_ratio < 0) or not np.isfinite(self.source_ratio).all() or self.source_ratio.sum() <= 0:
            raise ValueError("source_ratio must be finite, nonnegative, and have a positive sum")
        self.source_ratio /= self.source_ratio.sum()

        self.sources = np.asarray(dataset.sources)
        self.tasks = np.asarray(dataset.tasks)
        self.task_names = sorted(set(self.tasks.tolist()))
        self.groups = {
            task: {
                source: np.flatnonzero((self.tasks == task) & (self.sources == source))
                for source in ("original", "recovery")
            }
            for task in self.task_names
        }
        if not self.task_names:
            raise ValueError("Cannot sample an empty aligned dataset")
        if self.task_balanced:
            if self.micro_batch % len(self.task_names):
                raise ValueError("Stage-1 microbatch must divide evenly across tasks")
            self.per_task = self.micro_batch // len(self.task_names)
            if self.per_task < 1:
                raise ValueError("Stage-1 microbatch must include every task")
        else:
            self.per_task = None
        for task, groups in self.groups.items():
            available = [source for source in ("original", "recovery") if len(groups[source])]
            if not available:
                raise ValueError(f"No samples for task {task}")
            if self.task_balanced and not len(groups["original"]):
                raise ValueError(f"Stage-1 task {task} has no original samples")
        self._order_cache = OrderedDict()
        self._source_order_cache = OrderedDict()
        self._task_cache = {}

    def _source_quotas(self, task, total):
        groups = self.groups[task]
        available = np.asarray([len(groups["original"]) > 0, len(groups["recovery"]) > 0])
        weights = self.source_ratio * available
        if not weights.sum():
            weights = available.astype(np.float64)
        weights /= weights.sum()
        exact = weights * total
        quotas = np.floor(exact).astype(int)
        for index in np.argsort(-(exact - quotas))[: total - quotas.sum()]:
            quotas[index] += 1
        return {"original": int(quotas[0]), "recovery": int(quotas[1])}

    @staticmethod
    def _seeded_order(size, seed, task, source, epoch):
        sequence = np.random.SeedSequence([seed, _stable_int(task), _stable_int(source), epoch])
        return np.random.default_rng(sequence).permutation(size)

    def _sample(self, task, source, occurrence, block):
        group = self.groups[task][source]
        epoch, offset = divmod(occurrence, len(group))
        key = (task, source, epoch)
        order = self._order_cache.get(key)
        if order is None:
            order = self._seeded_order(len(group), self.seed, task, source, epoch)
            self._order_cache[key] = order
            while len(self._order_cache) > 64:
                self._order_cache.popitem(last=False)
        else:
            self._order_cache.move_to_end(key)
        return int(group[order[offset]])

    def _source_order(self, task, quotas, block):
        key = (task, block)
        value = self._source_order_cache.get(key)
        if value is None:
            sources = [source for source in ("original", "recovery")
                       for _ in range(quotas[source])]
            if not sources:
                raise ValueError(f"No available source for task {task}")
            shuffle = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.rank, block, _stable_int(task)])
            )
            value = tuple(sources[index] for index in shuffle.permutation(len(sources)))
            self._source_order_cache[key] = value
            while len(self._source_order_cache) > 64:
                self._source_order_cache.popitem(last=False)
        else:
            self._source_order_cache.move_to_end(key)
        return value

    def _task_for_block(self, block):
        task = self._task_cache.get(block)
        if task is None:
            if len(self.task_names) == 1:
                task = self.task_names[0]
            else:
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, block, self.rank]))
                task = self.task_names[int(rng.integers(len(self.task_names)))]
            self._task_cache[block] = task
            if len(self._task_cache) > 64:
                self._task_cache.pop(next(iter(self._task_cache)))
        return task

    def __iter__(self):
        for position in range(self.start_draw, self.start_draw + self.draws):
            block, slot = divmod(position, self.micro_batch)
            if self.task_balanced:
                task_slot, task_offset = divmod(slot, self.per_task)
                task = self.task_names[task_slot]
                quotas = self._source_quotas(task, self.per_task)
            else:
                task = self._task_for_block(block)
                quotas = self._source_quotas(task, self.micro_batch)
                task_offset = slot

            # A fixed, replayable permutation of source labels provides exact
            # local ratios without making sample order depend on DataLoader.
            source_order = self._source_order(task, quotas, block)
            if len(source_order) != (self.per_task if self.task_balanced else self.micro_batch):
                raise AssertionError("Sampler source quotas do not cover the microbatch")
            source = source_order[task_offset]
            occurrence_in_group = sum(1 for prior in source_order[:task_offset] if prior == source)
            if self.task_balanced:
                group_count_per_block = quotas[source]
                group_block = block * self.world_size + self.rank
                occurrence = group_block * group_count_per_block + occurrence_in_group
            else:
                group_count_per_block = quotas[source]
                group_block = block * self.world_size + self.rank
                occurrence = group_block * group_count_per_block + occurrence_in_group
            sample_index = self._sample(task, source, occurrence, block)
            draw_seed = int(np.random.SeedSequence([self.seed, self.rank, position, sample_index]).generate_state(1)[0])
            yield sample_index, draw_seed

    def __len__(self):
        return self.draws


def _stable_int(value):
    return int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:4], "little")
