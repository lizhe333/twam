"""Uncompressed, memory-mappable companions for compressed tactile feature caches."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
import warnings

import numpy as np


_FORMAT_VERSION = 1
_TOKEN_TAIL_SHAPE = (8, 512)


def _episode_paths(cache_root, episode_id):
    root = Path(cache_root)
    relative = Path(str(episode_id))
    if relative.is_absolute() or not relative.parts or any(part in {".", ".."} for part in relative.parts):
        raise ValueError(f"Invalid episode_id for feature cache: {episode_id!r}")
    base = root / relative
    return (
        base.parent / f"{base.name}.npz",
        base.parent / f"{base.name}.tokens.npy",
        base.parent / f"{base.name}.tokens.json",
        relative.with_name(f"{relative.name}.npz").as_posix(),
    )


def _source_signature(path):
    stat = Path(path).stat()
    return {"source_npz_size": int(stat.st_size), "source_npz_mtime_ns": int(stat.st_mtime_ns)}


def _same_signature(left, right):
    return (left["source_npz_size"] == right["source_npz_size"] and
            left["source_npz_mtime_ns"] == right["source_npz_mtime_ns"])


def _steps_equal(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return left.shape == right.shape and left.dtype == right.dtype and np.array_equal(left, right)


def _metadata_steps(metadata):
    raw = np.asarray(metadata["raw_step"], dtype=np.dtype(metadata["raw_step_dtype"]))
    shape = tuple(int(value) for value in metadata["raw_step_shape"])
    if raw.size != int(np.prod(shape, dtype=np.int64)):
        raise ValueError("raw_step values do not match raw_step_shape")
    return raw.reshape(shape)


def _valid_companion(source_npz, tokens_path, metadata_path, episode_id, source_relative,
                     expected_steps=None, expected_frames=None, compare_source_steps=False):
    """Return (memmap, reason); the reason is None only when the companion is valid."""
    source_npz, tokens_path, metadata_path = map(Path, (source_npz, tokens_path, metadata_path))
    if not tokens_path.exists() and not metadata_path.exists():
        return None, "companion_missing"
    if not tokens_path.is_file() or not metadata_path.is_file():
        return None, "companion_incomplete"
    try:
        source_before = _source_signature(source_npz)
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("format_version") != _FORMAT_VERSION:
            return None, "unsupported_companion_format"
        if metadata.get("episode_id") != str(episode_id):
            return None, "episode_id_mismatch"
        if metadata.get("source_npz") != source_relative:
            return None, "source_path_mismatch"
        if not _same_signature(metadata, source_before):
            return None, "source_npz_stat_mismatch"

        tokens = np.load(tokens_path, mmap_mode="r", allow_pickle=False)
        if not isinstance(tokens, np.memmap):
            return None, "tokens_are_not_memmapped"
        shape = tuple(int(value) for value in metadata["shape"])
        dtype = np.dtype(metadata["dtype"])
        if tuple(tokens.shape) != shape or tokens.dtype != dtype:
            return None, "tokens_metadata_mismatch"
        if tokens.ndim != 3 or tokens.shape[1:] != _TOKEN_TAIL_SHAPE or tokens.dtype != np.dtype(np.float16):
            return None, "tokens_shape_or_dtype_invalid"

        raw_step = _metadata_steps(metadata)
        if raw_step.ndim != 1:
            return None, "raw_step_shape_invalid"
        if expected_frames is not None and raw_step.shape[0] != int(expected_frames):
            return None, "raw_step_frame_count_mismatch"
        if tokens.shape[0] != raw_step.shape[0]:
            return None, "tokens_raw_step_length_mismatch"
        if expected_steps is not None and not _steps_equal(raw_step, expected_steps):
            return None, "raw_step_mismatch"

        if compare_source_steps:
            with np.load(source_npz, allow_pickle=False) as source:
                if not _steps_equal(raw_step, source["raw_step"]):
                    return None, "raw_step_source_mismatch"
        if not _same_signature(source_before, _source_signature(source_npz)):
            return None, "source_npz_changed_during_validation"
        return tokens, None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return None, f"companion_read_error:{type(exc).__name__}"


def _validate_source_arrays(tokens, raw_step, episode_id, expected_frames=None):
    if tokens.ndim != 3 or tokens.shape[1:] != _TOKEN_TAIL_SHAPE:
        raise ValueError(f"Invalid tokens shape for {episode_id}: {tokens.shape}; expected [frames,8,512]")
    if tokens.dtype != np.dtype(np.float16):
        raise ValueError(f"Invalid tokens dtype for {episode_id}: {tokens.dtype}; expected float16")
    if raw_step.ndim != 1 or raw_step.shape[0] != tokens.shape[0]:
        raise ValueError(f"Invalid raw_step shape for {episode_id}: {raw_step.shape}")
    if expected_frames is not None and tokens.shape[0] != int(expected_frames):
        raise ValueError(
            f"Frame count mismatch for {episode_id}: tokens={tokens.shape[0]} manifest={expected_frames}"
        )


def _write_temp_npy(path, value):
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise
    return Path(name)


def _write_temp_json(path, value):
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise
    return Path(name)


def _atomic_write_companion(tokens_path, metadata_path, tokens, metadata):
    tokens_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_tokens = _write_temp_npy(tokens_path, tokens)
    temporary_metadata = None
    try:
        temporary_metadata = _write_temp_json(metadata_path, metadata)
        # The metadata is replaced last, so readers never accept an unpaired new array.
        os.replace(temporary_tokens, tokens_path)
        os.replace(temporary_metadata, metadata_path)
    finally:
        for temporary in (temporary_tokens, temporary_metadata):
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass


def prepare_mmap_cache(rows, cache_root):
    """Create or reuse atomic float16 ``.tokens.npy`` companions for episode NPZs.

    The NPZ remains the source of truth. A companion is reusable only when its
    episode path, source file size/mtime, token shape/dtype, and raw-step metadata
    agree with the current source.
    """
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    converted = reused = converted_frames = reused_frames = 0
    episodes = []
    seen = set()

    for row in rows:
        episode_id = str(row["episode_id"])
        if episode_id in seen:
            raise ValueError(f"Duplicate episode_id in rows: {episode_id}")
        seen.add(episode_id)
        expected_frames = int(row["frames"]) if row.get("frames") is not None else None
        source_npz, tokens_path, metadata_path, source_relative = _episode_paths(cache_root, episode_id)
        if not source_npz.is_file():
            raise FileNotFoundError(f"Missing source tactile cache: {source_npz}")
        signature = _source_signature(source_npz)

        existing, reason = _valid_companion(
            source_npz, tokens_path, metadata_path, episode_id, source_relative,
            expected_frames=expected_frames, compare_source_steps=True,
        )
        if existing is not None:
            frames = int(existing.shape[0])
            del existing
            reused += 1
            reused_frames += frames
            episodes.append({"episode_id": episode_id, "status": "reused", "frames": frames})
            continue

        with np.load(source_npz, allow_pickle=False) as source:
            tokens = source["tokens"]
            raw_step = source["raw_step"]
        _validate_source_arrays(tokens, raw_step, episode_id, expected_frames)
        if not _same_signature(signature, _source_signature(source_npz)):
            raise RuntimeError(f"Source NPZ changed while preparing companion: {source_npz}")

        metadata = {
            "format_version": _FORMAT_VERSION,
            "episode_id": episode_id,
            "source_npz": source_relative,
            **signature,
            "shape": list(tokens.shape),
            "dtype": str(tokens.dtype),
            "raw_step": raw_step.tolist(),
            "raw_step_shape": list(raw_step.shape),
            "raw_step_dtype": str(raw_step.dtype),
        }
        _atomic_write_companion(tokens_path, metadata_path, tokens, metadata)
        converted += 1
        converted_frames += int(tokens.shape[0])
        episodes.append({"episode_id": episode_id, "status": "converted", "frames": int(tokens.shape[0]),
                         "previous_companion_state": reason})

    return {
        "episodes": len(episodes),
        "converted": converted,
        "reused": reused,
        "converted_frames": converted_frames,
        "reused_frames": reused_frames,
        "frames": converted_frames + reused_frames,
        "elapsed_seconds": time.perf_counter() - started,
        "episode_results": episodes,
    }


def load_tactile_tokens(cache_root, episode_id, expected_steps):
    """Load valid tactile tokens as a read-only memmap, otherwise use the NPZ source.

    Both paths preserve the source float16 dtype. The caller can select frames first
    and cast only those selected values to float32. A present but invalid companion
    is reported and ignored; the NPZ raw-step sequence is always checked before use.
    """
    expected_steps = np.asarray(expected_steps)
    if expected_steps.ndim != 1:
        raise ValueError(f"expected_steps must be one-dimensional, got {expected_steps.shape}")
    source_npz, tokens_path, metadata_path, source_relative = _episode_paths(cache_root, episode_id)
    tokens, reason = _valid_companion(
        source_npz, tokens_path, metadata_path, str(episode_id), source_relative,
        expected_steps=expected_steps,
    )
    if tokens is not None:
        return tokens
    if reason != "companion_missing":
        warnings.warn(
            f"Ignoring invalid tactile mmap companion for {episode_id}: {reason}; reading source NPZ",
            RuntimeWarning,
            stacklevel=2,
        )

    before = _source_signature(source_npz)
    with np.load(source_npz, allow_pickle=False) as source:
        source_tokens = source["tokens"]
        source_steps = source["raw_step"]
    _validate_source_arrays(source_tokens, source_steps, str(episode_id), len(expected_steps))
    if not _steps_equal(source_steps, expected_steps):
        raise ValueError(f"Source tactile raw_step does not match expected steps for {episode_id}")
    if not _same_signature(before, _source_signature(source_npz)):
        raise RuntimeError(f"Source NPZ changed while loading tactile features: {source_npz}")
    return source_tokens
