"""Streaming UniVTAC HDF5 to LeRobot dataset conversion and validation."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterator, Mapping, Sequence

import h5py
import numpy as np

from openpi.shared.univtac_constants import FPS, HORIZON, SEED, TASKS, prompt, task_names
from openpi.policies.univtac_images import decode_jpeg_rgb
from openpi.shared.univtac_runtime import atomic_json, configure, root as project_root


SCHEMA_VERSION = 1
SOURCE_SUBDIR = Path("UniVTAC/data/isaac45")
SOURCE_STEP_DELTA = 2
EXPECTED_GAP = {
    "task": "lift_can",
    "source_episode_id": 62,
    "frame_index": 227,
    "from_step": 594,
    "to_step": 586,
}
OUTPUT_REPO_PREFIX = "pi05_univtac"
OUTPUT_FEATURES = ("image", "wrist_image", "state", "actions")
LEROBOT_TIMESTAMP_TOLERANCE_SECONDS = 1e-4


@dataclass(frozen=True)
class Segment:
    task: str
    source_episode_id: int
    segment_index: int
    start_frame: int
    stop_frame: int  # exclusive
    valid_count: int
    raw_steps: tuple[int, int]
    source_path: str
    source_size_bytes: int

    @property
    def segment_id(self) -> str:
        return f"{self.task}:{self.source_episode_id}:{self.segment_index}"

    def as_record(self) -> dict[str, Any]:
        result = asdict(self)
        result["segment_id"] = self.segment_id
        result["raw_steps"] = list(self.raw_steps)
        return result


def _resolve_root(root: str | Path | None = None) -> Path:
    return Path(root).expanduser().resolve() if root is not None else project_root()


def _configure_data_environment() -> None:
    configure()
    # LeRobot 0cf8648 rejects the deprecated alias even when HF_LEROBOT_HOME
    # is also set. Keep the canonical cache path configured above.
    os.environ.pop("LEROBOT_HOME", None)


def _split_ids() -> tuple[list[int], list[int]]:
    """Return stable, shared per-task IDs for the seed-42 90/10 partition."""
    permutation = np.random.RandomState(SEED).permutation(100).tolist()
    return sorted(int(value) for value in permutation[:90]), sorted(int(value) for value in permutation[90:])


def split_source_ids(split: str) -> list[int]:
    """Expose the exact source IDs selected by a named dataset split."""
    train_ids, val_ids = _split_ids()
    if split == "full":
        return list(range(100))
    if split == "debug_train":
        return train_ids
    if split == "debug_val":
        return val_ids
    raise ValueError("split must be one of 'full', 'debug_train', or 'debug_val'")


def _episode_paths(base: Path, task: str) -> dict[int, Path]:
    folder = base / SOURCE_SUBDIR / task / "hdf5"
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing UniVTAC HDF5 task directory: {folder}")
    found: dict[int, Path] = {}
    for path in folder.glob("*.hdf5"):
        try:
            episode_id = int(path.stem)
        except ValueError as exc:
            raise ValueError(f"Unexpected nonnumeric UniVTAC episode file: {path}") from exc
        if episode_id in found:
            raise ValueError(f"Duplicate UniVTAC source episode ID {episode_id}: {path}")
        found[episode_id] = path
    expected = set(range(100))
    actual = set(found)
    if actual != expected:
        raise ValueError(
            f"Expected source episode IDs 0..99 for {task}; missing={sorted(expected-actual)}, "
            f"unexpected={sorted(actual-expected)}"
        )
    return found


def _expected_segment_bounds(task: str, episode_id: int, steps: np.ndarray) -> list[tuple[int, int]]:
    if steps.ndim != 1 or len(steps) < 2:
        raise ValueError(f"{task}/{episode_id}: expected a nonempty one-dimensional step sequence")
    breaks = np.flatnonzero(np.diff(steps) != SOURCE_STEP_DELTA)
    if not len(breaks):
        return [(0, len(steps))]
    allowed = (
        task == EXPECTED_GAP["task"]
        and episode_id == EXPECTED_GAP["source_episode_id"]
        and len(breaks) == 1
        and int(breaks[0]) == EXPECTED_GAP["frame_index"]
        and int(steps[breaks[0]]) == EXPECTED_GAP["from_step"]
        and int(steps[breaks[0] + 1]) == EXPECTED_GAP["to_step"]
    )
    if not allowed:
        records = [
            (int(index), int(steps[index]), int(steps[index + 1]))
            for index in breaks
        ]
        raise ValueError(f"Unexpected source step discontinuity in {task}/{episode_id}: {records}")
    boundary = EXPECTED_GAP["frame_index"] + 1
    return [(0, boundary), (boundary, len(steps))]


def scan(
    root: str | Path | None = None,
    split: str = "full",
    *,
    tasks: str | Sequence[str] | None = None,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    """Inspect source metadata and return a deterministic segment manifest.

    This reads one HDF5 episode at a time. It does not decode image payloads.
    ``debug_train`` selects 90 source IDs per task and ``debug_val`` selects the
    complementary 10, using one shared seed-42 split for every task.
    """
    base = _resolve_root(root)
    if split not in ("full", "debug_train", "debug_val"):
        raise ValueError("split must be one of 'full', 'debug_train', or 'debug_val'")
    selected_tasks = task_names(tasks) if isinstance(tasks, str) else tuple(tasks or TASKS)
    unknown = [task for task in selected_tasks if task not in TASKS]
    if unknown:
        raise ValueError(f"Unknown task(s): {unknown}")
    if len(set(selected_tasks)) != len(selected_tasks):
        raise ValueError(f"Task list contains duplicates: {selected_tasks}")
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive")

    train_ids, val_ids = _split_ids()
    wanted_ids = split_source_ids(split)
    if max_episodes is not None:
        wanted_ids = wanted_ids[:max_episodes]
    segments: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    task_summary: dict[str, dict[str, Any]] = {}
    total_frames = 0
    total_pairs = 0

    for task in selected_tasks:
        all_paths = _episode_paths(base, task)
        task_segments: list[dict[str, Any]] = []
        task_frames = 0
        task_pairs = 0
        task_max_frames = 0
        task_max_pairs = 0
        for episode_id in wanted_ids:
            path = all_paths[episode_id]
            source_rel = path.relative_to(base).as_posix()
            size_bytes = path.stat().st_size
            with h5py.File(path, "r") as handle:
                required = ("step", "embodiment/joint", "observation/head/rgb", "observation/wrist/rgb")
                missing = [key for key in required if key not in handle]
                if missing:
                    raise ValueError(f"{source_rel}: missing HDF5 dataset(s): {missing}")
                steps = np.asarray(handle["step"][:])
                joints = handle["embodiment/joint"]
                n_frames = len(steps)
                if joints.shape != (n_frames, 9):
                    expected_shape = (n_frames, 9)
                    raise ValueError(
                        f"{source_rel}: embodiment/joint has shape {joints.shape}; expected {expected_shape}"
                    )
                for camera_key in ("observation/head/rgb", "observation/wrist/rgb"):
                    if handle[camera_key].shape != (n_frames,):
                        expected_shape = (n_frames,)
                        raise ValueError(
                            f"{source_rel}: {camera_key} has shape {handle[camera_key].shape}; "
                            f"expected {expected_shape}"
                        )
                joint_values = np.asarray(joints[:], dtype=np.float32)
                if not np.isfinite(joint_values).all():
                    raise ValueError(f"{source_rel}: embodiment/joint contains NaN or infinity")
                bounds = _expected_segment_bounds(task, episode_id, steps)

            source_record = {
                "task": task,
                "source_episode_id": episode_id,
                "path": source_rel,
                "size_bytes": size_bytes,
                "raw_steps": [int(steps[0]), int(steps[-1])],
                "raw_frame_count": n_frames,
                "segment_count": len(bounds),
            }
            source_files.append(source_record)
            task_frames += n_frames
            task_max_frames = max(task_max_frames, n_frames)
            for segment_index, (start, stop) in enumerate(bounds):
                valid_count = stop - start - 1
                if valid_count <= 0:
                    raise ValueError(f"{source_rel}: source segment [{start}, {stop}) has no transition")
                raw_segment = Segment(
                    task=task,
                    source_episode_id=episode_id,
                    segment_index=segment_index,
                    start_frame=start,
                    stop_frame=stop,
                    valid_count=valid_count,
                    raw_steps=(int(steps[start]), int(steps[stop - 1])),
                    source_path=source_rel,
                    source_size_bytes=size_bytes,
                ).as_record()
                task_segments.append(raw_segment)
                task_pairs += valid_count
                task_max_pairs = max(task_max_pairs, valid_count)
        segments.extend(task_segments)
        total_frames += task_frames
        total_pairs += task_pairs
        task_summary[task] = {
            "source_episode_ids": list(wanted_ids),
            "source_episode_count": len(wanted_ids),
            "segment_count": len(task_segments),
            "raw_frame_count": task_frames,
            "valid_pair_count": task_pairs,
            "max_raw_frames_per_episode": task_max_frames,
            "max_valid_pairs_per_segment": task_max_pairs,
        }

    complete_scan = split == "full" and max_episodes is None and tuple(selected_tasks) == tuple(TASKS)
    if complete_scan:
        if len(segments) != 801 or total_pairs != 152079:
            raise ValueError(
                f"Unexpected full UniVTAC scan totals: {len(segments)} segments, {total_pairs} pairs; "
                "expected 801 segments and 152079 pairs"
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "pi05_univtac_source_manifest",
        "source_dataset": "UniVTAC Isaac Sim 4.5",
        "source_root": SOURCE_SUBDIR.as_posix(),
        "split": split,
        "split_seed": SEED,
        "split_rule": (
            "numpy RandomState(42).permutation(100): first 90 train, remaining 10 validation; "
            "same IDs for every task"
        ),
        "train_source_episode_ids": train_ids,
        "val_source_episode_ids": val_ids,
        "selected_tasks": list(selected_tasks),
        "selected_source_episode_ids": {task: list(wanted_ids) for task in selected_tasks},
        "partial_scan": max_episodes is not None or tuple(selected_tasks) != tuple(TASKS),
        "max_episodes_per_task": max_episodes,
        "fps": FPS,
        "source_step_delta": SOURCE_STEP_DELTA,
        "state_dim": 8,
        "action_dim": 8,
        "features": list(OUTPUT_FEATURES),
        "segment_count": len(segments),
        "raw_frame_count": total_frames,
        "valid_pair_count": total_pairs,
        "max_valid_pairs_per_segment": max((s["valid_count"] for s in segments), default=0),
        "task_summary": task_summary,
        "task_prompts": {task: prompt(task) for task in selected_tasks},
        "source_files": source_files,
        "segments": segments,
        "expected_source_gap": EXPECTED_GAP,
    }


def select_split(manifest: Mapping[str, Any], split: str) -> dict[str, Any]:
    """Filter a full scan to one deterministic source split without rereading HDF5."""
    if split not in ("full", "debug_train", "debug_val"):
        raise ValueError("split must be one of 'full', 'debug_train', or 'debug_val'")
    selected = split_source_ids(split)
    selected_set = set(selected)
    tasks = tuple(manifest["selected_tasks"])
    segments = [
        segment for segment in manifest["segments"]
        if int(segment["source_episode_id"]) in selected_set
    ]
    files = [
        source for source in manifest["source_files"]
        if int(source["source_episode_id"]) in selected_set
    ]
    summary: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_segments = [segment for segment in segments if segment["task"] == task]
        task_files = [source for source in files if source["task"] == task]
        summary[task] = {
            "source_episode_ids": selected,
            "source_episode_count": len(task_files),
            "segment_count": len(task_segments),
            "raw_frame_count": sum(int(source["raw_frame_count"]) for source in task_files),
            "valid_pair_count": sum(int(segment["valid_count"]) for segment in task_segments),
            "max_raw_frames_per_episode": max((int(source["raw_frame_count"]) for source in task_files), default=0),
            "max_valid_pairs_per_segment": max((int(segment["valid_count"]) for segment in task_segments), default=0),
        }
    return {
        **dict(manifest),
        "split": split,
        "selected_source_episode_ids": {task: selected for task in tasks},
        "partial_scan": manifest.get("partial_scan", False),
        "segment_count": len(segments),
        "raw_frame_count": sum(value["raw_frame_count"] for value in summary.values()),
        "valid_pair_count": sum(value["valid_pair_count"] for value in summary.values()),
        "max_valid_pairs_per_segment": max((s["valid_count"] for s in segments), default=0),
        "task_summary": summary,
        "source_files": files,
        "segments": segments,
    }


def iter_raw_samples(
    manifest: Mapping[str, Any],
    root: str | Path | None = None,
    *,
    include_images: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield state/action transitions from the selected source segments.

    The current frame supplies the state and camera images; the next frame's
    first eight joint values supply the action target. Segments are yielded
    independently, so a discontinuity can never be crossed.
    """
    base = _resolve_root(root)
    for segment in manifest["segments"]:
        source_path = base / segment["source_path"]
        start = int(segment["start_frame"])
        stop = int(segment["stop_frame"])
        task_prompts = manifest.get("task_prompts", {})
        task_prompt = (
            task_prompts[segment["task"]]
            if segment["task"] in task_prompts
            else prompt(segment["task"])
        )
        with h5py.File(source_path, "r") as handle:
            joints = np.asarray(handle["embodiment/joint"][start:stop], dtype=np.float32)
            steps = np.asarray(handle["step"][start:stop])
            head_dataset = handle["observation/head/rgb"] if include_images else None
            wrist_dataset = handle["observation/wrist/rgb"] if include_images else None
            for offset in range(stop - start - 1):
                frame_index = start + offset
                sample: dict[str, Any] = {
                    "task": segment["task"],
                    "prompt": task_prompt,
                    "source_episode_id": int(segment["source_episode_id"]),
                    "segment_index": int(segment["segment_index"]),
                    "segment_id": segment["segment_id"],
                    "frame_index": frame_index,
                    "step": int(steps[offset]),
                    "action_step": int(steps[offset + 1]),
                    "state": joints[offset, :8].copy(),
                    "actions": joints[offset + 1, :8].copy(),
                }
                if include_images:
                    assert head_dataset is not None and wrist_dataset is not None
                    sample["image"] = decode_jpeg_rgb(head_dataset[frame_index])
                    sample["wrist_image"] = decode_jpeg_rgb(wrist_dataset[frame_index])
                yield sample


def raw_samples(
    root: str | Path | None = None,
    *,
    split: str = "full",
    tasks: str | Sequence[str] | None = None,
    include_images: bool = False,
    manifest: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Convenience iterator that scans when no source manifest is supplied."""
    source_manifest = manifest if manifest is not None else scan(root, split, tasks=tasks)
    yield from iter_raw_samples(source_manifest, root, include_images=include_images)


def _repo_id(split: str, task: str) -> str:
    return f"{OUTPUT_REPO_PREFIX}/{split}/{task}"


def _dataset_path(output_root: Path, split: str, task: str) -> Path:
    return output_root / _repo_id(split, task)


def _episode_records(manifest: Mapping[str, Any], task: str) -> list[dict[str, Any]]:
    return [segment for segment in manifest["segments"] if segment["task"] == task]


def _task_manifest(manifest: Mapping[str, Any], task: str) -> dict[str, Any]:
    """Keep task-local segment and source-path metadata for the converter."""
    task_segments = _episode_records(manifest, task)
    task_files = [record for record in manifest["source_files"] if record["task"] == task]
    if not task_segments or task not in manifest.get("task_summary", {}):
        raise ValueError(f"Source manifest has no segments or summary for task {task!r}")
    expected_gap = manifest.get("expected_source_gap")
    if not isinstance(expected_gap, Mapping) or expected_gap.get("task") != task:
        expected_gap = None
    return {
        "schema_version": manifest["schema_version"],
        "kind": manifest["kind"],
        "source_dataset": manifest["source_dataset"],
        "source_root": manifest["source_root"],
        "split": manifest["split"],
        "split_seed": manifest["split_seed"],
        "split_rule": manifest["split_rule"],
        "selected_tasks": [task],
        "selected_source_episode_ids": {
            task: list(manifest["selected_source_episode_ids"][task])
        },
        "max_episodes_per_task": manifest.get("max_episodes_per_task"),
        "fps": manifest["fps"],
        "source_step_delta": manifest["source_step_delta"],
        "state_dim": manifest["state_dim"],
        "action_dim": manifest["action_dim"],
        "features": list(manifest["features"]),
        "task_summary": {task: manifest["task_summary"][task]},
        "task_prompts": {task: manifest["task_prompts"][task]},
        "source_files": task_files,
        "segments": task_segments,
        "expected_source_gap": dict(expected_gap) if expected_gap is not None else None,
    }


def _reuse_complete_dataset(dataset_path, *, task, source_manifest, source_root):
    """Reuse an existing complete dataset without modifying its metadata."""
    receipt_path = dataset_path / "data_manifest.json"
    if not receipt_path.is_file():
        raise FileExistsError(f"Existing dataset has no converter receipt: {dataset_path}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "complete" or receipt.get("task") != task or receipt.get("split") != source_manifest["split"]:
        raise FileExistsError(f"Existing dataset task/split is incompatible: {dataset_path}")
    validate(dataset_path, root=source_root)
    return "skipped_complete"


def _expected_features() -> dict[str, Any]:
    """Feature declaration for the pinned local LeRobot writer."""
    return {
        "image": {"dtype": "image", "shape": (270, 480, 3), "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": (270, 480, 3), "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
    }


def _write_task_dataset(
    task: str,
    split: str,
    source_manifest: Mapping[str, Any],
    output_root: Path,
    *,
    source_root: Path,
    image_writer_threads: int = 4,
) -> dict[str, Any]:
    """Write one task into a staging directory and atomically publish it."""
    # Import only after paths.configure() has established all cache locations.
    _configure_data_environment()
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover - installed in the training environment
        raise RuntimeError(
            "The pinned LeRobot package must be installed in the pi05-univtac environment"
        ) from exc

    final_path = _dataset_path(output_root, split, task)
    repo_id = _repo_id(split, task)
    task_manifest = _task_manifest(source_manifest, task)
    if final_path.exists():
        status = _reuse_complete_dataset(
            final_path,
            task=task,
            source_manifest=task_manifest,
            source_root=source_root,
        )
        return {"task": task, "status": status, "path": str(final_path)}

    final_path.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = final_path.parent / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_path = staging_parent / f"{task}.partial"
    stage_marker = staging_parent / f"{task}.partial.json"
    if staging_path.exists():
        if stage_marker.is_file():
            stage_state = json.loads(stage_marker.read_text())
            if (stage_state.get("task"), stage_state.get("split")) != (task, split):
                raise FileExistsError(f"Inconsistent staging data exists; refusing to overwrite: {staging_path}")
            if stage_state.get("status") == "complete":
                stage_receipt = staging_path / "data_manifest.json"
                if not stage_receipt.is_file():
                    raise FileExistsError(f"Completed staging data has no receipt: {staging_path}")
                _reuse_complete_dataset(
                    staging_path,
                    task=task,
                    source_manifest=task_manifest,
                    source_root=source_root,
                )
                stage_marker.unlink()
                staging_path.replace(final_path)
                return {"task": task, "status": "recovered_complete", "path": str(final_path)}
            if stage_state.get("status") != "incomplete":
                raise FileExistsError(f"Unrecognized staging state; refusing to overwrite: {stage_marker}")
            # The pinned writer cannot reopen a crashed writer. Restart only
            # this marked task; complete task datasets remain intact.
            shutil.rmtree(staging_path)
            stage_marker.unlink()
        elif (staging_path / "data_manifest.json").is_file():
            stage_receipt = json.loads((staging_path / "data_manifest.json").read_text())
            if stage_receipt.get("status") != "complete":
                raise FileExistsError(f"Unrecognized staging data exists; refusing to overwrite: {staging_path}")
            _reuse_complete_dataset(
                staging_path,
                task=task,
                source_manifest=task_manifest,
                source_root=source_root,
            )
            staging_path.replace(final_path)
            return {"task": task, "status": "recovered_complete", "path": str(final_path)}
        else:
            raise FileExistsError(f"Unrecognized staging data exists; refusing to overwrite: {staging_path}")
    elif stage_marker.exists():
        stage_state = json.loads(stage_marker.read_text())
        if (
            stage_state.get("status") != "incomplete"
            or (stage_state.get("task"), stage_state.get("split")) != (task, split)
        ):
            raise FileExistsError(f"Inconsistent staging marker exists; refusing to overwrite: {stage_marker}")
        stage_marker.unlink()
    atomic_json(stage_marker, {"status": "incomplete", "task": task, "split": split})

    dataset = None
    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=FPS,
            features=_expected_features(),
            robot_type="univtac_isaac45",
            root=staging_path,
            use_videos=False,
            image_writer_threads=image_writer_threads,
            image_writer_processes=0,
        )
        task_segments = _episode_records(source_manifest, task)
        for segment_number, segment in enumerate(task_segments, start=1):
            for sample in iter_raw_samples({"segments": [segment]}, source_root, include_images=True):
                dataset.add_frame(
                    {
                        "image": sample["image"],
                        "wrist_image": sample["wrist_image"],
                        "state": sample["state"],
                        "actions": sample["actions"],
                        "task": sample["prompt"],
                    }
                )
            dataset.save_episode()
            if segment_number == len(task_segments) or segment_number % 10 == 0:
                print(
                    f"data conversion {split}/{task}: {segment_number}/{len(task_segments)} segments",
                    file=sys.stderr,
                    flush=True,
                )
        stop_image_writer = getattr(dataset, "stop_image_writer", None)
        if stop_image_writer is not None:
            stop_image_writer()
        # Record source paths and all retained segment boundaries beside the
        # standard LeRobot files, then publish the completed task atomically.
        receipt = {
            "status": "complete",
            "schema_version": SCHEMA_VERSION,
            "repo_id": repo_id,
            "split": split,
            "task": task,
            "source_manifest": task_manifest,
            "frame_count": int(source_manifest["task_summary"][task]["valid_pair_count"]),
            "episode_count": int(source_manifest["task_summary"][task]["segment_count"]),
            "features": list(OUTPUT_FEATURES),
            "image_storage": "PNG files written by LeRobot image features (lossless after JPEG decode)",
            "fps": FPS,
        }
        atomic_json(staging_path / "data_manifest.json", receipt)
        atomic_json(stage_marker, {"status": "complete", "task": task, "split": split})
        stage_marker.unlink()
        final_path.parent.mkdir(parents=True, exist_ok=True)
        staging_path.replace(final_path)
        validate(final_path, root=source_root)
        return {"task": task, "status": "converted", "path": str(final_path)}
    except Exception:
        stop_image_writer = getattr(dataset, "stop_image_writer", None)
        if stop_image_writer is not None:
            try:
                stop_image_writer()
            except Exception:
                pass
        # Leave the marked partial task for an explicit safe restart. The other
        # completed task directories are never touched.
        raise


def convert(
    root: str | Path | None = None,
    output_root: str | Path | None = None,
    split: str = "full",
    *,
    tasks: str | Sequence[str] | None = None,
    source_manifest: Mapping[str, Any] | str | Path | None = None,
    max_episodes: int | None = None,
    image_writer_threads: int = 4,
) -> dict[str, Any]:
    """Convert each selected task to a resumable local LeRobot repository."""
    base = _resolve_root(root)
    _configure_data_environment()
    destination = Path(output_root).expanduser().resolve() if output_root is not None else base / "datasets"
    if split not in ("full", "debug_train", "debug_val"):
        raise ValueError("split must be one of 'full', 'debug_train', or 'debug_val'")
    if isinstance(source_manifest, (str, Path)):
        source = json.loads(Path(source_manifest).expanduser().read_text())
    elif source_manifest is not None:
        source = dict(source_manifest)
    else:
        source = scan(base, split, tasks=tasks, max_episodes=max_episodes)
    if source["split"] != split:
        if source["split"] == "full" and split in ("debug_train", "debug_val"):
            source = select_split(source, split)
        else:
            raise ValueError(f"Source manifest split {source['split']!r} does not match requested split {split!r}")
    selected_tasks = (
        tuple(tasks)
        if tasks is not None and not isinstance(tasks, str)
        else tuple(source["selected_tasks"])
    )
    if isinstance(tasks, str):
        selected_tasks = task_names(tasks)
    if any(task not in source["selected_tasks"] for task in selected_tasks):
        raise ValueError("Requested conversion task is absent from the source manifest")
    if source.get("max_episodes_per_task") is None:
        expected_source_ids = set(split_source_ids(split))
        for task in selected_tasks:
            manifest_ids = set(source.get("selected_source_episode_ids", {}).get(task, ()))
            if not expected_source_ids.issubset(manifest_ids):
                raise ValueError(
                    f"Source manifest does not contain all {split} episodes for {task}: "
                    f"missing {sorted(expected_source_ids - manifest_ids)}"
                )
    if max_episodes is not None and source.get("max_episodes_per_task") != max_episodes:
        raise ValueError("max_episodes must be applied during scan, not after manifest creation")

    task_results = []
    for task in selected_tasks:
        task_results.append(
            _write_task_dataset(
                task,
                split,
                source,
                destination,
                source_root=base,
                image_writer_threads=image_writer_threads,
            )
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pi05_univtac_conversion_report",
        "split": split,
        "output_root": str(destination),
        "task_results": task_results,
        "segment_count": sum(source["task_summary"][task]["segment_count"] for task in selected_tasks),
        "valid_pair_count": sum(source["task_summary"][task]["valid_pair_count"] for task in selected_tasks),
    }
    report_path = destination / OUTPUT_REPO_PREFIX / split / "conversion_manifest.json"
    atomic_json(report_path, {**report, "source_manifest": source})
    return report


def _read_jsonlines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Missing LeRobot metadata file: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _decode_png(payload: bytes, *, context: str) -> np.ndarray:
    try:
        from io import BytesIO
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - installed in the training environment
        raise RuntimeError("Pillow is required to validate embedded LeRobot images") from exc
    try:
        with Image.open(BytesIO(payload)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"Could not decode embedded PNG for {context}") from exc


def _embedded_image_bytes(table: Any, feature: str, row_index: int, *, context: str) -> bytes:
    value = table[feature][row_index].as_py()
    if isinstance(value, dict):
        payload = value.get("bytes")
    else:
        payload = None
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError(f"LeRobot {feature} frame is not embedded in Parquet for {context}")
    return bytes(payload)


def _as_column_array(table: Any, name: str, dtype: Any) -> np.ndarray:
    return np.asarray(table[name].to_pylist(), dtype=dtype)


def _validate_delta_queries(
    path: Path,
    repo_id: str,
    episode_actions: Sequence[np.ndarray],
    episode_lengths: Sequence[int],
) -> None:
    """Exercise the pinned LeRobot 50-action query and its end-of-episode clamp."""
    _configure_data_environment()
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover - installed in the training environment
        raise RuntimeError("The pinned LeRobot package is required for delta timestamp validation") from exc

    delta_timestamps = {"actions": [offset / FPS for offset in range(HORIZON)]}
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=path,
        delta_timestamps=delta_timestamps,
        download_videos=False,
    )
    offsets = np.arange(HORIZON, dtype=np.int64)
    for episode_index, (actions, length) in enumerate(zip(episode_actions, episode_lengths, strict=True)):
        episode_start = int(dataset.episode_data_index["from"][episode_index].item())
        episode_stop = int(dataset.episode_data_index["to"][episode_index].item())
        if episode_stop - episode_start != length:
            raise ValueError(f"LeRobot episode index {episode_index} has the wrong row range")
        for local_index in sorted({0, length - 1}):
            global_index = episode_start + local_index
            query_indices, padding = dataset._get_query_indices(global_index, episode_index)
            observed_indices = np.asarray(query_indices["actions"], dtype=np.int64)
            expected_local_indices = np.clip(local_index + offsets, 0, length - 1)
            expected_indices = episode_start + expected_local_indices
            if not np.array_equal(observed_indices, expected_indices):
                raise ValueError(
                    f"LeRobot action delta query crossed episode {episode_index} at local row {local_index}"
                )
            expected_padding = local_index + offsets >= length
            actual_padding = padding["actions_is_pad"].cpu().numpy()
            if not np.array_equal(actual_padding, expected_padding):
                raise ValueError(f"LeRobot action padding is incorrect for episode {episode_index}")
            observed_actions = dataset._query_hf_dataset(query_indices)["actions"].cpu().numpy()
            expected_actions = actions[expected_local_indices]
            if not np.array_equal(observed_actions, expected_actions):
                raise ValueError(
                    f"LeRobot 50-step action query differs at episode {episode_index}, row {local_index}"
                )
    del dataset


def validate(
    dataset_path: str | Path,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate LeRobot files against their source segments and task prompts."""
    path = Path(dataset_path).expanduser().resolve()
    source_root = _resolve_root(root)
    _configure_data_environment()
    if not path.is_dir():
        raise FileNotFoundError(path)
    receipt_path = path / "data_manifest.json"
    info_path = path / "meta" / "info.json"
    if not receipt_path.is_file() or not info_path.is_file():
        raise ValueError(f"Dataset is missing data_manifest.json or meta/info.json: {path}")
    receipt = json.loads(receipt_path.read_text())
    info = json.loads(info_path.read_text())
    if receipt.get("status") != "complete":
        raise ValueError(f"Dataset receipt is incomplete: {path}")
    manifest = receipt.get("source_manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"Dataset receipt is missing its source manifest: {receipt_path}")
    for name in ("source_root",):
        relative = Path(manifest[name])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Source manifest {name} must be relative to TWAM_ROOT: {relative}")

    task = receipt.get("task")
    split = receipt.get("split")
    repo_id = _repo_id(split, task) if split in ("full", "debug_train", "debug_val") and task in TASKS else None
    if repo_id is None or receipt.get("repo_id") != repo_id:
        raise ValueError(f"Dataset repo ID does not match task/split metadata: {receipt.get('repo_id')}")
    if set(receipt.get("features", ())) != set(OUTPUT_FEATURES):
        raise ValueError(f"Converter receipt has unexpected features: {receipt.get('features')}")
    if manifest.get("selected_tasks") != [task]:
        raise ValueError(f"Source manifest task list is inconsistent: {manifest.get('selected_tasks')}")
    segments = manifest.get("segments", [])
    if not segments or any(segment.get("task") != task for segment in segments):
        raise ValueError("Source manifest must contain the selected task's retained segments")
    expected_frames = int(receipt["frame_count"])
    expected_episodes = int(receipt["episode_count"])
    manifest_frames = sum(int(segment["valid_count"]) for segment in segments)
    if len(segments) != expected_episodes or manifest_frames != expected_frames:
        raise ValueError("Converter receipt counts do not match its source segment manifest")
    observed_frames = info.get("total_frames")
    observed_episodes = info.get("total_episodes")
    if observed_frames is None or int(observed_frames) != expected_frames:
        raise ValueError(f"LeRobot total_frames={observed_frames} but expected {expected_frames}: {path}")
    if observed_episodes is None or int(observed_episodes) != expected_episodes:
        raise ValueError(f"LeRobot total_episodes={observed_episodes} but expected {expected_episodes}: {path}")
    if int(info.get("fps", -1)) != FPS:
        raise ValueError(f"LeRobot FPS is {info.get('fps')}; expected {FPS}")
    info_features = info.get("features", {})
    structural_features = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    user_features = set(info_features) - structural_features
    if user_features != set(OUTPUT_FEATURES):
        raise ValueError(f"LeRobot metadata has unexpected user features: {sorted(user_features)}")
    for name, expected in _expected_features().items():
        feature = info_features[name]
        if (
            feature.get("dtype") != expected["dtype"]
            or tuple(feature.get("shape", ())) != tuple(expected["shape"])
            or feature.get("names") != expected["names"]
        ):
            raise ValueError(f"LeRobot feature {name} metadata is {feature}; expected {expected}")

    expected_prompt = manifest.get("task_prompts", {}).get(task)
    if not isinstance(expected_prompt, str) or not expected_prompt.strip():
        raise ValueError("Source manifest does not preserve the exact seen[0] task instruction")
    task_records = _read_jsonlines(path / "meta" / "tasks.jsonl")
    if task_records != [{"task_index": 0, "task": expected_prompt}]:
        raise ValueError(f"LeRobot task metadata does not contain only the expected prompt: {task_records}")
    episode_records = _read_jsonlines(path / "meta" / "episodes.jsonl")
    if len(episode_records) != expected_episodes:
        raise ValueError(f"LeRobot has {len(episode_records)} episode metadata rows, expected {expected_episodes}")

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - installed in the training environment
        raise RuntimeError("PyArrow is required to validate LeRobot episode Parquet files") from exc
    data_pattern = info.get("data_path")
    chunk_size = int(info.get("chunks_size", 0))
    if not isinstance(data_pattern, str) or chunk_size <= 0:
        raise ValueError("LeRobot metadata has an invalid Parquet data path/chunk size")
    source_files = {
        (record["task"], int(record["source_episode_id"])): record
        for record in manifest.get("source_files", [])
    }
    episode_actions: list[np.ndarray] = []
    episode_lengths: list[int] = []
    image_checks = 0
    global_index = 0
    structural_columns = ("timestamp", "frame_index", "episode_index", "index", "task_index")
    wanted_columns = set(OUTPUT_FEATURES) | set(structural_columns)
    for episode_index, (segment, episode_record) in enumerate(zip(segments, episode_records, strict=True)):
        context = f"{task} source episode {segment['source_episode_id']} segment {segment['segment_index']}"
        start = int(segment["start_frame"])
        stop = int(segment["stop_frame"])
        count = stop - start - 1
        if count != int(segment["valid_count"]):
            raise ValueError(f"Invalid source range in {context}: [{start}, {stop})")
        if (
            int(episode_record.get("episode_index", -1)) != episode_index
            or int(episode_record.get("length", -1)) != count
            or episode_record.get("tasks") != [expected_prompt]
        ):
            raise ValueError(f"LeRobot episode metadata is inconsistent for {context}: {episode_record}")
        source_record = source_files.get((task, int(segment["source_episode_id"])))
        if source_record is None:
            raise ValueError(f"Source episode metadata is missing for {context}")
        source_path = source_root / segment["source_path"]
        if source_path.stat().st_size != int(segment["source_size_bytes"]):
            raise ValueError(f"Source file size changed after scan for {context}: {source_path}")

        chunk = episode_index // chunk_size
        relative_data_path = data_pattern.format(episode_chunk=chunk, episode_index=episode_index)
        parquet_path = path / relative_data_path
        if not parquet_path.is_file():
            raise ValueError(f"Missing LeRobot Parquet episode for {context}: {parquet_path}")
        parquet_file = parquet.ParquetFile(parquet_path)
        available_columns = set(parquet_file.schema_arrow.names)
        if available_columns != wanted_columns:
            raise ValueError(f"Unexpected Parquet columns for {context}: {sorted(available_columns)}")
        table = parquet_file.read()
        if table.num_rows != count:
            raise ValueError(f"LeRobot Parquet row count {table.num_rows} != {count} for {context}")
        states = _as_column_array(table, "state", np.float32)
        actions = _as_column_array(table, "actions", np.float32)
        if states.shape != (count, 8) or actions.shape != (count, 8):
            raise ValueError(f"LeRobot state/action array shapes are invalid for {context}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"LeRobot state/action arrays contain NaN or infinity for {context}")
        frame_indices = _as_column_array(table, "frame_index", np.int64)
        episode_indices = _as_column_array(table, "episode_index", np.int64)
        row_indices = _as_column_array(table, "index", np.int64)
        task_indices = _as_column_array(table, "task_index", np.int64)
        timestamps = _as_column_array(table, "timestamp", np.float64)
        if not np.array_equal(frame_indices, np.arange(count)):
            raise ValueError(f"LeRobot frame_index values are inconsistent for {context}")
        if not np.all(episode_indices == episode_index):
            raise ValueError(f"LeRobot episode_index values are inconsistent for {context}")
        if not np.array_equal(row_indices, np.arange(global_index, global_index + count)):
            raise ValueError(f"LeRobot global index values are inconsistent for {context}")
        if not np.all(task_indices == 0):
            raise ValueError(f"LeRobot task_index values are inconsistent for {context}")
        if not np.allclose(
            timestamps,
            np.arange(count) / FPS,
            rtol=0.0,
            atol=LEROBOT_TIMESTAMP_TOLERANCE_SECONDS,
        ):
            max_error = float(np.max(np.abs(timestamps - np.arange(count) / FPS)))
            raise ValueError(
                f"LeRobot timestamps differ from index/FPS by {max_error:.9g}s for {context}; "
                f"tolerance is {LEROBOT_TIMESTAMP_TOLERANCE_SECONDS:.9g}s"
            )

        with h5py.File(source_path, "r") as handle:
            steps = np.asarray(handle["step"][:])
            joints = np.asarray(handle["embodiment/joint"][start:stop], dtype=np.float32)
            approved_bounds = _expected_segment_bounds(task, int(segment["source_episode_id"]), steps)
            segment_index = int(segment["segment_index"])
            if segment_index >= len(approved_bounds) or approved_bounds[segment_index] != (start, stop):
                raise ValueError(f"Source segment boundary changed after scan for {context}")
            observed_bounds = (int(steps[start]), int(steps[stop - 1]))
            expected_bounds = (int(segment["raw_steps"][0]), int(segment["raw_steps"][1]))
            if observed_bounds != expected_bounds:
                raise ValueError(f"Source raw step bounds changed after scan for {context}")
            if not np.array_equal(states, joints[:-1, :8]) or not np.array_equal(actions, joints[1:, :8]):
                raise ValueError(f"LeRobot state/action rows do not match source transitions for {context}")
            for row_index in sorted({0, count - 1}):
                source_frame = start + row_index
                for feature, source_key in (
                    ("image", "observation/head/rgb"),
                    ("wrist_image", "observation/wrist/rgb"),
                ):
                    payload = _embedded_image_bytes(table, feature, row_index, context=context)
                    observed = _decode_png(payload, context=f"{context} {feature} row {row_index}")
                    expected = decode_jpeg_rgb(handle[source_key][source_frame])
                    if observed.shape != (270, 480, 3) or not np.array_equal(observed, expected):
                        raise ValueError(
                            f"LeRobot {feature} PNG differs from the source JPEG pixels for {context}, "
                            f"row {row_index}"
                        )
                    image_checks += 1

        episode_actions.append(actions)
        episode_lengths.append(count)
        global_index += count

    if global_index != expected_frames:
        raise ValueError(f"Validated {global_index} rows but receipt declares {expected_frames}: {path}")
    _validate_delta_queries(path, repo_id, episode_actions, episode_lengths)
    return {
        "path": str(path),
        "repo_id": repo_id,
        "task": task,
        "split": split,
        "status": "valid",
        "frame_count": expected_frames,
        "episode_count": expected_episodes,
        "features": list(OUTPUT_FEATURES),
        "fps": FPS,
        "source_segments_checked": expected_episodes,
        "source_transitions_checked": expected_frames,
        "images_pixel_checked": image_checks,
        "delta_horizon_checked": HORIZON,
    }


def prepare(args: argparse.Namespace) -> Any:
    """Adapter for the project CLI; accepts its parsed subcommand namespace."""
    command = getattr(args, "data_command", None) or getattr(args, "command", None)
    if command == "scan":
        result = scan(
            getattr(args, "root", None),
            getattr(args, "split", "full"),
            tasks=getattr(args, "task", None),
            max_episodes=getattr(args, "max_episodes", None),
        )
        output = getattr(args, "output", None)
        if output:
            atomic_json(Path(output), result)
        return result
    if command == "convert":
        return convert(
            getattr(args, "root", None),
            getattr(args, "output_root", None),
            getattr(args, "split", "full"),
            tasks=getattr(args, "task", None),
            source_manifest=getattr(args, "source_manifest", None),
            max_episodes=getattr(args, "max_episodes", None),
            image_writer_threads=getattr(args, "image_writer_threads", 4),
        )
    if command == "validate":
        dataset = getattr(args, "dataset", None) or getattr(args, "path", None)
        if dataset is None:
            raise ValueError("validate requires a dataset path")
        return validate(dataset, root=getattr(args, "root", None))
    raise ValueError(f"Unsupported data command {command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="data_command", required=True)
    for name in ("scan", "convert"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--root", type=Path)
        sub.add_argument("--split", choices=("full", "debug_train", "debug_val"), default="full")
        sub.add_argument("--task", choices=(*TASKS, "all"), default="all")
        sub.add_argument("--max-episodes", type=int)
        if name == "scan":
            sub.add_argument("--output", type=Path)
        else:
            sub.add_argument("--output-root", type=Path)
            sub.add_argument("--source-manifest", type=Path)
            sub.add_argument("--image-writer-threads", type=int, default=4)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("dataset", type=Path)
    validate_parser.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    if args.data_command == "scan" and args.output is None:
        args.output = project_root() / "openpi" / "reports" / "univtac" / f"data-{args.split}.json"
    result = prepare(args)
    if args.data_command == "scan":
        printed = {
            key: result[key]
            for key in (
                "split", "selected_tasks", "segment_count", "raw_frame_count", "valid_pair_count",
                "max_valid_pairs_per_segment", "partial_scan",
            )
        }
        printed["manifest_path"] = str(args.output)
    else:
        printed = result
    print(json.dumps(printed, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by command-line smoke checks
    raise SystemExit(main())
