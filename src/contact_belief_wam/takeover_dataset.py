"""Command-aligned takeover samples with independent history and BC boundaries.

The raw trace is immutable. A separate, explicitly reviewed manifest proposes
policy ranges; original diagnostic ``policy_eligible`` flags are not rewritten.
Only observable inputs are returned to AlignedDataset's normal training path.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .feature_cache import load_tactile_tokens
from .paths import sha256
from .takeover import policy_prefix, tactile_indices


FORMAT = "wave0-hdmi-contact-v1"


def validate_contract(row, *, allow_review=False):
    if row.get("training_enabled") is not True and not allow_review:
        raise ValueError("Review-only takeover manifest: training is disabled")
    n = int(row["frames"]) - 1
    if n < 1 or row.get("context_range") != [0, n + 1] or row.get("segments") != [[0, n + 1]]:
        raise ValueError("Takeover context must retain the complete N+1 observations")
    if row.get("actual_command_key") != "actions/clipped_actual_target":
        raise ValueError("Takeover labels must use actual dispatched commands")
    previous = 0
    for interval in row["policy_supervision_ranges"]:
        start, stop = int(interval["start"]), int(interval["stop"])
        if not previous <= start < stop <= n:
            raise ValueError("Invalid or overlapping policy supervision ranges")
        if "decision_step" in interval and int(interval["decision_step"]) != start:
            raise ValueError("A feedback-dependent range must begin at its decision observation")
        previous = stop


def validate_trace(handle, row):
    """Validate o[t] -> actual a[t] -> o[t+1], including terminal partial action."""
    n = int(row["frames"]) - 1
    obs, actions = handle["observations"], handle["actions"]
    if handle.attrs.get("schema_version") != FORMAT:
        raise ValueError("Unsupported takeover trace schema")
    if len(obs["joint_q"]) != n + 1 or actions["clipped_actual_target"].shape != (n, 8):
        raise ValueError("Expected N actual commands and N+1 observations")
    for key in ("joint_q", "head_rgb", "wrist_rgb", "left_rgb_marker", "right_rgb_marker"):
        if key not in obs or len(obs[key]) != n + 1 or not obs[key + "_valid"][:].all():
            raise ValueError(f"Missing or invalid model observation: {key}")
    if not obs["observation_valid"][:].all() or not actions["action_success"][:].all():
        raise ValueError("Invalid observation/action in takeover trace")
    if not bool(handle.attrs["calibration_valid"]) or not actions["calibration_valid"][:].all():
        raise ValueError("Invalid action calibration")
    if not np.isfinite(obs["joint_q"][:]).all() or not np.isfinite(actions["clipped_actual_target"][:]).all():
        raise ValueError("Nonfinite joint/command values")
    substeps = actions["substeps_executed"][:]
    terminal_success = handle.attrs["terminal_status"] == "success"
    if not (np.all(substeps[:-1] == 2) and (substeps[-1] == 2 or terminal_success and substeps[-1] == 1)):
        raise ValueError("Unexpected physical substeps per strategy action")
    for obs_key, action_prefix in (("physics_step", "physics_step"), ("action_count", "action_count")):
        clock = obs[obs_key][:]
        begin, end = actions[action_prefix + "_begin"][:], actions[action_prefix + "_end"][:]
        if not (np.array_equal(clock[:-1], begin) and np.array_equal(clock[1:], end)
                and np.array_equal(end - begin, substeps)):
            raise ValueError(f"Discontinuous observation/action clock: {obs_key}")
    decisions = actions["decision_step"][:]
    if np.any(decisions < 0) or np.any(decisions > np.arange(n)):
        raise ValueError("Action decision references a future observation")
    for interval in row["policy_supervision_ranges"]:
        if "decision_step" in interval:
            start, stop = interval["start"], interval["stop"]
            if not np.all(decisions[start:stop] == interval["decision_step"]):
                raise ValueError("Policy range crosses a new feedback decision")
    return {"observations": n + 1, "actions": n, "physics_steps": int(substeps.sum()),
            "terminal_success": terminal_success, "terminal_substeps": int(substeps[-1]),
            "command_state_max_difference": float(np.abs(
                actions["clipped_actual_target"][:] - obs["joint_q"][1:]).max())}


def load_episode(path, row, cache_root, *, cameras=("head", "wrist")):
    from .data import resize

    if row.get("model_input_cache"):
        root = Path(row["model_input_cache"])
        meta = json.loads((root / "metadata.json").read_text())
        if (meta.get("schema") != "takeover-model-input-v1" or meta["trace_sha256"] != row["trace_sha256"]
                or meta["frames"] != row["frames"] or not set(cameras) <= set(meta["cameras"])):
            raise ValueError("Model-input cache differs from the prepared trace/view contract")
        # Preparation verifies checksums once. Workers read only selected frames,
        # without decompressing and hashing a 0.3–1 GB diagnostic HDF5 per draw.
        joints = np.load(root / "joints.npy", mmap_mode="r")
        commands = np.load(root / "commands.npy", mmap_mode="c")
        images = {camera: np.load(root / f"{camera}.npy", mmap_mode="r") for camera in cameras}
        if joints.shape != (row["frames"], 8) or commands.shape != (row["frames"] - 1, 8):
            raise ValueError("Invalid cached observation/command count")
        if any(p.shape != (row["frames"], 224, 224, 3) or p.dtype != np.uint8 for p in images.values()):
            raise ValueError("Invalid cached RGB shape or dtype")
    else:
        with h5py.File(path, "r") as handle:
            validate_trace(handle, row)
            joints = np.asarray(handle["observations/joint_q"][:], np.float32)
            commands = np.asarray(handle["actions/clipped_actual_target"][:], np.float32)
            # Trace pixels are raw numerical RGB. Do not JPEG-decode or swap channels.
            images = {camera: np.stack([resize(frame) for frame in handle[f"observations/{camera}_rgb"]])
                      for camera in cameras}
        if sha256(path) != row["trace_sha256"]:
            raise ValueError("Raw trace changed after the review contract was created")
    steps = np.arange(len(joints), dtype=np.int64)
    cache = Path(cache_root) / (row["episode_id"] + ".npz")
    with np.load(cache, allow_pickle=False) as features:
        expected = np.stack([tactile_indices(t) for t in steps])
        if not np.array_equal(features["window_indices"], expected):
            raise ValueError("Tactile windows must remain causal and continuous through takeover")
        if json.loads(str(features["metadata"])) != row["encoder_metadata"]:
            raise ValueError("Tactile encoder/preprocessing differs from reviewed contract")
        if str(features["trace_sha256"]) != row["trace_sha256"]:
            raise ValueError("Tactile cache is bound to a different trace")
    tactile = load_tactile_tokens(cache_root, row["episode_id"], steps)
    if not np.isfinite(tactile).all():
        raise ValueError("Nonfinite tactile features")
    return dict(joints=joints, commands=commands, images=images, tactile=tactile)


def sample_episode(dataset, row, episode, start, stop, current, rng, index):
    """Produce the same tensors as AlignedDataset, using explicit actual commands."""
    cfg = dataset.config
    joints, commands = episode["joints"], episode["commands"]
    n = len(commands)
    warmup = min(int(rng.integers(cfg.belief_warmup_min, cfg.belief_warmup_max + 1)),
                 (current - start) // cfg.k)
    length = cfg.belief_warmup_max + 1
    sequence_mask = np.arange(length) >= length - warmup - 1
    frames = np.clip(current - np.arange(length - 1, -1, -1) * cfg.k, start, stop - 1)
    padding = length - warmup - 1
    if padding:
        frames[:padding] = frames[padding]
    future_indices = np.minimum(frames + cfg.k, stop - 1)
    future_mask = (frames + cfg.k < stop) & sequence_mask
    action_indices = frames[:, None] + np.arange(cfg.chunk_size)[None, :]
    action_mask = (action_indices < n) & sequence_mask[:, None]
    executed_indices = frames[:, None] - cfg.k + np.arange(cfg.k)[None, :]
    executed_valid = (executed_indices >= start) & (executed_indices < frames[:, None]) & sequence_mask[:, None]
    executed_indices = np.clip(executed_indices, start, n - 1)
    processor = dataset.processors[row["task"]]
    normalized_state = processor.normalize(joints[frames], "state")
    low, high = [np.asarray(dataset.wm_norm["state"][k]) for k in ("q01", "q99")]
    actions, world_actions, executed, masks, tokens, token_masks = [], [], [], [], [], []
    for position, frame in enumerate(frames):
        allowed, mask = policy_prefix(commands, int(frame), row["policy_supervision_ranges"], cfg.chunk_size)
        if not mask.any():
            allowed[:] = joints[frame]  # finite observable padding, never an excluded command
        actions.append(processor.actions(allowed, joints[frame]))
        masks.append(mask & sequence_mask[position])
        # WM targets retain genuine dynamics, including unsuccessful prefix actions.
        targets = commands[np.minimum(action_indices[position], n - 1)]
        world_actions.append(dataset._world_actions(targets, joints[frame], dataset.wm_norm))
        anchor = joints[max(start, int(frame) - cfg.k)]
        history = commands[executed_indices[position]].copy()
        history[~executed_valid[position]] = anchor
        executed.append(dataset._world_actions(history, anchor, dataset.wm_norm)[:, :8])
        if processor.sp is None:
            ids, token_mask = np.zeros(1, np.int64), np.zeros(1, bool)
        else:
            ids, token_mask = processor.tokens(normalized_state[position])
        tokens.append(ids)
        token_masks.append(token_mask)
    views = {}
    for camera, pixels in episode["images"].items():
        key = "image" if camera == "head" else "wrist_image"
        views[key] = pixels[frames].transpose(0, 3, 1, 2).astype(np.float32) / 127.5 - 1
        views["future_" + key] = pixels[future_indices].transpose(0, 3, 1, 2).astype(np.float32) / 127.5 - 1
    return {**views, "visual_frame_index": frames.copy(), "future_visual_frame_index": future_indices,
            "state": normalized_state.astype(np.float32),
            "world_state": ((joints[frames] - low) / (high - low + 1e-6) * 2 - 1).astype(np.float32),
            "actions": np.stack(actions), "world_actions": np.stack(world_actions),
            "executed_actions": np.stack(executed), "executed_valid": executed_valid,
            "tactile": episode["tactile"][frames].astype(np.float32),
            "future_tactile": episode["tactile"][future_indices].astype(np.float32),
            "tokens": np.stack(tokens), "token_mask": np.stack(token_masks),
            "action_mask": action_mask, "policy_action_mask": np.stack(masks),
            "future_mask": future_mask, "sequence_mask": sequence_mask,
            "step": frames.copy(), "sample_index": index, "source": row["source"],
            "task": row["task"], "prompt": row["prompt"], "episode_id": row["episode_id"]}
