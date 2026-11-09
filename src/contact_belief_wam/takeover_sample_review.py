"""Prepare and inspect exactly one existing recovery / normal-success pair.

No simulator, policy inference, optimizer, training manifest enrollment, or raw
trace mutation is performed. The exported manifest is explicitly review-only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from .config import ContactConfig
from .data import AlignedDataset, AlignedProcessor, resize, prepare_aligned_batch
from .paths import PROJECT, WORKSPACE, atomic_json, sha256, write_jsonl
from .takeover import tactile_indices
from .takeover_dataset import FORMAT, validate_trace


def require(value, message):
    if not value:
        raise ValueError(message)


def candidate_ranges(handle, result):
    n = len(handle["actions/clipped_actual_target"])
    if result["kind"] == "natural":
        return [dict(start=0, stop=n, reason="normal_success")]
    require(result["kind"] == "oracle" and result["success"], "Expected successful privileged teacher")
    takeover = int(result["takeover_step"])
    phases = handle["actions/phase"].asstr()[:]
    decisions = handle["actions/decision_step"][:]
    sources = handle["actions/source"].asstr()[:]
    intervals = []
    for t in range(takeover, n):
        require(sources[t] == "oracle", "Unexpected post-takeover control source")
        if intervals and decisions[t] == intervals[-1]["decision_step"]:
            require(phases[t] == intervals[-1]["reason"], "Phase changed without a new decision")
            intervals[-1]["stop"] = t + 1
        else:
            require(decisions[t] == t, "Feedback decision must precede its action labels")
            intervals.append(dict(start=t, stop=t + 1, decision_step=t, reason=str(phases[t])))
    return intervals


def prepare_row(result_path, output):
    result = json.loads(result_path.read_text())
    require(result["success"], "Only the two existing successful branches are review candidates")
    trace = Path(result["trajectory"])
    with h5py.File(trace, "r") as handle:
        checkpoint = Path(str(handle.attrs["checkpoint"]))
        deployment = json.loads((checkpoint / "deployment.json").read_text())
        require(deployment["camera"] == "head_wrist", "Baseline must be dual-view")
        n = len(handle["actions/clipped_actual_target"])
        episode = f"{'recovery' if result['kind'] == 'oracle' else 'normal'}/{result['seed']}"
        with np.load(result_path.parent / "tactile_audit.npz", allow_pickle=False) as audit:
            features = np.asarray(audit["tokens"])
            encoder = json.loads(str(audit["encoder_metadata"]))
            windows = np.asarray(audit["window_indices"])
        require(features.shape == (n + 1, 8, 512), "Tactile feature count/shape mismatch")
        require(np.isfinite(features).all(), "Nonfinite tactile cache")
        require(np.array_equal(windows, np.stack([tactile_indices(t) for t in range(n + 1)])),
                "Audit cache has noncausal/reset tactile windows")
        row = dict(data_format=FORMAT, episode_id=episode, task="insert_HDMI", split="review",
                   source="recovery" if result["kind"] == "oracle" else "original",
                   demonstrator="privileged_oracle" if result["kind"] == "oracle" else "wave0",
                   seed=result["seed"], path=str(trace.resolve()), frames=n + 1, actions=n,
                   segments=[[0, n + 1]], context_range=[0, n + 1],
                   prompt=deployment["prompt"], actual_command_key="actions/clipped_actual_target",
                   policy_supervision_ranges=candidate_ranges(handle, result),
                   supervision_status="candidate_for_review", training_enabled=False,
                   training_approved=False, video_effect_accepted=result["kind"] == "oracle",
                   encoder_metadata=encoder, trace_sha256=sha256(trace),
                   source_result=str(result_path.resolve()), source_result_sha256=sha256(result_path),
                   task_stats=str(checkpoint / "norm_stats.json"),
                   wm_stats=str(checkpoint / "wm_norm_stats.json"),
                   source_cache_sha256=sha256(result_path.parent / "tactile_audit.npz"),
                   takeover_step=result.get("takeover_step"), timing=result.get("timing"),
                   original_policy_flags_unchanged=True,
                   context_semantics="all past observations and actual commands; no reset at BC boundaries")
        row["technical_trace_check"] = validate_trace(handle, row)
    cache = output / "cache" / (episode + ".npz")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, tokens=features.astype(np.float16), raw_step=np.arange(n + 1, dtype=np.int64),
                        frame_index=np.arange(n + 1), window_indices=windows,
                        metadata=json.dumps(encoder), trace_sha256=row["trace_sha256"])
    return row


def inspect_prefix(row):
    result = json.loads(Path(row["source_result"]).read_text())
    if result["kind"] != "oracle":
        return {"applicable": False, "reason": "uninterrupted_normal_rollout"}
    stop = row["takeover_step"] + 1
    with h5py.File(row["path"], "r") as branch, h5py.File(result["reference"], "r") as reference:
        checks = {key: bool(np.array_equal(branch["observations/" + key][:stop],
                                          reference["observations/" + key][:stop]))
                  for key in ("joint_q", "head_rgb", "wrist_rgb", "left_rgb_marker", "right_rgb_marker",
                              "physics_step", "action_count", "object_in_gripper_pose_world")}
        checks["actual_prefix_commands"] = bool(np.array_equal(
            branch["actions/clipped_actual_target"][:stop - 1], reference["actions/clipped_actual_target"][:stop - 1]))
    recheck = json.loads((Path(row["source_result"]).parent / "takeover_restore_check.json").read_text())
    require(all(checks.values()) and recheck["passed"], "Takeover prefix/restored input differs")
    return {"applicable": True, "observations_compared": stop, "checks": checks,
            "strict_at_takeover_passed": True, "strict_forward_replay_passed": result["replay"]["passed"],
            "limitation": "Ten-step cross-branch pixel replay differed; physical replay and takeover input passed. "
                          "This audit does not prove renderer-history independence."}


def audit_dataset(rows, output, tokenizer):
    import torch
    from torch.utils.data import DataLoader
    from .sequence import observation_at, targets_at

    torch.set_num_threads(4)
    cfg = ContactConfig(aligned=True, stage="stage2", mode="full", independent_tactile=True, camera_mode="head_wrist",
                        k=10, chunk_size=50, belief_warmup_min=4, belief_warmup_max=4)
    require(len({sha256(r["task_stats"]) for r in rows}) == 1, "Task normalization differs between traces")
    require(len({sha256(r["wm_stats"]) for r in rows}) == 1, "WM normalization differs between traces")
    kwargs = dict(manifest=rows, data_root=output, cache_root=output / "cache",
                  task_stats=rows[0]["task_stats"], wm_stats=rows[0]["wm_stats"], config=cfg,
                  split="all", tokenizer_path=tokenizer)
    try:
        AlignedDataset(**kwargs)
    except ValueError as exc:
        require("training is disabled" in str(exc), str(exc))
    else:
        raise ValueError("Review-only manifest was accepted without explicit review opt-in")
    ds = AlignedDataset(**kwargs, allow_review=True)
    inverse = AlignedProcessor(rows[0]["task_stats"], None, "")
    inverse_world = AlignedProcessor(rows[0]["wm_stats"], None, "")
    checks = []
    snapshot_dir = output / "samples"
    snapshot_dir.mkdir(exist_ok=True)
    for index, (rid, start, stop, t) in enumerate(ds.index):
        row, episode = rows[rid], ds._load(rid)
        q, commands = episode["joints"], episode["commands"]
        sample = ds[index]
        frames = sample["step"]
        intervals = row["policy_supervision_ranges"]
        eligible_stop = next((r["stop"] for r in intervals if r["start"] <= t < r["stop"]), t)
        count = min(50, eligible_stop - t)
        mask = sample["policy_action_mask"][-1]
        require(np.array_equal(mask, np.arange(50) < count), f"Incorrect policy prefix at {row['seed']}:{t}")
        decoded = inverse.absolute(sample["actions"][-1], q[t])
        if count:
            np.testing.assert_allclose(decoded[:count], commands[t:t + count], atol=2e-6, rtol=1e-6)
            np.testing.assert_allclose(decoded[count:], np.repeat(commands[t + count - 1:t + count],
                                       50 - count, axis=0), atol=2e-6, rtol=1e-6)
        else:
            np.testing.assert_allclose(decoded, np.repeat(q[t:t + 1], 50, axis=0), atol=2e-6, rtol=1e-6)
        history_start = max(0, t - 10)
        history = inverse_world.absolute(sample["executed_actions"][-1], q[history_start])
        np.testing.assert_allclose(history[sample["executed_valid"][-1]], commands[history_start:t],
                                   atol=2e-6, rtol=1e-6)
        np.testing.assert_array_equal(sample["tactile"], episode["tactile"][frames].astype(np.float32))
        np.testing.assert_array_equal(sample["image"][-1], episode["images"]["head"][t].transpose(2, 0, 1) / np.float32(127.5) - 1)
        np.testing.assert_array_equal(sample["wrist_image"][-1], episode["images"]["wrist"][t].transpose(2, 0, 1) / np.float32(127.5) - 1)
        require(frames[-1] == t and np.all(frames <= t), "Noncausal recurrent history")
        np.testing.assert_array_equal(frames[sample["sequence_mask"]],
                                      np.arange(t - min(4, t // 10) * 10, t + 1, 10))
        require(bool(sample["future_mask"][-1]) == (t + 10 <= row["actions"]), "WM transition lost at label boundary")
        require(not any(any(secret in key for secret in ("oracle", "prism", "slot", "truth")) for key in sample),
                "Privileged audit fields reached model batch")
        # Change only excluded future labels; the entire BC input must stay identical.
        poisoned = episode["commands"].copy()
        poison_start = t + count
        episode["commands"][poison_start:] += 123
        try:
            changed = ds[index]
            np.testing.assert_array_equal(sample["actions"][-1], changed["actions"][-1])
            np.testing.assert_array_equal(sample["executed_actions"], changed["executed_actions"])
        finally:
            episode["commands"][:] = poisoned
        if t in {0, 30, 39, 40, 59, 60, 109, 110, row["actions"] - 1}:
            np.savez_compressed(snapshot_dir / f"{row['seed']}-{t:04d}.npz",
                                **{k: v for k, v in sample.items() if isinstance(v, np.ndarray)})
        checks.append(dict(seed=row["seed"], frame=t, policy_valid_prefix=count,
                           context_frames=frames[sample["sequence_mask"]].tolist(),
                           executed_history=[history_start, t],
                           tactile_window=tactile_indices(t).tolist(), wm_future_valid=bool(sample["future_mask"][-1])))
        if t % 100 == 0:
            print(json.dumps({"audit_seed": row["seed"], "frame": t}), flush=True)
    batch = prepare_aligned_batch(next(iter(DataLoader(ds, batch_size=2, shuffle=False))), "cpu")
    obs, target = observation_at(batch, -1), targets_at(batch, -1)
    require(obs.image.shape == obs.wrist_image.shape == (2, 3, 224, 224), "Dual-view batch failure")
    require(target.actions.shape == (2, 50, 32), "Action batch failure")
    atomic_json(output / "loader_checks.json", checks)
    ds.close()
    return dict(anchors_checked=len(checks), policy_supervised_anchors=sum(x["policy_valid_prefix"] > 0 for x in checks),
                actual_command_roundtrip=True, all_boundary_masks_checked=True,
                excluded_future_poison_test_passed=True, recurrent_history_continuous=True,
                causal_tactile_windows=True, dual_view_model_input=True,
                wm_transitions_retained=True, privileged_fields_absent=True,
                review_manifest_training_guard=True, config=cfg.to_dict())


def check_encoder(rows, device="cpu"):
    from .tactile import AnyTouchEncoder
    encoder = AnyTouchEncoder(WORKSPACE / "AnyTouch2", device=device, threads=4)
    checks = []
    for row in rows:
        require(encoder.metadata == row["encoder_metadata"], "Encoder version changed")
        indices = ([34, 38, 40, 60, 110] if row["source"] == "recovery" else [0, 40, row["actions"]])
        with h5py.File(row["path"], "r") as f:
            windows = np.stack([np.stack([np.stack([f[f"observations/{side}_rgb_marker"][int(j)]
                                for j in tactile_indices(t)]) for side in ("left", "right")]) for t in indices])
        actual = encoder(windows)
        with np.load(Path(row["source_result"]).parent / "tactile_audit.npz") as f:
            expected = f["tokens"][indices]
        maximum = float(np.abs(actual - expected).max())
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=5e-4)
        checks.append(dict(seed=row["seed"], frames=indices, max_abs_error=maximum, passed=True))
        print(json.dumps({"encoder_recheck": checks[-1]}), flush=True)
    return checks


def make_frames(row, output):
    directory = output / "frames" / str(row["seed"])
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    with h5py.File(row["path"], "r") as f, np.load(Path(row["source_result"]).parent / "tactile_audit.npz") as z:
        obs, actions = f["observations"], f["actions"]
        q, commands = obs["joint_q"][:], actions["clipped_actual_target"][:]
        phases, decisions = actions["phase"].asstr()[:], actions["decision_step"][:]
        delta = z["cosine_delta"]
        raw_previous = None
        for t in range(row["frames"]):
            head, wrist = resize(obs["head_rgb"][t]), resize(obs["wrist_rgb"][t])
            touch_raw = [obs[f"{s}_rgb_marker"][t] for s in ("left", "right")]
            touch = [np.asarray(Image.fromarray(x).resize((224, 224), Image.Resampling.BILINEAR)) for x in touch_raw]
            mosaic = np.concatenate([np.concatenate([head, wrist], 1), np.concatenate(touch, 1)], 0)
            Image.fromarray(mosaic).save(directory / f"{t:04d}.jpg", quality=92)
            raw_delta = [0., 0.] if raw_previous is None else [float(np.abs(a.astype(float) - b).mean() / 255)
                                                for a, b in zip(touch_raw, raw_previous)]
            raw_previous = touch_raw
            interval = next((i for i in row["policy_supervision_ranges"] if i["start"] <= t < i["stop"]), None)
            valid_count = min(50, interval["stop"] - t) if interval else 0
            records.append(dict(t=t, phase=str(phases[t]) if t < len(commands) else "terminal_success",
                decision=int(decisions[t]) if t < len(commands) else None, q=q[t].tolist(),
                command=commands[t].tolist() if t < len(commands) else None,
                next_q=q[t + 1].tolist() if t < len(commands) else None,
                policy_valid_prefix=valid_count, label_stop=t + valid_count,
                touch_window=tactile_indices(t).tolist(), raw_touch_delta=raw_delta,
                latent_delta=float(delta[t]), physical_step=int(obs["physics_step"][t]),
                substeps=int(actions["substeps_executed"][t]) if t < len(commands) else None,
                audit_only_object_slot_mm=((obs["prism_pose_world"][t, :3] - obs["slot_pose_world"][t, :3]) * 1000).tolist()))
    with (output / f"frames-{row['seed']}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in record.items()})
    return dict(seed=row["seed"], title="真值辅助恢复" if row["source"] == "recovery" else "正常成功",
                actions=row["actions"], ranges=row["policy_supervision_ranges"], frames=records,
                timing=row["timing"], takeover=row["takeover_step"], source_result=row["source_result"],
                source_trace=row["path"], input_prompt=row["prompt"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    pilot = PROJECT / "data/recovery_wave0_v1/insert_HDMI/pilot"
    parser.add_argument("--recovery", type=Path, default=pilot / "oracle/3000001/attempt-05/result.json")
    parser.add_argument("--normal", type=Path, default=pilot / "natural/3000002/result.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/recovery_review_v1/insert_HDMI/pair_3000001_3000002")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_paths = [args.recovery.resolve(), args.normal.resolve()]
    originals = {str(p): sha256(p) for source in source_paths for p in
                 (source, source.parent / "training_contract.json")}
    rows = [prepare_row(path, output) for path in source_paths]
    write_jsonl(output / "review_manifest.jsonl", rows)
    atomic_json(output / "contracts.json", rows)
    prefix = [inspect_prefix(row) for row in rows]
    tokenizer = WORKSPACE / "models/pi05_base/big_vision/paligemma_tokenizer.model"
    loading = audit_dataset(rows, output, tokenizer)
    encoding = check_encoder(rows)
    episodes = [make_frames(row, output) for row in rows]
    immutable = all(sha256(path) == digest for path, digest in originals.items())
    immutable &= all(sha256(row["path"]) == row["trace_sha256"] for row in rows)
    require(immutable, "Source artifacts changed during review")
    report = dict(status="technical_checks_passed", training_enabled=False, expanded_collection=False,
                  training_started=False, raw_sources_unchanged=bool(immutable),
                  episodes=[dict(seed=r["seed"], actions=r["actions"], observations=r["frames"],
                                 candidate_policy_actions=sum(x["stop"] - x["start"] for x in r["policy_supervision_ranges"]),
                                 candidate_ranges=len(r["policy_supervision_ranges"])) for r in rows],
                  loader=loading, tactile_reencoding=encoding, prefix=prefix,
                  limitations=["Privileged teacher labels; no proof that student uses tactile feedback.",
                               "Cross-branch pixel replay issue remains; actual takeover input passes.",
                               "Two existing episodes only; no generalization or closed-loop policy evaluation."],
                  source_hashes=originals,
                  implementation_hashes={name: sha256(Path(__file__).with_name(name)) for name in
                                         ("data.py", "takeover_dataset.py", "takeover_sample_review.py", "takeover_review_page.py")})
    atomic_json(output / "validation.json", report)
    (output / "review_data.js").write_text("window.REVIEW=" + json.dumps(dict(episodes=episodes, validation=report),
                          ensure_ascii=False, allow_nan=False) + ";\n", encoding="utf-8")
    from .takeover_review_page import PAGE
    (output / "index.html").write_text(PAGE, encoding="utf-8")
    print(json.dumps(dict(output=str(output), episodes=report["episodes"], status=report["status"])), flush=True)


if __name__ == "__main__":
    main()
