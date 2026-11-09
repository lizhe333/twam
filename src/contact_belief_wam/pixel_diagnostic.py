"""Audit the existing branch pixel mismatch without changing training data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .data import resize
from .paths import PROJECT, WORKSPACE, atomic_json, sha256
from .takeover import tactile_indices


def cosine_distance(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    cosine = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)
    return np.mean(1 - np.clip(cosine, -1, 1), axis=-1)


def depth_difference(a, b, far_mm=34.):
    a, b = np.asarray(a, float), np.asarray(b, float)
    delta = np.abs(a - b)
    edge = (np.isclose(a, far_mm, atol=1e-5) != np.isclose(b, far_mm, atol=1e-5))
    outliers = np.argwhere(delta > .01)
    return dict(rmse_mm=float(np.sqrt(np.mean(delta ** 2))), mae_mm=float(delta.mean()),
                p99_mm=float(np.quantile(delta, .99)), pixels=int(delta.size),
                outlier_count_over_0_01mm=int(len(outliers)),
                outlier_fraction=float((delta > .01).mean()),
                all_outliers_are_far_clip_switches=bool(np.all(edge[delta > .01])),
                non_clip_rmse_mm=float(np.sqrt(np.mean(delta[~edge] ** 2))),
                far_clip_switches=[dict(y=int(y), x=int(x), before_mm=float(a[y, x]), after_mm=float(b[y, x]))
                                   for y, x in outliers[:20]])


def rgb_difference(a, b):
    delta = np.abs(a.astype(float) - b.astype(float))
    return dict(mae_levels=float(delta.mean()), rmse_levels=float(np.sqrt(np.mean(delta ** 2))),
                p99_levels=float(np.quantile(delta, .99)), fraction_over_10=float((delta > 10).mean()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT / "data/recovery_wave0_v1/insert_HDMI/pilot/oracle/3000001/attempt-05")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/recovery_review_v1/insert_HDMI/pair_3000001_3000002/pixel_diagnosis")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    result = json.loads((args.source / "result.json").read_text())
    probe = next((args.source / "restore_probe/trace").rglob("trajectory.h5"))
    evidence = result["new_contact_evidence"]
    stats = []
    with h5py.File(result["reference"], "r") as ref, h5py.File(probe, "r") as replay:
        a, b = ref["observations"], replay["observations"]
        for i in range(11):
            rgb = {key: rgb_difference(a[key][40 + i], b[key][i]) for key in
                   ("head_rgb", "wrist_rgb", "left_rgb_marker", "right_rgb_marker")}
            resized = {key: rgb_difference(resize(a[key][40 + i]), resize(b[key][i]))
                       for key in ("head_rgb", "wrist_rgb")}
            depth = {key: depth_difference(a[key][40 + i], b[key][i]) for key in ("left_depth", "right_depth")}
            stats.append(dict(step=40 + i, rgb=rgb, model_rgb_224=resized, depth=depth))
        # Match every replay frame against nearby reference states. This tests
        # a timestamp offset independently of the renderer-state hypothesis.
        time_alignment = {}
        for key in ("joint_q", "left_marker", "right_marker", "head_rgb", "wrist_rgb"):
            offsets = {}
            for lag in range(-3, 4):
                errors = [float(np.abs(a[key][40 + i + lag].astype(float) - b[key][i]).mean())
                          for i in range(1, 11)]
                offsets[str(lag)] = float(np.mean(errors))
            time_alignment[key] = dict(errors_by_reference_offset=offsets,
                                        best_offset=int(min(offsets, key=offsets.get)))
        raw_refs = {s: np.mean([a[f"{s}_rgb_marker"][t].astype(float) for t in evidence["baseline_indices"]], axis=0)
                    for s in ("left", "right")}
        all_windows, raw_changes = [], []
        for restored in (False, True):
            variant, scores = [], []
            def touch(side, t):
                return b[f"{side}_rgb_marker"][t - 40] if restored and t >= 40 else a[f"{side}_rgb_marker"][t]
            for t in range(40, 51):
                indices = tactile_indices(t)
                variant.append(np.stack([np.stack([touch(s, int(j)) for j in indices]) for s in ("left", "right")]))
                scores.append(max(float(np.abs(touch(s, int(j)).astype(float) - raw_refs[s]).mean() / 255)
                                  for s in ("left", "right") for j in indices))
            all_windows.append(np.stack(variant))
            raw_changes.append(scores)
    from .tactile import AnyTouchEncoder
    encoder = AnyTouchEncoder(WORKSPACE / "AnyTouch2", device="cpu", threads=4)
    features = []
    for variant in all_windows:
        features.append(np.concatenate([encoder(variant[i:i + 4]) for i in range(0, len(variant), 4)]))
    features = np.stack(features)
    with np.load(args.source / "tactile_audit.npz") as old:
        baseline = old["tokens"][evidence["baseline_indices"]].mean(0)
        contact_signal = float(cosine_distance(old["tokens"][40], old["tokens"][36]))
        before_contact = old["tokens"][34:41]
    signal = [cosine_distance(z, baseline) for z in features]
    gates = [np.asarray(raw) >= evidence["raw_threshold"] for raw in raw_changes]
    gates = [gate & (delta > evidence["latent_threshold"]) for gate, delta in zip(gates, signal)]
    mismatch = cosine_distance(features[0], features[1])
    feature_report = dict(encoder_metadata=encoder.metadata, observation_steps=list(range(40, 51)),
        branch_cosine_distance=mismatch.tolist(), branch_feature_rmse=np.sqrt(((features[0] - features[1]) ** 2).mean((1, 2))).tolist(),
        real_contact_cosine_change_36_to_40=contact_signal,
        max_branch_distance_over_contact_change=float(mismatch.max() / max(contact_signal, 1e-12)),
        baseline_new_contact_score=signal[0].tolist(), replay_new_contact_score=signal[1].tolist(),
        raw_threshold=evidence["raw_threshold"], latent_threshold=evidence["latent_threshold"],
        baseline_contact_gate=gates[0].tolist(), replay_contact_gate=gates[1].tolist(),
        contact_gate_identical=bool(np.array_equal(gates[0], gates[1])),
        first_takeover_input_distance=float(mismatch[0]), no_future_frames=True)
    np.savez_compressed(args.output / "existing_branch_features.npz", original=features[0], replay=features[1],
                        baseline_before_contact=before_contact, steps=np.arange(40, 51))
    output = dict(existing_pair_only=True, collection_enabled=False, training_enabled=False,
        source_hashes={str(p): sha256(p) for p in (Path(result["reference"]), probe, Path(result["trajectory"]))},
        pixel_metrics=stats, time_alignment=time_alignment, tactile_features=feature_report,
        depth_is_student_input=False, student_tactile_input="AnyTouch2 from raw rgb_marker, not depth or simulator pose",
        original_strict_checks_unchanged=True)
    atomic_json(args.output / "existing_pair_audit.json", output)
    print(json.dumps(dict(max_feature_cosine=float(mismatch.max()), contact_signal=contact_signal,
                         contact_gate_identical=feature_report["contact_gate_identical"],
                         best_reference_offsets={k:v["best_offset"] for k,v in time_alignment.items()})), flush=True)


if __name__ == "__main__":
    main()
