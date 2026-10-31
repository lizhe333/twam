"""Offline, causal tactile audit and self-contained review artifacts."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np

from .paths import WORKSPACE, atomic_json
from .takeover import tactile_indices, policy_prefix, fingerprint

AUDITOR_SOURCE = fingerprint([Path(__file__)])


def point_triangle_distance(points, triangles):
    """Minimum Euclidean distance; bounded caller batches avoid large products."""
    p = np.asarray(points, float)[:, None, :]
    tri = np.asarray(triangles, float)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac, ap = b-a, c-a, p-a
    d00 = (ab*ab).sum(-1); d01 = (ab*ac).sum(-1); d11 = (ac*ac).sum(-1)
    d20 = (ap*ab).sum(-1); d21 = (ap*ac).sum(-1)
    den = d00*d11-d01*d01
    v = (d11*d20-d01*d21)/np.maximum(den, 1e-30)
    w = (d00*d21-d01*d20)/np.maximum(den, 1e-30)
    inside = (v >= 0) & (w >= 0) & (v+w <= 1) & (den > 1e-30)
    best = np.where(inside, ((ap-v[..., None]*ab-w[..., None]*ac)**2).sum(-1), np.inf)
    for x, y in ((a,b), (b,c), (c,a)):
        edge = y-x
        fraction = np.clip(((p-x)*edge).sum(-1)/np.maximum((edge*edge).sum(-1), 1e-30), 0, 1)
        best = np.minimum(best, ((p-x-fraction[...,None]*edge)**2).sum(-1))
    return float(np.sqrt(best.min()))


def contact_distances(handle):
    """Native surface vertex/triangle distance; an audit proxy, not force.

Both directions are checked. Edge-only contacts can remain unconfirmed; the
collector must not invent a collision label when evidence is unavailable.
"""
    from scipy.spatial import cKDTree
    result = []
    obs = handle["observations"]
    for i in range(len(obs["joint_q"])):
        vertices = [obs[f"{name}_vertices_world"][i] for name in ("prism", "slot")]
        if not all(np.isfinite(x).all() for x in vertices):
            result.append(float("nan")); continue
        # Conservative AABB distance culls far-apart surfaces without labeling contact.
        lower = np.maximum(vertices[0].min(0)-vertices[1].max(0), vertices[1].min(0)-vertices[0].max(0))
        if np.linalg.norm(np.maximum(lower, 0)) > .003:
            result.append(float(np.linalg.norm(np.maximum(lower, 0)))); continue
        nearest = float(cKDTree(vertices[1]).query(vertices[0], k=1)[0].min())
        if nearest <= .0005:
            # A close native surface-vertex pair proves proximity directly;
            # do not perform quadratic triangle queries at every contact frame.
            result.append(nearest); continue
        distances = []
        for j, name in enumerate(("prism", "slot")):
            triangles = vertices[j][handle[f"{name}_triangles"][:]]
            other = vertices[1-j]
            selected = ((triangles.max(1) >= other.min(0)-.003) &
                        (triangles.min(1) <= other.max(0)+.003)).all(1)
            triangles = triangles[selected]
            if not len(triangles):
                continue
            # Batch small point sets to bound closest_point_naive's temporary arrays.
            points = other[((other >= vertices[j].min(0)-.003) &
                            (other <= vertices[j].max(0)+.003)).all(1)]
            for offset in range(0, len(points), 32):
                distances.append(point_triangle_distance(points[offset:offset+32], triangles))
        result.append(min(distances+[nearest]))
    return np.asarray(result)


def latent_trace(handle, encoder, batch_size=8):
    obs = handle["observations"]
    n = len(obs["joint_q"])
    features = []
    for begin in range(0, n, batch_size):
        windows = []
        for t in range(begin, min(n, begin+batch_size)):
            # h5py fancy indices reject repeated padding indices; read individually.
            windows.append(np.stack([np.stack([obs[f"{side}_rgb_marker"][int(i)] for i in tactile_indices(t)])
                                     for side in ("left", "right")]))
        features.append(encoder(np.stack(windows)))
    z = np.concatenate(features)
    reference = z[:8].mean(0)
    cosine = (z*reference).sum(-1) / np.maximum(np.linalg.norm(z, axis=-1)*np.linalg.norm(reference, axis=-1), 1e-12)
    delta = np.mean(1-np.clip(cosine, -1, 1), axis=-1)
    threshold = max(float(delta[:8].mean()+3*delta[:8].std()), 1e-6)
    return z, delta, threshold


def precontact_evidence(handle, z, contact, decision):
    """New contact information relative to the immediately preceding grasp.

An already elevated signal from an earlier approach/grip adjustment is not new
evidence. This audit reference uses only frames before external contact.
"""
    obs = handle["observations"]
    baseline = list(range(contact-8, contact))
    refs = {s:np.mean([obs[f"{s}_rgb_marker"][t].astype(float) for t in baseline],axis=0)
            for s in ("left","right")}
    raw = np.zeros(len(z))
    for t in range(contact-8,decision+1):
        raw[t] = max(float(np.abs(obs[f"{s}_rgb_marker"][t].astype(float)-refs[s]).mean()/255)
                     for s in ("left","right"))
    reference = z[baseline].mean(0)
    cosine = (z*reference).sum(-1)/np.maximum(np.linalg.norm(z,axis=-1)*np.linalg.norm(reference,axis=-1),1e-12)
    delta = np.mean(1-np.clip(cosine,-1,1),axis=-1)
    raw_threshold = float(raw[baseline].mean()+max(.01,3*raw[baseline].std()))
    latent_threshold = float(delta[baseline].mean()+max(1e-6,3*delta[baseline].std()))
    return dict(baseline_indices=baseline,raw_changes=raw.tolist(),raw_threshold=raw_threshold,
                latent_changes=delta.tolist(),latent_threshold=latent_threshold)


def progress_proxy_audit(handle, result, distances):
    """Audit-only falsification of the controller's FK-progress assumption.

    Ground truth never feeds a takeover time or a correction direction. A
    contradiction stops collection for diagnosis instead of treating arm travel
    through a slipping grasp as evidence of insertion or successful recovery.
    """
    obs = handle["observations"]
    ee = np.asarray(obs["observed_fk_world"])[:, 2]
    object_z = np.asarray(obs["prism_pose_world"])[:, 2] - np.asarray(obs["slot_pose_world"])[:, 2]
    grasp = np.asarray(obs["object_in_gripper_pose_world"])[:, :3]
    contradictions = []
    for event in result["detector"]["events"]:
        stop, start = event["step"], event["step"]-10
        if stop >= len(ee) or not np.any(np.asarray(distances[start:stop+1]) <= .0005):
            continue
        ee_progress = float(ee[start]-ee[stop])*1000
        object_progress = float(object_z[start]-object_z[stop])*1000
        relative_slip = float(np.linalg.norm(grasp[stop]-grasp[start]))*1000
        if (ee_progress >= .5 and event["progress_ratio"] >= .8
                and object_progress < .2*ee_progress and relative_slip >= .5
                and event["tactile_sustained"]):
            contradictions.append(dict(start=start, stop=stop, ee_progress_mm=ee_progress,
                object_progress_mm=object_progress, relative_grasp_motion_mm=relative_slip,
                command_tracking_ratio=event["progress_ratio"], detector_status=event["status"]))
    return dict(usable_for_this_failure=not bool(contradictions), contradictions=contradictions,
                audit_only=True, online_ground_truth_used=False,
                reason="arm_motion_with_object_stall_and_grasp_slip" if contradictions else "no_contradiction_observed")


def temporal_audit(result, distances, latent_delta, latent_threshold, *, raw_scores=None, raw_threshold=None):
    decision = result.get("takeover_step")
    if decision is None:
        return dict(passed=bool(result.get("success")), reason="normal_success" if result.get("success") else "baseline_failed")
    scores = np.asarray(result["detector"]["scores"] if raw_scores is None else raw_scores)
    raw_threshold = result["detector"]["tactile_threshold"] if raw_threshold is None else raw_threshold
    contact = np.flatnonzero(np.asarray(distances[:decision+1]) <= .0005)
    if not len(contact):
        return dict(passed=False, reason="native_surface_contact_not_confirmed")
    tc = int(contact[0])
    if tc < 8:
        return dict(passed=False, reason="no_verified_precontact_calibration")
    # Evidence must be present in the actual four-frame model window, not just
    # elsewhere in the video or a detector's longer accumulation window.
    valid = [t for t in range(tc, decision+1)
             if scores[tactile_indices(t)].max() >= raw_threshold and latent_delta[t] > latent_threshold]
    if not valid or decision not in valid:
        return dict(passed=False, reason="no_new_tactile_evidence_in_decision_input")
    recovery = next((i for i, r in enumerate(result["action_metadata"]) if r["source"] in ("teacher", "oracle")), None)
    order = recovery is not None and tc <= valid[0] <= decision <= recovery
    return dict(passed=bool(order), t_contact=tc, t_tactile=valid[0], t_decision=decision,
                t_recovery=recovery, index_units="strategy_observation_index",
                tactile_window_indices=tactile_indices(decision).tolist(),
                contact_evidence="native_UIPC_surface_distance_le_0.5mm_plus_stall_and_tactile",
                contact_is_force_measurement=False, human_contact_review_required=True)


def render(handle, result, out, delta, distances):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    obs = handle["observations"]
    decision = result.get("takeover_step")
    n = len(obs["joint_q"])
    def panel(t):
        images = [obs[k][t] for k in ("head_rgb", "wrist_rgb", "left_rgb_marker", "right_rgb_marker")]
        images = [cv2.resize(x, (320, 240)) for x in images]
        canvas = np.concatenate([np.concatenate(images[:2], 1), np.concatenate(images[2:], 1)], 0)
        source = "initial" if t==0 else result["action_metadata"][t-1]["phase"]
        cv2.putText(canvas, f"seed={result['seed']} t={t} {source}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 1)
        return canvas
    video = out/"preview.mp4"
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480",
               "-r", "20", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        for t in sorted(set(range(0, n, 3)) | {n-1}):
            process.stdin.write(np.ascontiguousarray(panel(t)).tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("Preview encoder failed")
    figure, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(result["detector"]["scores"], label="raw tactile change")
    axes[0].axhline(result["detector"].get("tactile_threshold") or 0, ls="--")
    axes[1].plot(delta, label="AnyTouch2 cosine change")
    # Audit truth is visibly separated from the controller's observable proxy.
    ee = obs["observed_fk_world"][:, 2]
    object_z = obs["prism_pose_world"][:, 2]-obs["slot_pose_world"][:, 2]
    axes[2].plot((ee[0]-ee)*1000, label="EE progress proxy mm")
    commanded = handle["actions/command_q_fk_ee_world"][:, 2]
    axes[2].plot(np.arange(1,len(commanded)+1),(ee[0]-commanded)*1000,alpha=.5,label="commanded EE depth mm")
    axes[2].plot((object_z[0]-object_z)*1000, label="object progress audit mm")
    axes[3].plot(distances*1000, label="native surface distance mm")
    axes[3].axhline(.5, ls="--")
    for ax in axes:
        if decision is not None: ax.axvline(decision, color="red", ls="--")
        ax.legend(); ax.grid(alpha=.2)
    axes[-1].set_xlabel("strategy observation index")
    figure.tight_layout(); figure.savefig(out/"curves.png", dpi=120); plt.close(figure)
    if decision is not None:
        cv2.imwrite(str(out/"decision.png"), cv2.cvtColor(panel(decision), cv2.COLOR_RGB2BGR))
        window = np.concatenate([np.concatenate([cv2.resize(obs[f"{s}_rgb_marker"][int(t)], (224,224))
                  for t in tactile_indices(decision)], 1) for s in ("left", "right")], 0)
        cv2.imwrite(str(out/"decision_touch_window.png"), cv2.cvtColor(window, cv2.COLOR_RGB2BGR))


def audit_result(path, encoder):
    path = Path(path); out = path.parent
    result = json.loads(path.read_text())
    with h5py.File(result["trajectory"], "r") as handle:
        n = len(handle["actions/clipped_actual_target"])
        valid = (len(handle["observations/joint_q"]) == n+1 and n <= 600
                 and bool(handle.attrs["calibration_valid"]) and bool(handle["observations/observation_valid"][:].all())
                 and bool(handle["actions/action_success"][:].all()))
        substeps = handle["actions/substeps_executed"][:]
        valid &= bool(np.all(substeps[:-1] == 2) and (n == 0 or substeps[-1] == 2 or (substeps[-1] == 1 and result.get("success"))))
        z, delta, threshold = latent_trace(handle, encoder)
        distances = contact_distances(handle)
        result["progress_proxy_audit"] = progress_proxy_audit(handle, result, distances)
        np.savez_compressed(out/"tactile_audit.npz", tokens=z, cosine_delta=delta,
            window_indices=np.stack([tactile_indices(t) for t in range(n+1)]),
            surface_distance_m=distances, encoder_metadata=json.dumps(encoder.metadata))
        decision = result.get("takeover_step")
        timing = None
        if decision is not None:
            contacts = np.flatnonzero(distances[:decision+1] <= .0005)
            if len(contacts) and contacts[0] >= 8:
                evidence = precontact_evidence(handle,z,int(contacts[0]),decision)
                result["new_contact_evidence"] = evidence
                timing = temporal_audit(result,distances,np.asarray(evidence["latent_changes"]),evidence["latent_threshold"],
                                        raw_scores=evidence["raw_changes"],raw_threshold=evidence["raw_threshold"])
        if timing is None:
            timing = temporal_audit(result, distances, delta, threshold)
        oracle = result.get("kind") == "oracle"
        result.update(timing=timing, latent_threshold=threshold, encoder=encoder.metadata,
                      alignment_valid=bool(valid), qualified=bool(valid and result.get("success") and timing["passed"] and not oracle),
                      user_approved=False, auditor_source_hashes=AUDITOR_SOURCE)
        if oracle:
            physical = result.get("physical_diagnostic_requested") and result.get("physical_replay", {}).get("passed") and result.get("physical_takeover_restore", {}).get("passed")
            result["oracle_diagnostic_success"] = bool(valid and result.get("success") and (result.get("replay", {}).get("passed") or physical))
            result["progress_proxy_audit"]["applies_to"] = "passive_detector_only_not_the_privileged_oracle"
            result["timing"]["does_not_establish_observation_only_teacher"] = True
            result["excluded_from_training"] = True
        if decision is not None:
            # Include enough history for current Belief warmup without resetting at takeover.
            start = max(0, decision-40)
            np.savez_compressed(out/"decision_input.npz", head=handle["observations/head_rgb"][decision],
                wrist=handle["observations/wrist_rgb"][decision], state=handle["observations/joint_q"][decision],
                tactile_window_indices=tactile_indices(decision), tactile_tokens=z[decision],
                executed_actions=handle["actions/clipped_actual_target"][start:decision],
                context_indices=np.arange(start, decision+1), context_tactile=z[start:decision+1])
        commands = handle["actions/clipped_actual_target"][:]
        for anchor in range(n):
            targets, mask = policy_prefix(commands, anchor, result["policy_supervision_ranges"])
            if mask.any() and not np.isfinite(targets).all():
                raise RuntimeError("Invalid policy target prefix")
        render(handle, result, out, delta, distances)
    atomic_json(path, result)
    atomic_json(out/"training_contract.json", dict(schema_version="takeover-supervision-v1",
        trajectory=result["trajectory"], context_range=result["context_range"],
        policy_supervision_ranges=result["policy_supervision_ranges"],
        actual_command_key="actions/clipped_actual_target", future_targets_must_use="policy_prefix",
        training_enabled=False, user_approved=False))
    body = html.escape(json.dumps({k:v for k,v in result.items() if k in ("seed","kind","status","strategy_actions","timing","qualified","progress_proxy_audit")}, ensure_ascii=False, indent=2))
    extra = ""
    if result.get("takeover_step") is not None:
        extra = (f'<p><a href="../../natural/{result["seed"]}/index.html">Corresponding natural WAVE-0 failure</a></p>'
                 '<p><img width="720" src="decision.png"></p><p><img width="896" src="decision_touch_window.png"></p>'
                 '<p><a href="decision_input.npz">Causal decision input and executed history</a></p>')
    (out/"index.html").write_text(f'<!doctype html><meta charset="utf-8"><title>HDMI pilot</title><h1>HDMI {result["seed"]}: {result["kind"]}</h1><pre>{body}</pre><video controls width="720" src="preview.mp4"></video><p><img width="900" src="curves.png"></p>{extra}<p><a href="result.json">Full audit and per-probe feedback</a></p><p><a href="training_contract.json">Supervision boundaries (training disabled)</a></p>', encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    from .tactile import AnyTouchEncoder
    encoder = AnyTouchEncoder(WORKSPACE/"AnyTouch2", device=args.device, threads=4)
    result = audit_result(args.result, encoder)
    print(json.dumps({k:result.get(k) for k in ("seed","kind","qualified","timing")}), flush=True)
    pilot = args.result.resolve().parent.parent.parent
    if (pilot/"pilot.json").is_file():
        hold = pilot/"diagnostic_hold.json"
        if not result["progress_proxy_audit"]["usable_for_this_failure"] and not hold.exists():
            atomic_json(hold, dict(reason="FK_progress_is_not_insertion_feedback", seed=result["seed"],
                result=str(args.result.resolve()), evidence=result["progress_proxy_audit"],
                action="Stop after this complete audit; revise observable progress and grasp feedback before more collection"))
        if hold.exists():
            print(f"Diagnostic hold: {hold}", flush=True)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
