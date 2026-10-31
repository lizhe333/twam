"""Three-class review exports, immutable imports and measured correction checks."""
from __future__ import annotations

from collections import Counter
import html
import json
from pathlib import Path
import shutil

import h5py
import numpy as np

from .paths import atomic_json, sha256, write_jsonl
from .mixed_recovery import KINDS, MixedLedger, targets


def self_correction_check(result, task):
    from scipy.spatial.transform import Rotation
    p = result["perturbation"]
    axis = np.asarray(p["offset_target_frame_mm"], float)
    requested = float(np.linalg.norm(axis))
    if requested <= 0:
        return dict(passed=False, reason="zero_requested_displacement")
    axis /= requested
    n, stop = result["strategy_actions"], p["stop"]
    with h5py.File(result["trajectory"], "r") as f:
        obs = f["observations"]
        obj = np.stack([obs["prism_pose_world"][stop], obs["prism_pose_world"][n]])
        name = "target_pose_world" if task == "insert_hole" else "slot_pose_world"
        target = np.stack([obs[name][stop], obs[name][n]])
    relative = Rotation.from_quat(target[:, [4,5,6,3]]).inv().apply(obj[:, :3]-target[:, :3])*1000
    error = relative[:, :2] @ axis
    actual = float(Rotation.from_quat(target[0, [4,5,6,3]]).inv().apply(
        np.asarray(p["actual_object_delta_world_mm"]))[:2] @ axis)
    reduction = float(abs(error[0])-abs(error[1]))
    passed = bool(result["success"] and p["stop"]-p["start"] == 10 and stop < n
                  and actual >= .25*requested and reduction >= .1)
    return dict(passed=passed, requested_mm=requested, actual_along_direction_mm=actual,
                error_after_perturb_mm=float(error[0]), final_error_mm=float(error[1]),
                absolute_error_reduction_mm=reduction,
                reason="measured_self_correction" if passed else "insufficient_physical_offset_or_correction")


def normalize_import(row, manifest, output):
    """Build a new manifest proposal; never edit old HDF5, metadata, or labels."""
    from .hole_preview import label_ranges
    kind = "oracle_recovery" if row["kind"] == "recovery" else row["kind"]
    source = Path(row["source_result"])
    if sha256(source) != row["source_result_sha256"] or sha256(row["path"]) != row["trace_sha256"]:
        raise ValueError("Imported result or raw trace changed")
    original = json.loads(source.read_text())
    if original.get("kind") == "oracle":
        from .oracle_batch_export import _legacy_attempt
        original = _legacy_attempt(source)
    selected = original.get("recovery" if kind == "oracle_recovery" else "natural", original)
    events = []
    if kind == "oracle_recovery":
        events = [dict(step=r["start"], stop=r["stop"], bc_eligible=True) for r in row["policy_supervision_ranges"]]
    perturb = row.get("perturbation") or selected.get("perturbation")
    with h5py.File(row["path"], "r") as f:
        sources = f["actions/source"].asstr()[:]
        phases = f["actions/phase"].asstr()[:]
        ranges, _ = label_ranges(sources, phases, f["actions/decision_step"][:], success=True, kind=kind,
            perturb_start=perturb["start"] if perturb else None, perturb_stop=perturb["stop"] if perturb else None,
            takeover=row.get("takeover_step"), teacher_events=events)
    result = dict(selected, kind=kind, task=row["task"], seed=row["seed"], success=True,
        checkpoint=row["checkpoint"], trajectory=row["path"], strategy_actions=row["actions"],
        policy_supervision_ranges=ranges, perturbation=perturb, events=events,
        takeover_step=row.get("takeover_step"), timing=row.get("timing"),
        natural_reference=row.get("natural_reference") or row.get("paired_perturbed_failure") or selected.get("perturbed_reference"),
        demonstrator="privileged_oracle" if kind == "oracle_recovery" else "wave0",
        collection_origin=row.get("collection_origin") or ("controlled_robot_perturbation" if perturb else "natural_failure" if kind == "oracle_recovery" else "unperturbed"),
        provenance=row.get("collection_provenance") or "existing_recorded_trace_import",
        imported_existing=True, original_result=str(source), original_result_sha256=sha256(source),
        original_manifest=str(manifest), original_episode_id=row["episode_id"],
        training_enabled=False, same_instance_prefix=True)
    dest = output/"imports"/row["task"]/kind/str(row["seed"])/"result.json"
    if dest.exists() and json.loads(dest.read_text()) != result:
        raise ValueError("Immutable normalized import changed")
    atomic_json(dest, result)
    # Reuse a source cache only when its own hash matches its old contract.
    new_cache = output/"cache"/row["task"]/kind/(str(row["seed"])+".npz")
    for parent in [Path(manifest).parent, *Path(manifest).parents]:
        candidate = parent/"cache"/(row["episode_id"]+".npz")
        if candidate.is_file():
            if sha256(candidate) != row["cache_sha256"]:
                raise ValueError("Imported tactile cache changed")
            if not new_cache.exists():
                new_cache.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(candidate, new_cache)
            break
    return dest


def export_result(ledger, path, encoder=None):
    from .hole_preview_export import export_sample, render_label_timeline
    path = Path(path)
    result = json.loads(path.read_text())
    task, seed, kind = result["task"], result["seed"], result["kind"]
    state = ledger.snapshot()
    current = [r for r in state["exports"] if r["task"] == task and r["seed"] == seed and r["kind"] == kind]
    if current:
        return dict(status="duplicate_seed_kind", episode_id=current[0]["episode_id"])
    if not result["success"] or result.get("eligible_for_export") is False:
        return dict(status="not_selected", reason=result.get("reason") or "native_failure")
    if not ledger.needs(task)[kind]:
        return dict(status="quota_full")
    if kind == "self_corrected":
        check = self_correction_check(result, task)
        if not check["passed"]:
            return dict(status="not_selected", reason=check["reason"], correction=check)
    checkpoint = Path(result["checkpoint"])
    row = export_sample(path, ledger.root, seed=seed, checkpoint=checkpoint, encoder=encoder)
    row.update(group_id=f"{task}/{seed}", source_branch=str(path),
               teacher_revision=result.get("teacher_revision"), original_result=result.get("original_result"),
               target_axis_world=result.get("target_axis_world"))
    folder = Path(row["review_page"]).parent
    render_label_timeline([row], folder/"training_label_timeline.png")
    page = (folder/"index.html").read_text()
    page = page.replace('<div id="jumps">', '<p><a href="training_label_timeline.png">训练标签时间轴</a></p><div id="jumps">')
    (folder/"index.html").write_text(page)
    atomic_json(folder/"contract.json", row)
    ledger.record_export(row)
    return dict(status="accepted", episode_id=row["episode_id"], row=row)


def import_existing(ledger, encoder=None):
    for manifest in ledger.snapshot()["config"].get("import_manifests", []):
        for line in Path(manifest).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("suitable_for_imitation") or row.get("training_enabled") is not False:
                continue
            kind = "oracle_recovery" if row["kind"] == "recovery" else row["kind"]
            if kind not in KINDS:
                continue
            s = ledger.snapshot()
            if any(r["task"] == row["task"] and r["seed"] == row["seed"] and r["kind"] == kind for r in s["exports"]):
                continue
            if not ledger.needs(row["task"])[kind]:
                continue
            signature = f"{manifest}#{row['episode_id']}"
            if any(r["source"] == signature for r in s["imports"]):
                continue
            try:
                normalized = normalize_import(row, manifest, ledger.root)
                outcome = export_result(ledger, normalized, encoder)
                ledger.record_import(signature, outcome["status"], episode_id=outcome.get("episode_id"))
                print(json.dumps(dict(event="import", source=signature, status=outcome["status"])), flush=True)
            except Exception as exc:
                ledger.record_import(signature, "rejected", reason=f"{type(exc).__name__}: {exc}")
                print(json.dumps(dict(event="import_rejected", source=signature, reason=str(exc))), flush=True)
            rebuild_index(ledger)


def rebuild_index(ledger):
    state, output = ledger.snapshot(), ledger.root
    rows = state["exports"]
    write_jsonl(output/"review_manifest.jsonl", rows)
    sections, counts, coverage = [], {}, {}
    titles = dict(normal="正常成功", self_corrected="扰动后自行纠正", oracle_recovery="Oracle 恢复")
    for task in state["config"]["tasks"]:
        c = MixedLedger.counts(state, task)
        counts[task] = c
        selected = [r for r in rows if r["task"] == task]
        attempts = [a for a in state["attempts"] if a["task"] == task]
        trials = [b["trial"] for a in attempts for b in a["branches"] if b["type"] == "perturb"]
        coverage[task] = dict(attempted={k: dict(Counter(str(t[k]) for t in trials)) for k in ("stage", "direction", "amplitude_mm")},
            accepted_origins=dict(Counter(r["collection_origin"] for r in selected)),
            accepted_perturbations=[dict(seed=r["seed"], kind=r["kind"], perturbation=r.get("perturbation")) for r in selected if r.get("perturbation")],
            native_target_axes=[dict(seed=a["seed"], axis=a.get("result", {}).get("target_axis_world")) for a in attempts])
        sections.append(f'<h2>{html.escape(task)} · {html.escape(state["task_status"][task])}</h2>')
        sections.append('<p>'+ '；'.join(f'{titles[k]} {c["success"][k]}/{targets(state["config"],task)[k]}' for k in KINDS)+'</p>')
        sections.append(f'<p>新 seed {c["natural"]}/100，扰动分支 {c["perturb"]}/200，Oracle 分支 {c["recovery"]}/100。</p>')
        folder = output/"manifests"/task
        folder.mkdir(parents=True, exist_ok=True)
        for kind in KINDS:
            subset = [r for r in selected if r["kind"] == kind]
            filename = "recovery" if kind == "oracle_recovery" else kind
            write_jsonl(folder/(filename+".jsonl"), subset)
            sections.append(f'<h3>{titles[kind]}</h3><ul>')
            for row in subset:
                url = Path(row["review_page"]).relative_to(output).as_posix()
                labels = sum(x["stop"]-x["start"] for x in row["policy_supervision_ranges"])
                origin = "人工扰动" if row["collection_origin"] == "controlled_robot_perturbation" else "自然失败" if kind == "oracle_recovery" else "无扰动"
                sections.append(f'<li><a href="{html.escape(url)}">{html.escape(row["episode_id"])}</a> · {origin} · {row["actions"]} 动作，{labels} 个有效 BC 标签</li>')
            sections.append('</ul>')
    summary = dict(status=state["status"], task_status=state["task_status"], counts=counts, coverage=coverage,
                   exported=len(rows), training_enabled=False, training_started=False,
                   imports=state["imports"], attempts=state["attempts"], updated_at=state["updated_at"])
    atomic_json(output/"summary.json", summary)
    sections.append('<h2>全部尝试</h2><p><a href="summary.json">数量、覆盖、失败与淘汰记录</a> · <a href="campaign.json">持久化账本</a></p>')
    for attempt in state["attempts"]:
        reason = attempt.get("result", {}).get("error") or attempt.get("result", {}).get("reason", "")
        sections.append(f'<p>{attempt["task"]}/{attempt["seed"]}：{html.escape(attempt["status"])} {html.escape(reason)}</p>')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>双任务三类恢复数据</title><style>body{font:17px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:0 20px;background:#101925;color:#e7edf5}a{color:#9fcaff}</style><h1>HDMI / insert_hole：50 条恢复＋40 条对照</h1><p>扰动由机器人真实执行，扰动动作不作为模仿标签。Oracle 使用仿真真值，学生输入不含真值。本批次未启用训练。</p>'+''.join(sections)+'</html>'
    temporary = output/"index.tmp.html"
    temporary.write_text(page)
    temporary.replace(output/"index.html")
    return summary
