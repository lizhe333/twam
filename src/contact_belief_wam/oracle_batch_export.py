"""Immutable, task-qualified exports for the privileged recovery campaign.

The campaign owns simulator attempts and quotas. This module only consumes
completed results: raw traces remain immutable, and generated manifests remain
disabled for training until a separate training enrollment is requested.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import html
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
from PIL import Image, ImageDraw

from .paths import WORKSPACE, atomic_json, read_jsonl, sha256, write_jsonl
from .takeover import tactile_indices
from .takeover_dataset import FORMAT, validate_contract, validate_trace

TASK_BUDGETS = {"insert_HDMI": 600, "insert_hole": 317}
_DEFAULT_ENCODER = None


def _default_encoder():
    global _DEFAULT_ENCODER
    if _DEFAULT_ENCODER is None:
        from .tactile import AnyTouchEncoder
        _DEFAULT_ENCODER = AnyTouchEncoder(WORKSPACE / "AnyTouch2", device="cpu", threads=4)
    return _DEFAULT_ENCODER


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _path(value, parent):
    path = Path(value)
    return (path if path.is_absolute() else parent / path).resolve()


def _load_json(path):
    return json.loads(Path(path).read_text())


def recovery_ranges(handle, recovery):
    """Use actual successful Teacher decisions, never infer labels from outcome alone."""
    n = len(handle["actions/clipped_actual_target"])
    takeover = int(recovery["takeover_step"])
    require(0 <= takeover < n, "Takeover must precede at least one recovery action")
    intervals, previous = [], takeover
    decisions = handle["actions/decision_step"][:]
    phases = handle["actions/phase"].asstr()[:] if "actions/phase" in handle else None
    for event in recovery["events"]:
        if "stop" not in event or int(event["stop"]) == int(event["step"]):
            continue  # Marker-only events have no action labels.
        start, stop = int(event["step"]), int(event["stop"])
        require(previous <= start < stop <= n, "Teacher events overlap or exceed recorded commands")
        previous = stop
        require(int(event.get("decision_step", start)) == start,
                "Teacher decision must reference its current observation")
        require(np.all(decisions[start:stop] == start), "Teacher event crosses a feedback decision")
        eligible = event.get("bc_eligible")
        require(isinstance(eligible, bool), "Every Teacher action event needs explicit bc_eligible")
        # Exploration and ineffective pushes remain valid WM transitions only.
        if not eligible or event.get("stalled", False) or event.get("phase") == "exploration":
            continue
        if phases is not None:
            require(np.all(phases[start:stop] == event["phase"]), "Teacher phase disagrees with trace")
        intervals.append(dict(start=start, stop=stop, decision_step=start,
                              phase=event["phase"], reason=event.get("reason", "eligible_teacher_action")))
    require(bool(intervals), "Recovery contains no eligible imitation labels")
    return intervals


def _legacy_attempt(result_path):
    """Adapt the explicitly approved old HDMI success without changing its source flags."""
    old = _load_json(result_path)
    require(old.get("kind") == "oracle" and old.get("success") is True,
            "Existing import requires a successful privileged recovery")
    trajectory = _path(old["trajectory"], result_path.parent)
    reference = _path(old["reference"], result_path.parent)
    with h5py.File(trajectory, "r") as branch, h5py.File(reference, "r") as natural:
        task = str(branch.attrs["task"])
        require(natural.attrs["terminal_status"] != "success", "Paired natural rollout did not fail")
        checkpoint = str(branch.attrs["checkpoint"])
        natural_n = len(natural["actions/clipped_actual_target"])
    events = []
    for item in old.get("oracle", old.get("teacher", {})).get("events", []):
        if "stop" not in item:
            continue
        event = dict(item, decision_step=int(item["step"]))
        eligible = not item.get("stalled", False) and item["phase"] != "exploration"
        if "insert" in item["phase"] and not item.get("success", False):
            # Legacy HDMI uses the slot-local z gap; lower means greater insertion.
            before = item.get("before", {}).get("object_in_slot_mm")
            after = item.get("after", {}).get("object_in_slot_mm")
            require(before is not None and after is not None,
                    "Legacy insertion event lacks recorded progress for BC screening")
            event["progress_mm"] = float(before[2] - after[2])
            eligible &= event["progress_mm"] >= .04
        event["bc_eligible"] = bool(eligible)
        events.append(event)
    restore = old.get("physical_takeover_restore", {})
    require(restore.get("passed") is True, "Existing recovery physical restore did not pass")
    return dict(task=task, seed=int(old["seed"]), budget=TASK_BUDGETS[task], checkpoint=checkpoint,
                natural=dict(success=False, status="failed", trajectory=str(reference), strategy_actions=natural_n),
                recovery=dict(success=True, status="success", trajectory=str(trajectory),
                              takeover_step=int(old["takeover_step"]), physical_restore=restore,
                              events=events, timing=old["timing"], strategy_actions=old["strategy_actions"]),
                imported_existing=True, tactile_audit=str(result_path.parent / "tactile_audit.npz"))


def _cache(row, output, encoder, source_audit=None):
    cache = output / "cache" / (row["episode_id"] + ".npz")
    expected = np.stack([tactile_indices(t) for t in range(row["frames"])])
    if source_audit is not None and source_audit.is_file():
        row["source_cache"] = str(source_audit)
        row["source_cache_sha256"] = sha256(source_audit)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            require(str(data["trace_sha256"]) == row["trace_sha256"], "Existing cache trace changed")
            metadata = json.loads(str(data["metadata"]))
            windows, tokens = data["window_indices"], data["tokens"]
        if encoder is not None:
            require(metadata == encoder.metadata, "Existing cache encoder changed")
    elif source_audit is not None and source_audit.is_file():
        with np.load(source_audit, allow_pickle=False) as data:
            tokens, windows = data["tokens"], data["window_indices"]
            metadata = json.loads(str(data["encoder_metadata"]))
            if "trace_sha256" in data:
                require(str(data["trace_sha256"]) == row["trace_sha256"], "Source audit belongs to another trace")
        source_metadata = _load_json(row["source_result"]).get("encoder")
        if source_metadata is not None:
            require(source_metadata == metadata, "Source audit encoder metadata differs from result")
        if encoder is not None:
            require(metadata == encoder.metadata, "Source cache encoder changed")
    else:
        if encoder is None:
            encoder = _default_encoder()
        metadata, windows, chunks = encoder.metadata, expected, []
        with h5py.File(row["path"], "r") as handle:
            for begin in range(0, row["frames"], 8):
                batch = np.stack([np.stack([np.stack([
                    handle[f"observations/{side}_rgb_marker"][int(i)] for i in expected[t]])
                    for side in ("left", "right")]) for t in range(begin, min(begin + 8, row["frames"]))])
                chunks.append(np.asarray(encoder(batch), np.float16))
        tokens = np.concatenate(chunks)
    require(metadata.get("offsets") == [-6, -4, -2, 0], "Encoder must use causal four-frame windows")
    require(np.array_equal(windows, expected), "Tactile windows contain future frames or history reset")
    require(tokens.shape == (row["frames"], 8, 512) and np.isfinite(tokens).all(), "Invalid tactile features")
    row["encoder_metadata"] = metadata
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, tokens=tokens.astype(np.float16), raw_step=np.arange(row["frames"]),
                            frame_index=np.arange(row["frames"]), window_indices=windows,
                            metadata=json.dumps(metadata), trace_sha256=row["trace_sha256"])
        temporary.replace(cache)
    row["cache_sha256"] = sha256(cache)
    return tokens


def audit_loading(row, output):
    """Exercise the actual adapter, normalization, BC boundaries and continuous history."""
    import torch
    from .config import ContactConfig
    from .data import AlignedDataset, AlignedProcessor, prepare_aligned_batch
    from .sequence import observation_at, targets_at
    torch.set_num_threads(4)
    cfg = ContactConfig(aligned=True, stage="stage2", mode="full", independent_tactile=True,
                        camera_mode="head_wrist", k=10, chunk_size=50,
                        belief_warmup_min=4, belief_warmup_max=4)
    kwargs = dict(manifest=[row], data_root=output, cache_root=output / "cache",
                  task_stats={row["task"]: row["task_stats"]}, wm_stats=row["wm_stats"],
                  config=cfg, split="all", tokenizer_path=None)
    try:
        AlignedDataset(**kwargs)
    except ValueError as exc:
        require("training is disabled" in str(exc), str(exc))
    else:
        raise ValueError("Exported manifest was admitted to training without explicit enrollment")
    ds = AlignedDataset(**kwargs, allow_review=True)
    inv = AlignedProcessor(row["task_stats"], None, "")
    inv_wm = AlignedProcessor(row["wm_stats"], None, "")
    boundary_anchors = {0, row["actions"] - 1}
    for interval in row["policy_supervision_ranges"]:
        boundary_anchors.update([max(0, interval["start"] - 1), interval["start"], interval["stop"] - 1])
    supervised, poisoned = 0, 0
    try:
        episode = ds._load(0)
        commands, joints = episode["commands"], episode["joints"]
        for index, (_, _, _, t) in enumerate(ds.index):
            sample = ds[index]
            interval = next((x for x in row["policy_supervision_ranges"] if x["start"] <= t < x["stop"]), None)
            count = min(50, interval["stop"] - t) if interval else 0
            np.testing.assert_array_equal(sample["policy_action_mask"][-1], np.arange(50) < count)
            labels = inv.absolute(sample["actions"][-1], joints[t])
            expected = np.repeat((commands[t + count - 1] if count else joints[t])[None], 50, axis=0)
            expected[:count] = commands[t:t + count]
            np.testing.assert_allclose(labels, expected, atol=3e-6, rtol=1e-6)
            begin = max(0, t - 10)
            history = inv_wm.absolute(sample["executed_actions"][-1], joints[begin])
            np.testing.assert_allclose(history[sample["executed_valid"][-1]], commands[begin:t], atol=3e-6, rtol=1e-6)
            frames = sample["step"][sample["sequence_mask"]]
            np.testing.assert_array_equal(frames, np.arange(t - min(4, t // 10) * 10, t + 1, 10))
            np.testing.assert_array_equal(sample["tactile"], episode["tactile"][sample["step"]].astype(np.float32))
            require(bool(sample["future_mask"][-1]) == (t + 10 <= row["actions"]), "WM history was truncated at a BC boundary")
            require(not any(any(s in key for s in ("oracle", "prism", "slot", "truth")) for key in sample),
                    "Privileged fields reached student input")
            if t in boundary_anchors:
                original = commands[t + count:].copy()
                try:
                    commands[t + count:] += 123
                    changed = ds[index]
                    for key in ("actions", "executed_actions", "tactile", "image", "wrist_image", "state"):
                        np.testing.assert_array_equal(sample[key], changed[key])
                finally:
                    commands[t + count:] = original
                poisoned += 1
            supervised += bool(count)
        from torch.utils.data import DataLoader
        batch = prepare_aligned_batch(next(iter(DataLoader(ds, batch_size=min(2, len(ds))))), "cpu")
        obs, target = observation_at(batch, -1), targets_at(batch, -1)
        require(obs.image.shape == obs.wrist_image.shape and obs.image.shape[1:] == (3, 224, 224),
                "Dual-view student batch is malformed")
        require(target.actions.shape[1:] == (50, 32), "Actual action target batch is malformed")
        return dict(passed=True, anchors_checked=len(ds), supervised_anchors=supervised,
                    excluded_future_poison_checks=poisoned, history_continuous=True,
                    causal_tactile_windows=True, actual_command_roundtrip=True,
                    dual_view_model_input=True, privileged_fields_absent=True,
                    training_disabled_guard=True, task_stats_sha256=sha256(row["task_stats"]),
                    wm_stats_sha256=sha256(row["wm_stats"]))
    finally:
        ds.close()


def _mosaic(handle, t):
    from .data import resize
    obs = handle["observations"]
    tiles = [resize(obs[key][t]) for key in ("head_rgb", "wrist_rgb", "left_rgb_marker", "right_rgb_marker")]
    return Image.fromarray(np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0))


def _video(trace, destination, reference=None):
    """Every recorded observation; shorter branches hold their terminal frame."""
    width = 896 if reference is not None else 448
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x480", "-r", "15", "-i", "-", "-an", "-c:v", "libx264",
               "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]
    with h5py.File(trace, "r") as actual:
        original = h5py.File(reference, "r") if reference is not None else None
        try:
            handles = [original, actual] if original is not None else [actual]
            lengths = [len(h["observations/joint_q"]) for h in handles]
            with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
                try:
                    for t in range(max(lengths)):
                        canvas = Image.new("RGB", (width, 480), "#141a24")
                        draw = ImageDraw.Draw(canvas)
                        for j, (handle, length) in enumerate(zip(handles, lengths)):
                            frame = min(t, length - 1)
                            canvas.paste(_mosaic(handle, frame), (j * 448, 32))
                            title = "WAVE-0 FAILED" if handle is original else "ORACLE RECOVERY" if original else "NORMAL SUCCESS"
                            draw.text((j * 448 + 5, 8), f"{title} | obs {frame}" + (" | END HELD" if t >= length else ""), fill="white")
                        process.stdin.write(np.asarray(canvas).tobytes())
                finally:
                    process.stdin.close()
                require(process.wait() == 0, "Review video encoding failed")
        finally:
            if original is not None:
                original.close()


REVIEW_PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>恢复训练样本逐帧查看</title>
<style>body{font:16px/1.6 system-ui;margin:24px auto;max-width:1200px;padding:0 18px;background:#111823;color:#e7edf5}a{color:#94c6ff}button,input{font:inherit}input{width:75%}.layout{display:flex;gap:20px;flex-wrap:wrap}.main{width:448px;max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-width:650px}#history{display:flex;gap:10px;flex-wrap:wrap}canvas{width:224px}video{max-width:100%}</style>
<h1 id="title"></h1><p>上排：head / wrist。下排：左 / 右触觉。标签使用实际执行动作；真值只用于 Teacher 和审核。训练尚未启用。</p>
<p><a href="../../../../index.html">批次索引</a> · <a href="frames.csv">逐帧 CSV</a> · <a href="validation.json">加载与边界检查</a> · <a href="contract.json">样本契约</a></p>
<video controls preload="metadata" src="comparison.mp4"></video>
<p><button id="prev">上一帧</button> <input id="slider" type="range" min="0" value="0"> <button id="next">下一帧</button></p>
<div class="layout"><div><img id="frame" class="main"><p id="time"></p></div><pre id="details"></pre></div>
<h2>实际可用的四帧触觉窗口</h2><div id="history"></div>
<script src="frames.js"></script><script>
const d=window.EPISODE,s=document.getElementById('slider');s.max=d.frames.length-1;
document.getElementById('title').textContent=d.episode_id;
const url=t=>'frames/'+String(t).padStart(4,'0')+'.jpg';
function show(){const t=+s.value,r=d.frames[t];document.getElementById('frame').src=url(t);document.getElementById('time').textContent='观测 '+t+' / '+s.max;document.getElementById('details').textContent=JSON.stringify(r,null,2);const h=document.getElementById('history');h.replaceChildren();for(const j of r.tactile_window){const box=document.createElement('div'),caption=document.createElement('div'),c=document.createElement('canvas'),im=new Image();c.width=448;c.height=224;caption.textContent='t = '+j;box.append(caption,c);h.append(box);im.onload=()=>c.getContext('2d').drawImage(im,0,224,448,224,0,0,448,224);im.src=url(j);}}
s.oninput=show;document.getElementById('prev').onclick=()=>{s.value=Math.max(0,+s.value-1);show()};document.getElementById('next').onclick=()=>{s.value=Math.min(+s.max,+s.value+1);show()};document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();s.value=Math.max(0,Math.min(+s.max,+s.value+(e.key==='ArrowLeft'?-1:1)));show()}});show();</script></html>'''


def build_review(row, output, tokens):
    directory = output / "review" / row["episode_id"]
    frames = directory / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    records = []
    features = tokens.astype(np.float32)
    previous = np.concatenate([features[:1], features[:-1]])
    cosine = np.mean(1 - np.clip((features * previous).sum(-1) /
                     np.maximum(np.linalg.norm(features, axis=-1) * np.linalg.norm(previous, axis=-1), 1e-12), -1, 1), axis=-1)
    with h5py.File(row["path"], "r") as handle:
        obs, commands = handle["observations"], handle["actions/clipped_actual_target"]
        phases = handle["actions/phase"].asstr()[:] if "actions/phase" in handle else ["wave0"] * row["actions"]
        decisions = handle["actions/decision_step"][:]
        previous_rgb = None
        for t in range(row["frames"]):
            _mosaic(handle, t).save(frames / f"{t:04d}.jpg", quality=87)
            pixels = [obs[f"{side}_rgb_marker"][t].astype(np.float32) for side in ("left", "right")]
            raw_delta = [0., 0.] if previous_rgb is None else [float(np.abs(a - b).mean() / 255.) for a, b in zip(pixels, previous_rgb)]
            previous_rgb = pixels
            interval = next((r for r in row["policy_supervision_ranges"] if r["start"] <= t < r["stop"]), None)
            count = min(50, interval["stop"] - t) if interval else 0
            records.append(dict(t=t, phase=str(phases[t]) if t < row["actions"] else "terminal_success",
                                decision_step=int(decisions[t]) if t < row["actions"] else None,
                                robot_state=obs["joint_q"][t].tolist(),
                                actual_command=commands[t].tolist() if t < row["actions"] else None,
                                executed_history=commands[max(0, t - 10):t].tolist(),
                                policy_valid_prefix=count, policy_label_stop=t + count,
                                policy_reason=interval.get("reason") if interval else "context_and_WM_only",
                                tactile_window=tactile_indices(t).tolist(),
                                raw_touch_delta=raw_delta, latent_delta_from_previous=float(cosine[t]),
                                physics_step=int(obs["physics_step"][t])))
    (directory / "frames.js").write_text("window.EPISODE=" + json.dumps(
        dict(episode_id=row["episode_id"], timing=row.get("timing"), frames=records), ensure_ascii=False, allow_nan=False) + ";\n")
    with (directory / "frames.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in record.items()})
    _video(row["path"], directory / "comparison.mp4", row.get("natural_reference"))
    (directory / "index.html").write_text(REVIEW_PAGE)
    row["review_page"] = str(directory / "index.html")
    row["video"] = str(directory / "comparison.mp4")


def rebuild_index(output_root, ledger_state=None):
    output = Path(output_root).resolve()
    manifest = output / "review_manifest.jsonl"
    rows = read_jsonl(manifest) if manifest.exists() else []
    effective = output / "effective_config.json"
    settings = _load_json(effective).get("tasks", {}) if effective.exists() else {}
    counts, links = {}, []
    for task in TASK_BUDGETS:
        subset = [r for r in rows if r["task"] == task]
        counts[task] = dict(recovery=sum(r["source"] == "recovery" for r in subset),
                            normal=sum(r["source"] == "original" for r in subset))
        folder = output / "manifests" / task
        folder.mkdir(parents=True, exist_ok=True)
        write_jsonl(folder / "recovery.jsonl", [r for r in subset if r["source"] == "recovery"])
        write_jsonl(folder / "normal.jsonl", [r for r in subset if r["source"] == "original"])
        target = int(settings.get(task, {}).get("recovery_target", 25))
        links.append(f"<h2>{html.escape(task)}：恢复 {counts[task]['recovery']} / {target}，正常 {counts[task]['normal']}</h2><ul>")
        for row in subset:
            link = Path(row["review_page"]).relative_to(output).as_posix()
            labels = sum(r["stop"] - r["start"] for r in row["policy_supervision_ranges"])
            links.append(f'<li><a href="{html.escape(link)}">{html.escape(row["episode_id"])}</a> — {row["actions"]} 动作，{labels} 个 BC 标签</li>')
        links.append("</ul>")
    if ledger_state is not None:
        links.append('<h2>全部尝试</h2><p><a href="export_summary.json">完整账本与统计</a></p><table>'
                     '<tr><th>任务 / seed</th><th>自然运行</th><th>恢复</th><th>结束原因</th></tr>')
        for attempt in ledger_state.get("attempts", []):
            result = attempt.get("result", {})
            natural = result.get("natural") or attempt.get("natural") or {}
            recovery = result.get("recovery") or {}
            reason = recovery.get("reason") or result.get("recovery_status") or natural.get("reason") or result.get("error", "")
            cells = (f'{attempt["task"]} / {attempt["seed"]}', natural.get("status", attempt.get("status", "")),
                     recovery.get("status", "未接管"), reason)
            links.append('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in cells) + '</tr>')
        links.append('</table>')
    summary = dict(counts=counts, exported=len(rows), training_enabled=False, training_started=False,
                   ledger=ledger_state, manifest=str(manifest))
    atomic_json(output / "export_summary.json", summary)
    (output / "index.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Recovery 采集批次</title>'
        '<style>body{font:16px/1.7 system-ui;max-width:1000px;margin:36px auto;padding:0 20px}a{color:#1763a9}td,th{text-align:left;padding:6px;border-bottom:1px solid #ddd}</style>'
        '<h1>HDMI / insert_hole 恢复样本</h1><p>真值辅助 Teacher；自然失败前缀保留为历史和 WM 转移，'
        '有效恢复动作按反馈决策分段。正常成功保留完整监督。未启动训练。</p>' + ''.join(links) + '</html>')
    return summary


def _export(result_path, output, attempt, encoder):
    task, seed = attempt["task"], int(attempt["seed"])
    require(task in TASK_BUDGETS, "Unsupported recovery campaign task")
    require(int(attempt["budget"]) == TASK_BUDGETS[task], "Task action budget changed")
    natural, recovery = attempt["natural"], attempt.get("recovery") or {}
    kind = "normal" if natural.get("success") else "recovery"
    selected = natural if kind == "normal" else recovery
    if selected.get("success") is not True:
        return []
    episode_id = f"{task}/{kind}/{seed}"
    manifest = output / "review_manifest.jsonl"
    rows = read_jsonl(manifest) if manifest.exists() else []
    existing = next((r for r in rows if r["episode_id"] == episode_id), None)
    if existing:
        require(existing["source_result_sha256"] == sha256(result_path) and
                existing["trace_sha256"] == sha256(existing["path"]), "Previously exported source changed")
        return [existing]
    effective = output / "effective_config.json"
    normal_target = int(_load_json(effective)["tasks"][task].get("normal_target", 25)) if effective.exists() else 25
    if kind == "normal" and sum(r["task"] == task and r["source"] == "original" for r in rows) >= normal_target:
        return []
    trace = _path(selected["trajectory"], result_path.parent)
    checkpoint = _path(attempt["checkpoint"], result_path.parent)
    deployment = _load_json(checkpoint / "deployment.json")
    require(deployment["camera"] == "head_wrist", "Baseline checkpoint is not dual view")
    row = dict(data_format=FORMAT, task=task, episode_id=episode_id, seed=seed, split="review", kind=kind,
               path=str(trace), source="original" if kind == "normal" else "recovery",
               demonstrator="wave0" if kind == "normal" else "privileged_oracle",
               prompt=deployment["prompt"], checkpoint=str(checkpoint),
               actual_command_key="actions/clipped_actual_target", training_enabled=False,
               training_approved=False, suitable_for_imitation=True,
               supervision_status="technical_checks_passed", source_result=str(result_path),
               source_result_sha256=sha256(result_path), trace_sha256=sha256(trace),
               task_stats=str(checkpoint / "norm_stats.json"), wm_stats=str(checkpoint / "wm_norm_stats.json"),
               original_policy_flags_unchanged=True, imported_existing=bool(attempt.get("imported_existing")),
               collection_provenance=attempt.get("collection_provenance"),
               context_semantics="continuous observed history and executed actions; no reset at BC boundaries",
               limitations=["Privileged Teacher imitation does not establish that the student uses tactile feedback.",
                            "Cross-branch pixel equality is not required for this imitation dataset."])
    with h5py.File(trace, "r") as handle:
        n = len(handle["actions/clipped_actual_target"])
        require(int(selected["strategy_actions"]) == n <= int(attempt["budget"]), "Recorded action count/budget mismatch")
        require(handle.attrs["terminal_status"] == "success", "Native task success is absent")
        if "task" in handle.attrs:
            require(handle.attrs["task"] == task, "Trace task differs from result")
        row.update(frames=n + 1, actions=n, context_range=[0, n + 1], segments=[[0, n + 1]])
        if kind == "recovery":
            require(natural.get("success") is False, "Recovery must have a failed natural reference")
            require(recovery.get("physical_restore", {}).get("passed") is True, "Physical restoration check failed")
            timing = recovery["timing"]
            times = [timing[k] for k in ("t_contact", "t_tactile", "t_decision", "t_recovery")]
            require(timing.get("passed") is True and all(isinstance(t, int) for t in times) and
                    0 <= times[0] <= times[1] <= times[2] <= times[3] == int(recovery["takeover_step"]),
                    "Contact evidence does not precede recovery")
            row.update(takeover_step=int(recovery["takeover_step"]), timing=timing,
                       natural_reference=str(_path(natural["trajectory"], result_path.parent)),
                       policy_supervision_ranges=recovery_ranges(handle, recovery))
            with h5py.File(row["natural_reference"], "r") as reference:
                require(reference.attrs["terminal_status"] != "success", "Native baseline actually succeeded")
                end = row["takeover_step"]
                for key in ("joint_q", "head_rgb", "wrist_rgb", "left_rgb_marker", "right_rgb_marker"):
                    require(np.array_equal(handle[f"observations/{key}"][:end + 1], reference[f"observations/{key}"][:end + 1]),
                            "Recorded natural prefix was replaced or its history reset")
                require(np.array_equal(handle["actions/clipped_actual_target"][:end],
                                       reference["actions/clipped_actual_target"][:end]), "Recorded prefix commands differ")
            row["natural_reference_sha256"] = sha256(row["natural_reference"])
        else:
            row.update(takeover_step=None, timing=None,
                       policy_supervision_ranges=[dict(start=0, stop=n, reason="normal_success")])
        validate_contract(row, allow_review=True)
        technical = validate_trace(handle, row)
    source_audit = attempt.get("tactile_audit") or selected.get("tactile_audit")
    tokens = _cache(row, output, encoder, _path(source_audit, result_path.parent) if source_audit else None)
    loading = audit_loading(row, output)
    build_review(row, output, tokens)
    require(sha256(trace) == row["trace_sha256"] and sha256(result_path) == row["source_result_sha256"],
            "Raw source changed during export")
    directory = Path(row["review_page"]).parent
    atomic_json(directory / "validation.json", dict(passed=True, trace=technical, loader=loading,
                source_unchanged=True, training_enabled=False, cross_branch_pixel_equality_required=False))
    atomic_json(directory / "contract.json", row)
    rows.append(row)
    rows.sort(key=lambda r: r["episode_id"])
    temporary = manifest.with_suffix(".tmp.jsonl")
    write_jsonl(temporary, rows)
    temporary.replace(manifest)
    rebuild_index(output)
    return [row]


def export_attempt(result_path: Path, output_root: Path, *, encoder=None):
    """Export one completed new-format result; returns zero or one admitted rows."""
    result_path, output = Path(result_path).resolve(), Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".export.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _export(result_path, output, _load_json(result_path), encoder)


def import_existing(result_path: Path, output_root: Path, *, encoder=None):
    """Import the approved existing success, reusing its causal AnyTouch audit."""
    result_path, output = Path(result_path).resolve(), Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".export.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _export(result_path, output, _legacy_attempt(result_path), encoder)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--import-existing", action="store_true")
    args = parser.parse_args(argv)
    rows = (import_existing if args.import_existing else export_attempt)(args.result, args.output)
    print(json.dumps({"exported": [r["episode_id"] for r in rows], "training_started": False}), flush=True)


if __name__ == "__main__":
    main()
