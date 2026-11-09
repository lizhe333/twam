"""Export three review-only samples without modifying any training manifest."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .paths import atomic_json, sha256, write_jsonl
from .oracle_batch_export import _cache, _mosaic, audit_loading, build_review, require
from .hole_preview import validate_preview_labels
from .takeover_dataset import FORMAT, validate_contract, validate_trace

TITLES = {"normal": "正常成功", "self_corrected": "扰动后 WAVE-0 自行纠正", "oracle_recovery": "真实受阻后 Oracle 恢复"}
ENGLISH = {"normal": "NORMAL / WAVE-0", "self_corrected": "PERTURB / WAVE-0 CORRECTION", "oracle_recovery": "BLOCKED / ORACLE RECOVERY"}


def export_sample(result_path, output, *, seed, checkpoint, encoder=None, episode_suffix=""):
    result_path, output = Path(result_path).resolve(), Path(output).resolve()
    result = json.loads(result_path.read_text())
    require(result.get("success") is True, "Only native successful branches can be preview labels")
    trace = Path(result["trajectory"])
    kind = result["kind"]
    n = int(result["strategy_actions"])
    deployment = json.loads((checkpoint / "deployment.json").read_text())
    require(deployment["camera"] == "head_wrist", "Dual-view checkpoint required")
    task = result.get("task", "insert_hole")
    require(task in {"insert_HDMI", "insert_hole"}, "Unsupported task")
    require(n <= {"insert_HDMI": 600, "insert_hole": 317}[task], "Complete branch exceeded native budget")
    row = dict(data_format=FORMAT, task=task, episode_id=f"{task}/{kind}/{seed}{episode_suffix}",
        seed=seed, split="review", kind=kind, path=str(trace),
        source="recovery" if kind == "oracle_recovery" else "original",
        demonstrator=result["demonstrator"], prompt=deployment["prompt"], checkpoint=str(checkpoint),
        actual_command_key="actions/clipped_actual_target", training_enabled=False, training_approved=False,
        suitable_for_imitation=True, supervision_status="technical_checks_passed",
        source_result=str(result_path), source_result_sha256=sha256(result_path), trace_sha256=sha256(trace),
        task_stats=str(checkpoint / "norm_stats.json"), wm_stats=str(checkpoint / "wm_norm_stats.json"),
        frames=n+1, actions=n, context_range=[0,n+1], segments=[[0,n+1]],
        policy_supervision_ranges=result["policy_supervision_ranges"],
        takeover_step=result.get("takeover_step"), timing=result.get("timing"),
        perturbation=result.get("perturbation"),
        collection_origin=result.get("collection_origin", "controlled_robot_perturbation" if result.get("perturbation") else "unperturbed" if kind == "normal" else "natural_failure"),
        group_id=f"{task}/{seed}", imported_existing=result.get("imported_existing", False),
        independent_evaluation=False, dynamics_valid="all_recorded_robot_commands",
        collection_provenance=result.get("provenance", "live_wave0_and_recorded_robot_perturbation"),
        teacher_revision=result.get("teacher_revision"),
        limitations=["Labels exclude deliberate offsets; absence of learned shortcuts requires later training/evaluation.",
                     "Oracle uses privileged poses. This preview does not demonstrate tactile-dependent student behavior."])
    with h5py.File(trace, "r") as handle:
        require(handle.attrs["terminal_status"] == "success", "Native success is absent")
        sources = handle["actions/source"].asstr()[:]
        perturb = result.get("perturbation")
        labels = validate_preview_labels(sources, row["policy_supervision_ranges"],
            perturb["start"] if perturb else None, perturb["stop"] if perturb else None)
        validate_contract(row, allow_review=True)
        technical = validate_trace(handle, row)
        if kind == "self_corrected":
            from scipy.spatial.transform import Rotation
            axis = np.asarray(perturb["offset_target_frame_mm"], float)
            requested = float(np.linalg.norm(axis))
            axis /= requested
            stop = perturb["stop"]
            obj = handle["observations/prism_pose_world"][[stop, n]]
            target = handle["observations/target_pose_world" if task == "insert_hole" else "observations/slot_pose_world"][[stop, n]]
            relative = Rotation.from_quat(target[:, [4,5,6,3]]).inv().apply(obj[:, :3]-target[:, :3])*1000
            error = relative[:, :2] @ axis
            actual = np.asarray(perturb["actual_object_delta_world_mm"])
            actual = float(np.dot(Rotation.from_quat(target[0, [4,5,6,3]]).inv().apply(actual)[:2], axis))
            correction = dict(requested_offset_mm=requested, actual_offset_along_direction_mm=actual,
                error_after_perturb_mm=float(error[0]), final_error_mm=float(error[1]),
                absolute_error_reduction_mm=float(abs(error[0])-abs(error[1])))
            require(actual >= .25*requested, "Perturbation did not produce a meaningful physical offset")
            require(correction["absolute_error_reduction_mm"] >= .1,
                    "Successful insertion alone does not demonstrate self-correction")
            row["self_correction_check"] = correction
        if kind == "oracle_recovery":
            require(result["physical_restore"]["passed"], "Physical restore failed")
            times = [result["timing"][k] for k in ("t_contact","t_tactile","t_decision","t_recovery")]
            require(all(isinstance(t,int) for t in times) and times == sorted(times), "Missing pre-action contact/tactile evidence")
            require(times[-1] == row["takeover_step"], "Recovery clock mismatch")
            reference_path = result.get("perturbed_reference") or result.get("natural_reference")
            require(bool(reference_path), "Recovery requires its failed reference")
            with h5py.File(reference_path, "r") as reference:
                require(reference.attrs["terminal_status"] != "success", "Reference did not fail")
                if result.get("same_instance_prefix", True):
                    end = row["takeover_step"]
                    for key in ("joint_q","head_rgb","wrist_rgb","left_rgb_marker","right_rgb_marker"):
                        require(np.array_equal(handle[f"observations/{key}"][:end+1], reference[f"observations/{key}"][:end+1]),
                                "Recorded history changed at Oracle takeover")
                    require(np.array_equal(handle["actions/clipped_actual_target"][:end],reference["actions/clipped_actual_target"][:end]),
                            "Executed history changed at Oracle takeover")
            row["natural_reference"] = str(reference_path)
            row["paired_perturbed_failure"] = str(reference_path) if perturb else None
    tokens = _cache(row, output, encoder)
    loading = audit_loading(row, output)
    build_review(row, output, tokens)
    directory = Path(row["review_page"]).parent
    # Add explicit label masks to the same frame data used for the review video.
    raw = (directory / "frames.js").read_text()
    data = json.loads(raw.removeprefix("window.EPISODE=").rstrip(";\n"))
    data.update(kind=kind, title=TITLES[kind], ranges=row["policy_supervision_ranges"], perturbation=perturb)
    for frame in data["frames"]:
        t, count = frame["t"], frame["policy_valid_prefix"]
        frame["control_source"] = str(sources[t]) if t < n else "terminal"
        frame["policy_mask_50"] = [i < count for i in range(50)]
        frame["label_action_indices"] = list(range(t,t+count))
        frame["perturb_bc_eligible"] = False
        frame["wm_transition_valid"] = t < n
    (directory / "frames.js").write_text("window.EPISODE="+json.dumps(data,ensure_ascii=False)+";\n")
    with (directory / "frames.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(data["frames"][0]));writer.writeheader()
        for frame in data["frames"]:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in frame.items()})
    (directory / "index.html").write_text(PAGE)
    if kind == "oracle_recovery":
        pair = directory / "failed_vs_recovery.mp4"
        if (directory / "comparison.mp4").exists():
            (directory / "comparison.mp4").replace(pair)
            row["paired_video"] = str(pair)
            page = (directory / "index.html").read_text()
            (directory / "index.html").write_text(page.replace('<div id="jumps">',
                '<p><a href="failed_vs_recovery.mp4">同分支失败／恢复配对视频</a></p><div id="jumps">'))
    render_video([row],directory / "comparison.mp4")
    require(sha256(trace)==row["trace_sha256"], "Raw trace changed during export")
    atomic_json(directory / "validation.json",dict(passed=True,trace=technical,loader=loading,labels=labels,
        history_continuous=True,training_enabled=False,source_unchanged=True))
    atomic_json(directory / "contract.json",row)
    return row


def annotated_frame(handle,row,t):
    t=min(t,row["frames"]-1)
    canvas=Image.new("RGB",(448,592),"#111a28")
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",14)
    draw.text((10,8),ENGLISH[row["kind"]],font=font,fill="white")
    draw.text((10,31),f"Observation {t} / {row['actions']}    (15 fps review)",font=font,fill="#b2c1d6")
    canvas.paste(_mosaic(handle,t),(0,64))
    for xy,text in (((7,68),"HEAD"),((231,68),"WRIST"),((7,292),"LEFT TOUCH"),((231,292),"RIGHT TOUCH")):
        draw.rectangle((xy[0]-3,xy[1]-2,xy[0]+105,xy[1]+19),fill="#111a28")
        draw.text(xy,text,font=font,fill="white")
    interval=next((r for r in row["policy_supervision_ranges"] if r["start"]<=t<r["stop"]),None)
    count=min(50,interval["stop"]-t) if interval else 0
    phase=handle["actions/phase"].asstr()[t] if t<row["actions"] else "native_success"
    color="#75e4ab" if count else "#ffc06d" if phase=="perturb" else "#aab8cc"
    draw.text((10,521),f"{phase}   |   BC {'ON' if count else 'OFF'}",font=font,fill=color)
    text=f"Labels a[{t}:{t+count}]   / 50 targets" if count else "Context / WM only; no policy labels"
    draw.text((10,545),text,font=font,fill=color)
    draw.text((10,569),"Perturb commands are excluded before flow input",font=font,fill="#b2c1d6")
    return canvas


def render_video(rows,destination):
    width=448*len(rows)
    command=["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{width}x592",
             "-r","15","-i","-","-an","-c:v","libx264","-threads","2","-pix_fmt","yuv420p","-movflags","+faststart",str(destination)]
    handles=[h5py.File(row["path"],"r") for row in rows]
    try:
        with subprocess.Popen(command,stdin=subprocess.PIPE) as proc:
            try:
                for t in range(max(r["frames"] for r in rows)):
                    canvas=Image.new("RGB",(width,592))
                    for i,(handle,row) in enumerate(zip(handles,rows)):
                        canvas.paste(annotated_frame(handle,row,t),(i*448,0))
                    proc.stdin.write(np.asarray(canvas).tobytes())
            finally:
                proc.stdin.close()
            require(proc.wait()==0,"Annotated review video failed")
    finally:
        for handle in handles:handle.close()


def render_label_timeline(rows,destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig,axes=plt.subplots(len(rows),1,figsize=(12,1.6*len(rows)+1),sharex=True,squeeze=False)
    for ax,row in zip(axes[:,0],rows):
        ax.broken_barh([(0,row["actions"])],(.15,.7),facecolors="#b2bdcc")
        ax.broken_barh([(r["start"],r["stop"]-r["start"]) for r in row["policy_supervision_ranges"]],
                       (.15,.7),facecolors="#33b97c",edgecolors="white",linewidth=.3)
        perturb=row.get("perturbation")
        if perturb:
            ax.broken_barh([(perturb["start"],perturb["stop"]-perturb["start"])],(.15,.7),facecolors="#f4b54e")
        if row.get("takeover_step") is not None:
            ax.axvline(row["takeover_step"],color="#405b91",linestyle="--")
            ax.text(row["takeover_step"]+2,1.02,f"Oracle @ {row['takeover_step']}",fontsize=9)
        ax.set_title(ENGLISH[row["kind"]],loc="left",fontsize=11)
        ax.set_yticks([]);ax.set_ylim(0,1.35);ax.set_xlim(0,317)
        ax.spines[["top","right","left"]].set_visible(False)
    axes[-1,0].set_xlabel("Actual strategy actions (including perturbation and takeover prefix)")
    fig.legend(handles=[Patch(color="#33b97c",label="BC eligible"),Patch(color="#f4b54e",label="Perturb: BC excluded"),
                        Patch(color="#b2bdcc",label="Context / WM only")],loc="upper center",ncol=3,frameon=False)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(destination,dpi=160);plt.close(fig)


PAGE='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Insert Hole 训练标签预览</title>
<style>body{margin:24px auto;max-width:1150px;padding:0 20px;background:#101925;color:#e7edf5;font:16px/1.6 system-ui}a{color:#9fcaff}video{max-width:100%;height:550px}button{padding:7px;margin:3px;cursor:pointer}#slider{width:70%}.layout{display:flex;gap:20px;flex-wrap:wrap}#frame{width:448px;max-width:100%}pre{white-space:pre-wrap;max-width:600px;overflow-wrap:anywhere}#history{display:flex;gap:8px;flex-wrap:wrap}canvas{width:224px}#badge{padding:12px;font-weight:700;border-left:6px solid}</style>
<h1 id="title"></h1><p><a href="../../../..//index.html">三类样本索引</a> · <a href="frames.csv">逐帧标签 CSV</a> · <a href="contract.json">训练样本契约</a> · <a href="validation.json">实际加载检查</a></p>
<p>上排 head / wrist，下排左右触觉。扰动命令只保留为历史和 WM 转移；BC 目标在构造 flow 输入前截断。此处仅供审核，训练未启用。</p>
<video controls preload="metadata" src="comparison.mp4"></video><div id="jumps"></div><p><button id="prev">上一帧</button><input id="slider" type="range" min="0" value="0"><button id="next">下一帧</button></p><div id="badge"></div>
<div class="layout"><div><img id="frame"><h3>真实可用的四帧触觉</h3><div id="history"></div></div><pre id="details"></pre></div>
<script src="frames.js"></script><script>const d=window.EPISODE,s=document.getElementById('slider');s.max=d.frames.length-1;document.getElementById('title').textContent=d.title;const url=t=>'frames/'+String(t).padStart(4,'0')+'.jpg';function show(){const t=+s.value,r=d.frames[t],b=document.getElementById('badge');document.getElementById('frame').src=url(t);b.style.color=r.policy_valid_prefix?'#75e4ab':r.phase==='perturb'?'#ffc06d':'#aab8cc';b.textContent='观测 '+t+' → '+r.phase+' | '+(r.policy_valid_prefix?'BC 有效：动作 ['+t+', '+r.policy_label_stop+')，其余 '+(50-r.policy_valid_prefix)+' 个位置不含未来标签':'BC 无标签；保留真实历史与动力学');document.getElementById('details').textContent=JSON.stringify(r,null,2);const h=document.getElementById('history');h.replaceChildren();for(const j of r.tactile_window){const box=document.createElement('div'),caption=document.createElement('div'),c=document.createElement('canvas'),im=new Image();c.width=448;c.height=224;caption.textContent='t = '+j;box.append(caption,c);h.append(box);im.onload=()=>c.getContext('2d').drawImage(im,0,224,448,224,0,0,448,224);im.src=url(j);}}const jumps=document.getElementById('jumps');for(const t of [...new Set([0,...(d.perturbation?[d.perturbation.start-1,d.perturbation.start,d.perturbation.stop]:[]),...(d.timing?[d.timing.t_decision]:[]),+s.max])]){const b=document.createElement('button');b.textContent='观测 '+t;b.onclick=()=>{s.value=t;show()};jumps.append(b)}s.oninput=show;document.getElementById('prev').onclick=()=>{s.value=Math.max(0,+s.value-1);show()};document.getElementById('next').onclick=()=>{s.value=Math.min(+s.max,+s.value+1);show()};document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();s.value=Math.max(0,Math.min(+s.max,+s.value+(e.key==='ArrowLeft'?-1:1)));show()}});show();</script></html>'''


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--oracle-result",type=Path);p.add_argument("--self-result",type=Path)
    p.add_argument("--encoder-device",choices=("cpu","cuda"),default="cpu");a=p.parse_args(argv)
    result=json.loads((a.run/"result.json").read_text());a.output.mkdir(parents=True,exist_ok=True)
    encoder=None
    if a.encoder_device=="cuda":
        from .tactile import AnyTouchEncoder
        from .paths import WORKSPACE
        encoder=AnyTouchEncoder(WORKSPACE/"AnyTouch2",device="cuda",threads=4)
    rows=[]
    for kind in TITLES:
        override=a.oracle_result if kind=="oracle_recovery" else a.self_result if kind=="self_corrected" else None
        if override:
            source=override;chosen=json.loads(source.read_text())
            require(chosen["kind"]==kind and chosen["success"],"Override is not a successful sample of this kind")
        else:
            candidates=[x for x in result["branches"] if x["kind"]==kind and x["success"]]
            if not candidates:continue
            chosen=candidates[0];trace=Path(chosen["trajectory"])
            # Source result sits above trace/seed-N/repeat-0/trajectory.h5.
            source=trace.parents[3]/"result.json"
        suffix="-"+chosen["trial"]["id"] if override and chosen.get("trial") else ""
        rows.append(export_sample(source,a.output,seed=result["seed"],checkpoint=Path(result["checkpoint"]),episode_suffix=suffix,encoder=encoder))
    write_jsonl(a.output/"review_manifest.jsonl",rows)
    if rows:
        render_video(rows,a.output/"three_way_comparison.mp4")
        render_label_timeline(rows,a.output/"training_label_timeline.png")
    cards=[]
    for row in rows:
        relative=Path(row["review_page"]).relative_to(a.output.resolve()).as_posix()
        labels=sum(r["stop"]-r["start"] for r in row["policy_supervision_ranges"])
        cards.append(f'<li><a href="{html.escape(relative)}">{TITLES[row["kind"]]}</a>：{row["actions"]} 个动作，{labels} 个可模仿标签；扰动标签 0。</li>')
    (a.output/"index.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Insert Hole 三类样本</title><style>body{background:#101925;color:#e7edf5;font:17px/1.8 system-ui;max-width:1400px;margin:25px auto;padding:0 20px}a{color:#9fcaff}video,img{width:100%}</style><h1>Insert Hole 三类训练样本预览</h1><p>相同 seed 的配对对照；每个完整分支最多 317 个策略动作。绿色为可模仿标签，橙色 perturb 为人为准备状态，始终不作为策略标签。双视角、左右触觉及实际执行历史保留连续。</p><video controls src="three_way_comparison.mp4"></video><img src="training_label_timeline.png" alt="训练标签时间轴"><ul>'+''.join(cards)+'</ul><p><a href="review_manifest.jsonl">审核清单</a> · <a href="summary.json">检查结果与全部尝试</a></p><p>Oracle 重试通过回放已记录的 WAVE-0 与扰动命令重建受阻前缀，再从同一次仿真的快照恢复；它不作为一次新的自然策略 rollout 统计。</p><p>未加入训练清单，未启动训练。标签排除检查不等于保证训练后的模型不会形成行为捷径，后续仍需正常行为回归和闭环评测。</p></html>')
    atomic_json(a.output/"summary.json",dict(samples=[dict(kind=r["kind"],actions=r["actions"],
        labels=sum(i["stop"]-i["start"] for i in r["policy_supervision_ranges"]),review=r["review_page"]) for r in rows],
        all_three_present=len(rows)==3,training_enabled=False,collection_expanded=False,source_run=str(a.run.resolve()),
        original_attempts=[dict(kind=b["kind"],success=b["success"],actions=b["strategy_actions"],
            trial=b.get("trial"),reason=b.get("reason"),path=b["trajectory"]) for b in result["branches"]],
        oracle_retry=str(a.oracle_result.resolve()) if a.oracle_result else None))
    print(json.dumps(dict(exported=[r["kind"] for r in rows],training_enabled=False)),flush=True)


if __name__=="__main__":main()
