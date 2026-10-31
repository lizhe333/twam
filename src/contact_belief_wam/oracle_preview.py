"""Review one explicitly privileged diagnostic, paired with its natural failure."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np

from .paths import WORKSPACE, atomic_json


def paired_video(reference, actual, result, path):
    import cv2
    n = len(actual["observations/joint_q"])
    m = len(reference["observations/joint_q"])
    def panel(handle, step, label, finished):
        obs = handle["observations"]
        t = min(step, len(obs["joint_q"])-1)
        output = np.zeros((480, 640, 3), np.uint8)
        for key, x, y, w, h, title in (
                ("head_rgb", 0, 45, 320, 180, "HEAD"),
                ("wrist_rgb", 320, 45, 320, 180, "WRIST"),
                ("left_rgb_marker", 0, 245, 320, 220, "LEFT TACTILE"),
                ("right_rgb_marker", 320, 245, 320, 220, "RIGHT TACTILE")):
            output[y:y+h,x:x+w] = cv2.resize(obs[key][t], (w,h))
            cv2.putText(output, title, (x+8,y+17), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,230,0), 1)
        suffix = " | END FRAME HELD" if finished else ""
        cv2.putText(output, f"{label} | step {t}{suffix}", (8,20), cv2.FONT_HERSHEY_SIMPLEX, .44, (255,255,255), 1)
        if handle is actual:
            phase = "initial" if t == 0 else result["action_metadata"][t-1]["phase"]
        else:
            phase = "same-seed WAVE-0, no intervention"
        cv2.putText(output, phase, (8,39), cv2.FONT_HERSHEY_SIMPLEX, .42, (120,220,255), 1)
        rel = (obs["prism_pose_world"][t][:3]-obs["slot_pose_world"][t][:3])*1000
        cv2.putText(output, f"GT object-slot xyz mm: {rel[0]:+.2f} {rel[1]:+.2f} {rel[2]:+.2f}",
                    (8,239), cv2.FONT_HERSHEY_SIMPLEX, .43, (255,255,255), 1)
        return output
    terminal = "SUCCESS" if result.get("success") else result["status"].upper()
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "1280x480", "-r", "15", "-i", "-", "-an", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    # Every observation, 15 fps: four times slower than the 60 Hz simulator.
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        for t in range(max(n,m)):
            left = panel(reference, t, "WAVE-0 / FINAL: FAILED", t >= m)
            right = panel(actual, t, f"GT-ASSISTED / FINAL: {terminal}", t >= n)
            process.stdin.write(np.concatenate([left,right], axis=1).tobytes())
        process.stdin.close()
        if process.wait():
            raise RuntimeError("Paired video encoding failed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    from .takeover_report import audit_result
    from .tactile import AnyTouchEncoder
    result = json.loads(args.result.read_text())
    physical = result.get("physical_diagnostic_requested") and result.get("physical_replay", {}).get("passed") and result.get("physical_takeover_restore", {}).get("passed")
    if result.get("kind") != "oracle" or not (result.get("replay", {}).get("passed") or physical):
        raise ValueError("Only a verified oracle branch can be presented as a paired recovery")
    encoder = AnyTouchEncoder(WORKSPACE / "AnyTouch2", device=args.device, threads=4)
    result = audit_result(args.result, encoder)
    out = args.result.parent
    with h5py.File(result["reference"], "r") as reference, h5py.File(result["trajectory"], "r") as actual:
        paired_video(reference, actual, result, out / "comparison.mp4")
        joint = actual["observations/joint_q"]
        n = len(joint)-1
        rel = actual["observations/prism_pose_world"][-1,:3] - actual["observations/slot_pose_world"][-1,:3]
        validation = dict(replay_passed=bool(result.get("replay", {}).get("passed")), physical_replay_passed=bool(physical), observations=len(joint), actions=n,
            n_plus_one=len(joint)==len(actual["actions/clipped_actual_target"])+1,
            budget_includes_prefix=n <= 600, substeps=actual["actions/substeps_executed"][:].tolist(),
            no_policy_positive_labels=not bool(actual["actions/policy_eligible"][:].any()),
            ground_truth_teacher=True, native_success=bool(result.get("success")),
            qualified_for_training=False, final_object_slot_mm=(rel*1000).tolist(),
            terminal_status=str(actual.attrs["terminal_status"]))
    atomic_json(out/"validation.json", validation)
    events = result.get("oracle", {}).get("events", [])
    lines = []
    for e in events:
        if "target" in e:
            lines.append(f'<tr><td>{e["step"]}–{e.get("stop", e["step"])}</td><td>{html.escape(e["phase"])}</td>'
                         f'<td>{e.get("actual_actions", 0)}</td><td>{html.escape(e["reason"])}</td></tr>')
    status = "恢复成功" if result.get("oracle_diagnostic_success") else "恢复未成功"
    pairing = ("同一 seed 重新运行 WAVE-0 得到新的失败参考；从同一次仿真的完整快照接管，经过十步动态复现校验。"
               "旧轨迹的跨初始化严格回放未通过，未将两者标成相同轨迹。" if result.get("reference_recreated_live")
               else "原失败轨迹的前缀逐步回放校验通过。")
    if not result.get("replay", {}).get("passed"):
        pairing = ("同一 seed 使用双视角 10k EMA WAVE-0 在 CPU 上重新运行得到失败参考。十步恢复检查的机器人、物体、"
                   "夹持相对位姿、原生表面和触觉标记坐标通过原容差；相机或触觉图像／深度像素未通过严格回放检查。"
                   "本视频仅展示真值辅助的物理恢复诊断，不是已验收的自然失败恢复样本，也不是旧 CUDA 录像的精确重放。")
    (out/"index.html").write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>HDMI 3000001 真值辅助恢复</title>
<style>body{{font:16px/1.65 system-ui;max-width:1280px;margin:32px auto;padding:0 20px;background:#10151d;color:#e9eef6}}a{{color:#9ac7ff}}video,img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #354252;text-align:left;padding:8px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
<h1>seed 3000001：{status}</h1>
<p>第 {result['takeover_step']} 步接管；整条分支实际执行 {result['strategy_actions']} / 600 个策略动作。{pairing}</p>
<p><strong>本次专家在线使用仿真真值。它验证物理恢复能力，不证明仅靠视觉／触觉就能恢复。数据独立保存，未加入训练。</strong></p>
<h2>失败／恢复对照</h2><p>左：本次同 seed WAVE-0；右：真值辅助恢复。双视角与左右触觉。四倍慢放；较早结束的一侧明确标记为末帧停留。</p>
<video controls preload="metadata" src="comparison.mp4"></video>
<h2>接管前模型可见输入</h2><img src="decision.png"><p>左右触觉四帧窗口，未使用未来帧：</p><img src="decision_touch_window.png">
<h2>反馈曲线</h2><img src="curves.png"><pre>{html.escape(json.dumps(result['timing'], ensure_ascii=False, indent=2))}</pre>
<h2>执行记录</h2><table><tr><th>策略步</th><th>阶段</th><th>实际动作</th><th>依据</th></tr>{''.join(lines)}</table>
<p><a href="result.json">完整结果与专家真值输入</a> · <a href="replay_check.json">严格回放校验</a> · <a href="physical_replay_check.json">十步物理校验</a> · <a href="physical_takeover_restore_check.json">接管物理校验</a> · <a href="validation.json">数据与预算检查</a> · <a href="decision_input.npz">接管时的连续输入历史</a></p></html>''', encoding="utf-8")
    print(json.dumps(dict(status=status, output=str(out), validation=validation | {"substeps": sorted(set(validation["substeps"]))}), ensure_ascii=False))


if __name__ == "__main__":
    main()
