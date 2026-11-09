"""Render the pixel audit as an offline report; never edit traces or manifests."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from .paths import PROJECT, WORKSPACE, atomic_json, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "data/recovery_review_v1/insert_HDMI/pair_3000001_3000002/pixel_diagnosis")
    args = parser.parse_args(argv)
    out = args.output.resolve()
    read = lambda path: json.loads(path.read_text())
    existing = read(out / "existing_pair_audit.json")
    native = read(out / "native_probe_v2/result.json")
    if native["status"] != "complete":
        raise RuntimeError("Native diagnostic did not complete")
    validation = read(out.parent / "validation.json")
    rows = [json.loads(line) for line in (out.parent / "review_manifest.jsonl").read_text().splitlines()]
    source = Path(rows[0]["source_result"]).parent
    strict = read(source / "replay_check.json")
    physical = read(source / "physical_replay_check.json")
    original_hashes = existing["source_hashes"] | validation["source_hashes"]
    for row in rows:
        original_hashes[row["path"]] = row["trace_sha256"]
    unchanged = {p: sha256(p) == value for p, value in original_hashes.items()}
    if not all(unchanged.values()):
        raise RuntimeError("An original source changed")
    if any(row["training_enabled"] or row["training_approved"] for row in rows):
        raise RuntimeError("Review-only training guard changed")

    tactile = existing["tactile_features"]
    depth = [v for step in existing["pixel_metrics"] for v in step["depth"].values()]
    variants = ["default", "aa_off", "aa_and_denoiser_off"]
    rgb_keys = ["head_rgb", "wrist_rgb", "left_rgb_marker", "right_rgb_marker"]
    branch = {mode:{key:max(row[key]["mae"] for row in native[mode+"_step_pixels"][1:])
                    for key in rgb_keys} for mode in variants}
    fixed = native["fixed_rng_render_only"]
    fixed_max = {key:max(row[key]["mae"] for row in fixed["differences_from_first"])
                 for key in rgb_keys}
    preprocessed = {key:max(row["model_rgb_224"][key]["mae_levels"] for row in existing["pixel_metrics"])
                    for key in rgb_keys[:2]}
    cached = {mode:native[mode+"_step_pixels"][0] for mode in variants}
    if any(value["mae"] != 0 for row in cached.values() for value in row.values()):
        raise RuntimeError("Primed snapshot cache comparison is not exact")
    stationary = {mode:native[mode+"_render_only"]["physical_invariant_errors"] for mode in variants}
    stationary["fixed_rng"] = fixed["physical_invariant_errors"]
    if any(value != 0 for row in stationary.values() for value in row.values()):
        raise RuntimeError("The render-only diagnostic changed physical state")

    # The report records a limited engineering judgment, not training approval.
    assessment = dict(
        seed=3000001, attempt="attempt-05", status="pixel_diagnosis_complete",
        bc_assessment="usable_for_privileged_teacher_IL; exact_cross_branch_pixels_not_required",
        training_approved=False, training_started=False, expanded_collection=False,
        new_training_episodes=0, original_strict_pixel_check_passed=strict["passed"],
        original_physical_check_passed=physical["passed"],
        original_sources_unchanged=unchanged,
        review_manifest_sha256=sha256(out.parent / "review_manifest.jsonl"),
        original_thresholds_unchanged=True,
        best_reference_offsets={k:v["best_offset"] for k,v in existing["time_alignment"].items()},
        existing_max_model_rgb_mae_levels=preprocessed,
        existing_max_tactile_cosine_distance=max(tactile["branch_cosine_distance"]),
        contact_cosine_change=tactile["real_contact_cosine_change_36_to_40"],
        relative_contact_distance=tactile["max_branch_distance_over_contact_change"],
        contact_gate_identical=tactile["contact_gate_identical"],
        depth_outliers=sum(x["outlier_count_over_0_01mm"] for x in depth),
        depth_max_outliers_per_image=max(x["outlier_count_over_0_01mm"] for x in depth),
        depth_all_outliers_far_clip=all(x["all_outliers_are_far_clip_switches"] for x in depth),
        branch_max_rgb_mae_levels=branch, fixed_rng_max_rgb_mae_levels=fixed_max,
        static_physical_invariants=stationary, diagnostic_actions=native["physical_strategy_actions"],
        cached_takeover_equality_is_not_renderer_restore_proof=True,
        renderer_history_restore_not_implemented=True,
        aa_disable_control_valid=False,
        aa_requested_value=native["aa_off_value"],
        aa_observed_final_value=fixed["render_settings"]["/rtx/post/aa/op"],
        renderer_subcomponent_attribution="AA-off request was overridden to DLSS=3; this is NOT a valid AA ablation. Rendering state/noise is localized, but individual RTX components are not isolated.",
        policy_action_sensitivity_tested=False,
        limitations=[
            "Contact evidence timing was checked; the teacher still uses privileged true pose/progress.",
            "A small AnyTouch cosine distance is not a bound on action errors or success rates.",
            "Only the existing ten-action same-command comparison was encoded; recovery actions diverge afterward.",
            "Cached images are restored, but native renderer history is not; exact pixel counterfactual claims remain invalid.",
            "The render-only diagnostic advances sensor timestamps/noise, but not physics or strategy actions.",
        ],
        source_locations={
            "snapshot_cache":"pi05-univtac/src/pi05_univtac/contact_restore.py:553",
            "marker_noise":"UniVTAC/third_party/TacEx/source/tacex/tacex/simulation_approaches/fem_based/mani_skill_sim.py:360",
            "marker_rgb":"UniVTAC/third_party/TacEx/source/tacex/tacex/gelsight_sensor.py:397",
            "depth_inf_to_far":"UniVTAC/third_party/TacEx/source/tacex/tacex/gelsight_sensor.py:620",
            "render_update":"UniVTAC/envs/_base_task.py:474",
        },
        reproducibility=dict(
            offline="PYTHONPATH=twam/src python -m contact_belief_wam.pixel_diagnostic",
            native="PYTHONPATH=twam/src:pi05-univtac/src bash twam/scripts/sim.sh -m contact_belief_wam.sim_pixel_diagnostic --reference <paired-baseline.h5> --output <new-diagnostic-dir> --headless --device cuda:0",
            report="PYTHONPATH=twam/src python -m contact_belief_wam.pixel_diagnostic_report",
            native_script_sha256=native["source_sha256"],
            report_source_sha256=sha256(__file__),
            note="Use installed simulator environment and graphics runtime; never overwrite an existing diagnostic."))
    atomic_json(out / "assessment.json", assessment)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    t = np.arange(40, 51)
    for key, label, color in [("head_rgb", "Head", "#2563eb"), ("wrist_rgb", "Wrist", "#dc2626")]:
        axes[0, 0].plot(t, [x["rgb"][key]["mae_levels"] for x in existing["pixel_metrics"]], color=color, label=label+" raw")
        axes[0, 0].plot(t, [x["model_rgb_224"][key]["mae_levels"] for x in existing["pixel_metrics"]], "--", color=color, label=label+" 224 input")
    axes[0, 0].set(title="Existing pair: camera difference", xlabel="Observation index", ylabel="Mean absolute difference (0-255)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(t[1:], tactile["branch_cosine_distance"][1:], "o-", label="Replay difference")
    axes[0, 1].axhline(tactile["real_contact_cosine_change_36_to_40"], color="#dc2626", label="Real contact: t36 to t40")
    axes[0, 1].set(yscale="log", title="Actual four-frame AnyTouch2 input", xlabel="Observation index", ylabel="Mean token cosine distance")
    axes[0, 1].legend(fontsize=8)
    for mode, label, color in zip(variants, ["Default", "AA-off request (overridden)", "DL denoiser off; AA unverified"], ["#2563eb", "#dc2626", "#16a34a"]):
        axes[1, 0].plot(np.arange(1, 11), [x["head_rgb"]["mae"] for x in native[mode+"_step_pixels"][1:]], color=color, label=label)
    axes[1, 0].set(title="Native control: same commands, head RGB", xlabel="Actions after restore", ylabel="Mean absolute difference (0-255)")
    axes[1, 0].legend(fontsize=8)
    variable = native["aa_and_denoiser_off_render_only"]["final_consecutive_difference"]
    x = np.arange(4)
    axes[1, 1].bar(x-.18, [variable[k]["mae"] for k in rgb_keys], .36, label="Normal RNG: last consecutive frames")
    axes[1, 1].bar(x+.18, [fixed_max[k] for k in rgb_keys], .36, label="Fixed RNG: max versus first frame")
    axes[1, 1].set(xticks=x, xticklabels=["Head", "Wrist", "Left touch", "Right touch"], title="Same physical state: render-only RGB", ylabel="Mean absolute difference (0-255)")
    axes[1, 1].legend(fontsize=7)
    for ax in axes.flat:
        ax.grid(alpha=.18)
    fig.savefig(out / "diagnostic_curves.png", dpi=160)
    plt.close(fig)

    table = "".join(f"<tr><td>{label}</td>"+"".join(f"<td>{branch[mode][k]:.4f}</td>" for k in rgb_keys)+"</tr>"
                    for mode,label in zip(variants,["默认渲染", "请求关闭 AA（未保持生效）", "关闭 DL 去噪；AA 状态未逐帧核验"]))
    fixed_text = "、".join(f"{label} {fixed_max[k]:.6f}" for k,label in zip(rgb_keys,["head", "wrist", "左触觉", "右触觉"]))
    ratio = 100*tactile["max_branch_distance_over_contact_change"]
    sources = "".join(f'<li><code>{html.escape(path)}</code></li>' for path in assessment["source_locations"].values())
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>3000001 · 跨分支像素诊断</title><style>
body{{margin:0;background:#f5f7fa;color:#172333;font:16px/1.8 system-ui,sans-serif}}main{{max-width:1080px;margin:32px auto;padding:0 22px}}h1{{font-size:29px;line-height:1.4}}h2{{font-size:21px}}section{{background:white;border:1px solid #d8e1e9;border-radius:12px;padding:22px;margin:18px 0}}a{{color:#1758a4}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;text-align:right;border-bottom:1px solid #dae2eb}}td:first-child,th:first-child{{text-align:left}}.note{{border-left:4px solid #d69a2c;padding-left:14px}}.muted{{color:#536578;font-size:14px}}code{{overflow-wrap:anywhere}}.scroll{{overflow-x:auto}}
</style><main><p class="muted">HDMI / seed 3000001 / attempt-05 / 仅诊断</p><h1>这条恢复可用于真值 Teacher 模仿学习；不要求跨分支逐像素相同</h1>
<p>数据格式、动作标签和历史连续性已检查通过。像素差异没有破坏已查的物理状态与观测时序对应，触觉差异没有改变已测窗口的接触证据判定。没有依据仅因这项差异拒收示范。严格跨分支像素复现是额外的仿真诊断，不是行为克隆的必要门槛。</p>
<p><a href="../index.html">返回两条示范逐帧核对</a> · <a href="assessment.json">机器可读结论</a> · <a href="existing_pair_audit.json">原始样本逐步指标</a> · <a href="native_probe_v2/result.json">受控仿真实验</a></p>
<section><h2>定位到了什么</h2>
<p><b>相机：</b>快照恢复了 Python 层的图像缓存和物理状态，没有恢复 RTX 内部渲染历史。固定物理状态和 Torch 随机数后再次渲染，图像仍会改变；同动作回放也重现了差异。已定位到未被快照覆盖的渲染状态／随机性，尚未逐项区分 RTX 内部的历史、采样与去噪贡献。</p>
<p class="note">抗锯齿控制没有保持生效：写入 <code>/rtx/post/aa/op=0</code> 后即时读取为 0，但实验结束读取为 3（DLSS）。因此“关闭 AA 后仍有差异”不能排除 DLSS，下面的该组数值仅保留为诊断记录，不作为有效 AA 消融。DL 去噪结束时读取为 false。</p>
<p><b>触觉 RGB：</b><code>draw_markers()</code> 每次绘制标记点都会执行 <code>torch.rand_like(...) * 80</code> 并限幅，随后与光学触觉图叠加。这个 80 是标记图内部的噪声参数，不是整幅学生输入的平均误差。固定 RNG 的静态对照最大 RGB 差异：{fixed_text}（0–255 等级）。</p>
<p><b>深度：</b>已有十步回放共发现 {assessment['depth_outliers']} 个超过 0.01 mm 的像素差异，每张 240×320 图最多一个；全部在有效深度与 34 mm 无穷值替代值之间切换。其余像素基本一致。深度不是学生的直接输入；它对触觉图的间接影响已通过实际 RGB 与 AnyTouch2 编码一起评估。</p>
<p class="note">接管瞬间“像素完全一致”包含恢复缓存的效果，不能作为渲染器历史已复原的证据。报告保留原严格回放失败结果，没有放宽阈值或修改原始图像。</p></section>
<section><h2>对这条数据的影响有多大</h2>
<p>已有探针的关节、触觉标记、head 和 wrist 在前后 ±3 帧搜索中，都以 <b>0 帧偏移</b> 最匹配。原物理等价检查通过，执行动作相同。模型 224×224 预处理后的最大平均像素误差：head <b>{preprocessed['head_rgb']:.3f}/255</b>，wrist <b>{preprocessed['wrist_rgb']:.3f}/255</b>。</p>
<p>按真实 <code>[t−6,t−4,t−2,t]</code> 窗口重算 AnyTouch2，最大跨分支余弦距离 <b>{max(tactile['branch_cosine_distance']):.6f}</b>；真实接触前后 t36→t40 的距离为 <b>{tactile['real_contact_cosine_change_36_to_40']:.6f}</b>。前者约为后者的 <b>{ratio:.2f}%</b>。两分支 t40–t50 的接触证据门控全部相同。这是特征尺度的比较，不是成功率或动作误差上界。</p>
<img src="diagnostic_curves.png" alt="原样本像素误差、AnyTouch 特征差异和渲染控制对照曲线">
<p class="muted">右下图是相同物理状态下的渲染测试。普通随机数列比较末两帧；固定随机数列取全部帧相对首帧的最大误差，两者使用不同汇总方式，不能用于精确百分比归因。</p></section>
<section><h2>同一状态的控制实验</h2><p>只执行既有前缀与十步动作，三种配置各在继续后恢复同一物理快照。下表取接管后第 1–10 步 RGB MAE 的最大值，单位为 0–255 像素等级；不把恢复的第 0 帧缓存计入新渲染结果。</p>
<div class="scroll"><table><thead><tr><th>配置</th><th>head</th><th>wrist</th><th>左触觉</th><th>右触觉</th></tr></thead><tbody>{table}</tbody></table></div>
<p>静态重绘时，策略动作计数、物理步数、UIPC 帧、关节及连接器／插槽顶点的差异均为 0。该测试会更新传感器时间戳、缓存与噪声，不能作为采集时额外插入渲染帧的建议。</p>
<p class="muted">初版诊断在相机懒加载刷新前拍快照，第 0 帧出现了额外缓存顺序误差；v2 与实际 Runner 一样先读取观测再拍快照，三种配置的第 0 帧均完全相同。初版不用于上表或训练判断。全部测试没有调用策略服务或恢复 Teacher。</p></section>
<section><h2>训练使用判断</h2><p><b>这条示范可以用于真值 Teacher 模仿学习。</b>继续使用本恢复分支实际记录的双视角、触觉和动作，保持历史连续；不要拿失败分支后续图像替换恢复观测，也不要把新渲染当成真实接触变化。</p>
<p>严格物理／像素反事实比较应分别报告物理等价与像素不等价。学生能否学会恢复、是否使用触觉，要由后续训练和闭环评测回答；本次没有测量策略对像素差异的动作敏感性，也不把此项额外实验当作开始模仿学习的前置门槛。</p>
<p>原始数据与源结果哈希一致；原清单仍为 <code>training_enabled=false</code>、<code>training_approved=false</code>。没有新增训练样本，没有扩量，没有训练。原有数据加载、标签边界和历史连续性检查结果保留。</p></section>
<section><h2>可复核证据与源码</h2><img src="existing_difference.png" alt="原始 t41 和跨分支 t41 及放大像素差分"><p class="muted">差分图作了显示放大；计算均使用未压缩原始 HDF5。</p><ul>{sources}</ul><p><a href="assessment.json">assessment.json</a> 记录源哈希、诊断命令、限制和训练隔离状态。诊断只用 seed 3000001，未将重放结果登记为新数据。</p></section>
</main></html>'''
    (out / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"report":str(out / "index.html"), "training_enabled":False, "fixed_rng_rgb":fixed_max}, ensure_ascii=False))


if __name__ == "__main__":
    main()
