# HDMI / insert_hole：50 条恢复与 40 条对照

配置 `configs/recovery_mixed_90.yaml`，入口 `scripts/collect_recovery_mixed.sh`。
输出独立写入 `data/recovery_mixed_v3/`，旧 `recovery_oracle_v2` 保持停止。
每任务累计 25 条恢复、10 条正常、10 条扰动后自行纠正；复核的已有样本计入。
只采集、导出、审核。清单保持 `training_enabled=false`，不启动训练。

```bash
bash scripts/collect_recovery_mixed.sh --dry-run
bash scripts/collect_recovery_mixed.sh
bash scripts/collect_recovery_mixed.sh --status
bash scripts/collect_recovery_mixed.sh --stop-after-current
bash scripts/collect_recovery_mixed.sh --resume
```

`--import-only` 只复核并导入旧样本；`--max-attempts N` 限制本次新增的基线 seed 数。
每次启动继续同一账本，重启不返还额度或重复失败分支。
单 GPU 锁与既有评测入口兼容；监督器锁避免同一批次重复启动。
技术错误停止对应任务，普通任务失败扣额度后继续。技术错误需检查原因，不能以重启跳过。

## 采集与判定

先让原版双视角 WAVE-0 10k EMA 完整运行。自然失败使用持续受阻快照恢复；
正常成功保存对照，并从本次仿真的接近快照建立扰动分支。
在 10 步决策边界，目标表面距离 2–6 mm 或 0.5–2 mm、尚未接触、已有至少六步历史且夹持稳定时保存快照。
夹持稳定按相对初始夹持的平移不超过 2 mm、旋转不超过 3° 检查。
不存在合格快照时记录跳过，不凭空制造可用状态。

扰动使用目标局部 ±X / ±Y，HDMI 幅度 0.5/1/2 mm，hole 为 0.75/1.5/3/4 mm。
用实际 IK/控制接口执行 10 个策略动作，不移动物体真值，不额外施加旋转。
按已预约的尝试数量平衡方向、幅度和接近阶段；调度不使用全局随机数。
扰动后重新推理 WAVE-0，待完整运行结束后决定是否需要 Oracle。
仅实际位移达到请求的 25%、纠正后对应误差至少减少 0.1 mm 的原生成功计为自行纠正。

受阻判定沿用物体真实轴向进展、原生接触、连续三观测触觉变化，并保留对称进展解除规则。
扰动过程本身产生的受阻候选不触发恢复；后续 WAVE-0 的真实插入尝试需重新确认。
hole 固定使用 `hole-bounded-contact-alignment-v1` 与 2° 姿态容差。
HDMI 保持现有有效 Teacher。所有动作含前缀、扰动、卸载、调整均计入 600/317 策略动作预算。

## 标签和来源

原始 HDF5 保留 N 动作与 N+1 观测。正常和有效自行纠正动作提供 BC 标签；
扰动动作、恢复前缀、无效 Teacher 动作及失败分支只用于上下文和可用动力学转移。
动作目标在控制来源、扰动及新反馈决策处截断，排除的未来值在 flow 输入构造前替换。
AnyTouch2 使用连续 `[t-6,t-4,t-2,t]` 窗口；实际动作与触觉历史不清空。

每个任务/seed/类别最多计入一条。`group_id=任务/seed` 用于后续按场景分组划分数据，
配对分支不能被当成独立测试 seed。来源区分自然失败、物理扰动，以及实时策略执行或记录动作回放。
旧样本用新标签提案导入，原 HDF5、原结果和旧账本不改写。

## 额度、验收与产物

每任务最多 100 个新 seed、200 个扰动分支、100 个 Oracle 分支；每 seed 最多四次扰动及一次 Oracle。
预约先于执行，额度持久化。两任务交替串行；每任务新采 1 正常、1 自纠、3 恢复且包含扰动恢复，
经自动加载/标签检查通过后，状态由 `validating` 转为 `collecting`，继续运行至目标或额度耗尽。

`index.html` 展示三类数量、逐帧页面、视频及来源；`summary.json` 保存覆盖和全部尝试。
`manifests/<task>/normal.jsonl`、`self_corrected.jsonl`、`recovery.jsonl` 为独立审核清单。
每条成功样本有动作标签时间轴、50 位置掩码、真实四帧触觉窗口、加载检查和原生成功证明。
恢复样本另有同分支失败/恢复配对视频。每次导出更新索引，满足每新增五条更新的要求。
物理恢复检查不要求渲染像素一致；只比较被复制的真实记录前缀以发现拼接错误。
停止或完成后仅清理已经退出的原生进程所产生的临时物理快照，保留正式轨迹、配置、结果、视频和日志。
