# 对齐版工程检查记录

2026-10-30 归档说明：本文保留历史工程检查结论；已完成的调试 stdout 和 smoke 视频已清理，文中这些旧路径不再表示文件仍然存在。精简验证证据保留在工作区 `contact-belief-wam/outputs/aligned_validation/`，清理清单见 `reports/artifact-cleanup-20261030/verification.json`。

本轮将训练与部署的动作表示、归一化、图像处理、tokenizer、优化器、学习率和 EMA 约定对齐到 `wave0/single-view`，保留 Contact Belief、触觉预测及恢复数据这些方法差异。正式入口为 `scripts/aligned_train.sh`，配置为 `configs/aligned_8xa800.yaml`。没有启动正式训练，也没有重新下载基础权重或制作上传包。

## 数据与训练约定

- Stage 1：八任务原始训练轨迹 720 条、插入任务恢复轨迹 100 条；使用参考实现固定的 80 条原始验证轨迹。每卡 batch 32 精确按任务均衡；两个插入任务各自按原始与恢复 1:1 采样。
- Stage 2：每个插入任务使用全部 100 条原始轨迹与 50 条恢复轨迹，按 1:1 采样。两个对比版本共用清单、统计、来源采样与策略动作 mask，分别训练自己的 Stage 1 teacher。
- 前七维动作相对于 chunk 起始状态，夹爪保持绝对值；32 维 flow 目标及末尾填充监督沿用参考实现。部署端恢复为 8 维绝对关节目标。
- 策略使用逐任务统计，世界模型使用八任务共享统计；历史动作使用上一反馈区间的起始状态作为相对动作锚点。统计直接复制参考版本，未用恢复数据重新计算。
- Stage 1 使用 8×32=256，最多 5,000 步及参考版本的验证/早停规则；Stage 2 两任务各 2×32=64，分别 10,000 步、并行执行。GA 均为 1。

数据统计见 `outputs/aligned_validation/data_preparation.json`；命令解析结果见同目录的 `full_plan.json` 和 `wave0_recovery_plan.json`。

## 已完成的模型检查

完整尺寸 `full` 模型已在 RTX 4090 上通过共享 Stage 1、两个任务 Stage 2 的真实数据前反向、参数更新和保存恢复。原始与恢复样本均参与检查；有梯度参数均产生梯度，冻结参数未改变。固定噪声的保存前与重新加载后输出最大差值均为 **0**，优化器、学习率位置、RNG 和 Stage 2 EMA 恢复检查通过。

仅执行计算验证更新：Stage 1 含一次保存前诊断失败的尝试，共两次；每个 Stage 2 一次。完成后的报告复查没有重复更新。报告位于 `outputs/aligned_debug/full/`。调试用了 CPU 主参数/优化器卸载与小 batch；这些设置没有写入正式训练配置。

CPU 测试：**149 passed、2 skipped**，日志 `outputs/aligned_cpu_tests_final.log`。CUDA 注意力与序列检查：**10 passed**，日志 `outputs/aligned_gpu_attention_tests.log`。两进程 CPU DDP 的递归序列、梯度检查点和跨 rank 梯度一致性检查通过，见 `outputs/aligned_validation/ddp_cpu.jsonl`；该检查不执行优化器更新。官方路径数值对齐采用缩小宽度的真实模块测试，完整尺寸模型另由上述真实 batch 检查覆盖。

完整模型的真实在线 RPC 已通过：50×8 动作有限，目标步号为 10；实际 AnyTouch2 CPU 编码及模型 CPU BF16 推理共约 5.9 秒。记录见 `outputs/aligned_validation/real_cpu_rpc.json`。`CBWAM_DEBUG_CPU_BF16=1` 仅用于本地服务 smoke，不属于正式训练配置。

配套 `wave0_recovery` 基线的共享 Stage 1、`insert_hole` Stage 2、`insert_HDMI` Stage 2 也全部通过相同检查，每个模型仅更新一次，固定噪声输出差值均为 **0**。三个更新均包含真实数据，训练参数无缺失梯度、冻结参数未变。报告已持久保存到 `outputs/aligned_validation/wave0_recovery/`，原始更新记录见该目录 `update_events.json`；大型 debug checkpoint 位于 `/tmp/cbwam-aligned-debug/wave0_recovery/`，不是正式训练输入。

基线 hole 的保存后检查曾因 CPU 模型服务与恢复过程共用系统内存被终止。已释放服务并修复恢复时重复持有优化器快照的问题；从完整保存的 checkpoint 继续核对，没有重复更新。保存后恢复逻辑的相关 CPU 测试 **14 passed**，日志 `outputs/aligned_debug_resume_tests.log`。

新进程恢复时还发现并修复了 CUDA 延迟初始化覆盖已恢复 RNG 的问题：先初始化 CUDA，再设置保存的随机状态。真实 CUDA 新进程回归测试 **1 passed**（`outputs/aligned_cuda_rng_tests.log`），训练/恢复相关 CPU 测试 **26 passed**（`outputs/aligned_restore_tests.log`）。修复后完整 checkpoint 恢复通过。最终基线 debug 进程正常退出，日志 `outputs/aligned_debug_wave0_recovery_verified.log`。

## 仿真闭环

实际设备为 RTX 4090 仿真、CPU BF16 策略推理、CPU AnyTouch2。均使用完整模型、EMA 和真实在线触觉；两个完整 episode 沿用原任务默认预算与成功/失败判据，短检查单独标明。

| 检查 | 策略步数 | 反馈次数 | 原生结果 | 工程结果 |
|---|---:|---:|---|---|
| `insert_hole`，seed 1000000，k=10 | 141 | 15 | failed | 正常终止，无运行异常 |
| `insert_HDMI`，seed 1000000，k=10 | 13 | 2 | success | 正常终止，无运行异常 |
| `insert_hole`，seed 1000000，k=5，短检查 | 10 | 2 | timeout | 达到指定短预算，无运行异常 |

三次检查的目标步号均严格等于当前步号加 k；首轮历史预测无效，之后每次历史预测均正确匹配。belief token 范数约为 22.6272–22.6274，没有出现数值发散；该范数也受到模块 LayerNorm 的影响，不能单凭范数认定记忆有效。CPU 推理平均约 3.8–4.0 秒/次。

汇总见 `outputs/aligned_validation/smoke_summary.json`。视频位于：

- `outputs/aligned_smoke/insert_hole/video/1000000_failed.mp4`
- `outputs/aligned_smoke/insert_HDMI/video/1000000_success.mp4`
- `outputs/aligned_smoke/k5/video/1000000_timeout.mp4`

各目录的 `episodes/1000000.json` 保存动作、反馈、预测误差和耗时，`episodes/1000000.png` 保存 belief 与预测误差曲线。两个任务的视频均已解码并检查末帧。只做过计算验证更新的模型在单一种子上的成功/失败不用于评估方法效果。

## 正式运行

在挂载同一路径的 8×A800 作业中运行：

```bash
bash /zhe_li/tactile-wam/contact-belief-wam/scripts/aligned_train.sh train \
  --config /zhe_li/tactile-wam/contact-belief-wam/configs/aligned_8xa800.yaml \
  --variant full \
  --output /zhe_li/tactile-wam/contact-belief-wam/runs/aligned-wave0-fast-v1 \
  --resume none
```

续训将最后一个参数改为 `--resume auto`。配套基线另提交一个作业，将 `--variant full` 改为 `--variant wave0_recovery`，它会使用独立输出子目录和独立 teacher。不要恢复旧版绝对动作 checkpoint，也不要用本地 debug checkpoint 初始化正式训练。

正式配置保留 FP32 参数、BF16 autocast、SDPA、fused AdamW、GPU FP32 EMA 和梯度检查点。A800 实际显存与吞吐尚未测量；本轮工程检查不代表已验证正式训练收敛、任务成功率或恢复能力。
