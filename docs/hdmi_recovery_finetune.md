# HDMI 单视角两阶段恢复微调

配置：`configs/finetune_insert_HDMI_4xa800.yaml`。入口：`scripts/finetune_hdmi.sh`。
沿用 `head-only-expert-only-8task-all-train-fast-mb16-003` 的 HDMI 10k 完整 TWAM 与 EMA。
新输出：`/zhe_li/tactile-wam/twam/runs/hdmi-head-only-recovery-finetune-001/full`。

## 数据与监督

按用户要求，145 条全部训练：100 条 Original、25 条 Oracle 恢复、10 条正常成功、10 条扰动后自行纠正。
不留验证集、不做验证或早停；Stage 1 使用第 1000 步的最终 checkpoint。
同 seed 的配对关系仍保留在 `group_id`，以后闭环评测使用未参与采集的新 seed。

新清单保存在 `/zhe_li/tactile-wam/twam/data/hdmi_finetune_v1`，已有采集清单保持只读。
独立新清单启用训练，原始 HDF5、审核视频、旧模型不改写。

- Stage 1 采样完整有效动力学，包含已记录的扰动、受阻、卸载和恢复动作；正常／恢复来源约 1:1。
- Stage 2 只从存在有效 BC 标签的位置采样，正常／恢复来源约 2:1。恢复来源也包含自行纠正样本。
- 扰动和无效动作不进入 BC 目标；动作窗口在控制和反馈决策边界截断，被排除的未来动作在 flow 输入前替换。
- 触觉与动作历史保留完整，AnyTouch2 继续使用连续的 `[t-6,t-4,t-2,t]`；没有真值输入。
- 学生只读取 head RGB 与左右触觉。记录中的 wrist 留作审核。
- 新轨迹的 head RGB、关节和实际命令预处理成只读内存映射缓存，避免每个训练采样反复解压数百 MB 的诊断 HDF5。生成缓存时验证原记录，提交和重启时验证缓存哈希。

## 权重和预算

| 阶段 | 更新内容 | GPU 与 batch | 步数和学习率 |
| --- | --- | --- | --- |
| Stage 1 | Visual WM、动作／状态编码器、Tactile WM、Belief 和 Corrector | 4 卡 × 8，全局 32 | 1000 步；视觉 WM 2.5e-6，接触模块 2.5e-5 |
| Stage 2 | 已有 Policy、Belief、Corrector、策略接口；两个 WM 冻结 | 4 卡 × 16，全局 64 | 2000 步；Policy 5e-6，接触模块／接口 1e-5 |

两个阶段依次使用同一组 GPU。BF16 autocast、FP32 参数、梯度检查点；Stage 2 使用分片优化器与分片 GPU EMA。
Stage 1 从现有 TWAM 提取完整 WM／接触权重，视觉特征来自其冻结的 WM 编码器。
Stage 2 先加载原 HDMI 策略与 EMA，再覆盖刚完成的 WM／接触模块，保留原来的 Policy。
新数据实验重置优化器和更新计数；中断后只恢复这个新 run 的 checkpoint。不要传旧 `--stage2-restart-from` 或 `--stage1-checkpoint`。

推荐 **单机 4 张 A800 80GB**。原单视角 HDMI 的相同 Stage 2 batch 在四卡上实测每卡峰值分配约 49.87 GiB、保留约 52.68 GiB，吞吐中位数约 26.58 条序列／秒。
按旧吞吐估算，2000 步 Stage 2 约 80 分钟；新 WM 微调和新数据吞吐尚未在 A800 实测，整项任务可先预留 2–3 小时，首次数据准备时间另计。
这些是旧任务测量和估算，不是本次训练的实测结果。

## 百度云提交

使用以前的环境与相同挂载路径 `/zhe_li/tactile-wam`，先同步本次代码及新增输入。
`data/hdmi_finetune_v1/upload_files.txt` 列出新增代码、清单和模型输入／触觉缓存，路径相对于工作区；不必为训练上传全部原始诊断视频和 HDF5。
`training_files.txt` 列出完整运行依赖，原来的 100 条专家数据、缓存及 10k 模型通常已在云端。

可在采集机器打包新增文件，再在云端同一路径解包：

```bash
cd /zhe_li/tactile-wam
tar -cf outputs/hdmi-finetune-inputs.tar \
  -T twam/data/hdmi_finetune_v1/upload_files.txt
# 云端：tar -xf hdmi-finetune-inputs.tar -C /zhe_li/tactile-wam
```

本次提交包已生成在 `/zhe_li/tactile-wam/outputs/hdmi-finetune-inputs.tar`，约 1 GiB。
它包含新增代码、训练清单和新增模型输入／触觉缓存；旧的 100 条专家数据、模型和 Python 环境仍使用原任务已有文件。

云端任务启动命令：

```bash
set -euo pipefail
cd /zhe_li/tactile-wam/twam
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CBW_PYTHON=/zhe_li/tactile-wam/contact-belief-wam/.venv/bin/python
bash scripts/finetune_hdmi.sh train
```

脚本自动设置 PYTHONPATH、离线模型加载与线程数。新目录首次运行开始微调；相同命令可恢复本次任务。
`bash scripts/finetune_hdmi.sh plan` 只显示分卡与阶段计划；`prepare` 准备／校验输入；`check` 用 CPU 检查真实样本加载和标签，不训练。

日志在新输出目录的 `logs/stage1.log`、`logs/insert_HDMI.log`，状态文件为 `train_status.json`。
最终模型在 `tasks/insert_HDMI/export/final`；中间 checkpoint 每 250 步保存，保留最后四份。
训练损失和成功率是不同指标；完成后比较原模型和新模型的正常插入、受阻恢复闭环表现。

## 本地检查结果

76 项相关测试通过；缓存整理后的 17 项专项测试也通过。
生产加载器检查覆盖 145 条数据的两个阶段、2391 个采样位置，以及 1005 次被排除未来动作的改写检查。
两阶段权重继承、标签边界、连续历史和全量训练配置均已检查；本机未启动实际训练，新配置尚未在 A800 上实跑。
精简报告：`data/hdmi_finetune_v1/implementation_checks.json`。

## 从本轮 2k EMA 追加到累计 5k

本轮 RTX 4090 的 50 个固定场景 seed 评测已完成：原模型 2/50，恢复微调 2k 模型 4/50。
新增四条成功、原先两条成功丢失；历史基线没有配对策略噪声，不能据此确认稳定提升。
恢复动作训练损失仍下降，因此先追加 3000 个策略更新，之后再比较闭环成功率。

配置：`configs/finetune_insert_HDMI_continue_4xa800.yaml`，入口：`scripts/finetune_hdmi_continue.sh`。
从 `runs/hdmi-head-only-recovery-finetune-001/full/tasks/insert_HDMI/export/final` 的完整模型和 EMA 初始化，
保留其 WM、Policy、Belief、Corrector，不覆盖回旧 Stage 1，也不重新跑 Stage 1。
只更新原 Stage 2 的 Policy、Belief、Corrector 及接口，Visual WM 和 Tactile WM 继续冻结。

这是权重续训：优化器、采样计数和学习率调度重新开始，不是原 2k 作业的精确断点续跑。
新增 3000 步，Policy 学习率峰值 5e-6，Belief／新增接口峰值 1e-5，100 步 warmup；
global batch 64，单机四张 A800 80GB，每卡 micro batch 16，保留正常／恢复来源 2:1。
同样的 145 条数据通过原冻结清单和缓存读取，标签、归一化和训练划分不变；
新准备记录写入 `data/hdmi_finetune_continue_v1`，云端无需重新提供原始审核视频或采集账本。

云端同步更新代码后，使用上次环境与挂载路径提交：

```bash
set -euo pipefail
cd /zhe_li/tactile-wam/twam
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CBW_PYTHON=/zhe_li/tactile-wam/contact-belief-wam/.venv/bin/python
bash scripts/finetune_hdmi_continue.sh train
```

`plan` 仅查看 GPU 计划，`prepare` 校验并复用冻结数据，`check` 检查 Stage 2 生产加载器。
相同 `train` 命令在中断后恢复这次新作业。不要直接修改旧 2k 配方的 steps 后 resume，
旧优化器／学习率契约不支持这样的精确恢复。

新输出为 `runs/hdmi-head-only-recovery-finetune-002-continue-to5k/full`。
每 500 个新增更新保存一次，保留全部六份；新目录 step 1000 / 2000 / 3000 分别对应累计 3k / 4k / 5k。
原 2k 权重保留，最终导出仍位于新目录的 `tasks/insert_HDMI/export/final`。
按上一轮约 2.4 秒／更新及保存开销，预留约 2.5–3 小时。这里没有启动本地正式训练或提交云作业。

## 最新提交：普通权重重跑两阶段，5k＋10k，不用 EMA

用户确认以专家数据训练好的完整 TWAM 普通权重为起点，重新运行恢复微调。
使用 `configs/finetune_insert_HDMI_fresh_noema_4xa800.yaml` 和 `scripts/finetune_hdmi_fresh.sh`。
这次不使用上面追加 3k 的入口。

- Stage 1：5000 步，更新 Visual WM、动作／状态编码器、Tactile WM、Belief 和 Corrector。
- Stage 2：10000 步，更新 Policy、Belief、Corrector 和策略接口，两个 WM 冻结。
- 起点：`runs/head-only-expert-only-8task-all-train-fast-mb16-003/full/tasks/insert_HDMI/export/final/model.safetensors`。
- 两阶段分别从第 0 步开始，各自使用新优化器；Stage 2 继承本次 Stage 1 的 WM／接触模块和原专家策略普通权重。
- `finetune_use_ema: false`，`ema_enabled: false`，`stage2.shard_ema: false`：不读取原 EMA，不维护／导出新 EMA。
- 未来评测使用普通 `model.safetensors`，服务参数显式设为 `use_ema=False`；部署元数据标注 `use_ema: false`。
- 原 145 条数据、标签、归一化和来源比例保持不变；独立准备记录为 `data/hdmi_finetune_fresh_noema_v1`。
- 两阶段依次使用单机 4 张 A800 80GB。每 1000 步保存，保留最后 10 份 checkpoint。

将 `outputs/hdmi-fresh-noema-s1-5k-s2-10k-code.tar.gz` 同步到云端相同路径，沿用已有环境、数据和专家模型，提交：

```bash
set -euo pipefail
cd /zhe_li/tactile-wam
tar -xzf outputs/hdmi-fresh-noema-s1-5k-s2-10k-code.tar.gz
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CBW_PYTHON=/zhe_li/tactile-wam/contact-belief-wam/.venv/bin/python
bash twam/scripts/finetune_hdmi_fresh.sh train
```

输出：`runs/hdmi-head-only-recovery-finetune-003-fresh-noema-s1-5k-s2-10k/full`。
相同命令会恢复这次新作业，既有两轮模型和结果保留。
上一轮实测 Stage 1 约 0.32 秒／更新、Stage 2 约 2.4 秒／更新；
本次预计 Stage 1 约半小时、Stage 2 约 7–8 小时，连同初始化和保存可预留 8–9 小时。
这是基于旧 EMA 作业的耗时估算，不是步数或时长限制；新无 EMA 配方尚未实际运行。
