# 头部单视角、纯专家轨迹的两阶段训练

2026-10-30 归档说明：未保存 checkpoint 且已被 `head-only-expert-only-8task-all-train-fast-mb16-003` 替代的 `fast-002` 运行记录已清理。该次尝试的配置与迁移说明保存在工作区 `reports/artifact-cleanup-20261030/retained-inputs/twam/runs/`；下文涉及它的命令和路径用于说明历史重启过程。

初版实验使用 `configs/aligned_8xa800_expert_only.yaml`，代码分支为 `expert-only-two-stage`。第一次优化使用 `aligned_8xa800_expert_only_fast.yaml`；每卡 16、取消梯度累积的续训命令见文末 `aligned_8xa800_expert_only_fast_mb16.yaml`。`camera_mode: head_only` 同时控制 Visual WM、触觉预测器的视觉条件、策略视觉输入和导出后的推理配置。AnyTouch2 继续使用左右触觉输入；单视角指 RGB 相机只使用 head。

复用的视觉基座是 `wave0-shared-8task-mb32-002` 选出的 `step_00004500`，以及该 checkpoint 配套的 `wm_norm_stats.json`。它由官方 WAVE-0 RoboTwin 世界模型初始化，再在 UniVTAC 八任务上训练。Visual WM 和 world action/state encoders 保持冻结。这里的 Contact Stage1 只训练新增触觉预测器、Belief Updater、Future Corrector 及接口，不重新训练视觉 WM，也不继承含 Recovery 的 Contact Stage1。

| 阶段 | 数据 | 训练范围 | 卡数、batch | 更新次数 |
| --- | --- | --- | --- | --- |
| Stage1 | 八任务全部 800 条 Original，无验证集 | Tactile WM、Belief、Corrector 及本阶段接口 | 8 卡 × 每卡 32，累积 1 次，全局 256 | 固定 5,000，无验证和早停，使用最后一步 checkpoint |
| Stage2，每个任务独立 | 对应任务自己的 100 条 Original | 策略、Belief、Corrector；冻结 Tactile WM | 每任务 4 卡 × 每卡 8，累积 2 次，全局 64；最多两个任务并行 | 每个任务 10,000，共八份策略 |

八个策略任务为 `insert_hole`、`insert_HDMI`、`insert_tube`、`pull_out_key`、`grasp_classify`、`lift_bottle`、`lift_can`、`put_bottle_in_shelf`。每个策略都从同一个新 Contact Stage1 checkpoint 和基础 π0.5 初始化，在独立目录训练；不会继承上一个任务的策略权重。

两阶段保留原来的损失、学习率、GPU EMA、优化器状态分片和计算复用。Stage2 继续让动作损失训练 Belief，不新增接触分类标签。训练过程不留出验证轨迹、不执行验证或早停；最终策略效果可通过独立仿真 rollout 检查。

`expert_only: true` 会在数据准备时过滤 Recovery；`data_mix: '1:0'` 用于采样。`stage1.validate_every: 0` 和 `wm_validation_episode_ids: []` 将全部 800 条专家轨迹用于训练，关闭验证，并让入口使用 Stage1 最终 checkpoint 衔接策略。即使源清单里还列着 Recovery、对应文件已经不可用，也不会读取它们。Original 的时间段及配套归一化保持一致。训练器额外检查输入清单只能包含 Original；外部 Stage1 初始化必须有纯专家训练身份记录。

派生清单保存在 `/zhe_li/tactile-wam/contact-belief-wam/data/manifests/aligned_expert_only_all_train`，统计副本保存在 `/zhe_li/tactile-wam/contact-belief-wam/data/stats/aligned_expert_only_all_train`。原始 HDF5、AnyTouch2 特征缓存、旧混合清单和 Recovery 训练结果保留。原 `aligned_8xa800.yaml`、`aligned_4xa800.yaml` 仍描述之前的双视角混合数据实验。

在百度云选择单机 8 张 A800，沿用原挂载路径，启动命令如下。入口会准备清单、训练新增接触模块，然后自动调度八个 Stage2。任务完成、对应四卡空闲后会启动下一个任务，最多两个任务并行。无需外层再套 `torchrun`。

```bash
set -euo pipefail
cd /zhe_li/tactile-wam/twam

export CBWAM_ROOT=/zhe_li/tactile-wam/twam
export TWAM_ROOT=/zhe_li/tactile-wam
export CBW_PYTHON=/zhe_li/tactile-wam/contact-belief-wam/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ ! -e "$CBWAM_ROOT/vendor/transformers" ]; then
  ln -sT /zhe_li/tactile-wam/contact-belief-wam/vendor/transformers "$CBWAM_ROOT/vendor/transformers"
fi

bash scripts/train.sh train \
  --config configs/aligned_8xa800_expert_only.yaml \
  --variant full \
  --output /zhe_li/tactile-wam/twam/runs/head-only-expert-only-8task-all-train-001 \
  --resume auto
```

首次启动的新目录里没有 checkpoint，因此 `--resume auto` 从新 Stage1 开始；中断后使用完全相同的代码、配置和目录可继续本次实验。命令不传 `--stage1-checkpoint`，避免跳过这次需要重训的接触模块。`world_init` 仍加载上面的纯视觉 WAVE-0 基座。

仅检查分卡计划可在最后一条命令后加 `--plan`；仅准备数据可将 `train` 改为 `prepare`。这两种检查都不会启动 GPU 训练。

新结果根目录为 `/zhe_li/tactile-wam/twam/runs/head-only-expert-only-8task-all-train-001/full`，其中 `logs/stage1.log` 记录接触模块训练，`logs/<task>.log` 分别记录八个任务策略训练。最终八份策略位于各自的 `tasks/<task>/export/final`。

## 优化版断点重启

`configs/aligned_8xa800_expert_only_fast.yaml` 保持全部 800 条专家轨迹、头部单视角、八个独立任务各 10k 步、现有损失、学习率和冻结范围。Belief 保持顺序递推、历史预热与两步反向传播。两步 Belief 构建完成后，把这两个时刻的 VLM／Action Expert 计算合并为一个 batch；增强、噪声和时间采样仍按原来的时刻顺序生成，损失仍按原序列和有效 mask 归一化。

每任务仍用四卡、每卡 8 条序列、累积 2 次、全局 batch 64；合并后的大模型前向通常包含 16 个时刻样本。两个任务并行。`stage2.shard_ema: true` 把 FP32 EMA 参数分摊到四张 GPU，训练期间不搬到 CPU，不增加每步 EMA 通信。当前参数量对应每卡约 3.15 GiB；保存时汇总为原有的单个 `ema.safetensors`，部署接口保持兼容。

先停止旧云端作业，再在单机八卡 A800 上执行：

```bash
set -euo pipefail
cd /zhe_li/tactile-wam/twam

export CBWAM_ROOT=/zhe_li/tactile-wam/twam
export TWAM_ROOT=/zhe_li/tactile-wam
export CBW_PYTHON=/zhe_li/tactile-wam/contact-belief-wam/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ ! -e "$CBWAM_ROOT/vendor/transformers" ]; then
  ln -sT /zhe_li/tactile-wam/contact-belief-wam/vendor/transformers \
    "$CBWAM_ROOT/vendor/transformers"
fi

bash scripts/train.sh train \
  --config configs/aligned_8xa800_expert_only_fast.yaml \
  --variant full \
  --stage1-checkpoint /zhe_li/tactile-wam/twam/runs/head-only-expert-only-8task-all-train-001/full/shared_wm/stage1/checkpoints/step_00005000 \
  --output /zhe_li/tactile-wam/twam/runs/head-only-expert-only-8task-all-train-fast-002 \
  --resume auto
```

Stage1 已完成，这条命令直接复用其最终 checkpoint。首次重启时，`stage2_restart_from` 从旧目录为各任务选择最新带 `complete.json` 的 checkpoint，恢复原始模型、分片优化器、EMA、更新步数、采样位置和随机状态；没有 checkpoint 的任务从相同 Stage1 正常初始化。每个任务仍训练到总计 10k 步。旧目录只读，新 checkpoint、日志和策略导出写入 `head-only-expert-only-8task-all-train-fast-002/full`。

兼容迁移只允许执行方式变化，默认拒绝 batch 变化，并拒绝训练数据、Belief 反向步数、学习率、教师权重或无关源码变化。下文每卡 16 的配置显式允许在全局 batch 和卡数不变时调整 microbatch／累积次数。任务目录内的 `performance_restart.json` 记录迁移来源。后续再次执行同一命令时优先使用新目录的 checkpoint，并恢复严格的代码／配置一致性检查。合并运算可能改变浮点舍入，不声称训练轨迹逐位一致。

模型和梯度等价检查使用真实小规模 WAVE/Contact 网络，包括 BF16、增强、有效性 mask、梯度检查点与仅由第二时刻动作损失回传到前一 Belief 的检查。分布式检查在 CPU Gloo 上执行；此开发机无法实测 A800 的显存峰值和加速幅度，重启后应以新日志中的 `global_sequences_per_second`、`timing`、`cuda_memory` 为准。

## 每卡 16、取消梯度累积

`configs/aligned_8xa800_expert_only_fast_mb16.yaml` 把 Stage2 从 `4 × 8 × 2 = 64` 改为 `4 × 16 × 1 = 64`。累积次数由入口自动计算。仍然是两任务各四卡同时训练，两个 Belief 时刻保持递推和反向传播，VLM／Action Expert 合并后一次前向包含 32 个时刻样本。GPU 分片 EMA、学习率、梯度裁剪、冻结范围、损失和每任务总计 10k 步全部保留，Stage1 继续复用已经完成的纯专家版本。

配置的 `stage2_restart_from` 指向 `head-only-expert-only-8task-all-train-001/full`，并设置 `stage2_restart_rebatch: true`。2026-10-27 用户停止 `fast-002` 时，Hole 和 HDMI 的日志到 4800 步，但该目录尚未保存 checkpoint；因此从 `001` 的完整 4000 步存档恢复，未保存的约 800 步需要重跑。这项显式迁移只允许纯专家 Stage2 在全局 batch、进程数不变时重新分组，恢复模型、优化器、EMA、更新步数、采样计数和 RNG 状态。样本在各 rank 上的分配、增强和噪声与样本的对应关系会变化，所以不保证训练轨迹逐位一致。没有启动过的另外六个任务正常从共享 Stage1 初始化。

旧云端作业已经停止，可直接使用下面命令。以后暂停时，可先确认各任务 `stage2/checkpoints/step_XXXXXXXX/complete.json` 已写完，减少未保存进度的损失；`.partial` 不能用于续训。入口会拒绝从“已有训练日志但没有完整 checkpoint”的任务静默重新训练。首次启动选取旧目录每任务最新完整 checkpoint，之后优先选择新目录自己的 checkpoint。

单机八卡 A800 的完整启动命令：

```bash
set -euo pipefail
cd /zhe_li/tactile-wam/twam

export CBWAM_ROOT=/zhe_li/tactile-wam/twam
export TWAM_ROOT=/zhe_li/tactile-wam
export CBW_PYTHON=/zhe_li/tactile-wam/contact-belief-wam/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ ! -e "$CBWAM_ROOT/vendor/transformers" ]; then
  ln -sT /zhe_li/tactile-wam/contact-belief-wam/vendor/transformers \
    "$CBWAM_ROOT/vendor/transformers"
fi

bash scripts/train.sh train \
  --config configs/aligned_8xa800_expert_only_fast_mb16.yaml \
  --variant full \
  --stage1-checkpoint /zhe_li/tactile-wam/twam/runs/head-only-expert-only-8task-all-train-001/full/shared_wm/stage1/checkpoints/step_00005000 \
  --output /zhe_li/tactile-wam/twam/runs/head-only-expert-only-8task-all-train-fast-mb16-003 \
  --resume auto
```

旧目录保留。新结果写入 `head-only-expert-only-8task-all-train-fast-mb16-003/full`，不会额外再训练 10k 步。重启后先观察前两步的 `memory.rank*.jsonl` 和至少 100 步的稳定速度；本机的 RTX 4090 不能代替 A800 测量训练峰值或加速比例。2026-10-27 的 batch 8 日志显示，累积第一轮反向之后会暂存约 12.6 GiB 额外梯度；取消累积有助于容纳更大的前向 batch，但实际显存峰值仍以云端日志为准。

## 10k 完成后自动评测

`scripts/watch_eval_10k.py` 监视当前 `fast-mb16-003/full` 下的八个任务，每 30 秒扫描一次。只接受 `tasks/<task>/stage2/checkpoints/step_00010000/complete.json` 已提交且训练身份一致的 checkpoint；不会用训练日志中的步数或尚未保存完的 `.partial` 目录触发评测。发现后校验模型和 EMA 的哈希，并固定一份推理文件，避免训练清理 checkpoint 后队列丢失权重。只读训练目录，不修改训练源码、优化器或进程。

评测在挂载同一数据目录的本机 RTX 4090（GPU 0）串行运行。每个任务使用头部 RGB、在线 AnyTouch2、EMA 权重和固定的 50 个 seed（1000000–1000049），沿用 WAVE-0 的 `bounded-actions-fixed-seeds-v2` 原生仿真流程。先做渲染与物理兼容检查，GPU 被其他计算进程占用时等待；不会占用云端训练用的八张 A800。评测期间继续扫描，新完成的任务会追加到队列。

在本机启动一次即可，退出终端后继续运行：

```bash
cd /zhe_li/tactile-wam/twam
mkdir -p /zhe_li/tactile-wam/outputs/eval_suites/tactile-head-only-expert-mb16-003-10k-fixed-seeds

nohup /zhe_li/tactile-wam/contact-belief-wam/.venv/bin/python -u \
  /zhe_li/tactile-wam/twam/scripts/watch_eval_10k.py \
  --run-root /zhe_li/tactile-wam/twam/runs/head-only-expert-only-8task-all-train-fast-mb16-003/full \
  --suite /zhe_li/tactile-wam/outputs/eval_suites/tactile-head-only-expert-mb16-003-10k-fixed-seeds \
  --gpu 0 --episodes 50 --poll-seconds 30 \
  >> /zhe_li/tactile-wam/outputs/eval_suites/tactile-head-only-expert-mb16-003-10k-fixed-seeds/monitor.log \
  2>&1 < /dev/null &
```

这个进程需要本机 GPU 和回环端口访问权限，应在普通终端运行。结果目录中的 `daemon.json` 是监视器状态，`manifest.json` 是持久化队列，`summary.csv` 汇总各任务成功率，`results/` 保留每个任务的正式评测结果和日志。查看进度：

```bash
tail -n 30 /zhe_li/tactile-wam/outputs/eval_suites/tactile-head-only-expert-mb16-003-10k-fixed-seeds/monitor.log
cat /zhe_li/tactile-wam/outputs/eval_suites/tactile-head-only-expert-mb16-003-10k-fixed-seeds/summary.csv
```

重复启动会由队列锁拦截；中断后用相同命令恢复，已完成结果不重复评测。单次评测失败最多重试一次，仍失败则保留日志并继续其他任务。GPU／仿真环境尚不可用时保留队列等待重试。全部八个任务评测完成或达到重试上限后，监视器退出。训练目录变更时同时指定新的 `--run-root` 和 `--suite`，避免混合实验结果。
