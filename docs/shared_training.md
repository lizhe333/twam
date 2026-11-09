# 八任务共享 WM 与独立恢复策略

> 历史配方说明：本页对应旧 shared/legacy 入口，不是当前默认训练流程。当前双视角两阶段、冻结范围与学习率见 [tactile_training.md](tactile_training.md)。

2026-10-30 已清理旧 `shared-wm8-k10-8xa800-bs512` 无 checkpoint 作业的日志；配置、数据清单和统计已迁入工作区 `reports/artifact-cleanup-20261030/retained-inputs/contact-belief-wam/runs/`。下文该作业路径仅作历史记录。

开发机只做数据准备、冻结特征缓存和每个模型至多两次更新的真实计算检查。正式作业由用户在百度云手动提交，挂载路径与开发机完全相同。主配置为 [`shared_8xa800_bs64.yaml`](../configs/shared_8xa800_bs64.yaml)，复用已有 π0.5、WAVE-0 权重与项目环境，不下载权重，不制作上传包。

## 训练内容

Stage 1 使用 `insert_hole`、`insert_HDMI`、`insert_tube`、`pull_out_key`、`grasp_classify`、`lift_bottle`、`lift_can`、`put_bottle_in_shelf` 共八个任务的原始演示，再加入两个插入任务各 50 条恢复演示，共 900 条。按最新配置，900 条全部用于训练，不保留验证集。先均匀采样任务，再在插入任务内部按 Original:Recovery=1:1 采样；没有恢复数据的六个任务只采样原始数据。

动作与状态统一使用八任务全部 900 条轨迹的 q01/q99。Stage 2 的两个策略继续使用同一份统计，并将其随各自 checkpoint 导出，避免共享 WM 在阶段切换时读到含义不同的归一化动作。8 维绝对关节目标、`action[t]=q[t+1]`、`k=10`、50 步 chunk 均保持不变。

| 阶段 | 更新模块 | 冻结模块 | 数据安排 |
|---|---|---|---|
| 共享 Stage 1，默认 5,000 步 | Visual WM、动作/状态编码器、Belief Updater、Contact Predictor、Future Corrector | 视觉塔、AnyTouch2；不训练策略主干 | 八任务原始数据＋两任务恢复数据 |
| 每任务 Stage 2，默认 10,000 步 | VLM 语言部分、latent queries、Resampler、Action Expert、belief 接口投影、Future Corrector | 视觉塔、AnyTouch2、共享 WM/动作/状态编码器、Belief Updater、Contact Predictor | 前 1,000 步仅 Original，之后 Original:Recovery=1:1 |

Stage 1 的 5,000 步用于八任务与新增触觉模块适配，可通过 `--stage1-steps` 修改；它不是收敛承诺。两个 Stage 2 分别从同一个 Stage 1 checkpoint 和 π0.5 base 初始化，互不继承对方策略的权重，顺序使用八张 GPU，阶段进程结束后释放资源。没有增加新的训练阶段、成功率门槛或 early stopping。

本版为保持接口稳定，默认在整个 Stage 2 冻结 Updater 和 Contact Predictor。可在配置中将 `stage2_freeze_belief`、`stage2_freeze_contact_predictor` 设为 false，作为新 run 的训练选择；不在运行中动态重建 DDP 参数。冻结权重不等于冻结 belief 状态：Updater 仍按序列递归，WM/Predictor 保留对输入的梯度。Stage 1 没训练的 belief→queries/expert 投影仍在 Stage 2 学习。

## 恢复动作监督

所有真实动作，包括人为偏移和故意顶住的动作，都用于世界模型目标与 belief 历史。策略监督另有逐动作 `policy_action_mask`：排除 `atom/tag=perturb` 的目标帧，以及采集日志中 `blocked_hold` 的物理步区间，保留正常接近、合理探测 push、退回、修正和插入。

mask 按下一帧关节目标对齐，覆盖 50 步 chunk 内每个位置。无可模仿目标的样本不稀释 flow loss。前 k 步存在被排除动作时，同时停用该窗口的 latent-action alignment 与策略分支未来视觉辅助监督，避免以另一条 loss 路径教策略制造错误。教师分支的未来视觉/触觉监督仍保留。阶段标签只用于监督 mask，不进入策略输入。

随机历史长度为 0–4，最后至多两次更新做截断反传。Stage 1 ranking 负例按真实 task ID 构造，不把不同任务任意当作同一任务动作。推理仍用最终实际下发动作的前 k 步重新预测接触预期。

## 准备与 4090 debug

完整准备命令会复用有效缓存，只提取缺失的 AnyTouch2 特征：

```bash
bash /zhe_li/tactile-wam/contact-belief-wam/scripts/shared_train.sh prepare \
  --config /zhe_li/tactile-wam/contact-belief-wam/configs/shared_8xa800_bs64.yaml \
  --device cuda --batch-size 16 --threads 8
```

共享统计与清单已保存到 `data/stats/shared.all_train.quantiles.json` 和 `data/manifests/all_train/shared.jsonl`。数据检查和完成记录见 `IMPLEMENTATION.md`，无需每次云端作业重新准备。

```bash
bash /zhe_li/tactile-wam/contact-belief-wam/scripts/shared_train.sh debug \
  --config /zhe_li/tactile-wam/contact-belief-wam/configs/shared_8xa800_bs64.yaml \
  --output /zhe_li/tactile-wam/contact-belief-wam/outputs/debug_shared_4090 \
  --device cuda --optimizer-device cpu --updates 2 --threads 8
```

debug 运行完整模型，Stage 1 首个 batch 包含八任务原始样本与两任务恢复样本；两个 Stage 2 各检查正常与恢复样本。前反向使用 4090，AdamW 参数副本和状态在 CPU，以避免 24GB 显存被优化器状态占满。没有额外冻结语言模型、缩小网络或使用 LoRA。A800 正式训练使用原来的 GPU AdamW，不使用此 debug optimizer。记录模型、优化器、调度器、RNG 存取，以及固定噪声动作/接触预测的一致性；debug checkpoint 只用于该 debug 的阶段交接，不供正式训练初始化。

已有 debug 使用其 `debug_run.yaml` 固定输入，后续修改正式数据配置不会重跑已完成更新。最终全量训练清单与统计单独核查。

结果在 `outputs/debug_shared_4090/debug_summary.json` 与每个模型目录的 `debug_report.json`、`debug.log`。重复同一 debug 命令会跳过已完成模型，不追加更新；未完成的报告需先检查错误。

## 百度云直接提交

三个模型顺序运行：共享 Stage 1 → `insert_hole` Stage 2 → `insert_HDMI` Stage 2。每卡 64，全局 512，GA=1，BF16。

已对照正在运行的 `wave0-shared-8task-mb32-002`：它的共享 Stage 1 使用 8 卡、每卡 32、全局 256；Stage 2 每任务 2 卡、每卡 32、全局 64，同波四任务并行。这里保留每卡 64、GA=1 和两个策略顺序训练的安排，所以每个阶段使用全部 8 卡，全局 batch 512。当前配置只在本地验证了命令解析，没有启动八卡正式训练。

现跑 WAVE-0 使用原始演示、相对关节动作及验证/early stopping；本项目继续使用全量原始与恢复数据、绝对关节动作和固定步数。它们尚不是数据与动作表示完全统一的对照实验。

正式配置 `gradient_checkpointing: false`，使用 GPU AdamW，不使用 CPU optimizer offload、调试小 batch 或额外冻结层。梯度检查点仅由 4090 debug 入口在自己的配置中开启。表中模块冻结是方法本身的训练安排，不是显存优化。

```bash
bash /zhe_li/tactile-wam/contact-belief-wam/scripts/shared_train.sh train \
  --config /zhe_li/tactile-wam/contact-belief-wam/configs/shared_8xa800_bs64.yaml \
  --num-gpus 8 --micro-batch 64 --global-batch 512 \
  --stage1-steps 5000 --stage2-steps 10000 --stage2-original-steps 1000 \
  --resume none
```

断点续训：

```bash
bash /zhe_li/tactile-wam/contact-belief-wam/scripts/shared_train.sh train \
  --config /zhe_li/tactile-wam/contact-belief-wam/runs/shared-wm8-k10-8xa800-bs512/pipeline_config.yaml \
  --resume auto
```

会跳过完成的 Stage 1 和已完成的任务策略，从当前模型最新完整 checkpoint 恢复 optimizer、scheduler、随机数与采样位置。Original-only 到混合数据的切换以绝对训练步数计算，续训不重做正常演示预热。

```text
runs/shared-wm8-k10-8xa800-bs512/
  pipeline_config.yaml
  pipeline_state.json
  shared_wm/stage1/checkpoints/step_00005000/
  shared_wm/logs/metrics.jsonl
  tasks/insert_hole/stage2/checkpoints/
  tasks/insert_hole/export/final/
  tasks/insert_HDMI/stage2/checkpoints/
  tasks/insert_HDMI/export/final/
```

`val_every: 0` 关闭验证集构建和阶段末验证。训练日志记录视觉初始/修正损失、触觉预测损失，按任务和 Original/Recovery 分组；每 1,000 步及阶段末保存。Stage 2 还记录 flow、latent alignment、belief 范数/更新幅度和恢复采样开关。阶段仍按固定步数切换，模型效果在训练后的仿真闭环中评估。每任务 `export/final` 可直接用于现有 serve/eval 接口。


性能诊断更新：训练优先读取 NPZ 配套的 `.tokens.npy`/`.tokens.json`，按帧取值后转 float32；NPZ 仍作为源文件保留。`prepare` 会创建这些配套缓存。图像按序列内唯一帧号解码；Stage 1 跳过不使用的语言 tokenization；分组指标一次传回 CPU。以上不改变 batch、序列采样、模型和损失。

日志新增 `timing/rank0/*_host_seconds`（取数据等待、预处理、指标归约）以及 `*_cuda_seconds`（前向、反向、optimizer），均为该日志区间内每个 micro-batch 的均值。CUDA event 只在日志边界汇总；计时含流上的通信/空隙，不能直接当作纯 kernel 时间。各项可能重叠，也不涵盖所有阶段，不能相加当作完整 step 时间。吞吐继续看 `sequences_per_second`。

共享文件更新不会热替换已运行的 Python 模块，需重新启动 worker。没有完整 checkpoint 时重启从第 0 步开始；建议新的 output 区分前后日志。实际 A800 提速需新作业测量，CPU 缓存计时不能替代端到端吞吐。详见 `reports/stage1_performance_audit.json`。
