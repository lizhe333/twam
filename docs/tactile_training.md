# 双视角两阶段训练

2026-10-30 归档说明：未保存 checkpoint、已被 `head-wrist-stage2-8gpu-003` 替代的 `head-wrist-stage2-8gpu-002` 日志已清理；配置与源码快照保留在工作区 `reports/artifact-cleanup-20261030/retained-inputs/twam/runs/`。本页涉及 `002` 的路径表示历史重启过程。

默认入口是 `bash scripts/train.sh`，配置为 `configs/aligned_4xa800.yaml`，两个阶段均占用四卡。`independent_tactile: true` 表示触觉预测使用自己的网络，不表示额外增加一个训练阶段。

## 模块和训练范围

| 模块 | Stage 1：共享接触预测与 Belief | Stage 2：各任务策略 |
| --- | --- | --- |
| 已训练的 head＋wrist Visual WM、world action/state encoders | 冻结 | 冻结 |
| WM 视觉目标编码器、AnyTouch2 | 冻结 | 冻结 |
| 独立 Tactile WM | 训练 | 冻结 |
| Belief Updater、Future Corrector | 训练 | 继续训练 |
| belief→world state 投影 | 训练 | 训练 |
| belief→VLM queries / Action Expert 接口 | 不参与 | 训练 |
| π0.5 策略主干、Action Expert、策略视觉塔 | 不参与 | 训练 |

Stage 1 使用未来触觉、修正后未来视觉和原有视觉预测目标。Visual WM 权重冻结，但 belief→state 输入仍可通过它获得梯度。Tactile WM 读取当前 head/wrist 冻结视觉特征、当前触觉、状态及真实动作前缀；两路视觉有独立的相机标识。它不读取 belief 或预测的未来视觉。

Stage 2 保留动作损失、未来视觉和对齐辅助损失。动作梯度经过 belief→Action Expert 和 Corrector 路径更新 Updater，不需要 blocked/normal 分类标签。冻结独立 Tactile WM 后，触觉预测误差仍记录，但不会直接训练 Updater。

Stage 1 最多 5,000 步，学习率 1e-4；Stage 2 每任务 10,000 步。Stage 2 策略和 Updater 的峰值学习率均为 2.5e-5，其他新接口和 Corrector 为 1e-4。Updater 有独立的 `belief` 优化器分组，参与 EMA 和 checkpoint 恢复。

## 初始化和数据

默认复用：

```text
/zhe_li/tactile-wam/outputs/wave0-shared-8task-head-wrist-001/shared_wm/stage1/checkpoints/step_00005000/pretrained_model
```

通过 `world_init` 加载这份纯 WAVE-0 WM。它不是 Contact Stage-1 checkpoint。冻结复用要求相机顺序 `[head, wrist]`、预测跨度、模型尺寸和共享归一化一致，拒绝截取相机、补随机参数或重新初始化不匹配的世界模块。π0.5 策略及冻结视觉目标编码器仍使用该 WM 配套的 `pi05_base`。

Stage 1：720 条八任务 Original＋100 条 Hole/HDMI Recovery；80 条 Original 验证。每批任务均衡，两个插入任务内部 Original:Recovery=1:1。Stage 2：每个任务只读取自己的 100 条 Original＋50 条 Recovery，两个任务从同一份 Stage 1 独立初始化。`policy_tasks` 可以改为八任务的任意非空子集；没有 Recovery 的任务使用 Original。

扰动和刻意持续顶住的动作保留为动力学输入和历史，只屏蔽相应动作的模仿监督。读取两路图像时使用相同的当前、历史和未来帧索引，缺少 wrist 会报错。AnyTouch2 缓存无需因双视角重算。

两个阶段都使用 k=10、chunk=50；动作前七维相对 chunk 起点，夹爪绝对值，策略和共享 WM 使用各自归一化。在线服务要求 `image`（head）和 `wrist_image`（wrist），仿真适配器根据 checkpoint 元数据自动发送两路图像。最终动作经过边界处理，再用于计算下一轮预期触觉。

## 运行

在仓库根目录，复用现有环境：

```bash
export CBW_PYTHON=/zhe_li/tactile-wam/contact-belief-wam/.venv/bin/python
export CBWAM_ROOT="$PWD"
# 复用旧环境时，正式训练的源码指纹检查仍需要这个目录。
# 旧工作区与本仓库使用同一固定版本的 Transformers 源码。
if [ ! -e vendor/transformers ]; then
  ln -sT /zhe_li/tactile-wam/contact-belief-wam/vendor/transformers vendor/transformers
fi
# 可选：首次生成配置，把派生数据和训练输出放入本 checkout；已有配置需按新配方更新。
"$CBW_PYTHON" scripts/configure_workspace.py --workspace /zhe_li/tactile-wam

bash scripts/train.sh prepare --config configs/workspace.local.yaml
bash scripts/train.sh train --config configs/workspace.local.yaml --variant full --plan

# 有可用训练 GPU 时执行；可以在共享接触模块完成后停下检查验证结果。
bash scripts/train.sh train --config configs/workspace.local.yaml --variant full --stop-after stage1
bash scripts/train.sh train --config configs/workspace.local.yaml --variant full --resume auto

# 同一个冻结的双视角 WM、相同数据及动作监督屏蔽的 WAVE-0 对照。
bash scripts/train.sh train --config configs/workspace.local.yaml --variant wave0_recovery
```

默认 Stage 1 使用四卡，每卡 32、梯度累积 2 次，有效 batch 保持 256；两个 Stage 2 各使用两卡并行，每卡 8、累积 4 次，每任务有效 batch 为 64。四卡备选配方设置 `CUDA_VISIBLE_DEVICES=0,1,2,3`；其 Stage 2 每卡优化器状态比八卡配方多一倍，显存仍需实测。步数、学习率与训练数据不变；实际耗时取决于吞吐。

八卡配置 `configs/aligned_8xa800.yaml` 的 Stage 1 使用八卡、每卡 32；Stage 2 两个任务各用四卡并行、每卡 8、累积 2 次，每任务有效 batch 仍为 64。之前未分片的 Stage 2 在每卡 32 和 16 时均触发 OOM。现在恢复成功 WAVE-0 训练使用的 Adam 状态分片（`shard_optimizer_state: true`），EMA 保留在 GPU（`ema_device: cuda`），Belief 仍对两个历史时刻反向传播。新配置峰值显存需在云端实测；每个任务的 `memory.rankNN.jsonl` 会记录各卡初始化、前两次更新的阶段显存和后续峰值。

分片只改变 Adam 状态的存放位置，不改变各参数组学习率。每张卡单独保存优化器分片，全部保存并校验后才发布完整 checkpoint；恢复时校验卡数、分片哈希和参数对应关系，避免保存/恢复时在 rank 0 聚合完整优化器而再次 OOM。

对照组没有接触扩展可训练，直接复用 `world_init` 后训练策略，不重新训练视觉 WM，也不加载完整方法的接触模块。增加策略任务时最多并行两个任务，并等待对应 GPU 空闲。

默认训练输出使用新的 `contact-head-wrist-two-stage-4gpu` 目录。已有本地配置不会自动覆盖，需将 Stage 1 的 `num_gpus` 改为 4，并保留 `micro_batch: 32`、`global_batch: 256`。旧八卡 run、三阶段或单视角 run 不能作为这份配方的精确续训；旧 checkpoint 仍可按自身配置读取。低层旧 `tactile` 模型/权重读取接口保留兼容，统一入口不再调度该阶段。

`vendor/transformers` 被 Git 忽略，单独同步仓库不会带上它。若云端没有旧源码目录，在创建上述链接前使用 `"$CBW_PYTHON" scripts/prepare_transformers.py` 下载并校验固定版本；离线时传入 `--archive /path/to/source.tar.gz`。`--plan` 仅检查分卡，不会验证这项正式训练依赖。若因缺少源码而退出，补齐后保留原输出目录并使用 `--resume auto` 重试。

### 已完成 Stage 1，调整 Stage 2 后继续

用 `--stage1-checkpoint` 指向已完成 Stage 1 选出的 checkpoint，并使用新输出目录。入口会验证权重哈希、双视角/latent 配置、共享归一化和配套视觉编码器，跳过 Stage 1；两个策略仍独立初始化。原 checkpoint 与训练身份记录保留不变。后续续训保持相同配置、源码、输出和 checkpoint 参数，使用 `--resume auto`。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash scripts/train.sh train \
  --config configs/aligned_8xa800.yaml \
  --variant full \
  --stage1-checkpoint /zhe_li/tactile-wam/twam/runs/head-wrist-two-stage-4gpu-001/full/shared_wm/stage1/checkpoints/step_00004500 \
  --output /zhe_li/tactile-wam/twam/runs/head-wrist-stage2-8gpu-002 \
  --resume auto
```

## 验证

`<output>/full/shared_wm/validation.jsonl` 记录各任务的触觉预测、视觉修正及原始视觉指标。默认按 `contact_total = tactile_future + visual_corrected` 的任务平均值选择 Stage 1 checkpoint，而不是仅看冻结视觉基座。每 500 步验证，每任务 128 个窗口，最早 2,000 步后按 patience=4 早停。

`tactile_persistence` 与 `tactile_gain` 检查预测是否优于直接保持当前触觉。Stage 2 使用全部 Original 后，Stage 1 留出集不能作为策略独立验证集；最终比较需使用独立仿真 seed、相同相机、数据和执行预算。

CPU 缩小模型测试检查冻结、梯度、双视角、阶段交接和推理；正式模型显存、多卡吞吐与闭环成功率仍需在训练/仿真 GPU 上验证。
