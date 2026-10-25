# 独立触觉 WM 与统一训练入口

当前入口是 `python -m contact_belief_wam train`，或 `bash scripts/train.sh`。在仓库根目录执行，使用项目 Python 环境。`scripts/aligned_train.sh` 仅保留为转发命令。

## 从哪里读代码

| 文件 | 职责 |
| --- | --- |
| `src/contact_belief_wam/train.py` | 配置、阶段顺序、GPU 进程、checkpoint 交接与续训 |
| `src/contact_belief_wam/trainer.py` | 各阶段共用的训练循环、验证、优化器和保存 |
| `src/contact_belief_wam/data.py` | HDF5/AnyTouch2 缓存、相对动作、归一化和采样 |
| `src/contact_belief_wam/serve.py` | 推理服务、历史 belief 和预测时间戳 |
| `src/contact_belief_wam/modules.py` | `TactileWorldModel`、Updater 和 Corrector |
| `src/contact_belief_wam/model.py` | 各模块的前向连接及冻结范围 |
| `src/contact_belief_wam/legacy/` | 旧绝对动作数据与训练代码，仅供旧实验使用 |

`aligned_data.py / aligned_trainer.py / aligned_pipeline.py / aligned_serve.py` 已移除；导入分别改为 `data / trainer / train / serve`。

## 阶段和目标

默认配置 `independent_tactile: true` 的完整方法依次执行：

1. **tactile**：只训练 `contact.predictor`，即独立 `TactileWorldModel`。读取当前冻结视觉特征、当前触觉 latent、状态和未来 k 步真实轨迹动作。使用自己的状态/动作投影和两层 Transformer Decoder，预测 t+k 的 8×512 触觉 latent。目标由冻结 AnyTouch2 编码得到；损失为 cosine distance + 0.5×SmoothL1。这个阶段不调用 Visual WM、Future Corrector 或 Belief Updater。
2. **stage1**：从 tactile checkpoint 严格加载 Predictor 并冻结，训练原有视觉 WM、状态/动作编码、Belief Updater 与 Future Corrector。独立触觉损失仍记录，但不加入该阶段的优化目标。只传递 Predictor 权重，不传递 tactile checkpoint 中未训练的 belief/corrector 权重。
3. **stage2**：加载 Stage1 checkpoint，为两个插入任务分别训练策略；冻结独立 Predictor、视觉 WM 和配置指定的 Updater，更新策略、接口投影与 Future Corrector。

这次改动**没有实现 blocked/normal 标签监督的 Belief 专项训练**。当前 Stage1 的 belief 仍由视觉预测目标间接训练。若采用专项诊断方案，需要另行定义标签、损失和验证集，不能把现有 Stage1 等同于该阶段。

触觉动作采用 shared-WM 归一化：前七维相对 chunk 起点，夹爪为绝对位置。后续训练和在线推理使用同一约定。在线触觉预测使用生成并经过边界处理的最终动作前 k 步；与下一次同时间戳的真实触觉比较。独立网络不读取 belief 或预测的未来视觉，因此视觉 WM 后续更新不会改变它的输入编码器。独立模型不允许在加载时修改 k；不同预测时域需要相应训练和验证。

旧 checkpoint 未设置该开关时，仍使用旧 `ContactPredictor` 结构。旧耦合预测器不能用于独立触觉阶段的权重交接，也不能续训新的配置。`wave0_recovery` 自动关闭该开关，跳过 tactile 阶段，保留 WAVE-0 基线原有训练目标与计算路径。完整方法增加了触觉训练预算，实验中应单独报告这一开销。

## 运行

先按 README 生成本地路径配置，再运行：

```bash
# 查看全部阶段和 GPU 分配，不启动训练
bash scripts/train.sh --config configs/workspace.local.yaml --variant full --plan

# 仅完成触觉训练和验证；只需配置 tactile.num_gpus 对应数量的 GPU
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train.sh \
  --config configs/workspace.local.yaml --variant full \
  --output /absolute/new/run --stop-after tactile

# 检查触觉指标后，用八卡继续 Stage1 和两个策略任务
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train.sh \
  --config configs/workspace.local.yaml --variant full \
  --output /absolute/new/run --resume auto

# WAVE-0 对照使用独立输出目录
bash scripts/train.sh --config configs/workspace.local.yaml \
  --variant wave0_recovery --output /absolute/new/baseline
```

也可以一次运行完整流程，不指定 `--stop-after`。`--stop-after stage1` 会在策略训练前停下。`--resume auto` 复用已完成且来源一致的阶段，或恢复未完成阶段的优化器、随机状态和采样位置。

## 触觉验证

`<output>/full/tactile_wm/validation.jsonl` 记录：

- `tactile_future`：正确动作条件下的预测误差，也是触觉 checkpoint 选择和早停指标。
- `tactile_persistence`：直接把当前触觉当成未来触觉的误差。
- `tactile_gain`：persistence 误差减去预测误差，正值说明优于该基线。
- `tactile_wrong_action` / `tactile_action_gap`：同任务错误动作条件下的误差及其相对正确动作的差值；无有效负例的样本不计入。该指标检查动作敏感性，不是反事实轨迹的真实监督。

Stage1 继续按视觉 `world_total` 选择 checkpoint，独立触觉权重已经固定。现有验证集仍是原始演示留出集；受阻、退回、重新对齐片段上的泛化评测，需要另外划分未参与训练的恢复 episode。平均 loss 下降不能替代这些诊断或闭环成功率。

代码回归测试使用缩小的网络验证梯度、冻结、接口和权重交接；不代表正式配置已经收敛。本次代码修改不会启动正式 GPU 训练。
