# 架构与数据约定

当前默认配置为 `configs/aligned_8xa800.yaml`，启用 `independent_tactile: true`。训练入口及操作见 [独立触觉训练](tactile_training.md) 和 [WAVE-0 对齐配方](aligned_training.md)。

## 模块

`ContactWAM` 是 WAVE-0 与接触模块的组合类。原有 Visual WM 保持 width 1024、12 层、16 heads、FFN 4096；独立 Tactile WM 和接触模块使用 width 512、8 heads、FFN 2048 的两层 decoder。

- **Visual WM**：读取当前视觉、状态及动作 latent，预测未来视觉变化。
- **TactileWorldModel**：读取当前冻结视觉特征、当前触觉、状态及真实动作前缀，用自己的动作/状态投影预测未来触觉。它不读取 belief 或预测的未来视觉。
- **BeliefUpdater**：读取当前真实触觉、上一轮对当前时刻的预测、两者差异、历史 belief、状态及已执行动作，输出 4×512 belief token。
- **FutureCorrector**：用 belief 和动作 latent 修正未来视觉。
- **Action Expert**：读取视觉/语言上下文、未来视觉条件及 belief 条件生成动作。

默认独立版本中，belief 调整 latent-action queries、视觉 WM state、FutureCorrector 和 Action Expert 条件，不输入 Tactile WM。旧 checkpoint 缺少 independent_tactile 字段时继续使用原来的 ContactPredictor：其输入包含修正后的未来视觉与 belief。两种 Predictor 权重不互换。

冻结 AnyTouch2 读取左右传感器各 4 帧历史，去掉特殊 token 后在时间维求平均、空间池化到每侧 2×2，得到总共 8×512 的触觉 latent。未来触觉目标使用同一冻结编码器；当前模型预测 t+k 一个时刻，不是完整未来序列。

## 在线反馈

策略先根据上一轮预测与当前真实触觉更新 belief，再生成动作。触觉 WM 使用最终生成、边界处理并转换到 shared-WM 归一化空间的前 k 步动作重新计算预期触觉。服务保存 belief、预期触觉和 target_step；只有 target_step 与下一次观测时刻一致时，预期才参与更新。真实未来触觉不会作为部署输入。

默认 k=10，动作 chunk 长度 50。独立触觉模型加载时不允许改 k；不同预测时域需要相应训练和验证。后续 belief 学习仍使用随机历史预热 0–4 步，默认保留末尾两步梯度。两个阶段共用这一历史约定。

## 数据和初始化

UniVTAC joint 有 9 列，本任务使用前 8 列。轨迹下一帧关节位置作为动作监督代理；aligned 数据处理将前 7 维转换为相对 chunk 起点的位置，第 8 维夹爪保持绝对值。WAVE-0 内部补齐到 32 维，flow matching 监督 32 维；仿真执行端仍接收 8 维绝对目标。

策略与世界模型分别复用参考任务统计和八任务共享统计。历史执行动作使用此前区间起点作为锚点。数据按 episode 和原始 step 连续性分段，历史窗口与未来目标不得跨段。

Stage1 使用八任务共享训练集：720 条原始训练轨迹与 100 条插入恢复轨迹；验证保留 80 条原始轨迹。Stage2 每个插入任务使用 100 条原始和 50 条恢复轨迹。扰动/受阻保持区间的动作可用于动力学学习，但按采集记录屏蔽策略模仿监督。额外的恢复诊断验证应使用未参与训练的 episode，不能把当前训练恢复集当作泛化验证集。

π0.5 与冻结视觉目标编码器使用八任务 WM 配套的 base；Visual WM 通过 world_init 加载已训练的 head＋wrist 版本，两个阶段都冻结其权重及动作/状态编码器。冻结复用严格检查相机顺序、预测跨度、尺寸和共享归一化，不截取相机或重新初始化参数。两路当前/未来 RGB 贯穿数据、模型和推理服务。

## 训练边界

默认顺序为 stage1 → stage2。Stage1 训练独立 Predictor、Updater、Corrector 和参与的接口，按触觉误差＋修正后视觉误差选择 checkpoint。Stage2 冻结两个 WM，继续训练 Updater、Corrector、策略与接口；策略视觉塔参与训练，未来视觉监督塔独立冻结。

独立 Predictor 不读取 belief，因此触觉误差不直接训练 Updater。Updater 从视觉预测/修正损失与 Stage2 的动作损失学习；不要求 blocked/normal 标签或分类头。回归测试验证代码连接、梯度和恢复行为，不代表正式闭环效果。旧训练与调试记录见 IMPLEMENTATION.md，不能作为双视角两阶段版本的训练结果。
