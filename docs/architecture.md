# 架构与数据约定

当前训练使用 [WAVE-0 对齐配置](aligned_training.md)。下文的绝对动作和冻结视觉塔约定属于旧入口；对齐入口采用 chunk 起点相对的前 7 维动作、绝对夹爪、32 维 flow 监督，并在 Stage 2 训练策略视觉塔，保留独立冻结的未来视觉目标塔。Contact Belief 和触觉预测的信息流保持不变。策略统计与共享 WM 统计分别复用参考实现，已执行动作以此前区间的起点为锚点；仿真接口仍接收 8 维绝对目标。

## 模型

策略主干沿用官方 PyTorch WAVE-0 / π0.5。WAVE-0 世界模型维持 width 1024、12 层、16 heads、FFN 4096；接触扩展使用 width 512、8 heads、FFN 2048 的两层 decoder。belief 状态由 4 个 512 维 token 表示。冻结 AnyTouch2 读入左右触觉相机各 4 帧的历史窗口，经原始投影特征池化为每侧 4 个 token，总计 8×512；cache 保留原始 step 和编码元数据以检查帧对齐。

`full` 模式的五条 belief 通路均开启：belief 调整 latent-action queries、世界模型 state、未来视觉 corrector、接触预测器和动作 Expert 条件。其余模式用于消融：`pi05` 不使用世界模型或 belief；`wave0` 使用世界模型而不使用 belief；`direct_touch` 只将触觉 belief 接入动作 Expert。正式全量训练以 `full` 为默认。

`k` 在执行、query 数和未来监督间隔中共用一个值。默认 `k=10`，也可用 `k=5`；策略 action chunk 长度为 50。belief 历史随机预热 0–4 步，反向梯度只保留末尾 1–2 步（默认 BPTT=2），避免把整段 episode 展开成计算图。

## 权重和输入适配

WAVE-0 的状态和动作输入内部按 32 维 padding 设计；本任务执行端是 8 维。不能按某个 14→8 的观察形状直接切片重置投影层。初始化应保留形状匹配的官方张量，只对确实改变的层显式初始化并记录加载清单。相机输入使用 UniVTAC `cam_high` 对应切片接入 WAVE-0 的高位相机特征。

官方 WAVE-0 世界模型严格加载已在本地实际通过：167 个张量加载、3 个形状适配转换、0 个重初始化、0 个未使用，记录在 `reports/official_world_load.json`。Stage 1 和 Stage 2 的有界真实 batch debug、checkpoint 保存和恢复都已完成。π0.5 base 复用本地已有的官方未微调 JAX `pi05_base`，经官方转换器转成 PyTorch；唯一重新初始化的是策略不使用的词表输出头。不会另行下载 PyTorch/Hugging Face π0.5 权重；该 base 不是旧任务微调策略。

## 数据时间轴

原始 UniVTAC HDF5 的 `embodiment/joint` 有 9 列，前 8 列用于本任务。当前观测状态取当前帧，动作监督取下一帧的 8 维**绝对关节目标**，不是关节增量。数据帧按 `step` 的连续性分段，未来目标不得跨越 episode 或断段。`k` 的执行/query/未来目标时间偏移保持一致。

每个任务单独用训练 split 汇总动作和状态的 `q01/q99` 统计并保存为 JSON。归一化为 `[-1,1]` 的线性缩放，超出分位数的值不裁剪。最终清单中每个任务有 100 条原始演示和 50 条含实测受阻、最终成功的恢复 episode，按 1:1 抽样；原始和恢复来源分别做 90/10 与 45/5 训练/验证划分。v1–v4 诊断数据、失败轨迹及显式排除项不进入训练。最终数量、帧统计和缓存核对见 [`../IMPLEMENTATION.md`](../IMPLEMENTATION.md)，采集控制器与扰动分布见 [`recovery.md`](recovery.md)。

AnyTouch2 输入从 HDF5 左右 `rgb_marker` 序列读取，按同一 episode 和分段构造历史窗口。冻结编码器已为全部清单 episode 生成 `[N,8,512]` `float16` cache；所有值有限，缓存帧数和 `raw_step` 均与 HDF5 一致。两任务合计 76,958 帧。缓存只包含 token、原始 step、帧索引和编码器元数据；训练仍读取完整 HDF5 相机、状态和动作。

## 训练阶段与记录

默认 `full` 模式以已转换的 pi05 base 和已加载的 WAVE-0 世界模块初始化。Stage 1 训练 2,000 步，更新世界模型及动作/状态条件编码器、Contact Belief、Future Corrector 和 Contact Predictor；视觉骨干与动作接口投影保持冻结。若无可用世界模块初始化，Stage 1 步数自动改为 20,000。Stage 2 训练 10,000 步，冻结视觉塔、视觉投影和 WAVE-0 世界模块，更新策略语言/动作部分、latent-action queries、future resampler 与 Contact Belief 扩展。AnyTouch2 始终冻结。`pi05` 与 `direct_touch` 没有 Stage 1 世界模型目标，会跳过该阶段。固定验证集每 500 步记录未来视觉和触觉损失，每 1,000 步保存 checkpoint；这些指标供 A800 训练时观察，不设通过门槛。

真实 batch 前向/反向、梯度和 checkpoint 恢复已在开发机完成；正式训练尚未开始，云端训练曲线和任务成功率要在用户手动提交 A800 作业后评估。启动与续训命令见 [`cloud.md`](cloud.md)。
