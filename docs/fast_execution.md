# Contact-Belief 执行优化

对照同工作区 `wave0-univtac-fast` 后，仅迁移适用于 Contact-Belief 的执行优化。

| 项目 | 当前处理 |
|---|---|
| 两个空相机 | Contact 从一开始只输入 head，`empty_cameras=0`；无需删除相机 |
| SDPA | 已接入四处 WAVE-0 自定义 VLM/Expert attention；完整模式 Stage 2 和闭环推理使用。世界模型原本已用 SDPA |
| fused AdamW | 新配置在 CUDA 上启用，CPU debug 保留普通 AdamW；分组、超参不变 |
| GPU FP32 EMA | 当前 Contact 训练没有 EMA，不新增这项训练机制 |
| DDP unused 检测 | 保留；递归分支及消融开关不能照搬对方的全部参数有梯度假设 |
| 每 10 步吞吐与计时 | 已有，保留数据等待、预处理、GPU 流前反向/optimizer 和指标统计计时 |

SDPA 保留 GQA、原缩放系数、additive block mask 及输出布局。PyTorch 自动选择可用内核；启用 SDPA 不等于保证 FlashAttention，也不保证训练逐位相同。模型结构、参数名称、belief 历史长度、数据和损失均未改。

新配置 `configs/shared_8xa800_fast.yaml` 相比 `shared_8xa800_bs64.yaml` 仅新增 `attention_backend: sdpa`、`fused_adamw: true`，并更换输出目录。保留 8×A800、每卡64、全局512、GA=1、BF16；不开梯度检查点，不增加offload或额外冻结。

用户手动提交新任务时可运行：

```bash
bash /zhe_li/tactile-wam/contact-belief-wam/scripts/shared_train.sh train \
  --config /zhe_li/tactile-wam/contact-belief-wam/configs/shared_8xa800_fast.yaml \
  --resume auto
```

新的 `runs/shared-wm8-k10-8xa800-bs512-fast02` 不存在时从头开始，之后相同命令恢复该版本最新完整 checkpoint。此命令不会自动继承旧 `perf01` 的训练进度。旧配置/旧checkpoint缺少开关时继续 eager/普通AdamW；不支持通过修改开关冒充原运行的精确续训。没有修改或停止已有云端任务，也没有启动正式训练。

验证：111 passed、2 skipped。覆盖缩小尺寸的真实 Contact/WAVE 模块的 eager/SDPA loss与可训练梯度数值一致性、分块mask/GQA、固定噪声动作和接触预测、配置及模型checkpoint恢复、optimizer/scheduler恢复和新旧续训contract。完整三阶段命令已 dry-run。CUDA测试在本次环境中跳过，未运行全尺寸GPU训练或测量A800吞吐；CPU测试不构成云端提速结论。
