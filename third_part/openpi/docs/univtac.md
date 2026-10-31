# π0.5 × UniVTAC

已经跑通的 UniVTAC 适配直接集成在 `openpi` 内部。模型仍为官方 JAX/NNX π0.5，使用原来的 AdamW、FSDP、EMA 和 Orbax checkpoint。单双视角通过 YAML 选择，共用实现。

本次迁移没有修改 `Pi0`、Gemma、SigLIP、flow matching、attention mask 或模型参数树；原有阅读注释保留。官方 `scripts/train.py` 和 `scripts/serve_policy.py` 只增加 `--univtac` 分发入口。

## 代码在哪里

| 职责 | 现在的位置（相对 `src/openpi/`） | 复用来源 |
| --- | --- | --- |
| 实验配置、`UniVTACData` | `training/univtac_config.py` | 原 `config.py`，相机和卡数差异改为配置 |
| 图像、状态、动作输入输出变换 | `policies/univtac_policy.py`、`univtac_images.py` | 已跑通的 head-only 版本，兼容双视角 |
| HDF5 → LeRobot 转换及片段处理 | `training/univtac_data.py` | 原 `data.py` |
| 归一化统计 | `training/univtac_statistics.py` | 原 `statistics.py` |
| base 权重加载 | `training/univtac_weights.py` | 原 `StrictBaseLoader`，继续检查参数名及形状 |
| 训练循环、更新和日志 | `training/univtac_trainer.py` | 原训练循环；复用官方 `init_train_state()` |
| 任务队列 | `training/univtac_queue.py` | 原顺序队列与恢复逻辑 |
| 推理与 websocket 服务 | `serving/univtac.py` | 原 `serving.py` |
| checkpoint 相机信息 | `policies/univtac_camera.py` | 原 `camera_contract.py` |
| 仿真输入、动作缓存和执行约定 | `policies/univtac_sim.py` | 原 `sim_policy.py` |
| 工作区路径、任务常量 | `shared/univtac_runtime.py`、`univtac_constants.py` | 原 `paths.py`、`constants.py` |

调用顺序：

```text
scripts/train.py --univtac train
  → training/univtac_cli.py
  → training/univtac_config.make_config()
  → training/univtac_trainer.run()
  → 官方 init_train_state()、data_loader
  → 原 UniVTAC train_step()：loss → 梯度 → AdamW → EMA
  → 官方 checkpoints.save_state() / restore_state()
```

模型端不再依赖外部 `pi05_univtac` 包或不同 variant 目录。旧适配目录保留为历史实现，原有评测和分析脚本继续可用。

## 配置

| `configs/univtac/` 中的文件 | 相机 | 每任务 FSDP 卡数 |
| --- | --- | --- |
| `head_only.yaml` | head | 2 |
| `head_wrist.yaml` | head、wrist | 2 |
| `head_wrist_4gpu.yaml` | head、wrist | 4 |
| `legacy_paper.yaml` | 按旧 head-only 版本选择 | 2 |

`legacy_paper.yaml` 保留旧版本的相机例外：`insert_tube`、`lift_bottle` 使用 head+wrist，其他六任务使用 head。它复现的是旧 head-only 适配版本的选择逻辑；历史保留的这两个任务原四卡训练若要续训，应使用 `head_wrist_4gpu.yaml`。

正式默认配置沿用已跑通代码：batch 64、10,000 步、seed 42、EMA 0.99、AdamW、梯度裁剪 1.0；学习率 warmup 1,000 步，峰值 `2.5e-5`，在第 10,000 步衰减至 `2.5e-6`。旧 README 的 batch 32 描述已经过时，这里以现有训练代码及 checkpoint 配方为准。

配置可选择相机、FSDP 卡数、batch、训练步数、worker 数、seed 和 EMA decay。已有 LoRA/overfit 调试模式仍使用原来的局部配置，不改变正式全参数微调。

## 复用现有环境和数据

```bash
cd /zhe_li/tactile-wam/openpi
export TWAM_ROOT=/zhe_li/tactile-wam
export OPENPI_ROOT="$TWAM_ROOT/openpi"
export PI05_PYTHON="$TWAM_ROOT/.venvs/openpi/bin/python"
export PYTHONPATH="$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
```

继续使用现有 `.venvs/openpi` 环境（JAX 0.5.3、Flax 0.10.2、Orbax 0.11.13），本次未重新安装环境。可选依赖组 `univtac` 声明数据准备使用的 h5py 和 PyYAML。

数据和资产继续使用工作区原路径：

- 原始 HDF5：`UniVTAC/data/isaac45/<task>/hdf5/`。
- 已转换 LeRobot 数据：`datasets/pi05_univtac/<split>/<task>/`。
- 既有统计：`assets/pi05_univtac/<split>/<task>/`。
- 官方 base：`models/pi05_base/`。
- checkpoint：`outputs/checkpoints/pi05_univtac_<task>_<mode>/<run>/`。

训练直接复用已转换的 LeRobot 数据，无需重新转换或计算统计。需要准备新数据时，入口是 `python -m openpi.training.univtac_data scan|convert|validate`；统计入口是 `python -m openpi.training.univtac_statistics --task TASK --split full`。

保留的数据语义：当前 8 维 joint 为 state，下一帧 joint 为 action；chunk 50、执行 10 步。前 7 维动作相对同一个 chunk 起点，夹爪保持绝对米制位置。State 和 Action 各自使用已有 q01/q99；8 维状态先生成语言 token，再补齐至 32 维。数据尾部重复最后一个有效目标，`lift_can/62` 在帧 228 的分段处理保持原样。RGB 保留原解码顺序，继续使用官方 resize-with-pad。

## 训练入口

先查看单任务或八任务计划，不加载模型、不启动训练：

```bash
bash scripts/train/univtac/train.sh plan \
  --config configs/univtac/head_wrist.yaml \
  --task insert_hole --run my-dual-run
```

实际训练将 `plan` 改为 `train`。双卡配置需让该进程看到两张 GPU；四卡配置需看到四张。单任务传 `--task insert_hole`，顺序训练八任务传 `--task all`。每个任务仍在新进程中从 base 初始化。恢复已有运行添加 `--resume`，使用原任务、相机、卡数及 run 名称。

八卡并行调度沿用原四组双卡队列：

```bash
"$PI05_PYTHON" scripts/train/univtac/train_parallel.py \
  --config configs/univtac/head_wrist.yaml --run my-dual-parallel
```

默认展示计划，添加 `--launch` 才执行。每组一个 JAX 进程，四任务并行、每组顺序处理两个任务；无需 torchrun。输出 run 名称仍按 `RUN-pair0` 至 `RUN-pair3` 组织，支持 `--resume`。

## 已有权重、EMA 与恢复

```python
from openpi.serving.univtac import TaskPolicy

policy = TaskPolicy(
    ("insert_hole",),
    checkpoint="/path/to/run/10000",
    mode="full",
)
result = policy.infer(observation)
# result["actions"] 为 [50, 8] 的绝对关节目标。
```

checkpoint 的目录结构、参数键和树形结构保留。官方 Orbax 保存逻辑仍将推理参数写入 `params/`：启用 EMA 时这里保存 EMA，未启用时保存当前参数；训练状态仍在 `train_state/`。推理直接加载这份导出的参数。

推理优先读取 checkpoint 自带的 `camera_contract.json`；没有相机记录的旧权重沿用原来的双视角回退。相机、任务和参数形状等必要的功能检查保留。

按要求，新入口不生成或检查哈希、源码指纹、参数字节摘要，也不要求旧 `state-N.json` 或源码来源记录。新训练仅保存可读的 `univtac_config.json` 和相机信息。恢复直接使用 Orbax 恢复参数、优化器、step 和 EMA。原上游不保存 DataLoader 游标，因此仍不承诺中断续训的数据顺序与不间断训练逐位相同。

## 服务和仿真

```bash
"$PI05_PYTHON" scripts/serve_policy.py --univtac \
  --task insert_hole --model-mode full --checkpoint /path/to/run/10000

"$PI05_PYTHON" scripts/eval/univtac/eval.py --list
```

`--list` 只列出已有 checkpoint。评测脚本复用原来的固定种子、动作预算、仿真进程和结果协议，并同时识别旧 `run_identity.json` 的任务/相机字段和新配置文件；不检查其中的指纹。

`UniVTAC/policy/Pi05Wam/deploy_policy.py` 增加 `OPENPI_ROOT` 路径支持，新评测入口自动设置它并启动本仓库模型服务。仿真环境启动和兼容性工具仍复用工作区的 `pi05-univtac/scripts/`；没有重装或改写 Isaac Sim 环境。

## 验证范围

简短结果见 [univtac-validation.json](univtac-validation.json)。验证包括真实 LeRobot 样本和变换、旧配置对齐、完整生产 checkpoint 加载、缩小的真实 π0.5/Gemma/SigLIP 训练更新、EMA 和 Orbax 恢复。

本次没有重跑完整训练、生产规模多卡续训或仿真成功率评测。迁移验证脚本、临时数据、checkpoint 和日志在完成后清理；保留正式代码、配置和这份简短记录。
