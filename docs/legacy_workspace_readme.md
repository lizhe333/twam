# Contact-Belief WAM

本项目在官方 WAVE-0 / π0.5 策略上加入触觉接触预期与 belief 反馈，用 UniVTAC 的插孔和 HDMI 插入任务研究受阻后的纠正动作。

## 开发与正式训练

开发机只用于代码开发、恢复数据采集、AnyTouch2 特征缓存，以及真实 batch 的 1–2 次更新调试和烟测。完整训练由用户在百度云 A800 环境手动启动；本仓库不把本地开发机当正式训练环境。

通过 wheel 在项目目录外调用工具时，设置 `CBWAM_ROOT=/path/to/contact-belief-wam` 指向项目根目录。根目录须包含 `pyproject.toml`、`src/contact_belief_wam/` 和 `scripts/cloud_train.sh`；源码 checkout、当前工作目录或 Python 环境旁的唯一项目目录会自动识别。

当前对齐版本使用 [`configs/aligned_8xa800.yaml`](configs/aligned_8xa800.yaml) 和独立入口 `scripts/aligned_train.sh`，具体配置、debug 和提交命令见 [`docs/aligned_training.md`](docs/aligned_training.md)，本轮实际检查记录见 [`docs/aligned_debug_report.md`](docs/aligned_debug_report.md)。历史实现记录见 [`IMPLEMENTATION.md`](IMPLEMENTATION.md)，采集控制器历史见 [`docs/recovery.md`](docs/recovery.md)。

## 设计约定

- 四种比较模式：`pi05`（π0.5 策略）、`wave0`（策略加 WAVE-0 世界模型）、`direct_touch`（触觉 belief 接入策略）、`full`（世界模型与触觉 belief 全部启用）。`full` 默认打开五条 belief 通路。
- 官方 WAVE-0 世界模型为 width 1024、12 层；新增 contact 模块为 width 512、8 heads、FFN 2048，belief 为 4×512。冻结的 AnyTouch2 对左右触觉视频分别编码，每侧 4 tokens，共 8×512。
- `k=10` 是默认统一的执行长度、latent-action query 数和未来目标间隔；支持 `k=5`。策略仍预测 50 步 action chunk。UniVTAC 原始 HDF5 有 9 维关节记录，模型取前 8 维。对齐版把未来关节目标的前 7 维减去 chunk 起点状态，夹爪保持绝对位置，并监督补齐后的 32 维；仿真收到的仍是还原后的 8 维绝对关节目标。
- 原始示范和恢复示范按 1:1 混合。每个任务已采集 50 条最终成功且至少一次实测受阻的恢复 episode；完整数据与缓存核对见 [`IMPLEMENTATION.md`](IMPLEMENTATION.md)。AnyTouch2 token cache 是触觉特征缓存，不能代替 HDF5 中的相机画面、状态或动作。

模型细节见 [`docs/architecture.md`](docs/architecture.md)，A800 启动与续训命令见 [`docs/cloud.md`](docs/cloud.md)。云端训练直接引用 A800 上的实际路径，不要求生成上传包或清单。

## 配置模板

当前对齐版以 `wave0-univtac-fast` 为参考：Stage 1 使用 8 卡、每卡 32、global batch 256；Stage 2 的两个插入任务分别使用 2 卡、每卡 32、每任务 global batch 64，同时执行，均为 GA=1。动作、图像、tokenizer、双统计归一化、优化器和学习率计划均按参考实现对齐。`full` 和 `wave0_recovery` 共用数据、统计与采样规则，分别训练自己的 Stage 1 teacher。

之前的 [`configs/shared_8xa800_bs64.yaml`](configs/shared_8xa800_bs64.yaml)、`scripts/shared_train.sh` 及下面单任务模板保留给旧实验。旧绝对动作 checkpoint 不用于续训新的对齐版本。

[`configs/train_insert_hole.yaml`](configs/train_insert_hole.yaml) 和 [`configs/train_insert_HDMI.yaml`](configs/train_insert_HDMI.yaml) 是 A800 run-config 模板。默认配置为 `full`、`k=10`、4 GPU、global batch 32、micro batch 2、bf16、Stage 1 自动选择 2k 初始化或 20k 从头训练、Stage 2 10k。配置里的路径应改为用户在 A800 上的实际路径；本地仿真 smoke 不代表 A800 训练已运行。

旧版单任务入口支持 1/2/3/4/8 GPU，其历史示例见 [`docs/cloud.md`](docs/cloud.md)。新的对齐版使用上面的固定分工与 [`docs/aligned_training.md`](docs/aligned_training.md) 命令；A800 显存和吞吐需在用户提交的作业中确认。

之前的三卡 A800 性能 debug 已完成：每卡 batch 16/32 的 Stage 2 吞吐分别为 17.99/22.63 序列每秒。用户选择正式训练每卡 64，不再追加 batch 性能测试；64 尚无 A800 实测。历史性能入口与记录见 [`docs/benchmark.md`](docs/benchmark.md)。

## 本地数据准备与有限调试

以下命令用于准备输入或做有界 smoke，不会启动正式训练。恢复采集需要可运行的 UniVTAC/Isaac 仿真；`--episodes` 表示成功 episode 目标数，当前进度见 [`IMPLEMENTATION.md`](IMPLEMENTATION.md)：

```bash
cd /zhe_li/tactile-wam/contact-belief-wam
./.venv/bin/cbwam prepare recovery \
  --task insert_hole \
  --data-root /zhe_li/tactile-wam/contact-belief-wam/data \
  --episodes 5 \
  --max-attempts 1000
```

确认原始 HDF5、恢复 HDF5 和 UniVTAC instructions 都存在后，再构建相对路径 manifest 和每任务训练 split 的 q01/q99 统计：

```bash
./.venv/bin/cbwam prepare manifest \
  --task both \
  --data-root /zhe_li/tactile-wam/contact-belief-wam/data \
  --original-root /zhe_li/tactile-wam/UniVTAC/data/isaac45 \
  --recovery-root /zhe_li/tactile-wam/contact-belief-wam/data/recovery
```

使用本机完整 AnyTouch2 源码和权重生成特征缓存。缓存按 episode 保存 tokens 与原始 step；训练仍读取 HDF5 的图像、关节状态、动作和原始触觉：

```bash
./.venv/bin/cbwam prepare cache \
  --task both \
  --data-root /zhe_li/tactile-wam/contact-belief-wam/data \
  --source /zhe_li/tactile-wam/AnyTouch2 \
  --device cpu \
  --batch-size 8 \
  --threads 8
```

真实 batch 的 debug 限定每阶段 1–2 次 optimizer update，不是训练任务。云端模板使用 `/workspace/...` 路径；先复制一份 run config 并将数据、cache、stats、π0.5 与 WAVE-0 路径改成本机实际位置，再运行：

```bash
cp configs/train_insert_hole.yaml /tmp/train_insert_hole.local-debug.yaml
# 编辑上面的副本，将路径字段改成本机实际文件。
./.venv/bin/cbwam debug \
  --run-config /tmp/train_insert_hole.local-debug.yaml \
  --output /zhe_li/tactile-wam/contact-belief-wam/reports/debug/insert_hole \
  --updates 1 \
  --stages stage1,stage2 \
  --device cpu \
  --threads 12
```

debug 会使用 manifest/cache 中的真实样本，检查前向、反向、参数更新和保存/恢复，并将结果写到指定目录的 `debug_report.json`。已核验的初始化权重来源与当前 debug 结果见 [`IMPLEMENTATION.md`](IMPLEMENTATION.md)；A800 启动、续训和输出路径设置见 [`docs/cloud.md`](docs/cloud.md)。

## 主要数据风险

插入动作位移较小，8 维绝对目标经过每任务训练集的 `q01/q99` 归一化后，可能出现有效动作分辨率不足。若视频显示末端插入抖动，应优先检查原始动作、反归一化结果和量化后的细小位移。旧 JAX 实验使用不同动作空间，不作为直接对比基线。
