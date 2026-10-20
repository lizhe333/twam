# 三卡 A800 batch 性能 debug

这次百度云作业只比较 batch，不启动两阶段正式训练。默认测试完整 `full` 模式的 Stage 2：相同 π0.5 / RoboTwin 世界模型初始化、真实混合数据和触觉缓存、BF16、梯度检查点、belief 随机预热 0–4 次与最后两次截断反传。触觉新增模块尚未经过 Stage 1 训练，损失只用来完成真实计算，不能据此判断收敛或任务效果。

每卡 batch 16/32 分别使用全局 batch 48/96，两组均为 GA=1。各自在独立 DDP 进程中预热 3 次 optimizer update，再测量 10 次；不保存模型，不执行验证和仿真，结束后退出。模型加载、AdamW 首次状态分配和预热不计入测量窗口。每组默认最多 1,800 秒（含模型加载），可通过 `--case-timeout` 修改。

## 提交命令

百度云与开发机使用相同挂载路径：`/zhe_li/tactile-wam/contact-belief-wam`。下面直接引用项目内已有的 `data/`、`models/`，输出也保存在挂载目录内，不使用 `/workspace` 占位路径。无需生成上传包，也不下载 π0.5 权重。`--bootstrap-env true` 安装项目依赖；已有可用的项目 `.venv` 时可以改成 `false`。

```bash
CBW_PROJECT=/zhe_li/tactile-wam/contact-belief-wam
CBW_DATA_ROOT="$CBW_PROJECT/data"
CBW_PI05_BASE="$CBW_PROJECT/models/pi05_base_local_debug"
CBW_TOKENIZER="$CBW_PROJECT/models/tokenizer"
CBW_WORLD_INIT="$CBW_PROJECT/models/wave0_robotwin_world"
CBW_TASK=insert_hole
CBW_OUTPUT="$CBW_PROJECT/runs/$CBW_TASK/benchmark-3xa800-bs16-bs32"

bash "$CBW_PROJECT/scripts/cloud_benchmark.sh" \
  --task "$CBW_TASK" --mode full \
  --data-root "$CBW_DATA_ROOT" \
  --manifest "$CBW_DATA_ROOT/manifests/$CBW_TASK.jsonl" \
  --stats "$CBW_DATA_ROOT/stats/$CBW_TASK.quantiles.json" \
  --tactile-cache "$CBW_DATA_ROOT/cache/anytouch2" \
  --pi05-base "$CBW_PI05_BASE" \
  --tokenizer-path "$CBW_TOKENIZER" \
  --world-init "$CBW_WORLD_INIT" \
  --output "$CBW_OUTPUT" \
  --python python3.10 --bootstrap-env true --device cuda \
  --num-gpus 3 --micro-batches 16 32 \
  --benchmark-stage stage2 --warmup-steps 3 --measure-steps 10 \
  --precision bf16 --gradient-checkpointing true \
  --k 10 --data-mix 1:1 \
  --belief-warmup-min 0 --belief-warmup-max 4 --belief-bptt-steps 2 \
  --lr-new 1e-4 --lr-policy 2.5e-5 --grad-clip 1.0 \
  --workers 2 --seed 42 --case-timeout 1800 --resume none
```

五个 belief 接入开关默认全开。可以添加 `--dry-run` 只打印两个子命令；本地检查时同时使用 `--bootstrap-env false`，不会启动模型或参数更新。正式提交时不加 `--dry-run`。每次 sweep 使用新的输出目录，避免混合不同次测量。

## 看什么结果

输出根目录的 `comparison.md` 是汇总表，`comparison.json` 保存完整结果及按端到端吞吐选出的 `recommended_micro_batch`。不单凭 GPU 利用率选 batch：大 batch 即使利用率更高，也可能吞吐更低或显存余量过小。结果包含：

- 完整 belief 序列/秒、平均与中位 step 时间；按完整训练更新计，包含取数据、预处理、前反向、DDP 同步及 AdamW 更新。
- 每卡平均 GPU 利用率、采样显存和功耗，仅统计测量窗口；使用实际物理 UUID 匹配设备，支持容器中的 GPU 编号重映射。
- 各 rank 的 PyTorch 峰值 allocated/reserved 显存，以及三卡的最大值。
- 数据等待、预处理和训练计算时间，便于识别 GPU 在等数据的情况。

每个 `bs16/`、`bs32/` 子目录保存 `run_config.yaml`、`launch.log`、`steps.jsonl`、`gpu_samples.jsonl`、`devices.json` 和 `result.json`。失败时保留 `failure_rank*.json`；若 batch 32 OOM，batch 16 的结果仍会保留，汇总标记 `oom`。两组都失败时作业返回失败，已有诊断仍保留。

## 然后重新提交正式任务

性能 debug 不产出可续训模型，正式训练从已有 π0.5 / 世界模型初始化开始，使用新的输出目录。根据结果，在 [`cloud_train.sh`](../scripts/cloud_train.sh) 的完整正式命令中选择：

```text
每卡 16：--num-gpus 3 --global-batch 48 --micro-batch 16
每卡 32：--num-gpus 3 --global-batch 96 --micro-batch 32
```

两组都会自动算出 `accumulation_steps: 1`。调度步数、EMA 状态及训练范围不会由 benchmark 自动改变。完整两阶段训练参数见 [`cloud.md`](cloud.md)。本机只完成脚本与命令检查，真实 A800 性能以这次云端 debug 的结果为准。
