# 百度云训练与结果取回

当前采用八任务共享 WM、两个独立插入策略的新入口，使用 8×A800、每卡 64、全局 512、GA=1，直接提交命令见 [`shared_training.md`](shared_training.md)。本文下方的 `cloud_train.sh` 是保留的单任务两阶段入口，不能代替共享训练。

正式训练由用户在百度云 A800 上手动启动。开发机用于代码开发、恢复数据采集、AnyTouch2 特征缓存，以及使用真实 batch 做有界调试和 smoke；不在开发机运行正式训练。当前数据、验证与训练状态统一见 [`../IMPLEMENTATION.md`](../IMPLEMENTATION.md)。

此前三卡 A800 的 Stage 2 性能 debug 已完成，每卡 batch 16/32 都成功，吞吐分别为 17.99/22.63 序列每秒。当时三卡配方采用每卡 batch 64、全局 batch 192、GA=1；当前正式配方已经改为上面的八卡配置。BS 64 及 Stage 1 的 A800 性能尚未实测，不把已有结果当作这两项的测量值。历史性能入口见 [`benchmark.md`](benchmark.md)，GPU 数支持 1/2/3/4/8。

当前百度云与开发机使用相同挂载路径 `/zhe_li/tactile-wam/contact-belief-wam`；性能 debug 命令已经按此填写。本文下方 `/workspace` 目录树仅为布局示例，不是当前作业的挂载位置。

## 历史三卡单任务配置

配置已保存为 [`train_insert_hole_3xa800_bs64.yaml`](../configs/train_insert_hole_3xa800_bs64.yaml)，复用已安装环境和现有权重。Stage 1 从官方世界模型初始化，默认训练 2,000 步；完成后加载其结果进入 Stage 2，训练 10,000 步。两个阶段均使用每卡 64、全局 192、GA=1。性能 debug 没有保存模型，不作为续训来源。

```bash
bash /zhe_li/tactile-wam/contact-belief-wam/scripts/cloud_train.sh \
  --run-config /zhe_li/tactile-wam/contact-belief-wam/configs/train_insert_hole_3xa800_bs64.yaml \
  --bootstrap-env false \
  --resume none
```

输出目录为 `/zhe_li/tactile-wam/contact-belief-wam/runs/insert_hole/full-k10-3xa800-bs192`。中断后可用相同命令将 `--resume none` 改为 `--resume auto`。开发机只检查命令解析，不运行该正式任务。

## 云端目录

以下是路径布局示例。无需生成云端上传包或上传清单；A800 上已有文件可直接通过 `cloud_train.sh` 的路径参数或 run config 引用，请将示例路径替换为用户实际路径：

```text
/workspace/contact-belief-data/
  hdf5/original/<task>/       # 原始 HDF5
  hdf5/recovery/<task>/       # 恢复 HDF5
  cache/anytouch2/            # 冻结 AnyTouch2 token cache
  manifests/<task>.jsonl      # 相对 data_root 的清单
  stats/<task>.quantiles.json # 每任务训练 split 的 q01/q99
/workspace/models/
  pi05_base_local_debug/      # 已验证的转换后 PyTorch 初始化权重
  tokenizer/                  # tokenizer 文件，可单独指定
  wave0_robotwin_world/       # 提取后的 RoboTwin WAVE-0 世界模型权重
/workspace/runs/<task>/full-k10/
```

AnyTouch2 cache 只保存触觉 tokens 与帧对齐信息，不能替代原始 HDF5 的相机、状态、动作或触觉画面。

## 首次训练

以下示例列出单机 8×A800 的 `insert_hole` 训练参数，全局 batch 128、每卡 16、梯度累积 1。两个阶段均使用这组 batch 参数；EMA、学习率调度和模型训练范围沿用当前实现，不包含此前仅讨论过的 WAVE-0 配方对齐修改。确认 A800 上已有可用项目环境，并将代码、数据、manifest、统计、cache、转换后的初始化权重、tokenizer 与 WAVE-0 权重参数改为实际路径。已有 `.venv` 时传 `--bootstrap-env false`，不需要 wheelhouse；首次安装可传 `true` 使用项目环境安装脚本。

```bash
export CBW_PROJECT=/zhe_li/tactile-wam/contact-belief-wam
export CBW_DATA_ROOT="$CBW_PROJECT/data"
export CBW_PI05_BASE="$CBW_PROJECT/models/pi05_base_local_debug"
export CBW_TOKENIZER="$CBW_PROJECT/models/tokenizer"
export CBW_WAVE0="$CBW_PROJECT/models/wave0_robotwin_world"
export CBW_RUN="$CBW_PROJECT/runs/insert_hole/full-k10-8xa800-bs128"

cd "$CBW_PROJECT"
bash scripts/cloud_train.sh \
  --task insert_hole \
  --mode full \
  --data-root "$CBW_DATA_ROOT" \
  --manifest "$CBW_DATA_ROOT/manifests/insert_hole.jsonl" \
  --stats "$CBW_DATA_ROOT/stats/insert_hole.quantiles.json" \
  --tactile-cache "$CBW_DATA_ROOT/cache/anytouch2" \
  --pi05-base "$CBW_PI05_BASE" \
  --world-init "$CBW_WAVE0" \
  --output "$CBW_RUN" \
  --python python3.10 \
  --bootstrap-env false \
  --tokenizer-path "$CBW_TOKENIZER" \
  --device cuda \
  --num-gpus 8 \
  --global-batch 128 \
  --micro-batch 16 \
  --precision bf16 \
  --gradient-checkpointing true \
  --k 10 \
  --data-mix 1:1 \
  --belief-warmup-min 0 \
  --belief-warmup-max 4 \
  --belief-bptt-steps 2 \
  --belief-to-latent-queries true \
  --belief-to-world-state true \
  --belief-to-future-corrector true \
  --belief-to-contact-predictor true \
  --belief-to-expert true \
  --stage1-steps auto \
  --stage1-pretrained-steps 2000 \
  --stage1-scratch-steps 20000 \
  --stage2-steps 10000 \
  --lr-new 1e-4 \
  --lr-policy 2.5e-5 \
  --grad-clip 1 \
  --val-every 500 \
  --val-batches 8 \
  --save-every 1000 \
  --seed 42 \
  --workers 2 \
  --resume none
```

每 GPU micro batch 为 16，`8 × 16 × 1 = 128`，因此解析出的 `accumulation_steps` 为 1。batch 按完整 belief 序列计；序列末尾最多两次更新保留梯度。八卡参数解析和两阶段 dry run 已检查，未在开发机启动正式训练，也未测量八卡 A800 的实际显存或吞吐。`full` 和 `wave0` 有世界模型目标：配置提供 WAVE-0 初始化时 Stage 1 为 2,000 步，没有初始化时为 20,000 步。`pi05` 与 `direct_touch` 没有世界模型训练目标，Stage 1 会跳过，直接执行配置的 Stage 2；这两种模式应将 `world_init` 设为 `none`，避免对不使用的权重做存在性检查。默认 `k=10` 在 execute/query/target 中统一，支持设为 `5`。五条 belief 通路在 `full` 模式默认全开。

上例用 `--resume none` 明确开始新 run。如果输出目录已有 `run_config.yaml`，不要用这个选项覆盖旧 run；改用下方续训命令，或选择新的输出目录。

## 使用 run config 续训

配置文件采用扁平 snake_case 字段，CLI 参数会覆盖同名 YAML 字段；配置内路径应指向 A800 上用户实际文件。重新提交时，`--resume auto` 先选择输出目录内最新的完整 Stage 2 checkpoint；没有完整 Stage 2 checkpoint 时，才选择最新的完整 Stage 1 checkpoint。续训须保留原 GPU 数和模型配置（包括 `k`、模式等），训练器会恢复 optimizer、scheduler、步数与随机状态。

```bash
export CBW_PROJECT=/zhe_li/tactile-wam/contact-belief-wam
export CBW_RUN_CONFIG="$CBW_PROJECT/runs/insert_hole/full-k10-8xa800-bs128/run_config.yaml"
cd "$CBW_PROJECT"
bash scripts/cloud_train.sh \
  --run-config "$CBW_RUN_CONFIG" \
  --resume auto
```

`--run-config` 且项目 `.venv` 已存在时，脚本会直接使用该环境，不再重新 bootstrap。若续训选中已到配置末步的完整 Stage 2 checkpoint，而进程在 `export/final/` 写完前中断，脚本会从该 checkpoint 恢复导出；只有检查最终导出文件齐备后才把 `run_state.json` 标为 `complete`。不完整导出不会被当作完成状态。首次 HDMI 训练可将 run config 和数据字段改成 A800 上的实际 HDMI 路径，再显式传 `--resume none`。

## 输出与验证记录

训练输出采用如下布局。具体 checkpoint 目录只会在相应阶段达到保存步数后生成；Stage 2 完成后导出 `export/final/`。

```text
/workspace/runs/<task>/full-k10/
  run_config.yaml
  data_manifest.jsonl
  quantiles.json
  initialization_report.json
  initialization_stage1.json
  run_state.json
  logs/
    metrics.jsonl
    train.log
    tensorboard/
  stage1/
    checkpoints/step_XXXXXXXX/
      complete.json
      pretrained_model/
      training/
  stage2/
    checkpoints/step_XXXXXXXX/
      complete.json
      pretrained_model/
      training/
  export/final/
    config.json
    model.safetensors
    quantiles.json
    tokenizer/
    deployment.json
```

`complete.json` 标记一份可用于自动续训选择的完整 checkpoint。全量模式的固定验证 split 按 `original`、`recovery` 分组记录视觉与触觉相关指标，例如 `val/original/visual_initial`、`val/recovery/tactile_future`、`val/original/visual_corrected`；不同阶段和模式产生的指标会不同。`val/all/...` 是合并统计。验证观测固定，按 500 步记录未来视觉/触觉损失；这些数值用于查看训练变化，不是通过门槛。Checkpoint 默认每 1,000 步保存，且阶段末步也保存。

## 将部署权重取回开发机

先将云端 `export/final/` 复制到项目下相同相对位置。把 `USER@HOST` 换成实际云端登录地址：

```bash
rsync -av --progress \
  "USER@HOST:/workspace/runs/insert_hole/full-k10/export/final/" \
  "/zhe_li/tactile-wam/contact-belief-wam/runs/insert_hole/full-k10/export/final/"
```

以下 `serve` 命令启动独立策略服务，使用本机 AnyTouch2 源码和检查点；项目 `.venv` 应是正式 Python 3.10 环境：

```bash
/zhe_li/tactile-wam/contact-belief-wam/.venv/bin/cbwam serve \
  --checkpoint /zhe_li/tactile-wam/contact-belief-wam/runs/insert_hole/full-k10/export/final \
  --host 127.0.0.1 \
  --port 18780 \
  --device cuda \
  --tactile-device cpu \
  --anytouch-source /zhe_li/tactile-wam/AnyTouch2 \
  --k 10 \
  --threads 12
```

批量 `eval` 会自行启动并关闭策略服务，所以运行下面命令时不要同时占用相同端口的独立 `serve`。评估要求开发机有匹配的 UniVTAC/Isaac 仿真环境和 CUDA。该链路已用本轮 debug checkpoint 完成两任务各三个完整 episode，以及 `k=5` 短闭环检查；具体设备和结果见 [`../IMPLEMENTATION.md`](../IMPLEMENTATION.md)。正式训练后的 checkpoint 仍需按下面命令评测其任务表现。

```bash
/zhe_li/tactile-wam/contact-belief-wam/.venv/bin/cbwam eval \
  --task insert_hole \
  --checkpoint /zhe_li/tactile-wam/contact-belief-wam/runs/insert_hole/full-k10/export/final \
  --output /zhe_li/tactile-wam/contact-belief-wam/runs/eval/insert_hole/full-k10 \
  --seeds 1000000,1000001,1000002 \
  --device cuda \
  --port 18780 \
  --k 10
```

