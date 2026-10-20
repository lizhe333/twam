from __future__ import annotations

import os
from pathlib import Path
import yaml

from .runtime import project_root

TASKS = ("insert_hole", "insert_HDMI", "insert_tube", "pull_out_key",
         "grasp_classify", "lift_bottle", "lift_can", "put_bottle_in_shelf")
IMAGE = "observation.images.head"
# Official WAVE-0 uses an inclusive action index: h=9 means ten actions.
# UniVTAC labels action[i] with q[t+i+1], so that action prefix reaches frame t+10.
FUTURE = "observation.world_model.future.h9.images.head"


def load_config(path, workspace=None, output=None, task=None, stage1=None):
    root = project_root()
    cfg = yaml.safe_load(Path(path).read_text())
    shared = cfg.get("task") == "shared_wm"
    if task:
        if not shared or task not in TASKS:
            raise ValueError("--task requires the shared eight-task recipe")
        cfg["task"] = task
    if cfg["task"] not in (*TASKS, "shared_wm"):
        raise ValueError("Unknown UniVTAC task")
    workspace = Path(workspace or os.environ.get("TWAM_ROOT", root.parent)).resolve()
    cfg["workspace"] = str(workspace)
    for key in ("base", "world_init", "tokenizer"):
        if cfg.get(key):
            p = Path(cfg[key])
            cfg[key] = str(p if p.is_absolute() else workspace / p)
    cfg["baseline"] = str(root / "baseline" / cfg["task"])
    cfg["output"] = str(Path(output).resolve() if output else root / "outputs" / cfg["task"] / "wave0-head-only-seed42")
    if shared:
        if cfg.get("tasks") != list(TASKS):
            raise ValueError("Shared recipe must list all eight tasks in the canonical order")
        cfg["shared_wm"] = True
        cfg["workflow_root"] = cfg["output"]
        cfg["wm_norm"] = str(Path(cfg["output"])/"artifacts/wm_norm_stats.json")
        cfg["output"] = str(Path(cfg["output"])/("shared_wm" if not task else "tasks/"+task))
        cfg["num_gpus"] = cfg["stage1" if not task else "stage2"]["num_gpus"]
        if stage1:
            cfg["stage1_checkpoint"] = str(Path(stage1).resolve())
    if cfg["global_batch"] != 64 or cfg["seed"] != 42:
        raise ValueError("Baseline recipe requires global batch 64 and seed 42; create a new recipe for other experiments")
    for name in ("stage1", "stage2"):
        stage_cfg = cfg[name]
        batch = stage_cfg.get("global_batch", cfg["global_batch"])
        devices = stage_cfg.get("num_gpus", cfg["num_gpus"])
        if batch % (devices*stage_cfg["micro_batch"]):
            raise ValueError(f"{name}: global batch must be divisible by GPUs × micro batch")
    if shared and cfg["stage1"]["micro_batch"] % (2*len(TASKS)):
        raise ValueError("Shared WM physical batch must contain at least two examples of every task")
    return cfg


def policy_config(run, stage, stage1=None):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.wave0.configuration_wave0 import WAVE0Config
    return WAVE0Config(
        world_model_stage=stage, world_model_base_model_path=run["base"],
        world_model_stage1_checkpoint_path=str(stage1) if stage1 else None,
        input_features={IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
                        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,))},
        # OpenPI's baseline supervises the zero-padded 32-D flow target too.
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(32,))},
        empty_cameras=2, chunk_size=50, n_action_steps=10,
        skip_empty_camera_compute=run.get("performance", {}).get("skip_empty_camera_compute", False),
        attention_backend=run.get("performance", {}).get("attention_backend", "eager"),
        world_model_future_horizons=(9,), wam_latent_action_num_queries=10,
        dtype="float32", device="cpu", gradient_checkpointing=True,
        freeze_vision_encoder=False, train_expert_only=False,
        compile_model=False, push_to_hub=False, wam_share_frozen_vision_encoder=False,
        wam_loss_weight=1.0, wam_loss_warmup_steps=1000,
        wam_alignment_decay_start_step=3000, wam_alignment_decay_end_step=10000,
        world_model_compute_zero_action_metric=False,
    )
