"""Task configurations use the pinned upstream JAX implementation without a registry patch."""
from __future__ import annotations

from openpi.shared.univtac_runtime import configure, root
configure()

import dataclasses
from pathlib import Path
import yaml
import flax.nnx as nnx
from openpi.models.pi0_config import Pi0Config
from openpi.training import config as upstream
from openpi.training import optimizer
from openpi import transforms as T
from openpi.shared import nnx_utils
from openpi.shared.univtac_constants import prompt, task_names
from openpi.policies.univtac_policy import UniVTACInputs, UniVTACOutputs
from openpi.training.univtac_weights import StrictBaseLoader


@dataclasses.dataclass(frozen=True)
class UniVTACData(upstream.DataConfigFactory):
    task: str = "lift_can"
    camera_mode: str = "head_wrist"

    def create(self, assets_dirs, model_config):
        delta_mask = T.make_bool_mask(7, -1)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=T.Group(
                inputs=[UniVTACInputs(camera_mode=self.camera_mode), T.DeltaActions(delta_mask)],
                outputs=[T.AbsoluteActions(delta_mask), UniVTACOutputs()],
            ),
            model_transforms=upstream.ModelTransformFactory(default_prompt=prompt(self.task))(model_config),
            action_sequence_keys=("actions",),
            prompt_from_task=True,
            use_quantile_norm=True,
        )


def make_config(task: str, mode: str = "full", *, run: str = "default", resume: bool = False,
                steps: int | None = None, batch_size: int | None = None,
                camera_mode: str | None = None, recipe=None) -> upstream.TrainConfig:
    task_names(task)
    options = load_recipe(recipe)
    camera_mode = camera_mode or options.get("task_cameras", {}).get(task, options.get("camera_mode", "head_wrist"))
    UniVTACInputs(camera_mode=camera_mode)  # Validate before allocating a model.
    if mode not in ("full", "smoke", "lora", "overfit", "base"):
        raise ValueError(mode)
    local = mode in ("lora", "overfit")
    fsdp_devices = 1 if local or mode == "base" else int(options.get("fsdp_devices", 4))
    if fsdp_devices not in (1, 2, 4):
        raise ValueError("fsdp_devices must be 1, 2, or 4")
    split = "debug_train" if mode in ("lora", "overfit", "base") else "full"
    model = Pi0Config(pi05=True, action_dim=32, action_horizon=50, discrete_state_input=True,
                      paligemma_variant="gemma_2b_lora" if local else "gemma_2b",
                      action_expert_variant="gemma_300m_lora" if local else "gemma_300m")
    name = f"pi05_univtac_{task}_{mode}"
    return upstream.TrainConfig(
        name=name, project_name="pi05-univtac", exp_name=run, model=model,
        weight_loader=StrictBaseLoader(str(root() / "models/pi05_base/params")),
        # Local acceptance trains only LoRA matrices. The upstream convenience
        # filter leaves the vision tower trainable; explicitly freezing all
        # non-LoRA leaves keeps this 24GB debug separate from full fine-tuning.
        freeze_filter=nnx.Not(nnx_utils.PathRegex(".*lora.*")) if local else nnx.Nothing,
        data=UniVTACData(
            task=task, camera_mode=camera_mode, repo_id=f"pi05_univtac/{split}/{task}",
            assets=upstream.AssetsConfig(assets_dir=str(root() / "assets/pi05_univtac"), asset_id=f"{split}/{task}"),
        ),
        assets_base_dir=str(root() / "assets/configs"),
        checkpoint_base_dir=str(root() / "outputs/checkpoints"),
        seed=options.get("seed", 42), batch_size=batch_size or (1 if local or mode == "base" else options.get("batch_size", 64)),
        num_workers=0 if local else options.get("num_workers", 2),
        num_train_steps=steps if steps is not None else (200 if mode == "overfit" else 20 if local else 100 if mode == "smoke" else options.get("num_train_steps", 10000)),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=10 if local else 1000,
            peak_lr=1e-4 if local else 2.5e-5,
            decay_steps=1000 if local else 10000,
            decay_lr=1e-5 if local else 2.5e-6,
        ),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None if local else options.get("ema_decay", 0.99),
        log_interval=1 if local or mode == "smoke" else 100,
        save_interval=100 if mode == "overfit" else 10 if local else 50 if mode == "smoke" else 1000,
        keep_period=None, overwrite=False, resume=resume, wandb_enabled=False,
        fsdp_devices=fsdp_devices,
        policy_metadata={"task": task, "action_horizon": 50, "action_dim": 8, "fps": 60,
                         "execute_steps": 10, "mode": mode, "prompt": prompt(task),
                         "camera_mode": camera_mode, "adapter_release": "openpi-univtac"},
    )


def load_recipe(path=None):
    """Experiment settings; existing defaults remain unchanged when omitted."""
    if path is None:
        return {}
    options = dict(path) if isinstance(path, dict) else yaml.safe_load(Path(path).read_text())
    if not isinstance(options, dict):
        raise ValueError("Expected a YAML mapping")
    allowed = {"camera_mode", "task_cameras", "fsdp_devices", "batch_size", "num_train_steps", "num_workers", "seed", "ema_decay"}
    if set(options) - allowed:
        raise ValueError(f"Unknown UniVTAC settings: {sorted(set(options) - allowed)}")
    UniVTACInputs(camera_mode=options.get("camera_mode", "head_wrist"))
    for task, camera in options.get("task_cameras", {}).items():
        task_names(task)
        UniVTACInputs(camera_mode=camera)
    for key in ("fsdp_devices", "batch_size", "num_train_steps"):
        if key in options and (type(options[key]) is not int or options[key] <= 0):
            raise ValueError(f"{key} must be a positive integer")
    if options.get("fsdp_devices", 4) not in (1, 2, 4):
        raise ValueError("fsdp_devices must be 1, 2, or 4")
    if options.get("batch_size", 64) % options.get("fsdp_devices", 4):
        raise ValueError("batch_size must be divisible by fsdp_devices")
    return options
