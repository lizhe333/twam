"""Warm-start both stages from a complete TWAM policy without resetting it."""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_model

from .config import ContactConfig


CORE = "base_policy.model."
VISION1 = CORE + "vision_backbone."
VISION2 = CORE + "world_vision_backbone."
WORLD = tuple(CORE + name + "." for name in
              ("world_model", "world_action_encoder", "world_state_encoder"))
COMPATIBLE = ("aligned", "mode", "camera_mode", "independent_tactile", "k", "chunk_size",
              "action_dim", "padded_dim", "width", "heads", "ffn", "belief_tokens", "tactile_tokens",
              "world_width", "world_depth", "world_heads", "world_ffn", "vision_width", "vision_pool",
              "expert_width", "tokenizer_max_length", "encoder_reference", "world_norm_stats",
              "belief_to_latent_queries", "belief_to_world_state", "belief_to_future_corrector",
              "belief_to_contact_predictor", "belief_to_expert")


def check_compatible(config, saved, *, source_stage):
    if saved.stage != source_stage or saved.mode != "full" or not saved.independent_tactile:
        raise ValueError(f"Expected a complete, independent-tactile {source_stage} TWAM")
    for name in COMPATIBLE:
        if getattr(config, name) != getattr(saved, name):
            raise ValueError(f"Fine-tuning checkpoint differs in {name}")
    if config.stage == "stage2" and source_stage == "stage2" and config.policy_norm_stats != saved.policy_norm_stats:
        raise ValueError("Fine-tuning must retain the policy normalization")


def stage1_source_key(key):
    """Read the frozen WM encoder, never the already adapted policy RGB tower."""
    if key.startswith(VISION1):
        return VISION2 + key[len(VISION1):]
    if key.startswith(WORLD) or key.startswith("contact."):
        return key
    raise ValueError(f"Unexpected Stage-1 tensor: {key}")


def stage2_overlay_key(key):
    if key.startswith(VISION2):
        return VISION1 + key[len(VISION2):]
    if key.startswith(WORLD) or key.startswith("contact."):
        return key
    return None


@torch.no_grad()
def copy_tensors(model, path, mapping, *, unchanged_prefix=None):
    """Require every mapped tensor; copy one tensor at a time to limit CPU RAM."""
    state = model.state_dict()
    with safe_open(str(path), framework="pt", device="cpu") as source:
        keys = set(source.keys())
        for dest, src in mapping.items():
            if src not in keys:
                raise ValueError(f"Fine-tuning checkpoint is missing {src}")
            value = source.get_tensor(src)
            if value.shape != state[dest].shape:
                raise ValueError(f"Fine-tuning shape differs: {src} -> {dest}")
            if unchanged_prefix and dest.startswith(unchanged_prefix) and not torch.equal(state[dest].cpu(), value):
                raise ValueError("Stage-1 frozen visual encoder changed; policy/world feature spaces would differ")
            state[dest].copy_(value)
    return len(mapping)


@torch.no_grad()
def overlay_ema(model, path, *, stage1=False):
    state = model.state_dict()
    source_to_dest = {stage1_source_key(k): k for k in state if k != "optimizer_step"} if stage1 else {k: k for k in state}
    mapping = {}
    with safe_open(str(path), framework="pt", device="cpu") as source:
        for key in source.keys():
            if key in source_to_dest:
                mapping[source_to_dest[key]] = key
            elif not stage1:
                raise ValueError(f"Unexpected EMA tensor: {key}")
    if not mapping:
        raise ValueError("EMA contains no applicable fine-tuning tensors")
    return copy_tensors(model, path, mapping)


def initialize_finetune(config, source, *, use_ema=True, stage1_checkpoint=None, stage2_only=False):
    from .model import ContactWAM

    if stage2_only and (config.stage != "stage2" or stage1_checkpoint):
        raise ValueError("Stage-2-only continuation requires Stage 2 without a Stage-1 overlay")
    source = Path(source)
    check_compatible(config, ContactConfig.load(source / "config.json"), source_stage="stage2")
    if config.stage not in {"stage1", "stage2"}:
        raise ValueError("Two-stage fine-tuning requires stage1 or stage2")
    model = ContactWAM(config)
    report = dict(finetune_from=str(source), use_ema=use_ema, optimizer_reset=True,
                  original_policy_preserved=config.stage == "stage2", source_stage="stage2")
    report["stage2_only"] = stage2_only
    if config.stage == "stage1":
        mapping = {key: stage1_source_key(key) for key in model.state_dict() if key != "optimizer_step"}
        report["transferred_tensors"] = copy_tensors(model, source / "model.safetensors", mapping)
    else:
        load_model(model, str(source / "model.safetensors"), strict=True)
        report["transferred_tensors"] = len(model.state_dict())
    if use_ema:
        report["ema_tensors"] = overlay_ema(model, source / "ema.safetensors", stage1=config.stage == "stage1")
    if config.stage == "stage2" and not stage2_only:
        if not stage1_checkpoint:
            raise ValueError("Fine-tuning Stage 2 requires the newly adapted Stage-1 checkpoint")
        teacher = Path(stage1_checkpoint) / "pretrained_model"
        check_compatible(config, ContactConfig.load(teacher / "config.json"), source_stage="stage1")
        mapping = {key: src for key in model.state_dict() if (src := stage2_overlay_key(key)) is not None}
        report["adapted_tensors"] = copy_tensors(model, teacher / "model.safetensors", mapping,
                                                 unchanged_prefix=VISION2)
        report["adapted_stage1"] = str(stage1_checkpoint)
    model.optimizer_step.zero_()
    model.configure_trainability()
    return model, report
