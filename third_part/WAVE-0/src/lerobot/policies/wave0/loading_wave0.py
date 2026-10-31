from __future__ import annotations

from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_model

from lerobot.policies.wave0.modeling_wave0 import WAVE0Policy
from lerobot.policies.wave0.image_processing_wave0 import augment
from lerobot.configs.univtac import policy_config

WORLD = ("world_model", "world_action_encoder", "world_state_encoder")


# Compatibility name; no subclass or additional parameter namespace.
BaselineWAVE0 = WAVE0Policy


def load_initial_world(policy, path):
    """Select ordered pretrained camera slots and adapt positions with strict shapes."""
    path = Path(path)
    if path.is_dir():
        path /= "model.safetensors"
    report = {"loaded": [], "converted": [], "unused": []}
    with safe_open(path, framework="pt", device="cpu") as source:
        keys = {k.removeprefix("model."): k for k in source.keys()}
        used = set()
        for name in WORLD:
            module = getattr(policy.model, name)
            values = {}
            for key, expected in module.state_dict().items():
                full = name+"."+key
                if full not in keys:
                    raise ValueError(f"Missing pretrained tensor: {full}")
                value = source.get_tensor(keys[full])
                used.add(full)
                if value.shape != expected.shape:
                    if full in ("world_model.camera_embedding", "world_model.future_queries"):
                        # Ordered source camera slots initialize the requested real views.
                        # Both embeddings remain trainable during Stage 1.
                        value = value[:, :expected.shape[1]].contiguous()
                        report["converted"].append({"key": full, "rule": f"first {expected.shape[1]} pretrained camera slots -> ordered real views"})
                    elif full == "world_action_encoder.position_embedding":
                        resized = expected.clone()
                        n = min(value.shape[1], expected.shape[1])
                        resized[:, :n] = value[:, :n]
                        value = resized
                        report["converted"].append({"key": full, "rule": f"copy {n} positions; extra positions unused by ten-action prefix"})
                    else:
                        raise ValueError(f"Unexpected shape: {full}: {value.shape} vs {expected.shape}")
                if value.shape != expected.shape:
                    raise ValueError(f"Invalid adapted shape: {full}")
                values[key] = value
                report["loaded"].append(full)
            module.load_state_dict(values, strict=True)
        report["unused"] = sorted(set(keys)-used)
        if report["unused"]:
            raise ValueError("Expected a world-only pretrained checkpoint")
    return report


def create_policy(run, stage, stage1=None, checkpoint=None):
    config = policy_config(run, stage, stage1)
    if checkpoint and stage == "stage2":
        # Avoid official from_pretrained's Stage-2 re-injection on resume: restore
        # all saved raw tensors exactly, including policy vision and step counter.
        policy = BaselineWAVE0(config)
        load_model(policy, str(Path(checkpoint)/"model.safetensors"), strict=True)
        report = {"resume_strict": True}
    else:
        policy = BaselineWAVE0.from_pretrained(
            str(checkpoint or run["base"]), config=config, strict=True, local_files_only=True)
        report = {"base": run["base"], "policy_body_strict": True}
        if stage == "stage1" and not checkpoint and run.get("world_init"):
            report["world"] = load_initial_world(policy, run["world_init"])
    policy.policy_augmentation = run["policy_augmentation"]
    if stage == "stage2":
        # Unused vocabulary head in the action expert is not an actuator head.
        policy.model.paligemma_with_expert.gemma_expert.lm_head.requires_grad_(False)
        if run.get("shared_wm"):
            from lerobot.datasets.univtac_shared import install_world_normalization
            install_world_normalization(policy, Path(run["baseline"])/"norm_stats.json", run["wm_norm"])
            report["normalization"] = "task-policy -> physical -> shared-WM (world inputs only)"
    return policy, report


def group_name(name):
    if any(name.startswith("model."+k+".") for k in WORLD):
        return "world"
    if ".world_vision_backbone." in name or ".vision_backbone." in name:
        return "target_vision"
    if ".wam_" in name:
        return "future_condition"
    if ".vision_tower." in name or ".multi_modal_projector." in name:
        return "policy_vision"
    if ".paligemma." in name:
        return "language"
    return "action_expert"


def gradient_report(policy):
    report = {}
    for name, p in policy.named_parameters():
        group = group_name(name)
        row = report.setdefault(group, {"trainable": 0, "frozen": 0, "gradient_squared_norm": 0., "missing": []})
        row["trainable" if p.requires_grad else "frozen"] += p.numel()
        if not p.requires_grad:
            if p.grad is not None:
                raise AssertionError(f"Frozen tensor has gradient: {name}")
        elif p.grad is None:
            row["missing"].append(name)
        else:
            if not torch.isfinite(p.grad).all():
                raise FloatingPointError(f"Nonfinite gradient: {name}")
            row["gradient_squared_norm"] += float(p.grad.float().square().sum())
    return report
