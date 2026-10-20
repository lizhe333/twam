"""Explicit official-checkpoint conversion; never silently skip policy/body tensors."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .paths import atomic_json

WORLD_NAMES = ("world_model", "world_action_encoder", "world_state_encoder")


def checkpoint_file(path):
    path = Path(path)
    path = path / "model.safetensors" if path.is_dir() else path
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return path


def canonical(key):
    for prefix in ("base_policy.model.", "model."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def extract_world(source, destination):
    source, destination = checkpoint_file(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with safe_open(source, framework="pt", device="cpu") as f:
        selected = {canonical(k): f.get_tensor(k) for k in f.keys()
                    if canonical(k).split(".")[0] in WORLD_NAMES}
    if not all(any(k.startswith(name + ".") for k in selected) for name in WORLD_NAMES):
        raise ValueError("Source does not contain all three official world modules")
    save_file(selected, str(destination / "model.safetensors"))
    source_config = source.parent / "config.json"
    cfg = json.loads(source_config.read_text()) if source_config.exists() else {}
    atomic_json(destination / "source_config.json", cfg)
    report = {"source": str(source), "keys": sorted(selected), "camera": "cam_high",
              "policy": "Policy backbone is separately initialized from lerobot/pi05_base"}
    atomic_json(destination / "extraction.json", report)
    return report


def load_world(model, source):
    """Take cam_high; copy compatible encoders including internally padded 32-D input."""
    source = checkpoint_file(source)
    dst = model.core.state_dict()
    selected = {k: v for k, v in dst.items() if k.split(".")[0] in WORLD_NAMES}
    report = {"source": str(source), "loaded": [], "converted": [], "reinitialized": [], "unused": []}
    with safe_open(source, framework="pt", device="cpu") as f:
        index = {canonical(k): k for k in f.keys()}
        updates = {}
        for key, value in selected.items():
            if key not in index:
                raise ValueError(f"Official world checkpoint is missing {key}")
            tensor = f.get_tensor(index[key])
            if tensor.shape == value.shape:
                updates[key] = tensor
                report["loaded"].append(key)
            elif key in {"world_model.camera_embedding", "world_model.future_queries"} and tensor.shape[1] == 3 and value.shape[1] == 1:
                sliced = tensor[:, :1]
                if sliced.shape != value.shape:
                    raise ValueError(f"Non-camera dimensions differ: {key}")
                updates[key] = sliced.contiguous()
                report["converted"].append({"key": key, "rule": "cam_high (camera index 0) -> head"})
            elif key == "world_action_encoder.position_embedding" and tensor.shape[0::2] == value.shape[0::2]:
                resized = value.clone()
                count = min(tensor.shape[1], value.shape[1])
                resized[:, :count] = tensor[:, :count]
                updates[key] = resized
                report["converted"].append({"key": key, "rule": f"copy position prefix {count}; initialize remaining positions"})
            elif key in {"world_action_encoder.input_projection.0.weight", "world_state_encoder.projection.0.weight"}:
                # Only a truly incompatible input layer is reinitialized. This is
                # normally unnecessary: both 14-D and 8-D inputs are padded to 32.
                report["reinitialized"].append({"key": key, "source_shape": list(tensor.shape), "target_shape": list(value.shape)})
            else:
                raise ValueError(f"Incompatible world tensor {key}: {tuple(tensor.shape)} -> {tuple(value.shape)}")
        report["unused"] = sorted(k for k in f.keys() if canonical(k) not in selected)
    model.core.load_state_dict(updates, strict=False)  # Every selected body key was checked above.
    return report


def initialize(config, *, world_init=None, stage1_checkpoint=None):
    from .model import ContactWAM, WAVE0Policy, wave_config
    checkpoint_file(config.pi05_base)
    base = WAVE0Policy.from_pretrained(config.pi05_base, config=wave_config(config),
                                      strict=True, local_files_only=True)
    model = ContactWAM(config, base_policy=base)
    if config.aligned and config.stage != "stage1" and config.use_world:
        # Inference construction omits the official Stage-2 loader; initialize
        # its independent frozen visual target tower from the same base weights.
        WAVE0Policy._load_frozen_vision_backbone(model.core.world_vision_backbone,
                                                str(checkpoint_file(config.pi05_base)))
    report = {"pi05_base": config.pi05_base, "policy_strict": True,
              "world_pretrained": False, "new_contact_modules": config.use_belief}
    source = stage1_checkpoint or world_init
    if source and str(source).lower() != "none" and config.use_world:
        report["world"] = load_world(model, source)
        report["world_pretrained"] = True
    if stage1_checkpoint and model.contact is not None:
        with safe_open(checkpoint_file(stage1_checkpoint), framework="pt", device="cpu") as f:
            state = {k.removeprefix("contact."): f.get_tensor(k) for k in f.keys() if k.startswith("contact.")}
        model.contact.load_state_dict(state, strict=True)
        report["contact_from_stage1"] = str(stage1_checkpoint)
    return model, report


def load_contact_resized(model, path):
    """Only the learned query length may differ between exported k=10 and k=5."""
    from safetensors.torch import load_model
    import tempfile
    path = checkpoint_file(path)
    expected = model.state_dict()
    with safe_open(path, framework="pt", device="cpu") as f:
        tensors = {}
        metadata = f.metadata()
        for key in f.keys():
            tensor = f.get_tensor(key)
            if key in expected and tensor.shape != expected[key].shape:
                if key != "base_policy.model.wam_latent_action_queries":
                    raise ValueError(f"Unexpected shape change when changing k: {key}")
                value = expected[key].clone()
                count = min(value.shape[1], tensor.shape[1])
                value[:, :count] = tensor[:, :count]
                tensor = value
            tensors[key] = tensor
    with tempfile.TemporaryDirectory(prefix="cbwam-load-") as tmp:
        resized = Path(tmp) / "model.safetensors"
        save_file(tensors, str(resized), metadata=metadata)
        load_model(model, str(resized), strict=True)
