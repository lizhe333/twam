"""Camera selection travels with checkpoints; old checkpoints remain dual-view."""
import json
from pathlib import Path


def camera_mode_for_checkpoint(checkpoint, task):
    checkpoint = Path(checkpoint)
    path = checkpoint / "camera_contract.json"
    if not path.exists():
        identity_path = checkpoint.parent / "run_identity.json"
        if identity_path.exists():
            identity = json.loads(identity_path.read_text())
            if identity.get("training", {}).get("camera_mode") is not None:
                raise RuntimeError(f"Checkpoint is missing its camera contract: {path}")
        return "head_wrist"
    contract = json.loads(path.read_text())
    if (contract.get("schema_version") != 1 or contract.get("task") != task
            or contract.get("camera_mode") not in ("head_only", "head_wrist")):
        raise ValueError(f"Invalid camera contract: {path}")
    return contract["camera_mode"]


def camera_contract(config, task):
    return {"schema_version": 1, "task": task, "camera_mode": config.data.camera_mode,
            "adapter_release": "openpi-univtac",
            "model_views": ["base_0_rgb"] if config.data.camera_mode == "head_only"
                           else ["base_0_rgb", "left_wrist_0_rgb"]}
