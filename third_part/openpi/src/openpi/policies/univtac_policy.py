"""One data contract for LeRobot samples and live Isaac camera observations."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from openpi.policies.univtac_images import resize_rgb, to_rgb_uint8


@dataclass(frozen=True)
class UniVTACInputs:
    camera_mode: str = "head_wrist"

    def __post_init__(self):
        if self.camera_mode not in ("head_only", "head_wrist"):
            raise ValueError(f"Unsupported camera mode: {self.camera_mode}")

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32).copy()
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError(f"Expected finite state (8,), got {state.shape}")
        head = resize_rgb(to_rgb_uint8(data["image"]))
        use_wrist = self.camera_mode == "head_wrist"
        wrist = resize_rgb(to_rgb_uint8(data["wrist_image"])) if use_wrist else np.zeros_like(head)
        result = {
            "state": state,
            "image": {"base_0_rgb": head, "left_wrist_0_rgb": wrist, "right_wrist_0_rgb": np.zeros_like(head)},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.bool_(use_wrist), "right_wrist_0_rgb": np.False_},
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32).copy()
            if actions.shape != (50, 8) or not np.isfinite(actions).all():
                raise ValueError(f"Expected finite action chunk (50,8), got {actions.shape}")
            result["actions"] = actions
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclass(frozen=True)
class UniVTACOutputs:
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)[..., :8].copy()
        if actions.shape != (50, 8) or not np.isfinite(actions).all():
            raise ValueError(f"Policy returned invalid actions: {actions.shape}")
        return {"actions": actions}
