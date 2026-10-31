"""Inference-only alignment with the existing JPEG / measured-q training data.

Kept outside src so installing this evaluator cannot change a running training
job's source identity. No model parameters or training targets are changed.
"""
import cv2
import numpy as np

from contact_belief_wam.sim_eval import Adapter, cpu


PROFILE = "jpeg-measured-history-v1"


def training_jpeg(image):
    """Match UniVTAC envs/utils/data.py and the training loader byte-for-byte.

    The writer passes rgb_marker directly to OpenCV, without RGB/BGR conversion
    or explicit quality settings; the loader decodes with IMREAD_COLOR.
    """
    image = cpu(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected tactile uint8 H x W x 3 image")
    okay, encoded = cv2.imencode(".jpg", image)
    if not okay:
        raise RuntimeError("Tactile JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape != image.shape:
        raise RuntimeError("Tactile JPEG decoding failed")
    return decoded


class TrainingAlignedAdapter(Adapter):
    def observe(self, observation):
        self.touch.append(np.stack([
            training_jpeg(observation["tactile"][f"{side}_tactile"]["rgb_marker"])
            for side in ("left", "right")
        ]))

    def execute(self, task, action):
        success, observation = super().execute(task, action)
        # The training history is q[t-k+1:t], relative to the previous replan
        # anchor q[t-k]. Use the observation AFTER the same two physical calls,
        # not the command sent to the controller. Keep command logs unchanged.
        measured = np.asarray(cpu(observation["embodiment"]["joint"]), dtype=np.float32)[:8].copy()
        if measured.shape != (8,) or not np.isfinite(measured).all():
            raise ValueError("Expected eight finite measured joint positions")
        self.executed[-1] = measured
        self.actions[-1]["measured_joint"] = measured.tolist()
        return success, observation
