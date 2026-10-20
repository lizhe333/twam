"""Shared offline/online transforms. UniVTAC JPEG arrays already have RGB channel order."""
from __future__ import annotations

import numpy as np

TOUCH_OFFSETS = (-6, -4, -2, 0)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
PREPROCESS_VERSION = "raw-rgb-torch-bilinear-aa224-clip-fourframe-v1"


def decode_rgb(encoded):
    import cv2
    if isinstance(encoded, np.ndarray) and encoded.ndim == 3:
        return encoded
    data = np.frombuffer(bytes(encoded), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid UniVTAC JPEG")
    # Writer uses cv2.imencode on numerical RGB arrays without a channel flip.
    return image


def window_indices(frame: int, segment_start: int = 0):
    return np.maximum(segment_start, frame + np.asarray(TOUCH_OFFSETS)).astype(np.int64)


def tactile_preprocess(windows):
    """[...,4,H,W,3] uint8 -> [...,4,3,224,224] float32, with no background subtraction."""
    import torch
    import torch.nn.functional as F
    x = torch.as_tensor(np.asarray(windows))
    if x.shape[-1] != 3 or x.shape[-4] != 4 or x.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 [...,4,H,W,3], got {x.shape}/{x.dtype}")
    lead = x.shape[:-3]
    x = x.reshape(-1, *x.shape[-3:]).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
    mean = x.new_tensor(CLIP_MEAN)[None, :, None, None]
    std = x.new_tensor(CLIP_STD)[None, :, None, None]
    return ((x - mean) / std).reshape(*lead, 3, 224, 224)


def vision_preprocess(images):
    """Match official WAVE-0 resize-with-pad and [-1,1] image normalization."""
    import torch
    from .paths import configure
    configure()
    from lerobot.policies.wave0.modeling_wave0 import resize_with_pad_torch
    image = torch.as_tensor(np.asarray(images)).float().permute(0, 3, 1, 2) / 255.0
    image = resize_with_pad_torch(image, 224, 224)
    return image * 2.0 - 1.0


class Quantiles:
    def __init__(self, stats):
        import json
        from pathlib import Path
        self.stats = json.loads(Path(stats).read_text()) if isinstance(stats, (str, Path)) else stats

    def normalize(self, array, key):
        q = self.stats[key]
        lo, hi = np.asarray(q["q01"], np.float32), np.asarray(q["q99"], np.float32)
        # Constant fingers/joints retain a finite scale; values outside quantiles aren't clipped.
        return (2 * (np.asarray(array, np.float32) - lo) / np.maximum(hi - lo, 1e-6) - 1).astype(np.float32)

    def unnormalize(self, array, key="action"):
        q = self.stats[key]
        lo, hi = np.asarray(q["q01"], np.float32), np.asarray(q["q99"], np.float32)
        return ((np.asarray(array, np.float32) + 1) / 2 * np.maximum(hi - lo, 1e-6) + lo).astype(np.float32)


class PromptTokenizer:
    def __init__(self, path, max_length=200):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.tokenizer.padding_side = "right"
        self.max_length = max_length

    def __call__(self, prompts, normalized_states):
        state = np.pad(np.asarray(normalized_states), ((0, 0), (0, 24)))
        bins = np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1
        texts = []
        for prompt, values in zip(prompts, bins):
            prompt = prompt.strip().replace("_", " ").replace("\n", " ")
            texts.append(f"Task: {prompt}, State: {' '.join(map(str, values))};\nAction: ")
        result = self.tokenizer(texts, return_tensors="pt", padding="max_length", max_length=self.max_length,
                                truncation=True, add_special_tokens=True)
        return result["input_ids"], result["attention_mask"].bool()
