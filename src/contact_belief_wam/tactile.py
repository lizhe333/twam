"""Frozen AnyTouch2 adapter; loads the original network without changing its source or weights."""
from __future__ import annotations

import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .paths import atomic_json, sha256
from .preprocessing import PREPROCESS_VERSION, TOUCH_OFFSETS, decode_rgb, tactile_preprocess, window_indices


class AnyTouchEncoder:
    def __init__(self, source, checkpoint=None, device="cpu", threads=8):
        import torch
        from transformers import AutoConfig
        from transformers.models.clip.modeling_clip import CLIPVisionTransformer
        self.source = Path(source)
        self.checkpoint = Path(checkpoint or self.source / "checkpoints/checkpoint-4frames.pth")
        # Resize/preprocessing stays on CPU even when the frozen model uses CUDA.
        torch.set_num_threads(threads)
        source_path = self.source / "model/tactile_mae.py"
        tree = ast.parse(source_path.read_text())
        # These old documentation decorators reference removed constants. Removing only
        # documentation and the unused positional-embedding import preserves computation.
        tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == "model.util.pos_embed")]
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node.decorator_list = [d for d in node.decorator_list if not (
                    isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id in
                    {"add_start_docstrings_to_model_forward", "replace_return_docstrings"})]
        namespace = {"__name__": "contact_belief_wam._original_anytouch", "CLIPVisionTransformer": CLIPVisionTransformer}
        exec(compile(ast.fix_missing_locations(tree), str(source_path), "exec"), namespace)
        cfg = AutoConfig.from_pretrained(self.source / "CLIP-B-16/config.json", local_files_only=True)
        cfg.vision_config._attn_implementation = "eager"
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = namespace["TactileVideoMAE"](SimpleNamespace(mask_ratio=0.0), cfg, 4, 1)
        original = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        original = original.get("model", original)
        weights = {k.replace("touch_mae_model.", ""): v for k, v in original.items()
                   if "touch_mae_model." in k and "decoder" not in k and "mask_token" not in k}
        if not weights:
            weights = {k: v for k, v in original.items() if k in self.model.state_dict()}
        missing = set(self.model.state_dict()) - set(weights)
        # This is a deterministic position index, not a learned missing parameter.
        if missing - {"new_position_ids"}:
            raise RuntimeError(f"Missing frozen AnyTouch2 parameters: {sorted(missing)}")
        for key in missing:
            weights[key] = self.model.state_dict()[key]
        self.model.load_state_dict(weights, strict=True)
        self.model.requires_grad_(False).eval().to(device)
        self.device = device
        self.metadata = {"encoder": "AnyTouch2", "frames": 4, "sensor_id": 3,
                         "source_sha256": sha256(source_path), "checkpoint_sha256": sha256(self.checkpoint),
                         "preprocess": PREPROCESS_VERSION, "offsets": list(TOUCH_OFFSETS),
                         "pool": "projected-patches-temporal-mean-spatial-2x2", "shape": [8, 512]}

    def encode_preprocessed(self, windows):
        """[B,2,4,3,224,224] -> [B,8,512]."""
        import torch
        import torch.nn.functional as F
        x = windows.to(self.device)
        b = x.shape[0]
        with torch.inference_mode():
            features = self.model(x.flatten(0, 1), torch.full((b * 2,), 3, dtype=torch.long, device=self.device))
            if features.shape[1:] != (398, 512):
                raise ValueError(f"Unexpected AnyTouch2 projection shape: {features.shape}")
            patches = features[:, 6:].reshape(b * 2, 2, 14, 14, 512).mean(1)
            pooled = F.adaptive_avg_pool2d(patches.permute(0, 3, 1, 2), (2, 2)).flatten(2).transpose(1, 2)
            return pooled.reshape(b, 8, 512).float().cpu().numpy()

    def __call__(self, windows):
        return self.encode_preprocessed(tactile_preprocess(windows))


def tactile_keys(handle):
    result = []
    for side in ("left", "right"):
        key = next((f"tactile/{side}_{name}/rgb_marker" for name in ("gsmini", "tactile")
                    if f"tactile/{side}_{name}/rgb_marker" in handle), None)
        if key is None:
            raise KeyError(f"Missing {side} rgb_marker tactile")
        result.append(key)
    return result


def cache_episodes(rows, data_root, cache_root, encoder, batch_size=8):
    import h5py
    data_root, cache_root = Path(data_root), Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    atomic_json(cache_root / "encoder.json", encoder.metadata)
    for row in rows:
        dest = cache_root / (row["episode_id"] + ".npz")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            with np.load(dest) as old:
                if (json.loads(str(old["metadata"])) == encoder.metadata and
                    old["tokens"].shape == (row["frames"], 8, 512)):
                    print(json.dumps({"cache": row["episode_id"], "status": "reused"}), flush=True)
                    continue
            raise ValueError(f"Stale tactile cache: {dest}")
        with h5py.File(data_root / row["path"], "r") as handle:
            keys = tactile_keys(handle)
            rgb = [[decode_rgb(frame) for frame in handle[key][:]] for key in keys]
            starts = np.zeros(row["frames"], np.int64)
            for start, stop in row["segments"]:
                starts[start:stop] = start
            outputs = []
            for offset in range(0, row["frames"], batch_size):
                frames = range(offset, min(row["frames"], offset + batch_size))
                windows = np.stack([np.stack([np.stack([rgb[side][j] for j in window_indices(i, starts[i])])
                                             for side in range(2)]) for i in frames])
                outputs.append(encoder(windows).astype(np.float16))
            tokens = np.concatenate(outputs)
            temporary = dest.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, tokens=tokens, raw_step=handle["step"][:],
                                frame_index=np.arange(row["frames"]), metadata=json.dumps(encoder.metadata))
            temporary.replace(dest)
        print(json.dumps({"cache": row["episode_id"], "status": "encoded", "frames": len(tokens)}), flush=True)
