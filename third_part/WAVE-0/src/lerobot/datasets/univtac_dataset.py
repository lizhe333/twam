from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image
import sentencepiece
import torch
from torch.utils.data import Dataset, Sampler

from lerobot.configs.univtac import IMAGE, FUTURE, WRIST, FUTURE_WRIST, camera_keys
from lerobot.utils.univtac_runtime import digest


from lerobot.policies.wave0.processor_wave0 import UniVTACProcessor as Processor
from lerobot.policies.wave0.image_processing_wave0 import resize, decode

class UniVTACDataset(Dataset):
    def __init__(self, run, *, episode_ids=None, stats=None, valid_future_only=False):
        self.run = run
        baseline = Path(run["baseline"])
        self.segments = json.loads((baseline / "segments.json").read_text())
        if episode_ids is not None:
            self.segments = [s for s in self.segments if s["source_episode_id"] in episode_ids]
        self.prompt = json.loads((baseline / "instruction.json").read_text())["seen"][0]
        self.processor = Processor(stats or baseline / "norm_stats.json", run["tokenizer"], self.prompt,
                                   run.get("camera_mode", "head_only"))
        self.index = [(i, t) for i, s in enumerate(self.segments)
                      for t in range(s["start_frame"], s["stop_frame"]-(10 if valid_future_only else 1))]
        self.handles = OrderedDict()

    def __len__(self):
        return len(self.index)

    def _file(self, segment):
        path = str(Path(self.run["workspace"]) / segment["source_path"])
        if path not in self.handles:
            self.handles[path] = h5py.File(path, "r")
        self.handles.move_to_end(path)
        while len(self.handles) > 8:
            self.handles.popitem(last=False)[1].close()
        return self.handles[path]

    def __getitem__(self, index):
        segment_id, frame = self.index[index]
        s = self.segments[segment_id]
        f = self._file(s)
        stop = s["stop_frame"]
        state = np.asarray(f["embodiment/joint"][frame, :8], np.float32)
        target = np.asarray(f["embodiment/joint"][frame+1:min(frame+51, stop), :8], np.float32)
        target = np.concatenate([target, np.repeat(target[-1:], 50-len(target), axis=0)])
        image = decode(f["observation/head/rgb"][frame])
        future_frame = min(frame+10, stop-1)
        future = decode(f["observation/head/rgb"][future_frame])
        wrist = decode(f["observation/wrist/rgb"][frame]) if WRIST in self.processor.camera_keys else None
        batch = self.processor.observation(image, state, wrist_rgb=wrist)
        batch.update({FUTURE: torch.from_numpy(future.transpose(2, 0, 1)).float()/255,
                      IMAGE+"_is_pad": torch.tensor([False, frame+10 >= stop]),
                      "action": torch.from_numpy(self.processor.actions(target, state)),
                      "world_model_task": self.run["task"], "sample_index": index})
        if wrist is not None:
            future_wrist = decode(f["observation/wrist/rgb"][future_frame])
            batch[FUTURE_WRIST] = torch.from_numpy(future_wrist.transpose(2, 0, 1)).float()/255
            batch[WRIST+"_is_pad"] = batch[IMAGE+"_is_pad"].clone()
        return batch

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def __getstate__(self):
        return {**self.__dict__, "handles": OrderedDict()}

class ReplaySampler(Sampler):
    """Epoch permutations partitioned by update/rank, exactly replayable after resume."""
    def __init__(self, size, seed, global_batch, world_size, rank, start_step, stop_step):
        if global_batch % world_size:
            raise ValueError("Global batch must divide world size")
        self.size, self.seed = size, seed
        self.batch, self.world, self.rank = global_batch, world_size, rank
        self.start, self.stop = start_step, stop_step

    def __iter__(self):
        cached_epoch, order = -1, None
        local = self.batch // self.world
        for step in range(self.start, self.stop):
            for position in range(step*self.batch+self.rank*local, step*self.batch+(self.rank+1)*local):
                epoch, offset = divmod(position, self.size)
                if epoch != cached_epoch:
                    order = np.random.default_rng(np.random.SeedSequence([self.seed, epoch])).permutation(self.size)
                    cached_epoch = epoch
                yield int(order[offset])

    def __len__(self):
        return (self.stop-self.start)*self.batch//self.world

def audit_data(run, verify_hashes=False):
    baseline = Path(run["baseline"])
    identity = json.loads((baseline / "run_identity.json").read_text())
    if digest(baseline / "norm_stats.json") != identity["hashes"]["norm_stats"]:
        raise ValueError("Normalization differs from the recorded head-only PI0.5 checkpoint")
    ds = UniVTACDataset(run)
    report = {"task": run["task"], "episodes": len({s["source_episode_id"] for s in ds.segments}),
              "segments": len(ds.segments), "pairs": len(ds),
              "stats_match_baseline": True, "full_source_hashes_checked": verify_hashes,
              "camera_mode": run.get("camera_mode", "head_only"),
              "future_offset_frames": 10, "official_inclusive_action_horizon": 9,
              "tokenized_state_dim": 8, "flow_loss_dim": 32, "samples": []}
    if sorted({s["source_episode_id"] for s in ds.segments}) != list(range(100)):
        raise ValueError("Expected exactly the baseline's 100 episodes")
    for s in ds.segments:
        p = Path(run["workspace"]) / s["source_path"]
        if p.stat().st_size != s["source_size_bytes"]:
            raise ValueError(f"Raw data size changed: {p}")
        if verify_hashes and digest(p) != s["source_sha256"]:
            raise ValueError(f"Raw data hash changed: {p}")
        with h5py.File(p, "r") as f:
            for key in camera_keys(run):
                view = key.rsplit(".", 1)[-1]
                if len(f[f"observation/{view}/rgb"]) < s["stop_frame"]:
                    raise ValueError(f"Missing aligned {view} frames: {p}")
            step = f["step"][s["start_frame"]:s["stop_frame"]]
            if len(step) != s["valid_count"]+1 or not np.all(np.diff(step)==2):
                raise ValueError(f"Raw timeline changed: {p}")
    offset = 0
    for s in ds.segments:
        for i in (offset, offset+s["valid_count"]-1):
            sample = ds[i]
            if not all(torch.isfinite(v).all() for v in sample.values() if isinstance(v, torch.Tensor)):
                raise ValueError("Nonfinite data")
        offset += s["valid_count"]
    for i in (0, len(ds)//2, len(ds)-1):
        sample = ds[i]
        report["samples"].append({"index": i, "action_shape": list(sample["action"].shape),
                                  "future_valid": not bool(sample[IMAGE+"_is_pad"][1])})
    ds.close()
    return report
