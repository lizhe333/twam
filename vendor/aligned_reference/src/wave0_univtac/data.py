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

from .config import IMAGE, FUTURE
from .runtime import digest


def resize(image):
    """OpenPI client PIL bilinear resize-with-pad, including integer rounding."""
    image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    w, h = image.size
    ratio = max(w / 224, h / 224)
    rw, rh = int(w / ratio), int(h / ratio)
    result = Image.new(image.mode, (224, 224), 0)
    result.paste(image.resize((rw, rh), Image.Resampling.BILINEAR), ((224-rw)//2, (224-rh)//2))
    return np.asarray(result).copy()


def decode(encoded):
    image = cv2.imdecode(np.frombuffer(bytes(encoded), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid UniVTAC JPEG")
    return resize(image)  # Preserve numerical RGB order; no OpenCV channel swap.


class Processor:
    def __init__(self, stats, tokenizer, prompt):
        self.stats = json.loads(Path(stats).read_text())["norm_stats"]
        self.sp = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer))
        self.prompt = prompt.strip().replace("_", " ").replace("\n", " ")

    def normalize(self, value, key):
        # Match OpenPI exactly: add epsilon to the range, do not clamp to epsilon.
        q = self.stats[key]
        lo, hi = np.asarray(q["q01"]), np.asarray(q["q99"])
        return (np.asarray(value) - lo) / (hi - lo + 1e-6) * 2 - 1

    def tokens(self, normalized_state):
        # Baseline tokenizes eight state entries BEFORE padding to 32.
        bins = np.digitize(normalized_state, np.linspace(-1, 1, 257)[:-1]) - 1
        text = f"Task: {self.prompt}, State: {' '.join(map(str, bins))};\nAction: "
        ids = self.sp.encode(text, add_bos=True)
        if len(ids) > 200:
            raise ValueError("Baseline task prompt exceeds 200 tokens")
        return np.asarray(ids + [0]*(200-len(ids)), np.int64), np.arange(200) < len(ids)

    def observation(self, image, state):
        state = self.normalize(np.asarray(state, np.float32), "state")
        ids, mask = self.tokens(state)
        return {IMAGE: torch.from_numpy(resize(image).transpose(2, 0, 1)).float()/255,
                "observation.state": torch.from_numpy(state.astype(np.float32)),
                "observation.language.tokens": torch.from_numpy(ids),
                "observation.language.attention_mask": torch.from_numpy(mask)}

    def actions(self, absolute, state):
        delta = np.asarray(absolute, np.float32).copy()
        delta[..., :7] -= np.asarray(state, np.float32)[:7]
        result = self.normalize(delta, "actions").astype(np.float32)
        return np.pad(result, ((0, 0), (0, 24)))

    def absolute(self, normalized, state):
        value = np.asarray(normalized)[..., :8]
        q = self.stats["actions"]
        lo, hi = np.asarray(q["q01"]), np.asarray(q["q99"])
        result = (value+1)/2 * (hi-lo+1e-6) + lo
        result[..., :7] += np.asarray(state)[:7]
        return result.astype(np.float32)


class UniVTACDataset(Dataset):
    def __init__(self, run, *, episode_ids=None, stats=None, valid_future_only=False):
        self.run = run
        baseline = Path(run["baseline"])
        self.segments = json.loads((baseline / "segments.json").read_text())
        if episode_ids is not None:
            self.segments = [s for s in self.segments if s["source_episode_id"] in episode_ids]
        self.prompt = json.loads((baseline / "instruction.json").read_text())["seen"][0]
        self.processor = Processor(stats or baseline / "norm_stats.json", run["tokenizer"], self.prompt)
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
        batch = self.processor.observation(image, state)
        batch.update({FUTURE: torch.from_numpy(future.transpose(2, 0, 1)).float()/255,
                      IMAGE+"_is_pad": torch.tensor([False, frame+10 >= stop]),
                      "action": torch.from_numpy(self.processor.actions(target, state)),
                      "world_model_task": self.run["task"], "sample_index": index})
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
