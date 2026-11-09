from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def project_root():
    candidates = [Path(os.environ["WAVE0_ROOT"])] if "WAVE0_ROOT" in os.environ else list(Path(__file__).resolve().parents)
    for path in candidates:
        if (path / "src/lerobot/policies/wave0/modeling_wave0.py").is_file() and (path / "configs/univtac").is_dir():
            return path.resolve()
    raise RuntimeError("Set WAVE0_ROOT to the integrated WAVE-0 repository")


def workspace_root():
    if "TWAM_ROOT" in os.environ:
        return Path(os.environ["TWAM_ROOT"]).resolve()
    return next((p for p in project_root().parents if (p / "UniVTAC").is_dir()), project_root().parent)


def configure():
    root = project_root()
    os.environ.setdefault("HF_HOME", str(root / ".cache/huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return root


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
