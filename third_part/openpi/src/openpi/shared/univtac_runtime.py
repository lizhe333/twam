"""Relocatable paths and process-local cache settings; import before JAX/LeRobot."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


def root() -> Path:
    return Path(os.environ.get("TWAM_ROOT", Path(__file__).resolve().parents[4])).resolve()


def configure() -> Path:
    base = root()
    settings = {
        "TWAM_ROOT": str(base),
        "HF_HOME": str(base / ".cache/huggingface"),
        "HF_LEROBOT_HOME": str(base / "datasets"),
        "OPENPI_DATA_HOME": str(base / "models/pi05_base"),
        "XDG_CACHE_HOME": str(base / ".cache"),
        "JAX_COMPILATION_CACHE_DIR": str(base / ".cache/jax"),
        "UV_CACHE_DIR": str(base / ".cache/uv"),
        "TORCH_HOME": str(base / ".cache/torch"),
        "WANDB_MODE": "offline",
        "WANDB_DIR": str(base / "outputs/wandb"),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2",
    }
    for key, value in settings.items():
        os.environ.setdefault(key, value)
    # The pinned LeRobot version deliberately rejects this deprecated alias.
    os.environ.pop("LEROBOT_HOME", None)
    return base


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        tmp = Path(f.name)
    tmp.replace(path)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]
