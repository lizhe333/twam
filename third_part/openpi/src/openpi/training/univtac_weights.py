"""Strict loading: only newly introduced LoRA leaves may be absent in the base."""
from dataclasses import dataclass
import flax.traverse_util as traversal
import numpy as np
from openpi.models import model


@dataclass(frozen=True)
class StrictBaseLoader:
    params_path: str

    def load(self, params):
        expected = traversal.flatten_dict(params, sep="/")
        restored = model.restore_params(self.params_path, restore_type=np.ndarray)
        found = traversal.flatten_dict(restored, sep="/")
        extra = set(found) - set(expected)
        missing = set(expected) - set(found)
        forbidden = {key for key in missing if "lora" not in key.lower()}
        if extra or forbidden:
            raise ValueError(f"Checkpoint mismatch; extra={sorted(extra)}, missing={sorted(forbidden)}")
        for key, value in found.items():
            if value.shape != expected[key].shape:
                raise ValueError(f"Checkpoint shape mismatch at {key}: {value.shape} != {expected[key].shape}")
        merged = {key: found[key].astype(ref.dtype, copy=False) if key in found else ref for key, ref in expected.items()}
        return traversal.unflatten_dict(merged, sep="/")
