"""Convert the local, unfinetuned official base for CPU debug while HF downloads.

This uses OpenPI's published tensor conversion functions. It is NOT the old
task-finetuned JAX policy and does not replace the cloud lerobot/pi05_base asset.
"""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import torch
from safetensors.torch import save_file
from contact_belief_wam.paths import PROJECT, WORKSPACE, atomic_json, sha256


def main():
    torch.set_num_threads(4)
    source = WORKSPACE / "models/pi05_base"
    script = WORKSPACE / "openpi/examples/convert_jax_model_to_pytorch.py"
    destination = PROJECT / "models/pi05_base_local_debug"
    if (destination / "model.safetensors").exists():
        # Compatibility completion for the first published tensor-only conversion.
        from safetensors.torch import load_file
        weights = load_file(str(destination / "model.safetensors"))
        report = json.loads((destination / "conversion.json").read_text())
        complete_lm_heads(weights, destination, report)
        return
    spec = importlib.util.spec_from_file_location("official_openpi_conversion", script)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    print("Restoring official unfinetuned JAX pi05_base on CPU", flush=True)
    params = converter.slice_initial_orbax_checkpoint(str(source), restore_precision="float32")
    pg_config = SimpleNamespace(
        vision_config=SimpleNamespace(hidden_size=1152, num_hidden_layers=27, num_attention_heads=16,
                                      intermediate_size=4304, patch_size=14, projection_dim=2048),
        text_config=SimpleNamespace(hidden_size=2048, num_hidden_layers=18, num_attention_heads=8,
                                    head_dim=256, intermediate_size=16384))
    print("Mapping published PaliGemma and action-expert tensors", flush=True)
    weights, expert = converter.slice_paligemma_state_dict(params["paligemma_params"], pg_config)
    weights.update(converter.slice_gemma_state_dict(
        expert, converter.openpi.models.gemma.get_config("gemma_300m"),
        num_expert=1, checkpoint_dir=str(source), pi05=True))
    for key in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        for source_key, output_key in (("kernel", "weight"), ("bias", "bias")):
            value = params["projection_params"][key][source_key]
            if isinstance(value, dict):
                value = value["value"]
            tensor = torch.from_numpy(np.asarray(value))
            weights[f"{key}.{output_key}"] = tensor.T if output_key == "weight" else tensor
    # Independent contiguous storage also makes duplicated tied-weight tensors safe.
    weights = {key: value.contiguous().clone() for key, value in weights.items()}
    destination.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(destination / "model.safetensors"))
    report = {"source": str(source), "source_kind": "official unfinetuned JAX pi05_base",
              "purpose": "pretrained CPU debug fallback; formal cloud default remains lerobot/pi05_base",
              "converter": str(script), "converter_sha256": sha256(script),
              "tensor_count": len(weights), "dtype": "float32",
              "checkpoint_sha256": sha256(destination / "model.safetensors"),
              "strict_wave0_loading": "performed separately, never assumed by this conversion"}
    atomic_json(destination / "conversion.json", report)
    complete_lm_heads(weights, destination, report)
    print(json.dumps(report), flush=True)


def complete_lm_heads(weights, destination, report):
    """Match the upstream PyTorch container's unused language output heads.

    JAX has no action-expert vocabulary head: actions use action_out_proj.
    OpenPI's official full-model converter leaves that unused head initialized.
    Preserve that distinction in provenance; every policy-used tensor is mapped.
    """
    expert_key = "paligemma_with_expert.gemma_expert.lm_head.weight"
    embed_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    head_key = "paligemma_with_expert.paligemma.lm_head.weight"
    if expert_key in weights and head_key in weights:
        print("PyTorch container heads already complete", flush=True)
        return
    weights[head_key] = weights.pop(embed_key)
    torch.manual_seed(42)
    weights[expert_key] = torch.empty(257152, 1024).normal_(std=.02)
    path = destination / "model.safetensors"
    tmp = path.with_suffix(".tmp")
    save_file(weights, str(tmp))
    tmp.replace(path)
    report.update(tensor_count=len(weights), checkpoint_sha256=sha256(path),
                  alias_conversion="PaliGemma embed_tokens stored under tied lm_head alias",
                  initialized_unused_tensors=[expert_key],
                  unused_head_reason="JAX action expert has no vocabulary head; official PyTorch converter also initializes this unused container tensor. Flow actions read action_out_proj, never lm_head.")
    atomic_json(destination / "conversion.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
