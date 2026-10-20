"""Real-batch engineering checks. Never runs the two formal training stages."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from .cloud import model_config
from .data import ContactDataset, collate_contact
from .paths import atomic_json
from .preprocessing import PromptTokenizer
from .sequence import SequenceTrainingModel, observation_at, prepare_batch
from .trainer import export_model, optimizer_for, validation


def run_debug(run, *, updates=1, stages=("stage1", "stage2"), device="cpu", threads=16):
    from .weights import initialize
    from .model import ContactWAM
    if not 1 <= updates <= 2:
        raise ValueError("Developer debug is limited to one or two optimizer updates per stage")
    torch.set_num_threads(threads)
    torch.manual_seed(run["seed"])
    output = Path(run["output"])
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = PromptTokenizer(run["tokenizer_path"])
    report = {"task": run["task"], "device": device, "formal_training": False, "stages": {}, "complete": False}
    previous = output / "debug_report.json"
    if previous.exists():
        prior = json.loads(previous.read_text())
        for stage in stages:
            if stage in prior.get("stages", {}):
                raise ValueError(f"Debug {stage} already completed in {output}; do not repeat optimizer updates")
        report["stages"] = prior.get("stages", {})
    origin = Path(run["pi05_base"]) / "conversion.json"
    if origin.is_file():
        report["base_weight_provenance"] = json.loads(origin.read_text())
    stage1 = output / "stage1/pretrained_model"
    if not (stage1 / "model.safetensors").is_file():
        stage1 = None
    try:
        for stage in stages:
            if stage == "stage1" and run["mode"] not in {"full", "wave0"}:
                continue
            cfg = model_config(run, stage)
            dataset = ContactDataset(run["manifest"], run["data_root"], run["tactile_cache"], run["stats"], cfg)
            eligible = [i for i, (_, start, stop, frame) in enumerate(dataset.index)
                        if frame - start >= 4 * cfg.k and frame + cfg.k < stop]
            if not eligible:
                raise ValueError("No real sequence supports the requested debug history")
            # Include both sources when available, without substituting synthetic data.
            choices = []
            for source in ("original", "recovery"):
                selected = [i for i in eligible if dataset.sources[i] == source]
                if selected:
                    choices.append(selected[len(selected) // 2])
            # Force a long enough sampled history to exercise the actual two-step graph.
            raw = collate_contact([dataset[(index, 0)] for index in choices])
            batch = prepare_batch(raw, tokenizer, device)
            started = time.monotonic()
            policy, initialization = initialize(cfg, world_init=run["world_init"], stage1_checkpoint=stage1 if stage == "stage2" else None)
            policy.to(device).train()
            wrapper = SequenceTrainingModel(policy)
            optimizer = optimizer_for(policy, run)
            frozen_versions = {name: parameter._version for name, parameter in policy.named_parameters() if not parameter.requires_grad}
            autocast = lambda: torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                                              enabled=cfg.dtype == "bfloat16")
            traces = []
            for number in range(updates):
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    result = wrapper(batch)
                if not torch.isfinite(result["loss"]):
                    raise FloatingPointError("Nonfinite debug loss")
                result["loss"].backward()
                active_full = cfg.mode == "full" and all(getattr(cfg, name) for name in (
                    "belief_to_latent_queries", "belief_to_world_state", "belief_to_future_corrector",
                    "belief_to_contact_predictor", "belief_to_expert"))
                missing_contact = [name for name, parameter in policy.named_parameters()
                                   if name.startswith("contact.") and parameter.requires_grad and parameter.grad is None]
                if active_full and missing_contact:
                    raise AssertionError(f"Active contact parameters missing gradients: {missing_contact}")
                gradients = {}
                for name, parameter in policy.named_parameters():
                    if parameter.grad is None:
                        continue
                    if not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"Nonfinite gradient: {name}")
                    group = name.split(".")[1] if name.startswith("contact.") else "policy_or_world"
                    gradients[group] = gradients.get(group, 0.) + float(parameter.grad.float().square().sum())
                norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), run["grad_clip"])
                optimizer.step()
                policy.optimizer_step.add_(1)
                changed_frozen = [name for name, parameter in policy.named_parameters()
                                  if name in frozen_versions and parameter._version != frozen_versions[name]]
                if changed_frozen:
                    raise AssertionError(f"Frozen parameters updated: {changed_frozen[:5]}")
                traces.append({"loss": float(result["loss"]), "grad_norm": float(norm),
                               "gradient_squared_norm_by_module": gradients, "frozen_parameter_count": len(frozen_versions),
                               "missing_contact_gradients": missing_contact})
                del result
            checkpoint = output / stage / "pretrained_model"
            policy.eval()
            export_model(policy, checkpoint, run)
            torch.save({"optimizer": optimizer.state_dict(), "step": int(policy.optimizer_step),
                        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state()}, checkpoint.parent / "debug_training.pt")
            obs = observation_at(batch, batch["state"].shape[1] - 1)
            noise = torch.randn(len(choices), cfg.chunk_size, 32, device=device)
            if stage == "stage2":
                with autocast():
                    before = policy.predict_action_chunk(obs, noise=noise).actions.cpu()
            else:
                with torch.no_grad(), autocast():
                    history = policy.history_step(obs, batch["actions"][:, -1])
                    before = torch.cat([v.flatten(1) for v in history if v is not None], 1).cpu()
            del wrapper, optimizer, policy
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            restored = ContactWAM.from_pretrained(checkpoint, stage=stage, device=device).eval()
            optimizer_restored = optimizer_for(restored, run)
            state = torch.load(checkpoint.parent / "debug_training.pt", map_location=device, weights_only=False)
            optimizer_restored.load_state_dict(state["optimizer"])
            if state["step"] != int(restored.optimizer_step):
                raise AssertionError("Checkpoint step mismatch")
            with torch.no_grad(), autocast():
                if stage == "stage2":
                    after = restored.predict_action_chunk(obs, noise=noise).actions.cpu()
                else:
                    history = restored.history_step(obs, batch["actions"][:, -1])
                    after = torch.cat([v.flatten(1) for v in history if v is not None], 1).cpu()
            difference = float((before - after).abs().max())
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            val_dataset = ContactDataset(run["manifest"], run["data_root"], run["tactile_cache"], run["stats"], cfg, split="val")
            validation_values = validation(SequenceTrainingModel(restored), val_dataset, tokenizer, device, 2, 1, autocast)
            report["stages"][stage] = {"updates": traces, "checkpoint_max_abs_difference": difference,
                "optimizer_restored": True, "initialization": initialization, "sources": raw["source"],
                "validation": validation_values, "seconds": time.monotonic() - started}
            atomic_json(output / "debug_report.json", report)
            if stage == "stage1":
                stage1 = checkpoint
            del restored, optimizer_restored, state, batch, raw
            gc.collect()
        report["complete"] = True
    except Exception:
        import traceback
        report["error"] = traceback.format_exc()
        raise
    finally:
        atomic_json(output / "debug_report.json", report)
    return report
