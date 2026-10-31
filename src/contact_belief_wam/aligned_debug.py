"""Bounded real-batch checks for the aligned Contact-Belief pipeline.

This is a developer debug entry, not a training launcher. It caps each model
at two updates, keeps its settings in a separate output tree, and checks the
same recurrent model, optimizer, EMA, and checkpoint paths used by training.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
import yaml

from .data import collate_contact
from .paths import atomic_json


def validate_debug_updates(updates: int) -> int:
    """Reject anything beyond the approved one- or two-update debug budget."""
    if isinstance(updates, bool) or int(updates) != updates or not 1 <= int(updates) <= 2:
        raise ValueError("Aligned debug is limited to one or two updates per model")
    return int(updates)


def validate_debug_micro_batch(micro_batch: int) -> int:
    if isinstance(micro_batch, bool) or int(micro_batch) != micro_batch or int(micro_batch) not in {1, 2}:
        raise ValueError("Aligned debug micro-batch must be 1 or 2")
    return int(micro_batch)


def select_debug_groups(tasks, sources, episode_ids, *, stage: str, task: str | None = None,
                        shared_tasks=(), eligible_indices=None):
    """Select task-preserving, source-labeled real examples for each stage.

    Stage 1 puts two original examples from distinct episodes of one task in a
    single forward so the ranking objective has a same-task negative. A
    recovery example from that task is a separate micro-batch. Stage 2 uses one
    original and one recovery example for its task, also as separate
    micro-batches so both sources are visible in the trace.
    """
    if not (len(tasks) == len(sources) == len(episode_ids)):
        raise ValueError("Aligned debug task/source/episode metadata lengths differ")
    indices = list(range(len(tasks))) if eligible_indices is None else [int(i) for i in eligible_indices]
    if any(i < 0 or i >= len(tasks) for i in indices):
        raise ValueError("Aligned debug candidate index is outside dataset metadata")

    def episodes_for(label, source):
        episodes = {}
        for index in indices:
            if tasks[index] == label and sources[index] == source:
                episodes.setdefault(str(episode_ids[index]), []).append(index)
        # A middle window avoids always testing the episode boundary.
        return {episode: values[len(values) // 2] for episode, values in episodes.items()}

    if stage in {"tactile", "stage1"}:
        candidates = list(shared_tasks)
        if task and task not in candidates:
            candidates.insert(0, task)
        for label in candidates:
            originals = episodes_for(label, "original")
            recovery = episodes_for(label, "recovery")
            if len(originals) >= 2 and recovery:
                chosen = sorted(originals.items())[:2]
                recovery_episode, recovery_index = sorted(recovery.items())[0]
                return [
                    {"source": "original", "task": label,
                     "episode_ids": [item[0] for item in chosen],
                     "indices": [item[1] for item in chosen]},
                    {"source": "recovery", "task": label,
                     "episode_ids": [recovery_episode], "indices": [recovery_index]},
                ]
        raise ValueError("Stage-1 debug needs two same-task original episodes and one recovery episode")
    if stage == "stage2":
        if not task:
            raise ValueError("Stage-2 debug requires its policy task")
        result = []
        for source in ("original", "recovery"):
            episodes = episodes_for(task, source)
            if not episodes:
                if source == "recovery":
                    continue
                raise ValueError(f"Stage-2 debug has no {source} episode for {task}")
            episode, index = sorted(episodes.items())[0]
            result.append({"source": source, "task": task,
                           "episode_ids": [episode], "indices": [index]})
        return result
    raise ValueError(f"Unknown aligned debug stage: {stage}")


def debug_settings(run, *, updates, device, optimizer_device, micro_batch, threads):
    """Create a report-only copy; never alter the formal recipe passed by caller."""
    updates = validate_debug_updates(updates)
    micro_batch = validate_debug_micro_batch(micro_batch)
    if optimizer_device not in {"cpu", "cuda"}:
        raise ValueError("Debug optimizer device must be cpu or cuda")
    if int(threads) < 1:
        raise ValueError("Debug thread count must be positive")
    recipe = copy.deepcopy(run)
    return {
        "formal_training": False,
        "variant": recipe.get("variant"),
        "mode": recipe.get("mode"),
        "updates_per_model": updates,
        "device": str(device),
        "optimizer_device": optimizer_device,
        "micro_batch": micro_batch,
        "threads": int(threads),
        "ema_device": "cpu" if optimizer_device == "cpu" else str(device),
        "gradient_checkpointing": bool(recipe.get("gradient_checkpointing", True)),
        "formal_recipe": {
            "stage1": copy.deepcopy(recipe.get("stage1", {})),
            "stage2": copy.deepcopy(recipe.get("stage2", {})),
            "ema_decay": recipe.get("ema_decay", 0.99),
            "global_data_mix": recipe.get("data_mix", "1:1"),
        },
    }


def reject_formal_output(path):
    """Prevent a bounded debug from being placed in a formal output tree."""
    root = Path(path)
    markers = ("train_status.json", "pipeline_config.yaml", "run_config.yaml")
    found = []
    if root.exists():
        found = [str(candidate) for marker in markers for candidate in root.rglob(marker)]
    if found:
        raise ValueError("Aligned debug output contains formal training state: " + ", ".join(found))


def _atomic_yaml(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        yaml.safe_dump(value, handle, sort_keys=False)
        temporary = Path(handle.name)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sample_identity(dataset, index):
    rid = int(dataset.index[index][0])
    row = dataset.rows[rid]
    return str(row["episode_id"])


def _eligible_indices(dataset, config):
    required_history = max(1, int(config.belief_bptt_steps))
    # WAVE-0 recovery has a one-step belief path and no historical warmup. Do
    # not exclude its first action window for a history it never consumes.
    minimum_past = max(0, required_history - 1) * int(config.k)
    eligible = []
    for index, (_rid, start, stop, frame) in enumerate(dataset.index):
        if frame - start < minimum_past or frame + config.k >= stop:
            continue
        eligible.append(index)
    return eligible


def _materialize_sample(dataset, index, seed, *, minimum_sequence_length, require_policy_target):
    """Choose a deterministic history draw with actual recurrent steps and labels."""
    for offset in range(2048):
        sample = dataset[(int(index), int((seed + offset) % (2**32)))]
        sequence_length = int(np.asarray(sample["sequence_mask"], dtype=np.bool_).sum())
        if sequence_length < minimum_sequence_length:
            continue
        if require_policy_target:
            target = np.asarray(sample["action_mask"], dtype=np.bool_)
            policy = np.asarray(sample.get("policy_action_mask", target), dtype=np.bool_)
            if not np.any(target & policy):
                continue
        for key in ("state", "world_state", "actions", "world_actions"):
            if key not in sample:
                raise ValueError(f"Aligned sample is missing required normalization field {key!r}")
            if not np.isfinite(np.asarray(sample[key])).all():
                raise FloatingPointError(f"Aligned sample has nonfinite {key}")
        return sample, {"index": int(index), "episode_id": _sample_identity(dataset, index),
                       "source": sample["source"], "task": sample["task"],
                       "valid_sequence_steps": sequence_length, "draw_seed": int((seed + offset) % (2**32))}
    raise ValueError(f"No debug window at dataset index {index} has enough recurrent steps and labels")


def _minimum_sequence_length(config):
    """Require enough sequence positions for the active truncated BPTT path."""
    return max(1, int(config.belief_bptt_steps))


def _make_groups(dataset, stage, task, tasks, config, seed, micro_batch):
    eligible = _eligible_indices(dataset, config)
    episode_ids = [_sample_identity(dataset, i) for i in range(len(dataset))]
    groups = select_debug_groups(dataset.tasks, dataset.sources, episode_ids,
                                 stage=stage, task=task, shared_tasks=tasks,
                                 eligible_indices=eligible)
    if stage == "stage1" and len(groups[0]["indices"]) != 2:
        raise AssertionError("Stage-1 ranking micro-batch must contain a same-task pair")
    if stage == "stage1" and micro_batch < 2:
        raise ValueError("Stage-1 aligned debug needs --micro-batch 2 to exercise same-task ranking")
    return groups


def _detach_metrics(result):
    return {"loss": result["loss"].detach().cpu(),
            "metrics": {name: {part: tensor.detach().cpu() for part, tensor in value.items()}
                        for name, value in result.get("metrics", {}).items()}}


class _LossAdapter(nn.Module):
    """Make the sequence mapping compatible with the reference offload helper."""
    def __init__(self, sequence_model):
        super().__init__()
        self.sequence_model = sequence_model

    def forward(self, batch):
        result = self.sequence_model(batch)
        return result["loss"], _detach_metrics(result)


class _ForwardBackward(nn.Module):
    """CPU-master training adapter used only by the bounded CUDA debug."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch, backward_scale):
        loss, metrics = self.policy(batch)
        if backward_scale is not None:
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite aligned debug loss")
            (loss / backward_scale).backward()
            loss = loss.detach()
        return loss, metrics


def _functional_call_cpu_masters(module, args, device, *, dtype=torch.bfloat16,
                                 parameter_overrides=None, save_activations=True):
    """Run a CPU-master module through temporary CUDA weights.

    This mirrors the reference project's debug-only offload path without adding
    a runtime dependency on that separate project.
    """
    if any(parameter.device.type != "cpu" for parameter in module.parameters()):
        raise ValueError("Aligned debug offload requires CPU master parameters")
    device = torch.device(device)
    overrides = parameter_overrides or {}
    parameters = {}
    for name, parameter in module.named_parameters():
        source = overrides.get(name, parameter)
        target_dtype = dtype if source.is_floating_point() else source.dtype
        parameters[name] = source.to(device=device, dtype=target_dtype, copy=True)
    named_parameters = list(module.named_parameters(remove_duplicate=False))
    original_buffers = list(module.named_buffers(remove_duplicate=False))
    buffers_original = dict(module.named_buffers())
    buffers = {name: value.to(device=device, copy=True) for name, value in buffers_original.items()}
    try:
        saved = (torch.autograd.graph.save_on_cpu(pin_memory=True)
                 if save_activations and device.type == "cuda" else nullcontext())
        # save_on_cpu can return the padded attention mask with its compact
        # stride. PyTorch's memory-efficient SDPA backward requires an aligned
        # row stride, so keep math/flash available and exclude that backend in
        # this debug-only execution context. Formal training is untouched.
        sdpa = _debug_sdpa_context(device)
        with sdpa, saved:
            return torch.func.functional_call(module, (parameters, buffers), args,
                                              tie_weights=True, strict=True)
    finally:
        # Explicitly restore aliases; one module can be registered at multiple
        # paths and functional_call temporarily rebinds those attributes.
        for name, parameter in named_parameters:
            parent, _, key = name.rpartition(".")
            setattr(module.get_submodule(parent), key, parameter)
        with torch.no_grad():
            for name, original in buffers_original.items():
                original.copy_(buffers[name].to(original.device))
        for name, buffer in original_buffers:
            parent, _, key = name.rpartition(".")
            setattr(module.get_submodule(parent), key, buffer)


def _debug_sdpa_context(device):
    if torch.device(device).type != "cuda":
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError as exc:
        raise RuntimeError("This debug offload requires PyTorch SDPA backend controls") from exc
    return sdpa_kernel([SDPBackend.MATH, SDPBackend.FLASH_ATTENTION])


def _offloaded_forward(policy, batch, device, *, backward_scale, execution_dtype=torch.bfloat16):
    if any(parameter.device.type != "cpu" for parameter in policy.parameters()):
        raise ValueError("Aligned debug offload requires CPU master parameters")
    wrapper = _ForwardBackward(policy)
    return _functional_call_cpu_masters(wrapper, (batch, backward_scale), device,
                                       dtype=execution_dtype)


class _PredictionAdapter(nn.Module):
    def __init__(self, policy, stage, device):
        super().__init__()
        self.policy = policy
        self.stage = stage
        self.device = torch.device(device)

    def forward(self, batch, noise):
        return _prediction_values(self.policy, batch, noise, self.stage, self.device)


def _offloaded_prediction(policy, batch, noise, stage, device, ema_values=None):
    adapter = _PredictionAdapter(policy, stage, device)
    overrides = {f"policy.{name}": value for name, value in (ema_values or {}).items()}
    with _debug_sdpa_context(device), torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda",
            cache_enabled=False):
        return _functional_call_cpu_masters(adapter, (batch, noise), device,
                                            parameter_overrides=overrides,
                                            save_activations=False)


def _state_digest(value):
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape)).encode())
            # ``view(dtype)`` rejects zero-dimensional tensors when the target
            # element size differs (for example the policy's scalar
            # optimizer_step buffer). Flatten first so scalar state participates
            # in the same stable byte-level digest as all other tensors.
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for part in item:
                visit(part)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def _gradient_summary(policy, optimized_ids):
    squares, active, missing = {}, {}, []
    finite = True
    for name, parameter in policy.named_parameters():
        if id(parameter) not in optimized_ids:
            if parameter.grad is not None:
                raise AssertionError(f"Frozen/unoptimized parameter received a gradient: {name}")
            continue
        active[name] = parameter
        if parameter.grad is None:
            missing.append(name)
            continue
        gradient = parameter.grad.detach()
        if not torch.isfinite(gradient).all():
            finite = False
            raise FloatingPointError(f"Nonfinite aligned debug gradient: {name}")
        if name.startswith("contact."):
            group = "contact." + name.split(".")[1]
        elif "vision_tower" in name or "multi_modal_projector" in name or "vision_backbone" in name:
            group = "policy_vision"
        elif any(part in name for part in ("world_model", "world_action_encoder", "world_state_encoder")):
            group = "world"
        else:
            group = "policy"
        squares[group] = squares.get(group, 0.0) + float(gradient.float().square().sum())
    norms = {name: float(value**0.5) for name, value in squares.items()}
    return norms, len(active), len(missing), finite, missing


def _required_gradient_groups(mode, stage, norms, freeze_world=False):
    required = ({"contact.predictor", "contact.updater", "contact.corrector"}
                if stage == "stage1" and freeze_world and mode == "full" else
                {"contact.predictor"} if stage == "tactile" else {"world"} if stage == "stage1" else {"policy"})
    if mode == "full":
        required.add(next((key for key in norms if key.startswith("contact.")), "contact"))
    missing = sorted(name for name in required
                     if not any(key == name or key.startswith(name + ".") for key in norms))
    if missing:
        raise AssertionError(f"Aligned debug has no gradient through active modules: {missing}")


def _metric_scalars(summary):
    result = {}
    for name, values in summary.get("metrics", {}).items():
        if "sum" not in values or "count" not in values:
            continue
        count = float(values["count"].float().sum())
        total = float(values["sum"].float().sum())
        result[name] = {"mean": total / max(1.0, count), "count": count}
    return result


def _apply_ema(policy, values):
    saved = {}
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if name in values:
                saved[name] = parameter.detach().clone()
                parameter.copy_(values[name].to(device=parameter.device, dtype=parameter.dtype))
    return saved


def _restore_parameters(policy, values):
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if name in values:
                parameter.copy_(values[name])


def _prediction_values(policy, batch, noise, stage, device):
    from .model import pad8
    from .sequence import observation_at

    valid = torch.nonzero(batch["sequence_mask"][0], as_tuple=False).flatten()
    if valid.numel() == 0:
        raise ValueError("Cannot run fixed-noise check on an empty sequence")
    index = int(valid[-1])
    obs = observation_at(batch, index)
    if stage == "tactile":
        action_stream = batch["world_actions"] if "world_actions" in batch else batch["actions"]
        state = obs.world_state if obs.world_state is not None else obs.state
        with torch.no_grad():
            expected = policy.contact.predictor(policy.encode_visual(obs.image, obs.wrist_image),
                action_stream[:, index, :policy.config.k], state, obs.tactile)
        return {"expected_contact": expected.detach().float().cpu()}
    if stage == "stage1":
        action_stream = batch["world_actions"] if "world_actions" in batch else batch["actions"]
        actions = action_stream[:, index, :policy.config.k]
        belief = policy.update_belief(obs)
        world_state = obs.world_state if obs.world_state is not None else obs.state
        current = policy.encode_visual(obs.image, obs.wrist_image)
        action_tokens = policy.core.world_action_encoder(
            pad8(actions).float(), pad8(world_state).float())
        initial, corrected, expected = policy.predict_future(
            current, world_state, action_tokens, belief, obs.tactile, actions=actions)
        values = {"belief": belief.detach().float().cpu(),
                  "world_initial": initial.detach().float().cpu(),
                  "world_corrected": corrected.detach().float().cpu()}
        if expected is not None:
            values["expected_contact"] = expected.detach().float().cpu()
        return values
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda",
                                         cache_enabled=False):
        prediction = policy.predict_action_chunk(obs, noise=noise)
    values = {"actions": prediction.actions.detach().float().cpu(),
              "belief": prediction.belief.detach().float().cpu()}
    if prediction.expected_contact is not None:
        values["expected_contact"] = prediction.expected_contact.detach().float().cpu()
    return values


def _assert_outputs_equal(before, after):
    if set(before) != set(after):
        raise AssertionError(f"Checkpoint outputs differ: {sorted(before)} != {sorted(after)}")
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    return max((float((before[key] - after[key]).abs().max()) if before[key].numel() else 0.0
                for key in before), default=0.0)


def _hardlink_export(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            try:
                os.link(item, target)
            except OSError:
                shutil.copy2(item, target)


def _worker_debug_settings(*, updates, execution_device, optimizer_device, offload, micro_batch, threads):
    return {"updates": updates, "device": str(execution_device),
            "optimizer_device": optimizer_device, "cpu_master_functional_offload": offload,
            "micro_batch": micro_batch, "threads": threads,
            "sdpa_backends": ["math", "flash_attention"] if offload else "default",
            "efficient_sdpa_disabled_for_offload": offload,
            "ema_device": "cpu" if offload else str(execution_device)}


def _resume_checkpoint_path(run, stage, task, updates, existing, expected_debug):
    """Return a complete bounded-debug checkpoint for post-save-only recovery."""
    if existing is not None:
        expected_report = {"complete": False, "formal_training": False,
                           "variant": run["variant"], "mode": run["mode"], "stage": stage,
                           "task": task or "shared_wm", "debug": expected_debug,
                           "formal_recipe": copy.deepcopy(run[stage])}
        mismatched = [key for key, value in expected_report.items()
                      if existing.get(key) != value]
        if mismatched:
            raise RuntimeError("Prior incomplete aligned debug report does not match this run: "
                               + ", ".join(mismatched))

        report_updates = existing.get("updates", [])
        if not isinstance(report_updates, list):
            raise RuntimeError("Prior aligned debug report has an invalid updates list")
        update_numbers = [item.get("update") for item in report_updates if isinstance(item, dict)]
        if len(update_numbers) != len(report_updates) or update_numbers != list(range(1, len(update_numbers) + 1)):
            raise RuntimeError("Prior aligned debug report has an invalid update history")
        if update_numbers and update_numbers != list(range(1, updates + 1)):
            raise RuntimeError("Prior aligned debug report update history disagrees with its checkpoint step")

    checkpoint = Path(run["output"]) / stage / "checkpoints" / f"step_{updates:08d}"
    if existing is not None and existing.get("checkpoint"):
        if Path(existing["checkpoint"]).resolve() != checkpoint.resolve():
            raise RuntimeError("Prior aligned debug report points at a different checkpoint")
    complete_path = checkpoint / "complete.json"
    training_path = checkpoint / "training.pt"
    model_path = checkpoint / "pretrained_model"
    if not complete_path.is_file() or not training_path.is_file():
        raise RuntimeError("Prior incomplete aligned debug has no complete saved checkpoint; "
                           "use a fresh output to avoid repeating updates")
    try:
        complete = json.loads(complete_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read aligned debug checkpoint marker: {complete_path}") from exc
    expected_complete = {"formal_training": False, "stage": stage,
                         "task": task or "shared_wm", "step": updates, "world_size": 1,
                         "training_contract": "bounded-debug-only"}
    if any(complete.get(key) != value for key, value in expected_complete.items()):
        raise RuntimeError("Saved aligned debug checkpoint marker does not match this run")
    model_files = ("config.json", "model.safetensors", "norm_stats.json", "wm_norm_stats.json",
                   "tokenizer.model", "deployment.json")
    if any(not (model_path / name).is_file() for name in model_files):
        raise RuntimeError("Saved aligned debug checkpoint is missing model export files")
    if stage == "stage2" and not (model_path / "ema.safetensors").is_file():
        raise RuntimeError("Saved stage-2 debug checkpoint is missing its EMA export")
    return checkpoint


def _resume_post_save_checks(run, stage, task, *, updates, device, optimizer_device,
                             micro_batch, threads, report, checkpoint, started):
    """Finish only the restore/inference checks for an already-saved debug step."""
    from .trainer import EMA, _load_dataset, make_optimizer, restore_rng, set_lr
    from .model import ContactWAM
    from .data import prepare_aligned_batch

    execution_device = torch.device(device)
    offload = optimizer_device == "cpu"
    state = torch.load(checkpoint / "training.pt", map_location="cpu", weights_only=False)
    expectations = state.get("post_save_expectations")
    required = {"model_signature", "optimizer_signature", "ema_signature", "fixed_noise_outputs"}
    if not isinstance(expectations, dict) or not required.issubset(expectations):
        raise RuntimeError("Saved aligned debug checkpoint has no complete post-save expectations")
    if state.get("step") != updates or state.get("schedule_step") != updates or state.get("world_size") != 1:
        raise RuntimeError("Saved aligned debug training state does not match its checkpoint step")
    if not isinstance(state.get("optimizer"), dict) or not isinstance(state["optimizer"].get("state"), dict):
        raise RuntimeError("Saved aligned debug checkpoint has invalid optimizer state")
    if _state_digest(state["optimizer"]["state"]) != expectations["optimizer_signature"]:
        raise RuntimeError("Saved aligned debug optimizer state failed its checkpoint integrity check")
    if not torch.is_tensor(state.get("fixed_noise")):
        raise RuntimeError("Saved aligned debug checkpoint has no fixed-noise tensor")
    if not isinstance(expectations["fixed_noise_outputs"], dict):
        raise RuntimeError("Saved aligned debug checkpoint has invalid fixed-noise expectations")
    next_lrs = state.get("next_learning_rates")
    if not isinstance(next_lrs, dict):
        raise RuntimeError("Saved aligned debug checkpoint has no next learning rates")
    saved_rng = state.get("rng")
    rng_probe_cpu = state.get("rng_probe_cpu")
    rng_probe_cuda = state.get("rng_probe_cuda")
    if saved_rng is None or not torch.is_tensor(rng_probe_cpu) or not torch.is_tensor(rng_probe_cuda):
        raise RuntimeError("Saved aligned debug checkpoint has incomplete RNG expectations")
    expected_before_save = expectations["fixed_noise_outputs"]
    fixed_noise_cpu = state["fixed_noise"]
    model_signature = expectations["model_signature"]
    optimizer_signature = expectations["optimizer_signature"]
    ema_signature = expectations["ema_signature"]
    if not isinstance(model_signature, str) or not isinstance(optimizer_signature, str):
        raise RuntimeError("Saved aligned debug checkpoint has invalid state signatures")
    if stage == "stage2" and not isinstance(ema_signature, str):
        raise RuntimeError("Saved stage-2 checkpoint has no EMA signature")
    if stage == "stage1" and ema_signature is not None:
        raise RuntimeError("Saved stage-1 checkpoint unexpectedly contains EMA state")

    from .config import ContactConfig
    config = ContactConfig(**run["model_config"])
    dataset = _load_dataset(run, stage)
    try:
        plan = _make_groups(dataset, stage, task, run["tasks"], config, run["seed"], micro_batch)
        expected_draws = sum(len(group["indices"]) for group in plan) * updates
        if state.get("draws") != expected_draws:
            raise RuntimeError("Saved aligned debug checkpoint draw count does not match its update budget")
        min_sequence = _minimum_sequence_length(config)
        sample_records = []
        for group_index, group in enumerate(plan):
            for position, index in enumerate(group["indices"]):
                _, record = _materialize_sample(
                    dataset, index,
                    int(run["seed"] + group_index * 10007 + position * 101),
                    minimum_sequence_length=min_sequence,
                    require_policy_target=(stage == "stage2"))
                sample_records.append(record)

        model_path = checkpoint / "pretrained_model"
        restored = ContactWAM.from_pretrained(model_path, stage=stage, device="cpu").eval()
        if _state_digest(restored.state_dict()) != model_signature:
            raise RuntimeError("Aligned debug model weights failed their checkpoint integrity check")
        if not offload:
            restored.to(execution_device)
        restored_optimizer = make_optimizer(
            restored, run, stage, device="cpu" if offload else execution_device)
        restored_optimizer.load_state_dict(state["optimizer"])
        if _state_digest(restored_optimizer.state_dict()["state"]) != optimizer_signature:
            raise RuntimeError("Aligned debug optimizer state did not round-trip")
        set_lr(restored_optimizer, run[stage], stage, int(state["schedule_step"]))
        restored_lrs = {group["name"]: float(group["lr"])
                        for group in restored_optimizer.param_groups}
        if restored_lrs != next_lrs:
            raise RuntimeError("Aligned debug LR schedule position did not round-trip")
        if int(restored.optimizer_step) != updates:
            raise RuntimeError("Aligned debug checkpoint optimizer step differs")

        # The optimizer and its serialized copy are only needed for the checks
        # above; release both before allocating EMA values or replaying inference.
        del state["optimizer"]
        del restored_optimizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        del state
        gc.collect()

        ema_restored = None
        if stage == "stage2":
            ema_restored = EMA(restored, float(run.get("ema_decay", 0.99)), device="cpu")
            ema_restored.load(model_path / "ema.safetensors", restored)
            if _state_digest(ema_restored.values) != ema_signature:
                raise RuntimeError("Aligned debug EMA state did not round-trip")
        elif ema_signature is not None:
            raise RuntimeError("Saved stage-1 checkpoint unexpectedly contains EMA state")

        restore_rng(saved_rng)
        torch.testing.assert_close(torch.rand(4), rng_probe_cpu, rtol=0, atol=0)
        torch.testing.assert_close(torch.rand(4, device=execution_device).cpu(),
                                   rng_probe_cuda, rtol=0, atol=0)

        last_group = len(plan) - 1
        replay_samples = []
        for position, index in enumerate(plan[last_group]["indices"]):
            sample, _ = _materialize_sample(
                dataset, index,
                int(run["seed"] + updates * 1000003 + last_group * 10007 + position * 101),
                minimum_sequence_length=min_sequence,
                require_policy_target=(stage == "stage2"))
            replay_samples.append(sample)
        replay_raw = collate_contact(replay_samples)
        replay_batch = prepare_aligned_batch(replay_raw, execution_device)
        noise = fixed_noise_cpu.to(execution_device)
        restored.eval()
        if offload:
            after = _offloaded_prediction(
                restored, replay_batch, noise, stage, execution_device,
                ema_values=None if ema_restored is None else ema_restored.values)
        else:
            if ema_restored is not None:
                _apply_ema(restored, ema_restored.values)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=True, cache_enabled=False):
                after = _prediction_values(restored, replay_batch, noise, stage, execution_device)
        difference = _assert_outputs_equal(expected_before_save, after)

        final_export = None
        if stage == "stage2":
            final_export = Path(run["output"]) / "export" / "final"
            _hardlink_export(model_path, final_export)
            expected_files = {"config.json", "model.safetensors", "ema.safetensors", "norm_stats.json",
                              "wm_norm_stats.json", "tokenizer.model", "deployment.json"}
            missing_files = sorted(name for name in expected_files if not (final_export / name).is_file())
            if missing_files:
                raise FileNotFoundError(f"Aligned debug export is missing {missing_files}")

        report.pop("error", None)
        report.update(complete=True, checkpoint=str(checkpoint),
                      export=str(final_export) if final_export else None,
                      initialization=report.get("initialization") or {
                          "resumed_from_saved_checkpoint": True},
                      samples=report.get("samples") or sample_records,
                      checkpoint_outputs_max_abs_difference=difference,
                      model_state_roundtrip=True, optimizer_restored=True,
                      lr_schedule_restored=True, ema_restored=(ema_restored is not None),
                      rng_restored=True, fixed_noise=True, ema_exported=(stage == "stage2"),
                      updates_recovered_from_checkpoint=updates,
                      update_traces_recovered=(len(report.get("updates", [])) == updates),
                      peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(execution_device) / 2**30,
                      seconds=time.monotonic() - started)
        return report
    finally:
        close = getattr(dataset, "close", None)
        if close:
            close()


def _debug_worker(run, stage, *, task=None, updates=2, device="cuda", optimizer_device="cpu",
                  micro_batch=2, threads=4, data_report=None):
    from .trainer import (EMA, _export, _load_dataset, _load_model, make_optimizer,
                                  rng_state, restore_rng, set_lr)
    from .model import ContactWAM
    from .sequence import SequenceTrainingModel
    from .data import prepare_aligned_batch

    updates = validate_debug_updates(updates)
    micro_batch = validate_debug_micro_batch(micro_batch)
    if stage not in {"tactile", "stage1", "stage2"}:
        raise ValueError(f"Unknown aligned debug stage: {stage}")
    execution_device = torch.device(device)
    if execution_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Aligned real-model debug requires the available developer CUDA device")
    if optimizer_device not in {"cpu", "cuda"}:
        raise ValueError("Debug optimizer device must be cpu or cuda")
    offload = optimizer_device == "cpu"
    if stage == "stage1" and micro_batch < 2:
        raise ValueError("Stage-1 debug requires micro-batch 2 for task-preserving ranking")
    torch.set_num_threads(threads)
    torch.manual_seed(int(run["seed"]))
    np.random.seed(int(run["seed"]))
    import random
    random.seed(int(run["seed"]))

    output = Path(run["output"])
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "debug_report.json"
    started = time.monotonic()
    worker_debug = _worker_debug_settings(
        updates=updates, execution_device=execution_device, optimizer_device=optimizer_device,
        offload=offload, micro_batch=micro_batch, threads=threads)
    report = {"complete": False, "formal_training": False, "variant": run["variant"],
              "mode": run["mode"], "stage": stage, "task": task or "shared_wm",
              "debug": worker_debug, "formal_recipe": copy.deepcopy(run[stage]),
              "data_report": data_report, "updates": [], "normalization_fields": {
                  "policy_state": "state", "world_state": "world_state",
                  "policy_actions": "actions", "world_actions": "world_actions"}}
    expected_checkpoint = output / stage / "checkpoints" / f"step_{updates:08d}"
    existing = None
    if report_path.exists():
        existing = json.loads(report_path.read_text())
        if existing.get("complete") is True:
            expected_complete = {"formal_training": False, "variant": run["variant"],
                                "mode": run["mode"], "stage": stage,
                                "task": task or "shared_wm", "debug": worker_debug,
                                "formal_recipe": copy.deepcopy(run[stage])}
            mismatched = [key for key, value in expected_complete.items()
                          if existing.get(key) != value]
            if mismatched:
                raise RuntimeError("Completed aligned debug report does not match this run: "
                                   + ", ".join(mismatched))
            return existing
    if existing is not None or expected_checkpoint.exists():
        if existing is not None:
            report = existing
        try:
            checkpoint = _resume_checkpoint_path(
                run, stage, task, updates, existing, worker_debug)
            _resume_post_save_checks(
                run, stage, task, updates=updates, device=device,
                optimizer_device=optimizer_device, micro_batch=micro_batch,
                threads=threads, report=report, checkpoint=checkpoint, started=started)
            return report
        except Exception:
            report["error"] = traceback.format_exc()
            raise
        finally:
            atomic_json(report_path, report)
    atomic_json(report_path, report)
    try:
        from .config import ContactConfig
        config = ContactConfig(**run["model_config"])
        dataset = _load_dataset(run, stage)
        plan = _make_groups(dataset, stage, task, run["tasks"], config, run["seed"], micro_batch)
        min_sequence = _minimum_sequence_length(config)
        samples_by_group = []
        sample_records = []
        for group_index, group in enumerate(plan):
            samples, records = [], []
            for position, index in enumerate(group["indices"]):
                seed = int(run["seed"] + group_index * 10007 + position * 101)
                sample, record = _materialize_sample(dataset, index, seed,
                    minimum_sequence_length=min_sequence,
                    require_policy_target=(stage == "stage2"))
                samples.append(sample)
                records.append(record)
            if stage == "stage1" and group["source"] == "original" and len(samples) != 2:
                raise AssertionError("Stage-1 ranking requires two original examples in one forward")
            samples_by_group.append(samples)
            sample_records.extend(records)

        model_device = execution_device if not offload else torch.device("cpu")
        policy, initialization = _load_model(run, stage)
        policy.to(model_device).train()
        wrapper = SequenceTrainingModel(policy)
        optimizer = make_optimizer(policy, run, stage, device=optimizer_device)
        optimized_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        trainable_names = [name for name, parameter in policy.named_parameters() if id(parameter) in optimized_ids]
        non_fp32 = [name for name, parameter in policy.named_parameters()
                    if id(parameter) in optimized_ids and parameter.dtype != torch.float32]
        if non_fp32:
            raise AssertionError(f"Aligned trainable parameters must stay FP32 masters: {non_fp32[:8]}")
        frozen_versions = {name: parameter._version for name, parameter in policy.named_parameters()
                           if id(parameter) not in optimized_ids}
        ema = (EMA(policy, float(run.get("ema_decay", 0.99)), device="cpu" if offload else execution_device)
               if stage == "stage2" else None)
        adapter = _LossAdapter(wrapper)
        policy.train()
        torch.cuda.reset_peak_memory_stats(execution_device)

        for update in range(updates):
            optimizer.zero_grad(set_to_none=True)
            group_results = []
            update_batches = []
            total_sequences = sum(len(group) for group in samples_by_group)
            for group_index, samples in enumerate(samples_by_group):
                draw_samples = []
                group_records = []
                for position, index in enumerate(plan[group_index]["indices"]):
                    seed = int(run["seed"] + (update + 1) * 1000003 + group_index * 10007 + position * 101)
                    sample, record = _materialize_sample(dataset, index, seed,
                        minimum_sequence_length=min_sequence,
                        require_policy_target=(stage == "stage2"))
                    draw_samples.append(sample)
                    group_records.append(record)
                raw = collate_contact(draw_samples)
                batch = prepare_aligned_batch(raw, execution_device)
                weight = len(draw_samples) / total_sequences
                if stage == "stage2":
                    policy.optimizer_step.fill_(update)
                autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=False)
                with autocast:
                    if offload:
                        summary, result = _offloaded_forward(adapter, batch, execution_device,
                            backward_scale=1.0 / weight, execution_dtype=torch.bfloat16)
                        # `result` carries detached loss/metrics from the real sequence forward.
                        loss_value = float(summary.detach())
                    else:
                        result = wrapper(batch)
                        if not torch.isfinite(result["loss"]):
                            raise FloatingPointError("Nonfinite aligned debug loss")
                        (result["loss"] * weight).backward()
                        result = _detach_metrics(result)
                        loss_value = float(result["loss"])
                if not torch.isfinite(torch.tensor(loss_value)):
                    raise FloatingPointError("Nonfinite aligned debug loss")
                group_results.append({"source": plan[group_index]["source"], "task": plan[group_index]["task"],
                                      "episode_ids": [item["episode_id"] for item in group_records],
                                      "loss": loss_value, "metrics": _metric_scalars(result)})
                update_batches.append((raw, batch))

            gradient_norms, active_count, missing_count, _finite, missing = _gradient_summary(policy, optimized_ids)
            _required_gradient_groups(run["mode"], stage, gradient_norms,
                                      freeze_world=run.get("stage1_freeze_world", False))
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.grad is not None],
                                                        float(run["grad_clip"]), foreach=False)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Nonfinite aligned debug gradient norm")
            set_lr(optimizer, run[stage], stage, update)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            policy.optimizer_step.fill_(update + 1)
            if ema is not None:
                ema.update(policy)
            changed_frozen = [name for name, parameter in policy.named_parameters()
                              if name in frozen_versions and parameter._version != frozen_versions[name]]
            if changed_frozen:
                raise AssertionError(f"Frozen/unoptimized parameter changed: {changed_frozen[:8]}")
            trace = {"update": update + 1, "group_losses": group_results,
                     "gradient_norm": float(grad_norm), "gradient_norm_by_module": gradient_norms,
                     "active_trainable_parameter_count": active_count,
                     "trainable_parameters_missing_gradients": missing_count,
                     "missing_trainable_gradient_names": missing[:16],
                     "frozen_parameter_count": len(frozen_versions),
                     "frozen_parameters_unchanged": True,
                     "recurrent_valid_steps": [record["valid_sequence_steps"] for record in sample_records],
                     "ranking_pair_same_task": bool(stage == "stage1"),
                     "ranking_metric": next((value["metrics"].get("rank") for value in group_results
                                              if value["source"] == "original"), None),
                     "learning_rates": {group["name"]: float(group["lr"]) for group in optimizer.param_groups},
                     "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(execution_device) / 2**30}
            if stage == "stage1":
                ranking_metric = trace["ranking_metric"]
                if ranking_metric is None or ranking_metric["count"] <= 0:
                    raise AssertionError("Stage-1 debug did not exercise valid same-task ranking pairs")
            report["updates"].append(trace)
            print(json.dumps({"event": "aligned_debug_update", "stage": stage, "task": task,
                              "variant": run["variant"], **trace}, allow_nan=False), flush=True)
            # Preserve a real aligned batch for deterministic checkpoint inference below.
            raw, batch = update_batches[-1]
            del update_batches

        # Capture a fixed action-noise tensor, optimizer, LR position, RNG, model,
        # and EMA state in the same isolated debug checkpoint.
        stage_root = output / stage
        checkpoint = stage_root / "checkpoints" / f"step_{updates:08d}"
        model_path = checkpoint / "pretrained_model"
        checkpoint.mkdir(parents=True, exist_ok=False)
        model_signature = _state_digest(policy.state_dict())
        _export(policy, model_path, run, ema=ema)
        set_lr(optimizer, run[stage], stage, updates)
        saved_next_lrs = {group["name"]: float(group["lr"]) for group in optimizer.param_groups}
        noise = torch.randn((1, config.chunk_size, config.padded_dim), device=execution_device)
        # Capture the exact expected deployment result while the updated
        # in-memory model still exists. A second load of the same checkpoint is
        # compared against this independent pre-export result below.
        was_training = policy.training
        policy.eval()
        if offload:
            expected_before_save = _offloaded_prediction(
                policy, batch, noise, stage, execution_device,
                ema_values=None if ema is None else ema.values)
        else:
            saved_ema_parameters = _apply_ema(policy, ema.values) if ema is not None else {}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=True, cache_enabled=False):
                expected_before_save = _prediction_values(policy, batch, noise, stage, execution_device)
            _restore_parameters(policy, saved_ema_parameters)
        policy.train(was_training)
        saved_rng = rng_state()
        expected_probe_cpu = torch.rand(4)
        expected_probe_cuda = torch.rand(4, device=execution_device).cpu()
        restore_rng(saved_rng)
        optimizer_signature = _state_digest(optimizer.state_dict()["state"])
        ema_signature = None if ema is None else _state_digest(ema.values)
        training_state = {"optimizer": optimizer.state_dict(), "step": updates,
                          "schedule_step": updates, "next_learning_rates": saved_next_lrs,
                          "world_size": 1, "draws": int(total_sequences * updates),
                          "rng": saved_rng, "rng_probe_cpu": expected_probe_cpu,
                          "rng_probe_cuda": expected_probe_cuda, "fixed_noise": noise.detach().cpu(),
                          # Keep independent pre-export expectations with the
                          # checkpoint so post-save checks can be repeated
                          # without performing another optimizer update.
                          "post_save_expectations": {
                              "model_signature": model_signature,
                              "optimizer_signature": optimizer_signature,
                              "ema_signature": ema_signature,
                              "fixed_noise_outputs": expected_before_save,
                          }}
        torch.save(training_state, checkpoint / "training.pt")
        # Drop the optimizer.state_dict() snapshot before loading its serialized
        # copy below; keeping this dictionary alive can roughly double CPU RAM.
        del training_state
        atomic_json(checkpoint / "complete.json", {"formal_training": False, "stage": stage,
                    "task": task or "shared_wm", "step": updates, "world_size": 1,
                    "training_contract": "bounded-debug-only"})

        # Release CPU optimizer state before allocating the inference copy on CUDA.
        del adapter, wrapper, optimizer, policy, ema, raw, batch, result
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        restored = ContactWAM.from_pretrained(model_path, stage=stage, device="cpu").eval()
        if _state_digest(restored.state_dict()) != model_signature:
            raise AssertionError("Aligned debug model weights did not round-trip through export")
        if not offload:
            restored.to(execution_device)
        restored_optimizer = make_optimizer(restored, run, stage,
                                             device="cpu" if offload else execution_device)
        state = torch.load(checkpoint / "training.pt", map_location="cpu", weights_only=False)
        restored_optimizer.load_state_dict(state["optimizer"])
        if _state_digest(restored_optimizer.state_dict()["state"]) != optimizer_signature:
            raise AssertionError("Aligned debug optimizer state did not round-trip")
        set_lr(restored_optimizer, run[stage], stage, int(state["schedule_step"]))
        restored_lrs = {group["name"]: float(group["lr"]) for group in restored_optimizer.param_groups}
        if restored_lrs != saved_next_lrs:
            raise AssertionError("Aligned debug LR schedule position did not round-trip")
        saved_rng = state["rng"]
        rng_probe_cpu = state["rng_probe_cpu"]
        rng_probe_cuda = state["rng_probe_cuda"]
        fixed_noise_cpu = state["fixed_noise"]
        del state["optimizer"]
        del restored_optimizer
        del state
        gc.collect()
        torch.cuda.empty_cache()
        ema_restored = None
        if stage == "stage2":
            ema_restored = EMA(restored, float(run.get("ema_decay", 0.99)), device="cpu")
            ema_restored.load(model_path / "ema.safetensors", restored)
            if _state_digest(ema_restored.values) != ema_signature:
                raise AssertionError("Aligned debug EMA state did not round-trip")
        if int(restored.optimizer_step) != updates:
            raise AssertionError("Aligned debug checkpoint optimizer step differs")
        restore_rng(saved_rng)
        torch.testing.assert_close(torch.rand(4), rng_probe_cpu, rtol=0, atol=0)
        torch.testing.assert_close(torch.rand(4, device=execution_device).cpu(),
                                   rng_probe_cuda, rtol=0, atol=0)

        # Recreate the last real source batch and compare its pre-export result
        # with the reloaded checkpoint's EMA deployment path under fixed noise.
        last_group = len(plan) - 1
        sample_indices = plan[last_group]["indices"]
        replay_samples, replay_records = [], []
        for position, index in enumerate(sample_indices):
            sample, record = _materialize_sample(dataset, index,
                int(run["seed"] + updates * 1000003 + last_group * 10007 + position * 101),
                minimum_sequence_length=min_sequence, require_policy_target=(stage == "stage2"))
            replay_samples.append(sample)
            replay_records.append(record)
        replay_raw = collate_contact(replay_samples)
        replay_batch = prepare_aligned_batch(replay_raw, execution_device)
        noise = fixed_noise_cpu.to(execution_device)
        restored.eval()
        if offload:
            after = _offloaded_prediction(
                restored, replay_batch, noise, stage, execution_device,
                ema_values=None if ema_restored is None else ema_restored.values)
        else:
            restored.to(execution_device)
            if ema_restored is not None:
                _apply_ema(restored, ema_restored.values)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=True, cache_enabled=False):
                after = _prediction_values(restored, replay_batch, noise, stage, execution_device)
        # `batch` and `replay_batch` are deterministically materialized from the
        # same index/seed; still check this explicitly before comparing outputs.
        difference = _assert_outputs_equal(expected_before_save, after)

        if stage == "stage2":
            final_export = output / "export" / "final"
            _hardlink_export(model_path, final_export)
            expected_files = {"config.json", "model.safetensors", "ema.safetensors", "norm_stats.json",
                              "wm_norm_stats.json", "tokenizer.model", "deployment.json"}
            missing_files = sorted(name for name in expected_files if not (final_export / name).is_file())
            if missing_files:
                raise FileNotFoundError(f"Aligned debug export is missing {missing_files}")
        else:
            final_export = None
        report.update(complete=True, checkpoint=str(checkpoint), export=str(final_export) if final_export else None,
                      initialization=initialization, samples=sample_records,
                      checkpoint_outputs_max_abs_difference=difference,
                      model_state_roundtrip=True,
                      optimizer_restored=True, lr_schedule_restored=True,
                      ema_restored=(ema_restored is not None), rng_restored=True,
                      fixed_noise=True, ema_exported=(stage == "stage2"),
                      trainable_parameter_count=len(trainable_names),
                      trainable_parameters_fp32=True,
                      peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(execution_device) / 2**30,
                      seconds=time.monotonic() - started)
        return report
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        if "dataset" in locals():
            close = getattr(dataset, "close", None)
            if close:
                close()
        atomic_json(report_path, report)


def run_debug_pipeline(run, *, updates=2, device="cuda", optimizer_device="cpu",
                       micro_batch=2, threads=4, data_report=None):
    """Run Stage 1 and both task policies in a variant-specific debug tree."""
    settings = debug_settings(run, updates=updates, device=device, optimizer_device=optimizer_device,
                              micro_batch=micro_batch, threads=threads)
    output = Path(run["output"])
    reject_formal_output(output)
    output.mkdir(parents=True, exist_ok=True)
    root_config = {"formal_training": False, "run": copy.deepcopy(run), "debug": settings}
    config_path = output / "debug_run.yaml"
    if config_path.exists():
        previous = yaml.safe_load(config_path.read_text())
        if previous != root_config:
            raise ValueError(f"Aligned debug settings changed for {output}; choose a fresh --output")
    else:
        _atomic_yaml(config_path, root_config)

    from . import train

    results = {}
    root_report = {"complete": False, "formal_training": False, "variant": run["variant"],
                   "mode": run["mode"], "debug": settings, "models": results}
    summary_path = output / "debug_summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if previous.get("complete"):
            return previous
        expected_summary = {"formal_training": False, "variant": run["variant"],
                            "mode": run["mode"], "debug": settings}
        mismatched = [key for key, value in expected_summary.items()
                      if previous.get(key) != value]
        if mismatched:
            raise RuntimeError(f"Prior incomplete aligned debug at {output} does not match this run: "
                               + ", ".join(mismatched))
        previous_models = previous.get("models", {})
        if not isinstance(previous_models, dict):
            raise RuntimeError(f"Prior incomplete aligned debug at {output} has invalid model results")
        for model_name in previous_models:
            if model_name not in {"tactile_wm", "shared_wm", *run["policy_tasks"]}:
                raise RuntimeError(f"Prior summary contains an unknown debug model: {model_name}")
            model_output = (output / model_name if model_name in {"tactile_wm", "shared_wm"}
                            else output / "tasks" / model_name)
            report_path = model_output / "debug_report.json"
            if not report_path.is_file() and not (model_output / "stage1" / "checkpoints").exists() \
                    and not (model_output / "stage2" / "checkpoints").exists() \
                    and not (model_output / "tactile" / "checkpoints").exists():
                raise RuntimeError(f"Prior summary records {model_name} without its report or checkpoint")
    atomic_json(summary_path, root_report)
    try:
        teacher = None
        if not (run["mode"] == "wave0" and run.get("stage1_freeze_world", False)):
            stage1_run = train.stage_run(run, "stage1")
            stage1_report = _debug_worker(stage1_run, "stage1", updates=updates, device=device,
                                      optimizer_device=optimizer_device, micro_batch=micro_batch,
                                      threads=threads, data_report=data_report)
            results["shared_wm"] = stage1_report
            atomic_json(summary_path, {**root_report, "models": results})
            teacher = Path(stage1_report["checkpoint"])
        for task in run["policy_tasks"]:
            stage2_run = train.stage_run(run, "stage2", task, checkpoint=teacher)
            report = _debug_worker(stage2_run, "stage2", task=task, updates=updates, device=device,
                                   optimizer_device=optimizer_device, micro_batch=1, threads=threads,
                                   data_report=data_report)
            results[task] = report
            atomic_json(summary_path, {**root_report, "models": results})
        root_report.update(complete=True, models=results, seconds=sum(value.get("seconds", 0.0)
                                                                       for value in results.values()))
        atomic_json(summary_path, root_report)
        print(json.dumps({"event": "aligned_debug_complete", "summary": str(summary_path),
                          "formal_training": False}), flush=True)
        return root_report
    except Exception:
        root_report["error"] = traceback.format_exc()
        root_report["models"] = results
        atomic_json(summary_path, root_report)
        raise
