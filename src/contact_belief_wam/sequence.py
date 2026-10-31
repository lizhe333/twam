"""One DDP forward owns an entire truncated recurrent sequence."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import fields

import numpy as np
import torch
from torch import nn

from .preprocessing import vision_preprocess
from .types import ContactObservation, ContactTargets


def prepare_batch(batch, tokenizer, device, *, include_language=True):
    result = dict(batch)
    b, t = batch["state"].shape[:2]
    for key in ("image", "future_image"):
        images = batch[key].numpy() if isinstance(batch[key], torch.Tensor) else np.asarray(batch[key])
        result[key] = vision_preprocess(images.reshape(-1, *images.shape[2:])).reshape(b, t, 3, 224, 224)
    if include_language:
        states = np.asarray(batch["state"]).reshape(-1, 8)
        prompts = [prompt for prompt in batch["prompt"] for _ in range(t)]
        ids, mask = tokenizer(prompts, states)
        result["tokens"], result["token_mask"] = ids.reshape(b, t, -1), mask.reshape(b, t, -1)
    else:
        result["tokens"] = torch.zeros((b, t, 1), dtype=torch.long)
        result["token_mask"] = torch.zeros((b, t, 1), dtype=torch.bool)
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in result.items()}


def observation_at(batch, index, belief=None, expected=None, valid=None, target_step=None):
    step = batch["step"][:, index]
    expected_valid = torch.zeros_like(step, dtype=torch.bool) if valid is None else valid
    if target_step is not None:
        expected_valid = expected_valid & (target_step == step)
    task = batch.get("task")
    if task is not None:
        length = batch["sequence_mask"].shape[1]
        if torch.is_tensor(task):
            if task.ndim >= 2 and task.shape[1] == length:
                task = task[:, index]
        elif len(task) and isinstance(task[0], (list, tuple)) and len(task[0]) == length:
            task = [sample[index] for sample in task]
    return ContactObservation(
        image=batch["image"][:, index], state=batch["state"][:, index], tactile=batch["tactile"][:, index],
        tokens=batch["tokens"][:, index], token_mask=batch["token_mask"][:, index],
        belief=belief, expected_contact=expected, expected_valid=expected_valid,
        executed_actions=batch["executed_actions"][:, index], executed_valid=batch["executed_valid"][:, index],
        belief_valid=valid, step=step, task=task,
        world_state=batch["world_state"][:, index] if "world_state" in batch else None,
        wrist_image=batch["wrist_image"][:, index] if "wrist_image" in batch else None)


def targets_at(batch, index):
    return ContactTargets(actions=batch["actions"][:, index], action_mask=batch["action_mask"][:, index],
                          future_image=batch["future_image"][:, index], future_tactile=batch["future_tactile"][:, index],
                          future_mask=batch["future_mask"][:, index],
                          policy_action_mask=(batch["policy_action_mask"][:, index]
                                              if "policy_action_mask" in batch else None),
                          world_actions=batch["world_actions"][:, index] if "world_actions" in batch else None,
                          future_wrist_image=batch["future_wrist_image"][:, index] if "future_wrist_image" in batch else None)


def select_valid(old, new, mask):
    if new is None:
        return old
    if old is None:
        old = torch.zeros_like(new)
    return torch.where(mask.reshape(mask.shape[0], *([1] * (new.ndim - 1))), new, old)


def _cat_records(records):
    """Time-major batching; retain the autograd edges of recurrent beliefs."""
    values = {}
    for field in fields(records[0]):
        parts = [getattr(record, field.name) for record in records]
        example = next((part for part in parts if part is not None), None)
        if example is None:
            values[field.name] = None
        elif torch.is_tensor(example):
            values[field.name] = torch.cat([
                part if part is not None else torch.zeros_like(example) for part in parts])
        elif isinstance(example, (list, tuple)):
            values[field.name] = [item for part in parts for item in part]
        else:
            raise TypeError(f"Cannot batch observation field {field.name}")
    return type(records[0])(**values)


def _batch_policy(policy, records, batch_size):
    """Batch expensive policy passes AFTER the differentiable belief recurrence.

    Sample augmentation/noise/time separately in the former time order. Belief
    and independent tactile dynamics contain no stochastic layers, so this
    preserves random draws as well as the per-sequence loss normalization.
    """
    observations, targets, beliefs, contacts, current, future = [], [], [], [], [], []
    images, noises, times = [], [], []
    for _, obs, target, features, belief, expected in records:
        observations.append(obs)
        targets.append(target)
        beliefs.append(belief)
        contacts.append(expected)
        current.append(features["current_visual"])
        future.append(features["future_visual"])
        images.append([policy.policy_image(view) for view in policy.images(obs.image, obs.wrist_image)])
        shape = (*target.actions.shape[:-1], policy.config.padded_dim)
        noises.append(policy.core.sample_noise(shape, target.actions.device))
        times.append(policy.core.sample_time(batch_size, target.actions.device))
    result = policy(_cat_records(observations), _cat_records(targets),
                    current_visual=torch.cat(current), future_visual=torch.cat(future),
                    updated_belief=torch.cat(beliefs), expected_contact=torch.cat(contacts),
                    policy_images=[torch.cat(views) for views in zip(*images)],
                    noise=torch.cat(noises), time=torch.cat(times))
    outputs = []
    for position, (index, obs, *_rest) in enumerate(records):
        chosen = slice(position * batch_size, (position + 1) * batch_size)
        output = {key: ({name: value[chosen] for name, value in item.items()}
                        if isinstance(item, dict) else item[chosen] if torch.is_tensor(item) else item)
                  for key, item in result.items()}
        if obs.belief is None:
            output["metrics"].pop("belief_delta", None)
        outputs.append((index, output))
    return outputs


class FrozenVisualCache:
    """Reuse only deterministic WM vision latents within one sequence batch.

    Keys use exact dataset frame indices AND batch row, never just a timestamp
    shared by different episodes. Policy vision remains trainable/augmented and
    always runs normally. Drop the entire cache when this forward returns.
    """
    def __init__(self, policy, batch):
        self.policy, self.batch = policy, batch
        self.indices = torch.stack([batch["visual_frame_index"],
                                    batch["future_visual_frame_index"]]).detach().cpu().tolist()
        self.values = {}

    def get(self, index, *, future=False):
        frame_indices = self.indices[int(future)]
        keys = [(row, frames[index]) for row, frames in enumerate(frame_indices)]
        missing = [row for row, key in enumerate(keys) if key not in self.values]
        if missing:
            prefix = "future_" if future else ""
            images = self.batch[prefix + "image"][:, index]
            rows = torch.tensor(missing, device=images.device)
            wrist = self.batch.get(prefix + "wrist_image")
            encoded = self.policy.encode_visual(images.index_select(0, rows),
                        wrist[:, index].index_select(0, rows) if wrist is not None else None)
            for position, row in enumerate(missing):
                self.values[keys[row]] = encoded[position]
        return torch.stack([self.values[key] for key in keys])


class SequenceTrainingModel(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch):
        cfg = self.policy.config
        b, length = batch["sequence_mask"].shape
        first_grad = max(0, length - cfg.belief_bptt_steps)
        if getattr(cfg, "stage", None) == "tactile":
            # Independent dynamics uses each valid window directly; no updater
            # warmup and no prediction-to-belief computation graph.
            first_grad = 0
        # Older/custom data without exact frame metadata retains its old path.
        cache_visual = (getattr(cfg, "aligned", False) and getattr(cfg, "use_world", False)
                        and getattr(cfg, "stage", None) in {"stage1", "stage2"}
                        and "visual_frame_index" in batch and "future_visual_frame_index" in batch)
        visual_cache = FrozenVisualCache(self.policy, batch) if cache_visual else None
        batch_policy = (getattr(cfg, "batch_stage2_policy", False)
                        and cfg.stage == "stage2" and cfg.mode == "full" and cfg.independent_tactile)
        active = batch.get("_sequence_active")
        if active is None:
            active = batch["sequence_mask"].any(dim=0).detach().cpu().tolist()
        deferred, outputs = [], []
        belief, expected, target_step = None, None, None
        history_valid = torch.zeros(b, dtype=torch.bool, device=batch["state"].device)
        expected_valid = history_valid.clone()
        action_losses, future_losses, loss_masks, future_masks, metrics = [], [], [], [], {}
        for index in range(length):
            valid = batch["sequence_mask"][:, index]
            if not active[index]:
                continue
            if index == first_grad:
                belief = belief.detach() if belief is not None else None
                expected = expected.detach() if expected is not None else None
                # Warmup runs under no_grad inside the caller's BF16 autocast.
                # Its cached FP32->BF16 weight casts also have no grad history;
                # reusing those casts silently freezes Linear projections in
                # the trainable suffix. Recast them with autograd enabled.
                torch.clear_autocast_cache()
            obs = observation_at(batch, index, belief, expected, expected_valid, target_step)
            obs.belief_valid = history_valid
            with torch.no_grad() if index < first_grad else nullcontext():
                if index < first_grad:
                    # At historical t_i, predict t_i+k from that interval's
                    # actions. These are the executed actions reported at the
                    # NEXT observation, not the preceding interval at t_i.
                    history_actions = (batch["world_actions"][:, index, :cfg.k] if "world_actions" in batch
                                       else batch["executed_actions"][:, index + 1])
                    features = ({"current_visual": visual_cache.get(index)}
                                if visual_cache is not None and cfg.mode == "full" else {})
                    new_belief, new_expected = self.policy.history_step(obs, history_actions, **features)
                else:
                    features = ({"current_visual": visual_cache.get(index),
                                 "future_visual": visual_cache.get(index, future=True)}
                                if visual_cache is not None else {})
                    target = targets_at(batch, index)
                    if batch_policy:
                        if not features:
                            features = {"current_visual": self.policy.encode_visual(obs.image, obs.wrist_image),
                                        "future_visual": self.policy.encode_visual(target.future_image, target.future_wrist_image)}
                        actions = target.world_actions if target.world_actions is not None else target.actions
                        new_belief, new_expected = self.policy.history_step(
                            obs, actions[:, :cfg.k], current_visual=features["current_visual"])
                        deferred.append((index, obs, target, features, new_belief, new_expected))
                    else:
                        output = self.policy(obs, target, **features)
                        new_belief, new_expected = output["belief"], output["expected_contact"]
                        outputs.append((index, output))
            belief = select_valid(belief, new_belief, valid)
            expected = select_valid(expected, new_expected, valid)
            history_valid = history_valid | valid
            expected_valid = valid & batch["future_mask"][:, index] & (new_expected is not None)
            target_step = batch["step"][:, index] + cfg.k
        if deferred:
            outputs = _batch_policy(self.policy, deferred, b)
        for index, output in outputs:
            valid = batch["sequence_mask"][:, index]
            action_losses.append(output["action_loss_per_sample"])
            future_losses.append(output["future_loss_per_sample"])
            action_valid = output.get("action_valid")
            loss_masks.append(valid if action_valid is None else valid & action_valid.bool())
            future_masks.append(valid & batch["future_mask"][:, index])
            for key, value in output["metrics"].items():
                mask = valid & batch["future_mask"][:, index] if key in {
                    "visual_initial", "visual_delta", "rank", "world_total", "tactile_future", "visual_corrected",
                    "alignment", "policy_visual_future", "contact_total", "tactile_persistence",
                    "tactile_gain"} else valid
                if key == "flow" and action_valid is not None:
                    mask = valid & action_valid.bool()
                if key in output.get("metric_masks", {}):
                    mask = valid & output["metric_masks"][key]
                policy_aux_is_valid = output.get("policy_aux_valid")
                if key in {"alignment", "policy_visual_future"} and policy_aux_is_valid is not None:
                    mask = valid & batch["future_mask"][:, index] & policy_aux_is_valid.bool()
                metrics.setdefault(key, []).append((value.detach(), mask))
        weights = torch.stack(loss_masks).float()
        action_per_sample = (torch.stack(action_losses) * weights).sum(0) / weights.sum(0).clamp_min(1)
        action_valid_samples = weights.sum(0) > 0
        future_weights = torch.stack(future_masks).float()
        future_per_sample = (torch.stack(future_losses) * future_weights).sum(0) / future_weights.sum(0).clamp_min(1)
        future_valid = future_weights.sum(0) > 0
        action_loss = action_per_sample.sum() / action_valid_samples.sum().clamp_min(1)
        loss = action_loss + future_per_sample.sum() / future_valid.sum().clamp_min(1)
        reduced = {}
        for key, entries in metrics.items():
            values = torch.stack([v for v, _ in entries])
            masks = torch.stack([m for _, m in entries]).float()
            reduced[key] = {"sum": (values * masks).sum(0), "count": masks.sum(0)}
        return {"loss": loss, "metrics": reduced, "belief": belief}
