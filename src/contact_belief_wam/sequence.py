"""One DDP forward owns an entire truncated recurrent sequence."""
from __future__ import annotations

from contextlib import nullcontext

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
        world_state=batch["world_state"][:, index] if "world_state" in batch else None)


def targets_at(batch, index):
    return ContactTargets(actions=batch["actions"][:, index], action_mask=batch["action_mask"][:, index],
                          future_image=batch["future_image"][:, index], future_tactile=batch["future_tactile"][:, index],
                          future_mask=batch["future_mask"][:, index],
                          policy_action_mask=(batch["policy_action_mask"][:, index]
                                              if "policy_action_mask" in batch else None),
                          world_actions=batch["world_actions"][:, index] if "world_actions" in batch else None)


def select_valid(old, new, mask):
    if new is None:
        return old
    if old is None:
        old = torch.zeros_like(new)
    return torch.where(mask.reshape(mask.shape[0], *([1] * (new.ndim - 1))), new, old)


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
        belief, expected, target_step = None, None, None
        history_valid = torch.zeros(b, dtype=torch.bool, device=batch["state"].device)
        expected_valid = history_valid.clone()
        action_losses, future_losses, loss_masks, future_masks, metrics = [], [], [], [], {}
        for index in range(length):
            valid = batch["sequence_mask"][:, index]
            if not valid.any():
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
                    new_belief, new_expected = self.policy.history_step(obs, history_actions)
                else:
                    output = self.policy(obs, targets_at(batch, index))
                    new_belief, new_expected = output["belief"], output["expected_contact"]
                    action_losses.append(output["action_loss_per_sample"])
                    future_losses.append(output["future_loss_per_sample"])
                    action_valid = output.get("action_valid")
                    loss_masks.append(valid if action_valid is None else valid & action_valid.bool())
                    future_masks.append(valid & batch["future_mask"][:, index])
                    for key, value in output["metrics"].items():
                        mask = valid & batch["future_mask"][:, index] if key in {
                            "visual_initial", "visual_delta", "rank", "world_total", "tactile_future", "visual_corrected",
                            "alignment", "policy_visual_future"} else valid
                        if key == "flow" and action_valid is not None:
                            mask = valid & action_valid.bool()
                        if key in output.get("metric_masks", {}):
                            mask = valid & output["metric_masks"][key]
                        policy_aux_is_valid = output.get("policy_aux_valid")
                        if key in {"alignment", "policy_visual_future"} and policy_aux_is_valid is not None:
                            mask = valid & batch["future_mask"][:, index] & policy_aux_is_valid.bool()
                        metrics.setdefault(key, []).append((value.detach(), mask))
            belief = select_valid(belief, new_belief, valid)
            expected = select_valid(expected, new_expected, valid)
            history_valid = history_valid | valid
            expected_valid = valid & batch["future_mask"][:, index] & (new_expected is not None)
            target_step = batch["step"][:, index] + cfg.k
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
