from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .config import ContactConfig
from .modules import ContactModules
from .paths import atomic_json, configure
from .types import ContactObservation, ContactPrediction, ContactTargets

configure()
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.wave0.configuration_wave0 import WAVE0Config
from lerobot.policies.wave0.modeling_wave0 import WAVE0Policy, WorldActionEncoder, make_att_2d_masks


def wave_config(cfg: ContactConfig):
    # The extension owns next-frame timing. Construct the upstream config with its
    # historical h+1 convention, then expose our actual observation horizon k.
    stage = "stage1" if cfg.world_only else "inference" if cfg.use_world else "disabled"
    value = WAVE0Config(
        world_model_stage=stage, world_model_future_horizons=(cfg.k - 1,),
        wam_latent_action_num_queries=cfg.k, chunk_size=cfg.chunk_size, n_action_steps=cfg.k,
        max_action_dim=cfg.padded_dim, max_state_dim=cfg.padded_dim,
        input_features={"observation.images.head": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
                        "observation.state": PolicyFeature(FeatureType.STATE, (8,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (32 if cfg.aligned else 8,))},
        dtype=cfg.dtype, device="cpu", freeze_vision_encoder=not cfg.aligned,
        wam_share_frozen_vision_encoder=not cfg.aligned, gradient_checkpointing=cfg.gradient_checkpointing,
        world_model_width=cfg.world_width, world_model_depth=cfg.world_depth,
        world_model_num_heads=cfg.world_heads, world_model_mlp_dim=cfg.world_ffn,
        world_model_base_model_path=cfg.pi05_base, num_inference_steps=cfg.num_inference_steps,
        compile_model=False, push_to_hub=False,
    )
    value.world_model_future_horizons = (cfg.k - 1 if cfg.aligned else cfg.k,)
    if cfg.aligned:
        value.wam_loss_weight = 1.0
        value.wam_loss_warmup_steps = 1000
        value.wam_alignment_decay_start_step = 3000
        value.wam_alignment_decay_end_step = 10000
    return value


def feature_distance(pred, target):
    dims = tuple(range(1, pred.ndim))
    cosine = (1 - F.cosine_similarity(pred.float(), target.float(), dim=-1)).flatten(1).mean(1)
    huber = F.smooth_l1_loss(pred.float(), target.float(), reduction="none").mean(dims)
    return cosine + .5 * huber


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


def action_target_mask(action_mask, policy_action_mask=None):
    """Combine real action availability with optional policy supervision."""
    mask = action_mask.to(dtype=torch.bool)
    if policy_action_mask is not None:
        policy_mask = torch.as_tensor(policy_action_mask, device=mask.device, dtype=torch.bool)
        if policy_mask.ndim == mask.ndim - 1 and policy_mask.shape == mask.shape[:1]:
            policy_mask = policy_mask.unsqueeze(-1)
        try:
            policy_mask = torch.broadcast_to(policy_mask, mask.shape)
        except RuntimeError as exc:
            raise ValueError(
                "policy_action_mask must be broadcastable to the action mask "
                f"shape {tuple(mask.shape)}, got {tuple(policy_mask.shape)}"
            ) from exc
        mask = mask & policy_mask
    return mask


def policy_aux_valid(action_mask, policy_action_mask, k):
    """Mark samples whose first k actions are all valid policy targets.

    Without an explicit policy mask, preserve the historical auxiliary-loss
    behavior. The target future mask continues to control the teacher losses.
    """
    if policy_action_mask is None:
        return torch.ones(action_mask.shape[0], dtype=torch.bool, device=action_mask.device)
    mask = action_target_mask(action_mask, policy_action_mask)
    if mask.shape[1] < k:
        raise ValueError(f"action mask has {mask.shape[1]} positions, fewer than k={k}")
    return mask[:, :k].all(dim=1)


def masked_action_flow(flow_tokens, action_mask, policy_action_mask=None):
    """Reduce flow loss over valid policy-supervised chunk positions only."""
    mask = action_target_mask(action_mask, policy_action_mask)
    if mask.shape != flow_tokens.shape:
        raise ValueError(
            f"action mask shape {tuple(mask.shape)} must match flow tokens {tuple(flow_tokens.shape)}"
        )
    per_sample = (flow_tokens * mask.to(flow_tokens.dtype)).sum(1) / mask.sum(1).clamp_min(1)
    return per_sample, mask.any(1)


def pad8(value):
    return F.pad(value, (0, 32 - value.shape[-1]))


class ContactWAM(nn.Module):
    name = "contact_wave0"

    def __init__(self, config: ContactConfig, base_policy=None):
        super().__init__()
        self.config = config
        self.base_policy = base_policy if base_policy is not None else WAVE0Policy(wave_config(config))
        if not config.world_only and config.use_world:
            # Unlike stock deployment, final actions must still be encoded for feedback.
            self.core.world_action_encoder = WorldActionEncoder(32, config.world_width, 2,
                                                                config.world_heads, config.chunk_size)
        self.contact = ContactModules(config) if config.use_belief else None
        self.register_buffer("optimizer_step", torch.zeros((), dtype=torch.long))
        # Only the VLM/Expert custom attention needs this switch; the world
        # model already uses PyTorch SDPA. No global monkeypatch of Transformers.
        from lerobot.policies.wave0.modeling_wave0 import modeling_gemma
        for module in self.base_policy.modules():
            if isinstance(module, modeling_gemma.GemmaAttention):
                module.wave0_attention_backend = config.attention_backend
        self.configure_trainability()

    @property
    def core(self):
        return self.base_policy.model

    def configure_trainability(self):
        cfg = self.config
        if cfg.stage == "tactile":
            self.requires_grad_(False)
            self.contact.predictor.requires_grad_(True)
            return
        self.requires_grad_(cfg.stage != "inference")
        if cfg.stage == "stage1":
            self.core.vision_backbone.requires_grad_(False)
            if self.contact:
                # No language queries or Expert in Stage 1; avoid optimizer/unused DDP state.
                self.contact.to_queries.requires_grad_(False)
                self.contact.to_expert.requires_grad_(False)
                self.contact.query_gate.requires_grad_(False)
                if cfg.independent_tactile:
                    self.contact.predictor.requires_grad_(False)
        elif cfg.stage == "stage2":
            pg = self.core.paligemma_with_expert.paligemma
            pg.vision_tower.requires_grad_(cfg.aligned)
            pg.multi_modal_projector.requires_grad_(cfg.aligned)
            if cfg.aligned:
                self.core.paligemma_with_expert.gemma_expert.lm_head.requires_grad_(False)
            if cfg.use_world:
                for name in ("world_action_encoder", "world_state_encoder", "world_model", "world_vision_backbone"):
                    getattr(self.core, name).requires_grad_(False)
            else:
                self.core.wam_latent_action_queries.requires_grad_(False)
                self.core.wam_latent_action_projection.requires_grad_(False)
            if self.contact and cfg.stage2_freeze_belief:
                self.contact.updater.requires_grad_(False)
            if self.contact and (cfg.stage2_freeze_contact_predictor or cfg.independent_tactile):
                self.contact.predictor.requires_grad_(False)
        if self.contact:
            if cfg.mode == "direct_touch":
                for name in ("corrector", "predictor", "to_queries", "to_state"):
                    getattr(self.contact, name).requires_grad_(False)
                self.contact.query_gate.requires_grad_(False)
                self.contact.state_gate.requires_grad_(False)
            else:
                for enabled, module in ((cfg.belief_to_latent_queries, self.contact.to_queries),
                                        (cfg.belief_to_world_state, self.contact.to_state),
                                        (cfg.belief_to_expert, self.contact.to_expert)):
                    if not enabled:
                        module.requires_grad_(False)
                if not cfg.belief_to_latent_queries:
                    self.contact.query_gate.requires_grad_(False)
                if not cfg.belief_to_world_state:
                    self.contact.state_gate.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.config.stage == "tactile":
            self.base_policy.eval()
            self.contact.eval()
            self.contact.predictor.train(mode)
            return self
        # Stock base_policy is constructed in inference mode to keep loading independent
        # of a Stage-1 path. Set actual component modes explicitly here.
        if self.config.world_only:
            self.core.vision_backbone.eval()
            self.core.world_action_encoder.train(mode)
            self.core.world_state_encoder.train(mode)
            self.core.world_model.train(mode)
        else:
            self.core.paligemma_with_expert.train(mode)
            pg = self.core.paligemma_with_expert.paligemma
            pg.vision_tower.train(mode and self.config.aligned)
            pg.multi_modal_projector.train(mode and self.config.aligned)
            if self.config.use_world:
                self.core.world_vision_backbone.eval()
            if self.config.use_world:
                self.core.world_model.eval()
                self.core.world_action_encoder.eval()
                self.core.world_state_encoder.eval()
        return self

    def update_belief(self, observation):
        if not self.config.use_belief:
            return observation.state.new_zeros(observation.state.shape[0], 4, self.config.width)
        return self.contact.updater(observation)

    def history_step(self, obs, actions):
        """Recur from historical inputs and already-executed actions, without future targets."""
        belief = self.update_belief(obs)
        expected = None
        if self.config.mode == "full":
            current = self.encode_visual(obs.image)
            world_state = obs.world_state if obs.world_state is not None else obs.state
            latent = self.core.world_action_encoder(pad8(actions[:, :self.config.k]).float(), pad8(world_state).float())
            _, _, expected = self.predict_future(current, world_state, latent, belief, obs.tactile, actions=actions)
        return belief, expected

    def encode_visual(self, image):
        with torch.no_grad():
            if self.config.world_only:
                from lerobot.policies.wave0.modeling_wave0 import spatial_pool_image_tokens
                return spatial_pool_image_tokens(self.core.vision_backbone(image), self.base_policy.config.world_model_spatial_pool_size).float()[:, None]
            return self.core.encode_world_images([image]).float()

    def predict_future(self, current, state, action_tokens, belief, tactile, *, actions=None, detach_for_flow=False):
        state_token = self.core.world_state_encoder(pad8(state).float())
        if self.config.mode == "full" and self.config.belief_to_world_state:
            state_token = state_token + self.contact.state_gate * self.contact.to_state(belief.mean(1))[:, None]
        initial = current + self.core.world_model(current, state_token, action_tokens)
        source = initial.detach() if detach_for_flow else initial
        if self.config.mode == "full":
            corrected = self.contact.corrector(source, belief, action_tokens)
            if self.config.independent_tactile:
                # VLM latent plans have no physical action prefix yet. Predict
                # contact only when the actual normalized actions are known.
                contact = (self.contact.predictor(current, actions[:, :self.config.k], state, tactile)
                           if actions is not None else None)
            else:
                contact = self.contact.predictor(corrected, action_tokens, tactile, belief)
        else:
            corrected, contact = source, None
        return initial, corrected, contact

    def encode_context(self, obs, belief):
        core, cfg = self.core, self.config
        b = obs.state.shape[0]
        image_masks = [torch.ones(b, dtype=torch.bool, device=obs.image.device)]
        embeddings, padding, blocks = core.embed_prefix([self.policy_image(obs.image)], image_masks, obs.tokens, obs.token_mask)
        context_length = embeddings.shape[1]
        if cfg.use_world:
            query = core.wam_latent_action_queries[:, :cfg.k].expand(b, -1, -1)
            if cfg.mode == "full" and cfg.belief_to_latent_queries:
                query = query + self.contact.query_gate * self.contact.to_queries(belief.mean(1))[:, None]
            embeddings = torch.cat([embeddings, query.to(embeddings.dtype)], 1)
            qp = torch.ones(b, cfg.k, dtype=torch.bool, device=padding.device)
            qb = torch.zeros_like(qp)
            qb[:, 0] = True
            padding, blocks = torch.cat([padding, qp], 1), torch.cat([blocks, qb], 1)
        attention = core._prepare_attention_masks_4d(make_att_2d_masks(padding, blocks))
        positions = padding.long().cumsum(1) - 1
        dtype = core.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        output, kv = core.paligemma_with_expert.forward_vlm_with_context_kv(
            prefix_embeds=embeddings.to(dtype), attention_mask=attention,
            position_ids=positions, context_length=context_length)
        action_latent = core.wam_latent_action_projection(output[:, -cfg.k:].float()) if cfg.use_world else None
        return action_latent, padding[:, :context_length], kv

    def expert_velocity(self, pad, kv, condition, noisy_actions, time):
        core = self.core
        action, action_pad, action_block, time_condition = core.embed_suffix(noisy_actions, time)
        n = condition.shape[1]
        condition_pad = torch.ones(pad.shape[0], n, dtype=torch.bool, device=pad.device)
        condition_block = torch.zeros_like(condition_pad)
        if n:
            condition_block[:, 0] = True
        all_pad = torch.cat([pad, condition_pad, action_pad], 1)
        all_block = torch.cat([torch.zeros_like(pad), condition_block, action_block.bool()], 1)
        attention = core._prepare_attention_masks_4d(make_att_2d_masks(all_pad, all_block)[:, pad.shape[1]:])
        positions = (all_pad.long().cumsum(1) - 1)[:, pad.shape[1]:]
        dtype = core.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype
        _, out = core.paligemma_with_expert.forward_expert_with_context_kv(
            context_key_values=kv, future_embeds=condition.to(dtype), action_embeds=action.to(dtype),
            attention_mask=attention, position_ids=positions,
            future_adarms_cond=torch.zeros_like(time_condition), action_adarms_cond=time_condition)
        return core.action_out_proj(out[:, -self.config.chunk_size:].float())

    def policy_image(self, image):
        if self.config.aligned and self.config.policy_augmentation and self.training:
            from .aligned_augmentation import augment
            return augment(image / 2 + .5) * 2 - 1
        return image

    def to_world_actions(self, actions):
        if not getattr(self.config, "aligned", False):
            return actions
        source, target = self.config.policy_norm_stats, self.config.world_norm_stats
        if source is None or target is None:
            raise ValueError("Aligned deployment requires policy and shared-WM normalization")
        src, dst = source["actions"], target["actions"]
        lo = actions.new_tensor(src["q01"])
        span = actions.new_tensor(src["q99"]) - lo + 1e-6
        dst_lo = actions.new_tensor(dst["q01"])
        dst_span = actions.new_tensor(dst["q99"]) - dst_lo + 1e-6
        normalized = ((actions[..., :8] + 1) * .5 * span + lo - dst_lo) / dst_span * 2 - 1
        return torch.cat([normalized, actions[..., 8:]], -1)

    def policy_condition(self, obs, belief, current):
        latent, pad, kv = self.encode_context(obs, belief)
        condition = []
        predicted = None
        if self.config.use_world:
            initial, corrected, _ = self.predict_future(current, obs.world_state if obs.world_state is not None else obs.state, latent, belief, obs.tactile,
                                                       detach_for_flow=True)
            condition.append(self.core.wam_future_condition_resampler(corrected))
            predicted = initial
        if self.config.use_belief and self.config.belief_to_expert:
            condition.append(self.contact.to_expert(belief))
        context = torch.cat(condition, 1) if condition else obs.state.new_empty(obs.state.shape[0], 0, self.config.expert_width)
        return context, pad, kv, latent, predicted

    def forward_tactile(self, obs, target):
        """Fit only tactile dynamics to frozen AnyTouch targets, without recurrence."""
        cfg = self.config
        current = self.encode_visual(obs.image)
        state = obs.world_state if obs.world_state is not None else obs.state
        actions = target.world_actions if target.world_actions is not None else target.actions
        expected = self.contact.predictor(current, actions[:, :cfg.k], state, obs.tactile.detach())
        future_touch = target.future_tactile.detach()
        loss = feature_distance(expected, future_touch)
        persistence = feature_distance(obs.tactile.detach(), future_touch)
        zero = torch.zeros_like(loss)
        valid = target.future_mask.bool()
        metrics = {"loss": loss, "tactile_future": loss, "tactile_persistence": persistence,
                   "tactile_gain": persistence - loss}
        masks = {key: valid for key in metrics}
        if not self.training:
            tasks = obs.task if obs.task is not None else [f"sample-{i}" for i in range(len(loss))]
            neg, neg_valid = WAVE0Policy._make_negative_actions(actions[:, :cfg.k], tasks)
            wrong = self.contact.predictor(current, neg, state, obs.tactile)
            metrics["tactile_wrong_action"] = feature_distance(wrong, future_touch)
            metrics["tactile_action_gap"] = metrics["tactile_wrong_action"] - loss
            for name in ("tactile_wrong_action", "tactile_action_gap"):
                masks[name] = valid & neg_valid
        return {"loss_per_sample": loss * valid, "future_loss_per_sample": loss,
                "action_loss_per_sample": zero, "action_valid": torch.zeros_like(valid),
                "belief": obs.state.new_zeros(len(loss), cfg.belief_tokens, cfg.width),
                "expected_contact": expected, "expectation_valid": valid,
                "metrics": metrics, "metric_masks": masks}

    def forward(self, obs: ContactObservation, target: ContactTargets, *, noise=None, time=None):
        if self.config.stage == "tactile":
            return self.forward_tactile(obs, target)
        cfg, b = self.config, obs.state.shape[0]
        world_state = obs.world_state if obs.world_state is not None else obs.state
        world_actions = target.world_actions if target.world_actions is not None else target.actions
        belief = self.update_belief(obs)
        zero = obs.state.new_zeros(b)
        metrics = {}
        if not cfg.use_world and not (cfg.use_belief and cfg.belief_to_expert):
            # The upstream multi-stream helper cannot reshape a zero-token
            # future stream. The official PI0.5 forward is the exact no-condition
            # path, and preserves identical baseline losses/gradients.
            images_mask = [torch.ones(b, dtype=torch.bool, device=obs.image.device)]
            flow_dimensions = self.core([self.policy_image(obs.image)], images_mask, obs.tokens, obs.token_mask,
                                        pad8(target.actions).float(), noise=noise, time=time)
            flow_tokens = flow_dimensions[:, :, :(32 if cfg.aligned else 8)].mean(-1)
            flow, action_valid = masked_action_flow(
                flow_tokens, target.action_mask, target.policy_action_mask
            )
            return {"loss_per_sample": flow, "action_loss_per_sample": flow, "future_loss_per_sample": zero,
                    "action_valid": action_valid,
                    "policy_aux_valid": action_valid,
                    "belief": belief, "expected_contact": None, "expectation_valid": torch.zeros_like(target.future_mask),
                    "metrics": {"loss": flow, "flow": flow, "belief_norm": belief.float().norm(dim=-1).mean(1)}}
        current = self.encode_visual(obs.image) if cfg.use_world or cfg.world_only else None
        future = self.encode_visual(target.future_image) if cfg.use_world or cfg.world_only else None
        expected = None
        world_loss, touch_loss, corrected_loss = zero, zero, zero
        if cfg.use_world or cfg.stage == "stage1":
            action_tokens = self.core.world_action_encoder(pad8(world_actions[:, :cfg.k]).float(), pad8(world_state).float())
            initial, corrected, expected = self.predict_future(current, world_state, action_tokens, belief, obs.tactile, actions=world_actions)
            visual = (1 - F.cosine_similarity(initial, future, dim=-1)).flatten(1).mean(1)
            delta = F.smooth_l1_loss(initial - current, future - current, reduction="none").flatten(1).mean(1)
            rank = zero
            if cfg.stage == "stage1":
                tasks = obs.task
                if tasks is None:
                    # Keep singleton/missing metadata samples within their own
                    # trajectory rather than pairing different tasks.
                    tasks = [f"sample-{index}" for index in range(b)]
                elif isinstance(tasks, str):
                    tasks = [tasks] * b
                elif torch.is_tensor(tasks):
                    tasks = tasks.detach().cpu().tolist()
                if len(tasks) != b:
                    raise ValueError(f"Expected one task label per sample, got {len(tasks)} for batch {b}")
                neg, valid_neg = WAVE0Policy._make_negative_actions(pad8(world_actions[:, :cfg.k]), tasks)
                if valid_neg.any():
                    neg_tokens = self.core.world_action_encoder(neg, pad8(world_state).float())
                    wrong, _, _ = self.predict_future(current, world_state, neg_tokens, belief, obs.tactile, actions=neg)
                    wrong_score = (1 - F.cosine_similarity(wrong, future, dim=-1)).flatten(1).mean(1)
                    rank = F.relu(.1 + visual - wrong_score) * valid_neg
            world_loss = visual + .5 * delta + .1 * rank
            metrics.update(visual_initial=visual, visual_delta=delta, rank=rank, world_total=world_loss)
            if cfg.mode == "full":
                touch_loss = feature_distance(expected, target.future_tactile)
                corrected_loss = feature_distance(corrected, future)
                metrics.update(tactile_future=touch_loss, visual_corrected=corrected_loss)
        if cfg.stage == "stage1":
            flow = zero
            action_valid = torch.zeros(b, dtype=torch.bool, device=zero.device)
            policy_aux_is_valid = torch.zeros_like(action_valid)
            future_loss = world_loss + corrected_loss
            if not cfg.independent_tactile:
                future_loss = future_loss + touch_loss
            loss = future_loss * target.future_mask
        else:
            policy_aux_is_valid = policy_aux_valid(
                target.action_mask, target.policy_action_mask, cfg.k
            )
            condition, pad, kv, latent, predicted = self.policy_condition(obs, belief, current)
            actions = pad8(target.actions).float()
            noise = self.core.sample_noise(actions.shape, actions.device) if noise is None else noise
            time = self.core.sample_time(b, actions.device) if time is None else time
            noisy = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
            velocity = self.expert_velocity(pad, kv, condition, noisy, time)
            flow_tokens = ((velocity - (noise - actions))[:, :, :(32 if cfg.aligned else 8)] ** 2).mean(-1)
            flow, action_valid = masked_action_flow(
                flow_tokens, target.action_mask, target.policy_action_mask
            )
            auxiliary = zero
            if cfg.use_world:
                alignment = feature_distance(latent, action_tokens.detach())
                predicted_visual = (1 - F.cosine_similarity(predicted, future, dim=-1)).flatten(1).mean(1)
                predicted_delta = F.smooth_l1_loss(predicted - current, future - current, reduction="none").flatten(1).mean(1)
                alignment_weight = self.base_policy.config.get_wam_alignment_weight(int(self.optimizer_step))
                auxiliary = (predicted_visual + .5 * predicted_delta + alignment_weight * alignment)
                auxiliary = auxiliary * policy_aux_is_valid.to(auxiliary.dtype)
                metrics.update(alignment=alignment, policy_visual_future=predicted_visual)
            warmup = self.base_policy.config.wam_loss_warmup_steps
            wam_scale = min(1., (int(self.optimizer_step) + 1) / warmup) if warmup > 0 else 1.
            future_loss = ((1. if cfg.aligned else .1) * wam_scale * auxiliary
                           + .1 * (touch_loss + corrected_loss))
            loss = flow + future_loss * target.future_mask
            metrics["flow"] = flow
        metrics.update(loss=loss, belief_norm=belief.float().norm(dim=-1).mean(1))
        if obs.belief is not None:
            metrics["belief_delta"] = (belief - obs.belief).float().norm(dim=-1).mean(1)
        return {"loss_per_sample": loss, "belief": belief, "expected_contact": expected,
                "action_loss_per_sample": flow, "action_valid": action_valid,
                "policy_aux_valid": policy_aux_is_valid,
                "future_loss_per_sample": future_loss,
                "metrics": metrics, "expectation_valid": target.future_mask if expected is not None else torch.zeros_like(target.future_mask)}

    @torch.no_grad()
    def predict_action_chunk(self, obs: ContactObservation, noise=None, action_transform=None):
        def transform_actions(actions):
            if action_transform is None:
                return actions
            transformed = action_transform(actions[..., :8])
            if not torch.is_tensor(transformed):
                transformed = torch.as_tensor(transformed, device=actions.device, dtype=actions.dtype)
            else:
                transformed = transformed.to(device=actions.device, dtype=actions.dtype)
            if transformed.shape != actions[..., :8].shape or not torch.isfinite(transformed).all():
                raise ValueError("action_transform must return finite normalized [..., 8] actions")
            return torch.cat((transformed, actions[..., 8:]), dim=-1)

        if self.config.world_only:
            raise ValueError("Stage 1 checkpoint does not contain an action policy")
        if obs.step is None:
            raise ValueError("ContactObservation.step is required to align cached contact predictions")
        self.eval()
        cfg = self.config
        if cfg.mode in {"pi05", "wave0"} or (not cfg.use_world and not cfg.belief_to_expert):
            # Baselines execute the vendored official sampler verbatim. This also
            # avoids numerical differences between its cache and our extended
            # Expert context implementation when no tactile conditions are used.
            masks = [torch.ones(obs.state.shape[0], dtype=torch.bool, device=obs.image.device)]
            kwargs = {"state": pad8(obs.world_state if obs.world_state is not None else obs.state), "world_images": [obs.image]} if cfg.use_world else {}
            actions = self.core.sample_actions([obs.image], masks, obs.tokens, obs.token_mask, noise=noise, **kwargs)
            actions = transform_actions(actions)
            belief = self.update_belief(obs)
            return ContactPrediction(actions[:, :, :8], belief, None, obs.step + cfg.k,
                                     {"belief_norm": belief.norm(dim=-1), "condition_tokens": 4 if cfg.use_world else 0})
        belief = self.update_belief(obs)
        current = self.encode_visual(obs.image) if cfg.use_world else None
        condition, pad, kv, _, _ = self.policy_condition(obs, belief, current)
        shape = (obs.state.shape[0], cfg.chunk_size, 32)
        actions = self.core.sample_noise(shape, obs.state.device) if noise is None else noise.clone()
        dt = -1.0 / cfg.num_inference_steps
        for index in range(cfg.num_inference_steps):
            time = actions.new_full((actions.shape[0],), 1.0 + index * dt)
            actions = actions + dt * self.expert_velocity(pad, kv, condition, actions, time)
        # Keep execution, world-action encoding, and tactile prediction on the
        # same normalized action plan when a deployment supplies joint limits.
        actions = transform_actions(actions)
        expected = None
        if cfg.mode == "full":
            # Padded action coordinates never represent real actuators.
            final_prefix = self.to_world_actions(pad8(actions[:, :cfg.k, :8]))
            world_state = obs.world_state if obs.world_state is not None else obs.state
            latent = self.core.world_action_encoder(final_prefix, pad8(world_state))
            _, _, expected = self.predict_future(current, world_state, latent, belief, obs.tactile, actions=final_prefix)
        return ContactPrediction(actions[:, :, :8], belief, expected, obs.step + cfg.k,
                                 {"belief_norm": belief.norm(dim=-1), "condition_tokens": condition.shape[1]})

    def save_pretrained(self, directory):
        from safetensors.torch import save_model
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "config.json", self.config.to_dict())
        save_model(self, str(directory / "model.safetensors"))

    @classmethod
    def from_pretrained(cls, directory, *, device="cpu", stage=None, k=None):
        from safetensors.torch import load_model
        directory = Path(directory)
        cfg = ContactConfig.load(directory / "config.json")
        old_k = cfg.k
        if stage and (stage in {"tactile", "stage1"}) != cfg.world_only:
            raise ValueError("Stage 1 uses a world-only architecture; use weights.initialize for Stage 1 → Stage 2")
        if stage:
            cfg = replace(cfg, stage=stage)
        if k is not None:
            if cfg.independent_tactile and k != old_k:
                raise ValueError("Independent tactile dynamics must use its trained prediction horizon k")
            cfg = replace(cfg, k=k)
        model = cls(cfg)
        if cfg.k == old_k:
            load_model(model, str(directory / "model.safetensors"), strict=True)
        else:
            from .weights import load_contact_resized
            load_contact_resized(model, directory / "model.safetensors")
        return model.to(device)


POLICY_REGISTRY = {"contact_wave0": ContactWAM}
