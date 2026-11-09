from __future__ import annotations

import torch
from torch import nn

from .config import ContactConfig
from .types import ContactObservation


def decoder(cfg, layers=2):
    layer = nn.TransformerDecoderLayer(cfg.width, cfg.heads, cfg.ffn, dropout=0.0,
                                       activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(cfg.width))


class BeliefUpdater(nn.Module):
    def __init__(self, cfg: ContactConfig):
        super().__init__()
        self.cfg = cfg
        self.initial = nn.Parameter(torch.randn(1, cfg.belief_tokens, cfg.width) * .02)
        self.types = nn.Parameter(torch.randn(7, cfg.width) * .02)
        self.action_position = nn.Parameter(torch.randn(1, cfg.chunk_size, cfg.width) * .02)
        self.state_proj = nn.Linear(8, cfg.width)
        self.action_proj = nn.Linear(8, cfg.width)
        self.touch_norm = nn.LayerNorm(cfg.width)
        self.decoder = decoder(cfg)

    def forward(self, obs: ContactObservation):
        touch = obs.tactile.float()
        b, _, d = touch.shape
        device = touch.device
        initial = self.initial.expand(b, -1, -1)
        valid_b = obs.belief_valid if obs.belief_valid is not None else torch.zeros(b, dtype=torch.bool, device=device)
        previous = torch.where(valid_b[:, None, None], obs.belief, initial) if obs.belief is not None else initial
        valid_e = obs.expected_valid if obs.expected_valid is not None else torch.zeros(b, dtype=torch.bool, device=device)
        expected = obs.expected_contact if obs.expected_contact is not None else torch.zeros_like(touch)
        expected = torch.where(valid_e[:, None, None], expected, torch.zeros_like(expected))
        diff = torch.where(valid_e[:, None, None], touch - expected, torch.zeros_like(touch))
        actions = obs.executed_actions if obs.executed_actions is not None else touch.new_zeros(b, self.cfg.k, 8)
        action_valid = obs.executed_valid if obs.executed_valid is not None else torch.zeros(b, self.cfg.k, dtype=torch.bool, device=device)
        memory = [self.touch_norm(touch) + self.types[0], self.touch_norm(expected) + self.types[1],
                  self.touch_norm(diff) + self.types[2], previous + self.types[3],
                  self.state_proj((obs.world_state if obs.world_state is not None else obs.state).float())[:, None] + self.types[4],
                  self.action_proj(actions.float()) + self.action_position[:, :actions.shape[1]] + self.types[5]]
        padding = torch.cat([torch.zeros(b, 8, dtype=torch.bool, device=device),
                             (~valid_e)[:, None].expand(-1, 16),
                             torch.zeros(b, self.cfg.belief_tokens + 1, dtype=torch.bool, device=device),
                             ~action_valid], dim=1)
        return self.decoder(previous + self.types[6], torch.cat(memory, 1), memory_key_padding_mask=padding)


class FutureCorrector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vision_in = nn.Linear(cfg.vision_width, cfg.width)
        self.action_in = nn.Linear(cfg.world_width, cfg.width)
        self.decoder = decoder(cfg)
        self.vision_out = nn.Linear(cfg.width, cfg.vision_width)
        self.gate = nn.Parameter(torch.tensor(.1))
        self.use_belief = cfg.belief_to_future_corrector

    def forward(self, future, belief, action):
        shape = future.shape
        query = self.vision_in(future.flatten(1, 2).float())
        memory = self.action_in(action.float())
        if self.use_belief:
            memory = torch.cat([memory, belief], dim=1)
        residual = self.vision_out(self.decoder(query, memory)).reshape(shape)
        return future + self.gate * residual


class ContactPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 8, cfg.width) * .02)
        self.vision_in = nn.Linear(cfg.vision_width, cfg.width)
        self.action_in = nn.Linear(cfg.world_width, cfg.width)
        self.touch_in = nn.Linear(cfg.width, cfg.width)
        self.decoder = decoder(cfg)
        self.output = nn.Linear(cfg.width, cfg.width)
        self.use_belief = cfg.belief_to_contact_predictor

    def forward(self, future, action, tactile, belief):
        memory = [self.vision_in(future.flatten(1, 2).float()), self.action_in(action.float()), self.touch_in(tactile.float())]
        if self.use_belief:
            memory.append(belief)
        queries = self.query.expand(tactile.shape[0], -1, -1) + self.touch_in(tactile.float())
        return self.output(self.decoder(queries, torch.cat(memory, 1)))


class TactileWorldModel(nn.Module):
    """Independent action-conditioned tactile dynamics; no belief or visual future.

    Inputs use the shared WM normalization and frozen visual features.
    """

    def __init__(self, cfg):
        super().__init__()
        self.camera_embedding = (nn.Parameter(torch.randn(1, 2, 1, cfg.width) * .02)
                                 if cfg.camera_mode == "head_wrist" else None)
        self.k = cfg.k
        self.query = nn.Parameter(torch.randn(1, cfg.tactile_tokens, cfg.width) * .02)
        self.action_position = nn.Parameter(torch.randn(1, cfg.k, cfg.width) * .02)
        self.types = nn.Parameter(torch.randn(4, cfg.width) * .02)
        self.vision_in = nn.Linear(cfg.vision_width, cfg.width)
        self.action_in = nn.Linear(cfg.action_dim, cfg.width)
        self.state_in = nn.Linear(cfg.action_dim, cfg.width)
        self.touch_in = nn.Linear(cfg.width, cfg.width)
        self.decoder = decoder(cfg)
        self.output = nn.Linear(cfg.width, cfg.width)

    def forward(self, current_visual, actions, state, tactile):
        if actions.shape[1] != self.k:
            raise ValueError(f"Expected {self.k} future actions, got {actions.shape[1]}")
        touch = self.touch_in(tactile.float())
        visual = self.vision_in(current_visual.float())
        if self.camera_embedding is not None:
            if visual.shape[1] != 2:
                raise ValueError("Dual-view tactile dynamics requires head and wrist features")
            visual = visual + self.camera_embedding
        memory = torch.cat([
            visual.flatten(1, 2) + self.types[0],
            touch + self.types[1],
            self.action_in(actions[..., :8].float()) + self.action_position + self.types[2],
            self.state_in(state[..., :8].float())[:, None] + self.types[3],
        ], dim=1)
        queries = self.query.expand(tactile.shape[0], -1, -1) + touch
        return self.output(self.decoder(queries, memory))


class ContactModules(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.updater = BeliefUpdater(cfg)
        self.corrector = FutureCorrector(cfg)
        self.predictor = TactileWorldModel(cfg) if cfg.independent_tactile else ContactPredictor(cfg)
        self.to_queries = nn.Linear(cfg.width, cfg.vision_width)
        self.to_state = nn.Linear(cfg.width, cfg.world_width)
        self.to_expert = nn.Sequential(nn.LayerNorm(cfg.width), nn.Linear(cfg.width, cfg.expert_width))
        self.query_gate = nn.Parameter(torch.tensor(.1))
        self.state_gate = nn.Parameter(torch.tensor(.1))
