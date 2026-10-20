from types import SimpleNamespace

import torch
from torch import nn

from contact_belief_wam.config import ContactConfig
from contact_belief_wam.model import ContactWAM, masked_action_flow, policy_aux_valid
from contact_belief_wam.modules import ContactModules
from contact_belief_wam.sequence import SequenceTrainingModel, observation_at, targets_at
from contact_belief_wam.types import ContactObservation


def _config(**overrides):
    values = {
        "stage": "stage2",
        "mode": "full",
        "k": 2,
        "chunk_size": 4,
        "width": 8,
        "heads": 2,
        "ffn": 16,
        "belief_tokens": 2,
        "world_width": 8,
        "world_heads": 2,
        "world_ffn": 16,
        "vision_width": 8,
        "expert_width": 8,
        "vision_pool": 2,
    }
    values.update(overrides)
    return ContactConfig(**values)


class _FakeCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = nn.Linear(1, 1)
        self.wam_latent_action_queries = nn.Parameter(torch.randn(1, 4, 8))
        self.wam_latent_action_projection = nn.Linear(8, 8)
        self.world_action_encoder = nn.Linear(1, 1)
        self.world_state_encoder = nn.Linear(1, 1)
        self.world_model = nn.Linear(1, 1)
        self.world_vision_backbone = nn.Linear(1, 1)
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = nn.Module()
        self.paligemma_with_expert.paligemma.vision_tower = nn.Linear(1, 1)
        self.paligemma_with_expert.paligemma.multi_modal_projector = nn.Linear(1, 1)


class _FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeCore()


def _configured_model(cfg):
    model = ContactWAM.__new__(ContactWAM)
    nn.Module.__init__(model)
    model.config = cfg
    model.base_policy = _FakePolicy()
    model.contact = ContactModules(cfg)
    model.configure_trainability()
    return model


def test_stage2_freezes_belief_and_predictor_parameters_without_freezing_policy_interfaces():
    cfg = _config(stage2_freeze_belief=True, stage2_freeze_contact_predictor=True)
    model = _configured_model(cfg)
    contact = model.contact

    assert all(not parameter.requires_grad for parameter in contact.updater.parameters())
    assert all(not parameter.requires_grad for parameter in contact.predictor.parameters())
    for module in (contact.corrector, contact.to_queries, contact.to_state, contact.to_expert):
        assert all(parameter.requires_grad for parameter in module.parameters())
    assert contact.query_gate.requires_grad
    assert contact.state_gate.requires_grad


def test_stage2_frozen_modules_keep_input_gradients_and_freezing_defaults_off():
    assert not ContactConfig().stage2_freeze_belief
    assert not ContactConfig().stage2_freeze_contact_predictor

    cfg = _config(stage2_freeze_belief=True, stage2_freeze_contact_predictor=True)
    contact = _configured_model(cfg).contact
    batch = 2
    previous = torch.randn(batch, cfg.belief_tokens, cfg.width, requires_grad=True)
    observation = ContactObservation(
        image=torch.zeros(batch, 3, 8, 8),
        state=torch.randn(batch, 8),
        tactile=torch.randn(batch, 8, cfg.width),
        belief=previous,
        belief_valid=torch.ones(batch, dtype=torch.bool),
        executed_actions=torch.randn(batch, cfg.k, 8),
        executed_valid=torch.ones(batch, cfg.k, dtype=torch.bool),
    )

    updated = contact.updater(observation)
    updated.square().mean().backward()
    assert previous.grad is not None
    assert torch.isfinite(previous.grad).all()
    assert torch.count_nonzero(previous.grad).item() > 0

    belief = torch.randn(batch, cfg.belief_tokens, cfg.width, requires_grad=True)
    future = torch.randn(batch, 2, 2, cfg.vision_width, requires_grad=True)
    actions = torch.randn(batch, cfg.k, cfg.world_width, requires_grad=True)
    tactile = torch.randn(batch, 8, cfg.width, requires_grad=True)
    prediction = contact.predictor(future, actions, tactile, belief)
    prediction.square().mean().backward()
    assert belief.grad is not None
    assert torch.isfinite(belief.grad).all()
    assert torch.count_nonzero(belief.grad).item() > 0


def test_policy_action_mask_intersects_real_action_mask_and_returns_sample_validity():
    flow = torch.tensor([[1.0, 4.0, 9.0], [1.0, 2.0, 3.0], [0.5, 0.5, 8.0]])
    action_mask = torch.tensor([[True, True, False], [True, True, True], [True, True, True]])
    policy_mask = torch.tensor([[False, True, True], [False, False, False], [True, False, True]])

    loss, valid = masked_action_flow(flow, action_mask, policy_mask)

    torch.testing.assert_close(loss, torch.tensor([4.0, 0.0, 4.25]))
    assert valid.tolist() == [True, False, True]


def test_policy_aux_valid_requires_every_first_k_action_to_be_real_and_policy_labeled():
    action_mask = torch.tensor(
        [[True, True, False], [True, False, True], [True, True, True], [True, True, True]]
    )
    policy_mask = torch.tensor(
        [[True, True, False], [True, True, True], [True, False, True], [True, True, False]]
    )

    valid = policy_aux_valid(action_mask, policy_mask, k=2)

    assert valid.tolist() == [True, False, False, True]
    # Omitting the optional mask keeps legacy auxiliary supervision unchanged.
    assert policy_aux_valid(action_mask, None, k=2).tolist() == [True, True, True, True]


def _sequence_batch():
    batch, length, chunk = 3, 2, 3
    policy_mask = torch.tensor(
        [
            [[True, True, False], [False, False, False]],
            [[False, False, False], [False, False, False]],
            [[True, True, False], [False, True, False]],
        ]
    )
    scalar = torch.zeros(batch, length, 1)
    return {
        "sequence_mask": torch.ones(batch, length, dtype=torch.bool),
        "step": torch.arange(length).expand(batch, -1),
        "state": scalar,
        "image": torch.zeros(batch, length, 3, 8, 8),
        "tactile": torch.zeros(batch, length, 8, 8),
        "tokens": torch.zeros(batch, length, 1, dtype=torch.long),
        "token_mask": torch.ones(batch, length, 1, dtype=torch.bool),
        "executed_actions": torch.zeros(batch, length, 2, 8),
        "executed_valid": torch.ones(batch, length, 2, dtype=torch.bool),
        "actions": torch.zeros(batch, length, chunk, 8),
        "action_mask": torch.ones(batch, length, chunk, dtype=torch.bool),
        "policy_action_mask": policy_mask,
        "future_image": torch.zeros(batch, length, 3, 8, 8),
        "future_tactile": torch.zeros(batch, length, 8, 8),
        "future_mask": torch.ones(batch, length, dtype=torch.bool),
        "task": ["insert_hole", "insert_hole", "insert_HDMI"],
    }


class _MaskedLossPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(k=2, belief_bptt_steps=2)
        self.calls = []

    def forward(self, observation, targets):
        call = int(observation.step[0])
        self.calls.append((observation.task, targets.policy_action_mask))
        values = ([2.0, 100.0, 4.0], [6.0, 200.0, 10.0])[call]
        action = torch.tensor(values, dtype=observation.state.dtype)
        valid = targets.action_mask.bool()
        if targets.policy_action_mask is not None:
            policy_mask = targets.policy_action_mask.bool()
            if policy_mask.ndim == 1:
                policy_mask = policy_mask[:, None]
            valid = valid & policy_mask
        aux_valid = policy_aux_valid(targets.action_mask, targets.policy_action_mask, k=2)
        metric = torch.tensor([1.0, 10.0, 100.0])
        return {
            "belief": None,
            "expected_contact": None,
            "action_loss_per_sample": action,
            "action_valid": valid.any(dim=1),
            "policy_aux_valid": aux_valid,
            "future_loss_per_sample": torch.zeros_like(action),
            "metrics": {"alignment": metric, "policy_visual_future": metric,
                        "tactile_future": metric},
        }


def test_sequence_masks_policy_targets_and_excludes_samples_without_any_policy_target():
    batch = _sequence_batch()
    second_targets = targets_at(batch, 1)
    torch.testing.assert_close(second_targets.policy_action_mask, batch["policy_action_mask"][:, 1])
    assert observation_at(batch, 0).task == batch["task"]

    policy = _MaskedLossPolicy()
    result = SequenceTrainingModel(policy)(batch)

    # Sample 0 contributes 2; sample 1 has no policy labels and is excluded;
    # sample 2 averages its two valid time steps (4 and 10) to 7.
    torch.testing.assert_close(result["loss"], torch.tensor(4.5))
    assert all(call[0] == batch["task"] for call in policy.calls)
    torch.testing.assert_close(result["metrics"]["alignment"]["sum"], torch.tensor([1.0, 0.0, 100.0]))
    torch.testing.assert_close(result["metrics"]["alignment"]["count"], torch.tensor([1.0, 0.0, 1.0]))
    torch.testing.assert_close(result["metrics"]["tactile_future"]["count"], torch.tensor([2.0, 2.0, 2.0]))


def test_sequence_keeps_teacher_future_loss_when_policy_auxiliary_is_masked():
    class TeacherOnlyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(k=2, belief_bptt_steps=2)

        def forward(self, observation, targets):
            values = torch.tensor([1.0, 2.0, 3.0])
            return {
                "belief": None,
                "expected_contact": None,
                "action_loss_per_sample": torch.zeros_like(values),
                "action_valid": torch.zeros_like(values, dtype=torch.bool),
                "policy_aux_valid": torch.zeros_like(values, dtype=torch.bool),
                "future_loss_per_sample": values,
                "metrics": {"alignment": values, "tactile_future": values},
            }

    result = SequenceTrainingModel(TeacherOnlyPolicy())(_sequence_batch())

    # Policy auxiliaries have no valid samples, but teacher future losses still
    # reduce over every sample with a valid future target.
    torch.testing.assert_close(result["loss"], torch.tensor(2.0))
    torch.testing.assert_close(result["metrics"]["alignment"]["count"], torch.zeros(3))
    torch.testing.assert_close(result["metrics"]["tactile_future"]["count"], torch.full((3,), 2.0))
