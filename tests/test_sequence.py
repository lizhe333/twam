from types import SimpleNamespace

import torch
from torch import nn

from contact_belief_wam.sequence import SequenceTrainingModel


def _batch(sequence_mask, step=None, future_mask=None):
    sequence_mask = torch.as_tensor(sequence_mask, dtype=torch.bool)
    batch_size, length = sequence_mask.shape
    if step is None:
        step = torch.arange(length).mul(2).expand(batch_size, -1).clone()
    else:
        step = torch.as_tensor(step, dtype=torch.long)
    if future_mask is None:
        future_mask = torch.ones(batch_size, length, dtype=torch.bool)
    else:
        future_mask = torch.as_tensor(future_mask, dtype=torch.bool)

    scalar = torch.arange(batch_size * length, dtype=torch.float32).reshape(batch_size, length, 1)
    action_prefix = 400 + torch.arange(batch_size * length, dtype=torch.float32).reshape(
        batch_size, length, 1, 1) * 10 + torch.arange(2, dtype=torch.float32).reshape(1, 1, 2, 1)
    action_chunk = torch.full((batch_size, length, 3, 1), -999.0)
    action_chunk[:, :-1, :2] = action_prefix[:, 1:]
    action_chunk[:, :, 2] = 9000 + scalar
    return {
        "sequence_mask": sequence_mask,
        "step": step,
        "state": scalar,
        "image": scalar + 10,
        "tactile": scalar + 20,
        "tokens": scalar + 30,
        "token_mask": torch.ones(batch_size, length, 1, dtype=torch.bool),
        "executed_actions": action_prefix,
        "executed_valid": torch.ones(batch_size, length, 2, dtype=torch.bool),
        "actions": action_chunk,
        "action_mask": torch.ones(batch_size, length, 3, dtype=torch.bool),
        "future_image": scalar + 10000,
        "future_tactile": scalar + 20000,
        "future_mask": future_mask,
    }


class TinyPolicy(nn.Module):
    """Small control-flow probe; it does not implement or validate the WAVE model."""

    def __init__(self, length, *, bptt_steps=2, outputs=None, final_future_from_belief=False):
        super().__init__()
        self.config = SimpleNamespace(k=2, belief_bptt_steps=bptt_steps)
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.outputs = outputs
        self.final_future_from_belief = final_future_from_belief
        self.length = length
        self.history_calls = []
        self.forward_calls = []
        self.train_beliefs = []

    @staticmethod
    def _base(belief, state):
        return torch.zeros_like(state[:, :1]) if belief is None else belief

    def history_step(self, observation, executed_prefix):
        # This deliberately accepts only the observation and executed actions;
        # future ContactTargets must never be passed on the history path.
        self.history_calls.append({
            "step": observation.step.detach().clone(),
            "state": observation.state.detach().clone(),
            "action": executed_prefix.detach().clone(),
            "grad_enabled": torch.is_grad_enabled(),
            "belief_valid": observation.belief_valid.detach().clone(),
            "has_future_image": hasattr(observation, "future_image"),
            "has_future_tactile": hasattr(observation, "future_tactile"),
        })
        belief = self._base(observation.belief, observation.state) + self.weight
        expected = self.weight.expand(observation.state.shape[0], 1)
        return belief, expected

    def forward(self, observation, targets):
        call_index = len(self.forward_calls)
        self.forward_calls.append({
            "step": observation.step.detach().clone(),
            "expected_valid": observation.expected_valid.detach().clone(),
            "grad_enabled": torch.is_grad_enabled(),
            "targets": targets,
        })
        belief = self._base(observation.belief, observation.state) + self.weight
        if belief.requires_grad:
            belief.retain_grad()
        self.train_beliefs.append(belief)
        expected = self.weight.expand(observation.state.shape[0], 1)

        if self.outputs is not None:
            action_loss = torch.as_tensor(self.outputs[call_index]["action"], dtype=torch.float32)
            future_loss = torch.as_tensor(self.outputs[call_index]["future"], dtype=torch.float32)
        else:
            action_loss = torch.zeros(observation.state.shape[0])
            if self.final_future_from_belief and call_index == 1:
                future_loss = belief[:, 0]
            else:
                future_loss = torch.zeros(observation.state.shape[0])
        return {
            "belief": belief,
            "expected_contact": expected,
            "action_loss_per_sample": action_loss,
            "future_loss_per_sample": future_loss,
            "metrics": {},
        }


def test_left_padding_skips_history_and_only_last_two_updates_keep_bptt_gradients():
    batch = _batch([[False, False, True, True, True]])
    policy = TinyPolicy(length=5, final_future_from_belief=True)
    result = SequenceTrainingModel(policy)(batch)

    assert len(policy.history_calls) == 1
    history = policy.history_calls[0]
    assert history["step"].tolist() == [4]
    assert history["state"].tolist() == [[2.0]]
    assert history["grad_enabled"] is False
    assert history["belief_valid"].tolist() == [False]
    # At historical step i, use the prefix already executed at observation i+1.
    # It equals the first k targets for i, while excluding same-observation
    # actions (the preceding interval) and the rest of the future action chunk.
    expected_prefix = batch["executed_actions"][:, 3]
    torch.testing.assert_close(batch["actions"][:, 2, :2], expected_prefix)
    torch.testing.assert_close(history["action"], expected_prefix)
    assert not torch.equal(history["action"], batch["executed_actions"][:, 2])
    assert history["action"].shape[1] == 2
    assert not history["has_future_image"]
    assert not history["has_future_tactile"]
    assert len(policy.forward_calls) == 2
    assert all(call["grad_enabled"] for call in policy.forward_calls)
    assert all(call["targets"].future_image.min() >= 10000 for call in policy.forward_calls)
    torch.testing.assert_close(result["belief"], torch.tensor([[3.0]]))

    result["loss"].backward()
    # The final update's loss reaches the previous trainable update, while the
    # earlier history step was evaluated under no_grad.
    torch.testing.assert_close(policy.train_beliefs[0].grad, torch.tensor([[0.5]]))
    torch.testing.assert_close(policy.weight.grad, torch.tensor(1.0))


def test_expected_contact_is_invalid_when_target_step_does_not_match():
    batch = _batch([[True, True, True, True]], step=[[0, 2, 4, 7]])
    policy = TinyPolicy(length=4)
    SequenceTrainingModel(policy)(batch)

    assert len(policy.forward_calls) == 2
    assert [call["step"].item() for call in policy.forward_calls] == [4, 7]
    assert [call["expected_valid"].item() for call in policy.forward_calls] == [True, False]


def test_future_loss_is_reduced_only_over_each_samples_effective_future_mask():
    batch = _batch(
        [[True, True], [True, True], [True, True]],
        future_mask=[[True, False], [False, True], [False, False]],
    )
    policy = TinyPolicy(
        length=2,
        outputs=[
            {"action": [1, 2, 3], "future": [10, 20, 30]},
            {"action": [3, 4, 5], "future": [100, 200, 300]},
        ],
    )
    result = SequenceTrainingModel(policy)(batch)

    # Action loss: mean over time per sample, then batch mean = 3.
    # Future loss: sample means are 10, 200, and masked out; then mean
    # over the two samples with at least one valid future target = 105.
    torch.testing.assert_close(result["loss"], torch.tensor(108.0))


def test_bf16_warmup_does_not_reuse_detached_linear_weight_casts():
    class LinearMemory(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(k=2, belief_bptt_steps=2)
            self.projection = nn.Linear(1, 1)
            self.history_had_grad = []

        def history_step(self, obs, actions):
            value = self.projection(obs.state)
            self.history_had_grad.append(value.requires_grad)
            return value, None

        def forward(self, obs, targets):
            value = self.projection(obs.state) + obs.belief
            return {"belief": value, "expected_contact": None,
                    "action_loss_per_sample": value[:, 0].float().square(),
                    "future_loss_per_sample": torch.zeros_like(value[:, 0]), "metrics": {}}

    policy = LinearMemory()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = SequenceTrainingModel(policy)(_batch([[True, True, True, True, True]]))
    result["loss"].backward()
    assert policy.history_had_grad == [False, False, False]
    assert policy.projection.weight.grad is not None
    assert torch.isfinite(policy.projection.weight.grad).all()
    assert policy.projection.weight.grad.abs().sum() > 0

