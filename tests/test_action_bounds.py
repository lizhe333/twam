from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from contact_belief_wam.model import ContactWAM
from contact_belief_wam.preprocessing import Quantiles
from contact_belief_wam.serve import PolicyService
from contact_belief_wam.sim_eval import Adapter, robot_action_bounds
from contact_belief_wam.types import ContactObservation, ContactPrediction


def _bounds(low=-1.0, high=1.0, gripper_high=2.0):
    return np.asarray([[low, high]] * 7 + [[0.0, gripper_high]], dtype=np.float32)


class _FakeServiceModel:
    def __init__(self, actions):
        self.actions = actions
        self.seen_action_transform = False

    def predict_action_chunk(self, observation, action_transform=None):
        if action_transform is not None:
            self.seen_action_transform = True
            actions = action_transform(self.actions)
        else:
            actions = self.actions
        belief = torch.zeros((1, 4, 8), dtype=torch.float32)
        return ContactPrediction(actions, belief, None, torch.tensor([2]), {})


def _fake_service(monkeypatch, bounds, qpos):
    from contact_belief_wam import serve as serve_module

    monkeypatch.setattr(serve_module, "vision_preprocess",
                        lambda images: torch.zeros((1, 3, 224, 224), dtype=torch.float32))
    normalized = (2 * (qpos - 0.0) / 10.0 - 1.0).astype(np.float32)
    policy = _FakeServiceModel(torch.from_numpy(normalized[None]))
    service = object.__new__(PolicyService)
    service.device = torch.device("cpu")
    service.model = policy
    service.cfg = SimpleNamespace(k=2, dtype="float32")
    service.norm = Quantiles({"action": {"q01": [0.0] * 8, "q99": [10.0] * 8},
                              "state": {"q01": [0.0] * 8, "q99": [10.0] * 8}})
    service.tokenizer = lambda prompts, states: (torch.zeros((1, 2), dtype=torch.long),
                                                  torch.ones((1, 2), dtype=torch.bool))
    service.encoder = None
    service.sessions = {}
    service.metadata = {"mode": "full", "k": 2}
    service.reset("episode", action_bounds=bounds)
    return service, policy


def test_service_clips_absolute_targets_before_model_feedback(monkeypatch):
    bounds = _bounds(low=1, high=9, gripper_high=4)
    qpos = np.full((50, 8), 5.0, dtype=np.float32)
    qpos[:, 0] = 9.01
    qpos[:, 7] = 8.0
    service, model = _fake_service(monkeypatch, bounds, qpos)

    result = service.infer({"episode": "episode", "step": 0, "image": np.zeros((224, 224, 3), np.uint8),
                            "state": np.full(8, 5, np.float32), "tactile_window": np.zeros((2, 4, 240, 320, 3), np.uint8),
                            "prompt": "test"})

    assert model.seen_action_transform
    assert result["actions"].shape == (50, 8)
    np.testing.assert_allclose(result["actions"][:, 0], 9.0, atol=1e-6)
    np.testing.assert_allclose(result["actions"][:, 7], 4.0, atol=1e-6)
    np.testing.assert_allclose(result["arm_limit_overshoots"], 0.01, atol=2e-5)


def test_service_without_bounds_keeps_baseline_predict_call(monkeypatch):
    bounds = None
    qpos = np.full((50, 8), 5.0, dtype=np.float32)
    service, model = _fake_service(monkeypatch, bounds, qpos)
    service.sessions["episode"].last_step = -1

    result = service.infer({"episode": "episode", "step": 0, "image": np.zeros((224, 224, 3), np.uint8),
                            "state": np.full(8, 5, np.float32), "tactile_window": np.zeros((2, 4, 240, 320, 3), np.uint8),
                            "prompt": "test"})

    assert not model.seen_action_transform
    assert "arm_limit_overshoots" not in result
    np.testing.assert_allclose(result["actions"], qpos, atol=1e-6)


def test_predict_action_chunk_transforms_before_world_action_encoder():
    cfg = SimpleNamespace(stage="stage2", world_only=False, mode="full", use_world=True, belief_to_expert=True,
                          chunk_size=50, k=2, num_inference_steps=1)
    captured = {}

    class FakeCore:
        @staticmethod
        def sample_noise(shape, device):
            return torch.ones(shape, device=device)

        @staticmethod
        def world_action_encoder(actions, state):
            captured["encoded_actions"] = actions.clone()
            return actions

    fake = SimpleNamespace(config=cfg, core=FakeCore(),
                           eval=lambda: None,
                           to_world_actions=lambda actions: actions,
                           update_belief=lambda obs: torch.ones((1, 4, 8)),
                           encode_visual=lambda image: torch.ones((1, 2, 8)),
                           policy_condition=lambda obs, belief, current: (torch.ones((1, 1, 8)), None, None, None, None),
                           expert_velocity=lambda pad, kv, condition, actions, time: torch.zeros_like(actions),
                           predict_future=lambda current, state, latent, belief, tactile, *, actions=None:
                               (None, None, torch.ones((1, 8, 8))))
    obs = ContactObservation(image=torch.zeros((1, 3, 8, 8)), state=torch.zeros((1, 8)), tactile=None,
                             step=torch.tensor([3]))

    prediction = ContactWAM.predict_action_chunk(fake, obs, action_transform=lambda actions: torch.zeros_like(actions))

    assert torch.count_nonzero(captured["encoded_actions"][:, :, :8]) == 0
    assert torch.count_nonzero(prediction.actions) == 0


class _FakeClient:
    metadata = {"k": 2}

    def __init__(self, reported_overshoot=0.0):
        self.calls = []
        self.reported_overshoot = reported_overshoot

    def call(self, **message):
        self.calls.append(message)
        if message["op"] == "infer":
            actions = np.full((50, 8), 0.5, dtype=np.float32)
            return {"actions": actions, "diagnostics": {},
                    "arm_limit_overshoots": np.asarray([self.reported_overshoot] + [0.0] * 49, np.float32)}
        return {"reset": message["episode"]}


def _fake_task():
    manager = SimpleNamespace(
        robot=SimpleNamespace(data=SimpleNamespace(joint_pos_limits=torch.tensor([[[-1.0, 1.0]] * 7]))),
        _arm_ids=torch.arange(7), gripper_max_qpos=2.0,
    )
    return SimpleNamespace(_robot_manager=manager)


def _observation():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    tactile = np.zeros((8, 8, 3), dtype=np.uint8)
    return {"tactile": {"left_tactile": {"rgb_marker": tactile}, "right_tactile": {"rgb_marker": tactile}},
            "observation": {"head": {"rgb": image}},
            "embodiment": {"joint": np.zeros(9, dtype=np.float32)}}


def test_adapter_sends_bounds_and_preserves_arm_overshoot_error():
    task = _fake_task()
    bounds = robot_action_bounds(task)
    client = _FakeClient(reported_overshoot=0.021)
    adapter = Adapter(client, "episode", "prompt", bounds)
    reset = client.calls[0]
    np.testing.assert_array_equal(reset["action_bounds"], bounds)

    with pytest.raises(ValueError, match="outside limits by 0.021000 rad"):
        adapter.next_action(task, _observation())

