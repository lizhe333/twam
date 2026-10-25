from copy import deepcopy
import json

import pytest

from contact_belief_wam.aligned_debug import (
    _eligible_indices,
    _functional_call_cpu_masters,
    _minimum_sequence_length,
    _prediction_values,
    _resume_checkpoint_path,
    _state_digest,
    debug_settings,
    reject_formal_output,
    select_debug_groups,
    validate_debug_micro_batch,
    validate_debug_updates,
)

import torch
from torch import nn


def test_debug_update_budget_is_strictly_bounded():
    assert validate_debug_updates(1) == 1
    assert validate_debug_updates(2) == 2
    for value in (0, 3, -1, 1.5, True):
        with pytest.raises(ValueError, match="one or two updates"):
            validate_debug_updates(value)


def test_debug_micro_batch_only_allows_realistic_single_or_rank_pair():
    assert validate_debug_micro_batch(1) == 1
    assert validate_debug_micro_batch(2) == 2
    for value in (0, 3, 1.5, False):
        with pytest.raises(ValueError, match="must be 1 or 2"):
            validate_debug_micro_batch(value)


def test_stage1_debug_pairs_same_task_originals_and_adds_recovery():
    tasks = ["insert_hole", "insert_hole", "insert_hole", "insert_hole", "insert_HDMI"]
    sources = ["original", "original", "recovery", "original", "original"]
    episodes = ["orig-a", "orig-b", "recovery-a", "orig-c", "hdmi-a"]

    groups = select_debug_groups(tasks, sources, episodes, stage="stage1",
                                 shared_tasks=("insert_hole", "insert_HDMI"))

    assert [(group["source"], group["task"]) for group in groups] == [
        ("original", "insert_hole"), ("recovery", "insert_hole")
    ]
    assert len(groups[0]["indices"]) == 2
    assert len(set(groups[0]["episode_ids"])) == 2
    assert groups[0]["episode_ids"] == ["orig-a", "orig-b"]
    assert groups[1]["episode_ids"] == ["recovery-a"]


def test_stage1_debug_rejects_unpaired_or_cross_task_ranking_examples():
    with pytest.raises(ValueError, match="two same-task original"):
        select_debug_groups(["insert_hole", "insert_HDMI"], ["original", "original"],
                            ["a", "b"], stage="stage1",
                            shared_tasks=("insert_hole", "insert_HDMI"))


def test_stage2_debug_requires_the_requested_task_in_both_sources():
    groups = select_debug_groups(
        ["insert_hole", "insert_hole", "insert_HDMI", "insert_HDMI"],
        ["original", "recovery", "original", "recovery"],
        ["hole-orig", "hole-rec", "hdmi-orig", "hdmi-rec"],
        stage="stage2", task="insert_HDMI")

    assert [(group["source"], group["episode_ids"]) for group in groups] == [
        ("original", ["hdmi-orig"]), ("recovery", ["hdmi-rec"])
    ]
    with pytest.raises(ValueError, match="no recovery episode"):
        select_debug_groups(["insert_hole"], ["original"], ["hole-orig"],
                            stage="stage2", task="insert_hole")


def test_debug_settings_preserve_the_formal_recipe():
    run = {"variant": "full", "mode": "full", "output": "/tmp/formal",
           "gradient_checkpointing": False, "ema_decay": 0.99,
           "stage1": {"num_gpus": 8, "micro_batch": 32, "global_batch": 256, "steps": 5000},
           "stage2": {"num_gpus": 2, "micro_batch": 32, "global_batch": 64, "steps": 10000}}
    before = deepcopy(run)

    settings = debug_settings(run, updates=2, device="cuda", optimizer_device="cpu",
                              micro_batch=2, threads=4)

    assert run == before
    assert settings["formal_training"] is False
    assert settings["optimizer_device"] == "cpu"
    assert settings["ema_device"] == "cpu"
    assert settings["formal_recipe"]["stage1"] == before["stage1"]
    assert settings["formal_recipe"]["stage2"] == before["stage2"]
    assert settings["formal_recipe"]["stage1"]["micro_batch"] == 32


def test_debug_rejects_a_formal_training_output_tree(tmp_path):
    formal = tmp_path / "formal"
    formal.mkdir()
    (formal / "tasks" / "insert_hole").mkdir(parents=True)
    (formal / "tasks" / "insert_hole" / "run_config.yaml").write_text("formal: true\n")

    with pytest.raises(ValueError, match="formal training state"):
        reject_formal_output(formal)

    reject_formal_output(tmp_path / "fresh-debug")


def test_wave0_recovery_needs_only_its_single_real_sequence_step():
    class Config:
        k = 10
        belief_bptt_steps = 1

    class Dataset:
        index = [(0, 10, 100, 10), (0, 10, 100, 20)]

    assert _minimum_sequence_length(Config()) == 1
    assert _eligible_indices(Dataset(), Config()) == [0, 1]


def test_full_mode_requires_two_positions_for_truncated_bptt():
    class Config:
        belief_bptt_steps = 2

    assert _minimum_sequence_length(Config()) == 2


def test_cpu_master_functional_call_keeps_gradients_on_original_parameters():
    class TinyLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(2.0))

        def forward(self, value):
            return (self.scale * value).sum()

    module = TinyLoss()
    value = torch.tensor([1.0, 3.0])
    output = _functional_call_cpu_masters(module, (value,), "cpu", dtype=torch.float32,
                                          save_activations=False)
    output.backward()

    assert output.item() == 8.0
    assert module.scale.grad.item() == 4.0


def test_state_digest_handles_scalar_tensors():
    assert _state_digest({"step": torch.tensor(3), "weight": torch.ones(2)})


def _saved_debug_checkpoint(tmp_path):
    checkpoint = tmp_path / "stage2" / "checkpoints" / "step_00000001"
    model = checkpoint / "pretrained_model"
    model.mkdir(parents=True)
    (checkpoint / "training.pt").write_bytes(b"checkpoint placeholder")
    for name in ("config.json", "model.safetensors", "norm_stats.json", "wm_norm_stats.json",
                 "tokenizer.model", "deployment.json", "ema.safetensors"):
        (model / name).write_bytes(b"export placeholder")
    (checkpoint / "complete.json").write_text(json.dumps({
        "formal_training": False, "stage": "stage2", "task": "insert_hole",
        "step": 1, "world_size": 1, "training_contract": "bounded-debug-only"}))
    run = {"output": str(tmp_path), "variant": "wave0_recovery", "mode": "wave0",
           "stage2": {"lr": 2.5e-5}}
    debug = {"updates": 1, "device": "cuda", "optimizer_device": "cpu",
             "cpu_master_functional_offload": True, "micro_batch": 1, "threads": 4,
             "sdpa_backends": ["math", "flash_attention"],
             "efficient_sdpa_disabled_for_offload": True, "ema_device": "cpu"}
    report = {"complete": False, "formal_training": False, "variant": run["variant"],
              "mode": run["mode"], "stage": "stage2", "task": "insert_hole",
              "debug": debug, "formal_recipe": run["stage2"], "updates": []}
    return run, debug, report, checkpoint


def test_resume_accepts_complete_checkpoint_with_incomplete_or_missing_report(tmp_path):
    run, debug, report, checkpoint = _saved_debug_checkpoint(tmp_path)

    assert _resume_checkpoint_path(run, "stage2", "insert_hole", 1, report, debug) == checkpoint
    assert _resume_checkpoint_path(run, "stage2", "insert_hole", 1, None, debug) == checkpoint


def test_resume_rejects_mismatched_or_corrupt_checkpoint_state(tmp_path):
    run, debug, report, checkpoint = _saved_debug_checkpoint(tmp_path)

    with pytest.raises(RuntimeError, match="does not match this run"):
        _resume_checkpoint_path(run, "stage2", "insert_hole", 1,
                                {**report, "debug": {**debug, "updates": 2}}, debug)
    with pytest.raises(RuntimeError, match="disagrees with its checkpoint step"):
        _resume_checkpoint_path(run, "stage2", "insert_hole", 1,
                                {**report, "updates": [{"update": 1}, {"update": 2}]}, debug)
    (checkpoint / "complete.json").write_text(json.dumps({"formal_training": False}))
    with pytest.raises(RuntimeError, match="marker does not match"):
        _resume_checkpoint_path(run, "stage2", "insert_hole", 1, None, debug)


def test_stage1_fixed_noise_probe_includes_world_prediction_for_wave0_path():
    class Config:
        k = 1

    class WorldActionEncoder(nn.Module):
        def forward(self, actions, state):
            return actions + state[:, None, :]

    class Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.world_action_encoder = WorldActionEncoder()

    class Policy:
        config = Config()
        core = Core()

        def update_belief(self, obs):
            return torch.zeros(obs.state.shape[0], 1, 1)

        def encode_visual(self, image):
            return image.mean(dim=(1, 2, 3))[:, None, None]

        def predict_future(self, current, state, tokens, belief, tactile, *, actions=None):
            initial = current + tokens.mean(dim=(1, 2))[:, None, None]
            return initial, initial, None

    batch = {
        "sequence_mask": torch.tensor([[True, True]]),
        "step": torch.tensor([[0, 1]]),
        "image": torch.ones(1, 2, 3, 2, 2),
        "state": torch.zeros(1, 2, 8),
        "world_state": torch.zeros(1, 2, 8),
        "tactile": torch.zeros(1, 2, 1, 1),
        "tokens": torch.zeros(1, 2, 1),
        "token_mask": torch.ones(1, 2, 1, dtype=torch.bool),
        "executed_actions": torch.zeros(1, 2, 1, 8),
        "executed_valid": torch.ones(1, 2, 1, dtype=torch.bool),
        "world_actions": torch.ones(1, 2, 1, 8),
    }

    result = _prediction_values(Policy(), batch, None, "stage1", torch.device("cpu"))

    assert {"belief", "world_initial", "world_corrected"} <= set(result)
    assert result["world_initial"].shape == (1, 1, 1)
    assert torch.count_nonzero(result["world_initial"]) > 0
