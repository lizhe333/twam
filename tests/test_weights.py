from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from contact_belief_wam.weights import load_world


def model():
    core = nn.Module()
    core.world_model = nn.Module()
    core.world_model.camera_embedding = nn.Parameter(torch.zeros(1, 1, 1, 4))
    core.world_model.future_queries = nn.Parameter(torch.zeros(1, 1, 16, 4))
    core.world_model.body = nn.Linear(4, 4)
    core.world_action_encoder = nn.Module()
    core.world_action_encoder.position_embedding = nn.Parameter(torch.zeros(1, 50, 4))
    core.world_action_encoder.input_projection = nn.Sequential(nn.Linear(64, 4))
    core.world_state_encoder = nn.Module()
    core.world_state_encoder.projection = nn.Sequential(nn.Linear(32, 4))
    return SimpleNamespace(core=core)


def source_tensors(target):
    source = {"model." + k: torch.ones_like(v) for k, v in target.core.state_dict().items()}
    source["model.world_model.camera_embedding"] = torch.arange(3.).reshape(1, 3, 1, 1).expand(1, 3, 1, 4).contiguous()
    source["model.world_model.future_queries"] = torch.ones(1, 3, 16, 4)
    source["model.world_action_encoder.position_embedding"] = torch.ones(1, 32, 4)
    return source


def test_load_world_preserves_compatible_padded_encoders(tmp_path):
    target = model()
    save_file(source_tensors(target), str(tmp_path / "model.safetensors"))
    report = load_world(target, tmp_path)
    assert report["reinitialized"] == []
    assert len(report["converted"]) == 3
    assert torch.equal(target.core.world_state_encoder.projection[0].weight, torch.ones(4, 32))
    positions = target.core.world_action_encoder.position_embedding
    assert torch.all(positions[:, :32] == 1)
    assert torch.all(positions[:, 32:] == 0)
    assert torch.all(target.core.world_model.camera_embedding == 0)


def test_load_world_reinitializes_only_incompatible_input_projection(tmp_path):
    target = model()
    source = source_tensors(target)
    source["model.world_state_encoder.projection.0.weight"] = torch.ones(4, 14)
    original = target.core.world_state_encoder.projection[0].weight.detach().clone()
    save_file(source, str(tmp_path / "model.safetensors"))
    report = load_world(target, tmp_path)
    assert [item["key"] for item in report["reinitialized"]] == ["world_state_encoder.projection.0.weight"]
    torch.testing.assert_close(target.core.world_state_encoder.projection[0].weight, original)


def test_load_world_rejects_missing_or_incompatible_body(tmp_path):
    target = model()
    source = source_tensors(target)
    source.pop("model.world_model.body.bias")
    save_file(source, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="missing world_model.body.bias"):
        load_world(target, tmp_path)
    source = source_tensors(target)
    source["model.world_model.body.weight"] = torch.ones(4, 5)
    save_file(source, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="Incompatible world tensor"):
        load_world(target, tmp_path)
