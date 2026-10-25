import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from contact_belief_wam.legacy.trainer import (
    check_resume_contract,
    optimizer_for,
    training_contract,
)


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.contact = nn.Linear(2, 2)
        self.policy_layer = nn.Linear(2, 2)
        self.shared = nn.Module()
        self.shared.wam_projection = nn.Linear(2, 2)


def _run(**overrides):
    return {"lr_new": 0.02, "lr_policy": 0.003, **overrides}


def _contract_run(tmp_path, **overrides):
    manifest = tmp_path / "manifest.jsonl"
    stats = tmp_path / "quantiles.json"
    manifest.write_text("{}\n")
    stats.write_text("{}\n")
    return {"manifest": str(manifest), "stats": str(stats), **overrides}


def test_optimizer_cpu_path_keeps_original_adamw_options_and_parameter_groups():
    policy = _Policy()
    optimizer = optimizer_for(policy, _run(fused_adamw=True), device="cpu")

    assert optimizer.defaults["betas"] == (0.9, 0.95)
    assert optimizer.defaults["eps"] == 1e-8
    assert optimizer.defaults["weight_decay"] == 0.01
    assert optimizer.defaults.get("fused") is not True
    assert optimizer.defaults.get("foreach") is None
    assert [group["name"] for group in optimizer.param_groups] == ["new", "policy"]
    assert [group["lr"] for group in optimizer.param_groups] == [0.02, 0.003]
    assert set(optimizer.param_groups[0]["params"]) == {
        *policy.contact.parameters(), *policy.shared.wam_projection.parameters()
    }
    assert set(optimizer.param_groups[1]["params"]) == set(policy.policy_layer.parameters())


def test_fused_is_selected_only_for_requested_available_cuda_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    run = _run(fused_adamw=True)

    cpu_optimizer = optimizer_for(_Policy(), run, device=torch.device("cpu"))
    requested_cuda_optimizer = optimizer_for(_Policy(), run, device=torch.device("cuda:0"))
    disabled_optimizer = optimizer_for(_Policy(), _run(fused_adamw=False), device="cuda:0")

    assert cpu_optimizer.defaults.get("fused") is not True
    assert requested_cuda_optimizer.defaults["fused"] is True
    assert requested_cuda_optimizer.defaults["foreach"] is None
    assert disabled_optimizer.defaults.get("fused") is not True


def test_optimizer_state_dict_can_resume_on_cpu():
    torch.manual_seed(31)
    policy = _Policy()
    optimizer = optimizer_for(policy, _run(), device="cpu")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))

    for parameter in policy.parameters():
        parameter.grad = torch.full_like(parameter, 0.1)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    # A serialized checkpoint owns its tensors; state_dict() by itself can
    # still reference the live optimizer's moment/step tensors.
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    model_state = {key: value.detach().clone() for key, value in policy.state_dict().items()}

    restored = _Policy()
    restored.load_state_dict(model_state)
    restored_optimizer = optimizer_for(restored, _run(), device="cpu")
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer, lambda step: 1 / (step + 1)
    )
    restored_optimizer.load_state_dict(optimizer_state)
    restored_scheduler.load_state_dict(scheduler.state_dict())

    assert [group["lr"] for group in restored_optimizer.param_groups] == [
        group["lr"] for group in optimizer.param_groups
    ]
    for first, second in zip(policy.parameters(), restored.parameters()):
        torch.testing.assert_close(first, second)
    for parameter in policy.parameters():
        parameter.grad = torch.full_like(parameter, -0.2)
    for parameter in restored.parameters():
        parameter.grad = torch.full_like(parameter, -0.2)
    optimizer.step()
    restored_optimizer.step()
    for first, second in zip(policy.parameters(), restored.parameters()):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_resume_accepts_legacy_contract_without_fused_adamw(tmp_path):
    run = _contract_run(tmp_path)
    legacy_contract = training_contract(run)

    assert legacy_contract["fused_adamw"] is False
    del legacy_contract["fused_adamw"]
    check_resume_contract(run, {"training_contract": legacy_contract})


def test_resume_rejects_changing_fused_adamw_setting(tmp_path):
    saved_run = _contract_run(tmp_path, fused_adamw=False)
    saved_contract = training_contract(saved_run)
    current_run = _contract_run(tmp_path, fused_adamw=True)

    with pytest.raises(ValueError, match="fused_adamw"):
        check_resume_contract(current_run, {"training_contract": saved_contract})
