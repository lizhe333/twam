import pytest
import torch
from torch import nn

from contact_belief_wam.debug_optimizer import CPUOffloadedAdamW


def _optimizer(parameters, *, lr=0.03):
    return CPUOffloadedAdamW(
        parameters,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        foreach=False,
    )


def test_cpu_offloaded_adamw_matches_cpu_adamw_for_two_updates():
    initial = torch.tensor([[0.4, -0.2], [0.7, 1.1]], dtype=torch.float32)
    ordinary_parameter = nn.Parameter(initial.clone())
    offloaded_parameter = nn.Parameter(initial.clone())
    ordinary = torch.optim.AdamW(
        [ordinary_parameter], lr=0.03, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=0.01, foreach=False,
    )
    offloaded = _optimizer([offloaded_parameter])

    for gradient in (
        torch.tensor([[0.2, -0.1], [0.5, 0.4]]),
        torch.tensor([[-0.3, 0.7], [0.1, -0.2]]),
    ):
        ordinary_parameter.grad = gradient.clone()
        offloaded_parameter.grad = gradient.clone()
        ordinary.step()
        offloaded.step()
        ordinary.zero_grad(set_to_none=True)
        offloaded.zero_grad(set_to_none=True)
        torch.testing.assert_close(offloaded_parameter, ordinary_parameter, rtol=0, atol=0)

    assert offloaded.step_count == 2
    assert offloaded_parameter.grad is None
    assert all(parameter.device.type == "cpu" for group in offloaded.param_groups for parameter in group["params"])
    assert all(
        value.device.type == "cpu"
        for state in offloaded.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )


def test_optimizer_checkpoint_restores_moments_lr_and_reclones_model_weights():
    reference_parameter = nn.Parameter(torch.tensor([0.5, -0.4]))
    first_parameter = nn.Parameter(reference_parameter.detach().clone())
    resumed_parameter = nn.Parameter(torch.tensor([99.0, -99.0]))
    reference = torch.optim.AdamW(
        [reference_parameter], lr=0.02, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=0.01, foreach=False,
    )
    first = _optimizer([first_parameter], lr=0.02)

    first_gradient = torch.tensor([0.3, -0.6])
    reference_parameter.grad = first_gradient.clone()
    first_parameter.grad = first_gradient.clone()
    reference.step()
    first.step()
    reference.zero_grad(set_to_none=True)
    first.zero_grad(set_to_none=True)
    torch.testing.assert_close(first_parameter, reference_parameter, rtol=0, atol=0)
    checkpoint_weights = first_parameter.detach().clone()
    optimizer_checkpoint = first.state_dict()

    with torch.no_grad():
        resumed_parameter.copy_(checkpoint_weights)
    resumed = _optimizer([resumed_parameter], lr=0.9)
    resumed.load_state_dict(optimizer_checkpoint)

    assert resumed.step_count == 1
    assert resumed.param_groups[0]["lr"] == pytest.approx(0.02)
    torch.testing.assert_close(resumed.param_groups[0]["params"][0], checkpoint_weights)

    second_gradient = torch.tensor([-0.2, 0.1])
    reference_parameter.grad = second_gradient.clone()
    resumed_parameter.grad = second_gradient.clone()
    reference.step()
    resumed.step()
    torch.testing.assert_close(resumed_parameter, reference_parameter, rtol=0, atol=0)
    assert resumed.step_count == 2


def test_lambda_lr_can_schedule_cpu_parameter_groups():
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = _optimizer([{"params": [parameter], "lr": 0.04, "name": "debug"}])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1))

    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()

    assert optimizer.param_groups[0]["name"] == "debug"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.02)


def test_low_precision_parameters_keep_standard_adamw_moment_dtype():
    parameter = nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.bfloat16))
    optimizer = _optimizer([parameter], lr=0.01)
    parameter.grad = torch.tensor([0.2, -0.3], dtype=torch.bfloat16)

    optimizer.step()

    master = optimizer.param_groups[0]["params"][0]
    state = optimizer.state[master]
    assert master.dtype == torch.bfloat16
    assert state["exp_avg"].dtype == torch.bfloat16
    assert state["exp_avg_sq"].dtype == torch.bfloat16
    assert state["step"].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not visible")
def test_cuda_model_parameters_receive_cpu_adamw_updates():
    initial = torch.tensor([0.3, -0.8, 1.2])
    ordinary_parameter = nn.Parameter(initial.clone())
    cuda_parameter = nn.Parameter(initial.to("cuda"))
    ordinary = torch.optim.AdamW([ordinary_parameter], lr=0.01, foreach=False)
    offloaded = _optimizer([cuda_parameter], lr=0.01)
    gradient = torch.tensor([0.4, 0.2, -0.7])

    ordinary_parameter.grad = gradient.clone()
    cuda_parameter.grad = gradient.to("cuda")
    ordinary.step()
    offloaded.step()

    torch.testing.assert_close(cuda_parameter.cpu(), ordinary_parameter, rtol=0, atol=0)
    assert all(parameter.device.type == "cpu" for group in offloaded.param_groups for parameter in group["params"])
