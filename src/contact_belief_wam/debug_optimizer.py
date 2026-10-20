"""CPU-offloaded AdamW for bounded, single-device debug runs.

The optimized model stays on its execution device. Each trainable parameter has
an optimizer-owned CPU clone with the same dtype; gradients and updated values
are copied across only at optimizer steps. This is intended for memory-bounded
debugging and is not used by the formal trainer.

When resuming, load the model checkpoint before loading this optimizer state.
``load_state_dict`` refreshes the CPU parameter clones from the current model
parameters because PyTorch optimizer state dicts contain moments, not weights.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def _materialize_groups(params):
    """Turn parameter iterables into reusable groups while preserving options."""
    values = list(params)
    if not values:
        return []
    if isinstance(values[0], Mapping):
        groups = []
        for value in values:
            group = dict(value)
            if "params" not in group:
                raise ValueError("each parameter group must contain 'params'")
            group["params"] = list(group["params"])
            groups.append(group)
        return groups
    return [{"params": values}]


def _cpu_state(value: Any):
    if torch.is_tensor(value):
        return value.to(device="cpu")
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    return value


class CPUOffloadedAdamW(torch.optim.AdamW):
    """AdamW whose parameters, gradients, and moments live on CPU.

    ``param_groups`` and ``state`` follow the regular PyTorch optimizer API,
    which keeps schedulers such as ``LambdaLR`` compatible. Their parameters
    are CPU clones; original model parameters are available in
    ``original_param_groups`` and receive each updated value after ``step``.
    The clone dtype matches the original parameter dtype rather than using
    fp32 master weights, so the optimizer follows ordinary AdamW numerics for
    the model's configured dtype.
    """

    def __init__(self, params, *args, **kwargs):
        groups = _materialize_groups(params)
        original_groups = []
        cpu_groups = []
        clones = {}

        for group in groups:
            originals = group["params"]
            cpu_params = []
            for parameter in originals:
                if not isinstance(parameter, nn.Parameter):
                    raise TypeError("CPUOffloadedAdamW expects nn.Parameter objects")
                if not parameter.is_leaf:
                    raise ValueError("can't optimize a non-leaf Tensor")
                master = clones.get(id(parameter))
                if master is None:
                    master = nn.Parameter(
                        parameter.detach().to(device="cpu", copy=True),
                        requires_grad=parameter.requires_grad,
                    )
                    clones[id(parameter)] = master
                cpu_params.append(master)
            original_groups.append(originals)
            cpu_groups.append({**group, "params": cpu_params})

        # CPU AdamW's foreach implementation can allocate temporary tensors
        # proportional to optimizer state. Keep the memory use predictable.
        if kwargs.get("foreach") not in (None, False):
            raise ValueError("CPUOffloadedAdamW requires foreach=False")
        if kwargs.get("fused") not in (None, False):
            raise ValueError("CPUOffloadedAdamW does not support fused=True")
        kwargs["foreach"] = False
        kwargs["fused"] = False
        super().__init__(cpu_groups, *args, **kwargs)

        self.original_param_groups = original_groups
        self._step_count = 0

    @torch.no_grad()
    def _sync_gradients_to_cpu(self):
        for original_group, cpu_group in zip(self.original_param_groups, self.param_groups):
            for original, cpu_parameter in zip(original_group, cpu_group["params"]):
                gradient = original.grad
                if gradient is None:
                    cpu_parameter.grad = None
                    continue
                cpu_gradient = gradient.detach().to(device="cpu", dtype=cpu_parameter.dtype)
                if cpu_parameter.grad is None or cpu_parameter.grad.is_sparse != cpu_gradient.is_sparse:
                    cpu_parameter.grad = cpu_gradient.clone()
                else:
                    cpu_parameter.grad.copy_(cpu_gradient)

    @torch.no_grad()
    def _sync_parameters_to_model(self):
        for original_group, cpu_group in zip(self.original_param_groups, self.param_groups):
            for original, cpu_parameter in zip(original_group, cpu_group["params"]):
                original.copy_(cpu_parameter.to(device=original.device, dtype=original.dtype))

    @torch.no_grad()
    def _sync_parameters_to_cpu(self):
        for original_group, cpu_group in zip(self.original_param_groups, self.param_groups):
            for original, cpu_parameter in zip(original_group, cpu_group["params"]):
                cpu_parameter.copy_(original.detach().to(device="cpu", dtype=cpu_parameter.dtype))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._sync_gradients_to_cpu()
        super().step()
        self._sync_parameters_to_model()
        self._step_count += 1
        return loss

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)
        for group in self.original_param_groups:
            for parameter in group:
                if set_to_none:
                    parameter.grad = None
                elif parameter.grad is not None:
                    parameter.grad.detach_()
                    parameter.grad.zero_()

    def state_dict(self):
        state = super().state_dict()
        state["cpu_offloaded_step_count"] = self._step_count
        return state

    def load_state_dict(self, state_dict):
        state = dict(state_dict)
        self._step_count = int(state.pop("cpu_offloaded_step_count", 0))
        # Model weights are stored by the model checkpoint, not the optimizer.
        # Re-seed the CPU clones from those restored weights before moments are
        # loaded, so no pre-resume optimizer copy can bias the next update.
        self._sync_parameters_to_cpu()
        result = super().load_state_dict(state)
        for parameter, values in self.state.items():
            self.state[parameter] = _cpu_state(values)
        return result

    @property
    def step_count(self):
        return self._step_count
