"""Single-GPU debug: CPU master weights/gradients/Adam, CUDA model execution.

Keep functional_call active THROUGH backward so activation-checkpoint recomputes
see the CUDA copies too. Differentiable copies send gradients to CPU masters.
This is deliberately separate from the formal DDP training path.
"""
from contextlib import nullcontext

import torch


class _ForwardBackward(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch, backward_scale):
        loss, metrics = self.policy(batch)
        if backward_scale is not None:
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            (loss / backward_scale).backward()
            loss = loss.detach()
        return loss, metrics


def offloaded_forward(policy, batch, device, *, backward_scale=None, execution_dtype=None):
    if any(p.device.type != "cpu" for p in policy.parameters()):
        raise ValueError("Debug offload requires CPU master parameters")
    wrapper = _ForwardBackward(policy)
    original_parameters = list(wrapper.named_parameters(remove_duplicate=False))
    original_buffers = list(wrapper.named_buffers(remove_duplicate=False))
    parameters = {name: p.to(device=device, dtype=execution_dtype if p.is_floating_point() else p.dtype,
                              copy=True) for name, p in wrapper.named_parameters()}
    originals = dict(wrapper.named_buffers())
    buffers = {name: b.to(device=device, copy=True) for name, b in originals.items()}
    try:
        saved = torch.autograd.graph.save_on_cpu(pin_memory=True) if torch.device(device).type == "cuda" else nullcontext()
        with saved:
            return torch.func.functional_call(wrapper, (parameters, buffers),
                                              (batch, backward_scale), tie_weights=True, strict=True)
    finally:
        # Restore every alias explicitly: the same child module may be registered
        # under multiple paths, which otherwise can leave a temporary copy bound.
        for name, parameter in original_parameters:
            parent, _, key = name.rpartition(".")
            setattr(wrapper.get_submodule(parent), key, parameter)
        # Preserve forward-updated counters, just as ordinary module execution does.
        with torch.no_grad():
            for name, original in originals.items():
                original.copy_(buffers[name].to(original.device))
        for name, buffer in original_buffers:
            parent, _, key = name.rpartition(".")
            setattr(wrapper.get_submodule(parent), key, buffer)
