"""Preserve task-policy and shared-world normalization at the world input boundary."""
import json
from pathlib import Path
import torch

def bridge_tensor(value, source, target):
    """Invert task quantiles and apply WM quantiles; preserve padded dimensions."""
    lo = value.new_tensor(source["q01"])
    span = value.new_tensor(source["q99"])-lo+1e-6
    dst_lo = value.new_tensor(target["q01"])
    dst_span = value.new_tensor(target["q99"])-dst_lo+1e-6
    first = ((value[..., :8]+1)*.5*span+lo-dst_lo)/dst_span*2-1
    return torch.cat([first, value[..., 8:]], dim=-1)

def install_world_normalization(policy, task_stats, wm_stats):
    """Hooks preserve official parameter names and affect ONLY WM inputs.

    The policy prompt/flow target keeps the original task statistics. The frozen
    teacher and inference state encoder see the shared Stage-1 coordinate system.
    No learnable parameters or mutable batch context are introduced.
    """
    if getattr(policy, "_world_norm_handles", None):
        raise ValueError("World normalization already installed")
    source = json.loads(Path(task_stats).read_text())["norm_stats"]
    target = json.loads(Path(wm_stats).read_text())["norm_stats"]

    def state_hook(module, args):
        return (bridge_tensor(args[0], source["state"], target["state"]), *args[1:])

    def action_hook(module, args):
        actions, state, *rest = args
        return (bridge_tensor(actions, source["actions"], target["actions"]),
                bridge_tensor(state, source["state"], target["state"]) if state is not None else None, *rest)

    handles = [policy.model.world_state_encoder.register_forward_pre_hook(state_hook)]
    if policy.model.world_action_encoder is not None:
        handles.append(policy.model.world_action_encoder.register_forward_pre_hook(action_hook))
    policy._world_norm_handles = handles
