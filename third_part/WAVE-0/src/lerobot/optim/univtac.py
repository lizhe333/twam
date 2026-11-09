import math
import torch
from lerobot.policies.wave0.loading_wave0 import group_name
from lerobot.optim.optimizer_shards import make_sharded_adamw

def lr_at(step, peak, final, warmup, steps):
    if step < warmup:
        start = peak/(warmup+1)
        return start+(peak-start)*step/max(1, warmup)
    fraction = min(1., (step-warmup)/max(1, steps-warmup))
    return final+(peak-final)*.5*(1+math.cos(math.pi*fraction))

def make_optimizer(policy, run, stage):
    cfg = run[stage]
    performance = performance_config(run)
    groups = {"policy": [], "new": []}
    for name, p in policy.named_parameters():
        if p.requires_grad:
            groups["new" if group_name(name) in ("world", "future_condition") else "policy"].append(p)
    params = []
    for name, values in groups.items():
        if values:
            peak = cfg.get("lr_new", cfg["lr"]) if name=="new" else cfg["lr"]
            params.append({"params": values, "peak": peak, "name": name, "lr": peak})
    device = next((p.device for p in policy.parameters()), torch.device("cpu"))
    fused = bool(performance.get("fused_adamw", False)) and device.type == "cuda"
    optimizer_options = {"betas": (.9, .95), "eps": 1e-8,
                         "weight_decay": cfg["weight_decay"]}
    if fused:
        optimizer_options.update(fused=True, foreach=None)
    else:
        optimizer_options["foreach"] = False
    if stage == "stage2" and performance.get("shard_optimizer_state", False):
        return make_sharded_adamw(params, **optimizer_options)
    return torch.optim.AdamW(params, **optimizer_options)

def set_lr(optimizer, cfg, step):
    for g in optimizer.param_groups:
        final = cfg["final_lr"]*g["peak"]/cfg["lr"]
        g["lr"] = lr_at(step, g["peak"], final, cfg["warmup"], cfg["steps"])

def performance_config(run):
    performance = run.get("performance", {})
    if performance is None:
        return {}
    if not isinstance(performance, dict):
        raise ValueError("performance must be a mapping")
    return performance

def select_ema_device(run, device, *, cpu_offload=False):
    requested = performance_config(run).get("ema_device", "cpu")
    if requested not in ("cpu", "cuda"):
        raise ValueError("performance.ema_device must be 'cpu' or 'cuda'")
    device = torch.device(device)
    if requested == "cuda" and device.type == "cuda" and not cpu_offload:
        return device
    return torch.device("cpu")

def ddp_find_unused_parameters(run):
    # Retain legacy behavior unless the opt-in performance section says otherwise.
    return performance_config(run).get("find_unused_parameters", True)
