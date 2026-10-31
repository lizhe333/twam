"""GPU Adam state sharding; checkpoint each rank without GPU consolidation."""
from pathlib import Path

import torch
import torch.distributed as dist


FORMAT = "zero_local_v1"


def is_sharded(optimizer):
    return getattr(optimizer, "_univtac_sharded", False)


def local_optimizer(optimizer):
    return optimizer.optim if is_sharded(optimizer) else optimizer


def make_sharded_adamw(groups, **options):
    if not dist.is_initialized() or dist.get_world_size() < 2:
        raise ValueError("Sharded AdamW requires at least two distributed ranks")
    from torch.distributed.optim import ZeroRedundancyOptimizer

    optimizer = ZeroRedundancyOptimizer(
        groups, optimizer_class=torch.optim.AdamW,
        # Preserve independent parameter storage for upstream safetensors
        # save_model/load_model. Bucket views otherwise fail its full-storage
        # validation. State sharding itself does not require parameter views.
        parameters_as_bucket_view=False, overlap_with_ddp=False, **options,
    )
    optimizer._univtac_sharded = True
    return optimizer


def parameter_names(optimizer, policy):
    names = {id(p): n for n, p in policy.named_parameters()}
    return [[names[id(p)] for p in group["params"]]
            for group in local_optimizer(optimizer).param_groups]


def save_local_state(optimizer, policy, directory, rank, world):
    """Called by EVERY rank, after rank zero creates the pending directory.

    Only checkpoint I/O touches the CPU. No consolidate_state_dict() call or
    full optimizer allocation on rank zero, including during resume.
    """
    if not is_sharded(optimizer):
        raise ValueError("Expected sharded optimizer")
    torch.save({"format": FORMAT, "rank": rank, "world_size": world,
                "parameter_names": parameter_names(optimizer, policy),
                "optimizer": optimizer.optim.state_dict()},
               Path(directory) / f"optimizer.rank{rank:02d}.pt")


def load_local_state(optimizer, policy, directory, rank, world):
    if not is_sharded(optimizer):
        raise ValueError("Expected sharded optimizer")
    value = torch.load(Path(directory) / f"optimizer.rank{rank:02d}.pt",
                       map_location="cpu", weights_only=False)
    if (value.get("format") != FORMAT or value.get("rank") != rank
            or value.get("world_size") != world
            or value.get("parameter_names") != parameter_names(optimizer, policy)):
        raise ValueError("Optimizer shard rank, world size, or parameter mapping changed")
    optimizer.optim.load_state_dict(value["optimizer"])
    # The wrapper's groups drive LR schedules; match the restored local values.
    for local, outer in zip(optimizer.optim.param_groups, optimizer.param_groups):
        outer.update({k: v for k, v in local.items() if k != "params"})
