"""Adam state sharding, following the successful local dual-view WAVE-0 trainer.

Checkpoint each rank separately: never consolidate full Adam state onto GPU 0.
"""
from pathlib import Path

import torch
import torch.distributed as dist


FORMAT = "zero_local_v1"
DENSE_FORMAT = "adamw_v1"


def is_sharded(optimizer):
    return getattr(optimizer, "_twam_sharded", False)


def local_optimizer(optimizer):
    return optimizer.optim if is_sharded(optimizer) else optimizer


def make_sharded_adamw(groups, **options):
    if not dist.is_initialized() or dist.get_world_size() < 2:
        raise ValueError("Sharded AdamW requires at least two distributed ranks")
    from torch.distributed.optim import ZeroRedundancyOptimizer

    optimizer = ZeroRedundancyOptimizer(
        groups, optimizer_class=torch.optim.AdamW,
        # Safetensors exports need independent parameter storage, not views
        # into optimizer buckets. Sharding the moments does not need views.
        parameters_as_bucket_view=False, overlap_with_ddp=False, **options,
    )
    optimizer._twam_sharded = True
    return optimizer


def parameter_names(optimizer, policy):
    names = {id(p): name for name, p in policy.named_parameters()}
    return [[names[id(p)] for p in group["params"]]
            for group in local_optimizer(optimizer).param_groups]


def shard_name(rank):
    return f"optimizer.rank{rank:02d}.pt"


def save_local_state(optimizer, policy, directory, rank, world_size):
    """Every rank writes its own shard after the pending directory exists."""
    if not is_sharded(optimizer):
        raise ValueError("Expected sharded optimizer")
    torch.save({"format": FORMAT, "rank": rank, "world_size": world_size,
                "parameter_names": parameter_names(optimizer, policy),
                "optimizer": optimizer.optim.state_dict()},
               Path(directory) / shard_name(rank))


def load_local_state(optimizer, policy, directory, rank, world_size):
    if not is_sharded(optimizer):
        raise ValueError("Expected sharded optimizer")
    value = torch.load(Path(directory) / shard_name(rank), map_location="cpu", weights_only=False)
    if (value.get("format") != FORMAT or value.get("rank") != rank
            or value.get("world_size") != world_size
            or value.get("parameter_names") != parameter_names(optimizer, policy)):
        raise ValueError("Optimizer shard rank, world size, or parameter mapping changed")
    optimizer.optim.load_state_dict(value["optimizer"])
    # The outer optimizer groups drive scheduling; restore their LR/metadata too.
    for local, outer in zip(optimizer.optim.param_groups, optimizer.param_groups):
        outer.update({key: val for key, val in local.items() if key != "params"})


def restore_optimizer(optimizer, policy, directory, state, rank, world_size):
    expected = FORMAT if is_sharded(optimizer) else DENSE_FORMAT
    if state.get("optimizer_format", DENSE_FORMAT) != expected:
        raise ValueError("Checkpoint optimizer format differs from the current optimizer")
    if is_sharded(optimizer):
        load_local_state(optimizer, policy, directory, rank, world_size)
    else:
        optimizer.load_state_dict(state["optimizer"])
