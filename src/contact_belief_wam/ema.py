"""FP32 EMA, optionally partitioned across the existing data-parallel GPUs."""
from pathlib import Path
import shutil

import torch
import torch.distributed as dist


class EMA:
    """FP32 EMA for every trainable Stage-2 parameter, stored on rank zero."""
    is_sharded = False

    def __init__(self, policy, decay=0.99, device="cuda", *, initialize=True, parameter_names=None):
        self.decay = float(decay)
        self.device = torch.device(device)
        self.parameter_names = set(parameter_names) if parameter_names is not None else None
        self.values = ({name: parameter.detach().to(device=self.device, dtype=torch.float32).clone()
                        for name, parameter in policy.named_parameters()
                        if parameter.requires_grad and (self.parameter_names is None or name in self.parameter_names)}
                       if initialize else {})

    @torch.no_grad()
    def update(self, policy):
        if self.device.type == "cpu":
            # Transfer one parameter at a time; a foreach list would retain
            # another full FP32 model in host RAM on every EMA update.
            for name, parameter in policy.named_parameters():
                if name in self.values:
                    value = parameter.detach().to(device="cpu", dtype=torch.float32)
                    self.values[name].mul_(self.decay).add_(value, alpha=1.0 - self.decay)
                    del value
            return
        names, parameters = [], []
        for name, parameter in policy.named_parameters():
            if name in self.values:
                names.append(name)
                parameters.append(parameter.detach().to(device=self.device, dtype=torch.float32))
        values = [self.values[name] for name in names]
        if values:
            torch._foreach_mul_(values, self.decay)
            torch._foreach_add_(values, parameters, alpha=1.0 - self.decay)

    def save(self, path):
        from safetensors.torch import save_file
        save_file({name: value.detach().cpu().contiguous() for name, value in self.values.items()}, str(path))

    def load(self, path, policy):
        from safetensors import safe_open
        expected = {name: parameter for name, parameter in policy.named_parameters() if parameter.requires_grad}
        values = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(expected):
                raise ValueError("EMA checkpoint keys differ from trainable model parameters")
            for name in expected:
                if self.parameter_names is not None and name not in self.parameter_names:
                    continue
                value = handle.get_tensor(name)
                if value.dtype != torch.float32 or value.shape != expected[name].shape:
                    raise ValueError(f"EMA checkpoint tensor differs for {name}")
                values[name] = value.to(self.device)
        self.values = values


class ShardedEMA(EMA):
    """Disjoint parameters on each GPU, identical EMA arithmetic, no step collective.

    Only checkpoint serialization stages tensors on CPU. Exports retain the
    standard single safetensors file used by inference and older dense EMA.
    """
    is_sharded = True

    def __init__(self, policy, decay=0.99, device="cuda", *, initialize=True):
        if not dist.is_initialized():
            raise ValueError("Sharded EMA requires an initialized process group")
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        loads = [0] * self.world_size
        names = []
        ordered = sorted(((name, p.numel()) for name, p in policy.named_parameters() if p.requires_grad),
                         key=lambda item: (-item[1], item[0]))
        for name, size in ordered:
            owner = min(range(self.world_size), key=lambda rank: (loads[rank], rank))
            loads[owner] += size
            if owner == self.rank:
                names.append(name)
        super().__init__(policy, decay, device, initialize=initialize, parameter_names=names)

    def save(self, path):
        raise RuntimeError("Sharded EMA must be exported collectively")

    def save_collective(self, path):
        from safetensors.torch import load_file, save_file
        path = Path(path)
        parts = path.with_name(path.name + ".parts")
        if self.rank == 0:
            parts.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        EMA.save(self, parts / f"rank{self.rank:02d}.safetensors")
        dist.barrier()
        if self.rank == 0:
            values = {}
            for rank in range(self.world_size):
                shard = load_file(str(parts / f"rank{rank:02d}.safetensors"), device="cpu")
                if values.keys() & shard.keys():
                    raise ValueError("Overlapping EMA ownership")
                values.update(shard)
            save_file(values, str(path))
            del values
            shutil.rmtree(parts)
        dist.barrier()

