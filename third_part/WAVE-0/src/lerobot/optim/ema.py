"""Established FP32 EMA; checkpoint keys and update arithmetic are unchanged."""
import torch
from safetensors.torch import load_file, save_file

class EMA:
    """Float32 EMA stored on the selected device, normally CPU."""
    def __init__(self, policy, decay, device="cpu", *, initialize=True):
        self.decay = decay
        self.device = torch.device(device)
        self.values = ({n: p.detach().to(device=self.device, dtype=torch.float32).clone()
                        for n, p in policy.named_parameters() if p.requires_grad}
                       if initialize else {})

    @torch.no_grad()
    def update(self, policy):
        if self.device.type == "cuda":
            names, params = [], []
            for name, parameter in policy.named_parameters():
                if name in self.values:
                    names.append(name)
                    params.append(parameter.detach().to(device=self.device, dtype=torch.float32))
            values = [self.values[name] for name in names]
            if values:
                torch._foreach_mul_(values, self.decay)
                torch._foreach_add_(values, params, alpha=1-self.decay)
            return
        for n,p in policy.named_parameters():
            if n in self.values:
                value = p.detach().to(device=self.device, dtype=torch.float32)
                self.values[n].mul_(self.decay).add_(value, alpha=1-self.decay)

    def load(self, path, policy):
        # Load directly onto the selected device so resume does not allocate an
        # initialization EMA and then replace it, or stage a second full copy.
        values = load_file(str(path), device=str(self.device))
        expected = {n: p for n, p in policy.named_parameters() if p.requires_grad}
        if set(values) != set(expected):
            raise ValueError("EMA checkpoint parameter keys differ from the policy")
        for name, value in values.items():
            if value.dtype != torch.float32 or value.shape != expected[name].shape:
                raise ValueError(f"EMA checkpoint tensor differs for {name}")
        self.values = values

    def save(self, path):
        # safetensors writes CPU tensors. On CPU, .cpu() is a view and avoids a
        # duplicate model allocation; on CUDA, the copy happens only at save time.
        save_file({n: v.detach().cpu().contiguous() for n, v in self.values.items()}, str(path))

def make_ema(policy, run, stage, rank, device, *, initialize=True):
    if stage != "stage2" or rank != 0 or not run.get("ema_enabled", True):
        return None
    return EMA(policy, run["ema_decay"], device, initialize=initialize)
