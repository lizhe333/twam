"""Two-rank CPU/Gloo check of production accumulation, EMA and checkpoint helpers.

Uses a tiny linear model, not a WAVE-0 quality or A800 memory benchmark.
Run: python -m torch.distributed.run --standalone --nproc_per_node=2 scripts/debug/univtac/check_distributed.py
"""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from safetensors.torch import save_model, load_model
from lerobot.utils.univtac_runtime import configure, atomic_json
ROOT=configure()
from lerobot.configs.univtac import load_config
from lerobot.scripts.lerobot_train import EMA, make_optimizer, set_lr, save_checkpoint, restore_rng, witnesses


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=torch.nn.Linear(3,2)

    def forward(self,x):
        return self.linear(x)

    def save_pretrained(self,path):
        path=Path(path)
        path.mkdir(parents=True)
        save_model(self,str(path/"model.safetensors"))


dist.init_process_group("gloo")
rank,world=dist.get_rank(),dist.get_world_size()
torch.set_num_threads(1)
torch.manual_seed(42)
model=Tiny()
wrapped=DistributedDataParallel(model,gradient_as_bucket_view=True)
run=load_config(ROOT/"configs/univtac/insert_hole.yaml")
folder=[tempfile.mkdtemp(prefix="wave0-ddp-",dir="/tmp") if rank==0 else None]
dist.broadcast_object_list(folder)
run["output"]=folder[0]
optimizer=make_optimizer(model,run,"stage2")
set_lr(optimizer,run["stage2"],0)
ema=EMA(model,.99) if rank==0 else None
for micro in range(2):
    with wrapped.no_sync() if micro==0 else nullcontext():
        loss=wrapped(torch.full((1,3),float(rank*2+micro+1))).square().mean()/2
        loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
if ema:
    ema.update(model)
values=torch.cat([p.detach().flatten() for p in model.parameters()])
copies=[torch.empty_like(values) for _ in range(world)]
dist.all_gather(copies,values)
assert all(torch.equal(values,x) for x in copies)
torch.manual_seed(100+rank)
path=save_checkpoint(model,optimizer,ema,run,"stage2",1,{"debug":True,"fingerprint":"cpu-gloo-helper-smoke"},
                     world,rank,{"model":"tiny linear"})
expected=torch.rand(8)
state=torch.load(path/"training.pt",weights_only=False)
restore_rng(state["rng"][rank])
assert torch.equal(expected,torch.rand(8))
restored=Tiny()
load_model(restored,str(path/"pretrained_model/model.safetensors"),strict=True)
assert all(torch.equal(a,b) for a,b in zip(model.parameters(),restored.parameters()))
opt=make_optimizer(restored,run,"stage2")
opt.load_state_dict(state["optimizer"])
if rank==0:
    from safetensors.torch import load_file
    recovered_ema=EMA(restored,.99)
    recovered_ema.values=load_file(str(path/"ema.safetensors"))
    assert witnesses(restored,opt,recovered_ema)==json.loads((path/"witnesses.json").read_text())
    result={"passed":True,"world_size":world,"backend":"gloo","model":"tiny linear",
            "gradient_accumulation":2,"parameters_synchronized":True,"rng_restored_each_rank":True,
            "optimizer_and_ema_restored":True,"A800_or_NCCL_test":False}
    atomic_json(ROOT/"reports/distributed-cpu.json",result)
    print(json.dumps(result))
dist.barrier()
dist.destroy_process_group()
