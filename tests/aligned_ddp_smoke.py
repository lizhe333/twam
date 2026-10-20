"""Two local CPU ranks exercise recurrent DDP/checkpointing without training a full policy."""
import json
import sys
from dataclasses import replace
from pathlib import Path
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_fast_attention import tiny_architecture, cfg, inputs
from contact_belief_wam.model import ContactWAM, pad8
from contact_belief_wam.sequence import SequenceTrainingModel


def worker(rank, rendezvous, output):
    monkeypatch = pytest.MonkeyPatch()
    tiny_architecture.__wrapped__(monkeypatch)
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank, world_size=2)
    try:
        for stage in ('stage1','stage2'):
            torch.manual_seed(5)
            policy = ContactWAM(replace(cfg('sdpa', True), aligned=True, stage=stage))
            wrapped = torch.nn.parallel.DistributedDataParallel(
                SequenceTrainingModel(policy), find_unused_parameters=True,
                gradient_as_bucket_view=True, broadcast_buffers=False)
            obs, targets, _, _ = inputs()
            repeat = lambda x: x[:,None].repeat(1,3,*([1]*(x.ndim-1)))
            batch = dict(image=repeat(obs.image), future_image=repeat(targets.future_image),
                state=repeat(obs.state), world_state=repeat(obs.state),
                tactile=repeat(obs.tactile), future_tactile=repeat(targets.future_tactile),
                actions=repeat(pad8(targets.actions)),world_actions=repeat(pad8(targets.actions)),
                tokens=repeat(obs.tokens),token_mask=repeat(obs.token_mask),
                action_mask=torch.ones(2,3,4,dtype=torch.bool),
                policy_action_mask=torch.ones(2,3,4,dtype=torch.bool),
                future_mask=torch.ones(2,3,dtype=torch.bool),
                executed_actions=torch.randn(2,3,2,8),executed_valid=torch.ones(2,3,2,dtype=torch.bool),
                sequence_mask=torch.tensor([[True,True,True],[rank==0,True,True]]),
                step=torch.tensor([[0,2,4],[0,2,4]]), task=['insert_hole']*2)
            losses=[]
            for step in range(2):
                policy.zero_grad(set_to_none=True)
                torch.manual_seed(rank+step+8)
                result=wrapped(batch)
                result['loss'].backward()
                assert torch.isfinite(result['loss'])
                gradient=next(p.grad for p in policy.parameters() if p.grad is not None)
                copies=[torch.zeros_like(gradient) for _ in range(2)]
                dist.all_gather(copies,gradient)
                torch.testing.assert_close(copies[0],copies[1],rtol=0,atol=0)
                losses.append(float(result['loss']))
            if rank==0:
                with open(output,'a') as f:
                    f.write(json.dumps({'stage':stage,'ranks':2,'device':'cpu','gradient_checkpointing':True,
                                        'two_forward_backward_iterations':True,'losses':losses})+'\n')
            del wrapped,policy
    finally:
        dist.destroy_process_group()
        monkeypatch.undo()


if __name__=='__main__':
    output=str(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory(prefix='aligned-ddp-') as directory:
        mp.spawn(worker,args=(str(Path(directory)/'rendezvous'),output),nprocs=2,join=True)
