"""Two local CPU ranks exercise recurrent DDP/checkpointing without training a full policy."""
import json
import sys
from contextlib import nullcontext
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
            policy = ContactWAM(replace(cfg('sdpa', True), aligned=True, stage=stage,
                camera_mode='head_wrist', independent_tactile=True,
                stage1_freeze_world=True, stage2_freeze_belief=False))
            wrapped = torch.nn.parallel.DistributedDataParallel(
                SequenceTrainingModel(policy), find_unused_parameters=True,
                gradient_as_bucket_view=True, broadcast_buffers=False)
            obs, targets, _, _ = inputs()
            repeat = lambda x: x[:,None].repeat(1,3,*([1]*(x.ndim-1)))
            # Matching current/future frames exercise the actual frozen-vision
            # cache in a recurrent DDP forward, including across no_grad warmup.
            images = torch.stack([obs.image, targets.future_image,
                                  torch.rand_like(obs.image), torch.rand_like(obs.image)], dim=1)
            wrists = torch.rand_like(images)
            batch = dict(image=images[:, :3], future_image=images[:, 1:],
                wrist_image=wrists[:, :3], future_wrist_image=wrists[:, 1:],
                visual_frame_index=torch.tensor([[0,2,4],[0,2,4]]),
                future_visual_frame_index=torch.tensor([[2,4,6],[2,4,6]]),
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
            accumulation = 2
            for step in range(2):
                policy.zero_grad(set_to_none=True)
                step_loss = 0.
                for micro in range(accumulation):
                    torch.manual_seed(rank + step * accumulation + micro + 8)
                    sync = wrapped.no_sync() if micro < accumulation - 1 else nullcontext()
                    with sync:
                        with torch.autocast('cpu', dtype=torch.bfloat16):
                            result=wrapped(batch)
                        assert torch.isfinite(result['loss'])
                        (result['loss'] / accumulation).backward()
                    step_loss += float(result['loss']) / accumulation
                assert all(p.grad is None for p in policy.core.world_model.parameters())
                if stage == 'stage2':
                    assert all(p.grad is None for p in policy.contact.predictor.parameters())
                gradient=next(p.grad for p in policy.contact.updater.parameters() if p.grad is not None)
                assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
                copies=[torch.zeros_like(gradient) for _ in range(2)]
                dist.all_gather(copies,gradient)
                torch.testing.assert_close(copies[0],copies[1],rtol=0,atol=0)
                losses.append(step_loss)
            if rank==0:
                with open(output,'a') as f:
                    f.write(json.dumps({'stage':stage,'ranks':2,'device':'cpu','gradient_checkpointing':True,
                                        'camera_mode':'head_wrist','autocast':'bfloat16',
                                        'frozen_visual_reuse':True,
                                        'two_forward_backward_iterations':True,
                                        'accumulation_steps':accumulation,'losses':losses})+'\n')
            del wrapped,policy
    finally:
        dist.destroy_process_group()
        monkeypatch.undo()


if __name__=='__main__':
    output=str(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory(prefix='aligned-ddp-') as directory:
        mp.spawn(worker,args=(str(Path(directory)/'rendezvous'),output),nprocs=2,join=True)
