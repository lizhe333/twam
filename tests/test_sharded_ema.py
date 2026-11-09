"""Real distributed checkpoint/restore and dense-to-sharded EMA equivalence."""
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file

from contact_belief_wam import trainer
from contact_belief_wam.ema import EMA, ShardedEMA
from test_optimizer_shards import Policy, recipe


def worker(rank, root, world):
    root = Path(root)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(),
                            rank=rank, world_size=world, timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(40)
        policy = Policy()
        dense = EMA(policy, decay=.97, device="cpu")
        sharded = ShardedEMA(policy, decay=.97, device="cpu")
        names = [None] * world
        dist.all_gather_object(names, set(sharded.values))
        assert set.union(*names) == set(dense.values)
        assert sum(map(len, names)) == len(dense.values)
        assert all(names)
        run = recipe(root)
        optimizer = trainer.make_optimizer(policy, run, "stage2", device="cpu")
        for step in range(2):
            torch.manual_seed(50 + step)
            optimizer.zero_grad(set_to_none=True)
            policy(torch.randn(4, 4)).square().mean().backward()
            optimizer.step()
            dense.update(policy)
            sharded.update(policy)
            for name, value in sharded.values.items():
                torch.testing.assert_close(value, dense.values[name], rtol=0, atol=0)
        contract = {"shard_optimizer_state": True, "shard_ema": True}
        identity = {"fingerprint": "ema-test"}
        checkpoint = trainer._atomic_checkpoint(
            policy, optimizer, sharded, run, "stage2", 2, contract, identity,
            world, rank, [None] * world, 8, {})
        trainer._validate_resume(checkpoint, "stage2", contract, identity, world, rank=rank)
        restored = ShardedEMA(policy, decay=.97, device="cpu", initialize=False)
        restored.load(checkpoint / "pretrained_model/ema.safetensors", policy)
        for name, value in restored.values.items():
            torch.testing.assert_close(value, dense.values[name], rtol=0, atol=0)
        if rank == 0:
            full = load_file(str(checkpoint / "pretrained_model/ema.safetensors"))
            assert full.keys() == dense.values.keys()
            for name, value in full.items():
                torch.testing.assert_close(value, dense.values[name], rtol=0, atol=0)
            dense.save(root / "legacy-ema.safetensors")
        dist.barrier()
        restored.load(root / "legacy-ema.safetensors", policy)
        with torch.no_grad():
            for p in policy.parameters():
                p.add_(.03)
        dense.update(policy)
        restored.update(policy)
        for name, value in restored.values.items():
            torch.testing.assert_close(value, dense.values[name], rtol=0, atol=0)
        restored.save_collective(root / "final/ema.safetensors")
        if rank == 0:
            full = load_file(str(root / "final/ema.safetensors"))
            for name, value in full.items():
                torch.testing.assert_close(value, dense.values[name], rtol=0, atol=0)
            assert not list(root.rglob("*.parts"))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_distributed_ema_matches_dense_and_exports_complete_weights(tmp_path, world):
    (tmp_path / "stats.json").write_text("{}")
    (tmp_path / "tokenizer.model").write_bytes(b"test")
    mp.spawn(worker, args=(str(tmp_path), world), nprocs=world, join=True)
