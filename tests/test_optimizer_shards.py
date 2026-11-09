"""Real Gloo ranks: sharded Adam equals dense Adam, including exact resume."""
from contextlib import nullcontext
import copy
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from safetensors.torch import load_file, save_file
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from contact_belief_wam import optimizer_shards, trainer


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 9)
        self.contact = torch.nn.Module()
        self.contact.updater = torch.nn.Linear(9, 9)
        self.contact.to_expert = torch.nn.Linear(9, 7)
        self.contact.unused = torch.nn.Linear(7, 3)
        self.head = torch.nn.Linear(7, 3)
        self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        self.config = SimpleNamespace(camera_mode="head_wrist")

    def forward(self, x):
        return self.head(self.contact.to_expert(self.contact.updater(self.encoder(x)).tanh()))

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), str(Path(path) / "model.safetensors"))


def recipe(root):
    return {"output": str(root / "run"), "task": "insert_hole", "variant": "full", "mode": "full",
            "task_stats": str(root / "stats.json"), "wm_stats": str(root / "stats.json"),
            "tokenizer_path": str(root / "tokenizer.model"), "fused_adamw": False,
            "stage2": {"lr": .02, "lr_new": .01, "lr_belief": .005, "warmup": 1,
                       "final_lr": .002, "steps": 10, "weight_decay": .01,
                       "shard_optimizer_state": True, "ema_device": "cpu"}}


def batch(step, rank, micro):
    generator = torch.Generator().manual_seed(1000 + step * 100 + rank * 10 + micro)
    return torch.randn(2, 4, generator=generator), torch.randn(2, 3, generator=generator)


def update(policy, wrapped, optimizer, reference, dense, ema, reference_ema, run, step, rank, world):
    accumulation = 8 // world
    for opt in (optimizer, dense):
        opt.zero_grad(set_to_none=True)
        trainer.set_lr(opt, run["stage2"], "stage2", step)
    for micro in range(accumulation):
        with wrapped.no_sync() if micro < accumulation - 1 else nullcontext():
            x, y = batch(step, rank, micro)
            ((wrapped(x) - y).square().mean() / accumulation).backward()
        for other_rank in range(world):
            x, y = batch(step, other_rank, micro)
            ((reference(x) - y).square().mean() / (accumulation * world)).backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
    optimizer.step()
    dense.step()
    for actual, expected in zip(policy.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    if ema:
        ema.update(policy)
        reference_ema.update(reference)
        for name, value in ema.values.items():
            torch.testing.assert_close(value, reference_ema.values[name], rtol=2e-6, atol=2e-7)


def rank_test(rank, root, world):
    root = Path(root)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(),
                            rank=rank, world_size=world, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(34)
        policy = Policy()
        reference = copy.deepcopy(policy)
        run = recipe(root)
        optimizer = trainer.make_optimizer(policy, run, "stage2", device="cpu")
        dense_run = copy.deepcopy(run)
        dense_run["stage2"]["shard_optimizer_state"] = False
        dense = trainer.make_optimizer(reference, dense_run, "stage2", device="cpu")
        wrapped = torch.nn.parallel.DistributedDataParallel(
            policy, find_unused_parameters=True, gradient_as_bucket_view=True)
        ema = trainer.EMA(policy, device="cpu") if rank == 0 else None
        reference_ema = trainer.EMA(reference, device="cpu") if rank == 0 else None
        for step in range(3):
            update(policy, wrapped, optimizer, reference, dense, ema, reference_ema, run, step, rank, world)
        local_names = optimizer_shards.parameter_names(optimizer, policy)
        local_count = sum(p.numel() for g in optimizer.optim.param_groups for p in g["params"])
        full_count = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        assert 0 < local_count < full_count
        assert len(optimizer.optim.state) > 0
        rng_states = [None] * world
        dist.all_gather_object(rng_states, trainer.rng_state())
        contract = {"shard_optimizer_state": True, "ema_device": "cpu"}
        identity = {"fingerprint": "fixture"}
        path = trainer._atomic_checkpoint(policy, optimizer, ema, run, "stage2", 3,
                                          contract, identity, world, rank, rng_states, 24, {})
        meta = trainer._validate_resume(path, "stage2", contract, identity, world, rank=rank)
        assert len(meta["optimizer_shards_sha256"]) == world
        state = torch.load(path / "training.pt", weights_only=False)
        assert state["optimizer"] is None and state["optimizer_format"] == optimizer_shards.FORMAT
        assert state["draws_per_rank"] == 24 and len(state["rng_by_rank"]) == world
        restored = Policy()
        restored.load_state_dict(load_file(str(path / "pretrained_model/model.safetensors")))
        resumed = trainer.make_optimizer(restored, run, "stage2", device="cpu")
        optimizer_shards.restore_optimizer(resumed, restored, path, state, rank, world)
        assert optimizer_shards.parameter_names(resumed, restored) == local_names
        for old_group, new_group in zip(optimizer.optim.param_groups, resumed.optim.param_groups):
            assert new_group["lr"] == old_group["lr"]
            for old, new in zip(old_group["params"], new_group["params"]):
                for key, value in optimizer.optim.state.get(old, {}).items():
                    torch.testing.assert_close(resumed.optim.state[new][key], value, rtol=0, atol=0)
        restored_ema = trainer.EMA(restored, device="cpu", initialize=False) if rank == 0 else None
        if restored_ema:
            restored_ema.load(path / "pretrained_model/ema.safetensors", restored)
            for name in ema.values:
                torch.testing.assert_close(restored_ema.values[name], ema.values[name], rtol=0, atol=0)
        resumed_ddp = torch.nn.parallel.DistributedDataParallel(
            restored, find_unused_parameters=True, gradient_as_bucket_view=True)
        update(restored, resumed_ddp, resumed, reference, dense, restored_ema, reference_ema,
               run, 3, rank, world)
        with pytest.raises(ValueError, match="optimizer format"):
            optimizer_shards.restore_optimizer(dense, reference, path, state, rank, world)
        # A mislabeled shard must not silently attach moments to other weights.
        shard = path / optimizer_shards.shard_name(rank)
        wrong = torch.load(shard, weights_only=False)
        wrong["parameter_names"][0][0] = "wrong_parameter"
        bad = root / f"bad-{rank}"
        bad.mkdir()
        torch.save(wrong, bad / shard.name)
        with pytest.raises(ValueError, match="mapping changed"):
            optimizer_shards.load_local_state(resumed, restored, bad, rank, world)
        shard.write_bytes(b"corrupted optimizer")
        with pytest.raises(ValueError, match="shard integrity"):
            trainer._validate_resume(path, "stage2", contract, identity, world, rank=rank)
        (root / f"passed-{rank}.json").write_text(json.dumps({
            "local": local_count, "full": full_count,
            "names": [name for group in local_names for name in group]}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_updates_checkpoint_resume_and_shard_validation(tmp_path, world):
    (tmp_path / "stats.json").write_text("{}")
    (tmp_path / "tokenizer.model").write_bytes(b"fixture")
    mp.spawn(rank_test, args=(str(tmp_path), world), nprocs=world, join=True)
    rows = [json.loads((tmp_path / f"passed-{i}.json").read_text()) for i in range(world)]
    assert sum(row["local"] for row in rows) == rows[0]["full"]
    names = [name for row in rows for name in row["names"]]
    assert len(names) == len(set(names))
    assert set(names) == {name for name, p in Policy().named_parameters() if p.requires_grad}


def test_sharded_optimizer_requires_distributed_group(tmp_path):
    with pytest.raises(ValueError, match="distributed ranks"):
        trainer.make_optimizer(Policy(), recipe(tmp_path), "stage2", device="cpu")


def test_dense_checkpoint_still_roundtrips(tmp_path):
    (tmp_path / "stats.json").write_text("{}")
    (tmp_path / "tokenizer.model").write_bytes(b"fixture")
    run = recipe(tmp_path)
    run["stage2"]["shard_optimizer_state"] = False
    policy = Policy()
    optimizer = trainer.make_optimizer(policy, run, "stage2", device="cpu")
    policy(torch.ones(2, 4)).sum().backward()
    optimizer.step()
    ema = trainer.EMA(policy, device="cpu")
    contract = {"shard_optimizer_state": False, "ema_device": "cpu"}
    identity = {"fingerprint": "dense-fixture"}
    path = trainer._atomic_checkpoint(policy, optimizer, ema, run, "stage2", 1,
                                      contract, identity, 1, 0, [trainer.rng_state()], 2, {})
    trainer._validate_resume(path, "stage2", contract, identity, 1)
    state = torch.load(path / "training.pt", weights_only=False)
    assert state["optimizer_format"] == optimizer_shards.DENSE_FORMAT
    resumed = trainer.make_optimizer(policy, run, "stage2", device="cpu")
    optimizer_shards.restore_optimizer(resumed, policy, path, state, 0, 1)
    for parameter, old in optimizer.state.items():
        for key, value in old.items():
            torch.testing.assert_close(resumed.state[parameter][key], value, rtol=0, atol=0)
