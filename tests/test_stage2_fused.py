"""The real small WAVE/Contact model: batching must preserve two-step BPTT."""
from dataclasses import replace
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import copy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from contact_belief_wam.model import ContactWAM
from contact_belief_wam.sequence import SequenceTrainingModel
from test_fast_attention import tiny_architecture
from test_stage2_compute_reuse import config, batch, instrument


@pytest.mark.parametrize("camera,mixed,bf16,augmentation,length", [
    ("head_only", False, False, False, 5),
    ("head_wrist", True, False, True, 5),
    ("head_only", True, True, True, 5),
    ("head_only", False, False, True, 2),
])
def test_fused_policy_preserves_losses_gradients_randomness_and_bptt(
        tiny_architecture, monkeypatch, camera, mixed, bf16, augmentation, length):
    if augmentation:
        # Exercise the actual 224px random augmentation with the reduced vision tower.
        from contact_belief_wam import aligned_augmentation
        original_augment = aligned_augmentation.augment
        def resized_augment(images):
            large = torch.nn.functional.interpolate(images, (224, 224), mode="bilinear", align_corners=False)
            return torch.nn.functional.interpolate(original_augment(large), images.shape[-2:],
                                                    mode="bilinear", align_corners=False)
        monkeypatch.setattr(aligned_augmentation, "augment", resized_augment)
    torch.manual_seed(81)
    cfg = replace(config(checkpoint=True), camera_mode=camera, policy_augmentation=augmentation)
    serial = ContactWAM(cfg).train()
    fused = ContactWAM(replace(cfg, batch_stage2_policy=True)).train()
    fused.load_state_dict(serial.state_dict())
    data = batch(mixed)
    if length == 2:
        data = {k: v[:, -2:] if torch.is_tensor(v) and v.ndim >= 2 and v.shape[1] == 5 else v
                for k, v in data.items()}
    records = []
    for model in (serial, fused):
        counts, handles = instrument(model)
        beliefs = []
        def remember(module, args, output):
            if output.requires_grad:
                output.retain_grad()
            beliefs.append(output)
        handles.append(model.contact.updater.register_forward_hook(remember))
        torch.manual_seed(802)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            result = SequenceTrainingModel(model)(data)
        calls = counts["vlm_prefill_calls"]
        result["loss"].backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        assert all(not belief.requires_grad for belief in beliefs[:-2])
        assert all(belief.grad is not None and belief.grad.abs().sum() > 0 for belief in beliefs[-2:])
        for handle in handles:
            handle.remove()
        records.append((result, gradients, torch.get_rng_state(), calls))
    (a, ga, ra, ca), (b, gb, rb, cb) = records
    assert ca == 2 and cb == 1
    assert ga.keys() == gb.keys()
    rtol, atol = (0.025, 0.006) if bf16 else (3e-4, 3e-5)
    for key in ("loss", "belief"):
        torch.testing.assert_close(a[key], b[key], rtol=rtol, atol=atol)
    for name in ga:
        torch.testing.assert_close(ga[name], gb[name], rtol=rtol, atol=atol,
                                   msg=lambda message: name + ": " + message)
    assert a["metrics"].keys() == b["metrics"].keys()
    for key in a["metrics"]:
        for field in ("sum", "count"):
            torch.testing.assert_close(a["metrics"][key][field], b["metrics"][key][field], rtol=rtol, atol=atol)
    assert torch.equal(ra, rb)


def test_last_action_loss_backpropagates_into_previous_belief(tiny_architecture):
    torch.manual_seed(93)
    policy = ContactWAM(replace(config(), camera_mode="head_only", batch_stage2_policy=True)).train()
    data = batch()
    beliefs = []
    def remember(module, args, value):
        if value.requires_grad:
            value.retain_grad()
        beliefs.append(value)
    def last_action_only(module, args, output):
        mask = output["action_loss_per_sample"].new_ones(len(output["action_loss_per_sample"]))
        mask[:data["state"].shape[0]] = 0
        output["action_loss_per_sample"] = output["action_loss_per_sample"] * mask
        output["future_loss_per_sample"] = output["future_loss_per_sample"] * 0
    hook = policy.contact.updater.register_forward_hook(remember)
    final_loss = policy.register_forward_hook(last_action_only)
    result = SequenceTrainingModel(policy)(data)
    result["loss"].backward()
    assert all(not value.requires_grad for value in beliefs[:-2])
    assert beliefs[-2].grad is not None and beliefs[-2].grad.abs().sum() > 0
    assert beliefs[-1].grad is not None and beliefs[-1].grad.abs().sum() > 0
    hook.remove()
    final_loss.remove()


def _distributed_fused_worker(rank, root):
    from contact_belief_wam import trainer
    from contact_belief_wam.ema import ShardedEMA, EMA
    from test_optimizer_shards import recipe
    root = Path(root)
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        with pytest.MonkeyPatch.context() as patch:
            tiny_architecture.__wrapped__(patch)
            torch.manual_seed(98)
            policy = ContactWAM(replace(config(), camera_mode="head_only", batch_stage2_policy=True)).train()
            reference = copy.deepcopy(policy)
            reference.config = replace(reference.config, batch_stage2_policy=False)
            wrapped = torch.nn.parallel.DistributedDataParallel(
                SequenceTrainingModel(policy), find_unused_parameters=True, gradient_as_bucket_view=True)
            run = recipe(root)
            optimizer = trainer.make_optimizer(policy, run, "stage2", device="cpu")
            dense_run = copy.deepcopy(run)
            dense_run["stage2"]["shard_optimizer_state"] = False
            dense = trainer.make_optimizer(reference, dense_run, "stage2", device="cpu")
            ema, ema_reference = ShardedEMA(policy, device="cpu"), EMA(reference, device="cpu")
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                dense.zero_grad(set_to_none=True)
                for micro in range(2):
                    torch.manual_seed(1000 + step * 100 + micro * 10 + rank)
                    data = batch(mixed=True)
                    with wrapped.no_sync() if micro == 0 else nullcontext():
                        (wrapped(data)["loss"] / 2).backward()
                    for other in range(2):
                        torch.manual_seed(1000 + step * 100 + micro * 10 + other)
                        (SequenceTrainingModel(reference)(batch(mixed=True))["loss"] / 4).backward()
                assert policy.contact.updater.initial.grad.abs().sum() > 0
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
                torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
                optimizer.step()
                dense.step()
                ema.update(policy)
                ema_reference.update(reference)
                for name, value in policy.named_parameters():
                    torch.testing.assert_close(value, dict(reference.named_parameters())[name], atol=2e-4, rtol=1e-3)
                for name, value in ema.values.items():
                    torch.testing.assert_close(value, ema_reference.values[name], atol=2e-5, rtol=1e-3)
    finally:
        dist.destroy_process_group()


def test_fused_ddp_accumulation_and_gpu_sharding_math(tmp_path):
    mp.spawn(_distributed_fused_worker, args=(str(tmp_path),), nprocs=2, join=True)


def test_old_model_checkpoint_loads_with_only_batching_flag_changed(tiny_architecture, tmp_path):
    from contact_belief_wam import trainer
    torch.manual_seed(97)
    cfg = replace(config(), camera_mode="head_only")
    old = ContactWAM(cfg)
    old.optimizer_step.fill_(4000)
    source = tmp_path / "old/full"
    checkpoint = source / "tasks/insert_hole/stage2/checkpoints/step_00004000"
    old.save_pretrained(checkpoint / "pretrained_model")
    values = replace(cfg, batch_stage2_policy=True).to_dict()
    values.pop("type")
    run = {"task": "insert_hole", "mode": "full", "stage2_restart_from": str(source),
           "model_config": values, "policy_norm_stats": cfg.policy_norm_stats,
           "world_norm_stats": cfg.world_norm_stats}
    loaded, _ = trainer._load_model(run, "stage2", checkpoint)
    assert loaded.config.batch_stage2_policy and loaded.config.belief_bptt_steps == 2
    assert int(loaded.optimizer_step) == 4000
    assert old.state_dict().keys() == loaded.state_dict().keys()
    for name, value in old.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)


def _distributed_rebatch_worker(rank, root):
    from contact_belief_wam import trainer
    from contact_belief_wam.ema import ShardedEMA
    from test_optimizer_shards import recipe
    root = Path(root)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        with pytest.MonkeyPatch.context() as patch:
            tiny_architecture.__wrapped__(patch)
            torch.manual_seed(101)
            accumulated = ContactWAM(replace(config(), camera_mode="head_only",
                                    policy_augmentation=False, batch_stage2_policy=True)).train()
            larger = copy.deepcopy(accumulated)
            models = [accumulated, larger]
            # Hold per-sample stochastic inputs fixed when checking batch algebra;
            # real rebatching retains the distribution, not RNG-to-sample assignment.
            for policy in models:
                patch.setattr(policy.core, "sample_noise", lambda shape, device: torch.full(shape, .25, device=device))
                patch.setattr(policy.core, "sample_time", lambda size, device: torch.full((size,), .6, device=device))
            wrappers = [torch.nn.parallel.DistributedDataParallel(SequenceTrainingModel(policy),
                        find_unused_parameters=True, gradient_as_bucket_view=True) for policy in models]
            run = recipe(root)
            optimizers = [trainer.make_optimizer(policy, run, "stage2", device="cpu") for policy in models]
            emas = [ShardedEMA(policy, device="cpu") for policy in models]
            for step in range(2):
                torch.manual_seed(2000 + rank + step * 100)
                parts = [batch(mixed=True), batch(mixed=True)]
                combined = {key: torch.cat([part[key] for part in parts]) if torch.is_tensor(value)
                            else sum([part[key] for part in parts], []) for key, value in parts[0].items()}
                losses = []
                for wrapper, optimizer, micros in zip(wrappers, optimizers, (parts, [combined])):
                    optimizer.zero_grad(set_to_none=True)
                    total = 0.
                    for index, data in enumerate(micros):
                        with wrapper.no_sync() if index < len(micros) - 1 else nullcontext():
                            loss = wrapper(data)["loss"] / len(micros)
                            total += float(loss.detach())
                            loss.backward()
                    losses.append(total)
                assert losses[0] == pytest.approx(losses[1], rel=3e-4, abs=3e-5)
                assert larger.contact.updater.initial.grad.abs().sum() > 0
                for (name, old), (_, new) in zip(accumulated.named_parameters(), larger.named_parameters()):
                    assert (old.grad is None) == (new.grad is None), name
                    if old.grad is not None:
                        torch.testing.assert_close(old.grad, new.grad, rtol=5e-4, atol=3e-5,
                                                   msg=lambda message: name + ": " + message)
                for policy, optimizer, ema in zip(models, optimizers, emas):
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
                    optimizer.step()
                    ema.update(policy)
                for old, new in zip(accumulated.parameters(), larger.parameters()):
                    torch.testing.assert_close(old, new, rtol=1e-3, atol=2e-4)
                assert emas[0].values.keys() == emas[1].values.keys()
                for name, old in emas[0].values.items():
                    torch.testing.assert_close(old, emas[1].values[name], rtol=1e-3, atol=2e-5)
    finally:
        dist.destroy_process_group()


def test_fused_ddp_double_microbatch_without_accumulation(tmp_path):
    mp.spawn(_distributed_rebatch_worker, args=(str(tmp_path),), nprocs=2, join=True)
