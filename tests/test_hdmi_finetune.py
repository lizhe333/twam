from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from test_fast_attention import tiny_architecture, cfg, inputs
from contact_belief_wam import model as cm, train, trainer
from contact_belief_wam.finetune_weights import (
    initialize_finetune, stage1_source_key, stage2_overlay_key, check_compatible)
from contact_belief_wam.finetune_data import split_groups


def configuration(stage="stage2"):
    stats = {name: {"q01": [-1.] * 8, "q99": [1.] * 8} for name in ("state", "actions")}
    return replace(cfg("sdpa"), aligned=True, stage=stage, independent_tactile=True,
                   stage1_freeze_world=False, stage2_freeze_belief=False,
                   stage2_freeze_contact_predictor=True, world_norm_stats=stats, policy_norm_stats=stats)


def test_two_stage_finetune_really_updates_wms_and_keeps_the_ema_policy(tiny_architecture, tmp_path):
    torch.manual_seed(96)
    config = configuration()
    source = cm.ContactWAM(config)
    ema = trainer.EMA(source, device="cpu")
    with torch.no_grad():
        for p in source.parameters():
            if p.requires_grad:
                p.add_(.001)
        source.optimizer_step.fill_(10000)
    directory = tmp_path / "source"
    source.save_pretrained(directory)
    ema.save(directory / "ema.safetensors")
    ema_state = load_file(str(directory / "ema.safetensors"))
    original = source.state_dict()
    ema_by_storage = {original[key].data_ptr(): value for key, value in ema_state.items()}
    stage1, report = initialize_finetune(configuration("stage1"), directory)
    assert report["optimizer_reset"] and stage1.optimizer_step.item() == 0
    for key, tensor in stage1.state_dict().items():
        if key == "optimizer_step":
            continue
        src = stage1_source_key(key)
        torch.testing.assert_close(tensor, ema_state.get(src, original[src]), rtol=0, atol=0)
    frozen = {n: p.detach().clone() for n, p in stage1.named_parameters() if not p.requires_grad}
    recipe = {"stage1": {"lr": 1e-3, "lr_world": 1e-4, "weight_decay": .01}, "fused_adamw": False}
    optimizer = trainer.make_optimizer(stage1, recipe, "stage1", device="cpu")
    obs, target, noise, time = inputs()
    stage1.train()
    stage1(obs, target)["loss_per_sample"].mean().backward()
    for module in (stage1.core.world_model, stage1.core.world_action_encoder,
                   stage1.contact.predictor, stage1.contact.updater, stage1.contact.corrector):
        assert sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None) > 0
    optimizer.step()
    for name, p in stage1.named_parameters():
        if name in frozen:
            torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
    teacher = tmp_path / "adapted"
    stage1.save_pretrained(teacher / "pretrained_model")
    stage2, report = initialize_finetune(config, directory, stage1_checkpoint=teacher)
    assert report["original_policy_preserved"] and stage2.optimizer_step.item() == 0
    updated = stage1.state_dict()
    for key, tensor in stage2.state_dict().items():
        if key == "optimizer_step":
            continue
        overlay = stage2_overlay_key(key)
        expected = updated[overlay] if overlay is not None else ema_by_storage.get(original[key].data_ptr(), original[key])
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0, msg=key)
    stage2.train()
    stage2(obs, target, noise=noise, time=time)["action_loss_per_sample"].mean().backward()
    for module in (stage2.core.world_model, stage2.core.world_action_encoder, stage2.contact.predictor):
        assert all(not p.requires_grad and p.grad is None for p in module.parameters())
    for module in (stage2.contact.updater, stage2.contact.corrector):
        assert sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None) > 0


def test_transfer_rejects_missing_world_and_changed_frozen_visual_encoder(tiny_architecture, tmp_path):
    config = configuration()
    source = cm.ContactWAM(config)
    source.save_pretrained(tmp_path / "source")
    stage1, _ = initialize_finetune(configuration("stage1"), tmp_path / "source", use_ema=False)
    with torch.no_grad():
        next(stage1.core.vision_backbone.parameters()).add_(1)
    stage1.save_pretrained(tmp_path / "teacher/pretrained_model")
    with pytest.raises(ValueError, match="frozen visual encoder changed"):
        initialize_finetune(config, tmp_path / "source", use_ema=False, stage1_checkpoint=tmp_path / "teacher")
    weights = load_file(str(tmp_path / "source/model.safetensors"))
    del weights[next(k for k in weights if ".world_model." in k)]
    save_file(weights, str(tmp_path / "source/model.safetensors"))
    with pytest.raises(ValueError, match="missing"):
        initialize_finetune(configuration("stage1"), tmp_path / "source", use_ema=False)


def test_world_lr_is_lower_and_decay_preserves_ratio(tiny_architecture):
    policy = cm.ContactWAM(configuration("stage1"))
    config = dict(lr=2.5e-5, lr_world=2.5e-6, final_lr=2.5e-6, warmup=100, steps=1000, weight_decay=.01)
    opt = trainer.make_optimizer(policy, {"stage1": config, "fused_adamw": False}, "stage1", device="cpu")
    assert {g["name"]: g["lr"] for g in opt.param_groups} == pytest.approx({"world": 2.5e-6, "contact": 2.5e-5})
    trainer.set_lr(opt, config, "stage1", 1000)
    assert {g["name"]: g["lr"] for g in opt.param_groups} == pytest.approx({"world": 2.5e-7, "contact": 2.5e-6})


def test_group_split_is_deterministic_and_keeps_paired_categories_together():
    rows = [dict(group_id=f"seed/{i}", kind=kind) for i in range(12)
            for kind in (["oracle_recovery", "normal"] if i < 8 else ["normal", "self_corrected"])]
    held = split_groups(rows, seed=42, recovery_groups=3, control_groups=1)
    assert held == split_groups(rows[::-1], seed=42, recovery_groups=3, control_groups=1)
    assert len(held) == 4
    train_ids = {r["group_id"] for r in rows if r["group_id"] not in held}
    assert not train_ids.intersection(held)
    assert split_groups(rows, seed=42, recovery_groups=0, control_groups=0) == []


def test_hdmi_recipe_reuses_four_gpus_sequentially_and_keeps_old_recipe(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    run = train.resolve_config("configs/finetune_insert_HDMI_4xa800.yaml", output=tmp_path)
    plan = train.plan(run)
    assert plan["stage1"]["gpus"] == plan["stage2"][0]["gpus"] == ["0", "1", "2", "3"]
    assert len(plan["stage2"]) == 1 and plan["stage2"][0]["task"] == "insert_HDMI"
    assert not run["stage1_freeze_world"] and run["policy_supervised_only"] and not run["expert_only"]
    assert run["stage1"]["steps"] == 1000 and run["stage2"]["steps"] == 2000
    assert run["stage1"]["validate_every"] == 0 and run["wm_validation_episode_ids"] == []
    old = train.resolve_config("configs/aligned_8xa800_expert_only_fast_mb16.yaml", output=tmp_path / "old")
    assert len(old["tasks"]) == 8 and old["expert_only"] and old["stage1_freeze_world"]


def test_incompatible_input_contract_is_rejected():
    config = configuration()
    with pytest.raises(ValueError, match="camera_mode"):
        check_compatible(config, replace(config, camera_mode="head_wrist"), source_stage="stage2")
    with pytest.raises(ValueError, match="policy normalization"):
        check_compatible(replace(config, policy_norm_stats={}), config, source_stage="stage2")


def test_policy_continuation_keeps_full_ema_and_frozen_wms(tiny_architecture, tmp_path):
    config = configuration()
    source = cm.ContactWAM(config)
    ema = trainer.EMA(source, device="cpu")
    with torch.no_grad():
        for p in source.parameters():
            if p.requires_grad:
                p.add_(.125)
        source.optimizer_step.fill_(2000)
    source.save_pretrained(tmp_path)
    ema.save(tmp_path / "ema.safetensors")
    ema_state = load_file(str(tmp_path / "ema.safetensors"))
    original = source.state_dict()
    ema_by_storage = {original[k].data_ptr(): v for k, v in ema_state.items()}
    continued, report = initialize_finetune(config, tmp_path, stage2_only=True)
    assert report["stage2_only"] and report["optimizer_reset"]
    assert "adapted_stage1" not in report
    assert continued.optimizer_step.item() == 0
    for key, tensor in continued.state_dict().items():
        if key != "optimizer_step":
            expected = ema_by_storage.get(original[key].data_ptr(), original[key])
            torch.testing.assert_close(tensor, expected, rtol=0, atol=0, msg=key)
    frozen = {n: p.detach().clone() for n, p in continued.named_parameters() if not p.requires_grad}
    recipe = {"stage2": {"lr": 1e-3, "lr_new": 1e-3, "lr_belief": 1e-3, "weight_decay": 0.},
              "fused_adamw": False}
    optimizer = trainer.make_optimizer(continued, recipe, "stage2", device="cpu")
    obs, target, noise, time = inputs()
    continued.train()
    continued(obs, target, noise=noise, time=time)["action_loss_per_sample"].mean().backward()
    for module in (continued.core.world_model, continued.core.world_action_encoder, continued.contact.predictor):
        assert all(not p.requires_grad and p.grad is None for p in module.parameters())
    for module in (continued.contact.updater, continued.contact.corrector):
        assert sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None) > 0
    optimizer.step()
    for name, p in continued.named_parameters():
        if name in frozen:
            torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="without a Stage-1 overlay"):
        initialize_finetune(config, tmp_path, stage2_only=True, stage1_checkpoint=tmp_path)
    with pytest.raises(ValueError, match="without a Stage-1 overlay"):
        initialize_finetune(configuration("stage1"), tmp_path, stage2_only=True)


def test_continuation_launch_skips_stage1_and_preserves_old_recipe(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    cfg = "configs/finetune_insert_HDMI_continue_4xa800.yaml"
    run = train.resolve_config(cfg, output=tmp_path)
    schedule = train.plan(run)
    assert schedule["stage1"]["nproc_per_node"] == 0
    assert schedule["stage1"]["reuse_full_checkpoint"] == run["finetune_from"]
    assert schedule["stage2"][0]["gpus"] == ["0", "1", "2", "3"]
    assert run["stage2"]["steps"] == 3000
    assert run["finetune_prepared_from"] != run["finetune_data_dir"]
    with pytest.raises(ValueError, match="cannot run or overlay Stage 1"):
        train.plan(run, stop_after="stage1")
    monkeypatch.setattr(train, "_prepare", lambda run: None)
    monkeypatch.setattr(train, "_require_prepared", lambda run: None)
    def task_config(run, stage, task, checkpoint):
        assert stage == "stage2" and checkpoint is None
        return {**run, "task": task, "output": str(tmp_path / "task"),
                "policy_norm_stats": {}, "world_norm_stats": {}, "stage1_checkpoint": None}
    monkeypatch.setattr(train, "stage_run", task_config)
    def jobs(records, env, status, persist, **kwargs):
        assert len(records) == 1 and records[0]["stage"] == "stage2"
        assert "--stage1-checkpoint" not in records[0]["command"]
        Path(records[0]["result"]).write_text(json.dumps(dict(complete=True, identity="test")))
    monkeypatch.setattr(train, "_run_jobs", jobs)
    def forbidden(*args, **kwargs):
        pytest.fail("Stage 1 must not run during policy continuation")
    monkeypatch.setattr(train, "_run_prediction_stage", forbidden)
    status = train.main(cfg, output=tmp_path, resume="auto")
    assert status["state"] == "done"
    assert status["stage1"]["state"] == "reused_from_full_policy"


def test_fresh_raw_recipe_retrains_both_stages_without_ema(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    run = train.resolve_config("configs/finetune_insert_HDMI_fresh_noema_4xa800.yaml", output=tmp_path)
    schedule = train.plan(run)
    assert schedule["stage1"]["gpus"] == schedule["stage2"][0]["gpus"] == ["0", "1", "2", "3"]
    assert "reuse_full_checkpoint" not in schedule["stage1"]
    assert not run.get("finetune_stage2_only")
    assert not run["finetune_use_ema"] and not run["ema_enabled"] and not run["stage2"]["shard_ema"]
    assert "head-only-expert-only-8task-all-train-fast-mb16-003" in run["finetune_from"]
    assert run["stage1"]["steps"] == 5000 and run["stage2"]["steps"] == 10000
    assert run["finetune_data_dir"] != run["finetune_prepared_from"]


def test_raw_two_stage_initialization_never_loads_source_ema(tiny_architecture, tmp_path):
    config = configuration()
    source = cm.ContactWAM(config)
    source.optimizer_step.fill_(10000)
    source.save_pretrained(tmp_path / "source")
    # Any attempted EMA read must fail; only the ordinary source model exists.
    stage1, first_report = initialize_finetune(configuration("stage1"), tmp_path / "source", use_ema=False)
    assert not first_report["use_ema"] and "ema_tensors" not in first_report
    assert stage1.optimizer_step.item() == 0
    original = source.state_dict()
    for key, value in stage1.state_dict().items():
        if key != "optimizer_step":
            torch.testing.assert_close(value, original[stage1_source_key(key)], rtol=0, atol=0)
    with torch.no_grad():
        for module in (stage1.core.world_model, stage1.contact.updater):
            for p in module.parameters():
                p.add_(.01)
    stage1.save_pretrained(tmp_path / "adapted/pretrained_model")
    stage2, second_report = initialize_finetune(config, tmp_path / "source", use_ema=False,
                                               stage1_checkpoint=tmp_path / "adapted")
    assert not second_report["use_ema"] and "ema_tensors" not in second_report
    assert stage2.optimizer_step.item() == 0
    for key, value in stage2.state_dict().items():
        if key != "optimizer_step":
            overlay = stage2_overlay_key(key)
            expected = stage1.state_dict()[overlay] if overlay is not None else original[key]
            torch.testing.assert_close(value, expected, rtol=0, atol=0, msg=key)


def test_noema_checkpoint_export_and_resume_do_not_require_shadow_weights(tmp_path, monkeypatch):
    from test_optimizer_shards import Policy, recipe
    (tmp_path / "stats.json").write_text("{}")
    (tmp_path / "tokenizer.model").write_bytes(b"test")
    policy = Policy()
    run = recipe(tmp_path)
    run["ema_enabled"] = False
    run["stage2"]["shard_optimizer_state"] = False
    def forbidden(*args, **kwargs):
        pytest.fail("EMA must not be instantiated when disabled")
    monkeypatch.setattr(trainer, "EMA", forbidden)
    monkeypatch.setattr(trainer, "ShardedEMA", forbidden)
    for rank in range(4):
        assert trainer.make_ema(policy, run, "stage2", device="cpu", rank=rank) is None
    optimizer = trainer.make_optimizer(policy, run, "stage2", device="cpu")
    policy(torch.randn(4, 4)).square().mean().backward()
    optimizer.step()
    contract = {"shard_optimizer_state": False, "ema_enabled": False}
    identity = {"fingerprint": "no-ema-test"}
    path = trainer._atomic_checkpoint(policy, optimizer, None, run, "stage2", 1, contract,
                                      identity, 1, 0, [trainer.rng_state()], 4, {})
    meta = trainer._validate_resume(path, "stage2", contract, identity, 1)
    assert meta["ema_sha256"] is None
    assert not (path / "pretrained_model/ema.safetensors").exists()
    deployment = json.loads((path / "pretrained_model/deployment.json").read_text())
    assert deployment["use_ema"] is False and deployment["ema_file"] is None
    state = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
    restored = Policy()
    restored.load_state_dict(load_file(str(path / "pretrained_model/model.safetensors")))
    restored_optimizer = trainer.make_optimizer(restored, run, "stage2", device="cpu")
    restored_optimizer.load_state_dict(state["optimizer"])
    x = torch.randn(4, 4)
    for model, opt in ((policy, optimizer), (restored, restored_optimizer)):
        opt.zero_grad()
        model(x).square().mean().backward()
        opt.step()
    for actual, expected in zip(restored.parameters(), policy.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
