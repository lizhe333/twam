from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from contact_belief_wam import train, trainer
from contact_belief_wam.paths import sha256


def test_reference_lr_schedule_and_scaled_new_group_final_lr():
    peak, final, warmup, steps = 2.5e-5, 2.5e-6, 1000, 10000
    assert trainer.lr_at(0, peak, final, warmup, steps) == pytest.approx(peak / (warmup + 1))
    assert trainer.lr_at(warmup, peak, final, warmup, steps) == pytest.approx(peak)
    assert trainer.lr_at(steps, peak, final, warmup, steps) == pytest.approx(final)

    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_expert = nn.Linear(2, 2)
            self.wam_future_condition = nn.Linear(2, 2)
            self.contact = nn.Linear(2, 2)

    policy = TinyPolicy()
    run = {"stage2": {"lr": peak, "lr_new": 1e-4, "warmup": warmup,
                       "final_lr": final, "steps": steps, "weight_decay": 1e-10},
           "fused_adamw": False}
    optimizer = trainer.make_optimizer(policy, run, "stage2", device="cpu")
    trainer.set_lr(optimizer, run["stage2"], "stage2", steps)
    values = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert values == pytest.approx({"policy": final, "new": final * 4})


def test_stage1_optimizer_only_includes_world_and_contact_modules():
    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_expert = nn.Linear(2, 2)
            self.world_model = nn.Linear(2, 2)
            self.contact = nn.Linear(2, 2)

    policy = TinyPolicy()
    optimizer = trainer.make_optimizer(
        policy,
        {"stage1": {"lr": 1e-4, "weight_decay": 0.01}, "fused_adamw": False},
        "stage1", device="cpu")
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimized == {id(parameter) for parameter in policy.world_model.parameters()} | {
        id(parameter) for parameter in policy.contact.parameters()}


def test_fp32_ema_updates_all_trainable_parameters_and_roundtrips(tmp_path):
    model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
    model[1].weight.requires_grad_(False)
    ema = trainer.EMA(model, decay=0.99, device="cpu")
    assert set(ema.values) == {"0.weight", "0.bias", "1.bias"}
    for value in ema.values.values():
        assert value.dtype == torch.float32
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    before = {key: value.clone() for key, value in ema.values.items()}
    ema.update(model)
    for name, value in ema.values.items():
        assert torch.allclose(value, before[name] * 0.99 + dict(model.named_parameters())[name] * 0.01)
    destination = tmp_path / "ema.safetensors"
    ema.save(destination)
    restored = trainer.EMA(model, device="cpu", initialize=False)
    restored.load(destination, model)
    assert set(restored.values) == set(ema.values)
    assert all(torch.equal(restored.values[name], ema.values[name]) for name in ema.values)


def test_fixed_validation_batches_are_balanced_and_exact():
    tasks = [f"task_{index}" for index in range(8)]

    class Dataset:
        def __init__(self):
            self.tasks = [task for task in tasks for _ in range(128)]

    batches = trainer._fixed_validation_sampler(Dataset(), tasks, 128, 8, 0, 32)
    assert len(batches) == 4
    assert all(len(batch) == 32 for batch in batches)
    for batch in batches:
        task_counts = {task: 0 for task in tasks}
        for index, _ in batch:
            task_counts[Dataset().tasks[index]] += 1
        assert set(task_counts.values()) == {4}


def test_stage1_validation_reports_common_world_loss_and_task_source_components(monkeypatch):
    tasks = [f"task_{index}" for index in range(8)]

    class Dataset:
        def __init__(self):
            self.tasks = [task for task in tasks for _ in range(128)]
            self.sources = ["original"] * len(self.tasks)

        def __len__(self):
            return len(self.tasks)

        def __getitem__(self, item):
            index, _seed = item
            return {"task": self.tasks[index], "source": self.sources[index]}

    class Model:
        training = True

        def eval(self):
            self.training = False

        def train(self, value=True):
            self.training = value

        def __call__(self, batch):
            values = torch.tensor([tasks.index(task) + 1 for task in batch["task"]], dtype=torch.float32)
            ones = torch.ones_like(values)
            return {"metrics": {
                "world_total": {"sum": values.square(), "count": values},
                "visual_initial": {"sum": 2 * values, "count": ones},
                "tactile_future": {"sum": 3 * values, "count": ones},
                "visual_corrected": {"sum": 4 * values, "count": ones},
            }}

    run = {"tasks": tasks, "stage1": {"validation_samples_per_task": 128, "micro_batch": 32}}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = trainer.validate_prediction_stage(Model(), Dataset(), lambda batch, _device: batch,
                                            run, torch.device("cpu"), 1, 0)
    assert result["loss"] == pytest.approx(4.5)
    assert result["metrics"]["visual_initial"] == pytest.approx(9.0)
    assert result["pooled_metrics"]["world_total"] == pytest.approx(204 / 36)
    assert result["per_task"]["task_7"]["world_total"] == pytest.approx(8.0)
    assert result["per_source"]["original"]["tactile_future"] == pytest.approx(13.5)
    assert result["per_task_source"]["task_7"]["original"]["visual_corrected"] == pytest.approx(32.0)


@pytest.mark.parametrize("gpus,accumulation", [(4, 2), (8, 1)])
def test_pipeline_plan_uses_configured_stage1_gpus_and_two_parallel_policies(monkeypatch, tmp_path, gpus, accumulation):
    devices = [str(i + 2) for i in range(gpus)]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(devices))
    run = train.resolve_config(f"configs/aligned_{gpus}xa800.yaml", output=tmp_path)
    result = train.plan(run)
    assert result["stage1"]["gpus"] == devices
    assert result["stage1"]["nproc_per_node"] == gpus
    assert run["stage1"]["accumulation_steps"] == accumulation
    assert [job["task"] for job in result["stage2"]] == ["insert_hole", "insert_HDMI"]
    per_task = run["stage2"]["num_gpus"]
    assert [job["gpus"] for job in result["stage2"]] == [devices[:per_task], devices[per_task:2 * per_task]]
    assert [job["nproc_per_node"] for job in result["stage2"]] == [per_task, per_task]
    assert [job["concurrency"] for job in result["stage2"]] == [2, 2]
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert train.plan(run)["stage1"]["gpus"] == list(map(str, range(gpus)))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(ValueError, match=f"{gpus} visible GPUs"):
        train.plan(run)


def test_debug_pipeline_does_not_require_formal_gpu_plan(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    monkeypatch.setattr(train, "configure", lambda: None)
    monkeypatch.setattr(train, "resolve_config",
                        lambda *args, **kwargs: {"output": str(tmp_path), "mode": "full"})
    monkeypatch.setattr(train, "_prepare", lambda run: {"prepared": True})
    monkeypatch.setattr(train, "plan",
                        lambda _run: pytest.fail("debug must not validate the formal GPU map"))
    debug = SimpleNamespace(run_debug_pipeline=lambda run, **kwargs: {
        "output": run["output"], "micro_batch": kwargs["micro_batch"]})
    monkeypatch.setitem(sys.modules, "contact_belief_wam.aligned_debug", debug)
    result = train.main(command="debug", output=tmp_path / "debug", device="cpu")
    assert result["micro_batch"] == 2


def test_config_resolves_isolated_variants_and_reference_batch_schedule(tmp_path):
    full = train.resolve_config(variant="full", output=tmp_path / "runs")
    wave0 = train.resolve_config(variant="wave0_recovery", output=tmp_path / "runs")
    assert full["output"] == str(tmp_path / "runs" / "full")
    assert wave0["output"] == str(tmp_path / "runs" / "wave0_recovery")
    assert (full["mode"], wave0["mode"]) == ("full", "wave0")
    assert full["stage1"]["num_gpus"] == 4
    assert full["stage1"]["accumulation_steps"] == 2
    assert full["stage2"]["accumulation_steps"] == 4
    assert full["stage1"]["global_batch"] == 256
    assert full["stage2"]["global_batch"] == 64
    assert full["stage1"]["steps"] == 5000 and full["stage2"]["steps"] == 10000
    assert full["stage1"]["validation_samples_per_task"] == 128
    assert full["model_config"]["k"] == 10
    assert full["model_config"]["attention_backend"] == "sdpa"
    assert wave0["model_config"]["belief_warmup_min"] == 0
    assert wave0["model_config"]["belief_warmup_max"] == 0
    assert wave0["model_config"]["belief_bptt_steps"] == 1


@pytest.mark.parametrize("batch", [0, -128, 100, 256.0])
def test_config_rejects_invalid_accumulation_batch(tmp_path, batch):
    import yaml
    source = Path(__file__).parents[1] / "configs/aligned_4xa800.yaml"
    config = yaml.safe_load(source.read_text())
    config["stage1"]["global_batch"] = batch
    target = tmp_path / "invalid.yaml"
    target.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="global_batch"):
        train.resolve_config(target)


def test_resolved_model_config_preserves_smoke_horizon_override(tmp_path):
    import yaml

    source = Path(__file__).parents[1] / "configs/aligned_8xa800.yaml"
    config = yaml.safe_load(source.read_text())
    config["k"] = 5
    target = tmp_path / "smoke.yaml"
    target.write_text(yaml.safe_dump(config))
    resolved = train.resolve_config(target, variant="full", output=tmp_path / "smoke")
    assert resolved["model_config"]["k"] == 5


def test_pipeline_training_command_uses_torchrun_and_explicit_resume(tmp_path):
    config = tmp_path / "run_config.yaml"
    command = train._training_command(config, "stage2", resume=tmp_path / "checkpoint")
    assert "torch.distributed.run" in command
    assert command[command.index("--nproc_per_node") + 1] == "2"
    assert command[command.index("--resume") + 1] == str(tmp_path / "checkpoint")
    assert "--stage1-checkpoint" not in command


def test_stage1_handoff_reads_result_from_stage_directory_sibling(tmp_path):
    checkpoint = tmp_path / "shared_wm/stage1/checkpoints/step_00000500"
    model = checkpoint / "pretrained_model/model.safetensors"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"stage1 model")
    identity = {"fingerprint": "stage1-run"}
    (checkpoint / "complete.json").write_text(json.dumps({
        "stage": "stage1", "model_sha256": sha256(model),
    }))
    (checkpoint / "run_identity.json").write_text(json.dumps(identity))
    (checkpoint.parents[2] / "result.json").write_text(json.dumps({
        "complete": True, "selected_checkpoint": str(checkpoint),
    }))
    assert train._verify_prediction_checkpoint(checkpoint, identity) == checkpoint


def test_checkpoint_resume_contract_rejects_changed_optimizer_or_model_bytes(tmp_path):
    checkpoint = tmp_path / "stage1/checkpoints/step_00000010"
    checkpoint.mkdir(parents=True)
    model = checkpoint / "pretrained_model/model.safetensors"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"model")
    training = checkpoint / "training.pt"
    training.write_bytes(b"optimizer state")
    contract = {"lr": 1e-4, "global_batch": 256}
    identity = {"fingerprint": "run-1"}
    (checkpoint / "complete.json").write_text(json.dumps({
        "stage": "stage1", "world_size": 8, "training_contract": contract,
        "model_sha256": sha256(model), "training_sha256": sha256(training)}))
    (checkpoint / "run_identity.json").write_text(json.dumps(identity))
    trainer._validate_resume(checkpoint, "stage1", contract, identity, 8)
    with pytest.raises(ValueError, match="contract changed"):
        trainer._validate_resume(checkpoint, "stage1", {**contract, "lr": 2e-4}, identity, 8)
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        trainer._validate_resume(checkpoint, "stage1", contract, identity, 8)
