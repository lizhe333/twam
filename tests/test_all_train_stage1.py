"""Exercise the formal trainer's no-validation lifecycle with CPU-only stand-ins.

CUDA placement and the large model are replaced; optimization, checkpoint
serialization, resume validation and final-teacher handoff use the real code.
"""
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.torch import load_file, save_file
import torch

from contact_belief_wam import sequence, train, trainer
from contact_belief_wam.data import SHARED_TASKS
from contact_belief_wam.timing import TrainingTiming
from test_expert_only_training import RECIPE, source_data


@pytest.mark.parametrize("resume_step", [1, 2])
def test_all_train_saves_final_teacher_and_resumes_without_validation(source_data, tmp_path, monkeypatch, resume_step):
    run = train.resolve_config(RECIPE, output=tmp_path / "run")
    for key in ("aligned_manifest_dir", "policy_manifest_dir", "task_stats_dir", "wm_stats"):
        run.pop(key)
    run.update(source_data())
    train._prepare(run)
    run = train.stage_run(run, "stage1")
    run.update(workers=0, fused_adamw=False, save_every=1)
    run["stage1"].update(num_gpus=1, micro_batch=8, global_batch=8, steps=2, warmup=0)
    output = Path(run["output"])
    for key, value in {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}.items():
        monkeypatch.setenv(key, value)

    # Override only this module's CUDA plumbing; leave global torch and real
    # optimizer/tensor/checkpoint operations on CPU.
    proxy = SimpleNamespace(**torch.__dict__)
    proxy.cuda = SimpleNamespace(is_available=lambda: True, set_device=lambda device: None,
                                 get_rng_state=lambda: None)
    proxy.device = lambda *args, **kwargs: torch.device("cpu")
    proxy.autocast = lambda *args, **kwargs: nullcontext()
    monkeypatch.setattr(trainer, "torch", proxy)
    monkeypatch.setattr(trainer, "_cuda_memory", lambda device: {})
    monkeypatch.setattr(trainer, "TrainingTiming", lambda enabled: TrainingTiming(enabled=False))
    monkeypatch.setattr(trainer, "_source_identity", lambda *args: {"fingerprint": "cpu-all-train"})
    monkeypatch.setattr(trainer, "_move_batch", lambda raw, device: raw)

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.contact = torch.nn.Linear(1, 1, bias=False)
        def forward(self, batch):
            return {"loss": (self.contact(batch["x"]) - 2).square().mean(), "metrics": {}}

    def load_model(run, stage, resume_path=None):
        policy = Policy()
        if resume_path:
            policy.load_state_dict(load_file(str(Path(resume_path) / "pretrained_model/model.safetensors")))
        return policy, {"cpu_standin": True}

    def export(policy, path, run, ema=None):
        Path(path).mkdir(parents=True)
        save_file(policy.state_dict(), str(Path(path) / "model.safetensors"))

    class Dataset:
        tasks = list(SHARED_TASKS)
        sources = ["original"] * len(tasks)
        def __len__(self):
            return len(self.tasks)
        def __getitem__(self, index):
            index = index[0] if isinstance(index, tuple) else index
            return {"x": np.ones(1, np.float32), "task": self.tasks[index], "source": "original"}

    loads = []
    def load_dataset(run, stage, validation=False):
        assert not validation, "No validation dataset may be opened in an all-training run"
        loads.append(stage)
        return Dataset()

    monkeypatch.setattr(trainer, "_load_model", load_model)
    monkeypatch.setattr(trainer, "_export", export)
    monkeypatch.setattr(trainer, "_load_dataset", load_dataset)
    monkeypatch.setattr(sequence, "SequenceTrainingModel", lambda policy: policy)
    monkeypatch.setattr(trainer, "validate_prediction_stage", lambda *args, **kwargs: pytest.fail("Validation ran"))

    trainer.train(run, "stage1", threads=1)
    final = output / "stage1/checkpoints/step_00000002"
    first = output / "stage1/checkpoints/step_00000001"
    expected_weights = load_file(str(final / "pretrained_model/model.safetensors"))
    result = json.loads((output / "result.json").read_text())
    assert result["step"] == 2 and result["selection"]["method"] == "final"
    assert result["selection"]["best_loss"] is None and not result["selection"]["stopped_early"]
    assert result["selected_checkpoint"] == result["checkpoint"] == str(final)
    assert json.loads((first / "complete.json").read_text())["selected_checkpoint"] is None
    assert json.loads((final / "complete.json").read_text())["selected_checkpoint"] == str(final)
    assert not (output / "validation.jsonl").exists()
    assert train._verify_prediction_checkpoint(final) == final

    # Simulate either interruption after an intermediate checkpoint, or after
    # saving the final checkpoint but before publishing result.json.
    (output / "result.json").unlink()
    if resume_step == 1:
        shutil.rmtree(final)
    trainer.train(run, "stage1", resume=first if resume_step == 1 else final, threads=1)
    restored = json.loads((output / "result.json").read_text())
    assert restored["step"] == 2 and restored["selected_checkpoint"] == restored["checkpoint"] == str(final)
    assert loads == ["stage1", "stage1"]
    actual_weights = load_file(str(final / "pretrained_model/model.safetensors"))
    for key, value in expected_weights.items():
        torch.testing.assert_close(actual_weights[key], value, rtol=0, atol=0)
    assert train._verify_prediction_checkpoint(final) == final
