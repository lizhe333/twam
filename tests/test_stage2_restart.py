"""Completed Stage 1 can initialize a new Stage-2 batch/GPU recipe."""
import json
from pathlib import Path

import pytest
import yaml

from contact_belief_wam import train
from contact_belief_wam.config import ContactConfig
from contact_belief_wam.paths import sha256


@pytest.fixture
def teacher_run(tmp_path):
    run = train.resolve_config("configs/aligned_8xa800.yaml", output=tmp_path / "new")
    stats = {key: {"q01": [-1.] * 8, "q99": [1.] * 8} for key in ("state", "actions")}
    base = tmp_path / "base.safetensors"
    base.write_bytes(b"frozen visual encoder and policy initialization")
    stats_file = tmp_path / "stats.json"
    stats_file.write_text(json.dumps(stats))
    run.update(pi05_base=str(base), wm_norm_stats=stats,
               task_stats_by_task={task: str(stats_file) for task in run["policy_tasks"]},
               policy_manifests={task: "unused" for task in run["policy_tasks"]})
    teacher = tmp_path / "old/shared_wm/stage1/checkpoints/step_00004500"
    model = teacher / "pretrained_model/model.safetensors"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"trained world and contact modules")
    config = ContactConfig(**{**run["model_config"], "world_norm_stats": stats})
    (model.parent / "config.json").write_text(json.dumps(config.to_dict()))
    (teacher / "complete.json").write_text(json.dumps({
        "stage": "stage1", "model_sha256": sha256(model), "identity": "old-source-and-recipe",
    }))
    (teacher / "run_identity.json").write_text(json.dumps({
        "stage": "stage1", "fingerprint": "old-source-and-recipe", "pi05_base_sha256": sha256(base),
    }))
    (teacher.parents[2] / "result.json").write_text(json.dumps({
        "complete": True, "selected_checkpoint": str(teacher),
    }))
    return teacher, run


def test_eight_gpu_stage2_plan_reuses_teacher_and_keeps_global_batch(monkeypatch, teacher_run):
    teacher, run = teacher_run
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    run["stage1_init_checkpoint"] = str(teacher)
    schedule = train.plan(run)
    assert schedule["stage1"]["gpus"] == []
    assert schedule["stage1"]["reuse_checkpoint"] == str(teacher)
    assert [job["gpus"] for job in schedule["stage2"]] == [["0", "1", "2", "3"], ["4", "5", "6", "7"]]
    assert [job["nproc_per_node"] for job in schedule["stage2"]] == [4, 4]
    assert (run["stage2"]["micro_batch"], run["stage2"]["global_batch"], run["stage2"]["accumulation_steps"]) == (8, 64, 2)
    assert run["stage2"]["shard_optimizer_state"] is True
    assert run["stage2"]["ema_device"] == "cuda"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    with pytest.raises(ValueError, match="8 visible GPUs"):
        train.plan(run)


@pytest.mark.parametrize("key,value", [("camera_mode", "head_only"), ("k", 5),
                                       ("world_norm_stats", {"changed": True})])
def test_teacher_transfer_checks_latent_contract(teacher_run, key, value):
    teacher, run = teacher_run
    assert train._verify_stage1_initialization(teacher, run) == teacher
    target = teacher / "pretrained_model/config.json"
    config = json.loads(target.read_text())
    config[key] = value
    target.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=key):
        train._verify_stage1_initialization(teacher, run)


def test_teacher_transfer_checks_model_and_encoder_bytes(teacher_run):
    teacher, run = teacher_run
    base = Path(run["pi05_base"])
    original = base.read_bytes()
    base.write_bytes(b"different visual encoder")
    with pytest.raises(ValueError, match="pi05_base"):
        train._verify_stage1_initialization(teacher, run)
    base.write_bytes(original)
    (teacher / "pretrained_model/model.safetensors").write_bytes(b"damaged teacher")
    with pytest.raises(RuntimeError, match="hash check"):
        train._verify_stage1_initialization(teacher, run)


def test_stage2_restart_skips_stage1_and_keeps_teacher_artifacts(monkeypatch, teacher_run):
    teacher, run = teacher_run
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    original = {p: p.read_bytes() for p in teacher.rglob('*') if p.is_file()}
    monkeypatch.setattr(train, "resolve_config", lambda *a, **kw: dict(run))
    monkeypatch.setattr(train, "_prepare", lambda r: None)
    monkeypatch.setattr(train, "_require_prepared", lambda r: None)
    monkeypatch.setattr(train, "_run_prediction_stage", lambda *a: pytest.fail("Stage 1 retrained"))
    calls = []
    def jobs(records, env, status, persist, **kwargs):
        for record in records:
            command = record["command"]
            assert command[command.index("--nproc_per_node") + 1] == "4"
            config = yaml.safe_load(Path(command[command.index("--run-config") + 1]).read_text())
            assert config["stage1_checkpoint"] == str(teacher)
            assert config["stage1_checkpoint_identity"] == "old-source-and-recipe"
            assert config["stage2"]["micro_batch"] == 8
            calls.append(record["task"])
            Path(record["result"]).write_text(json.dumps({"complete": True, "identity": record["task"]}))
    monkeypatch.setattr(train, "_run_jobs", jobs)
    result = train.main(stage1_checkpoint=teacher, resume="auto")
    assert result["state"] == "done" and result["stage1"]["state"] == "reused"
    assert calls == ["insert_hole", "insert_HDMI"]
    assert all(path.read_bytes() == data for path, data in original.items())


@pytest.mark.parametrize("arguments", [{"variant": "wave0_recovery"}, {"command": "debug"},
                                      {"stop_after": "stage1"}])
def test_teacher_flag_rejects_incompatible_workflows(teacher_run, arguments):
    teacher, _ = teacher_run
    with pytest.raises(ValueError, match="--stage1-checkpoint"):
        train.main(stage1_checkpoint=teacher, plan_only=True, **arguments)
