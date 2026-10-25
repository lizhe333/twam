import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from contact_belief_wam import shared_pipeline as pipeline


def configuration(tmp_path):
    run = {"task": "shared_wm", "mode": "full", "shared_tasks": ["insert_hole", "insert_HDMI"],
           "policy_tasks": ["insert_hole", "insert_HDMI"], "data_root": str(tmp_path / "data"),
           "shared_manifest": str(tmp_path / "shared.jsonl"), "shared_stats": str(tmp_path / "stats.json"),
           "tactile_cache": str(tmp_path / "cache"), "pi05_base": str(tmp_path / "base"),
           "tokenizer_path": str(tmp_path / "tokenizer"), "world_init": "none", "output": str(tmp_path / "run"),
           "num_gpus": 3, "micro_batch": 64, "global_batch": 192,
           "stage1_steps": 5, "stage2_steps": 10, "stage2_original_steps": 2,
           "stage2_freeze_belief": True, "stage2_freeze_contact_predictor": True}
    Path(run["shared_manifest"]).write_text("{}\n")
    Path(run["shared_stats"]).write_text("{}\n")
    for task in run["policy_tasks"]:
        manifest = Path(run["data_root"]) / "manifests" / f"{task}.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}\n")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(run))
    return path


def test_dry_run_is_one_teacher_then_two_task_policies(tmp_path, monkeypatch, capsys):
    path = configuration(tmp_path)
    monkeypatch.setattr(pipeline, "subprocess", SimpleNamespace(run=lambda *a, **k: pytest.fail("Dry-run launched a worker")))
    pipeline.main(["train", "--config", str(path), "--dry-run"])
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(r["stage"], r["task"]) for r in rows] == [("stage1", "shared_wm"), ("stage2", "insert_hole"), ("stage2", "insert_HDMI")]
    assert all(r["GA"] == 1 and r["global_batch"] == 192 for r in rows)
    assert not (tmp_path / "run").exists()


def test_both_policies_use_same_teacher_stats_and_mask(tmp_path):
    args = pipeline.parser().parse_args(["train", "--config", str(configuration(tmp_path))])
    run = pipeline.resolve(args)
    teacher = tmp_path / "teacher/pretrained_model"
    for task in run["policy_tasks"]:
        specific = pipeline.stage_run(run, "stage2", task, teacher)
        assert specific["stage1_checkpoint"] == str(teacher)
        assert specific["stats"] == run["shared_stats"]
        assert specific["manifest"].endswith(f"/{task}.jsonl")
        assert specific["policy_action_mask"] and specific["stage2_freeze_belief"]
        assert not specific["task_balanced"]
    assert pipeline.stage_run(run, "stage1")["task_balanced"]


def test_resume_skips_completed_teacher_and_policy(tmp_path, monkeypatch):
    path = configuration(tmp_path)
    calls = []
    monkeypatch.setattr(pipeline, "check_data", lambda run: [])
    monkeypatch.setattr(pipeline.cloud, "restore_final_export", lambda *a, **k: None)

    def launch(command, **kwargs):
        config = yaml.safe_load(Path(command[command.index("--run-config") + 1]).read_text())
        stage = command[command.index("--stage") + 1]
        steps = int(command[command.index("--steps") + 1])
        calls.append((stage, config["task"], config.get("stage1_checkpoint")))
        checkpoint = Path(config["output"]) / stage / "checkpoints" / f"step_{steps:08d}"
        (checkpoint / "pretrained_model").mkdir(parents=True)
        (checkpoint / "complete.json").write_text(json.dumps({"stage": stage, "step": steps}))

    monkeypatch.setattr(pipeline, "subprocess", SimpleNamespace(run=launch))
    pipeline.main(["train", "--config", str(path)])
    assert len(calls) == 3
    assert calls[1][2] == calls[2][2]
    pipeline.main(["train", "--config", str(path), "--resume", "auto"])
    assert len(calls) == 3
    state = json.loads((tmp_path / "run/pipeline_state.json").read_text())
    assert state["complete"] and len(state["tasks"]) == 2
    with pytest.raises(FileExistsError):
        pipeline.main(["train", "--config", str(path)])


def test_device_override_preserves_ga_one(tmp_path):
    args = pipeline.parser().parse_args(["train", "--config", str(configuration(tmp_path)), "--num-gpus", "8"])
    run = pipeline.resolve(args)
    assert (run["global_batch"], run["accumulation_steps"]) == (512, 1)


def test_formal_entry_rejects_debug_offload(tmp_path):
    with pytest.raises(ValueError, match="debug-only"):
        pipeline.main(["train", "--config", str(configuration(tmp_path)), "--optimizer-device", "cpu"])


def test_resume_checks_data_and_curriculum_before_overwriting(tmp_path, monkeypatch):
    from contact_belief_wam.legacy.trainer import training_contract
    path = configuration(tmp_path)
    run = pipeline.resolve(pipeline.parser().parse_args(["train", "--config", str(path), "--resume", "auto"]))
    case = pipeline.stage_run(run, "stage2", "insert_hole", tmp_path / "teacher")
    directory = Path(case["output"])
    checkpoint = directory / "stage2/checkpoints/step_00000002"
    checkpoint.mkdir(parents=True)
    (checkpoint / "complete.json").write_text(json.dumps({"step": 2, "stage": "stage2",
                                                         "training_contract": training_contract(case)}))
    (directory / "run_config.yaml").write_text("original config\n")
    case["stage2_original_steps"] += 1
    monkeypatch.setattr(pipeline, "subprocess", SimpleNamespace(run=lambda *a, **k: pytest.fail("Invalid resume launched")))
    with pytest.raises(ValueError, match="stage2_original_steps"):
        pipeline.execute_stage(case, "stage2", 10)
    assert (directory / "run_config.yaml").read_text() == "original config\n"


def test_resume_contract_detects_changed_statistics(tmp_path):
    from contact_belief_wam.legacy.trainer import training_contract, check_resume_contract
    path = configuration(tmp_path)
    run = pipeline.resolve(pipeline.parser().parse_args(["train", "--config", str(path)]))
    case = pipeline.stage_run(run, "stage1")
    saved = {"training_contract": training_contract(case)}
    check_resume_contract(case, saved)
    Path(case["stats"]).write_text('{"q01": [0]}\n')
    with pytest.raises(ValueError, match="stats_sha256"):
        check_resume_contract(case, saved)


def test_all_data_recipe_disables_validation_and_uses_task_manifests(tmp_path):
    path = configuration(tmp_path)
    config = yaml.safe_load(path.read_text())
    config.update(train_on_all_data=True, val_every=500, policy_manifest_dir=str(tmp_path / "all_train"))
    path.write_text(yaml.safe_dump(config))
    run = pipeline.resolve(pipeline.parser().parse_args(["train", "--config", str(path)]))
    assert run["val_every"] == 0
    specific = pipeline.stage_run(run, "stage2", "insert_hole", tmp_path / "teacher")
    assert specific["manifest"] == str(tmp_path / "all_train/insert_hole.jsonl")
    assert specific["stats"] == run["shared_stats"]
