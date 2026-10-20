import json

import yaml

import contact_belief_wam.cloud as cloud
from contact_belief_wam.cloud import find_resume, resolve


def _flat_run_config():
    return {
        "task": "insert_hole",
        "mode": "full",
        "data_root": "/workspace/contact-belief-data",
        "manifest": "/workspace/contact-belief-data/manifests/insert_hole.jsonl",
        "stats": "/workspace/contact-belief-data/stats/insert_hole.quantiles.json",
        "tactile_cache": "/workspace/contact-belief-data/cache/anytouch2",
        "pi05_base": "/workspace/models/pi05_base",
        "world_init": "/workspace/models/wave0_robotwin_world",
        "output": "/workspace/runs/insert_hole/full-k10",
        "num_gpus": 4,
        "global_batch": 32,
        "micro_batch": 2,
        "gradient_checkpointing": True,
    }


def test_resolve_loads_flat_config_and_cli_values_override_it(tmp_path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(_flat_run_config()))

    run, dry_run = resolve([
        "--run-config", str(config_path),
        "--task", "insert_HDMI",
        "--global-batch", "64",
        "--gradient-checkpointing", "false",
        "--dry-run",
    ])

    assert dry_run
    assert run["task"] == "insert_HDMI"
    assert run["mode"] == "full"
    assert run["global_batch"] == 64
    assert run["gradient_checkpointing"] is False
    assert run["data_root"] == "/workspace/contact-belief-data"
    assert run["tokenizer_path"] == run["pi05_base"]
    assert run["accumulation_steps"] == 8


def test_resolve_computes_gradient_accumulation_from_global_batch():
    run, dry_run = resolve([
        "--data-root", "/data",
        "--manifest", "/data/manifest.jsonl",
        "--stats", "/data/stats.json",
        "--tactile-cache", "/data/cache",
        "--pi05-base", "/models/pi05",
        "--output", "/runs/test",
        "--num-gpus", "4",
        "--global-batch", "32",
        "--micro-batch", "2",
        "--dry-run",
    ])

    assert dry_run
    assert run["accumulation_steps"] == 4


def test_eight_gpu_dry_run_uses_ga_one_and_prints_both_stages(tmp_path, monkeypatch, capsys):
    manifest, stats = tmp_path / "manifest.jsonl", tmp_path / "stats.json"
    manifest.write_text("{}\n")
    stats.write_text("{}\n")
    output = tmp_path / "run"
    config = _flat_run_config()
    config.update(manifest=str(manifest), stats=str(stats), output=str(output), world_init="none")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    args = ["--run-config", str(config_path), "--num-gpus", "8",
            "--global-batch", "128", "--micro-batch", "16",
            "--stage1-steps", "2000", "--stage2-steps", "10000", "--dry-run"]

    def unexpected_launch(*args, **kwargs):
        raise AssertionError("A dry run must not launch training")

    monkeypatch.setattr(cloud.subprocess, "run", unexpected_launch)
    run, dry_run = resolve(args)
    assert dry_run
    assert run["accumulation_steps"] == 1
    cloud.main(args)
    commands = capsys.readouterr().out.strip().splitlines()
    assert len(commands) == 2
    assert all("--num_processes 8" in command and "--multi_gpu" in command for command in commands)
    assert "--stage stage1 --steps 2000" in commands[0]
    assert "--stage stage2 --steps 10000" in commands[1]
    assert not output.exists()


def test_resume_auto_prefers_latest_complete_stage2_checkpoint(tmp_path):
    def checkpoint(stage, name, step, complete=True):
        path = tmp_path / stage / "checkpoints" / name
        path.mkdir(parents=True)
        if complete:
            (path / "complete.json").write_text(json.dumps({"stage": stage, "step": step}))
        return path

    checkpoint("stage1", "step_003000", 3000)
    checkpoint("stage2", "step_000500", 500)
    latest_stage2 = checkpoint("stage2", "step_001000", 1000)
    checkpoint("stage2", "step_002000", 2000, complete=False)

    selected = find_resume({"resume": "auto", "output": str(tmp_path)})

    assert selected == latest_stage2


def _stage2_resume_args(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    manifest = data / "manifest.jsonl"
    stats = data / "quantiles.json"
    manifest.write_text("{}\n")
    stats.write_text("{}\n")
    output = tmp_path / "run"
    output.mkdir()
    (output / "run_config.yaml").write_text("previous run\n")
    checkpoint = output / "stage2/checkpoints/step_00000002"
    checkpoint.mkdir(parents=True)
    (checkpoint / "complete.json").write_text(json.dumps({
        "stage": "stage2", "step": 2, "draws_per_rank": 0, "num_gpus": 1,
    }))
    (output / "run_state.json").write_text(json.dumps({
        "stage": "stage2", "checkpoint": str(checkpoint), "step": 2,
    }))
    args = [
        "--task", "insert_hole", "--mode", "pi05", "--data-root", str(data),
        "--manifest", str(manifest), "--stats", str(stats), "--tactile-cache", str(data / "cache"),
        "--pi05-base", str(tmp_path / "pi05"), "--world-init", "none", "--output", str(output),
        "--num-gpus", "1", "--global-batch", "2", "--micro-batch", "2", "--precision", "no",
        "--stage1-steps", "0", "--stage2-steps", "2", "--resume", "auto",
    ]
    return args, output, checkpoint


def _write_fake_final_export(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.safetensors").touch()
    (path / "config.json").write_text("{}\n")
    (path / "quantiles.json").write_text("{}\n")
    (path / "deployment.json").write_text("{}\n")
    tokenizer = path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer_config.json").write_text("{}\n")


def test_resume_completed_stage2_restores_missing_final_export(monkeypatch, tmp_path):
    args, output, checkpoint = _stage2_resume_args(tmp_path)
    commands = []

    def fake_run(command, check, env):
        commands.append(command)
        export = output / "export/final"
        _write_fake_final_export(export)

    monkeypatch.setattr(cloud.subprocess, "run", fake_run)
    cloud.main(args)

    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--stage") + 1] == "stage2"
    assert command[command.index("--steps") + 1] == "2"
    assert command[command.index("--resume") + 1] == str(checkpoint)
    assert cloud.final_export_ready(output / "export/final")
    state = json.loads((output / "run_state.json").read_text())
    assert state["stage"] == "complete"
    assert state["checkpoint"] == str(checkpoint)


def test_missing_final_export_does_not_mark_run_complete(monkeypatch, tmp_path):
    args, output, _ = _stage2_resume_args(tmp_path)
    monkeypatch.setattr(cloud.subprocess, "run", lambda *args, **kwargs: None)

    try:
        cloud.main(args)
    except RuntimeError as error:
        assert "Final export is incomplete" in str(error)
    else:
        raise AssertionError("cloud marked a run complete without a final export")

    state = json.loads((output / "run_state.json").read_text())
    assert state["stage"] == "stage2"
