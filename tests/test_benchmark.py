import json

import pytest
import yaml

from contact_belief_wam import benchmark, cloud


def arguments(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    manifest, stats = data / "manifest.jsonl", data / "stats.json"
    manifest.write_text("{}\n")
    stats.write_text("{}\n")
    return ["--task", "insert_hole", "--mode", "full", "--data-root", str(data),
            "--manifest", str(manifest), "--stats", str(stats), "--tactile-cache", str(data),
            "--pi05-base", str(data), "--tokenizer-path", str(data), "--world-init", "none",
            "--output", str(tmp_path / "sweep"), "--num-gpus", "3", "--resume", "none",
            "--micro-batches", "16", "32", "--warmup-steps", "3", "--measure-steps", "10"]


def test_three_gpu_sweep_resolves_48_and_96_sequences_with_ga_one(tmp_path):
    _, _, cases, dry_run = benchmark.resolve_cases([*arguments(tmp_path), "--dry-run"])
    assert dry_run
    assert [(r["micro_batch"], r["global_batch"], r["accumulation_steps"]) for r in cases] == [(16, 48, 1), (32, 96, 1)]
    for case in cases:
        command = benchmark.benchmark_command(case)
        assert command[command.index("--num_processes") + 1] == "3"
        assert "--multi_gpu" in command
        assert command[command.index("--module") + 1] == "contact_belief_wam.benchmark"
        assert case["benchmark"] == {"stage": "stage2", "warmup_steps": 3, "measure_steps": 10,
                                     "purpose": "performance_only_no_checkpoint"}


@pytest.mark.parametrize("micro,global_batch", [(16, 48), (32, 96)])
def test_formal_entry_also_accepts_three_gpu_ga_one(tmp_path, micro, global_batch):
    args = arguments(tmp_path)
    # Shared flags only, before benchmark-specific options.
    args = args[:args.index("--micro-batches")]
    run, _ = cloud.resolve([*args, "--micro-batch", str(micro), "--global-batch", str(global_batch), "--dry-run"])
    assert run["accumulation_steps"] == 1


def test_dry_run_has_no_launch_or_output_side_effects(tmp_path, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise AssertionError("Dry run must not launch anything")

    monkeypatch.setattr(benchmark, "launch_case", fail)
    benchmark.main([*arguments(tmp_path), "--dry-run"])
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["global_batch"] for r in rows] == [48, 96]
    assert not (tmp_path / "sweep").exists()


def test_oom_preserves_successful_case_and_does_not_start_formal_training(tmp_path, monkeypatch):
    calls = []

    def fake_launch(command, log_path, timeout):
        run = yaml.safe_load(open(command[command.index("--worker-config") + 1]).read())
        assert "contact_belief_wam.legacy.trainer" not in command
        assert run["accumulation_steps"] == 1
        calls.append(run["micro_batch"])
        if run["micro_batch"] == 16:
            (log_path.parent / "result.json").write_text(json.dumps({
                "status": "ok", "micro_batch": 16, "global_batch": 48, "sequences_per_second": 12.,
                "mean_step_seconds": 4., "gpu": {"mean_utilization_pct": 85.},
                "peak_allocated_gib": 60., "peak_reserved_gib": 65.,
            }))
            return 0, False
        log_path.write_text("torch.OutOfMemoryError: CUDA out of memory\n")
        return 1, False

    monkeypatch.setattr(benchmark, "launch_case", fake_launch)
    benchmark.main(arguments(tmp_path))
    report = json.loads((tmp_path / "sweep/comparison.json").read_text())
    assert calls == [16, 32]
    assert [r["status"] for r in report["results"]] == ["ok", "oom"]
    assert report["recommended_micro_batch"] == 16
    assert (tmp_path / "sweep/comparison.md").is_file()
    assert not list((tmp_path / "sweep").rglob("*.safetensors"))


def test_all_failed_cases_still_write_comparison_then_fail_job(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "launch_case", lambda *args: (1, False))
    with pytest.raises(SystemExit, match="No benchmark case completed"):
        benchmark.main(arguments(tmp_path))
    report = json.loads((tmp_path / "sweep/comparison.json").read_text())
    assert len(report["results"]) == 2
    assert report["recommended_micro_batch"] is None


def test_selection_uses_throughput_rather_than_utilization_alone():
    report = benchmark.comparison([
        {"status": "ok", "micro_batch": 16, "sequences_per_second": 15., "gpu": {"mean_utilization_pct": 85.}},
        {"status": "ok", "micro_batch": 32, "sequences_per_second": 14., "gpu": {"mean_utilization_pct": 99.}},
    ])
    assert report["recommended_micro_batch"] == 16


def test_benchmark_does_not_resume_formal_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="Performance runs start independently"):
        benchmark.resolve_cases([*arguments(tmp_path), "--resume", "auto"])


def test_cleanup_kills_surviving_workers_after_launcher_exits(monkeypatch):
    events, clock = [], [0.]

    class ExitedLauncher:
        pid = 123456789

        def poll(self):
            return 1

        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr(benchmark.os, "killpg", lambda pid, sig: events.append((pid, sig)))
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(benchmark.time, "sleep", lambda duration: clock.__setitem__(0, clock[0] + duration))
    benchmark._stop_process(ExitedLauncher())
    assert (ExitedLauncher.pid, benchmark.signal.SIGTERM) in events
    assert (ExitedLauncher.pid, benchmark.signal.SIGKILL) in events
