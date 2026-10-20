import json
import threading

from contact_belief_wam import benchmark_telemetry
from contact_belief_wam.benchmark_telemetry import (
    GpuSampler,
    _parse_nvidia_smi_output,
    summarize_gpu_samples,
)


def _write_samples(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _sample(timestamp, gpu_uuid, utilization, used, total, power, name="GPU"):
    return {
        "timestamp": timestamp,
        "gpu_uuid": gpu_uuid,
        "name": name,
        "utilization_gpu_pct": utilization,
        "memory_used_mib": used,
        "memory_total_mib": total,
        "power_draw_w": power,
    }


def test_summary_uses_only_measured_window_and_averages_per_gpu_means(tmp_path):
    path = tmp_path / "gpu.jsonl"
    _write_samples(
        path,
        [
            _sample(9, "GPU-A", 100, 900, 4096, 300),
            _sample(10, "GPU-A", 20, 100, 4096, 50, "A"),
            _sample(10.5, "GPU-A", 60, 150, 4096, 70, "A"),
            _sample(11, "GPU-B", 80, 500, 8192, 120, "B"),
            _sample(12, "GPU-B", 0, 10, 8192, 20),
        ],
    )

    summary = summarize_gpu_samples(path, 10, 11)

    assert summary["mean_utilization_pct"] == 60
    assert summary["errors"] == []
    assert summary["per_gpu"] == [
        {
            "uuid": "GPU-A",
            "name": "A",
            "mean_utilization_pct": 40,
            "peak_memory_used_mib": 150,
            "memory_total_mib": 4096,
            "mean_power_w": 60,
            "sample_count": 2,
        },
        {
            "uuid": "GPU-B",
            "name": "B",
            "mean_utilization_pct": 80,
            "peak_memory_used_mib": 500,
            "memory_total_mib": 8192,
            "mean_power_w": 120,
            "sample_count": 1,
        },
    ]


def test_summary_filters_by_physical_uuid_and_preserves_requested_order(tmp_path):
    path = tmp_path / "gpu.jsonl"
    _write_samples(
        path,
        [
            _sample(5, "GPU-A", 10, 100, 1000, 20),
            _sample(5, "GPU-B", 90, 500, 2000, 80),
        ],
    )

    summary = summarize_gpu_samples(path, 0, 10, gpu_uuids=["GPU-B", "GPU-MISSING"])

    assert summary["mean_utilization_pct"] == 90
    assert [gpu["uuid"] for gpu in summary["per_gpu"]] == ["GPU-B", "GPU-MISSING"]
    assert summary["per_gpu"][0]["peak_memory_used_mib"] == 500
    assert summary["per_gpu"][1]["sample_count"] == 0
    assert summary["per_gpu"][1]["mean_utilization_pct"] is None


def test_nvidia_smi_parser_maps_na_values_to_null():
    rows = _parse_nvidia_smi_output(
        'GPU-A, "NVIDIA, Example", 35, N/A, 8192, [Not Supported]\n'
        "GPU-B, Example GPU, N/A, 1024, 16384, 125.5\n"
    )

    assert rows == [
        {
            "gpu_uuid": "GPU-A",
            "name": "NVIDIA, Example",
            "utilization_gpu_pct": 35,
            "memory_used_mib": None,
            "memory_total_mib": 8192,
            "power_draw_w": None,
        },
        {
            "gpu_uuid": "GPU-B",
            "name": "Example GPU",
            "utilization_gpu_pct": None,
            "memory_used_mib": 1024,
            "memory_total_mib": 16384,
            "power_draw_w": 125.5,
        },
    ]


def test_missing_nvidia_smi_is_recorded_and_stop_is_safe(tmp_path, monkeypatch):
    called = threading.Event()

    def missing_nvidia_smi(*args, **kwargs):
        called.set()
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(benchmark_telemetry.subprocess, "run", missing_nvidia_smi)
    path = tmp_path / "gpu.jsonl"
    sampler = GpuSampler(path, interval=0.01)

    sampler.stop()
    sampler.start()
    assert called.wait(timeout=2)
    sampler.stop()

    assert any("nvidia-smi is unavailable" in error for error in sampler.errors)
    summary = summarize_gpu_samples(path, 0, 1e20)
    assert any("nvidia-smi is unavailable" in error for error in summary["errors"])
