"""Lightweight, read-only NVIDIA GPU telemetry for batch benchmarks."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import threading
import time
from io import StringIO
from pathlib import Path
from typing import Any, Iterable


_NVIDIA_SMI_QUERY = (
    "nvidia-smi",
    "--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total,power.draw",
    "--format=csv,noheader,nounits",
)
_SUBPROCESS_TIMEOUT_SECONDS = 5


def _number_or_none(value: str) -> float | None:
    value = value.strip()
    if not value or value.lower() in {"n/a", "na", "not supported", "[not supported]", "unknown"}:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _parse_nvidia_smi_output(output: str) -> list[dict[str, Any]]:
    """Parse the six CSV fields returned by the sampler's nvidia-smi query."""
    samples: list[dict[str, Any]] = []
    for line_number, row in enumerate(csv.reader(StringIO(output), skipinitialspace=True), start=1):
        if not row or (len(row) == 1 and not row[0].strip()):
            continue
        if len(row) != 6:
            raise ValueError(f"expected 6 CSV columns on line {line_number}, got {len(row)}")
        gpu_uuid, name, utilization, memory_used, memory_total, power = row
        gpu_uuid = gpu_uuid.strip()
        if not gpu_uuid:
            raise ValueError(f"missing GPU UUID on line {line_number}")
        samples.append(
            {
                "gpu_uuid": gpu_uuid,
                "name": name.strip(),
                "utilization_gpu_pct": _number_or_none(utilization),
                "memory_used_mib": _number_or_none(memory_used),
                "memory_total_mib": _number_or_none(memory_total),
                "power_draw_w": _number_or_none(power),
            }
        )
    return samples


class GpuSampler:
    """Poll ``nvidia-smi`` on a background thread and append one JSONL row per GPU.

    Each GPU row has ``timestamp``, ``gpu_uuid``, ``name``,
    ``utilization_gpu_pct``, ``memory_used_mib``, ``memory_total_mib``, and
    ``power_draw_w`` fields. Unsupported ``nvidia-smi`` values are written as
    JSON ``null``. Errors are also exposed through :attr:`errors` and, when
    possible, written as timestamped JSONL records with an ``error`` field.
    """

    def __init__(self, path: str | Path, interval: float = 1.0):
        if interval <= 0:
            raise ValueError("interval must be greater than zero")
        self.path = Path(path)
        self.interval = float(interval)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._errors: list[str] = []
        self._started = False

    @property
    def errors(self) -> list[str]:
        """Return a snapshot of errors encountered by the background sampler."""
        with self._lock:
            return list(self._errors)

    def start(self) -> GpuSampler:
        """Start polling once; repeated calls while active are harmless."""
        with self._lock:
            if self._started:
                return self
            self._started = True
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._sample_loop,
                name="gpu-telemetry-sampler",
                daemon=True,
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        """Stop and join the sampler. Safe when it was never started or failed."""
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def __enter__(self) -> GpuSampler:
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.stop()
        return False

    def _record_error(self, message: str, stream: Any = None) -> None:
        with self._lock:
            self._errors.append(message)
        if stream is not None:
            record = {"timestamp": time.time(), "error": message}
            try:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
            except OSError as exc:
                with self._lock:
                    self._errors.append(f"could not write telemetry error record: {exc}")

    def _sample_loop(self) -> None:
        try:
            stream = self.path.open("a", encoding="utf-8")
        except OSError as exc:
            self._record_error(f"could not open GPU telemetry file {self.path}: {exc}")
            return

        with stream:
            failures = 0
            while not self._stop_event.is_set():
                try:
                    result = subprocess.run(
                        _NVIDIA_SMI_QUERY,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
                    )
                    if result.returncode != 0:
                        detail = (result.stderr or "").strip()
                        suffix = f": {detail[:500]}" if detail else ""
                        raise RuntimeError(f"nvidia-smi exited with status {result.returncode}{suffix}")

                    timestamp = time.time()
                    for gpu in _parse_nvidia_smi_output(result.stdout or ""):
                        record = {"timestamp": timestamp, **gpu}
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    failures = 0
                    delay = self.interval
                except FileNotFoundError as exc:
                    self._record_error(f"nvidia-smi is unavailable: {exc}", stream)
                    break
                except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                    self._record_error(f"GPU telemetry query failed: {exc}", stream)
                    failures += 1
                    delay = min(30.0, max(1.0, self.interval) * (2 ** min(failures - 1, 5)))

                if self._stop_event.wait(delay):
                    break


def _selected_gpu_list(gpu_uuids: Iterable[str] | None) -> list[str] | None:
    if gpu_uuids is None:
        return None
    if isinstance(gpu_uuids, str):
        return [gpu_uuids]
    return list(dict.fromkeys(gpu_uuids))


def summarize_gpu_samples(
    path: str | Path,
    start_time: float,
    end_time: float,
    gpu_uuids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Summarize JSONL GPU samples inside the measured wall-clock window.

    GPU selection uses the physical UUID recorded by ``nvidia-smi``. The
    ``mean_utilization_pct`` result is an unweighted mean of per-GPU means.
    """
    selected_order = _selected_gpu_list(gpu_uuids)
    selected = set(selected_order) if selected_order is not None else None
    aggregates: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    def get_aggregate(gpu_uuid: str) -> dict[str, Any]:
        return aggregates.setdefault(
            gpu_uuid,
            {
                "uuid": gpu_uuid,
                "name": None,
                "utilization_values": [],
                "memory_used_values": [],
                "memory_total_values": [],
                "power_values": [],
                "sample_count": 0,
            },
        )

    try:
        source = Path(path).open("r", encoding="utf-8")
    except OSError as exc:
        errors.append(f"could not read GPU telemetry file {path}: {exc}")
        source = None

    if source is not None:
        with source:
            for line_number, line in enumerate(source, start=1):
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, TypeError) as exc:
                    errors.append(f"invalid JSON on GPU telemetry line {line_number}: {exc}")
                    continue
                if not isinstance(record, dict):
                    errors.append(f"invalid GPU telemetry record on line {line_number}: expected an object")
                    continue

                timestamp = record.get("timestamp")
                try:
                    timestamp = float(timestamp)
                except (TypeError, ValueError):
                    timestamp = None
                if timestamp is not None and not math.isfinite(timestamp):
                    timestamp = None

                if "error" in record:
                    if timestamp is None or start_time <= timestamp <= end_time:
                        errors.append(str(record["error"]))
                    continue
                if timestamp is None or not start_time <= timestamp <= end_time:
                    continue
                gpu_uuid = record.get("gpu_uuid")
                if not isinstance(gpu_uuid, str) or not gpu_uuid:
                    errors.append(f"missing GPU UUID on telemetry line {line_number}")
                    continue
                if selected is not None and gpu_uuid not in selected:
                    continue

                aggregate = get_aggregate(gpu_uuid)
                aggregate["sample_count"] += 1
                name = record.get("name")
                if aggregate["name"] is None and isinstance(name, str) and name:
                    aggregate["name"] = name
                for source_key, target_key in (
                    ("utilization_gpu_pct", "utilization_values"),
                    ("memory_used_mib", "memory_used_values"),
                    ("memory_total_mib", "memory_total_values"),
                    ("power_draw_w", "power_values"),
                ):
                    value = record.get(source_key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                        aggregate[target_key].append(float(value))

    if selected_order is not None:
        for gpu_uuid in selected_order:
            get_aggregate(gpu_uuid)

    per_gpu: list[dict[str, Any]] = []
    for aggregate in aggregates.values():
        utilization_values = aggregate["utilization_values"]
        used_values = aggregate["memory_used_values"]
        total_values = aggregate["memory_total_values"]
        power_values = aggregate["power_values"]
        per_gpu.append(
            {
                "uuid": aggregate["uuid"],
                "name": aggregate["name"],
                "mean_utilization_pct": (
                    sum(utilization_values) / len(utilization_values) if utilization_values else None
                ),
                "peak_memory_used_mib": max(used_values) if used_values else None,
                "memory_total_mib": max(total_values) if total_values else None,
                "mean_power_w": sum(power_values) / len(power_values) if power_values else None,
                "sample_count": aggregate["sample_count"],
            }
        )

    if selected_order is not None:
        order = {gpu_uuid: index for index, gpu_uuid in enumerate(selected_order)}
        per_gpu.sort(key=lambda item: order.get(item["uuid"], len(order)))
    means = [row["mean_utilization_pct"] for row in per_gpu if row["mean_utilization_pct"] is not None]
    return {
        "per_gpu": per_gpu,
        "mean_utilization_pct": sum(means) / len(means) if means else None,
        "errors": errors,
    }
