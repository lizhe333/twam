"""Short cloud-only batch sweep; no validation, checkpoints, or formal training handoff."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback

import yaml

from . import cloud
from .paths import atomic_json


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--micro-batches", nargs="+", type=int, default=[16, 32],
                   help="Per-GPU sequence batches; global batch = GPU count × this value, GA=1")
    p.add_argument("--benchmark-stage", choices=["stage1", "stage2"], default="stage2")
    p.add_argument("--warmup-steps", type=int, default=3, help="Unmeasured optimizer updates")
    p.add_argument("--measure-steps", type=int, default=10, help="Measured optimizer updates")
    p.add_argument("--case-timeout", type=int, default=1800, help="Maximum seconds per independent run")
    p.add_argument("--worker-config", help=argparse.SUPPRESS)
    return p


def resolve_cases(argv=None):
    options, remaining = parser().parse_known_args(argv)
    if options.worker_config:
        return options, None, None, False
    # The benchmark derives batch sizes itself. Pass an internally consistent
    # placeholder through the shared parser, including when GPU count is three.
    base, dry_run = cloud.resolve([*remaining, "--global-batch", "24", "--micro-batch", "1"])
    if base["resume"] != "none":
        raise ValueError("Performance runs start independently; use --resume none and a new output")
    if (options.warmup_steps < 1 or options.measure_steps < 1 or options.case_timeout < 1
            or any(b < 1 for b in options.micro_batches)
            or len(set(options.micro_batches)) != len(options.micro_batches)):
        raise ValueError("Use positive warmup/measurement/timeout values and distinct positive micro batches")
    cases = []
    for micro in options.micro_batches:
        run = {**base, "micro_batch": micro, "global_batch": base["num_gpus"] * micro,
               "accumulation_steps": 1, "output": str(Path(base["output"]) / f"bs{micro}"),
               "benchmark": {"stage": options.benchmark_stage, "warmup_steps": options.warmup_steps,
                             "measure_steps": options.measure_steps, "purpose": "performance_only_no_checkpoint"}}
        cases.append(run)
    return options, base, cases, dry_run


def benchmark_command(run):
    command = [sys.executable, "-m", "accelerate.commands.launch", "--num_processes", str(run["num_gpus"]),
               "--num_machines", "1", "--mixed_precision", run["precision"], "--dynamo_backend", "no"]
    if run["num_gpus"] > 1:
        command.append("--multi_gpu")
    return command + ["--module", "contact_belief_wam.benchmark", "--worker-config",
                      str(Path(run["output"]) / "run_config.yaml")]


def _stop_process(process):
    # Only the new session created for this benchmark case is signalled.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 15
    while True:
        process.poll()  # Reap the leader, but also check its surviving workers.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        if time.monotonic() >= deadline:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            break
        time.sleep(.1)
    process.wait(timeout=5)


def launch_case(command, log_path, timeout):
    with Path(log_path).open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, start_new_session=True, env=os.environ.copy())

        def forward_output():
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)

        reader = threading.Thread(target=forward_output, daemon=True)
        reader.start()
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop_process(process)
        except BaseException:
            _stop_process(process)
            raise
        finally:
            # torchrun may have exited before an OOM/hung worker. Clean only
            # this case's process group before starting the next batch size.
            _stop_process(process)
            reader.join(timeout=10)
        return process.returncode, timed_out


def collect_case_result(run, returncode, timed_out=False):
    directory = Path(run["output"])
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    if returncode == 0 and result.get("status") == "ok":
        return result
    failures = [json.loads(p.read_text()) for p in directory.glob("failure_rank*.json")]
    log = directory / "launch.log"
    tail = ""
    if log.is_file():
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 32768))
            tail = stream.read().decode("utf-8", errors="replace")
    errors = json.dumps(failures).lower() + tail.lower()
    oom = "outofmemoryerror" in errors or "cuda out of memory" in errors or "cuda error: out of memory" in errors
    return {"status": "timeout" if timed_out else "oom" if oom else "failed", "returncode": returncode,
            "num_gpus": run["num_gpus"], "micro_batch": run["micro_batch"],
            "global_batch": run["global_batch"], "accumulation_steps": 1,
            "failures": failures, "log": str(log)}


def comparison(results):
    good = [r for r in results if r.get("status") == "ok" and r.get("sequences_per_second", 0) > 0]
    best = max(good, key=lambda r: r["sequences_per_second"]) if good else None
    return {"purpose": "performance_only_no_checkpoint", "results": results,
            "recommended_micro_batch": best["micro_batch"] if best else None,
            "selection": "Highest measured end-to-end sequence throughput among completed cases; inspect memory headroom too."}


def write_comparison(output, results):
    report = comparison(results)
    atomic_json(Path(output) / "comparison.json", report)
    lines = ["# A800 batch performance debug", "", "GA=1; batch counts complete belief sequences.", "",
             "| Per GPU | Global | Status | Sequences/s | Mean step (s) | GPU util (%) | Peak allocated / reserved (GiB) |",
             "|---:|---:|---|---:|---:|---:|---:|"]

    def number(value):
        return "—" if value is None else f"{value:.2f}"

    for row in results:
        lines.append(f"| {row['micro_batch']} | {row['global_batch']} | {row['status']} | "
                     f"{number(row.get('sequences_per_second'))} | {number(row.get('mean_step_seconds'))} | "
                     f"{number(row.get('gpu', {}).get('mean_utilization_pct'))} | "
                     f"{number(row.get('peak_allocated_gib'))} / {number(row.get('peak_reserved_gib'))} |")
    lines += ["", f"Throughput recommendation (per GPU): {report['recommended_micro_batch']}", "",
              "Loading and warmup are excluded. No checkpoint is saved and no formal training starts automatically."]
    (Path(output) / "comparison.md").write_text("\n".join(lines) + "\n")
    return report


def _device_uuid(properties):
    value = getattr(properties, "uuid", None)
    if value:
        value = str(value)
        return value if value.startswith(("GPU-", "MIG-")) else "GPU-" + value
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                                          "--format=csv,noheader,nounits"], text=True, timeout=5)
        for row in output.splitlines():
            pid, uuid = [part.strip() for part in row.split(",", 1)]
            if pid == str(os.getpid()):
                return uuid
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def benchmark_worker(run):
    import torch
    import torch.distributed as dist
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader

    from .benchmark_telemetry import GpuSampler, summarize_gpu_samples
    from .data import ContactDataset, MixedSequenceSampler, collate_contact
    from .preprocessing import PromptTokenizer
    from .sequence import SequenceTrainingModel, prepare_batch
    from .trainer import optimizer_for
    from .weights import initialize

    if run.get("device", "cuda") != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This performance debug requires the cloud CUDA job; --dry-run is available locally")
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision=run["precision"],
                              step_scheduler_with_optimizer=False,
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    if accelerator.num_processes != run["num_gpus"] or run["accumulation_steps"] != 1:
        raise ValueError("GPU count / GA differs from the resolved benchmark configuration")
    rank, output = accelerator.process_index, Path(run["output"])
    settings = run["benchmark"]
    stage, warmup, measured = settings["stage"], settings["warmup_steps"], settings["measure_steps"]
    sampler_gpu = None
    try:
        set_seed(run["seed"])
        if accelerator.is_main_process:
            print(json.dumps({"event": "loading", "stage": stage, "micro_batch": run["micro_batch"],
                              "global_batch": run["global_batch"], "GA": 1}), flush=True)
        cfg = cloud.model_config(run, stage)
        policy, report = initialize(cfg, world_init=run["world_init"])
        wrapper = SequenceTrainingModel(policy)
        optimizer = optimizer_for(policy, run)
        # Retain the formal schedule's horizon instead of decaying LR to zero in
        # this short performance run. We do not assess convergence here.
        horizon = int(run["stage2_steps"] if stage == "stage2" else run["stage1_pretrained_steps"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: .5 * (1 + math.cos(math.pi * min(step, horizon) / max(1, horizon))))
        wrapper, optimizer, scheduler = accelerator.prepare(wrapper, optimizer, scheduler)
        set_seed(run["seed"] + rank)
        properties = torch.cuda.get_device_properties(accelerator.device)
        device = {"rank": rank, "name": properties.name, "uuid": _device_uuid(properties),
                  "total_memory_gib": properties.total_memory / 2**30}
        devices = [None] * accelerator.num_processes
        if dist.is_initialized():
            dist.all_gather_object(devices, device)
        else:
            devices = [device]
        tokenizer = PromptTokenizer(run["tokenizer_path"])
        dataset = ContactDataset(run["manifest"], run["data_root"], run["tactile_cache"], run["stats"], cfg,
                                 seed=run["seed"])
        ratio = tuple(float(v) for v in run["data_mix"].split(":"))
        micro, steps = run["micro_batch"], warmup + measured
        sample_order = MixedSequenceSampler(dataset, draws=steps * micro, ratio=ratio, seed=run["seed"], rank=rank)
        loader = DataLoader(dataset, batch_size=micro, sampler=sample_order, collate_fn=collate_contact,
                            num_workers=run.get("workers", 2), pin_memory=True,
                            generator=torch.Generator().manual_seed(run["seed"] + rank))
        if accelerator.is_main_process:
            atomic_json(output / "initialization_report.json", report)
            atomic_json(output / "devices.json", devices)
            sampler_gpu = GpuSampler(output / "gpu_samples.jsonl")
            sampler_gpu.start()
        wrapper.train()
        optimizer.zero_grad(set_to_none=True)
        iterator = iter(loader)
        rows, window_start, clock_start = [], None, None
        for index in range(steps):
            torch.cuda.synchronize(accelerator.device)
            accelerator.wait_for_everyone()
            if index == warmup:
                torch.cuda.reset_peak_memory_stats(accelerator.device)
                window_start, clock_start = time.time(), time.perf_counter()
            started = time.perf_counter()
            raw = next(iterator)
            loaded = time.perf_counter()
            batch = prepare_batch(raw, tokenizer, accelerator.device)
            torch.cuda.synchronize(accelerator.device)
            prepared = time.perf_counter()
            with accelerator.accumulate(wrapper):
                with accelerator.autocast():
                    result = wrapper(batch)
                if not torch.isfinite(result["loss"]):
                    raise FloatingPointError(f"Nonfinite loss at benchmark update {index + 1}")
                accelerator.backward(result["loss"])
                norm = accelerator.clip_grad_norm_(wrapper.parameters(), run["grad_clip"])
                if not torch.isfinite(norm):
                    raise FloatingPointError(f"Nonfinite gradient at benchmark update {index + 1}")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            accelerator.unwrap_model(wrapper).policy.optimizer_step.fill_(index + 1)
            torch.cuda.synchronize(accelerator.device)
            finished = time.perf_counter()
            timings = torch.tensor([finished - started, loaded - started, prepared - loaded, finished - prepared,
                                    torch.cuda.max_memory_allocated(accelerator.device) / 2**30,
                                    torch.cuda.max_memory_reserved(accelerator.device) / 2**30],
                                   device=accelerator.device, dtype=torch.float64)
            counts = torch.stack([batch["sequence_mask"].sum(), batch["sequence_mask"][:, -cfg.belief_bptt_steps:].sum()])
            if dist.is_initialized():
                dist.all_reduce(timings, op=dist.ReduceOp.MAX)
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            values = timings.cpu().tolist()
            row = dict(zip(("step_seconds", "data_wait_seconds", "prepare_seconds", "compute_seconds",
                            "peak_allocated_gib", "peak_reserved_gib"), values))
            row.update(step=index + 1, phase="warmup" if index < warmup else "measure",
                       loss_rank0=float(result["loss"].detach()), grad_norm=float(norm),
                       history_updates=int(counts[0]), gradient_updates=int(counts[1]))
            if index >= warmup:
                rows.append(row)
            if accelerator.is_main_process:
                with (output / "steps.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps({"event": "benchmark_step", "micro_batch": micro, **row}), flush=True)
            del result, batch, raw
        accelerator.wait_for_everyone()
        elapsed, window_end = time.perf_counter() - clock_start, time.time()
        rank_memory = {**device, "peak_allocated_gib": torch.cuda.max_memory_allocated(accelerator.device) / 2**30,
                       "peak_reserved_gib": torch.cuda.max_memory_reserved(accelerator.device) / 2**30}
        memory = [None] * accelerator.num_processes
        if dist.is_initialized():
            dist.all_gather_object(memory, rank_memory)
        else:
            memory = [rank_memory]
        if accelerator.is_main_process:
            sampler_gpu.stop()
            uuids = [d["uuid"] for d in devices if d["uuid"]]
            gpu = summarize_gpu_samples(output / "gpu_samples.jsonl", window_start, window_end, gpu_uuids=uuids)
            gpu["errors"] = list(dict.fromkeys([*gpu["errors"], *sampler_gpu.errors]))
            if len(uuids) != len(devices):
                gpu["errors"].append("Some worker GPU UUIDs were unavailable; utilization excludes unidentified devices")
            summary = {"status": "ok", "stage": stage, "num_gpus": accelerator.num_processes,
                       "micro_batch": micro, "global_batch": run["global_batch"], "accumulation_steps": 1,
                       "warmup_steps": warmup, "measure_steps": measured, "measured_seconds": elapsed,
                       "sequences_per_second": run["global_batch"] * measured / elapsed,
                       "mean_step_seconds": statistics.mean(r["step_seconds"] for r in rows),
                       "median_step_seconds": statistics.median(r["step_seconds"] for r in rows),
                       "mean_data_wait_seconds": statistics.mean(r["data_wait_seconds"] for r in rows),
                       "mean_prepare_seconds": statistics.mean(r["prepare_seconds"] for r in rows),
                       "mean_compute_seconds": statistics.mean(r["compute_seconds"] for r in rows),
                       "peak_allocated_gib": max(m["peak_allocated_gib"] for m in memory),
                       "peak_reserved_gib": max(m["peak_reserved_gib"] for m in memory),
                       "devices": memory, "gpu": gpu, "measurement_window": [window_start, window_end],
                       "purpose": "performance_only_no_checkpoint"}
            atomic_json(output / "result.json", summary)
        accelerator.wait_for_everyone()
        accelerator.end_training()
    except BaseException as error:
        atomic_json(output / f"failure_rank{rank}.json",
                    {"rank": rank, "error_type": type(error).__name__, "message": str(error),
                     "traceback": traceback.format_exc()})
        raise
    finally:
        if sampler_gpu is not None:
            sampler_gpu.stop()


def main(argv=None):
    options, base, cases, dry_run = resolve_cases(argv)
    if options.worker_config:
        return benchmark_worker(yaml.safe_load(Path(options.worker_config).read_text()))
    output = Path(base["output"])
    if dry_run:
        for run in cases:
            print(json.dumps({"micro_batch": run["micro_batch"], "global_batch": run["global_batch"],
                              "num_gpus": run["num_gpus"], "accumulation_steps": 1,
                              "benchmark": run["benchmark"], "command": benchmark_command(run)}))
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output directory for a new performance sweep: {output}")
    for name in ("data_root", "manifest", "stats", "tactile_cache", "pi05_base", "tokenizer_path"):
        if not Path(base[name]).exists():
            raise FileNotFoundError(f"{name}: {base[name]}")
    if base["world_init"].lower() != "none" and not Path(base["world_init"]).exists():
        raise FileNotFoundError(base["world_init"])
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for run in cases:
        directory = Path(run["output"])
        directory.mkdir()
        (directory / "run_config.yaml").write_text(yaml.safe_dump(run, sort_keys=False))
        print(json.dumps({"event": "benchmark_start", "micro_batch": run["micro_batch"],
                          "global_batch": run["global_batch"], "num_gpus": run["num_gpus"],
                          "accumulation_steps": 1, "output": str(directory)}), flush=True)
        code, timed_out = launch_case(benchmark_command(run), directory / "launch.log", options.case_timeout)
        result = collect_case_result(run, code, timed_out)
        results.append(result)
        write_comparison(output, results)
        print(json.dumps({"event": "benchmark_result", **result}), flush=True)
    report = write_comparison(output, results)
    print(json.dumps({"event": "benchmark_complete", "comparison": str(output / "comparison.md"),
                      "recommended_micro_batch": report["recommended_micro_batch"]}), flush=True)
    if not any(r["status"] == "ok" for r in results):
        raise SystemExit("No benchmark case completed; inspect comparison.json and bs*/launch.log")


if __name__ == "__main__":
    main()
