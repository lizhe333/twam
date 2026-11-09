"""Bounded pilot orchestration; never starts training or an expansion phase."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import html
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys

from .paths import PROJECT, WORKSPACE, atomic_json
from .takeover import PilotLedger, TakeoverConfig, fingerprint, seed_overlap

DEFAULT_CHECKPOINT = WORKSPACE / "outputs/wave0-shared-8task-head-wrist-gpu-003/tasks/insert_HDMI/stage2/checkpoints/step_00010000"
DEFAULT_OUTPUT = PROJECT / "data/recovery_wave0_v1/insert_HDMI/pilot"


class DiagnosticHold(RuntimeError):
    """A complete audit disproved a prerequisite for further collection."""


def checkpoint_identity(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    deployment = json.loads((checkpoint/"deployment.json").read_text())
    complete = json.loads((checkpoint/"complete.json").read_text())
    expected = dict(task="insert_HDMI", camera="head_wrist", chunk_size=50, execute_steps=10,
        physical_steps_per_action=2, policy_step_budget=600, use_ema=True, strategy_hz=60, physics_hz=120)
    if any(deployment.get(k) != v for k,v in expected.items()) or complete.get("step") != 10000 or complete.get("stage") != "stage2":
        raise ValueError("Checkpoint does not match the approved dual-view WAVE-0 10k EMA contract")
    files = [checkpoint/n for n in ("deployment.json", "complete.json", "ema.safetensors", "norm_stats.json", "wm_norm_stats.json", "pretrained_model/config.json")]
    return dict(checkpoint=str(checkpoint), hashes=fingerprint(files), deployment=expected)


def gpu_info():
    output = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"], text=True)
    uuid, free = output.strip().split(",")
    return uuid.strip(), int(free)


def run_child(command, path, env):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(path.with_suffix(".process.json"), dict(pid=process.pid, command=command))
        try:
            return process.wait()
        finally:
            # Kill only this attempt's process group, including an orphan server.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=10)
            # The leader may have exited while its policy service is still alive.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def write_index(ledger):
    lines = ["<!doctype html><meta charset='utf-8'><title>HDMI takeover pilot</title>",
             "<h1>HDMI 小批采集</h1>", f"<pre>{html.escape(json.dumps(ledger.state['counts'], ensure_ascii=False, indent=2))}</pre>",
             f"<p>Status: {html.escape(ledger.state['status'])}. Expansion requires user review.</p>"]
    if (ledger.root/"diagnostic_hold.json").exists():
        lines.append("<p>采集因反馈前提未通过而停止，尚未完成 3＋2。<a href='diagnostic_hold.json'>诊断依据</a></p>")
    if (ledger.root/"diagnosis.html").exists():
        lines.append("<p><a href='diagnosis.html'>自然失败与正常成功：视频和曲线对照</a></p>")
    lines.append("<ul>")
    for row in ledger.state["attempts"]:
        folder = f"{row['kind']}/{row['seed']}"
        target = f"{folder}/index.html" if (ledger.root/folder/"index.html").exists() else f"{folder}/worker.log"
        reason = row.get("error") or (row.get("timing") or {}).get("reason", "")
        lines.append(f"<li><a href='{target}'>{row['seed']} {row['kind']}</a>: {html.escape(row['status'])}; qualified={row.get('qualified',False)}; {html.escape(str(reason))}</li>")
    lines.append("</ul><p><a href='pilot.json'>All attempts and provenance</a></p>")
    (ledger.root/"index.html").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--port", type=int, default=18891)
    args = p.parse_args(argv)
    args.output = args.output.resolve()
    cfg = TakeoverConfig()
    identity = checkpoint_identity(args.checkpoint)
    roots = [WORKSPACE/"contact-belief-wam/data/manifests", PROJECT/"data/manifests", WORKSPACE/"outputs/eval_suites",
             WORKSPACE/"contact-belief-wam/data/hdf5", PROJECT/"data/hdf5"]
    audit = seed_overlap(roots, range(cfg.seed_start, cfg.seed_start+cfg.natural_limit))
    if audit["conflicts"]:
        raise RuntimeError(f"Seed overlap: {audit['conflicts']}")
    args.output.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__).with_name(n) for n in ("takeover.py", "takeover_collect.py", "sim_takeover.py", "takeover_report.py")]
    atomic_json(args.output/"preflight.json", dict(config=asdict(cfg), checkpoint=identity, seed_audit=audit,
        source_hashes=fingerprint(source_files), training_enabled=False))
    if args.preflight_only:
        print(json.dumps(dict(status="preflight_passed", output=str(args.output), manifests_checked=audit["manifests_checked"],
                              collection_on_diagnostic_hold=(args.output/"diagnostic_hold.json").exists())))
        return 0
    ledger = PilotLedger(args.output, cfg, identity)
    lock = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    old = {s:signal.signal(s, interrupted) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        if ledger.state["status"] == "awaiting_user_review":
            print("Pilot is awaiting user review; no rollout started.")
            return 0
        if (args.output/"diagnostic_hold.json").exists():
            raise DiagnosticHold(f"Resolve the recorded feedback assumption first: {args.output/'diagnostic_hold.json'}")
        uuid, free = gpu_info()
        lock = (WORKSPACE/"outputs/eval_suites"/f".{uuid}.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if gpu_info()[1] < 20*1024:
            raise RuntimeError(f"GPU occupied ({free} MiB free); no worker was started")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", args.port))
        env = dict(os.environ, CBWAM_ROOT=str(PROJECT), TWAM_ROOT=str(WORKSPACE),
            PYTHONPATH=os.pathsep.join([str(PROJECT/"src"),str(WORKSPACE/"pi05-univtac/src"),str(WORKSPACE/"openpi/packages/openpi-client/src")]),
            PI05_SIM_GRAPHICS_LIBRARY_PATH=str(WORKSPACE/".cache/sim-graphics-runtime/lib"),
            OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
            HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
        atomic_json(args.output/"daemon.json", dict(pid=os.getpid(), status="collecting", running=True, gpu_uuid=uuid))

        def attempt(kind, seed, reference=None, takeover_step=None):
            directory = args.output/kind/str(seed)
            row = ledger.reserve(kind, seed, directory=str(directory), source_hashes=fingerprint(source_files))
            command = ["bash", str(PROJECT/"scripts/sim.sh"), "-m", "contact_belief_wam.sim_takeover",
                "--output",str(directory),"--checkpoint",str(args.checkpoint.resolve()),"--seed",str(seed),
                "--kind",kind,"--port",str(args.port),"--headless","--device","cuda:0",
                "--model-python",sys.executable]
            if reference:
                command += ["--reference",str(reference),"--takeover-step",str(takeover_step)]
            code = run_child(command, directory/"worker.log", env)
            result_path = directory/"result.json"
            if not result_path.exists():
                ledger.finish(row,dict(status="error", error=f"worker_exit_{code}_without_result"))
                raise RuntimeError(f"Worker failed: {directory/'worker.log'}")
            result = json.loads(result_path.read_text())
            if result["status"] == "error":
                ledger.finish(row,dict(status="error",error=result.get("error"),trajectory=result.get("trajectory")))
                raise RuntimeError(f"Attempt needs diagnosis: {result.get('error')}")
            audit_command = [sys.executable,"-m","contact_belief_wam.takeover_report",str(result_path),"--device","cuda"]
            code = run_child(audit_command, directory/"audit.log", env)
            if code not in (0, 3):
                ledger.finish(row,dict(status="audit_error",trajectory=result["trajectory"]))
                raise RuntimeError(f"Audit failed: {directory/'audit.log'}")
            result = json.loads(result_path.read_text())
            selected = result["qualified"] and (kind=="recovery" or ledger.counts()["normal_success"] < cfg.normal_target)
            ledger.finish(row,dict(status=result["status"], qualified=selected, result=str(result_path),
                trajectory=result["trajectory"], timing=result.get("timing")))
            write_index(ledger)
            print(json.dumps(dict(seed=seed,kind=kind,counts=ledger.counts(),qualified=selected)),flush=True)
            if code == 3:
                raise DiagnosticHold(f"Feedback prerequisite failed: {args.output/'diagnostic_hold.json'}")
            return result

        for seed in range(cfg.seed_start, cfg.seed_start+cfg.natural_limit):
            if ledger.target_reached(): break
            previous = next((r for r in ledger.state["attempts"] if r["kind"]=="natural" and r["seed"]==seed),None)
            if previous:
                path = args.output/"natural"/str(seed)/"result.json"
                if previous["status"] in ("error","interrupted","audit_error") or not path.exists(): continue
                natural = json.loads(path.read_text())
            else:
                natural = attempt("natural",seed)
            if (not natural.get("success") and natural["detector"].get("first_confirmed") is not None
                    and ledger.counts()["recovery"] < cfg.recovery_limit
                    and ledger.counts()["recovery_success"] < cfg.recovery_target
                    and not any(r["kind"]=="recovery" and r["seed"]==seed for r in ledger.state["attempts"])):
                attempt("recovery",seed,natural["trajectory"],natural["detector"]["first_confirmed"])
            if ledger.counts()["recovery"] >= cfg.recovery_limit and ledger.counts()["recovery_success"] < cfg.recovery_target:
                break
        ledger.pause_for_review("target_reached" if ledger.target_reached() else "attempt_allowance_exhausted")
        write_index(ledger)
    except DiagnosticHold as exc:
        ledger.state.update(status="needs_diagnosis", stop_reason="feedback_proxy_invalid", diagnostic=str(exc))
        ledger.save(); write_index(ledger)
        print(str(exc), flush=True)
    except BaseException as exc:
        ledger.state.update(status="needs_diagnosis", error=f"{type(exc).__name__}: {exc}")
        ledger.save(); write_index(ledger)
        raise
    finally:
        atomic_json(args.output/"daemon.json",dict(pid=os.getpid(),status=ledger.state["status"],running=False))
        if lock is not None: lock.close()
        ledger.close()
        for s,handler in old.items(): signal.signal(s,handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
