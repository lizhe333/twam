"""Serial mixed-recovery supervisor. Collection and review only; never training."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import sys

from .paths import PROJECT, WORKSPACE
from .mixed_recovery import MixedLedger, load_config
from .oracle_batch_campaign import checkpoint_identity, seed_audit, durable_json, now
from .takeover import fingerprint


def owned_workers(directory):
    directory = Path(directory).resolve()
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc/"cmdline").read_bytes().decode(errors="replace").split("\0")
            if "contact_belief_wam.sim_mixed_recovery" not in command or "--output" not in command:
                continue
            if Path(command[command.index("--output")+1]).resolve() != directory:
                continue
            if (proc/"stat").read_text().rsplit(")", 1)[1].split()[0] not in {"Z", "X"}:
                found.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


def preflight(config):
    checked = {task: checkpoint_identity(cfg["checkpoint"], task, cfg["budget"]) for task,cfg in config["tasks"].items()}
    roots = [WORKSPACE/"contact-belief-wam/data/manifests", PROJECT/"data/manifests",
             WORKSPACE/"contact-belief-wam/data/hdf5", PROJECT/"data/hdf5", WORKSPACE/"outputs/eval_suites"]
    roots += list((PROJECT/"data").glob("recovery*")) + list((PROJECT/"data").glob("*preview*"))
    audit = seed_audit(config, roots)
    if audit["conflicts"]:
        raise ValueError(f"New seeds overlap existing data: {audit['conflicts']}")
    for manifest in config.get("import_manifests", []):
        if not Path(manifest).is_file():
            raise ValueError(f"Missing import manifest: {manifest}")
    modules = ("mixed_recovery", "mixed_recovery_campaign", "mixed_recovery_export", "sim_mixed_recovery",
               "hole_preview", "hole_preview_export", "hole_preview_teacher", "sim_hole_preview",
               "sim_oracle_batch", "sim_takeover", "oracle_batch_teacher", "oracle_batch_detector",
               "takeover_dataset", "takeover", "data")
    return dict(checkpoints=checked, source_hashes=fingerprint([Path(__file__).with_name(n+".py") for n in modules]),
                seed_audit=audit, training_enabled=False)


def reconcile(ledger):
    """Interrupted reservations remain consumed; completed artifacts can be exported."""
    for attempt in ledger.snapshot()["attempts"]:
        directory = Path(attempt["directory"])
        live = owned_workers(directory)
        if live:
            raise RuntimeError(f"Previous simulator still owns this attempt: {live}")
        if attempt["status"] == "running":
            path = directory/"result.json"
            result = json.loads(path.read_text()) if path.exists() else dict(status="interrupted", reason="worker_interrupted_no_automatic_retry")
            ledger.finish_seed(attempt["task"], attempt["seed"], result)


def export_finished(ledger):
    from .mixed_recovery_export import export_result
    from .oracle_batch_cleanup import cleanup_completed_native_dump
    for attempt in ledger.snapshot()["attempts"]:
        if attempt["status"] == "running":
            continue
        if owned_workers(attempt["directory"]):
            raise RuntimeError("Cannot export or clean a running attempt")
        for path in attempt["result_paths"]:
            receipt = Path(path).parent/"export_result.json"
            if receipt.exists():
                continue
            try:
                outcome = export_result(ledger, path)
                durable_json(receipt, {k:v for k,v in outcome.items() if k != "row"})
                print(json.dumps(dict(event="export", task=attempt["task"], seed=attempt["seed"],
                    result=outcome["status"], branch=path)), flush=True)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                durable_json(receipt, dict(status="technical_error", reason=reason))
                ledger.stop_task(attempt["task"], reason)
                print(json.dumps(dict(event="technical_error", task=attempt["task"], branch=path, reason=reason)), flush=True)
                break
        path = Path(attempt["directory"])/"result.json"
        if attempt["status"] == "completed" and path.exists():
            cleanup_completed_native_dump(attempt["directory"], child_completed=True, result=json.loads(path.read_text()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT/"configs/recovery_mixed_90.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--stop-after-current", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Explicitly clear this batch's stop marker")
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--import-only", action="store_true")
    args = parser.parse_args(argv)
    if args.max_attempts is not None and args.max_attempts < 1:
        parser.error("max-attempts must be positive")
    config = load_config(args.config)
    root = Path(config["output_root"])
    if args.status:
        if not (root/"campaign.json").exists():
            print(json.dumps(dict(status="not_started", training_enabled=False)))
        else:
            s = MixedLedger(root).snapshot()
            print(json.dumps(dict(status=s["status"], task_status=s["task_status"],
                counts={t:MixedLedger.counts(s,t) for t in config["tasks"]},
                daemon=json.loads((root/"daemon.json").read_text()) if (root/"daemon.json").exists() else None,
                training_enabled=False), ensure_ascii=False, indent=2))
        return 0
    if args.stop_after_current:
        if not root.is_dir():
            parser.error("Campaign has not started")
        (root/"STOP_AFTER_CURRENT").touch()
        print("Stop requested after the current branch.")
        return 0
    print(json.dumps(dict(event="preflight_started", training_enabled=False)), flush=True)
    checked = preflight(config)
    if args.dry_run:
        print(json.dumps(dict(status="preflight_passed", **checked), ensure_ascii=False), flush=True)
        return 0
    root.mkdir(parents=True, exist_ok=True)
    with (root/"supervisor.lock").open("a") as supervisor_lock:
        fcntl.flock(supervisor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {k:checked[k] for k in ("checkpoints", "source_hashes")}
        ledger = MixedLedger(root, config, identity)
        reconcile(ledger)
        if args.resume:
            (root/"STOP_AFTER_CURRENT").unlink(missing_ok=True)
        durable_json(root/"effective_config.json", config)
        durable_json(root/"preflight.json", checked)
        from .mixed_recovery_export import import_existing, rebuild_index
        from .takeover_collect import gpu_info, run_child
        gpu_lock = None
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}")
        handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            uuid, free = gpu_info()
            gpu_lock = (WORKSPACE/"outputs/eval_suites"/f".{uuid}.lock").open("a")
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if free < 20*1024:
                raise RuntimeError(f"GPU occupied: {free} MiB free")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", config["port"]))
            ledger.set_status("importing")
            durable_json(root/"daemon.json", dict(pid=os.getpid(), running=True, started_at=now(), gpu_uuid=uuid))
            import_existing(ledger)
            export_finished(ledger)
            rebuild_index(ledger)
            if args.import_only:
                ledger.set_status("imports_complete")
                return 0
            started = 0
            ledger.set_status("collecting")
            while not (root/"STOP_AFTER_CURRENT").exists():
                if args.max_attempts is not None and started >= args.max_attempts:
                    ledger.set_status("stopped", reason="requested_attempt_cap")
                    break
                item = ledger.next_seed()
                if item is None:
                    states = ledger.snapshot()["task_status"]
                    ledger.set_status("complete" if all(v == "complete" for v in states.values()) else "stopped", reason="targets_or_allowances_or_task_errors")
                    break
                task, seed = item
                directory = ledger.reserve_seed(task, seed)
                directory.mkdir(parents=True, exist_ok=True)
                rebuild_index(ledger)
                env = dict(os.environ)
                env.update(PI05_SIM_GRAPHICS_LIBRARY_PATH=str(WORKSPACE/".cache/sim-graphics-runtime/lib"),
                           PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
                command = ["bash", str(PROJECT/"scripts/sim.sh"), "-m", "contact_belief_wam.sim_mixed_recovery",
                    "--config", str(root/"effective_config.json"), "--campaign-root", str(root),
                    "--task", task, "--seed", str(seed), "--output", str(directory),
                    "--headless", "--device", config["device"]]
                print(json.dumps(dict(event="seed_started", task=task, seed=seed)), flush=True)
                code = run_child(command, directory/"worker.log", env)
                path = directory/"result.json"
                result = json.loads(path.read_text()) if path.exists() else dict(status="error", error=f"worker_exited_{code}_without_result")
                ledger.finish_seed(task, seed, result)
                export_finished(ledger)
                rebuild_index(ledger)
                started += 1
                print(json.dumps(dict(event="seed_finished", task=task, seed=seed, counts=MixedLedger.counts(ledger.snapshot(),task))), flush=True)
            else:
                ledger.set_status("stopped", reason="stop_after_current")
        except BaseException as exc:
            ledger.set_status("interrupted" if isinstance(exc, KeyboardInterrupt) else "error", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            durable_json(root/"daemon.json", dict(pid=os.getpid(), running=False, stopped_at=now(), status=ledger.snapshot()["status"]))
            rebuild_index(ledger)
            if gpu_lock is not None:
                gpu_lock.close()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
