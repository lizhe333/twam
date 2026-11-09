"""Run exactly one privileged recovery diagnostic for the requested HDMI seed.

No automatic search, expansion, training, or edits to the original data. Retries
are separate explicitly launched branches and consume the existing pilot quota.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys

from .paths import PROJECT, WORKSPACE, atomic_json
from .takeover import PilotLedger, TakeoverConfig, fingerprint
from .takeover_collect import DEFAULT_CHECKPOINT, DEFAULT_OUTPUT, checkpoint_identity, gpu_info, run_child


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--takeover-step", type=int, default=40)
    p.add_argument("--config", type=Path)
    p.add_argument("--paired-live", action="store_true", help="Reproduce this same seed live and use an in-process full-state snapshot")
    p.add_argument("--policy-device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--physical-diagnostic", action="store_true",
                   help="Allow oracle-only physical validation while retaining a failed strict image replay")
    args = p.parse_args(argv)
    if args.physical_diagnostic and not args.paired_live:
        p.error('--physical-diagnostic requires --paired-live')
    if args.takeover_step <= 0 or args.takeover_step >= 600 or args.takeover_step % 10:
        p.error("takeover must be a decision boundary inside the original 600 actions")
    seed, root = 3000001, DEFAULT_OUTPUT
    source = root / "natural" / str(seed) / "result.json"
    natural = json.loads(source.read_text())
    if natural.get("success") or natural.get("status") != "failed":
        raise ValueError("Requested seed is not a recorded natural failure")
    reference = Path(natural["trajectory"])
    uuid, free = gpu_info()
    gpu_lock = (WORKSPACE / "outputs/eval_suites" / f".{uuid}.lock").open("a")
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if free < 8 * 1024:
        raise RuntimeError(f"GPU occupied; {free} MiB free")
    identity = checkpoint_identity(DEFAULT_CHECKPOINT)
    ledger = PilotLedger(root, TakeoverConfig(), identity)
    row = None
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    old = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        number = ledger.counts()["recovery"] + 1
        directory = root / "oracle" / str(seed) / f"attempt-{number:02d}"
        if directory.exists():
            raise RuntimeError(f"Refusing to overwrite diagnostic {directory}")
        config = json.loads(args.config.read_text()) if args.config else {}
        from .oracle_recovery import DEFAULTS
        if config.keys() - DEFAULTS.keys():
            raise ValueError("Unknown oracle settings")
        config = dict(DEFAULTS, **config)
        sources = [Path(__file__).with_name(n) for n in
                   ("oracle_trial.py", "oracle_recovery.py", "sim_takeover.py", "sim_oracle_pair.py", "takeover.py", "paths.py")]
        sources += [WORKSPACE/"UniVTAC/policy/Wave0Wam/deploy_policy.py",
                    WORKSPACE/"wave0-univtac-dual-view/scripts/policy-server.py",
                    WORKSPACE/"pi05-univtac/src/pi05_univtac/contact_recovery.py"]
        row = ledger.reserve_oracle(seed, directory=str(directory), takeover_step=args.takeover_step,
                                    source_hashes=fingerprint(sources), config=config, paired_live=args.paired_live,
                                    policy_device=args.policy_device, physical_diagnostic=args.physical_diagnostic)
        directory.mkdir(parents=True)
        atomic_json(directory / "oracle_config.json", config)
        atomic_json(directory / "request.json", dict(seed=seed, reference=str(reference),
            reference_sha256=fingerprint([reference]), takeover_step=args.takeover_step,
            purpose="one_privileged_recovery_preview", training_enabled=False,
            physical_diagnostic=args.physical_diagnostic, policy_device=args.policy_device,
            natural_expansion_enabled=False, automatic_retry_enabled=False,
            diagnostic_hold_preserved=(root / "diagnostic_hold.json").exists()))
        env = dict(os.environ, CBWAM_ROOT=str(PROJECT), TWAM_ROOT=str(WORKSPACE),
            PYTHONPATH=os.pathsep.join([str(PROJECT/"src"), str(WORKSPACE/"pi05-univtac/src"),
                                       str(WORKSPACE/"openpi/packages/openpi-client/src")]),
            PI05_SIM_GRAPHICS_LIBRARY_PATH=str(WORKSPACE/".cache/sim-graphics-runtime/lib"),
            OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
            HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", PYTHONUTF8="1")
        atomic_json(directory / "daemon.json", dict(pid=os.getpid(), running=True, gpu_uuid=uuid))
        worker = "contact_belief_wam.sim_oracle_pair" if args.paired_live else "contact_belief_wam.sim_takeover"
        command = ["bash", str(PROJECT/"scripts/sim.sh"), "-m", worker,
            "--output", str(directory), "--checkpoint", str(DEFAULT_CHECKPOINT), "--seed", str(seed),
            "--reference", str(reference), "--takeover-step", str(args.takeover_step),
            "--oracle-config", str(directory / "oracle_config.json"), "--headless", "--device", "cuda:0"]
        if not args.paired_live:
            command += ["--kind", "oracle"]
        else:
            command += ["--policy-device", args.policy_device]
            if args.physical_diagnostic:
                command += ["--physical-diagnostic"]
        print(json.dumps(dict(status="starting_one_oracle_branch", directory=str(directory),
                              recovery_attempts_consumed=ledger.counts()["recovery"])), flush=True)
        code = run_child(command, directory / "worker.log", env)
        result_path = directory / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else dict(
            status="error", success=False, error=f"worker_exit_{code}_without_result")
        ledger.finish(row, {k: result[k] for k in ("status", "success", "error", "trajectory", "strategy_actions", "replay") if k in result} |
                      dict(result=str(result_path), qualified=False))
        ledger.state.update(status="needs_diagnosis", stop_reason="oracle_preview_ready" if result.get("success") else "oracle_attempt_needs_review")
        ledger.save()
        atomic_json(directory / "daemon.json", dict(pid=os.getpid(), running=False, exit_code=code))
        print(json.dumps(dict(directory=str(directory), status=result["status"], success=result.get("success"),
                              actions=result.get("strategy_actions"), error=result.get("error"))), flush=True)
        return code or (1 if result["status"] in ("error", "interrupted") else 0)
    except BaseException as exc:
        if row is not None and row["status"] == "running":
            ledger.finish(row, dict(status="interrupted", qualified=False, error=str(exc)))
        raise
    finally:
        ledger.close()
        gpu_lock.close()
        for sig, handler in old.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
