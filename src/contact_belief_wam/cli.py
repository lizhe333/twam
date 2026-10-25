from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .paths import PROJECT, WORKSPACE, configure, read_jsonl


def main(argv=None):
    configure()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "train":
        from .train import cli_main
        return cli_main(argv[1:])
    parser = argparse.ArgumentParser(prog="cbwam")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Recovery collection, manifests/statistics, tactile caches, or official-world extraction")
    prep.add_argument("action", choices=["recovery", "manifest", "cache", "world"])
    prep.add_argument("--task", choices=["insert_hole", "insert_HDMI", "both"], default="both")
    prep.add_argument("--data-root", type=Path, default=PROJECT / "data")
    prep.add_argument("--original-root", type=Path)
    prep.add_argument("--recovery-root", type=Path)
    prep.add_argument("--manifest", type=Path)
    prep.add_argument("--output", type=Path)
    prep.add_argument("--source", type=Path)
    prep.add_argument("--config", type=Path)
    prep.add_argument("--episodes", type=int, default=50)
    prep.add_argument("--max-attempts", type=int, default=1000)
    prep.add_argument("--device", default="cpu")
    prep.add_argument("--batch-size", type=int, default=8)
    prep.add_argument("--threads", type=int, default=8)
    debug = sub.add_parser("debug", help="Real-batch forward/backward and checkpoint checks, at most two updates per stage")
    debug.add_argument("--run-config", type=Path, required=True)
    debug.add_argument("--output", type=Path, required=True)
    debug.add_argument("--updates", type=int, choices=[1, 2], default=1)
    debug.add_argument("--stages", default="stage1,stage2")
    debug.add_argument("--device", default="cpu")
    debug.add_argument("--threads", type=int, default=12)
    server = sub.add_parser("serve")
    server.add_argument("--checkpoint", type=Path, required=True)
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=18780)
    server.add_argument("--device", default="cpu")
    server.add_argument("--tactile-device", default="cpu")
    server.add_argument("--anytouch-source", type=Path, default=WORKSPACE / "AnyTouch2")
    server.add_argument("--anytouch-checkpoint", type=Path)
    server.add_argument("--k", type=int)
    server.add_argument("--threads", type=int, default=12)
    evaluate = sub.add_parser("eval", help="Launch a policy server and the independent UniVTAC adapter")
    evaluate.add_argument("--task", required=True, choices=["insert_hole", "insert_HDMI"])
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--seeds", default="1000000,1000001,1000002")
    evaluate.add_argument("--device", default="cpu", help="Policy device; simulation uses cuda:0")
    evaluate.add_argument("--port", type=int, default=18780)
    evaluate.add_argument("--k", type=int)
    evaluate.add_argument("--max-policy-steps", type=int)
    sub.add_parser("train", help="Training workflow (use train --help for stages and configuration)")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        tasks = ["insert_hole", "insert_HDMI"] if args.task == "both" else [args.task]
        if args.action == "manifest":
            from .legacy.data import build_manifest
            print(json.dumps(build_manifest(args.data_root, args.original_root, args.recovery_root, tasks), indent=2))
        elif args.action == "cache":
            from .tactile import AnyTouchEncoder, cache_episodes
            rows = read_jsonl(args.manifest) if args.manifest else [row for task in tasks for row in read_jsonl(args.data_root / "manifests" / f"{task}.jsonl")]
            encoder = AnyTouchEncoder(args.source or WORKSPACE / "AnyTouch2", device=args.device, threads=args.threads)
            cache_episodes(rows, args.data_root, args.output or args.data_root / "cache/anytouch2", encoder, args.batch_size)
        elif args.action == "world":
            from .weights import extract_world
            if args.source is None or args.output is None:
                parser.error("prepare world requires --source and --output")
            print(json.dumps(extract_world(args.source, args.output), indent=2))
        else:
            for task in tasks:
                output = (args.output / task if args.task == "both" else args.output) if args.output else args.data_root / "recovery" / task
                config = args.config or PROJECT / "configs" / f"recovery_{task}.yaml"
                command = ["bash", str(PROJECT / "scripts/sim.sh"), "-m", "contact_belief_wam.sim_collect",
                           "--task", task, "--output", str(output), "--config", str(config),
                           "--episodes", str(args.episodes), "--max-attempts", str(args.max_attempts),
                           "--headless", "--device", "cuda:0"]
                subprocess.run(command, check=True)
                _verify_recovery_collection(output / "collection.json", output / "hdf5", args.episodes)
    elif args.command == "debug":
        import yaml
        from .cloud import DEFAULTS
        from .debug import run_debug
        run = {**DEFAULTS, **yaml.safe_load(args.run_config.read_text()), "output": str(args.output)}
        run["tokenizer_path"] = run.get("tokenizer_path") or run["pi05_base"]
        run_debug(run, updates=args.updates, stages=args.stages.split(","), device=args.device, threads=args.threads)
    elif args.command == "serve":
        from .serve import serve
        values = vars(args)
        values.pop("command")
        serve(**values)
    elif args.command == "eval":
        evaluate_with_server(args)


def evaluate_with_server(args):
    from .protocol import Client
    requested_seeds = _parse_seeds(args.seeds)
    args.output.mkdir(parents=True, exist_ok=True)
    evaluation_path = args.output / "evaluation.json"
    command = [sys.executable, "-m", "contact_belief_wam", "serve", "--checkpoint", str(args.checkpoint.resolve()),
               "--port", str(args.port), "--device", args.device]
    if args.k is not None:
        command += ["--k", str(args.k)]
    with (args.output / "policy-server.log").open("w") as log:
        # Keep the service and its private AnyTouch child in their own process
        # group. Normal shutdown signals only the service so its handler can
        # close and reap the child; the isolated group is a safe last resort if
        # that cleanup does not finish.
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            for _ in range(1800):
                if server.poll() is not None:
                    raise RuntimeError(f"Policy server failed; see {log.name}")
                try:
                    client = Client(port=args.port, timeout=10)
                    client.close()
                    break
                except (OSError, TimeoutError):
                    time.sleep(1)
            else:
                raise TimeoutError("Policy server did not start within 30 minutes")
            command = ["bash", str(PROJECT / "scripts/sim.sh"), "-m", "contact_belief_wam.sim_eval", "--task", args.task,
                       "--output", str(args.output.resolve()), "--seeds", args.seeds, "--port", str(args.port),
                       "--headless", "--device", "cuda:0"]
            if args.max_policy_steps:
                command += ["--max-policy-steps", str(args.max_policy_steps)]
            # Do not mistake a prior run's summary for output from this launch.
            evaluation_path.unlink(missing_ok=True)
            subprocess.run(command, check=True)
            _verify_evaluation_results(evaluation_path, requested_seeds)
        finally:
            if server.poll() is None:
                server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if hasattr(os, "killpg"):
                    try:
                        os.killpg(server.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    server.kill()
                server.wait()


def _parse_seeds(seed_text):
    try:
        seeds = [int(value.strip()) for value in seed_text.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid evaluation seed list: {seed_text!r}") from exc
    if not seeds:
        raise ValueError("Evaluation requires at least one seed")
    return seeds


def _verify_recovery_collection(report_path, hdf5_root, requested_episodes):
    """Verify that a successful simulator exit produced the requested usable HDF5s."""
    try:
        report = json.loads(Path(report_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Recovery collection incomplete: 0/{requested_episodes} qualifying episodes; "
            f"missing or unreadable {Path(report_path).name} at {report_path}"
        ) from exc
    attempts = report.get("attempts") if isinstance(report, dict) else None
    if not isinstance(attempts, list):
        raise RuntimeError(
            f"Recovery collection incomplete: 0/{requested_episodes} qualifying episodes; "
            f"{Path(report_path).name} has no attempts list"
        )

    eligible_seeds = set()
    hdf5_seeds = set()
    hdf5_root = Path(hdf5_root)
    for record in attempts:
        if not isinstance(record, dict):
            continue
        seed = record.get("seed")
        blockages = record.get("blockages")
        if (isinstance(seed, bool) or not isinstance(seed, int)
                or record.get("success") is not True
                or isinstance(blockages, bool) or not isinstance(blockages, (int, float))
                or blockages < 1
                or record.get("excluded_from_training", False)):
            continue
        eligible_seeds.add(seed)
        if (hdf5_root / f"{seed}.hdf5").is_file():
            hdf5_seeds.add(seed)

    actual = len(hdf5_seeds)
    if actual < requested_episodes:
        missing_hdf5 = len(eligible_seeds - hdf5_seeds)
        raise RuntimeError(
            f"Recovery collection incomplete: {actual}/{requested_episodes} qualifying episodes "
            f"with HDF5; collection has {len(eligible_seeds)} eligible records and "
            f"{missing_hdf5} missing HDF5 file(s); see {report_path}"
        )
    return actual


def _verify_evaluation_results(evaluation_path, requested_seeds):
    try:
        summary = json.loads(evaluation_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Simulator exited without a readable {evaluation_path.name}") from exc
    episodes = summary.get("episodes") if isinstance(summary, dict) else None
    if not isinstance(episodes, list):
        raise RuntimeError(f"{evaluation_path.name} has no episode list")

    completed = Counter()
    for episode in episodes:
        if not isinstance(episode, dict) or episode.get("complete_rollout") is not True:
            continue
        seed = episode.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            continue
        completed[seed] += 1
    missing = Counter(requested_seeds) - completed
    if missing:
        details = ", ".join(f"{seed} (missing {count})" for seed, count in sorted(missing.items()))
        raise RuntimeError(f"Simulator exited without complete rollouts for requested seeds: {details}")


if __name__ == "__main__":
    main()
