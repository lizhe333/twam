"""Wait for the existing evaluation suite to release its GPU lock, then collect recovery data.

This launcher never stops or changes another experiment. Its log is safe to tail.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", default="0")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-attempts", type=int, default=1000)
    p.add_argument("--task", choices=["both", "insert_hole", "insert_HDMI"], default="both")
    args = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    workspace = project.parent
    uuid = subprocess.check_output(["nvidia-smi", "-i", args.gpu, "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    shared_lock = workspace / "outputs/eval_suites" / f".{uuid}.lock"
    shared_lock.parent.mkdir(parents=True, exist_ok=True)
    with shared_lock.open("a") as lock:
        print(json.dumps({"status": "waiting_for_existing_gpu_suite", "lock": str(shared_lock)}), flush=True)
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                free = int(subprocess.check_output(["nvidia-smi", "-i", args.gpu, "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
                if free >= 11000:
                    break
                fcntl.flock(lock, fcntl.LOCK_UN)
            except BlockingIOError:
                pass
            time.sleep(30)
        print(json.dumps({"status": "starting_recovery", "free_mib": free}), flush=True)
        tasks = ["insert_hole", "insert_HDMI"] if args.task == "both" else [args.task]
        for task in tasks:
            command = ["bash", str(project / "scripts/sim.sh"), "-m", "contact_belief_wam.sim_collect",
                       "--task", task, "--output", str(project / "data/recovery" / task),
                       "--config", str(project / "configs" / f"recovery_{task}.yaml"),
                       "--episodes", str(args.episodes), "--max-attempts", str(args.max_attempts),
                       "--headless", "--device", "cuda:0"]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu}
            subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
