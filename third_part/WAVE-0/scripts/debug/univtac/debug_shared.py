"""Bounded real-model checks, sequential on CPU or CUDA; never a formal training launcher."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
from lerobot.utils.univtac_runtime import configure, atomic_json
configure()
from lerobot.configs.univtac import TASKS

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--config", default=str(ROOT/"configs/univtac/shared_wm_8xa800.yaml"))
p.add_argument("--output", type=Path, required=True)
p.add_argument("--workspace", type=Path, default=ROOT.parent)
p.add_argument("--tasks", nargs="+", choices=TASKS, default=["insert_hole", "insert_HDMI"])
p.add_argument("--threads", type=int, default=8)
p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
p.add_argument("--debug-cpu-offload", action="store_true")
p.add_argument("--resume", action="store_true")
p.add_argument("--verify-hashes", action="store_true")
args = p.parse_args()
out = args.output.resolve()
env = {**os.environ, "PYTHONPATH": str(ROOT/"src")}
common = ["--config", str(Path(args.config).resolve()), "--workspace", str(args.workspace.resolve()), "--output", str(out)]


def invoke(command, name):
    log = out/"logs"/(name+".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "debug_check", "name": name, "log": str(log)}), flush=True)
    with log.open("a") as f:
        subprocess.run([sys.executable, "-m", "lerobot.scripts.lerobot_train", "--univtac", *command, *common],
                       env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


invoke(["prepare", *(["--verify-hashes"] if args.verify_hashes else [])], "prepare")
for stage, tasks in (("stage1", [None]), ("stage2", args.tasks)):
    for task in tasks:
        specific = ["--task", task, "--stage1-checkpoint", selected] if task else []
        training = ["train", "--stage", stage, "--device", args.device, "--threads", str(args.threads), *specific]
        if args.debug_cpu_offload:
            training += ["--debug-cpu-offload"]
        for count in (1, 2):
            invoke([*training, "--debug-updates", str(count), *(["--resume"] if count==2 or args.resume else [])],
                   f"{stage}-{task or 'shared'}-update{count}")
        invoke([*training, "--debug-updates", "2", "--resume", "--verify-only"], f"{stage}-{task or 'shared'}-restore")
        if stage=="stage1":
            selected = json.loads((out/"shared_wm/stage1/result.json").read_text())["selected_checkpoint"]
        else:
            invoke(["infer-smoke", "--task", task, "--device", args.device, "--threads", str(args.threads)], f"inference-{task}")
report = {"passed": True, "formal_training": False, "simulator_rollout": False,
          "device": args.device,
          "debug_cpu_offload": args.debug_cpu_offload,
          "shared_wm_tasks": list(TASKS), "real_stage2_tasks": args.tasks, "updates_per_stage": 2,
          "shared_wm": json.loads((out/"shared_wm/stage1/result.json").read_text()),
          "tasks": {task: {"train": json.loads((out/"tasks"/task/"stage2/result.json").read_text()),
                           "restore": json.loads((out/"tasks"/task/"stage2/restore_report.json").read_text()),
                           "inference": json.loads((out/"tasks"/task/"inference_report.json").read_text())}
                    for task in args.tasks}}
atomic_json(out/"debug_report.json", report)
print(json.dumps(report, indent=2))
