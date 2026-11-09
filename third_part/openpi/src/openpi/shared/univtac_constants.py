from __future__ import annotations

import json
from openpi.shared.univtac_runtime import root

TASKS = (
    "grasp_classify", "insert_HDMI", "insert_hole", "insert_tube",
    "lift_bottle", "lift_can", "pull_out_key", "put_bottle_in_shelf",
)
HORIZON = 50
EXECUTE_STEPS = 10
FPS = 60
SEED = 42
OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
EXPECTED_SEGMENTS = 801
EXPECTED_PAIRS = 152079
MAX_ACTIONS = dict(zip(TASKS, (56, 117, 211, 272, 388, 318, 235, 410)))
NATIVE_LIMITS = {task: (600 if task == "insert_HDMI" else 500 if task == "lift_bottle" else 300) for task in TASKS}


def task_names(task: str) -> tuple[str, ...]:
    if task == "all":
        return TASKS
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose from {TASKS}")
    return (task,)


def prompt(task: str) -> str:
    task_names(task)
    text = json.loads((root() / "UniVTAC/instructions" / f"{task}.json").read_text())["seen"][0]
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Missing seen[0] instruction for {task}")
    return text


def action_budget(task: str, native_limit: int | None = None) -> int:
    import math
    return max(native_limit or NATIVE_LIMITS[task], math.ceil(1.5 * MAX_ACTIONS[task]))
