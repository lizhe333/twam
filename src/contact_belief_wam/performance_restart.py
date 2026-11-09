"""Explicit transfer of training state across the Stage-2 execution optimization.

Ordinary resume stays strict. A transfer uses a new output directory, preserves
the old files, and admits only execution changes. Rebatching requires an explicit
opt-in and an unchanged global batch and rank count; training/data/teacher
contracts and unaffected source files must still match.
"""
from copy import deepcopy
import json
from pathlib import Path


EXECUTION_KEYS = {"batch_stage2_policy", "shard_ema"}
CHANGED_SOURCES = {f"src/contact_belief_wam/{name}.py" for name in (
    "config", "data", "model", "sequence", "trainer", "train", "ema", "performance_restart")}


def source_directory(run):
    source = run.get("stage2_restart_from")
    return Path(source).resolve() / "tasks" / run["task"] if source else None


def is_transfer(run, checkpoint):
    source = source_directory(run)
    return source is not None and Path(checkpoint).resolve().is_relative_to(source / "stage2/checkpoints")


def normalized_recipe(recipe, *, rebatch=False):
    value = deepcopy(recipe)
    for key in ("config", "output", "workflow_root", "stage2_restart_from", "stage1_init_checkpoint",
                "stage1_checkpoint", "stage1_checkpoint_identity", "batch_stage2_policy",
                "stage2_restart_rebatch"):
        value.pop(key, None)
    value.get("model_config", {}).pop("batch_stage2_policy", None)
    value.get("stage2", {}).pop("shard_ema", None)
    if rebatch:
        for key in ("micro_batch", "accumulation_steps"):
            value.get("stage2", {}).pop(key, None)
    return value


def _rebatch_keys(saved, current):
    if current["recipe"].get("stage2_restart_rebatch") is not True:
        return set()
    for identity in (saved, current):
        contract, recipe = identity["contract"], identity["recipe"]
        if (identity.get("stage") != "stage2" or not contract.get("expert_only")
                or contract.get("data_mix") != "1:0"):
            raise ValueError("Rebatch restart requires expert-only Stage 2 with data_mix 1:0")
        keys = ("micro_batch", "accumulation_steps", "world_size", "global_batch")
        if any(type(contract.get(key)) is not int or contract[key] <= 0 for key in keys):
            raise ValueError("Rebatch restart requires positive integer batch dimensions")
        if contract["micro_batch"] * contract["accumulation_steps"] * contract["world_size"] != contract["global_batch"]:
            raise ValueError("Rebatch restart must preserve a consistent global batch")
        for key in ("micro_batch", "accumulation_steps", "global_batch"):
            if recipe.get("stage2", {}).get(key) != contract[key]:
                raise ValueError(f"Rebatch restart recipe and contract differ: {key}")
        if recipe["stage2"].get("num_gpus") != contract["world_size"]:
            raise ValueError("Rebatch restart recipe and contract differ: num_gpus")
    return {"micro_batch", "accumulation_steps"}


def check_compatibility(saved, current):
    rebatch_keys = _rebatch_keys(saved, current)
    allowed = EXECUTION_KEYS | rebatch_keys
    old_contract = {k: v for k, v in saved["contract"].items() if k not in allowed}
    new_contract = {k: v for k, v in current["contract"].items() if k not in allowed}
    if old_contract != new_contract:
        raise ValueError("Performance restart cannot change the training/data/batch contract")
    if normalized_recipe(saved["recipe"], rebatch=bool(rebatch_keys)) != normalized_recipe(
            current["recipe"], rebatch=bool(rebatch_keys)):
        raise ValueError("Performance restart cannot change the training recipe")
    for key in ("teacher_sha256", "teacher_complete", "pi05_base_sha256", "stage"):
        if saved.get(key) != current.get(key):
            raise ValueError(f"Performance restart changed {key}")
    old, new = saved["source_hashes"], current["source_hashes"]
    unexpected = {name for name in old.keys() | new.keys()
                  if old.get(name) != new.get(name)} - CHANGED_SOURCES
    if unexpected:
        raise ValueError(f"Performance restart changed unrelated sources: {sorted(unexpected)}")


def validate_transfer(path, run, identity, world_size, rank=None):
    from .trainer import _validate_resume
    path = Path(path).resolve()
    if not is_transfer(run, path) or path.is_relative_to(Path(run["output"]).resolve()):
        raise ValueError("Performance restart must read a separate source run")
    saved = json.loads((path / "run_identity.json").read_text())
    check_compatibility(saved, identity)
    meta = _validate_resume(path, "stage2", saved["contract"], saved, world_size, rank=rank)
    report = {"source_checkpoint": str(path), "source_identity": saved["fingerprint"],
            "target_identity": identity["fingerprint"], "step": meta["step"],
            "preserved": ["parameters", "optimizer", "EMA", "learning_rate_step", "sampler_draws", "RNG"],
            "note": "Equivalent objective and BPTT; batched kernels can change numerical rounding."}
    if identity["recipe"].get("stage2_restart_rebatch"):
        keys = ("micro_batch", "accumulation_steps", "global_batch", "world_size")
        report["batch_transition"] = {
            "from": {key: saved["contract"][key] for key in keys},
            "to": {key: identity["contract"][key] for key in keys},
        }
        report["note"] = ("Same global batch, objective, BPTT and optimizer/EMA update cadence. "
                          "RNG and draw counters are restored, but rebatching changes sample-to-rank "
                          "and random-draw assignment; the training trajectory is not bitwise identical.")
    return report
