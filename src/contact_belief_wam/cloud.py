"""Manual cloud-job entry: resolve a run, then launch and release each DDP stage."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from .config import ContactConfig
from .paths import PROJECT, atomic_json


def boolean(value):
    if isinstance(value, bool):
        return value
    if value.lower() not in {"true", "false"}:
        raise argparse.ArgumentTypeError("Expected true or false")
    return value.lower() == "true"


DEFAULTS = dict(task="insert_hole", mode="full", num_gpus=4, global_batch=32, micro_batch=2,
                precision="bf16", gradient_checkpointing=True, k=10, data_mix="1:1",
                belief_warmup_min=0, belief_warmup_max=4, belief_bptt_steps=2,
                stage1_steps="auto", stage1_pretrained_steps=2000, stage1_scratch_steps=20000,
                stage2_steps=10000, lr_new=1e-4, lr_policy=2.5e-5, grad_clip=1.,
                val_every=500, save_every=1000, val_batches=8, seed=42, resume="none",
                python="python3.10", bootstrap_env=True, wheelhouse=str(PROJECT / "wheelhouse"),
                world_init="none", workers=2, tokenizer_path="", device="cuda")
for field in fields(ContactConfig):
    if field.name.startswith("belief_to_"):
        DEFAULTS[field.name] = True


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-config")
    for key in ("task", "mode", "data_root", "manifest", "stats", "tactile_cache", "pi05_base",
                "world_init", "output", "python", "wheelhouse", "precision", "data_mix", "resume",
                "tokenizer_path", "device"):
        p.add_argument("--" + key.replace("_", "-"))
    for key in ("num_gpus", "global_batch", "micro_batch", "k", "belief_warmup_min", "belief_warmup_max",
                "belief_bptt_steps", "stage1_pretrained_steps", "stage1_scratch_steps", "stage2_steps",
                "val_every", "save_every", "val_batches", "seed", "workers"):
        p.add_argument("--" + key.replace("_", "-"), type=int)
    p.add_argument("--stage1-steps")
    for key in ("lr_new", "lr_policy", "grad_clip"):
        p.add_argument("--" + key.replace("_", "-"), type=float)
    for key in ("bootstrap_env", "gradient_checkpointing", *(k for k in DEFAULTS if k.startswith("belief_to_"))):
        p.add_argument("--" + key.replace("_", "-"), type=boolean)
    p.add_argument("--dry-run", action="store_true", help="Print resolved stage commands; do not initialize or train")
    return p


def resolve(argv=None):
    ns = vars(parser().parse_args(argv))
    config = dict(DEFAULTS)
    if ns["run_config"]:
        config.update(yaml.safe_load(Path(ns["run_config"]).read_text()))
    config.update({k: v for k, v in ns.items() if v is not None and k not in {"run_config", "dry_run"}})
    for key in ("data_root", "manifest", "stats", "tactile_cache", "pi05_base", "output"):
        if not config.get(key):
            raise ValueError(f"--{key.replace('_', '-')} is required")
    if config["mode"] not in {"pi05", "wave0", "direct_touch", "full"}:
        raise ValueError("Invalid mode")
    divisor = config["num_gpus"] * config["micro_batch"]
    if config["num_gpus"] not in {1, 2, 3, 4, 8} or config["global_batch"] % divisor:
        raise ValueError("Use 1/2/3/4/8 GPUs and a global batch divisible by GPUs × micro batch")
    config["accumulation_steps"] = config["global_batch"] // divisor
    if config["accumulation_steps"] < 1:
        raise ValueError("Global batch is smaller than one micro batch")
    if config["precision"] not in {"bf16", "no"}:
        raise ValueError("Precision must be bf16 or no")
    config["tokenizer_path"] = config["tokenizer_path"] or config["pi05_base"]
    if config["stage1_steps"] != "auto":
        config["stage1_steps"] = int(config["stage1_steps"])
    return config, ns["dry_run"]


def complete_checkpoints(run, stage):
    candidates = []
    for path in (Path(run) / stage / "checkpoints").glob("step_*"):
        if (path / "complete.json").is_file():
            metadata = json.loads((path / "complete.json").read_text())
            candidates.append((metadata["step"], path))
    return sorted(candidates)


def find_resume(config):
    value = str(config["resume"])
    if value == "none":
        return None
    if value != "auto":
        path = Path(value).resolve()
        if not (path / "complete.json").is_file():
            raise ValueError(f"Incomplete checkpoint: {path}")
        return path
    for stage in ("stage2", "stage1"):
        candidates = complete_checkpoints(config["output"], stage)
        if candidates:
            return candidates[-1][1]
    return None


def model_config(run, stage):
    names = {f.name for f in fields(ContactConfig)}
    values = {k: v for k, v in run.items() if k in names}
    values.update(stage=stage, dtype="bfloat16" if run["precision"] == "bf16" else "float32")
    return ContactConfig(**values)


def training_command(config, output, stage, steps, resume=None):
    command = [sys.executable, "-m", "accelerate.commands.launch", "--num_processes", str(config["num_gpus"]),
               "--num_machines", "1", "--mixed_precision", config["precision"], "--dynamo_backend", "no"]
    if config["num_gpus"] > 1:
        command.append("--multi_gpu")
    command += ["--module", "contact_belief_wam.trainer", "--run-config", str(output / "run_config.yaml"),
                "--stage", stage, "--steps", str(steps)]
    if resume:
        command += ["--resume", str(resume)]
    return command


def final_export_ready(path):
    path = Path(path)
    has_weights = any(path.glob("model*.safetensors")) or any(path.glob("pytorch_model*.bin"))
    tokenizer = path / "tokenizer"
    return (has_weights and (path / "config.json").is_file() and (path / "quantiles.json").is_file()
            and (path / "deployment.json").is_file() and tokenizer.is_dir()
            and any(tokenizer.iterdir()))


def restore_final_export(config, output, stage2_ran=False):
    checkpoints = complete_checkpoints(output, "stage2")
    if not checkpoints:
        raise RuntimeError("Stage 2 has no complete checkpoint; cannot export a final model")
    checkpoint = checkpoints[-1][1]
    metadata = json.loads((checkpoint / "complete.json").read_text())
    if metadata["step"] < int(config["stage2_steps"]):
        raise RuntimeError("Stage 2 has not reached its configured final step")

    export = output / "export/final"
    state_path = output / "run_state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    # trainer writes run_state as stage2 when the complete checkpoint lands;
    # cloud changes it to complete only after the final export is verified.
    trusted_previous_export = state.get("stage") == "complete"
    if not final_export_ready(export) or not (stage2_ran or trusted_previous_export):
        # deployment.json is the exporter's last file. Removing it before a
        # retry prevents an interrupted overwrite from looking complete.
        deployment = export / "deployment.json"
        if deployment.is_file():
            deployment.unlink()
        command = training_command(config, output, "stage2", metadata["step"], checkpoint)
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True, env=os.environ.copy())

    if not final_export_ready(export):
        raise RuntimeError(f"Final export is incomplete: {export}")
    return checkpoint, metadata


def main(argv=None):
    config, dry_run = resolve(argv)
    output = Path(config["output"])
    use_world = config["mode"] in {"full", "wave0"}
    has_world = str(config["world_init"]).lower() != "none"
    if has_world:
        from .weights import checkpoint_file
        checkpoint_file(config["world_init"])
    stage1_steps = config["stage1_steps"]
    if stage1_steps == "auto":
        stage1_steps = config["stage1_pretrained_steps"] if has_world else config["stage1_scratch_steps"]
    if not use_world:
        stage1_steps = 0  # No world-model objective exists for pi05/direct_touch.
    config["resolved_stage1_steps"] = int(stage1_steps)
    for key in ("manifest", "stats"):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(config[key])
    resume = find_resume(config)
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)
        old_config = output / "run_config.yaml"
        if old_config.exists() and config["resume"] == "none":
            raise FileExistsError(f"Run exists; use --resume auto or a new output: {output}")
        old_config.write_text(yaml.safe_dump(config, sort_keys=False))
        shutil.copy2(config["manifest"], output / "data_manifest.jsonl")
        shutil.copy2(config["stats"], output / "quantiles.json")
    resume_meta = json.loads((resume / "complete.json").read_text()) if resume else {}
    stage2_ran = False
    for stage, steps in (("stage1", int(stage1_steps)), ("stage2", int(config["stage2_steps"]))):
        if steps <= 0 or (stage == "stage1" and resume_meta.get("stage") == "stage2"):
            continue
        stage_resume = resume if resume_meta.get("stage") == stage else None
        if stage_resume and resume_meta["step"] >= steps:
            continue
        command = training_command(config, output, stage, steps, stage_resume)
        print(" ".join(command), flush=True)
        if not dry_run:
            subprocess.run(command, check=True, env=os.environ.copy())
            stage2_ran = stage == "stage2"
    if not dry_run:
        state = {"stage": "complete", "export": None}
        if int(config["stage2_steps"]) > 0:
            checkpoint, metadata = restore_final_export(config, output, stage2_ran=stage2_ran)
            state.update(export="export/final", checkpoint=str(checkpoint), step=metadata["step"])
        atomic_json(output / "run_state.json", state)


if __name__ == "__main__":
    main()
