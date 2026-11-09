"""Official NNX/JAX optimizer, FSDP and Orbax training with auditable acceptance metrics.

Model construction and checkpoint semantics come directly from the pinned upstream
trainer. The step below follows its loss/update/EMA sequence, adding group diagnostics.
"""
from __future__ import annotations

from openpi.shared.univtac_runtime import configure, root, atomic_json, project_root
configure()
import dataclasses
import functools
import importlib.util
import json
import logging
from pathlib import Path
import time

import flax.nnx as nnx
import flax.traverse_util as traverse
import jax
import jax.numpy as jnp
import numpy as np
import optax
from openpi.models import model as model_api
from openpi.training import checkpoints, data_loader, sharding

from openpi.training.univtac_config import make_config
from openpi.policies.univtac_camera import camera_contract


def upstream_trainer():
    spec = importlib.util.spec_from_file_location("pi05_pinned_train", project_root() / "scripts/train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parameter_group(path: str) -> str:
    if path.startswith("PaliGemma/img/"):
        return "vision"
    if path.startswith("PaliGemma/llm/") and not any(part.endswith("_1") for part in path.split("/")):
        return "language"
    return "action_expert"


def group_norms(tree):
    flat = traverse.flatten_dict(tree.to_pure_dict(), sep="/")
    groups = {key: [] for key in ("vision", "language", "action_expert")}
    for path, array in flat.items():
        groups[parameter_group(path)].append(array)
    return {key: optax.global_norm(value) if value else jnp.array(0.0) for key, value in groups.items()}


def train_step(config, rng, state, batch):
    model = nnx.merge(state.model_def, state.params)
    model.train()
    observation, actions = batch

    def loss_fn(current, loss_rng):
        return jnp.mean(current.compute_loss(loss_rng, observation, actions, train=True))

    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, config.trainable_filter))(
        model, jax.random.fold_in(rng, state.step))
    params = state.params.filter(config.trainable_filter)
    updates, opt_state = state.tx.update(grads, state.opt_state, params)
    updated = optax.apply_updates(params, updates)
    # Actual parameter displacement also detects updates rounded away by dtype.
    displacement = jax.tree.map(lambda new, old: new - old, updated, params)
    nnx.update(model, updated)
    all_params = nnx.state(model)
    result = dataclasses.replace(state, step=state.step + 1, params=all_params, opt_state=opt_state)
    if state.ema_decay is not None:
        result = dataclasses.replace(result, ema_params=jax.tree.map(
            lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
            state.ema_params, all_params))
    metrics = {"loss": loss, "grad_norm": optax.global_norm(grads),
               "update_norm": optax.global_norm(displacement)}
    for group, value in group_norms(grads).items():
        metrics[f"grad_{group}"] = value
    for group, value in group_norms(displacement).items():
        metrics[f"update_{group}"] = value
    return result, metrics


class FixedLoader:
    def __init__(self, config):
        self.config = config.data.create(config.assets_dirs, config.model)
        raw = data_loader.create_torch_dataset(self.config, config.model.action_horizon, config.model)
        transformed = data_loader.transform_dataset(raw, self.config)
        self.indices = np.random.default_rng(42).choice(len(raw), size=16, replace=False).tolist()
        self.samples = [transformed[index] for index in self.indices]

    def batch(self, index):
        sample = jax.tree.map(lambda value: np.asarray(value)[None], self.samples[index])
        return model_api.Observation.from_dict(sample), sample["actions"]

    def __iter__(self):
        rng = np.random.default_rng(42)
        while True:
            for index in rng.permutation(16):
                yield self.batch(int(index))

    def data_config(self):
        return self.config


def memory_stats() -> dict:
    result = {}
    for index, device in enumerate(jax.local_devices()):
        stats = device.memory_stats() or {}
        result[str(index)] = {key: int(stats[key]) for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit") if key in stats}
    return result


def run(task, mode, run_name, *, resume=False, steps=None, batch_size=None, recipe=None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = make_config(task, mode, run=run_name, resume=resume, steps=steps, batch_size=batch_size, recipe=recipe)
    jax.config.update("jax_compilation_cache_dir", str(root() / ".cache/jax"))
    if mode in ("full", "smoke") and (jax.device_count() != config.fsdp_devices or jax.process_count() != 1 or any(d.platform != "gpu" for d in jax.devices())):
        raise RuntimeError(f"Full/smoke requires exactly {config.fsdp_devices} visible GPUs in one JAX process")
    # Keep existing directories intact unless the caller explicitly resumes.
    if config.checkpoint_dir.exists() and not resume:
        raise FileExistsError(f"Run exists: {config.checkpoint_dir}; use --resume or a new --run")
    if resume:
        verify_resume_camera(config, task)
    manager, resuming = checkpoints.initialize_checkpoint_dir(config.checkpoint_dir, keep_period=None,
                                                               overwrite=False, resume=resume)
    atomic_json(config.checkpoint_dir / "univtac_config.json", {
        "task": task, "mode": mode, "run": run_name, "camera_mode": config.data.camera_mode,
        "batch_size": config.batch_size, "fsdp_devices": config.fsdp_devices,
        "num_train_steps": config.num_train_steps, "ema_decay": config.ema_decay, "seed": config.seed,
    })
    contract = camera_contract(config, task)
    atomic_json(config.checkpoint_dir / "camera_contract.json", contract)
    mesh = sharding.make_mesh(config.fsdp_devices)
    batch_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = FixedLoader(config) if mode == "overfit" else data_loader.create_data_loader(config, sharding=batch_sharding, shuffle=True)
    if loader.data_config().norm_stats is None:
        raise RuntimeError("Normalization missing; run prepare first")
    iterator = iter(loader)
    batch = next(iterator)
    rng, init_rng = jax.random.split(jax.random.key(config.seed))
    state, state_sharding = upstream_trainer().init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(state)
    if resuming:
        state = checkpoints.restore_state(manager, state, loader)
        if int(state.step) != manager.latest_step():
            raise ValueError("Restored optimizer step does not match the selected checkpoint")
        atomic_json(config.checkpoint_dir / "resume_verified.json", {
            "step": int(state.step), "state_and_optimizer": "restored by Orbax",
            "dataloader_cursor": "not restored by upstream; reshuffled deterministically",
        })
    start = int(state.step)
    if start > config.num_train_steps:
        raise ValueError(f"Target steps {config.num_train_steps} is below saved step {start}")
    pstep = jax.jit(functools.partial(train_step, config),
                   in_shardings=(replicated, state_sharding, batch_sharding),
                   out_shardings=(state_sharding, replicated), donate_argnums=(1,))
    fixed_report = None
    if mode == "overfit":
        @jax.jit
        def eval_step(params, fixed_batch):
            model = nnx.merge(state.model_def, params)
            model.eval()
            return jnp.mean(model.compute_loss(jax.random.key(4242), *fixed_batch, train=False))
        def evaluate():
            return float(np.mean([float(eval_step(state.params, loader.batch(i))) for i in range(16)]))
        previous = config.checkpoint_dir / "overfit_samples.json"
        if previous.exists():
            fixed_report = json.loads(previous.read_text())
            if fixed_report["indices"] != loader.indices:
                raise AssertionError("Fixed overfit samples changed")
        else:
            fixed_report = {"indices": loader.indices, "noise_seed": 4242, "before": evaluate()}
            atomic_json(previous, fixed_report)
    group_updates = {key: 0.0 for key in ("vision", "language", "action_expert")}
    report_path = config.checkpoint_dir / "training_report.json"
    if resuming and report_path.exists():
        group_updates.update(json.loads(report_path.read_text()).get("group_update_max", {}))
    # A process can stop after saving its final checkpoint but before publishing
    # final.json. Reconstruct diagnostics and finish that report on explicit resume.
    log_path = config.checkpoint_dir / "metrics.jsonl"
    if resuming and log_path.exists():
        for line in log_path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # A torn last log line is not a completed metric record.
            if record.get("step", start + 1) <= start:
                for key in group_updates:
                    group_updates[key] = max(group_updates[key], record.get(f"update_{key}", 0.0))
    begin = last_log = time.monotonic()
    accumulated = []
    step_durations = []
    previous_step_start = None
    try:
        for _ in range(start, config.num_train_steps):
            step_start = time.monotonic()
            if previous_step_start is not None and int(state.step) >= start + 5:
                step_durations.append(step_start - previous_step_start)
            previous_step_start = step_start
            with sharding.set_mesh(mesh):
                state, metrics = pstep(rng, state, batch)
            step = int(state.step)
            values = {key: float(value) for key, value in jax.device_get(metrics).items()}
            if not np.isfinite(list(values.values())).all():
                raise FloatingPointError(f"Nonfinite training metric at step {step}: {values}")
            if values["grad_norm"] <= 0 or values["update_norm"] <= 0:
                raise AssertionError(f"No trainable gradient/update at step {step}")
            for key in group_updates:
                group_updates[key] = max(group_updates[key], values[f"update_{key}"])
            accumulated.append(values)
            if step % config.log_interval == 0 or step == config.num_train_steps:
                now = time.monotonic()
                record = {key: float(np.mean([v[key] for v in accumulated])) for key in values}
                record.update(step=step, learning_rate=float(config.lr_schedule.create()(step - 1)),
                              examples_per_second=len(accumulated) * config.batch_size / (now - last_log),
                              memory=memory_stats())
                if step_durations:
                    record["steady_step_seconds_median"] = float(np.median(step_durations))
                    record["steady_examples_per_second"] = config.batch_size / record["steady_step_seconds_median"]
                with log_path.open("a") as f:
                    f.write(json.dumps(record, allow_nan=False) + "\n")
                logging.info("step=%d loss=%.6f grad=%.5f update=%.5f", step, values["loss"], values["grad_norm"], values["update_norm"])
                accumulated.clear()
                step_durations.clear()
                last_log = now
            if step % config.save_interval == 0 or step == config.num_train_steps:
                checkpoints.save_state(manager, state, loader, step)
                manager.wait_until_finished()
                atomic_json(config.checkpoint_dir / str(step) / "camera_contract.json", contract)
            if step < config.num_train_steps:
                batch = next(iterator)
        if fixed_report is not None:
            fixed_report.update(after=evaluate())
            fixed_report["relative_drop"] = 1 - fixed_report["after"] / fixed_report["before"]
            atomic_json(config.checkpoint_dir / "overfit_result.json", fixed_report)
        if mode == "smoke" and not all(value > 0 for value in group_updates.values()):
            raise AssertionError(f"Missing full-parameter updates: {group_updates}")
        passed = fixed_report is None or fixed_report["relative_drop"] >= 0.2
        report = {"passed": passed, "mode": mode, "task": task, "start_step": start, "end_step": int(state.step),
                  "devices": [str(d) for d in jax.devices()], "elapsed_seconds": time.monotonic() - begin,
                  "group_update_max": group_updates, "checkpoint": str(config.checkpoint_dir / str(int(state.step))),
                   "overfit": fixed_report, "resume_verified": resuming}
        atomic_json(config.checkpoint_dir / str(int(state.step)) / "camera_contract.json", contract)
        atomic_json(report_path, report)
        atomic_json(config.checkpoint_dir / "final.json", {"step": int(state.step), "relative_checkpoint": str(int(state.step)), "passed": passed})
        if not passed:
            raise AssertionError(f"Overfit fixed-noise loss decrease {fixed_report['relative_drop']:.1%} < 20%")
        return report
    finally:
        manager.wait_until_finished()
        manager.close()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--mode", choices=["lora", "overfit", "smoke", "full"], required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--steps", type=int)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    run(a.task, a.mode, a.run, resume=a.resume, steps=a.steps, recipe=a.config)


def verify_resume_camera(config, task):
    from openpi.policies.univtac_camera import camera_mode_for_checkpoint
    directory = config.checkpoint_dir
    steps = sorted((p for p in directory.glob("*") if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    if steps and camera_mode_for_checkpoint(steps[-1], task) != config.data.camera_mode:
        raise ValueError("Requested camera mode differs from the checkpoint")
    settings = directory / "univtac_config.json"
    if settings.exists():
        saved = json.loads(settings.read_text())
        if saved.get("ema_decay") != config.ema_decay:
            raise ValueError("EMA configuration differs from the resumed training state")


if __name__ == "__main__":
    main()
