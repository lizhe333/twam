#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset, make_univtac_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint as save_lerobot_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)

_STAGE1_FIXED_GRAPH_POLICY_TYPES = frozenset(
    {
        "wave0",
    }
)
_STAGE1_DIAGNOSTIC_POLICY_TYPES = frozenset(
    {
        "wave0",
    }
)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    rabc_weights_provider=None,
    policy_forward_kwargs: dict[str, Any] | None = None,
    materialize_output_dict: bool = True,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        rabc_weights_provider: Optional RABCWeights instance for sample weighting.
        policy_forward_kwargs: Optional policy-specific forward controls.
        materialize_output_dict: Collect diagnostics and convert logged tensors to Python values.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()
    unwrapped_policy = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    if has_method(unwrapped_policy, "set_train_logging_step"):
        unwrapped_policy.set_train_logging_step(materialize_output_dict)

    # Get RA-BC weights if enabled
    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = rabc_weights_provider.compute_batch_weights(batch)

    policy_forward_kwargs = policy_forward_kwargs or {}

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        # Use per-sample loss when RA-BC is enabled for proper weighting
        if rabc_batch_weights is not None:
            # Get per-sample losses
            per_sample_loss, output_dict = policy.forward(batch, reduction="none", **policy_forward_kwargs)

            # Apply RA-BC weights: L_RA-BC = Σ(w_i * l_i) / (Σw_i + ε)
            # rabc_batch_weights is already normalized to sum to batch_size
            epsilon = 1e-6
            loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + epsilon)
            # Log raw mean weight (before normalization) - this is the meaningful metric
            output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
            output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
            output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
        else:
            loss, output_dict = policy.forward(batch, **policy_forward_kwargs)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(unwrapped_policy, "update"):
        unwrapped_policy.update()

    if materialize_output_dict:
        output_dict = {
            key: (
                value.detach().item()
                if isinstance(value, torch.Tensor) and value.numel() == 1
                else value.detach().cpu().tolist()
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in output_dict.items()
        }
    else:
        output_dict = {}

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # Most policies retain find_unused_parameters=True for conditional computation.
    # The listed Stage-1 WAM policies have fixed trainable graphs and use every
    # trainable parameter on every iteration, so the extra traversal is unnecessary.
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        fixed_graph = (
            cfg.policy.type in _STAGE1_FIXED_GRAPH_POLICY_TYPES
            and getattr(cfg.policy, "world_model_stage", None) == "stage1"
        )
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=not fixed_graph,
            gradient_as_bucket_view=fixed_graph,
            static_graph=fixed_graph,
        )
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when policy.device is set to CPU.
        force_cpu = cfg.policy.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    # All ranks must finish cfg.validate() before rank 0 initializes W&B,
    # which may create output_dir. Process-group initialization alone is not
    # a barrier: a slower rank could otherwise mistake this run's directory
    # for a previous run and fail the resume=False overwrite guard.
    accelerator.wait_for_everyone()

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
        
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        # Convert CLI peft config to dict for overrides
        peft_cli_overrides = dataclasses.asdict(cfg.peft)
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    # For SARM, always provide dataset_meta for progress normalization
    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Load precomputed SARM progress for RA-BC if enabled
    # Generate progress using: src/lerobot/policies/sarm/compute_rabc_weights.py
    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights

        # Get chunk_size from policy config
        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")

        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info(f"Loading SARM progress for RA-BC from {cfg.rabc_progress_path}")
        logging.info(f"Using chunk_size={chunk_size} from policy config, head_mode={head_mode}")
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if has_method(cfg.policy, "make_training_sampler"):
        sampler = cfg.policy.make_training_sampler(dataset, seed=cfg.seed)
        shuffle = sampler is None
    elif hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=cfg.dataloader_prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.dataloader_persistent_workers and cfg.num_workers > 0,
        # Default ordered delivery preserves sampler order across worker timing.
    )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Keep global batch size for logging; MetricsTracker handles world size internally.
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        materialize_output_dict = cfg.log_freq > 0 and (step + 1) % cfg.log_freq == 0 and is_main_process
        policy_forward_kwargs = None
        if (
            cfg.policy.type in _STAGE1_DIAGNOSTIC_POLICY_TYPES
            and getattr(cfg.policy, "world_model_stage", None) == "stage1"
        ):
            # Stage-1 diagnostics do not contribute to the loss. Compute them only
            # for a batch that will actually be emitted by the logger.
            policy_forward_kwargs = {
                "compute_diagnostics": cfg.log_freq > 0 and (step + 1) % cfg.log_freq == 0
            }

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            rabc_weights_provider=rabc_weights,
            policy_forward_kwargs=policy_forward_kwargs,
            materialize_output_dict=materialize_output_dict,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        if is_main_process:
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if output_dict:
                scalar_output = {
                    key: value for key, value in output_dict.items() if isinstance(value, (int, float, bool))
                }
                if scalar_output:
                    logging.info("loss components: %s", pformat(scalar_output, compact=True))
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                # Log RA-BC statistics if enabled
                if rabc_weights is not None:
                    rabc_stats = rabc_weights.get_stats()
                    wandb_log_dict.update(
                        {
                            "rabc_delta_mean": rabc_stats["delta_mean"],
                            "rabc_delta_std": rabc_stats["delta_std"],
                            "rabc_num_frames": rabc_stats["num_frames"],
                        }
                    )
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_lerobot_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if cfg.policy.use_peft:
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()



# UniVTAC keeps its validated DDP/accumulation/checkpoint protocol. The LeRobot
# dataset route above retains its Accelerate protocol; neither changes defaults.
_UNIVTAC_READY = False
_UNIVTAC_EXPORTS = {'check_shared_teacher', 'prune_checkpoints', 'create_policy', 'group_name', 'UniVTACDataset', 'np', 'save_checkpoint', 'profile_due', 'ddp_find_unused_parameters', 'object_digest', 'atomic_json', 'is_sharded', 'DataLoader', 'gradient_report', 'load_local_state', 'EMA', 'dist', 'resume_identity', 'lr_at', 'select_ema_device', 'assert_finite', 'set_lr', 'identity', 'witnesses', 'probe', 'fcntl', 'make_ema', 'performance_config', 'DistributedDataParallel', 'json', 'Path', 'digest', 'random', 'sync_if_cuda', 'latest', 'ReplaySampler', 'make_optimizer', 'SHARD_FORMAT', 'tensor_witness', 'restore_rng', 'os', 'validate_world'}


def _load_univtac_support():
    """Keep optional HDF5/UniVTAC dependencies out of the default training path."""
    global _UNIVTAC_READY
    if _UNIVTAC_READY:
        return
    import fcntl
    import json
    import os
    from pathlib import Path
    import random
    import numpy as np
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader
    from lerobot.datasets.univtac_dataset import UniVTACDataset, ReplaySampler
    from lerobot.policies.wave0.loading_wave0 import create_policy, group_name, gradient_report
    from lerobot.optim.ema import EMA, make_ema
    from lerobot.optim.univtac import (lr_at, make_optimizer, set_lr, performance_config,
                                      select_ema_device, ddp_find_unused_parameters)
    from lerobot.optim.optimizer_shards import FORMAT as SHARD_FORMAT, is_sharded, load_local_state
    from lerobot.utils.univtac_runtime import atomic_json, digest, object_digest
    from lerobot.utils.univtac_train_utils import (
        sync_if_cuda, profile_due, assert_finite, identity, latest, check_shared_teacher,
        validate_world, restore_rng, witnesses, tensor_witness, probe, save_checkpoint,
        prune_checkpoints, resume_identity,
    )
    globals().update({name: value for name, value in locals().items() if name in _UNIVTAC_EXPORTS})
    _UNIVTAC_READY = True


def __getattr__(name):
    if name in _UNIVTAC_EXPORTS:
        _load_univtac_support()
        return globals()[name]
    raise AttributeError(name)


def train_univtac(run, stage, *, debug=False, updates=1, resume=False, device="cuda", threads=8, verify_only=False,
          debug_cpu_offload=False):
    _load_univtac_support()
    if debug and not 1 <= updates <= 2:
        raise ValueError("Local debug is limited to 1–2 optimizer updates per stage")
    if verify_only and not resume:
        raise ValueError("verify-only requires --resume")
    if not debug and (device != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("Formal training requires CUDA; use debug for bounded local CPU checks")
    performance = performance_config(run)
    log_every = int(performance.get("log_every", run.get("log_every", 100)))
    if log_every <= 0:
        raise ValueError("log_every must be a positive integer")
    profile_updates = bool(performance.get("profile_updates", False))
    torch.set_num_threads(threads)
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if debug_cpu_offload:
        if not debug or device != "cuda" or world != 1:
            raise ValueError("CPU offload is only supported for single-GPU CUDA debug")
        run = {**run, "debug_cpu_offload": True}
    cpu_offload = debug_cpu_offload and stage == "stage2"
    if not debug and world != run["num_gpus"]:
        raise ValueError("Launch with the configured number of torchrun processes")
    if device == "cuda":
        torch.cuda.set_device(local_rank)
        dev = torch.device("cuda", local_rank)
    else:
        dev = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl" if dev.type=="cuda" else "gloo")
    out = Path(run["output"])
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/f".{stage}.lock").open("w") if rank==0 else None
    if lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    shared_stage1 = run.get("shared_wm") and stage=="stage1"
    if run.get("shared_wm") and ((stage=="stage1") != (run["task"]=="shared_wm")):
        raise ValueError("Shared Stage 1 uses all tasks; Stage 2 requires --task")
    stage1 = check_shared_teacher(run, debug) if run.get("shared_wm") and stage=="stage2" else None
    ident = identity(run, debug)
    checkpoint = latest(out, stage) if resume else None
    if not resume and (out/stage).exists():
        raise FileExistsError("Stage output exists; use --resume or a fresh --output")
    attempt = out/stage/"run_identity.json"
    integrated_ident = ident
    migration_path = out/stage/"source_migration.json"
    if attempt.exists():
        ident = resume_identity(ident, json.loads(attempt.read_text()), migration_path)
    if stage=="stage2" and not run.get("shared_wm"):
        stage1 = latest(out, "stage1")
        if stage1 is None:
            raise ValueError("Complete Stage 1 first")
        meta = json.loads((stage1/"complete.json").read_text())
        if not debug and meta["step"] != run["stage1"]["steps"]:
            raise ValueError("Stage 1 has not completed its configured training budget")
        teacher_ident = json.loads((stage1/"run_identity.json").read_text())
        if not attempt.exists():
            # A legacy Stage 1 may be followed by its first Stage 2 here.
            ident = resume_identity(ident, teacher_ident, migration_path)
        if teacher_ident != ident:
            raise ValueError("Stage 1 and Stage 2 must share the same recipe identity")
    # Every rank must finish checking the old directory state before rank 0
    # creates it. Otherwise a late rank rejects this launch's own output.
    if world > 1:
        dist.barrier()
    if rank==0:
        atomic_json(attempt, ident)
        if ident != integrated_ident:
            atomic_json(migration_path, {"legacy_fingerprint": ident["fingerprint"],
                                         "integrated_identity": integrated_ident})
    if checkpoint:
        saved = json.loads((checkpoint/"run_identity.json").read_text())
        if ident != saved:
            raise ValueError("Recipe/code/data identity changed; use a new output directory")
    result_path = out/stage/"result.json"
    if resume and not verify_only and checkpoint and result_path.exists():
        old_result = json.loads(result_path.read_text())
        if (old_result.get("identity")==ident["fingerprint"] and old_result.get("complete")
                and (not debug or old_result["step"]>=updates)):
            if world>1:
                dist.destroy_process_group()
            return {**old_result, "skipped_completed": True}
    torch.manual_seed(run["seed"])
    np.random.seed(run["seed"])
    random.seed(run["seed"])
    started = time.monotonic()
    print(json.dumps({"event": "loading", "task": run["task"], "stage": stage, "device": str(dev), "debug": debug}), flush=True)
    policy, loading = create_policy(run, stage, stage1/"pretrained_model" if stage1 else None,
                                    checkpoint/"pretrained_model" if checkpoint else None)
    policy.to("cpu" if cpu_offload else dev)
    optimizer = make_optimizer(policy, run, stage)
    ema_device = select_ema_device(run, dev, cpu_offload=cpu_offload)
    ema = make_ema(policy, run, stage, rank, ema_device, initialize=checkpoint is None)
    print(json.dumps({"event": "ema_configuration", "rank": rank, "stage": stage,
                      "enabled": bool(stage == "stage2" and run.get("ema_enabled", True)),
                      "shadow_allocated": ema is not None}), flush=True)
    if is_sharded(optimizer):
        local_numel = sum(p.numel() for g in optimizer.optim.param_groups for p in g["params"])
        print(json.dumps({"event": "gpu_optimizer_layout", "rank": rank,
                          "optimizer": "ZeroRedundancyOptimizer/AdamW",
                          "estimated_local_fp32_moments_gib": local_numel*8/1024**3,
                          "ema_device": str(ema_device) if ema else None,
                          "ema_owner_rank": 0, "cpu_optimizer_offload": False}), flush=True)
    start = 0
    saved_rng = None
    restore_audit = None
    if checkpoint:
        state = torch.load(checkpoint/"training.pt", map_location="cpu", weights_only=False)
        if state["world_size"] != world:
            raise ValueError("Resume requires the same world size")
        if is_sharded(optimizer):
            if state.get("optimizer_format") != SHARD_FORMAT:
                raise ValueError("Expected a checkpoint with sharded optimizer states")
            load_local_state(optimizer, policy, checkpoint, rank, world)
        else:
            optimizer.load_state_dict(state["optimizer"])
        if ema:
            ema.load(checkpoint/"ema.safetensors", policy)
        start = state["step"]
        saved_rng = state["rng"][rank]
        if debug and rank==0:
            old = json.loads((checkpoint/"witnesses.json").read_text())
            if witnesses(policy, optimizer, ema) != old:
                raise AssertionError("Parameter/optimizer/EMA restore witness differs")
            restore_audit = {"parameters_optimizer_ema_restored": True}
        del state  # Do not retain a second CPU copy of every optimizer slot on GPU resume.
    if world > 1:
        wrapped = DistributedDataParallel(policy, device_ids=[local_rank] if dev.type=="cuda" else None,
                                          find_unused_parameters=ddp_find_unused_parameters(run),
                                          gradient_as_bucket_view=True,
                                          broadcast_buffers=False)
    else:
        wrapped = policy
    if shared_stage1:
        from lerobot.datasets.univtac_shared import SharedDataset, BalancedReplaySampler
        dataset = make_univtac_dataset(run, stage)
    else:
        dataset = make_univtac_dataset(run, stage)
    from torch.utils.data._utils.collate import default_collate
    diagnostic = default_collate([child[i] for child in dataset.datasets for i in (0, len(child)//2)]) if shared_stage1 else default_collate([dataset[0], dataset[len(dataset)//2]])
    diagnostic = {k:v.to(dev) if torch.is_tensor(v) else v for k,v in diagnostic.items()}
    if checkpoint and debug:
        restored_probe = probe(policy, diagnostic, dev, cpu_offload=cpu_offload)
        old_probe = json.loads((checkpoint/"audit.json").read_text())["probe"]
        difference = abs(restored_probe["loss"]-old_probe["loss"])
        if difference > 1e-6:
            raise AssertionError(f"Checkpoint forward changed: {difference}")
        if rank==0:
            restore_audit["probe_max_abs_difference"] = difference
    if verify_only:
        if not checkpoint or not debug:
            raise ValueError("verify-only requires debug resume")
        if rank==0:
            atomic_json(out/stage/"restore_report.json", restore_audit)
        dataset.close()
        if world>1:
            dist.destroy_process_group()
        return restore_audit
    # Never carry open h5py handles across DataLoader worker forks.
    dataset.close()
    limit = updates if debug else run[stage]["steps"]
    global_batch = (16 if shared_stage1 else 2)*world if debug else run[stage].get("global_batch", run["global_batch"])
    micro = (16 if shared_stage1 else 2) if debug and stage=="stage1" else 1 if debug else run[stage]["micro_batch"]
    if global_batch % (micro*world):
        raise ValueError("Global batch must divide micro_batch * world_size")
    accumulation = global_batch//(micro*world)
    sampler = BalancedReplaySampler(dataset.sizes, run["seed"], global_batch, world, rank, micro, start, limit) if shared_stage1 else ReplaySampler(len(dataset), run["seed"], global_batch, world, rank, start, limit)
    loader = DataLoader(dataset, batch_size=micro, sampler=sampler, num_workers=0 if debug else run["workers"],
                        generator=torch.Generator().manual_seed(run["seed"]+rank), pin_memory=dev.type=="cuda")
    policy.train()
    if saved_rng:
        restore_rng(saved_rng)
    else:
        torch.manual_seed(run["seed"]+rank)
    optimizer.zero_grad(set_to_none=True)
    audit = {"loading": loading, "restore": restore_audit, "updates": [], "debug": debug}
    selection = {"best_loss": None, "best_checkpoint": None, "stale": 0, "stopped_early": False}
    if checkpoint and shared_stage1:
        selection = json.loads((checkpoint/"audit.json").read_text()).get("selection", selection)
    if shared_stage1:
        audit["training_pairs_per_task"] = dict(zip(run["tasks"], dataset.sizes))
    step, micro_index, loss_sum = start, 0, None
    last_log_time, last_log_step = time.monotonic(), start
    iterator = iter(loader)
    while step < limit and not selection["stopped_early"]:
        next_step = step + 1
        log_due = debug or next_step % log_every == 0 or next_step == limit
        profile_update = profile_due(profile_updates, next_step, log_every, limit)
        if profile_update:
            # Drain prior asynchronous work before measuring this sampled update.
            sync_if_cuda(dev)
        update_started = time.monotonic()
        phase_seconds = {"data_h2d": 0., "forward_backward": 0., "optimizer": 0., "ema": 0.} \
            if profile_update else None
        set_lr(optimizer, run[stage], step)
        before = {n:tensor_witness(p) for n,p in policy.named_parameters() if p.requires_grad} if debug else None
        for micro_index in range(accumulation):
            phase_started = time.monotonic() if profile_update else None
            raw = next(iterator)
            if shared_stage1:
                from collections import Counter
                if min(Counter(raw["world_model_task"]).values()) < 2 or set(raw["world_model_task"]) != set(run["tasks"]):
                    raise AssertionError("Shared WM physical batch lost task balance or same-task negatives")
            batch = {k:v.to(dev) if torch.is_tensor(v) else v for k,v in raw.items()}
            if profile_update:
                sync_if_cuda(dev)
                phase_seconds["data_h2d"] += time.monotonic() - phase_started
            phase_started = time.monotonic() if profile_update else None
            if stage=="stage2":
                # Upstream increments per forward. Pin to optimizer updates so
                # gradient accumulation does not accelerate WAM schedules.
                policy.model.stage2_step.fill_(step)
            sync = nullcontext() if world==1 or micro_index==accumulation-1 else wrapped.no_sync()
            with sync, torch.autocast(dev.type, dtype=torch.bfloat16, cache_enabled=not cpu_offload):
                if cpu_offload:
                    from lerobot.utils.debug_offload import offloaded_forward
                    loss, metrics = offloaded_forward(policy, batch, dev, backward_scale=accumulation,
                                                      execution_dtype=torch.bfloat16)
                else:
                    loss, metrics = wrapped(batch)
                    assert_finite(loss, "Nonfinite training loss")
                    (loss/accumulation).backward()
            loss_part = loss.detach()/accumulation
            loss_sum = loss_part if loss_sum is None else loss_sum + loss_part
            if profile_update:
                sync_if_cuda(dev)
                phase_seconds["forward_backward"] += time.monotonic() - phase_started
        phase_started = time.monotonic() if profile_update else None
        gradients = gradient_report(policy) if debug else None
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), run["grad_clip"])
        assert_finite(grad_norm, "Nonfinite gradient norm")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if stage=="stage2":
            policy.model.stage2_step.fill_(step)
        if profile_update:
            sync_if_cuda(dev)
            phase_seconds["optimizer"] += time.monotonic() - phase_started
        phase_started = time.monotonic() if profile_update else None
        if ema:
            ema.update(policy)
        if profile_update:
            sync_if_cuda(dev)
            phase_seconds["ema"] += time.monotonic() - phase_started
        needs_row_values = debug or log_due
        loss_value = float(loss_sum) if needs_row_values else None
        grad_norm_value = float(grad_norm) if needs_row_values else None
        update_seconds = time.monotonic() - update_started
        interval_seconds = time.monotonic() - last_log_time if log_due else None
        interval_steps = step - last_log_step if log_due else None
        step_per_second = interval_steps / interval_seconds if interval_seconds and interval_seconds > 0 else None
        row = {"step": step, "loss": loss_value, "grad_norm": grad_norm_value,
               "lr": {g["name"]:g["lr"] for g in optimizer.param_groups}, "metrics": metrics,
               "update_seconds": update_seconds,
               "step_per_second": step_per_second,
               "global_samples_per_second": (global_batch*interval_steps/interval_seconds
                                              if interval_seconds and interval_seconds > 0 else None),
               "phase_seconds": phase_seconds,
               "timing_sampled": profile_update,
               "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(dev)/1024**3 if dev.type=="cuda" else None,
               "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(dev)/1024**3 if dev.type=="cuda" else None}
        if debug:
            row["debug_cpu_offload"] = cpu_offload
            row["gradients"] = gradients
            row["changed_groups"] = sorted({group_name(n) for n,p in policy.named_parameters()
                                             if p.requires_grad and tensor_witness(p)!=before[n]})
            required = {"world"} if stage=="stage1" else {"policy_vision", "language", "action_expert", "future_condition"}
            if not required.issubset(row["changed_groups"]):
                raise AssertionError(f"No update in required groups: {required-set(row['changed_groups'])}")
            if any(g["missing"] for g in gradients.values()):
                raise AssertionError(f"Trainable tensors without gradient: {gradients}")
            audit["updates"].append(row)
        if rank==0 and log_due:
            log = out/stage/"train.jsonl"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as handle:
                handle.write(json.dumps(row)+"\n")
            print(json.dumps({"event":"update", "task":run["task"], "stage":stage, "step":step,
                              "loss":loss_value, "step_per_second":step_per_second,
                              "phase_seconds":phase_seconds,
                              "cuda_peak_allocated_gib":row["cuda_peak_allocated_gib"],
                              "cuda_peak_reserved_gib":row["cuda_peak_reserved_gib"],
                              "seconds":time.monotonic()-started}), flush=True)
        if log_due:
            last_log_time, last_log_step = time.monotonic(), step
        loss_sum = None
        validation_due = shared_stage1 and (debug or step%run[stage]["validate_every"]==0 or step==limit)
        if validation_due:
            validation = validate_world(policy, run, dev, world, rank, debug)
            validation["step"] = step
            if rank==0:
                with (out/stage/"validation.jsonl").open("a") as handle:
                    handle.write(json.dumps(validation)+"\n")
                print(json.dumps({"event": "validation", **validation}), flush=True)
            if selection["best_loss"] is None or validation["loss"] < selection["best_loss"]:
                selection.update(best_loss=validation["loss"], stale=0,
                                 best_checkpoint=str(out/stage/"checkpoints"/f"step_{step:08d}"))
            else:
                selection["stale"] += 1
            selection["stopped_early"] = bool(not debug and step>=run[stage]["early_stop_min_steps"]
                                              and selection["stale"]>=run[stage]["early_stop_patience"])
            audit["validation"] = validation
            audit["selection"] = selection.copy()
        if debug or validation_due or step%run["save_every"]==0 or step==limit:
            if debug:
                audit["probe"] = probe(policy, diagnostic, dev, cpu_offload=cpu_offload)
            checkpoint = save_checkpoint(policy, optimizer, ema, run, stage, step, ident, world, rank, audit)
            if rank==0 and not debug:
                prune_checkpoints(run, stage, selection["best_checkpoint"])
    dataset.close()
    result = {"task":run["task"], "stage":stage, "step":step, "checkpoint":str(checkpoint),
              "formal_training":not debug,"seconds":time.monotonic()-started,
              "world_size":world,"global_batch":global_batch, "micro_batch":micro,
              "accumulation":accumulation, "complete":True}
    result["identity"] = ident["fingerprint"]
    if shared_stage1:
        result.update(selected_checkpoint=selection["best_checkpoint"], selection=selection)
    if rank==0:
        atomic_json(out/stage/"result.json", result)
    if world>1:
        dist.destroy_process_group()
    return result

def main():
    import sys
    if "--univtac" in sys.argv:
        sys.argv.remove("--univtac")
        from lerobot.scripts.univtac_cli import main as univtac_main
        return univtac_main()
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
