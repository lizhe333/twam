"""Native JAX policy server; simulation connects through the official wire protocol."""
from __future__ import annotations
from openpi.shared.univtac_runtime import configure, root
configure()
from pathlib import Path
import numpy as np


def load_model(config, checkpoint):
    import jax.numpy as jnp
    from openpi.models import model
    # The default upstream loader intersects trees. Disable that behavior so
    # an incompatible checkpoint cannot appear to have loaded successfully.
    params = model.restore_params(Path(checkpoint) / "params", dtype=jnp.bfloat16)
    return config.model.load(params, remove_extra_params=False)


def policy_for(model, config, checkpoint=None, shared_sampler=None):
    from openpi import transforms as T
    from openpi.policies.policy import Policy
    from openpi.shared import normalize
    if checkpoint is not None:
        from openpi.policies.univtac_camera import camera_mode_for_checkpoint
        if config.data.camera_mode != camera_mode_for_checkpoint(checkpoint, config.data.task):
            raise RuntimeError("Inference camera selection does not match the checkpoint")
    data = config.data.create(config.assets_dirs, config.model)
    stats = data.norm_stats if checkpoint is None else normalize.load(Path(checkpoint) / "assets" / data.asset_id)
    if stats is None:
        raise RuntimeError("Task-specific normalization is required")
    policy = Policy(model,
                    transforms=[*data.data_transforms.inputs, T.Normalize(stats, use_quantiles=True), *data.model_transforms.inputs],
                    output_transforms=[*data.model_transforms.outputs, T.Unnormalize(stats, use_quantiles=True), *data.data_transforms.outputs],
                    metadata=config.policy_metadata)
    if shared_sampler is not None:
        policy._sample_actions = shared_sampler
    return policy


class TaskPolicy:
    def __init__(self, tasks, *, checkpoint=None, mode="base", recipe=None):
        from openpi.training.univtac_config import make_config
        self.tasks = tuple(tasks)
        if checkpoint is not None and len(self.tasks) != 1:
            raise ValueError("Each trained checkpoint belongs to exactly one task")
        from openpi.policies.univtac_camera import camera_mode_for_checkpoint
        configs = {task: make_config(task, mode, recipe=recipe, camera_mode=camera_mode_for_checkpoint(checkpoint, task)
                                    if checkpoint is not None else None) for task in self.tasks}
        self.model = load_model(configs[self.tasks[0]], checkpoint or root() / "models/pi05_base")
        self.policies = {}
        sampler = None
        for task, config in configs.items():
            policy = policy_for(self.model, config, checkpoint, sampler)
            sampler = policy._sample_actions
            self.policies[task] = policy

    def infer(self, obs, *, noise=None):
        task = obs.get("task", self.tasks[0] if len(self.tasks) == 1 else None)
        if task not in self.policies:
            raise ValueError(f"Task must be one of {self.tasks}, got {task!r}")
        policy = self.policies[task]
        # Copy all mutable arrays: upstream input transforms operate in place.
        clean = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in obs.items() if key != "task"}
        return policy.infer(clean, noise=noise)


def serve(args):
    import jax
    from openpi.shared.univtac_constants import task_names
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    policy = TaskPolicy(task_names(args.task), checkpoint=args.checkpoint, mode=args.model_mode, recipe=getattr(args, "config", None))
    metadata = {"tasks": list(policy.tasks), "action_shape": [50, 8], "devices": [str(d) for d in jax.devices()],
                "model_mode": args.model_mode, "checkpoint": str(args.checkpoint or "official pi05_base"),
                "camera_modes": {task: p.metadata.get("camera_mode") for task, p in policy.policies.items()}}
    print(metadata, flush=True)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()
