"""UniVTAC policy adapter; run only with scripts/sim.sh, never import the policy model here."""
from __future__ import annotations

import argparse
from collections import deque
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from .paths import WORKSPACE, atomic_json
from .protocol import Client


def cpu(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def robot_action_bounds(task):
    manager = task._robot_manager
    limits = cpu(manager.robot.data.joint_pos_limits)[0, cpu(manager._arm_ids)]
    bounds = np.empty((8, 2), dtype=np.float32)
    bounds[:7] = limits
    bounds[7] = (0, manager.gripper_max_qpos)
    if bounds.shape != (8, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError(f"Invalid robot joint bounds: {bounds}")
    return bounds


class Adapter:
    def __init__(self, client, episode, prompt, action_bounds=None):
        self.client, self.episode, self.prompt = client, episode, prompt
        self.k = client.metadata["k"]
        self.action_bounds = None if action_bounds is None else np.asarray(action_bounds, dtype=np.float32)
        if self.action_bounds is not None and (self.action_bounds.shape != (8, 2) or
                                               not np.isfinite(self.action_bounds).all() or
                                               np.any(self.action_bounds[:, 0] > self.action_bounds[:, 1])):
            raise ValueError("Expected finite robot action bounds [8,2] with lower <= upper")
        self.touch = deque(maxlen=7)
        self.executed, self.actions, self.updates = [], [], []
        self.chunk, self.position, self.step = None, 0, 0
        self.client.call(op="reset", episode=episode, action_bounds=self.action_bounds)

    def observe(self, observation):
        self.touch.append(np.stack([cpu(observation["tactile"][f"{side}_tactile"]["rgb_marker"]).copy()
                                    for side in ("left", "right")]))

    def next_action(self, task, observation):
        if not self.touch:
            self.observe(observation)
        if self.chunk is None or self.position == self.k:
            frames = list(self.touch)
            window = np.stack([frames[max(0, len(frames) - 1 + offset)] for offset in (-6, -4, -2, 0)], axis=1)
            result = self.client.call(op="infer", episode=self.episode, step=self.step,
                                      image=cpu(observation["observation"]["head"]["rgb"]),
                                      state=cpu(observation["embodiment"]["joint"])[:8],
                                      tactile_window=window, executed_actions=np.asarray(self.executed, np.float32).reshape(-1, 8),
                                      prompt=self.prompt)
            self.chunk = np.asarray(result["actions"], np.float32)
            if self.chunk.shape != (50, 8) or not np.isfinite(self.chunk).all():
                raise ValueError("Expected finite 50 × 8 policy chunk")
            reported_overshoots = result.get("arm_limit_overshoots")
            self.chunk_arm_limit_overshoots = None
            if reported_overshoots is not None:
                self.chunk_arm_limit_overshoots = np.asarray(reported_overshoots, np.float32)
                if (self.chunk_arm_limit_overshoots.shape != (50,) or
                        not np.isfinite(self.chunk_arm_limit_overshoots).all()):
                    raise ValueError("Expected finite arm-limit overshoots for the 50-action chunk")
            elif self.action_bounds is not None:
                raise ValueError("Bounded policy service did not report the original arm-limit overshoots")
            self.position, self.executed = 0, []
            self.updates.append(result["diagnostics"])
        action = self.chunk[self.position].copy()
        bounds = self.action_bounds if self.action_bounds is not None else robot_action_bounds(task)
        bounded = np.clip(action, bounds[:, 0], bounds[:, 1])
        overshoot = float(np.abs(bounded[:7] - action[:7]).max())
        if self.chunk_arm_limit_overshoots is not None:
            overshoot = max(overshoot, float(self.chunk_arm_limit_overshoots[self.position]))
        if overshoot > .02:
            raise ValueError(f"Policy joint target outside limits by {overshoot:.6f} rad")
        action[:] = bounded
        self.position += 1
        return action

    def execute(self, task, action):
        import torch
        value = torch.as_tensor(action, dtype=torch.float32, device=task._robot_manager.device)
        before = task.step_count
        success = False
        for _ in range(2):
            okay, success = task.take_action(value, action_type="qpos", force=True)
            if not okay or not task.plan_success:
                raise RuntimeError("Simulator rejected a policy action")
            if success:
                break
        advanced = task.step_count - before
        if advanced != 2 and not (advanced == 1 and success):
            raise RuntimeError(f"Expected two physical steps, got {advanced}")
        self.executed.append(action.copy())
        self.actions.append({"step": self.step, "physical_step": task.step_count, "action": action.tolist()})
        self.step += 1
        obs = task._get_observations()
        self.observe(obs)  # Every strategy step, including the k-1 steps without inference.
        return bool(success), obs

    def close(self):
        self.client.call(op="close", episode=self.episode)


def plot_belief(path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    data = json.loads(Path(path).read_text())
    if not data["updates"]:
        return
    updates = data["updates"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    steps = [u["step"] for u in updates]
    norms = np.asarray([u["belief_norm"] for u in updates])
    for i in range(norms.shape[1]):
        axes[0].plot(steps, norms[:, i], label=f"belief {i}")
    axes[0].set_ylabel("token L2 norm")
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].set_ylim(0, max(1., float(norms.max()) * 1.1))
    axes[0].legend()
    axes[1].plot(steps, [u["contact_prediction_error"] if u["contact_prediction_error"] is not None else np.nan for u in updates])
    axes[1].set_ylabel("contact cosine error")
    axes[2].plot(steps, [u["inference_seconds"] for u in updates])
    axes[2].set_ylabel("inference seconds")
    axes[2].set_xlabel("episode strategy step")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(Path(path).with_suffix(".png"), dpi=140)
    plt.close(fig)


def main(argv=None):
    os.chdir(WORKSPACE / "UniVTAC")
    sys.path.insert(0, str(WORKSPACE / "UniVTAC"))
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["insert_hole", "insert_HDMI"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", default="1000000,1000001,1000002")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18780)
    p.add_argument("--max-policy-steps", type=int)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(argv)
    args.headless, args.enable_cameras, args.livestream = True, True, 0
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app
    task = client = None
    fatal = None
    try:
        import yaml
        module = importlib.import_module(f"envs.{args.task}")
        cfg = module.TaskCfg()
        cfg.save_dir, cfg.decimation, cfg.video_frequency = args.output, 1, 2
        cfg.save_frequency, cfg.render_frequency = 2, 0
        cfg.tactile_sensor_type = "gsmini"
        cfg.obs_data_type = yaml.safe_load((WORKSPACE / "UniVTAC/task_config/clean.yml").read_text())["observations"]
        cfg.reset_time_limit = 300
        cfg.sim.device = args.device or "cuda:0"
        budget = args.max_policy_steps or {"insert_hole": 317, "insert_HDMI": 600}[args.task]
        cfg.step_lim = 2 * budget
        task = module.Task(cfg, mode="eval")
        client = Client(args.host, args.port)
        prompt = json.loads((WORKSPACE / "UniVTAC/instructions" / f"{args.task}.json").read_text())["seen"][0]
        records = []
        for seed in map(int, args.seeds.split(",")):
            begin, adapter = time.monotonic(), None
            result = {"seed": seed, "task": args.task, "status": "error", "complete_rollout": False,
                      "policy_step_budget": budget, "simulation_device": cfg.sim.device, **client.metadata}
            try:
                task.reset(seed=seed, instructions=[prompt])
                if not task.plan_success:
                    raise RuntimeError("Native pre_move failed")
                adapter = Adapter(client, f"{args.task}:{seed}", prompt, robot_action_bounds(task))
                obs = task._get_observations()
                for _ in range(budget):
                    action = adapter.next_action(task, obs)
                    success, obs = adapter.execute(task, action)
                    if success:
                        result["status"] = "success"
                        break
                    if task.check_early_stop():
                        result["status"] = "failed"
                        break
                else:
                    result["status"] = "timeout"
                result["complete_rollout"] = True
            except Exception as exc:
                result.update(error=str(exc), traceback=traceback.format_exc())
            finally:
                if adapter:
                    result.update(actions=adapter.actions, updates=adapter.updates, policy_steps=adapter.step)
                    adapter.close()
                else:
                    result.update(actions=[], updates=[], policy_steps=0)
                task.clean_cache(result=result["status"])
                result["elapsed_seconds"] = time.monotonic() - begin
                path = args.output / "episodes" / f"{seed}.json"
                # Convert NumPy diagnostics without discarding values.
                result = json.loads(json.dumps(result, default=lambda a: a.tolist()))
                atomic_json(path, result)
                plot_belief(path)
                records.append({k: v for k, v in result.items() if k not in {"actions", "updates"}})
                atomic_json(args.output / "evaluation.json", {"episodes": records,
                            "complete_rollouts": sum(r["complete_rollout"] for r in records)})
                print(json.dumps(records[-1]), flush=True)
        if not all(r["complete_rollout"] for r in records):
            raise RuntimeError("Runtime errors occurred; these episodes do not count as completed smoke rollouts")
    except Exception:
        fatal = traceback.format_exc()
        atomic_json(args.output / "fatal.json", {"complete_rollout": False, "traceback": fatal})
        print(fatal, flush=True)
    finally:
        if client:
            client.close()
        if task:
            task.close()
        app.close()
    if fatal:
        raise RuntimeError(fatal)


if __name__ == "__main__":
    main()
