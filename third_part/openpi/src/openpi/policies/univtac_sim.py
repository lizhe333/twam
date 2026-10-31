"""Simulator-side code: no JAX, OpenPI model, or LeRobot imports."""
from __future__ import annotations
import inspect
import numpy as np
from openpi.shared.univtac_constants import EXECUTE_STEPS, HORIZON, prompt


class InvalidPolicyOutput(ValueError):
    policy_output_error = True


class FiniteWebsocketClient:
    def __init__(self, host="127.0.0.1", port=8000, timeout=600):
        from openpi_client import msgpack_numpy
        import websockets.sync.client
        import websockets.exceptions

        self._codec = msgpack_numpy
        self._packer = msgpack_numpy.Packer()
        self.timeout = timeout
        self._communication_errors = (
            OSError,
            websockets.exceptions.ConnectionClosed,
            websockets.exceptions.InvalidHandshake,
            websockets.exceptions.InvalidMessage,
        )
        connect_options = dict(compression=None, max_size=None, open_timeout=30)
        # Isaac Sim 5.1 pins websockets 12, whose synchronous client has
        # neither proxy discovery nor keepalive. Both options arrived later.
        parameters = inspect.signature(websockets.sync.client.connect).parameters
        for option in ('proxy', 'ping_interval'):
            if option in parameters:
                connect_options[option] = None
        self._ws = self._transport_call(
            "connect",
            lambda: websockets.sync.client.connect(
                f"ws://{host}:{port}", **connect_options,
            ),
        )
        self.metadata = self._codec.unpackb(self._receive())

    def _transport_call(self, operation, call):
        try:
            return call()
        except TimeoutError as exc:
            raise TimeoutError(f"Policy websocket {operation} timed out: {exc}") from exc
        except self._communication_errors as exc:
            raise ConnectionError(f"Policy websocket {operation} failed: {exc}") from exc

    def _receive(self):
        response = self._transport_call("receive", lambda: self._ws.recv(timeout=self.timeout))
        if isinstance(response, str):
            raise ConnectionError(f"Policy server reported an execution error: {response}")
        return response

    def infer(self, observation):
        self._transport_call("send", lambda: self._ws.send(self._packer.pack(observation)))
        response = self._receive()
        return self._codec.unpackb(response)

    def close(self):
        self._transport_call("close", self._ws.close)


def cpu_array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def observation_for_policy(observation, task):
    return {"state": cpu_array(observation["embodiment"]["joint"])[:8].astype(np.float32).copy(),
            "image": cpu_array(observation["observation"]["head"]["rgb"]).copy(),
            "wrist_image": cpu_array(observation["observation"]["wrist"]["rgb"]).copy(),
            "prompt": prompt(task), "task": task}


class ChunkController:
    def __init__(self, client, task_name, *, arm_clip_tolerance=1e-5):
        if not 0 <= arm_clip_tolerance <= 0.02:
            raise ValueError('Arm clipping tolerance must be between 0 and 0.02 radians')
        self.client, self.task_name = client, task_name
        self.arm_clip_tolerance = arm_clip_tolerance
        self.reset()

    def reset(self):
        self.chunk = None
        self.position = 0
        self.requests = 0
        self.gripper_clips = 0
        self.arm_clips = 0
        self.max_arm_clip_rad = 0.0

    def next_action(self, task, observation):
        if self.chunk is None or self.position == EXECUTE_STEPS:
            response = self.client.infer(observation_for_policy(observation, self.task_name))
            self.chunk = np.asarray(response["actions"], dtype=np.float32)
            if self.chunk.shape != (HORIZON, 8) or not np.isfinite(self.chunk).all():
                raise InvalidPolicyOutput(f"Expected finite (50,8), got {self.chunk.shape}")
            self.position = 0
            self.requests += 1
        action = self.chunk[self.position].copy()
        self.position += 1
        manager = task._robot_manager
        limits = cpu_array(manager.robot.data.joint_pos_limits)[0, cpu_array(manager._arm_ids)]
        bounded = np.clip(action[:7], limits[:, 0], limits[:, 1])
        correction = float(np.max(np.abs(action[:7] - bounded)))
        if correction > self.arm_clip_tolerance:
            raise InvalidPolicyOutput(
                f"Arm target outside joint limits by {correction:.6f} rad "
                f"(allowed {self.arm_clip_tolerance:.6f}): {action[:7].tolist()}")
        # Bounded regression overshoot is corrected at the actuator boundary.
        # Gross/non-finite outputs still terminate this episode without execution.
        if correction > 0:
            self.arm_clips += 1
            self.max_arm_clip_rad = max(self.max_arm_clip_rad, correction)
        action[:7] = bounded
        original = float(action[7])
        action[7] = np.clip(action[7], 0.0, manager.gripper_max_qpos)
        self.gripper_clips += int(float(action[7]) != original)
        return action


def execute_action(task, action):
    import torch
    tensor = torch.as_tensor(action, dtype=torch.float32, device=task._robot_manager.device)
    start = task.step_count
    success = False
    for _ in range(2):
        okay, success = task.take_action(tensor, action_type="qpos", force=True)
        if not okay or not task.plan_success:
            raise RuntimeError("Simulator rejected action or stopped advancing physics")
        if success:
            break
    steps = task.step_count - start
    if steps != (1 if success and steps == 1 else 2):
        raise RuntimeError(f"60 Hz action advanced {steps} physical steps; expected 2 (or 1 on success)")
    return success, steps
