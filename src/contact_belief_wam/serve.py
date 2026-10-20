from __future__ import annotations

from dataclasses import dataclass
import json
import multiprocessing as mp
from pathlib import Path
import signal
import threading
import time
import traceback

import numpy as np
import torch

from .paths import WORKSPACE
from .preprocessing import PromptTokenizer, Quantiles, vision_preprocess
from .protocol import pack, unpack
from .types import ContactObservation


def _encoder_worker(connection, source, checkpoint, device, threads):
    try:
        from .tactile import AnyTouchEncoder
        encoder = AnyTouchEncoder(source, checkpoint, device=device, threads=threads)
        connection.send({"metadata": encoder.metadata})
        while True:
            value = connection.recv()
            if value is None:
                break
            connection.send({"tokens": encoder(value)})
    except Exception:
        connection.send({"error": traceback.format_exc()})
    finally:
        connection.close()


class EncoderProcess:
    def __init__(self, source, checkpoint=None, device="cpu", threads=8):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_encoder_worker, args=(child, str(source), checkpoint, device, threads))
        self._closed = False
        try:
            self.process.start()
        except BaseException:
            child.close()
            self.connection.close()
            raise
        child.close()
        try:
            self.metadata = self.receive()["metadata"]
        except BaseException:
            # Includes SIGTERM/SystemExit while waiting for the worker handshake.
            self.close()
            raise

    def receive(self):
        if not self.connection.poll(1800):
            raise TimeoutError("AnyTouch2 process did not return within 30 minutes")
        value = self.connection.recv()
        if "error" in value:
            raise RuntimeError(value["error"])
        return value

    def __call__(self, windows):
        self.connection.send(windows)
        return self.receive()["tokens"]

    def close(self):
        if self._closed:
            return
        self._closed = True

        def alive():
            try:
                return self.process.is_alive()
            except (AssertionError, ValueError):
                return False

        def join(timeout):
            try:
                self.process.join(timeout=timeout)
            except (AssertionError, ValueError):
                pass

        try:
            if getattr(self.process, "pid", None) is not None and alive():
                try:
                    self.connection.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
                join(timeout=10)
                if alive():
                    try:
                        self.process.terminate()
                    except (OSError, ProcessLookupError):
                        pass
                    join(timeout=5)
                if alive():
                    # This is the one child created by this EncoderProcess.
                    try:
                        self.process.kill()
                    except (OSError, ProcessLookupError):
                        pass
                    join(timeout=5)
            if getattr(self.process, "pid", None) is not None and not alive():
                try:
                    self.process.close()
                except (AttributeError, ValueError):
                    pass
        finally:
            try:
                self.connection.close()
            except OSError:
                pass


@dataclass
class Session:
    belief: object = None
    expected: object = None
    target_step: int = -1
    last_step: int = -1
    action_bounds: object = None


class PolicyService:
    def __init__(self, checkpoint, device="cpu", k=None, anytouch_source=None, anytouch_checkpoint=None,
                 tactile_device="cpu", threads=16):
        from .model import ContactWAM
        self.device = torch.device(device)
        torch.set_num_threads(threads)
        self.model = ContactWAM.from_pretrained(checkpoint, stage="inference", device=device, k=k).eval()
        self.cfg = self.model.config
        self.norm = Quantiles(Path(checkpoint) / "quantiles.json")
        self.tokenizer = PromptTokenizer(Path(checkpoint) / "tokenizer", self.cfg.tokenizer_max_length)
        self.encoder = EncoderProcess(anytouch_source or WORKSPACE / "AnyTouch2", anytouch_checkpoint,
                                      tactile_device) if self.cfg.use_belief else None
        self.sessions = {}
        self.lock = threading.Lock()
        self._closed = False
        self.metadata = {"mode": self.cfg.mode, "k": self.cfg.k, "chunk_size": 50, "action_dim": 8,
                         "action_space": "absolute_joint", "physical_steps_per_action": 2,
                         "policy_device": str(device), "tactile_device": tactile_device if self.encoder else "disabled"}

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.encoder is not None:
            self.encoder.close()

    def reset(self, episode, action_bounds=None):
        bounds = None
        if action_bounds is not None:
            bounds = np.asarray(action_bounds, dtype=np.float32)
            if bounds.shape != (8, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]):
                raise ValueError("action_bounds must be finite [8,2] lower/upper absolute joint limits")
            bounds = bounds.copy()
        self.sessions[episode] = Session(action_bounds=bounds)
        return {"reset": episode, "belief_cleared": True, "prediction_cleared": True}

    def _bounded_action_transform(self, bounds):
        limit_diagnostics = {}

        def transform(normalized_actions):
            # model.predict_action_chunk exposes normalized absolute targets.
            # Clip in robot joint space, then return to the training scale so
            # the world encoder/contact predictor consume the executed plan.
            values = normalized_actions.detach().float().cpu().numpy()
            qpos = self.norm.unnormalize(values, "action")
            bounded = np.clip(qpos, bounds[:, 0], bounds[:, 1])
            overshoots = np.abs(qpos[..., :7] - bounded[..., :7]).max(axis=-1)
            limit_diagnostics["arm_limit_overshoots"] = overshoots
            normalized = self.norm.normalize(bounded, "action")
            return torch.as_tensor(normalized, device=normalized_actions.device, dtype=normalized_actions.dtype)

        return transform, limit_diagnostics

    @torch.inference_mode()
    def infer(self, message):
        allowed = {"op", "episode", "step", "image", "state", "tactile_window", "executed_actions", "prompt"}
        if set(message) - allowed:
            raise ValueError(f"Unexpected policy inputs: {sorted(set(message) - allowed)}")
        episode, step = str(message["episode"]), int(message["step"])
        if episode not in self.sessions:
            raise ValueError("Reset the episode before the first observation")
        session = self.sessions[episode]
        if step < 0 or step <= session.last_step:
            raise ValueError("Policy steps must increase within an episode")
        begin = time.monotonic()
        tokens = self.encoder(np.asarray(message["tactile_window"], np.uint8)[None]) if self.encoder else np.zeros((1, 8, 512), np.float32)
        touch_elapsed = time.monotonic() - begin
        state = self.norm.normalize(np.asarray(message["state"], np.float32)[None], "state")
        ids, token_mask = self.tokenizer([message["prompt"]], state)
        executed = np.asarray(message.get("executed_actions", []), np.float32).reshape(-1, 8)
        if len(executed) > self.cfg.k:
            raise ValueError("Too many executed actions in one feedback interval")
        if session.last_step >= 0 and step - session.last_step != len(executed):
            raise ValueError("Executed actions do not account for the observed policy-step interval")
        action_history = np.zeros((1, self.cfg.k, 8), np.float32)
        action_valid = np.zeros((1, self.cfg.k), bool)
        if len(executed):
            action_history[:, -len(executed):] = self.norm.normalize(executed, "action")
            action_valid[:, -len(executed):] = True
        as_tensor = lambda x: torch.as_tensor(x, device=self.device)
        expected_valid = session.expected is not None and session.target_step == step
        obs = ContactObservation(image=vision_preprocess(np.asarray(message["image"])[None]).to(self.device),
                                 state=as_tensor(state), tactile=as_tensor(tokens), tokens=ids.to(self.device),
                                 token_mask=token_mask.to(self.device), belief=session.belief,
                                 expected_contact=session.expected, expected_valid=as_tensor([expected_valid]),
                                 belief_valid=as_tensor([session.belief is not None]),
                                 executed_actions=as_tensor(action_history), executed_valid=as_tensor(action_valid),
                                 step=as_tensor([step]))
        predicted_error = None
        if expected_valid:
            predicted_error = float((1 - torch.nn.functional.cosine_similarity(session.expected.float(), obs.tactile, dim=-1)).mean())
        limit_diagnostics = None
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.cfg.dtype == "bfloat16"):
            if session.action_bounds is None:
                prediction = self.model.predict_action_chunk(obs)
            else:
                action_transform, limit_diagnostics = self._bounded_action_transform(session.action_bounds)
                prediction = self.model.predict_action_chunk(obs, action_transform=action_transform)
        actions = self.norm.unnormalize(prediction.actions.float().cpu().numpy()[0])
        if session.action_bounds is not None:
            actions = np.clip(actions, session.action_bounds[:, 0], session.action_bounds[:, 1])
        if actions.shape != (50, 8) or not np.isfinite(actions).all():
            raise FloatingPointError("Policy did not produce finite 50 × 8 actions")
        diagnostics = {"step": step, "belief_norm": prediction.belief.float().norm(dim=-1).cpu().numpy()[0],
                       "belief_delta": float((prediction.belief - session.belief).float().norm(dim=-1).mean()) if session.belief is not None else None,
                       "contact_prediction_error": predicted_error, "expectation_valid": expected_valid,
                       "tactile_seconds": touch_elapsed, "inference_seconds": time.monotonic() - begin,
                       "target_step": int(prediction.target_step.item()), **self.metadata}
        session.belief, session.expected = prediction.belief, prediction.expected_contact
        session.target_step, session.last_step = diagnostics["target_step"], step
        result = {"actions": actions, "diagnostics": diagnostics}
        if limit_diagnostics is not None:
            overshoots = np.asarray(limit_diagnostics.get("arm_limit_overshoots"), dtype=np.float32)
            if overshoots.shape != (1, 50) or not np.isfinite(overshoots).all():
                raise ValueError(f"Expected finite arm-limit overshoots [1,50], got {overshoots.shape}")
            result["arm_limit_overshoots"] = overshoots[0]
        return result

    def handle(self, message):
        with self.lock:
            if message.get("op") == "reset":
                return self.reset(str(message["episode"]), message.get("action_bounds"))
            if message.get("op") == "close":
                self.sessions.pop(str(message["episode"]), None)
                return {"closed": True}
            if message.get("op") == "infer":
                return self.infer(message)
            raise ValueError("Unknown operation")


def serve(checkpoint, host="127.0.0.1", port=18780, **kwargs):
    from websockets.sync.server import serve as websocket_serve

    def exit_through_cleanup(signum, frame):
        raise SystemExit(128 + signum)

    # Popen.terminate() in the eval command sends SIGTERM. Convert its default
    # immediate process exit into normal stack unwinding so the service finally
    # block can stop and reap its private AnyTouch worker.
    old_sigterm = None
    if threading.current_thread() is threading.main_thread():
        old_sigterm = signal.signal(signal.SIGTERM, exit_through_cleanup)
    service = None
    def handler(socket):
        socket.send(pack(service.metadata))
        for value in socket:
            try:
                result = service.handle(unpack(value))
            except Exception:
                result = {"error": traceback.format_exc()}
            socket.send(pack(result))
    try:
        service_type = PolicyService
        config_path = Path(checkpoint) / "config.json"
        if config_path.is_file() and json.loads(config_path.read_text()).get("aligned", False):
            from .aligned_serve import AlignedPolicyService
            service_type = AlignedPolicyService
        service = service_type(checkpoint, **kwargs)
        print(json.dumps({"ready": True, "host": host, "port": port, **service.metadata}), flush=True)
        with websocket_serve(handler, host, port, max_size=None, compression=None, ping_interval=None) as server:
            server.serve_forever()
    finally:
        try:
            if service is not None:
                service.close()
        finally:
            if old_sigterm is not None:
                signal.signal(signal.SIGTERM, old_sigterm)
