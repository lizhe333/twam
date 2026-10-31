from __future__ import annotations

from dataclasses import dataclass
import json
import os
import multiprocessing as mp
from pathlib import Path
import signal
import threading
import time
import traceback

import numpy as np
import torch
from safetensors.torch import load_file

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


def inference_autocast_settings(device):
    """Resolve serving autocast; CPU BF16 is an explicit smoke-test opt-in."""
    device = torch.device(device)
    if device.type == 'cuda':
        return True, 'cuda:bfloat16'
    if device.type == 'cpu' and os.environ.get('CBWAM_DEBUG_CPU_BF16') == '1':
        return True, 'cpu:bfloat16'
    return False, 'disabled'


def prepare_inference_model(model, device, backbone_precision='float32'):
    """Convert only the policy backbone, after loading EMA and before GPU allocation."""
    if backbone_precision not in ('float32', 'bfloat16'):
        raise ValueError(f'Unsupported backbone precision: {backbone_precision}')
    if any(p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('Inference preparation requires a CPU-loaded model')
    before = sum(p.numel() * p.element_size() for p in model.parameters())
    if backbone_precision == 'bfloat16':
        model.core.paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16')
    after = sum(p.numel() * p.element_size() for p in model.parameters())
    model.to(device).eval()
    return dict(backbone_precision=backbone_precision, world_contact_precision='float32',
                parameter_bytes_before=before, parameter_bytes_after=after)


class AlignedPolicyService(PolicyService):
    def __init__(self, checkpoint, device='cpu', k=None, anytouch_source=None, anytouch_checkpoint=None,
                 tactile_device='cpu', threads=16, use_ema=True, backbone_precision='float32'):
        from .data import AlignedProcessor
        from .model import ContactWAM
        path = Path(checkpoint)
        self.device = torch.device(device)
        self._autocast_enabled, self._effective_autocast = inference_autocast_settings(self.device)
        torch.set_num_threads(threads)
        print(json.dumps({'serving_phase': 'load_checkpoint_on_cpu', 'checkpoint': str(path)}), flush=True)
        self.model = ContactWAM.from_pretrained(path, stage='inference', device='cpu', k=k).eval()
        self.cfg = self.model.config
        if not self.cfg.aligned:
            raise ValueError('Aligned service requires an aligned checkpoint')
        if use_ema:
            print(json.dumps({'serving_phase': 'load_ema_on_cpu'}), flush=True)
            values = load_file(str(path / 'ema.safetensors'), device='cpu')
            expected = self.model.state_dict()
            query_key = 'base_policy.model.wam_latent_action_queries'
            if query_key in values and values[query_key].shape != expected[query_key].shape:
                source, target = values[query_key], expected[query_key]
                if source.ndim == target.ndim == 3 and source.shape[0] == target.shape[0] and source.shape[2] == target.shape[2] and source.shape[1] >= target.shape[1]:
                    values[query_key] = source[:, :target.shape[1]].contiguous()
            if not values or any(n not in expected or v.shape != expected[n].shape for n, v in values.items()):
                raise ValueError('EMA keys/shapes do not match this model')
            self.model.load_state_dict(values, strict=False)
            del values, expected
        precision = prepare_inference_model(self.model, self.device, backbone_precision)
        print(json.dumps({'serving_phase': 'policy_on_device', 'device': str(self.device), **precision}), flush=True)
        self.processor = AlignedProcessor(path / 'norm_stats.json', path / 'tokenizer.model', '')
        self.world_processor = AlignedProcessor(path / 'wm_norm_stats.json', path / 'tokenizer.model', '')
        self.encoder = EncoderProcess(anytouch_source or WORKSPACE / 'AnyTouch2', anytouch_checkpoint,
                                      tactile_device) if self.cfg.use_belief else None
        self.sessions = {}
        self.lock = threading.Lock()
        self._closed = False
        self.metadata = {'mode': self.cfg.mode, 'k': self.cfg.k, 'chunk_size': self.cfg.chunk_size,
                         'camera_mode': self.cfg.camera_mode,
                         'action_dim': 8, 'action_space': 'absolute_joint',
                         'model_action_space': 'first7_chunk_relative_absolute_gripper',
                         'physical_steps_per_action': 2, 'policy_device': str(device),
                         'tactile_device': tactile_device if self.encoder else 'disabled', 'use_ema': use_ema,
                         'effective_autocast': self._effective_autocast,
                         'checkpoint': str(path.resolve()),
                         'encoder': self.encoder.metadata if self.encoder else None, **precision}

    def _autocast_context(self):
        # Defaults to disabled for CPU to preserve existing evaluation behavior.
        # The environment switch is read once at service construction so the
        # diagnostics describe the mode used for every inference in the run.
        enabled = getattr(self, '_autocast_enabled', self.device.type == 'cuda')
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=enabled)

    def reset(self, episode, action_bounds=None):
        result = super().reset(episode, action_bounds)
        self.sessions[episode].anchor_state = None
        return result

    def bounded_transform(self, bounds, anchor):
        diagnostics = {}
        def transform(actions):
            values = actions.detach().float().cpu().numpy()
            absolute = self.processor.absolute(values, anchor)
            bounded = np.clip(absolute, bounds[:, 0], bounds[:, 1])
            diagnostics['arm_limit_overshoots'] = np.abs(absolute[..., :7] - bounded[..., :7]).max(-1)
            # Processor.actions accepts [chunk,8]; policy transform uses [B,chunk,8].
            encoded = np.stack([self.processor.actions(sample, anchor)[..., :8] for sample in bounded])
            return torch.as_tensor(encoded, dtype=actions.dtype, device=actions.device)
        return transform, diagnostics

    @torch.inference_mode()
    def infer(self, message):
        allowed = {'op','episode','step','image','wrist_image','state','tactile_window','executed_actions','prompt'}
        if set(message) - allowed:
            raise ValueError(f'Unexpected policy inputs: {sorted(set(message)-allowed)}')
        dual_view = getattr(self.cfg, 'camera_mode', 'head_only') == 'head_wrist'
        if dual_view and message.get('wrist_image') is None:
            raise ValueError('head_wrist checkpoint requires wrist_image')
        episode, step = str(message['episode']), int(message['step'])
        if episode not in self.sessions:
            raise ValueError('Reset episode before inference')
        session = self.sessions[episode]
        if step < 0 or step <= session.last_step:
            raise ValueError('Policy steps must increase within an episode')
        begin = time.monotonic()
        tactile = (self.encoder(np.asarray(message['tactile_window'], np.uint8)[None]) if self.encoder
                   else np.zeros((1,8,self.cfg.width), np.float32))
        touch_elapsed = time.monotonic() - begin
        raw_state = np.asarray(message['state'], np.float32)
        if raw_state.shape != (8,) or not np.isfinite(raw_state).all():
            raise ValueError('Expected finite raw eight-joint state')
        self.processor.prompt = str(message['prompt']).strip().replace('_',' ').replace('\n',' ')
        observation = self.processor.observation(np.asarray(message['image']), raw_state)
        wrist_image = None
        if dual_view:
            from .data import resize
            pixels = resize(np.asarray(message['wrist_image'])).transpose(2, 0, 1)
            wrist_image = torch.from_numpy(pixels).float()[None].to(self.device) / 127.5 - 1
        world_state = self.world_processor.normalize(raw_state, 'state').astype(np.float32)
        executed = np.asarray(message.get('executed_actions', []), np.float32).reshape(-1,8)
        if len(executed) > self.cfg.k or not np.isfinite(executed).all():
            raise ValueError('Invalid executed action interval')
        if session.last_step >= 0 and step-session.last_step != len(executed):
            raise ValueError('Executed actions do not match elapsed policy steps')
        history = np.zeros((1,self.cfg.k,8), np.float32)
        valid_actions = np.zeros((1,self.cfg.k), bool)
        if len(executed):
            if session.anchor_state is None:
                raise ValueError('Executed actions require a previous interval anchor')
            history[:, -len(executed):] = self.world_processor.actions(executed, session.anchor_state)[..., :8]
            valid_actions[:, -len(executed):] = True
        tensor = lambda value: torch.as_tensor(value, device=self.device)
        expected_valid = session.expected is not None and session.target_step == step
        obs = ContactObservation(
            image=observation['observation.images.head'][None].to(self.device)*2-1,
            wrist_image=wrist_image,
            state=observation['observation.state'][None].to(self.device),
            world_state=tensor(world_state[None]), tactile=tensor(tactile),
            tokens=observation['observation.language.tokens'][None].to(self.device),
            token_mask=observation['observation.language.attention_mask'][None].to(self.device),
            belief=session.belief, belief_valid=tensor([session.belief is not None]),
            expected_contact=session.expected, expected_valid=tensor([expected_valid]),
            executed_actions=tensor(history), executed_valid=tensor(valid_actions), step=tensor([step]))
        error = (float((1-torch.nn.functional.cosine_similarity(session.expected.float(), obs.tactile,dim=-1)).mean())
                 if expected_valid else None)
        transform, bounds_info = (self.bounded_transform(session.action_bounds, raw_state)
                                  if session.action_bounds is not None else (None,None))
        with self._autocast_context():
            prediction = self.model.predict_action_chunk(obs, action_transform=transform)
        actions = self.processor.absolute(prediction.actions[0].float().cpu().numpy(), raw_state)
        if session.action_bounds is not None:
            actions = np.clip(actions, session.action_bounds[:,0], session.action_bounds[:,1])
        if actions.shape != (50,8) or not np.isfinite(actions).all():
            raise FloatingPointError('Invalid absolute action chunk')
        diagnostics = {'step':step, 'belief_norm':prediction.belief.float().norm(dim=-1).cpu().numpy()[0],
            'belief_delta':float((prediction.belief-session.belief).float().norm(dim=-1).mean()) if session.belief is not None else None,
            'contact_prediction_error':error, 'expectation_valid':expected_valid,
            'tactile_seconds':touch_elapsed, 'inference_seconds':time.monotonic()-begin,
            'target_step':int(prediction.target_step.item()), **self.metadata}
        session.belief, session.expected = prediction.belief, prediction.expected_contact
        session.last_step, session.target_step = step, diagnostics['target_step']
        session.anchor_state = raw_state.copy()
        result = {'actions':actions, 'diagnostics':diagnostics}
        if bounds_info is not None:
            result['arm_limit_overshoots'] = bounds_info['arm_limit_overshoots'][0]
        return result


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
            service_type = AlignedPolicyService
        elif kwargs.pop('backbone_precision', 'float32') != 'float32':
            raise ValueError('Backbone precision override requires an aligned checkpoint')
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
