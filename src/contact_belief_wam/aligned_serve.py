"""Aligned delta-action service. The simulator still receives absolute joint targets."""
from pathlib import Path
import os
import threading
import time

import numpy as np
import torch
from safetensors.torch import load_file

from .paths import WORKSPACE
from .serve import PolicyService, EncoderProcess
from .types import ContactObservation


def inference_autocast_settings(device):
    """Resolve serving autocast; CPU BF16 is an explicit smoke-test opt-in."""
    device = torch.device(device)
    if device.type == 'cuda':
        return True, 'cuda:bfloat16'
    if device.type == 'cpu' and os.environ.get('CBWAM_DEBUG_CPU_BF16') == '1':
        return True, 'cpu:bfloat16'
    return False, 'disabled'


class AlignedPolicyService(PolicyService):
    def __init__(self, checkpoint, device='cpu', k=None, anytouch_source=None, anytouch_checkpoint=None,
                 tactile_device='cpu', threads=16, use_ema=True):
        from .aligned_data import AlignedProcessor
        from .model import ContactWAM
        path = Path(checkpoint)
        self.device = torch.device(device)
        self._autocast_enabled, self._effective_autocast = inference_autocast_settings(self.device)
        torch.set_num_threads(threads)
        self.model = ContactWAM.from_pretrained(path, stage='inference', device=device, k=k).eval()
        self.cfg = self.model.config
        if not self.cfg.aligned:
            raise ValueError('Aligned service requires an aligned checkpoint')
        if use_ema:
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
        self.processor = AlignedProcessor(path / 'norm_stats.json', path / 'tokenizer.model', '')
        self.world_processor = AlignedProcessor(path / 'wm_norm_stats.json', path / 'tokenizer.model', '')
        self.encoder = EncoderProcess(anytouch_source or WORKSPACE / 'AnyTouch2', anytouch_checkpoint,
                                      tactile_device) if self.cfg.use_belief else None
        self.sessions = {}
        self.lock = threading.Lock()
        self._closed = False
        self.metadata = {'mode': self.cfg.mode, 'k': self.cfg.k, 'chunk_size': self.cfg.chunk_size,
                         'action_dim': 8, 'action_space': 'absolute_joint',
                         'model_action_space': 'first7_chunk_relative_absolute_gripper',
                         'physical_steps_per_action': 2, 'policy_device': str(device),
                         'tactile_device': tactile_device if self.encoder else 'disabled', 'use_ema': use_ema,
                         'effective_autocast': self._effective_autocast}

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
        allowed = {'op','episode','step','image','state','tactile_window','executed_actions','prompt'}
        if set(message) - allowed:
            raise ValueError(f'Unexpected policy inputs: {sorted(set(message)-allowed)}')
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
