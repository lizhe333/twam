"""Native policies for an isolated, matched A800 comparison."""
import json
import os
from pathlib import Path

import numpy as np

from pair_contract import INPUT_PROFILE, PROFILE, RNG_PROFILE, validate_render

RUNTIME = Path(os.environ['PAIR_RUNTIME']).resolve()
FAMILY = os.environ['PAIR_FAMILY']


class SeededClient:
    def __init__(self, client):
        self.client = client
        self.metadata = client.metadata
        self.scene, self.step = None, 0

    def infer(self, observation):
        if self.scene is None:
            raise RuntimeError('Set scene seed before requesting actions')
        result = self.client.infer(dict(observation, pair_scene_seed=self.scene, pair_policy_step=self.step))
        self.step += 10
        return result

    def close(self):
        return self.client.close()


class AuditMixin:
    def __init__(self, args):
        self._audited_seed = None
        super().__init__(args)

    def audit(self, task, observation):
        seed = int(task.cfg.seed)
        import carb.settings
        actual = validate_render(carb.settings.get_settings())
        if seed == self._audited_seed:
            return
        from PIL import Image
        from pi05_univtac.sim_policy import cpu_array
        pixels = cpu_array(observation['observation']['head']['rgb']).copy()
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3 or pixels.std() < 1:
            raise RuntimeError('Formal episode has an invalid/blank head RGB image')
        directory = task.save_root / 'input-audit'
        directory.mkdir(exist_ok=True)
        # Save the actual camera input, not the compressed video or an enhanced image.
        Image.fromarray(pixels).save(directory / f'{seed}-head.png')
        np.savez_compressed(directory / f'{seed}-initial.npz', image=pixels,
                            state=cpu_array(observation['embodiment']['joint'])[:8])
        record = dict(profile=PROFILE, family=FAMILY, scene_seed=seed,
                      render_settings=actual, input_shape=list(pixels.shape),
                      gpu_uuid=os.environ['PAIR_GPU_UUID'], rng_profile=RNG_PROFILE,
                      runtime_fingerprint=os.environ['PAIR_RUNTIME_FINGERPRINT'],
                      input_alignment=INPUT_PROFILE if FAMILY == 'tactile' else 'base-head-only',
                      note='Settings and input validity checked; image quality requires review of saved RGB.')
        (directory / f'{seed}.json').write_text(json.dumps(record, indent=2) + '\n')
        (task.save_root / 'render_contract.json').write_text(json.dumps(record, indent=2) + '\n')
        self._audit_record, self._audited_seed = record, seed
        if FAMILY == 'base':
            self.client.scene, self.client.step = seed, 0

    def reset(self):
        self._audited_seed = None
        return super().reset()

    def record_episode(self, task, outcome):
        super().record_episode(task, outcome)
        if self._audited_seed == int(task.cfg.seed):
            task.metadata['policy_execution']['comparison'] = self._audit_record


if FAMILY == 'tactile':
    # Loaded from the pinned src before importing the old native bridge.
    from aligned_feedback import TrainingAlignedAdapter
    from policy.TactileWave0Wam import deploy_policy as native
    native.PROJECT = RUNTIME

    class Policy(AuditMixin, native.Policy):
        def eval(self, task, observation):
            self.audit(task, observation)
            if self.adapter is None:
                self.adapter = TrainingAlignedAdapter(self.client, f'{self.task_name}:{task.cfg.seed}',
                    native.prompt(self.task_name), native.robot_action_bounds(task))
            return super().eval(task, observation)
elif FAMILY == 'base':
    from policy.Wave0Wam import deploy_policy as native
    native.TWAM_ROOT = RUNTIME / 'base_workspace'

    class Policy(AuditMixin, native.Policy):
        def __init__(self, args):
            super().__init__(args)
            self.client = SeededClient(self.client)
            self.controller.client = self.client

        def eval(self, task, observation):
            self.audit(task, observation)
            return super().eval(task, observation)
else:
    raise ValueError(f'Unknown comparison policy family: {FAMILY}')
