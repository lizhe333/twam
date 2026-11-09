"""Aligned tactile-only policy; record actual RGB/timing for each transfer episode."""
import json
import os
from pathlib import Path

import numpy as np

from aligned_feedback import TrainingAlignedAdapter
from contact_belief_wam.sim_eval import cpu
from pro6k_contract import INPUT_PROFILE, PROFILE, CONTACT_RESET_PATCH, validate_render, validate_timing
from policy.TactileWave0Wam import deploy_policy as native

native.PROJECT = Path(os.environ['PRO6K_RUNTIME'])


class TransferAdapter(TrainingAlignedAdapter):
    def execute(self, task, action):
        before = task._physics_step_count
        success, observation = super().execute(task, action)
        elapsed = task._physics_step_count - before
        if elapsed != 2 and not (success and elapsed == 1):
            raise RuntimeError(f'Expected two actual physics ticks per strategy action; got {elapsed}')
        return success, observation


class Policy(native.Policy):
    def __init__(self, args):
        self._audited_seed = None
        super().__init__(args)

    def eval(self, task, observation):
        seed = int(task.cfg.seed)
        if self._audited_seed != seed:
            import carb.settings
            from PIL import Image
            validate_timing(task)
            settings = validate_render(carb.settings.get_settings())
            pixels = cpu(observation['observation']['head']['rgb']).copy()
            if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3 or pixels.std() < 1:
                raise RuntimeError('Invalid/blank model RGB input')
            directory = task.save_root / 'input-audit'
            directory.mkdir(exist_ok=True)
            Image.fromarray(pixels).save(directory / f'{seed}-head.png')
            np.savez_compressed(directory / f'{seed}-initial.npz', image=pixels,
                                state=cpu(observation['embodiment']['joint'])[:8])
            self._audit = dict(profile=PROFILE, input_alignment=INPUT_PROFILE,
                contact_reset_patch=CONTACT_RESET_PATCH,
                task=self.task_name, scene_seed=seed, training_simulator='4.5', evaluation_simulator='5.1',
                gpu_uuid=os.environ['PRO6K_GPU_UUID'], render_settings=settings,
                gpu_binding=json.loads(os.environ['PRO6K_GPU_BINDING']),
                runtime_fingerprint=os.environ['PRO6K_RUNTIME_FINGERPRINT'],
                strategy_hz=60, physical_hz=120, decimation=1, action_repeats=2)
            (directory / f'{seed}.json').write_text(json.dumps(self._audit, indent=2) + '\n')
            self._audited_seed = seed
        if self.adapter is None:
            self.adapter = TransferAdapter(self.client, f'{self.task_name}:{seed}',
                native.prompt(self.task_name), native.robot_action_bounds(task))
        return super().eval(task, observation)

    def reset(self):
        self._audited_seed = None
        return super().reset()

    def record_episode(self, task, outcome):
        super().record_episode(task, outcome)
        if self._audited_seed == int(task.cfg.seed):
            task.metadata['policy_execution']['transfer'] = self._audit
