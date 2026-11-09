from pathlib import Path
from types import SimpleNamespace
import sys
import json

import numpy as np
import pytest
import torch

from contact_belief_wam.serve import prepare_inference_model


def test_inference_conversion_keeps_world_and_contact_fp32():
    class Backbone(torch.nn.Linear):
        def to_bfloat16_for_selected_params(self, precision):
            assert precision == 'bfloat16'
            self.bfloat16()
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = torch.nn.Module()
            self.core.paligemma_with_expert = Backbone(3, 3)
            self.world = torch.nn.Linear(3, 3)
            self.contact = torch.nn.Linear(3, 3)
    model = Model()
    original = model.world.weight.detach().clone()
    report = prepare_inference_model(model, 'cpu', 'bfloat16')
    assert model.core.paligemma_with_expert.weight.dtype == torch.bfloat16
    assert model.world.weight.dtype == model.contact.weight.dtype == torch.float32
    torch.testing.assert_close(model.world.weight, original, atol=0, rtol=0)
    assert report['parameter_bytes_after'] < report['parameter_bytes_before']


@pytest.mark.parametrize('camera', ['head_only', 'head_wrist'])
def test_native_feedback_uses_past_frames_executed_actions_and_resets(camera):
    workspace = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(workspace / 'UniVTAC'))
    from policy.TactileWave0Wam.deploy_policy import Policy
    class Client:
        metadata = dict(k=10, camera_mode=camera)
        def __init__(self):
            self.calls = []
        def call(self, **message):
            self.calls.append(message)
            if message['op'] == 'infer':
                return dict(actions=np.zeros((50, 8), np.float32), diagnostics={},
                            arm_limit_overshoots=np.zeros(50, np.float32))
            return {}
    class Task:
        def __init__(self):
            self.cfg = SimpleNamespace(seed=1000000, step_lim=600)
            self.step_count = self.take_action_cnt = 0
            self.plan_success = True
            self._robot_manager = SimpleNamespace(device='cpu', _arm_ids=np.arange(7), gripper_max_qpos=1,
                robot=SimpleNamespace(data=SimpleNamespace(joint_pos_limits=np.array([[[-1, 1]] * 7]))))
        def _get_observations(self):
            touch = np.full((8, 8, 3), self.step_count // 2, np.uint8)
            return dict(tactile={f'{side}_tactile': {'rgb_marker': touch} for side in ('left', 'right')},
                        observation={v: {'rgb': np.zeros((8, 8, 3), np.uint8)}
                                     for v in (('head',) if camera == 'head_only' else ('head', 'wrist'))},
                        embodiment={'joint': np.zeros(8, np.float32)})
        def take_action(self, value, **kwargs):
            self.step_count += 1
            self.take_action_cnt += 1
            return True, False
    policy = object.__new__(Policy)
    policy.task_name = 'insert_HDMI'
    policy.client, policy.adapter, policy._expanded_step_limit = Client(), None, False
    task = Task()
    for _ in range(11):
        policy.eval(task, task._get_observations())
    messages = [x for x in policy.client.calls if x['op'] == 'infer']
    assert [x['step'] for x in messages] == [0, 10]
    assert task.cfg.step_lim == 1200 and task.step_count == 22
    assert messages[0]['executed_actions'].shape == (0, 8)
    assert messages[1]['executed_actions'].shape == (10, 8)
    np.testing.assert_array_equal(messages[0]['tactile_window'][0, :, 0, 0, 0], [0, 0, 0, 0])
    np.testing.assert_array_equal(messages[1]['tactile_window'][0, :, 0, 0, 0], [4, 6, 8, 10])
    assert ('wrist_image' in messages[0]) == (camera == 'head_wrist')
    assert not any('future' in key or 'target' in key for key in messages[1])
    policy.reset()
    assert policy.adapter is None and policy.client.calls[-1]['op'] == 'close'


@pytest.mark.parametrize('camera', ['head_only', 'head_wrist'])
def test_native_policy_reads_camera_from_checkpoint(tmp_path, monkeypatch, camera):
    workspace = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(workspace / 'UniVTAC'))
    from policy.TactileWave0Wam import deploy_policy as native
    (tmp_path / 'config.json').write_text(json.dumps(dict(camera_mode=camera)))
    deployment = dict(task='insert_tube', variant='full', mode='full', aligned=True,
                      camera=camera, simulator_action='absolute_joint', action_chunk_length=50,
                      executed_action_prefix=10, use_ema=True)
    (tmp_path / 'deployment.json').write_text(json.dumps(deployment))
    monkeypatch.setenv('TACTILE_WAVE0_CHECKPOINT', str(tmp_path))
    monkeypatch.setenv('PI05_SERVER_LOG', str(tmp_path / 'server.log'))
    class Process:
        pid = 123456789
        def poll(self): return None
        def terminate(self): pass
        def wait(self, timeout=None): return 0
    class Client:
        metadata = dict(k=10, chunk_size=50, action_dim=8, camera_mode=camera, mode='full',
                        physical_steps_per_action=2, use_ema=True, tactile_device='cuda',
                        backbone_precision='bfloat16', checkpoint=str(tmp_path))
        def __init__(self, *args, **kwargs): pass
        def close(self): pass
    monkeypatch.setattr(native.subprocess, 'Popen', lambda *args, **kwargs: Process())
    monkeypatch.setattr(native, 'Client', Client)
    args = dict(task_name='insert_tube', pi05_eval_protocol='bounded-actions-fixed-seeds-v2',
                pi05_port=18781, pi05_python=sys.executable)
    policy = native.Policy(args)
    assert policy.camera_mode == camera
    task = SimpleNamespace(metadata={}, take_action_cnt=20, cfg=SimpleNamespace(step_lim=600))
    policy.record_episode(task, 'success')
    assert task.metadata['policy_execution']['camera'] == camera
    policy.close()
    deployment['camera'] = 'head_only' if camera == 'head_wrist' else 'head_wrist'
    (tmp_path / 'deployment.json').write_text(json.dumps(deployment))
    with pytest.raises(ValueError, match='deployment contract'):
        native.Policy(args)
