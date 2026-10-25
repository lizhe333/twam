from types import SimpleNamespace
from dataclasses import replace
import json

import numpy as np
import pytest
import torch
from test_fast_attention import tiny_architecture, cfg

from contact_belief_wam.data import AlignedProcessor
from contact_belief_wam.serve import AlignedPolicyService
from contact_belief_wam.serve import inference_autocast_settings
from contact_belief_wam.types import ContactPrediction


def processor(scale):
    stats = {key: {'q01': [-scale]*8, 'q99': [scale]*8} for key in ('state', 'actions')}
    result = AlignedProcessor(stats, None, 'insert')
    result.tokens = lambda state: (np.zeros(200, np.int64), np.ones(200, bool))
    return result


def service():
    value = object.__new__(AlignedPolicyService)
    value.device = torch.device('cpu')
    value.cfg = SimpleNamespace(k=2, width=512)
    value.processor, value.world_processor = processor(1), processor(3)
    value.encoder = None
    value.metadata = {}
    value.sessions = {}
    value.seen = []
    def predict(obs, action_transform=None):
        value.seen.append(obs)
        actions = torch.ones(1, 50, 8) * .2
        if action_transform is not None:
            actions = action_transform(actions)
        return ContactPrediction(actions, torch.zeros(1,4,512), torch.zeros(1,8,512), obs.step+2, {})
    value.model = SimpleNamespace(predict_action_chunk=predict)
    return value


def message(step, state, executed=()):
    return dict(episode='test', step=step, state=state, executed_actions=executed,
                image=np.zeros((240,320,3),np.uint8),prompt='insert hole')


def test_previous_interval_anchor_and_reset():
    value = service()
    value.reset('test')
    anchor = np.arange(8,dtype=np.float32)/10
    first = value.infer(message(0,anchor))
    executed = first['actions'][:2]
    current = anchor + .3
    value.infer(message(2,current,executed))
    obs = value.seen[-1]
    expected = value.world_processor.actions(executed,anchor)[:,:8]
    np.testing.assert_allclose(obs.executed_actions[0].numpy(), expected)
    assert bool(obs.expected_valid[0])
    np.testing.assert_allclose(obs.world_state[0], value.world_processor.normalize(current,'state'),rtol=1e-6)
    value.reset('test')
    value.infer(message(0,current))
    assert value.seen[-1].belief is None
    assert not value.seen[-1].expected_valid.any()
    assert not value.seen[-1].executed_valid.any()


def test_relative_action_bounds_roundtrip():
    value = service()
    bounds = np.array([[-.1,.1]]*7+[[0,.15]],np.float32)
    value.reset('test',bounds)
    result = value.infer(message(0,np.full(8,.05,np.float32)))
    np.testing.assert_allclose(result['actions'][:,:7],.1,atol=1e-6)
    np.testing.assert_allclose(result['actions'][:,7],.15,atol=1e-6)
    np.testing.assert_allclose(result['arm_limit_overshoots'],.15,atol=1e-6)


def test_ema_query_prefix_survives_k_override(tiny_architecture,tmp_path):
    import shutil
    from safetensors.torch import save_file
    from contact_belief_wam.model import ContactWAM
    from contact_belief_wam.paths import WORKSPACE
    model = ContactWAM(replace(cfg('sdpa'),aligned=True,mode='wave0',k=4))
    model.save_pretrained(tmp_path)
    values = {name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad}
    key = 'base_policy.model.wam_latent_action_queries'
    values[key].fill_(.25)
    save_file(values,str(tmp_path/'ema.safetensors'))
    stats = {'norm_stats': processor(1).stats}
    for name in ('norm_stats.json','wm_norm_stats.json'):
        (tmp_path/name).write_text(json.dumps(stats))
    shutil.copyfile(WORKSPACE/'models/pi05_base/big_vision/paligemma_tokenizer.model',tmp_path/'tokenizer.model')
    value = AlignedPolicyService(tmp_path,k=2,threads=1)
    try:
        assert value.model.core.wam_latent_action_queries.shape[1] == 2
        torch.testing.assert_close(value.model.core.wam_latent_action_queries,
                                   values[key][:,:2],atol=0,rtol=0)
    finally:
        value.close()


def test_debug_cpu_bf16_is_explicit_and_cpu_fp32_remains_default(monkeypatch):
    monkeypatch.delenv('CBWAM_DEBUG_CPU_BF16', raising=False)
    assert inference_autocast_settings('cpu') == (False, 'disabled')
    monkeypatch.setenv('CBWAM_DEBUG_CPU_BF16', '1')
    assert inference_autocast_settings('cpu') == (True, 'cpu:bfloat16')
    assert inference_autocast_settings('cuda') == (True, 'cuda:bfloat16')


@pytest.mark.parametrize('mode', ['wave0', 'full'])
def test_cpu_bf16_service_context_runs_tiny_real_policy(tiny_architecture, tmp_path, monkeypatch, mode):
    from contact_belief_wam import data, serve
    from contact_belief_wam.model import ContactWAM

    stats = {key: {'q01': [-1.0] * 8, 'q99': [1.0] * 8} for key in ('state', 'actions')}
    config = replace(cfg('sdpa'), aligned=True, mode=mode, k=2, chunk_size=50,
                     policy_norm_stats=stats, world_norm_stats=stats)
    ContactWAM(config).save_pretrained(tmp_path)
    (tmp_path / 'norm_stats.json').write_text(json.dumps(stats))
    (tmp_path / 'wm_norm_stats.json').write_text(json.dumps(stats))
    (tmp_path / 'tokenizer.model').write_bytes(b'test-only')

    class StubProcessor:
        def __init__(self, *_args):
            self.prompt = ''

        def observation(self, _image, _state):
            return {
                'observation.images.head': torch.full((3, 28, 28), 0.5),
                'observation.state': torch.zeros(8),
                'observation.language.tokens': torch.ones(12, dtype=torch.long),
                'observation.language.attention_mask': torch.ones(12, dtype=torch.bool),
            }

        def normalize(self, value, _key):
            return np.asarray(value, dtype=np.float32)

        def absolute(self, value, _anchor):
            return np.asarray(value, dtype=np.float32)

    monkeypatch.setenv('CBWAM_DEBUG_CPU_BF16', '1')
    monkeypatch.setattr(data, 'AlignedProcessor', StubProcessor)
    # The real model's contact input is zero tactile for this execution check.
    monkeypatch.setattr(serve, 'EncoderProcess', lambda *_args, **_kwargs: None)
    service = AlignedPolicyService(tmp_path, device='cpu', threads=1, use_ema=False)
    autocast_seen = []
    original_predict = service.model.predict_action_chunk

    def check_autocast(*args, **kwargs):
        autocast_seen.append(torch.is_autocast_enabled('cpu'))
        return original_predict(*args, **kwargs)

    service.model.predict_action_chunk = check_autocast
    try:
        service.reset('test')
        result = service.infer(message(0, np.zeros(8, dtype=np.float32)))
        assert result['actions'].shape == (50, 8)
        assert np.isfinite(result['actions']).all()
        assert autocast_seen == [True]
        assert result['diagnostics']['effective_autocast'] == 'cpu:bfloat16'
    finally:
        service.close()
