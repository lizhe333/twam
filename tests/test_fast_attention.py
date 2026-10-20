"""Numerical checks on real reduced-width Contact/WAVE modules, without model downloads."""
from types import SimpleNamespace

import pytest
import torch

from contact_belief_wam.config import ContactConfig
from contact_belief_wam import model as contact
from contact_belief_wam.types import ContactObservation, ContactTargets
from lerobot.policies.wave0 import modeling_wave0 as m


@pytest.fixture
def tiny_architecture(monkeypatch):
    from transformers import PaliGemmaConfig
    torch.set_num_threads(1)
    original_pali, original_expert = m.PaliGemmaForConditionalGeneration, m.GemmaForCausalLM
    def pali(config):
        config.text_config.vocab_size = 64
        config._vocab_size = 64
        config.image_token_index = 63
        config.vision_config.projection_dim = 32
        return original_pali(config)
    def expert(config):
        config.vocab_size = 64
        return original_expert(config)
    monkeypatch.setattr(m, 'PaliGemmaForConditionalGeneration', pali)
    monkeypatch.setattr(m, 'GemmaForCausalLM', expert)
    monkeypatch.setattr(m, 'get_gemma_config', lambda _: m.GemmaConfig(32, 2, 64, 4, 1, 8))
    monkeypatch.setattr(m, 'CONFIG_MAPPING', {'gemma': m.CONFIG_MAPPING['gemma'], 'paligemma': lambda: PaliGemmaConfig(
        vision_config=dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                           intermediate_size=64, image_size=28, patch_size=14, vision_use_head=False),
        text_config=dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=1, head_dim=8, intermediate_size=64, vocab_size=64))})
    original_config = contact.wave_config
    def small_config(cfg):
        value = original_config(cfg)
        value.image_resolution = (28, 28)
        value.world_model_spatial_pool_size = 2
        value.wam_future_condition_spatial_pool_size = 2
        return value
    monkeypatch.setattr(contact, 'wave_config', small_config)


def cfg(backend, checkpoint=False):
    return ContactConfig(stage='stage2', dtype='float32', width=32, heads=4, ffn=64,
                         world_width=32, world_depth=2, world_heads=4, world_ffn=64,
                         vision_width=32, expert_width=32, k=2, chunk_size=4,
                         attention_backend=backend, gradient_checkpointing=checkpoint,
                         num_inference_steps=2, stage2_freeze_belief=True,
                         stage2_freeze_contact_predictor=True)


def inputs():
    obs = ContactObservation(image=torch.rand(2,3,28,28), state=torch.randn(2,8),
        tactile=torch.randn(2,8,32), tokens=torch.randint(0,60,(2,12)),
        token_mask=torch.tensor([[True]*10+[False]*2]*2), step=torch.tensor([0,2]),
        task=['insert_hole']*2)
    target = ContactTargets(actions=torch.randn(2,4,8), action_mask=torch.ones(2,4,dtype=torch.bool),
        future_image=torch.rand(2,3,28,28), future_tactile=torch.randn(2,8,32),
        future_mask=torch.ones(2,dtype=torch.bool))
    return obs, target, torch.randn(2,4,32), torch.tensor([.3,.7])


@pytest.mark.parametrize('checkpoint', [False, True])
def test_contact_loss_gradients_and_inference_match(tiny_architecture, checkpoint):
    torch.manual_seed(31)
    eager = contact.ContactWAM(cfg('eager', checkpoint)).train()
    fast = contact.ContactWAM(cfg('sdpa', checkpoint)).train()
    fast.load_state_dict(eager.state_dict(), strict=True)
    assert fast.base_policy.config.empty_cameras == 0
    backends = [x.wave0_attention_backend for x in fast.modules() if isinstance(x, m.modeling_gemma.GemmaAttention)]
    assert backends and set(backends) == {'sdpa'}
    obs, target, noise, time = inputs()
    outputs = []
    for policy in (eager, fast):
        result = policy(obs, target, noise=noise.clone(), time=time)
        result['loss_per_sample'].mean().backward()
        gradients = {n:p.grad.clone() for n,p in policy.named_parameters() if p.grad is not None}
        assert gradients
        assert all(p.grad is None for p in policy.parameters() if not p.requires_grad)
        outputs.append((result['loss_per_sample'].detach(), gradients))
    torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=2e-6, rtol=2e-5)
    assert outputs[0][1].keys() == outputs[1][1].keys()
    for name, grad in outputs[0][1].items():
        torch.testing.assert_close(grad, outputs[1][1][name], atol=2e-5, rtol=2e-4, msg=lambda msg:name+' '+msg)
    a = eager.predict_action_chunk(obs, noise=noise.clone())
    b = fast.predict_action_chunk(obs, noise=noise.clone())
    torch.testing.assert_close(a.actions, b.actions, atol=5e-5, rtol=2e-4)
    torch.testing.assert_close(a.expected_contact, b.expected_contact, atol=5e-5, rtol=2e-4)


def test_checkpoint_preserves_backend_and_fixed_noise(tiny_architecture, tmp_path):
    torch.manual_seed(13)
    policy = contact.ContactWAM(cfg('sdpa')).eval()
    obs, _, noise, _ = inputs()
    expected = policy.predict_action_chunk(obs, noise=noise)
    policy.save_pretrained(tmp_path)
    restored = contact.ContactWAM.from_pretrained(tmp_path).eval()
    assert restored.config.attention_backend == 'sdpa'
    actual = restored.predict_action_chunk(obs, noise=noise)
    torch.testing.assert_close(expected.actions, actual.actions, atol=0, rtol=0)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_block_mask_gqa_attention_forward_and_input_gradients(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(4)
    dtype = torch.float32 if device == 'cpu' else torch.bfloat16
    tensors = [torch.randn(*shape, device=device, dtype=dtype) for shape in
               [(2,4,19,32), (2,1,25,32), (2,1,25,32)]]
    mask = torch.zeros(2,1,19,25,device=device,dtype=dtype)
    mask[:,:,:7,7:] = -1e4
    mask[:,:,:,22:] = -1e4
    outputs = []
    for backend in ('eager','sdpa'):
        q,k,v = [x.clone().requires_grad_() for x in tensors]
        module = SimpleNamespace(num_key_value_groups=4, training=True, wave0_attention_backend=backend)
        y,_ = m.wave0_attention_forward(module,q,k,v,mask,.125)
        y.float().square().mean().backward()
        outputs.append((y.detach(), q.grad, k.grad, v.grad))
    for a,b in zip(*outputs):
        torch.testing.assert_close(a,b,atol=2e-3 if device=='cuda' else 1e-6, rtol=.04 if device=='cuda' else 1e-5)


def test_legacy_config_defaults_to_eager():
    original = ContactConfig().to_dict()
    original.pop('attention_backend')
    assert ContactConfig.from_dict(original).attention_backend == 'eager'
    with pytest.raises(ValueError, match='attention backend'):
        ContactConfig(attention_backend='invalid')
