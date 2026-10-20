import copy
from dataclasses import replace

import torch

from test_fast_attention import tiny_architecture, cfg, inputs
from contact_belief_wam import model as cm
from contact_belief_wam.config import ContactConfig
from contact_belief_wam.sequence import SequenceTrainingModel
from lerobot.policies.wave0.modeling_wave0 import WAVE0Policy


def stats():
    return {key:{'q01':[-1.]*8,'q99':[1.]*8} for key in ('state','actions')}


def configuration(mode='full', stage='stage2'):
    return replace(cfg('sdpa'), aligned=True, mode=mode, stage=stage,
                   policy_norm_stats=stats(), world_norm_stats=stats())


def test_aligned_trainability_and_two_visual_towers(tiny_architecture):
    policy=cm.ContactWAM(configuration())
    pg=policy.core.paligemma_with_expert.paligemma
    assert all(p.requires_grad for p in pg.vision_tower.parameters())
    assert all(p.requires_grad for p in pg.multi_modal_projector.parameters())
    assert all(not p.requires_grad for p in policy.core.world_vision_backbone.parameters())
    assert {id(p) for p in pg.vision_tower.parameters()}.isdisjoint(
        {id(p) for p in policy.core.world_vision_backbone.parameters()})
    policy.train()
    assert pg.vision_tower.training
    assert not policy.core.world_vision_backbone.training
    assert policy.base_policy.config.output_features['action'].shape == (32,)
    assert policy.base_policy.config.world_model_future_horizons == (1,)


def test_aligned_world_bridge_gradients(tiny_architecture):
    config=configuration()
    config.world_norm_stats['actions']['q99']=[3.]*8
    model=cm.ContactWAM(config)
    value=torch.zeros(2,4,32,requires_grad=True)
    result=model.to_world_actions(value)
    expected=(1.*(2+1e-6)/2/(4+1e-6)*2-1)
    torch.testing.assert_close(result[...,:8],torch.full((2,4,8),expected))
    assert torch.count_nonzero(result[...,8:])==0
    result.sum().backward()
    assert value.grad[...,:8].abs().sum()>0


def test_wave0_aligned_stage2_matches_official_loss_and_gradients(tiny_architecture):
    torch.manual_seed(20)
    ours=cm.ContactWAM(configuration('wave0')).train()
    reference_config=copy.deepcopy(ours.base_policy.config)
    reference_config.world_model_stage='stage2'
    reference_config.world_model_stage1_checkpoint_path='/fixture/unused'
    reference=WAVE0Policy(reference_config).train()
    missing,extra=reference.model.load_state_dict(ours.core.state_dict(),strict=False)
    assert missing==['stage2_step'] and not extra
    reference.model.paligemma_with_expert.gemma_expert.lm_head.requires_grad_(False)
    for module in reference.modules():
        if hasattr(module,'num_key_value_groups'):
            module.wave0_attention_backend='sdpa'
    obs,target,noise,time=inputs()
    target.actions=cm.pad8(target.actions)
    target.world_actions=target.actions
    obs.world_state=obs.state
    target.policy_action_mask=torch.ones_like(target.action_mask)
    reference.model.sample_noise=lambda shape,device:noise.clone()
    reference.model.sample_time=lambda b,device:time.clone()
    batch={'observation.images.head':obs.image/2+.5,
           'observation.world_model.future.h1.images.head':target.future_image/2+.5,
           'observation.images.head_is_pad':torch.zeros(2,2,dtype=torch.bool),
           'observation.state':obs.state,'action':target.actions,
           'observation.language.tokens':obs.tokens,'observation.language.attention_mask':obs.token_mask,
           'world_model_task':['insert_hole']*2}
    a=ours(obs,target,noise=noise,time=time)['loss_per_sample'].mean()
    b,_=reference(batch)
    torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-5)
    a.backward();b.backward()
    expected=dict(reference.model.named_parameters())
    for name,p in ours.core.named_parameters():
        q=expected[name]
        assert (p.grad is None)==(q.grad is None),name
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=3e-4,msg=lambda msg:name+' '+msg)


def test_wave0_aligned_stage1_matches_official_world_objective(tiny_architecture):
    torch.manual_seed(27)
    ours=cm.ContactWAM(configuration('wave0','stage1')).train()
    reference=WAVE0Policy(copy.deepcopy(ours.base_policy.config)).train()
    reference.model.load_state_dict(ours.core.state_dict(),strict=True)
    obs,target,_,_=inputs()
    obs.world_state=obs.state
    target.world_actions=cm.pad8(target.actions)
    batch={'observation.images.head':obs.image/2+.5,
           'observation.world_model.future.h1.images.head':target.future_image/2+.5,
           'observation.images.head_is_pad':torch.zeros(2,2,dtype=torch.bool),
           'observation.state':obs.state,'action':target.world_actions,
           'world_model_task':['insert_hole']*2}
    torch.manual_seed(11)
    actual=ours(obs,target)['loss_per_sample'].mean()
    torch.manual_seed(11)
    expected,_=reference(batch)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    actual.backward();expected.backward()
    targets=dict(reference.model.named_parameters())
    for name,p in ours.core.named_parameters():
        other=targets[name]
        assert (p.grad is None)==(other.grad is None),name
        if p.grad is not None:
            torch.testing.assert_close(p.grad,other.grad,atol=1e-5,rtol=2e-4)
