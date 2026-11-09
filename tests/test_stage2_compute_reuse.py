"""Loss/gradient equivalence and compute counts for frozen-feature reuse."""
from collections import Counter
from dataclasses import replace
from types import MethodType

import pytest
import torch

from test_fast_attention import tiny_architecture, cfg, inputs
from contact_belief_wam import model as candidate
from contact_belief_wam.model import pad8
from contact_belief_wam.sequence import SequenceTrainingModel


class UncachedReference(candidate.ContactWAM):
    """Former history path: execute the entire visual/tactile prediction tree.

    Paired with batches lacking cache metadata, this is the original training
    computation, including the visual futures discarded during history warmup.
    """
    def history_step(self, obs, actions):
        """Recur from historical inputs and already-executed actions, without future targets."""
        belief = self.update_belief(obs)
        expected = None
        if self.config.mode == "full":
            current = self.encode_visual(obs.image, obs.wrist_image)
            world_state = obs.world_state if obs.world_state is not None else obs.state
            latent = self.core.world_action_encoder(pad8(actions[:, :self.config.k]).float(), pad8(world_state).float())
            _, _, expected = self.predict_future(current, world_state, latent, belief, obs.tactile, actions=actions)
        return belief, expected


def config(stage='stage2', checkpoint=True, independent=True):
    stats = {'actions': {'q01': [-1.] * 8, 'q99': [1.] * 8}}
    return replace(cfg('sdpa', checkpoint), aligned=True, stage=stage,
                   camera_mode='head_wrist', independent_tactile=independent,
                   stage1_freeze_world=True, stage2_freeze_belief=False,
                   policy_norm_stats=stats, world_norm_stats=stats)


def batch(mixed=False):
    b, length, h = 4, 5, 4
    frames = torch.arange(0, length * 2, 2).repeat(b, 1)
    valid = torch.ones(b, length, dtype=torch.bool)
    stops = torch.full((b, 1), 11)
    if mixed:
        for row, warmup in enumerate((4, 0, 1, 2)):
            valid[row, :length - warmup - 1] = False
            frames[row, :length - warmup - 1] = frames[row, length - warmup - 1]
        stops[1], stops[3] = 9, 10
    future = torch.minimum(frames + 2, stops - 1)
    row = torch.arange(b)[:, None]
    image = torch.rand(b, 11, 3, 28, 28)
    wrist = torch.rand_like(image)
    touch = torch.randn(b, 11, 8, 32)
    state = torch.randn(b, 11, 8)
    action = torch.randn(b, length, h, 32)
    action[..., 8:] = 0
    policy_mask = torch.ones(b, length, h, dtype=torch.bool)
    policy_mask[2, 3, :2] = False
    return dict(image=image[row, frames], future_image=image[row, future],
                wrist_image=wrist[row, frames], future_wrist_image=wrist[row, future],
                tactile=touch[row, frames], future_tactile=touch[row, future],
                state=state[row, frames], world_state=state[row, frames],
                actions=action, world_actions=action.clone(),
                tokens=torch.randint(0, 60, (b, length, 12)),
                token_mask=torch.ones(b, length, 12, dtype=torch.bool),
                action_mask=torch.ones(b, length, h, dtype=torch.bool), policy_action_mask=policy_mask,
                future_mask=((frames + 2) < stops) & valid, sequence_mask=valid,
                executed_actions=torch.randn(b, length, 2, 8),
                executed_valid=torch.ones(b, length, 2, dtype=torch.bool),
                step=frames, visual_frame_index=frames, future_visual_frame_index=future,
                task=['insert_hole'] * b)


def instrument(policy):
    counts = Counter()
    handles = []
    def count(name):
        def hook(module, arguments):
            counts[name + '_calls'] += 1
            if name == 'frozen_vision':
                counts['frozen_vision_images'] += len(arguments[0])
        return hook
    target = policy.core.vision_backbone if policy.config.world_only else policy.core.world_vision_backbone
    modules = [('frozen_vision', target), ('visual_wm', policy.core.world_model),
               ('world_action', policy.core.world_action_encoder)]
    if policy.contact is not None:
        modules += [('corrector', policy.contact.corrector), ('tactile_wm', policy.contact.predictor)]
    for name, module in modules:
        handles.append(module.register_forward_pre_hook(count(name)))
    if not policy.config.world_only:
        vlm = policy.core.paligemma_with_expert
        old = vlm.forward_vlm_with_context_kv
        def prefill(self, **kwargs):
            counts['vlm_prefill_calls'] += 1
            return old(**kwargs)
        vlm.forward_vlm_with_context_kv = MethodType(prefill, vlm)
    return counts, handles


@pytest.mark.parametrize('stage', ['stage1', 'stage2'])
@pytest.mark.parametrize('mixed', [False, True])
@pytest.mark.parametrize('bf16', [False, True])
def test_loss_gradients_and_rng_match(tiny_architecture, stage, mixed, bf16):
    torch.manual_seed(41)
    old = UncachedReference(config(stage)).train()
    new = candidate.ContactWAM(config(stage)).train()
    new.load_state_dict(old.state_dict())
    data = batch(mixed)
    records = []
    for model, use_cache in [(old, False), (new, True)]:
        model_batch = data if use_cache else {key: value for key, value in data.items()
                                              if key not in {'visual_frame_index', 'future_visual_frame_index'}}
        counts, handles = instrument(model)
        torch.manual_seed(333)
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
            result = SequenceTrainingModel(model)(model_batch)
        forward_counts = dict(counts)
        result['loss'].backward()
        grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        assert grads and all(torch.isfinite(g).all() for g in grads.values())
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        for handle in handles:
            handle.remove()
        records.append((result, grads, torch.get_rng_state(), forward_counts))
    (a, ga, ra, ca), (b, gb, rb, cb) = records
    # Different frozen encoder batch sizes can change BF16 kernel rounding.
    rtol, atol = (0.01, 0.003) if bf16 else (2e-5, 2e-6)
    torch.testing.assert_close(a['loss'], b['loss'], rtol=rtol, atol=atol)
    torch.testing.assert_close(a['belief'], b['belief'], rtol=rtol, atol=atol)
    assert ga.keys() == gb.keys()
    for name, gradient in ga.items():
        torch.testing.assert_close(gradient, gb[name], rtol=rtol, atol=atol,
                                   msg=lambda message: name + ': ' + message)
    for key in a['metrics']:
        for field in ('sum', 'count'):
            torch.testing.assert_close(a['metrics'][key][field], b['metrics'][key][field], rtol=rtol, atol=atol)
    assert torch.equal(ra, rb)
    assert cb['visual_wm_calls'] < ca['visual_wm_calls']
    assert cb['corrector_calls'] < ca['corrector_calls']
    assert cb['frozen_vision_images'] < ca['frozen_vision_images']
    assert cb.get('vlm_prefill_calls') == ca.get('vlm_prefill_calls')


@pytest.mark.parametrize('independent', [False, True])
def test_inference_contact_matches_full_prediction_path(tiny_architecture, independent):
    torch.manual_seed(43)
    policy = candidate.ContactWAM(config(independent=independent)).eval()
    obs, _, noise, _ = inputs()
    obs.wrist_image = torch.rand_like(obs.image)
    counts, handles = instrument(policy)
    prediction = policy.predict_action_chunk(obs, noise=noise)
    for handle in handles:
        handle.remove()
    assert counts['visual_wm_calls'] == (1 if independent else 2)
    assert counts['corrector_calls'] == (1 if independent else 2)
    assert counts['world_action_calls'] == (0 if independent else 1)
    # The final tactile prediction must still use the actual generated actions
    # after policy-to-world normalization, just as the former full WM path did.
    with torch.no_grad():
        current = policy.encode_visual(obs.image, obs.wrist_image)
        actions = policy.to_world_actions(pad8(prediction.actions[:, :policy.config.k]))
        latent = policy.core.world_action_encoder(actions, pad8(obs.state))
        _, _, expected = policy.predict_future(current, obs.state, latent, prediction.belief,
                                               obs.tactile, actions=actions)
    torch.testing.assert_close(prediction.expected_contact, expected, rtol=0, atol=0)


def test_legacy_coupled_history_matches(tiny_architecture):
    torch.manual_seed(44)
    old = UncachedReference(config(independent=False)).train()
    new = candidate.ContactWAM(config(independent=False)).train()
    new.load_state_dict(old.state_dict())
    obs, target, _, _ = inputs()
    obs.wrist_image = torch.rand_like(obs.image)
    with torch.no_grad():
        a = old.history_step(obs, target.actions)
        b = new.history_step(obs, target.actions)
    for expected, actual in zip(a, b):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_wave0_baseline_skips_history_vision_and_reuses_training_targets(tiny_architecture):
    torch.manual_seed(45)
    settings = replace(config(), mode='wave0', camera_mode='head_only')
    old = UncachedReference(settings).train()
    new = candidate.ContactWAM(settings).train()
    new.load_state_dict(old.state_dict())
    data = batch()
    results = []
    for policy, use_cache in ((old, False), (new, True)):
        counts, handles = instrument(policy)
        supplied = data if use_cache else {key: value for key, value in data.items()
                                           if key not in {'visual_frame_index', 'future_visual_frame_index'}}
        torch.manual_seed(123)
        result = SequenceTrainingModel(policy)(supplied)
        results.append((result['loss'], dict(counts), torch.get_rng_state()))
        for handle in handles:
            handle.remove()
    (a, ca, ra), (b, cb, rb) = results
    torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    assert torch.equal(ra, rb)
    assert ca['frozen_vision_images'] == 16
    assert cb['frozen_vision_images'] == 12
    assert ca['visual_wm_calls'] == cb['visual_wm_calls'] == 4
