import torch

from contact_belief_wam.config import ContactConfig
from contact_belief_wam.modules import BeliefUpdater, ContactPredictor, FutureCorrector
from contact_belief_wam.types import ContactObservation


def _config(**overrides):
    values = {
        "k": 2,
        "chunk_size": 6,
        "width": 16,
        "heads": 4,
        "ffn": 32,
        "belief_tokens": 4,
        "world_width": 20,
        "vision_width": 12,
        "expert_width": 24,
        "vision_pool": 2,
    }
    values.update(overrides)
    return ContactConfig(**values)


def _observation(cfg, batch=2, belief=None, belief_valid=None, expected_contact=None, expected_valid=None):
    device = belief.device if belief is not None else torch.device("cpu")
    return ContactObservation(
        image=torch.zeros(batch, 3, 8, 8, device=device),
        state=torch.randn(batch, 8, device=device),
        tactile=torch.randn(batch, 8, cfg.width, device=device),
        belief=belief,
        belief_valid=belief_valid,
        expected_contact=expected_contact,
        expected_valid=expected_valid,
        executed_actions=torch.randn(batch, cfg.k, 8, device=device),
        executed_valid=torch.ones(batch, cfg.k, dtype=torch.bool, device=device),
    )


def _assert_finite_parameter_grads(module):
    grads = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    assert grads and all(grad is not None for grad in grads)
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert any(torch.count_nonzero(grad).item() for grad in grads)


def _assert_finite_input_grad(value):
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()
    assert torch.count_nonzero(value.grad).item()


def test_belief_updater_shape_and_finite_gradients():
    torch.manual_seed(11)
    cfg = _config()
    module = BeliefUpdater(cfg)
    belief = torch.randn(2, cfg.belief_tokens, cfg.width, requires_grad=True)
    obs = _observation(
        cfg,
        belief=belief,
        belief_valid=torch.ones(2, dtype=torch.bool),
        expected_contact=torch.randn(2, 8, cfg.width),
        expected_valid=torch.tensor([True, False]),
    )

    updated = module(obs)
    assert updated.shape == (2, cfg.belief_tokens, cfg.width)
    assert torch.isfinite(updated).all()
    updated.square().mean().backward()
    _assert_finite_parameter_grads(module)
    _assert_finite_input_grad(belief)


def test_invalid_expected_contact_is_never_read():
    torch.manual_seed(12)
    cfg = _config()
    module = BeliefUpdater(cfg).eval()
    belief = torch.randn(2, cfg.belief_tokens, cfg.width)
    expected_valid = torch.zeros(2, dtype=torch.bool)
    expected = torch.randn(2, 8, cfg.width, requires_grad=True)
    obs = _observation(
        cfg,
        belief=belief,
        belief_valid=torch.ones(2, dtype=torch.bool),
        expected_contact=expected,
        expected_valid=expected_valid,
    )

    actual = module(obs)
    obs.expected_contact = torch.randn(2, 8, cfg.width) * 1e4
    alternative = module(obs)
    torch.testing.assert_close(actual, alternative, rtol=0, atol=0)

    grad = torch.autograd.grad(actual.square().sum(), expected)[0]
    assert torch.count_nonzero(grad).item() == 0


def test_two_updater_steps_keep_the_bptt_gradient_to_previous_belief():
    torch.manual_seed(13)
    cfg = _config()
    module = BeliefUpdater(cfg)
    initial_belief = torch.randn(2, cfg.belief_tokens, cfg.width, requires_grad=True)
    valid = torch.ones(2, dtype=torch.bool)
    first = module(_observation(cfg, belief=initial_belief, belief_valid=valid))
    second = module(_observation(cfg, belief=first, belief_valid=valid))
    probe = torch.randn_like(second)

    (second * probe).sum().backward()

    _assert_finite_input_grad(initial_belief)


def test_future_corrector_shape_and_finite_gradients():
    torch.manual_seed(14)
    cfg = _config()
    module = FutureCorrector(cfg)
    future = torch.randn(2, 2, 3, cfg.vision_width, requires_grad=True)
    belief = torch.randn(2, cfg.belief_tokens, cfg.width, requires_grad=True)
    action = torch.randn(2, 5, cfg.world_width, requires_grad=True)

    corrected = module(future, belief, action)
    assert corrected.shape == future.shape
    assert torch.isfinite(corrected).all()
    corrected.square().mean().backward()
    _assert_finite_parameter_grads(module)
    for value in (future, belief, action):
        _assert_finite_input_grad(value)


def test_contact_predictor_shape_and_finite_gradients():
    torch.manual_seed(15)
    cfg = _config()
    module = ContactPredictor(cfg)
    future = torch.randn(2, 2, 3, cfg.vision_width, requires_grad=True)
    action = torch.randn(2, 5, cfg.world_width, requires_grad=True)
    tactile = torch.randn(2, 8, cfg.width, requires_grad=True)
    belief = torch.randn(2, cfg.belief_tokens, cfg.width, requires_grad=True)

    prediction = module(future, action, tactile, belief)
    assert prediction.shape == (2, 8, cfg.width)
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    _assert_finite_parameter_grads(module)
    for value in (future, action, tactile, belief):
        _assert_finite_input_grad(value)


def test_disabled_belief_switches_make_corrector_and_predictor_independent_of_belief():
    torch.manual_seed(16)
    cfg = _config(belief_to_future_corrector=False, belief_to_contact_predictor=False)
    corrector = FutureCorrector(cfg).eval()
    predictor = ContactPredictor(cfg).eval()
    future = torch.randn(2, 2, 3, cfg.vision_width)
    action = torch.randn(2, 5, cfg.world_width)
    tactile = torch.randn(2, 8, cfg.width)
    belief_a = torch.randn(2, cfg.belief_tokens, cfg.width, requires_grad=True)
    belief_b = torch.randn(2, cfg.belief_tokens, cfg.width) * 100

    corrected_a = corrector(future, belief_a, action)
    corrected_b = corrector(future, belief_b, action)
    prediction_a = predictor(future, action, tactile, belief_a)
    prediction_b = predictor(future, action, tactile, belief_b)
    torch.testing.assert_close(corrected_a, corrected_b, rtol=0, atol=0)
    torch.testing.assert_close(prediction_a, prediction_b, rtol=0, atol=0)

    torch.autograd.grad(corrected_a.sum() + prediction_a.sum(), belief_a, allow_unused=True)
    # Both disabled paths omit belief from their computation graph.
    assert belief_a.grad is None
