from dataclasses import replace
import pytest
import torch

from test_fast_attention import tiny_architecture, cfg, inputs
from contact_belief_wam import model as cm, train, trainer
from contact_belief_wam.modules import TactileWorldModel
from contact_belief_wam.weights import load_tactile_predictor


def configuration(stage="tactile"):
    return replace(cfg("sdpa"), aligned=True, stage=stage, independent_tactile=True,
                   belief_warmup_min=0, belief_warmup_max=0, belief_bptt_steps=1)


def test_independent_dynamics_uses_action_order_state_and_touch():
    torch.manual_seed(3)
    config = configuration()
    module = TactileWorldModel(config).eval()
    visual = torch.randn(2, 1, 4, config.vision_width)
    actions, state, touch = torch.randn(2, 2, 8), torch.randn(2, 8), torch.randn(2, 8, config.width)
    reference = module(visual, actions, state, touch)
    assert reference.shape == touch.shape
    for other in (module(visual, actions.flip(1), state, touch),
                  module(visual, actions + 1, state, touch),
                  module(visual, actions, state + 1, touch),
                  module(visual, actions, state, touch + 1)):
        assert not torch.allclose(reference, other)


def test_tactile_stage_bypasses_belief_and_visual_prediction_and_only_updates_predictor(tiny_architecture, monkeypatch):
    torch.manual_seed(4)
    policy = cm.ContactWAM(configuration()).train()
    obs, target, _, _ = inputs()
    target.future_image = None  # Future RGB is not a tactile-training input.
    target.future_tactile.requires_grad_()
    for module in (policy.contact.updater, policy.contact.corrector,
                   policy.core.world_model, policy.core.world_action_encoder, policy.core.world_state_encoder):
        monkeypatch.setattr(module, "forward", lambda *a, **k: pytest.fail("unrelated module ran"))
    frozen = {n: p.detach().clone() for n, p in policy.named_parameters() if not p.requires_grad}
    recipe = {"tactile": {"lr": 1e-3, "weight_decay": .01}, "fused_adamw": False}
    optimizer = trainer.make_optimizer(policy, recipe, "tactile", device="cpu")
    result = policy(obs, target)
    result["loss_per_sample"].mean().backward()
    assert target.future_tactile.grad is None
    for name, parameter in policy.named_parameters():
        if name.startswith("contact.predictor."):
            assert parameter.requires_grad and parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        else:
            assert not parameter.requires_grad and parameter.grad is None
    optimizer.step()
    for name, parameter in policy.named_parameters():
        if name in frozen:
            torch.testing.assert_close(parameter, frozen[name], atol=0, rtol=0)
    torch.testing.assert_close(result["metrics"]["tactile_persistence"],
                               cm.feature_distance(obs.tactile, target.future_tactile))


def test_tactile_checkpoint_transfer_excludes_belief_and_roundtrips(tiny_architecture, tmp_path):
    torch.manual_seed(5)
    source = cm.ContactWAM(configuration()).eval()
    obs, target, _, _ = inputs()
    expected = source(obs, target)["expected_contact"]
    source.save_pretrained(tmp_path)
    restored = cm.ContactWAM.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(restored(obs, target)["expected_contact"], expected)
    destination = cm.ContactWAM(configuration("stage1"))
    prior = {n: p.detach().clone() for n, p in destination.contact.updater.named_parameters()}
    load_tactile_predictor(destination, tmp_path)
    assert all(not p.requires_grad for p in destination.contact.predictor.parameters())
    for name, parameter in destination.contact.updater.named_parameters():
        torch.testing.assert_close(parameter, prior[name], atol=0, rtol=0)
    for name, parameter in destination.contact.predictor.named_parameters():
        torch.testing.assert_close(parameter, dict(source.contact.predictor.named_parameters())[name])
    destination.config.world_norm_stats = {"different": True}
    with pytest.raises(ValueError, match="world_norm_stats"):
        load_tactile_predictor(destination, tmp_path)
    with pytest.raises(ValueError, match="horizon"):
        cm.ContactWAM.from_pretrained(tmp_path, k=3)


def test_independent_prediction_does_not_depend_on_corrected_future_or_belief(tiny_architecture):
    policy = cm.ContactWAM(configuration("stage1")).eval()
    obs, target, _, _ = inputs()
    current = policy.encode_visual(obs.image)
    latent = policy.core.world_action_encoder(cm.pad8(target.actions[:, :2]), cm.pad8(obs.state))
    first = policy.predict_future(current, obs.state, latent, torch.zeros(2, 4, 32), obs.tactile,
                                  actions=target.actions)[2]
    second = policy.predict_future(current, obs.state, latent + 50, torch.ones(2, 4, 32) * 20,
                                   obs.tactile, actions=target.actions)[2]
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_independent_inference_predicts_the_executed_action_prefix(tiny_architecture):
    config = configuration("stage2")
    stats = {name: {"q01": [-1.] * 8, "q99": [1.] * 8} for name in ("state", "actions")}
    config.policy_norm_stats = config.world_norm_stats = stats
    policy = cm.ContactWAM(config).eval()
    obs, _, noise, _ = inputs()
    prediction = policy.predict_action_chunk(obs, noise=noise, action_transform=lambda a: a.clamp(-.1, .1))
    actions = policy.to_world_actions(cm.pad8(prediction.actions[:, :config.k]))
    expected = policy.contact.predictor(policy.encode_visual(obs.image), actions, obs.state, obs.tactile)
    torch.testing.assert_close(prediction.expected_contact, expected)
    assert prediction.actions.abs().max() <= .1


def test_tactile_stage_selects_checkpoint_by_tactile_error(monkeypatch):
    class Dataset:
        tasks = ["insert_hole"] * 4

        def __getitem__(self, item):
            return {"task": "insert_hole", "source": "original"}

    class Model:
        training = True
        def eval(self): self.training = False
        def train(self, mode=True): self.training = mode
        def __call__(self, batch):
            return {"metrics": {name: {"sum": torch.full((2,), value), "count": torch.ones(2)}
                                for name, value in {"tactile_future": .25, "tactile_persistence": .75}.items()}}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    run = {"tasks": ["insert_hole"], "tactile": {"validation_samples_per_task": 4, "micro_batch": 2}}
    result = trainer.validate_prediction_stage(Model(), Dataset(), lambda batch, device: batch, run,
                                     torch.device("cpu"), 1, 0, stage="tactile")
    assert result["selection_metric"] == "tactile_future"
    assert result["loss"] == pytest.approx(.25)
    assert result["metrics"]["tactile_persistence"] == pytest.approx(.75)


def test_stage1_refuses_random_frozen_tactile_weights(monkeypatch):
    run = {"model_config": {"independent_tactile": True}, "mode": "full", "world_init": "unused"}
    with pytest.raises(ValueError, match="tactile_checkpoint"):
        trainer._load_model(run, "stage1")


def test_plan_and_command_expose_independent_stage(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "_physical_gpus", lambda: list(map(str, range(8))))
    run = {"config": "recipe", "variant": "full", "mode": "full", "workspace": str(tmp_path),
           "output": str(tmp_path), "independent_tactile": True, "tactile": {"num_gpus": 2}}
    assert train.plan(run)["tactile"]["gpus"] == ["0", "1"]
    run["mode"] = "wave0"
    assert "tactile" not in train.plan(run)
    command = train._training_command("run.yaml", "tactile", nproc=1)
    assert command[command.index("--nproc_per_node") + 1] == "1"
    assert command[command.index("--stage") + 1] == "tactile"


def test_tactile_can_run_alone_on_two_gpus(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    run = {"config": "recipe", "variant": "full", "mode": "full", "workspace": str(tmp_path),
           "output": str(tmp_path), "independent_tactile": True, "tactile": {"num_gpus": 2}}
    result = train.plan(run, stop_after="tactile")
    assert result["tactile"]["gpus"] == ["3", "5"]
    with pytest.raises(ValueError, match="8 visible GPUs"):
        train.plan(run)


def test_workflow_passes_tactile_checkpoint_then_can_stop_before_world_training(monkeypatch, tmp_path):
    run = {"config": "recipe", "variant": "full", "mode": "full", "workspace": str(tmp_path),
           "output": str(tmp_path / "run"), "independent_tactile": True,
           "world_init": "/unused/world", "tactile": {"num_gpus": 2}}
    monkeypatch.setattr(train, "resolve_config", lambda *a, **k: run.copy())
    monkeypatch.setattr(train, "_physical_gpus", lambda: list(map(str, range(8))))
    monkeypatch.setattr(train, "_prepare", lambda run: None)
    monkeypatch.setattr(train, "_require_prepared", lambda run: None)
    monkeypatch.setattr(train, "_pipeline_env", lambda threads: {})
    calls = []
    checkpoint = tmp_path / "tactile_checkpoint"
    def prediction_stage(value, stage, *args):
        calls.append(stage)
        if stage == "stage1":
            assert value["tactile_checkpoint"] == str(checkpoint)
        return checkpoint
    monkeypatch.setattr(train, "_run_prediction_stage", prediction_stage)
    monkeypatch.setattr(train, "_run_jobs", lambda *a, **k: pytest.fail("policy jobs started"))
    result = train.main(stop_after="tactile")
    assert calls == ["tactile"]
    assert result["state"] == "tactile_complete"
    calls.clear()
    result = train.main(resume="auto", stop_after="stage1")
    assert calls == ["tactile", "stage1"]
    assert result["state"] == "stage1_complete"


def test_sequence_tactile_phase_never_warms_up_belief_and_masks_wrong_actions():
    from contact_belief_wam.sequence import SequenceTrainingModel
    from test_sequence import TinyPolicy, _batch
    class TactileProbe(TinyPolicy):
        def forward(self, observation, targets):
            result = super().forward(observation, targets)
            result["metrics"]["tactile_wrong_action"] = torch.ones(1)
            result["metric_masks"] = {"tactile_wrong_action": torch.zeros(1, dtype=torch.bool)}
            return result
    policy = TactileProbe(3, bptt_steps=1)
    policy.config.stage = "tactile"
    result = SequenceTrainingModel(policy)(_batch([[True, True, True]]))
    assert policy.history_calls == []
    assert len(policy.forward_calls) == 3
    assert result["metrics"]["tactile_wrong_action"]["count"].sum() == 0


def test_public_train_entry_reports_plan_and_configuration_errors(monkeypatch, capsys):
    from contact_belief_wam.cli import main
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert main(["train", "--plan", "--stop-after", "tactile"]) == 0
    assert '"tactile"' in capsys.readouterr().out
    assert main(["train", "--plan", "--stop-after", "tactile", "--variant", "wave0_recovery"]) == 1
    assert "requires full mode" in capsys.readouterr().err
