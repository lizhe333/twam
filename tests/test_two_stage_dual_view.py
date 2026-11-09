"""Two-view data, frozen visual teacher, and action-supervised belief transfer."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from test_fast_attention import tiny_architecture, cfg, inputs
from contact_belief_wam import model as cm, train, trainer, weights
from contact_belief_wam.config import ContactConfig
from contact_belief_wam.data import AlignedDataset, AlignedTaskBalancedSampler, collate_contact, prepare_aligned_batch
from contact_belief_wam.sequence import observation_at, targets_at


def dual_config(stage):
    return replace(cfg("sdpa"), aligned=True, stage=stage, independent_tactile=True,
                   camera_mode="head_wrist", stage1_freeze_world=True,
                   stage2_freeze_belief=False)


def dual_inputs():
    obs, target, noise, time = inputs()
    obs.wrist_image = torch.rand_like(obs.image)
    target.future_wrist_image = torch.rand_like(target.future_image)
    return obs, target, noise, time


def assert_finite_nonzero_grad(module):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    assert gradients and all(g.isfinite().all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_joint_stage1_updates_contact_modules_and_preserves_both_camera_world_weights(tiny_architecture):
    torch.manual_seed(22)
    policy = cm.ContactWAM(dual_config("stage1")).train()
    obs, target, _, _ = dual_inputs()
    assert policy.encode_visual(obs.image, obs.wrist_image).shape == (2, 2, 4, 32)
    before = {name: p.detach().clone() for name, p in policy.named_parameters() if not p.requires_grad}
    optimizer = trainer.make_optimizer(policy, {"stage1": {"lr": 1e-3, "weight_decay": .01}}, "stage1")
    result = policy(obs, target)
    torch.testing.assert_close(result["future_loss_per_sample"],
                               result["metrics"]["world_total"] + result["metrics"]["contact_total"])
    result["loss_per_sample"].mean().backward()
    for module in (policy.contact.predictor, policy.contact.updater, policy.contact.corrector):
        assert_finite_nonzero_grad(module)
    assert not policy.core.world_model.training
    optimizer.step()
    for name, p in policy.named_parameters():
        if name in before:
            assert p.grad is None
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)


def test_stage2_action_loss_reaches_belief_with_frozen_predictor_and_both_views(tiny_architecture):
    torch.manual_seed(23)
    policy = cm.ContactWAM(dual_config("stage2")).train()
    obs, target, noise, time = dual_inputs()
    obs.image.requires_grad_()
    obs.wrist_image.requires_grad_()
    output = policy(obs, target, noise=noise, time=time)
    output["action_loss_per_sample"].mean().backward()
    assert_finite_nonzero_grad(policy.contact.updater)
    assert_finite_nonzero_grad(policy.contact.corrector)
    assert obs.image.grad.abs().sum() > 0
    assert obs.wrist_image.grad.abs().sum() > 0
    for module in (policy.contact.predictor, policy.core.world_model,
                   policy.core.world_action_encoder, policy.core.world_state_encoder):
        assert all(not p.requires_grad and p.grad is None for p in module.parameters())
    recipe = {"stage2": {"lr": 2.5e-5, "lr_new": 1e-4, "lr_belief": 2.5e-5, "weight_decay": 1e-10}}
    optimizer = trainer.make_optimizer(policy, recipe, "stage2")
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert groups["belief"]["lr"] == 2.5e-5 and groups["new"]["lr"] == 1e-4
    assert {id(p) for p in groups["belief"]["params"]} == {id(p) for p in policy.contact.updater.parameters()}


def test_both_policy_tasks_load_same_joint_checkpoint_without_cross_task_updates(tiny_architecture, tmp_path, monkeypatch):
    stats = {key: {"q01": [-1.] * 8, "q99": [1.] * 8} for key in ("state", "actions")}
    stage1 = cm.ContactWAM(replace(dual_config("stage1"), world_norm_stats=stats)).eval()
    shared = tmp_path / "shared/pretrained_model"
    stage1.save_pretrained(shared)
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").touch()
    monkeypatch.setattr(cm.WAVE0Policy, "from_pretrained", lambda path, config, **kw: cm.WAVE0Policy(config))
    monkeypatch.setattr(cm.WAVE0Policy, "_load_frozen_vision_backbone", lambda *a, **k: None)
    config = replace(dual_config("stage2"), pi05_base=str(base), world_norm_stats=stats)
    hole, _ = weights.initialize(config, stage1_checkpoint=shared)
    with torch.no_grad():
        next(hole.contact.updater.parameters()).add_(1.)
    hdmi, _ = weights.initialize(config, stage1_checkpoint=shared)
    for name, p in hdmi.contact.state_dict().items():
        torch.testing.assert_close(p, stage1.contact.state_dict()[name], rtol=0, atol=0)
    assert hdmi.core.world_model.camera_embedding.shape[1] == 2
    exported = tmp_path / "hdmi"
    hdmi.save_pretrained(exported)
    restored = cm.ContactWAM.from_pretrained(exported).eval()
    obs, _, noise, _ = dual_inputs()
    restored.config.policy_norm_stats = restored.config.world_norm_stats = stats
    expected = restored.predict_action_chunk(obs, noise=noise)
    with pytest.raises(ValueError, match="wrist RGB"):
        restored.predict_action_chunk(replace(obs, wrist_image=None), noise=noise)
    assert expected.actions.shape == (2, 4, 8)


def test_dual_view_dataset_keeps_current_future_and_history_aligned(tmp_path):
    cv2 = pytest.importorskip("cv2")
    frames = 8
    encoded = {}
    with h5py.File(tmp_path / "episode.hdf5", "w") as handle:
        handle.create_dataset("step", data=np.arange(frames) * 2)
        handle.create_dataset("embodiment/joint", data=np.zeros((frames, 9)))
        handle.create_dataset("atom/tag", data=np.array([b"move"] * 3 + [b"perturb"] + [b"move"] * 4))
        for camera, offset in (("head", 10), ("wrist", 100)):
            images = [cv2.imencode(".jpg", np.full((16, 16, 3), offset + i * 10, np.uint8))[1] for i in range(frames)]
            node = handle.create_dataset(f"observation/{camera}/rgb", (frames,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
            for i, image in enumerate(images):
                node[i] = image
            encoded[camera] = images
    cache = tmp_path / "cache/original/insert_hole"
    cache.mkdir(parents=True)
    np.savez(cache / "0.npz", tokens=np.zeros((frames, 8, 512), np.float16), raw_step=np.arange(frames) * 2)
    row = {"task": "insert_hole", "source": "original", "split": "train", "frames": frames,
           "episode_id": "original/insert_hole/0", "path": "episode.hdf5", "segments": [[0, frames]],
           "prompt": "insert", "policy_action_excluded_ranges": [{"start_exclusive": 7, "end_inclusive": 8}]}
    stats = {key: {"q01": [-1.] * 8, "q99": [1.] * 8} for key in ("state", "actions")}
    config = ContactConfig(camera_mode="head_wrist", k=2, chunk_size=4, belief_warmup_min=1, belief_warmup_max=1)
    dataset = AlignedDataset([row], tmp_path, tmp_path / "cache", {"insert_hole": stats}, stats, config,
                             valid_future_only=False)
    try:
        sample = dataset[2]
        np.testing.assert_array_equal(sample["step"], [0, 2])
        np.testing.assert_array_equal(sample["visual_frame_index"], [0, 2])
        np.testing.assert_array_equal(sample["future_visual_frame_index"], [2, 4])
        np.testing.assert_allclose(sample["wrist_image"][:, 0, 0, 0], np.array([100, 120]) / 127.5 - 1, atol=1e-6)
        np.testing.assert_allclose(sample["future_wrist_image"][:, 0, 0, 0], np.array([120, 140]) / 127.5 - 1, atol=1e-6)
        assert sample["action_mask"].all() and sample["sequence_mask"].all()
        np.testing.assert_array_equal(sample["policy_action_mask"][-1], [False, False, True, True])
        batch = collate_contact([sample])
        assert observation_at(batch, 1).wrist_image.shape == (1, 3, 224, 224)
        assert targets_at(batch, 1).future_wrist_image.shape == (1, 3, 224, 224)
        # Cache keys must be decoded frame indices, including left padding
        # and the clipped final target, not raw simulator steps or t+k.
        first, last = dataset[0], dataset[6]
        np.testing.assert_array_equal(first["visual_frame_index"], [0, 0])
        np.testing.assert_array_equal(first["future_visual_frame_index"], [2, 2])
        np.testing.assert_array_equal(last["visual_frame_index"], [4, 6])
        np.testing.assert_array_equal(last["future_visual_frame_index"], [6, 7])
        moved = prepare_aligned_batch(batch, "meta")
        assert moved["image"].device.type == "meta"
        assert moved["visual_frame_index"].device.type == "cpu"
        assert moved["future_visual_frame_index"].device.type == "cpu"
    finally:
        dataset.close()


def test_stage1_selection_tracks_tactile_and_corrected_visual_not_frozen_base():
    class Dataset:
        tasks = ["insert_hole"] * 4
        def __getitem__(self, item):
            return {"task": "insert_hole", "source": "original"}
    class Model:
        training = True
        def eval(self): self.training = False
        def train(self, mode=True): self.training = mode
        def __call__(self, batch):
            return {"metrics": {key: {"sum": torch.full((2,), value), "count": torch.ones(2)}
                                for key, value in {"world_total": 50., "contact_total": .7,
                                                   "tactile_future": .2, "visual_corrected": .5}.items()}}
    run = {"tasks": ["insert_hole"], "stage1": {"micro_batch": 2, "validation_samples_per_task": 4,
                                               "selection_metric": "contact_total"}}
    result = trainer.validate_prediction_stage(Model(), Dataset(), lambda b, d: b, run, torch.device("cpu"), 1, 0)
    assert result["selection_metric"] == "contact_total" and result["loss"] == pytest.approx(.7)


def test_original_only_tasks_and_baseline_reuse_plan(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    run = train.resolve_config(variant="wave0_recovery")
    assert train.plan(run)["stage1"]["reuse_world_init"] == run["world_init"]
    run["policy_tasks"] = ["pull_out_key"]
    assert train.plan(run)["stage2"][0]["task"] == "pull_out_key"
    data = SimpleNamespace(tasks=["pull_out_key"] * 8, sources=["original"] * 8)
    sampler = AlignedTaskBalancedSampler(data, 4, micro_batch=4, task_balanced=False)
    assert len(list(sampler)) == 4


def test_world_reuse_rejects_camera_or_normalization_mismatch(tmp_path):
    folder = tmp_path / "pretrained_model"
    folder.mkdir()
    (folder / "model.safetensors").touch()
    config = ContactConfig(camera_mode="head_wrist", stage1_freeze_world=True, world_norm_stats={"test": 1})
    metadata = {"type": "wave0", "input_features": {"observation.images.head": {}, "observation.images.wrist": {}},
                "world_model_future_horizons": [9], "chunk_size": 50, "world_model_width": 1024,
                "world_model_depth": 12, "world_model_num_heads": 16, "world_model_mlp_dim": 4096,
                "world_model_spatial_pool_size": 4}
    (folder / "config.json").write_text(json.dumps(metadata))
    (tmp_path / "wm_norm_stats.json").write_text(json.dumps(config.world_norm_stats))
    weights.validate_world_contract(config, folder)
    with pytest.raises(ValueError, match="camera order"):
        weights.validate_world_contract(replace(config, camera_mode="head_only"), folder)
    with pytest.raises(ValueError, match="normalization"):
        weights.validate_world_contract(replace(config, world_norm_stats={"test": 2}), folder)


@pytest.mark.parametrize("variant", ["full", "wave0_recovery"])
def test_pipeline_hands_shared_teacher_to_independent_policy_jobs(monkeypatch, tmp_path, variant):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    run = train.resolve_config(variant=variant, output=tmp_path)
    stats = {key: {"q01": [-1.] * 8, "q99": [1.] * 8} for key in ("state", "actions")}
    stats_file = tmp_path / "stats.json"
    stats_file.write_text(json.dumps(stats))
    run.update(wm_norm_stats=stats, task_stats_by_task={task: str(stats_file) for task in run["policy_tasks"]},
               policy_manifests={task: "unused" for task in run["policy_tasks"]})
    monkeypatch.setattr(train, "resolve_config", lambda *a, **kw: run)
    monkeypatch.setattr(train, "_prepare", lambda run: None)
    monkeypatch.setattr(train, "_require_prepared", lambda run: None)
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    (teacher / "complete.json").write_text(json.dumps({"identity": "shared-contact"}))
    calls = []
    def stage1(value, stage, *args):
        assert variant == "full"
        calls.append(stage)
        return teacher
    monkeypatch.setattr(train, "_run_prediction_stage", stage1)
    def jobs(records, env, status, persist, **kwargs):
        import yaml
        for record in records:
            config_file = record["command"][record["command"].index("--run-config") + 1]
            config = yaml.safe_load(Path(config_file).read_text())
            assert config["task"] == record["task"]
            assert config["stage1_checkpoint"] == (str(teacher) if variant == "full" else None)
            assert config["world_init"] == run["world_init"]
            assert config["model_config"]["camera_mode"] == "head_wrist"
            assert config["model_config"]["stage2_freeze_belief"] is False
            Path(record["result"]).write_text(json.dumps({"complete": True, "identity": record["task"]}))
    monkeypatch.setattr(train, "_run_jobs", jobs)
    result = train.main(variant=variant)
    assert result["state"] == "done"
    assert calls == (["stage1"] if variant == "full" else [])


def test_simulator_adapter_sends_both_camera_frames():
    from contact_belief_wam.sim_eval import Adapter
    class Client:
        metadata = {"k": 10, "camera_mode": "head_wrist"}
        calls = []
        def call(self, **message):
            self.calls.append(message)
            return {"actions": np.zeros((50, 8), np.float32), "diagnostics": {},
                    "arm_limit_overshoots": np.zeros(50, np.float32)}
    client = Client()
    bounds = np.tile([-1., 1.], (8, 1))
    adapter = Adapter(client, "episode", "insert", action_bounds=bounds)
    head, wrist = np.zeros((16, 16, 3), np.uint8), np.ones((16, 16, 3), np.uint8)
    observation = {"observation": {"head": {"rgb": head}, "wrist": {"rgb": wrist}},
                   "embodiment": {"joint": np.zeros(9)},
                   "tactile": {f"{side}_tactile": {"rgb_marker": head} for side in ("left", "right")}}
    adapter.next_action(None, observation)
    request = client.calls[-1]
    np.testing.assert_array_equal(request["image"], head)
    np.testing.assert_array_equal(request["wrist_image"], wrist)
