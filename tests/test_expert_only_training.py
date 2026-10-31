"""Original-only data isolation and the head-only two-stage training path."""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from contact_belief_wam import model, train, trainer
from contact_belief_wam.data import (
    AlignedTaskBalancedSampler, SHARED_TASKS, VALIDATION_EPISODE_IDS, prepare_data,
)
from contact_belief_wam.paths import read_jsonl, write_jsonl
from test_fast_attention import cfg, inputs, tiny_architecture
from test_stage2_restart import teacher_run


RECIPE = "configs/aligned_8xa800_expert_only.yaml"


@pytest.fixture
def source_data(tmp_path):
    def build(recoveries=50, materialize_recovery=False):
        root, baseline = tmp_path / "data", tmp_path / "baseline"
        stats = {key: {"q01": [-1.] * 8, "q99": [1.] * 8} for key in ("state", "actions")}
        for task in SHARED_TASKS:
            directory = baseline / task
            directory.mkdir(parents=True)
            (directory / "segments.json").write_text(json.dumps([
                {"source_episode_id": i, "start_frame": 0, "stop_frame": 40} for i in range(100)
            ]))
            (directory / "norm_stats.json").write_text(json.dumps(stats))
            (directory / "instruction.json").write_text(json.dumps({"seen": [f"do {task}"]}))
            rows = []
            for source, count in (("original", 100), ("recovery", recoveries if task in train.POLICY_TASKS else 0)):
                for i in range(count):
                    relative = f"raw/{source}/{task}/{i}.hdf5"
                    if source == "original" or materialize_recovery:
                        raw = root / relative
                        raw.parent.mkdir(parents=True, exist_ok=True)
                        raw.write_bytes(b"data preparation only fingerprints raw files")
                    rows.append({"episode_id": f"{source}/{task}/{i}", "task": task,
                                 "source": source, "path": relative, "split": "train",
                                 "segments": [[0, 40]], "frames": 40})
            write_jsonl(root / "manifests" / f"{task}.jsonl", rows)
        wm_stats, tokenizer = tmp_path / "wm.json", tmp_path / "tokenizer.model"
        wm_stats.write_text(json.dumps(stats))
        tokenizer.touch()
        return {"expert_only": True, "tasks": list(SHARED_TASKS), "workspace": str(tmp_path),
                "data_root": str(root), "baseline_dir": str(baseline),
                "reference_wm_stats": str(wm_stats), "tokenizer_path": str(tokenizer)}
    return build


@pytest.mark.parametrize("recoveries", [0, 7, 50])
@pytest.mark.parametrize("validate_every", [0, 500])
def test_expert_preparation_excludes_recovery_even_when_its_files_are_missing(source_data, recoveries, validate_every):
    run = source_data(recoveries)
    run["stage1"] = {"validate_every": validate_every}
    holdouts = set(VALIDATION_EPISODE_IDS) if validate_every else set()
    originals = {p: p.read_bytes() for p in (Path(run["data_root"]) / "manifests").glob("*.jsonl")}
    prepared = prepare_data(run)
    shared = read_jsonl(prepared["shared_manifest"])
    expected = {("original", "train"): 800 - 8 * len(holdouts)}
    if holdouts:
        expected["original", "val"] = 8 * len(holdouts)
    assert Counter((r["source"], r["split"]) for r in shared) == expected
    assert set(prepared["validation_episode_ids"]) == holdouts
    assert prepared["summary"]["stage1_train_recovery"] == 0
    for task in SHARED_TASKS:
        held_out = {int(r["episode_id"].rsplit("/", 1)[-1]) for r in shared
                    if r["task"] == task and r["split"] == "val"}
        assert held_out == holdouts
        rows = read_jsonl(prepared["policy_manifests"][task])
        assert len(rows) == 100
        assert {(r["task"], r["source"], r["split"]) for r in rows} == {(task, "original", "train")}
    assert all(p.read_bytes() == data for p, data in originals.items())
    assert prepare_data(run)["episode_fingerprint"] == prepared["episode_fingerprint"]


def test_mixed_preparation_and_its_files_are_preserved(source_data):
    run = source_data(materialize_recovery=True)
    mixed = prepare_data({**run, "expert_only": False})
    protected = [Path(mixed["shared_manifest"]), Path(mixed["preparation"])]
    protected += [Path(p) for p in mixed["policy_manifests"].values()]
    before = {p: p.read_bytes() for p in protected}
    assert mixed["summary"]["stage1_train_recovery"] == 100
    expert = prepare_data(run)
    assert expert["shared_manifest"] != mixed["shared_manifest"]
    assert expert["wm_stats"] != mixed["wm_stats"]
    with pytest.raises(ValueError, match="separate.*aligned_manifest_dir"):
        prepare_data({**run, "aligned_manifest_dir": mixed["aligned_manifest_dir"]})
    assert all(p.read_bytes() == value for p, value in before.items())


def test_mixed_preparation_still_requires_recovery(source_data):
    run = source_data(recoveries=0)
    with pytest.raises(ValueError, match="50 recovery"):
        prepare_data({**run, "expert_only": False})


def test_head_only_recipe_keeps_two_stage_budget_and_starts_new_stage1(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    expert, mixed = train.resolve_config(RECIPE), train.resolve_config("configs/aligned_8xa800.yaml")
    assert expert["camera_mode"] == expert["model_config"]["camera_mode"] == "head_only"
    assert expert["expert_only"] and expert["data_mix"] == "1:0"
    validation_keys = {"validate_every", "selection_metric", "validation_samples_per_task",
                       "early_stop_patience", "early_stop_min_steps"}
    assert {k: v for k, v in expert["stage1"].items() if k not in validation_keys} == {
        k: v for k, v in mixed["stage1"].items() if k not in validation_keys}
    assert expert["stage2"] == mixed["stage2"]
    assert expert["stage1"]["validate_every"] == 0 and expert["wm_validation_episode_ids"] == []
    for key in ("output", "shared_manifest", "policy_manifest_dir", "wm_stats", "world_init"):
        assert expert[key] != mixed[key]
    assert expert["stage1_freeze_world"] and not expert["stage2_freeze_belief"]
    assert expert["stage2_freeze_contact_predictor"] and expert["independent_tactile"]
    plan = train.plan(expert)
    assert plan["stage1"]["nproc_per_node"] == 8
    assert "reuse_checkpoint" not in plan["stage1"] and "reuse_world_init" not in plan["stage1"]
    assert [p["task"] for p in plan["stage2"]] == list(SHARED_TASKS)
    assert [p["gpus"] for p in plan["stage2"]] == [["0", "1", "2", "3"], ["4", "5", "6", "7"]] * 4


def test_expert_config_rejects_recovery_sampling(tmp_path):
    raw = yaml.safe_load((Path(__file__).parents[1] / RECIPE).read_text())
    raw["data_mix"] = "1:1"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="expert_only requires data_mix"):
        train.resolve_config(path)


def test_disabled_validation_cannot_silently_leave_expert_episodes_out(source_data, tmp_path):
    run = source_data()
    run.update(stage1={"validate_every": 0}, wm_validation_episode_ids=list(VALIDATION_EPISODE_IDS))
    with pytest.raises(ValueError, match="wm_validation_episode_ids"):
        prepare_data(run)
    raw = yaml.safe_load((Path(__file__).parents[1] / RECIPE).read_text())
    raw["wm_validation_episode_ids"] = list(VALIDATION_EPISODE_IDS)
    path = tmp_path / "bad-holdout.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="wm_validation_episode_ids"):
        train.resolve_config(path)


@pytest.mark.parametrize("stage,validation", [("stage1", False), ("stage1", True), ("stage2", False)])
def test_expert_trainer_rejects_a_mixed_manifest(tmp_path, stage, validation):
    path = tmp_path / "mixed.jsonl"
    write_jsonl(path, [{"source": "original"}, {"source": "recovery"}])
    run = train.resolve_config(RECIPE)
    run.update(shared_manifest=str(path), task_manifest=str(path), task_stats="unused")
    with pytest.raises(ValueError, match="Original-only manifest"):
        trainer._load_dataset(run, stage, validation=validation)


@pytest.mark.parametrize("stage,micro,ranks", [("stage1", 32, 8), ("stage2", 8, 4)])
def test_expert_sampler_balances_tasks_and_replays_on_each_rank(stage, micro, ranks):
    tasks = SHARED_TASKS if stage == "stage1" else ("insert_hole",)
    data = SimpleNamespace(tasks=[task for task in tasks for _ in range(100)],
                           sources=["original"] * (100 * len(tasks)))
    selections = []
    for rank in range(ranks):
        kwargs = dict(micro_batch=micro, task_balanced=stage == "stage1", source_ratio=(1, 0),
                      world_size=ranks, rank=rank, seed=42)
        draws = list(AlignedTaskBalancedSampler(data, micro * 2, **kwargs))
        assert draws[micro:] == list(AlignedTaskBalancedSampler(data, micro, start_draw=micro, **kwargs))
        assert Counter(data.tasks[i] for i, _ in draws[:micro]) == {t: micro // len(tasks) for t in tasks}
        assert all(data.sources[i] == "original" for i, _ in draws)
        selected = {i for i, _ in draws[:micro]}
        assert all(selected.isdisjoint(other) for other in selections)
        selections.append(selected)


def test_expert_stage2_rejects_unknown_or_recovery_teacher_provenance(teacher_run):
    teacher, run = teacher_run
    run["expert_only"] = True
    path = teacher / "run_identity.json"
    identity = json.loads(path.read_text())
    for contract in ({}, {"expert_only": False}):
        path.write_text(json.dumps({**identity, "contract": contract}))
        with pytest.raises(ValueError, match="expert-only Stage-1 checkpoint"):
            train._verify_stage1_initialization(teacher, run)
    path.write_text(json.dumps({**identity, "contract": {"expert_only": True}}))
    assert train._verify_stage1_initialization(teacher, run) == teacher


def test_expert_pipeline_trains_stage1_then_hands_it_to_eight_independent_policies(source_data, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    run = train.resolve_config(RECIPE, output=tmp_path / "new-run")
    for key in ("aligned_manifest_dir", "policy_manifest_dir", "task_stats_dir", "wm_stats"):
        run.pop(key)
    run.update(source_data())
    monkeypatch.setattr(train, "resolve_config", lambda *a, **kw: run)
    monkeypatch.setattr(train, "_require_prepared", lambda r: None)
    teacher = Path(run["output"]) / "shared_wm/stage1/checkpoints/step_00005000"
    calls = []
    def stage1(value, stage, placement, *args):
        calls.append(stage)
        assert placement["nproc_per_node"] == 8
        assert value["stage1_checkpoint"] is None and not value.get("stage1_init_checkpoint")
        shared = read_jsonl(value["shared_manifest"])
        assert len(shared) == 800 and {r["split"] for r in shared} == {"train"}
        contract = trainer.training_contract(train.stage_run(value, stage), stage, 8)
        assert contract["expert_only"] is True and contract["data_mix"] == "1:0"
        assert contract["validate_every"] == 0 and contract["checkpoint_selection"] == "final"
        assert "early_stop_patience" not in contract
        teacher.mkdir(parents=True)
        (teacher / "complete.json").write_text(json.dumps({"identity": "new-expert-stage1"}))
        return teacher
    monkeypatch.setattr(train, "_run_prediction_stage", stage1)
    def jobs(records, env, status, persist, **kwargs):
        for record in records:
            command = record["command"]
            value = yaml.safe_load(Path(command[command.index("--run-config") + 1]).read_text())
            calls.append(record["task"])
            assert value["stage1_checkpoint"] == str(teacher)
            assert value["stage1_checkpoint_identity"] == "new-expert-stage1"
            assert value["model_config"]["camera_mode"] == "head_only"
            assert value["stage2"]["steps"] == 10000
            assert value["output"] == str(Path(run["output"]) / "tasks" / record["task"])
            rows = read_jsonl(value["task_manifest"])
            assert len(rows) == 100 and {(r["task"], r["source"]) for r in rows} == {(record["task"], "original")}
            Path(record["result"]).write_text(json.dumps({"complete": True, "identity": record["task"]}))
    monkeypatch.setattr(train, "_run_jobs", jobs)
    assert train.main(resume="auto")["state"] == "done"
    assert calls == ["stage1", *SHARED_TASKS]


def test_eight_policy_jobs_wait_for_their_gpu_group(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    records = train.plan(train.resolve_config(RECIPE))["stage2"]
    live, launched = [], []
    class Process:
        def __init__(self, record):
            self.pid = 10000 + len(launched)
            self.gpus = set(record["gpus"])
            self.remaining = 6 if record["task"] == "insert_hole" else 1
            self.active = True
        def poll(self):
            self.remaining -= 1
            if self.remaining > 0:
                return None
            self.active = False
            return 0
    def spawn(record, env):
        active = [p for p in live if p.active]
        assert len(active) < 2
        assert all(p.gpus.isdisjoint(record["gpus"]) for p in active)
        launched.append(record["task"])
        process = Process(record)
        live.append(process)
        return process
    monkeypatch.setattr(train, "_spawn", spawn)
    monkeypatch.setattr(train, "_stop_processes", lambda processes: None)
    monkeypatch.setattr(train.time, "sleep", lambda seconds: None)
    train._run_jobs(records, {}, {}, lambda: None, concurrent=2)
    assert launched[:3] == ["insert_hole", "insert_HDMI", "pull_out_key"]
    assert set(launched) == set(SHARED_TASKS) and len(launched) == 8
    assert all(r["state"] == "done" for r in records)


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_head_only_contact_gradients_without_wrist_input(tiny_architecture, stage):
    torch.manual_seed(26)
    config = replace(cfg("sdpa"), aligned=True, stage=stage, camera_mode="head_only",
                     independent_tactile=True, stage1_freeze_world=True, stage2_freeze_belief=False)
    policy = model.ContactWAM(config).train()
    obs, target, noise, time = inputs()
    assert obs.wrist_image is None and target.future_wrist_image is None
    assert policy.encode_visual(obs.image).shape == (2, 1, 4, 32)
    result = policy(obs, target, noise=noise, time=time)
    result["loss_per_sample" if stage == "stage1" else "action_loss_per_sample"].mean().backward()
    modules = [policy.contact.updater, policy.contact.corrector]
    if stage == "stage1":
        modules.append(policy.contact.predictor)
    else:
        assert all(not p.requires_grad and p.grad is None for p in policy.contact.predictor.parameters())
    for module in modules:
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(g.isfinite().all() for g in gradients)
        assert sum(g.abs().sum().item() for g in gradients) > 0
    for module in (policy.core.world_model, policy.core.world_action_encoder, policy.core.world_state_encoder):
        assert all(not p.requires_grad and p.grad is None for p in module.parameters())
