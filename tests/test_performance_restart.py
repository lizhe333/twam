from copy import deepcopy
import json
from pathlib import Path

import pytest

from contact_belief_wam import train
from contact_belief_wam.performance_restart import check_compatibility, is_transfer


def identities():
    old = {"stage": "stage2", "contract": {"global_batch": 64, "micro_batch": 8,
            "world_size": 4, "stage1_checkpoint": "teacher", "lr": .000025},
           "recipe": {"config": "old.yaml", "mode": "full", "belief_bptt_steps": 2,
                      "model_config": {"belief_bptt_steps": 2}, "stage2": {"ema_device": "cuda"}},
           "source_hashes": {"src/contact_belief_wam/model.py": "old", "vendor/model.py": "pinned"},
           "teacher_sha256": "teacher", "teacher_complete": {"step": 5000}, "pi05_base_sha256": "base"}
    new = deepcopy(old)
    new["contract"].update(shard_ema=True, batch_stage2_policy=True)
    new["recipe"].update(config="fast.yaml", stage1_init_checkpoint="teacher-path",
                         stage2_restart_from="old/full", batch_stage2_policy=True)
    new["recipe"]["model_config"]["batch_stage2_policy"] = True
    new["recipe"]["stage2"]["shard_ema"] = True
    new["source_hashes"].update({"src/contact_belief_wam/model.py": "new", "src/contact_belief_wam/ema.py": "new"})
    return old, new


def test_performance_restart_accepts_only_execution_changes():
    old, new = identities()
    check_compatibility(old, new)


@pytest.mark.parametrize("change", ["batch", "bptt", "teacher", "data", "source", "ema_device"])
def test_performance_restart_rejects_changed_training(change):
    old, new = identities()
    if change == "batch": new["contract"]["micro_batch"] = 16
    if change == "bptt": new["recipe"]["belief_bptt_steps"] = 1
    if change == "teacher": new["teacher_sha256"] = "different"
    if change == "data": new["contract"]["manifest_sha256"] = "different"
    if change == "source": new["source_hashes"]["vendor/model.py"] = "different"
    if change == "ema_device": new["recipe"]["stage2"]["ema_device"] = "cpu"
    with pytest.raises(ValueError):
        check_compatibility(old, new)


def test_restart_uses_latest_complete_source_then_prefers_own_checkpoint(tmp_path):
    source = tmp_path / "old/full"
    output = tmp_path / "new/full/tasks/insert_hole"
    run = {"output": str(output), "task": "insert_hole", "stage2_restart_from": str(source)}
    def checkpoint(root, step):
        path = root / f"stage2/checkpoints/step_{step:08d}"
        path.mkdir(parents=True)
        (path / "complete.json").write_text(json.dumps({"step": step}))
        return path
    previous = checkpoint(source / "tasks/insert_hole", 3000)
    (previous.parent / "step_00004000.partial").mkdir()
    assert train._find_resume(run, "stage2", "auto") == previous
    assert is_transfer(run, previous)
    own = checkpoint(output, 4000)
    assert train._find_resume(run, "stage2", "auto") == own
    assert not is_transfer(run, own)
    assert train._find_resume(run, "stage2", "none") is None


def test_fast_recipe_preserves_batch_bptt_and_gpu_ema():
    old = train.resolve_config("configs/aligned_8xa800_expert_only.yaml")
    fast = train.resolve_config("configs/aligned_8xa800_expert_only_fast.yaml")
    assert fast["batch_stage2_policy"] and fast["stage2"]["shard_ema"]
    assert fast["stage2"]["ema_device"] == "cuda"
    assert fast["stage1"] == old["stage1"]
    assert {k: v for k, v in fast["stage2"].items() if k != "shard_ema"} == old["stage2"]
    assert fast["belief_bptt_steps"] == old["belief_bptt_steps"] == 2
    assert fast["policy_tasks"] == old["policy_tasks"]


def rebatch_identities():
    old, new = identities()
    for identity in (old, new):
        identity["contract"].update(expert_only=True, data_mix="1:0", accumulation_steps=2)
        identity["recipe"]["stage2"].update(num_gpus=4, micro_batch=8, global_batch=64,
                                           accumulation_steps=2)
    new["contract"].update(micro_batch=16, accumulation_steps=1)
    new["recipe"]["stage2"].update(micro_batch=16, accumulation_steps=1)
    new["recipe"]["stage2_restart_rebatch"] = True
    return old, new


def test_explicit_rebatch_preserves_the_optimizer_batch():
    old, new = rebatch_identities()
    check_compatibility(old, new)
    del new["recipe"]["stage2_restart_rebatch"]
    with pytest.raises(ValueError, match="batch contract"):
        check_compatibility(old, new)


@pytest.mark.parametrize("change", ["global_batch", "ranks", "accumulation", "recipe", "expert_only", "lr", "bptt"])
def test_rebatch_rejects_changes_beyond_batch_grouping(change):
    old, new = rebatch_identities()
    if change == "global_batch":
        new["contract"].update(global_batch=128, accumulation_steps=2)
        new["recipe"]["stage2"].update(global_batch=128, accumulation_steps=2)
    if change == "ranks":
        new["contract"].update(world_size=2, accumulation_steps=2)
        new["recipe"]["stage2"].update(num_gpus=2, accumulation_steps=2)
    if change == "accumulation": new["contract"]["accumulation_steps"] = 2
    if change == "recipe": new["recipe"]["stage2"]["micro_batch"] = 8
    if change == "expert_only": new["contract"]["expert_only"] = False
    if change == "lr": new["contract"]["lr"] *= 2
    if change == "bptt": new["recipe"]["model_config"]["belief_bptt_steps"] = 1
    with pytest.raises(ValueError):
        check_compatibility(old, new)


def test_restart_does_not_silently_discard_progress_before_first_checkpoint(tmp_path):
    source = tmp_path / "old/full"
    run = {"output": str(tmp_path / "new/full/tasks/insert_hole"), "task": "insert_hole",
           "stage2_restart_from": str(source)}
    with pytest.raises(FileNotFoundError, match="source does not exist"):
        train._find_resume(run, "stage2", "auto")
    task_dir = source / "tasks/insert_hole"
    task_dir.mkdir(parents=True)
    # Scheduled tasks with only a config may still start normally.
    (task_dir / "run_config.yaml").write_text("task: insert_hole\n")
    assert train._find_resume(run, "stage2", "auto") is None
    (task_dir / "train.jsonl").write_text('{"step": 4900}\n')
    partial = task_dir / "stage2/checkpoints/step_00005000.partial"
    partial.mkdir(parents=True)
    with pytest.raises(ValueError, match="no complete checkpoint"):
        train._find_resume(run, "stage2", "auto")
    complete = partial.with_suffix("")
    partial.rename(complete)
    (complete / "complete.json").write_text('{"step": 5000}')
    assert train._find_resume(run, "stage2", "auto") == complete


def test_mb16_recipe_keeps_budget_gpu_ema_and_two_step_bptt(monkeypatch):
    old = train.resolve_config("configs/aligned_8xa800_expert_only_fast.yaml")
    new = train.resolve_config("configs/aligned_8xa800_expert_only_fast_mb16.yaml")
    # fast-002 stopped before saving; the preceding run still owns complete state.
    assert new["stage2_restart_from"] == old["stage2_restart_from"]
    assert new["stage2_restart_rebatch"] is True
    assert new["stage1"] == old["stage1"]
    assert (new["stage2"]["micro_batch"], new["stage2"]["accumulation_steps"],
            new["stage2"]["global_batch"]) == (16, 1, 64)
    assert {k: v for k, v in old["stage2"].items() if k not in {"micro_batch", "accumulation_steps"}} == {
        k: v for k, v in new["stage2"].items() if k not in {"micro_batch", "accumulation_steps"}}
    assert new["model_config"] == old["model_config"]
    assert new["belief_bptt_steps"] == 2 and new["stage2"]["ema_device"] == "cuda"
    assert new["stage2"]["shard_ema"] and new["stage2"]["steps"] == 10000
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    new["stage1_init_checkpoint"] = "existing-stage1"
    schedule = train.plan(new)
    assert len(schedule["stage2"]) == 8
    assert all(job["nproc_per_node"] == 4 for job in schedule["stage2"])


def test_rebatch_keeps_global_expert_samples_after_resume():
    from types import SimpleNamespace
    from contact_belief_wam.data import AlignedTaskBalancedSampler
    data = SimpleNamespace(tasks=["insert_hole"] * 137, sources=["original"] * 137)
    start = 5000 * 16
    samples = {}
    for micro in (8, 16):
        ranks = [list(AlignedTaskBalancedSampler(data, 48, rank=rank, world_size=4,
                      start_draw=start, micro_batch=micro, task_balanced=False, source_ratio=(1, 0)))
                 for rank in range(4)]
        samples[micro] = [sorted(index for rows in ranks for index, seed in rows[step * 16:(step + 1) * 16])
                          for step in range(3)]
    assert samples[8] == samples[16]
