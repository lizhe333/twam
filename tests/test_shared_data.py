import json
from types import SimpleNamespace

import h5py
import numpy as np

from contact_belief_wam.config import ContactConfig
from contact_belief_wam.data import ContactDataset, MixedSequenceSampler, _blocked_hold_ranges
from contact_belief_wam.paths import write_jsonl
from contact_belief_wam.shared_data import SHARED_TASKS, prepare_shared_data


def _write_episode(path, frames=4, joint_value=0.0, tags=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = 2 * np.arange(frames, dtype=np.int64)
    joints = np.full((frames, 9), joint_value, dtype=np.float32)
    joints[:, :8] += np.arange(frames, dtype=np.float32)[:, None]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("step", data=steps)
        handle.require_group("embodiment").create_dataset("joint", data=joints)
        handle.require_group("observation/head").create_dataset(
            "rgb", data=np.zeros((frames, 2, 2, 3), dtype=np.uint8),
        )
        for side in ("left", "right"):
            handle.require_group(f"tactile/{side}_gsmini").create_dataset(
                "rgb_marker", data=np.zeros((frames, 2, 2, 3), dtype=np.uint8),
            )
        if tags is not None:
            handle.require_group("atom").create_dataset("tag", data=np.asarray(tags, dtype="S16"))
    return steps


def test_prepare_shared_data_builds_all_manifests_and_train_only_stats(tmp_path, monkeypatch):
    import contact_belief_wam.data as data_module

    workspace = tmp_path / "workspace"
    instructions = workspace / "UniVTAC/instructions"
    instructions.mkdir(parents=True)
    for task in SHARED_TASKS:
        (instructions / f"{task}.json").write_text(json.dumps({"seen": [f"do {task}"]}))
    monkeypatch.setattr(data_module, "WORKSPACE", workspace)

    original_root = tmp_path / "raw"
    for index, task in enumerate(SHARED_TASKS):
        value = 1000 if task == "insert_hole" else index * 10
        _write_episode(original_root / task / "hdf5/0.hdf5", joint_value=value)
    _write_episode(original_root / "insert_hole/hdf5/1.hdf5", joint_value=0)

    data_root = tmp_path / "prepared"
    old_rows = [
        {"episode_id": "original/insert_hole/0", "split": "val"},
        {"episode_id": "original/insert_hole/1", "split": "train"},
    ]
    write_jsonl(data_root / "manifests/insert_hole.jsonl", old_rows)
    result = prepare_shared_data(data_root, original_root=original_root, recovery_root=tmp_path / "recovery")

    assert set(result["tasks"]) == set(SHARED_TASKS)
    assert result["shared"]["episodes"] == len(SHARED_TASKS) + 1
    assert result["shared"]["train_episodes"] == len(SHARED_TASKS)
    rows = [json.loads(line) for line in (data_root / "manifests/shared.jsonl").read_text().splitlines()]
    assert {row["task"] for row in rows} == set(SHARED_TASKS)
    assert rows[0]["split"] == "val"  # Existing task split remains unchanged.

    stats = json.loads((data_root / "stats/shared.quantiles.json").read_text())
    assert stats["metadata"]["split"] == "train"
    assert stats["metadata"]["tasks"] == list(SHARED_TASKS)
    # The held-out insert_hole episode has joint values above all train rows.
    assert max(stats["state"]["q99"]) < 100
    assert max(stats["action"]["q99"]) < 100

    # The explicit all-train mode creates separate artifacts and includes the
    # old validation episode in the shared quantiles without rewriting old ones.
    preserved_paths = [
        data_root / "manifests" / "shared.jsonl",
        data_root / "stats" / "shared.quantiles.json",
        *(data_root / "manifests" / f"{task}.jsonl" for task in SHARED_TASKS),
        *(data_root / "stats" / f"{task}.quantiles.json" for task in SHARED_TASKS),
    ]
    preserved_bytes = {path: path.read_bytes() for path in preserved_paths}
    all_train_result = prepare_shared_data(
        data_root,
        original_root=original_root,
        recovery_root=tmp_path / "recovery",
        train_on_all_data=True,
    )

    all_train_manifest = data_root / "manifests/all_train/shared.jsonl"
    all_train_stats_path = data_root / "stats/shared.all_train.quantiles.json"
    assert all_train_result["shared"]["manifest"] == str(all_train_manifest)
    assert all_train_result["shared"]["stats"] == str(all_train_stats_path)
    assert all_train_result["shared"]["train_episodes"] == len(SHARED_TASKS) + 1
    assert all_train_result["shared"]["val_episodes"] == 0
    all_rows = [json.loads(line) for line in all_train_manifest.read_text().splitlines()]
    assert all(row["split"] == "train" for row in all_rows)
    val_episode = next(row for row in all_rows if row["episode_id"] == "original/insert_hole/0")
    assert val_episode["split"] == "train"
    for task in SHARED_TASKS:
        task_rows = [json.loads(line) for line in
                     (data_root / "manifests/all_train" / f"{task}.jsonl").read_text().splitlines()]
        assert all(row["split"] == "train" for row in task_rows)
    all_train_stats = json.loads(all_train_stats_path.read_text())
    assert all_train_stats["metadata"]["scope"] == "shared_tasks_all_train"
    assert all_train_stats["metadata"]["split"] == "all_train"
    # This held-out episode contains 1000+ joint values and must affect full-data stats.
    assert max(all_train_stats["state"]["q99"]) > 900
    assert max(all_train_stats["action"]["q99"]) > 900
    assert all(path.read_bytes() == content for path, content in preserved_bytes.items())


def test_task_balanced_sampler_mixes_within_task_handles_missing_recovery_and_resumes():
    tasks = np.asarray(["insert_hole"] * 21 + ["lift_can"])
    sources = np.asarray(["original"] * 20 + ["recovery"] + ["original"])
    dataset = SimpleNamespace(tasks=tasks, sources=sources)

    all_draws = list(MixedSequenceSampler(dataset, 4000, seed=81, task_balanced=True,
                                          original_only_draws=1200))
    resumed = list(MixedSequenceSampler(dataset, 3100, seed=81, task_balanced=True,
                                        original_only_draws=1200, start_draw=900))
    assert resumed == all_draws[900:]

    task_counts = {task: 0 for task in set(tasks)}
    task_sources = {task: {source: 0 for source in ("original", "recovery")} for task in task_counts}
    for index, _draw_seed in all_draws:
        task = tasks[index]
        source = sources[index]
        task_counts[task] += 1
        task_sources[task][source] += 1
    assert abs(task_counts["insert_hole"] / len(all_draws) - 0.5) < 0.04
    assert abs(task_counts["lift_can"] / len(all_draws) - 0.5) < 0.04
    assert task_sources["lift_can"]["recovery"] == 0
    assert abs(task_sources["insert_hole"]["original"] / task_counts["insert_hole"] - 0.65) < 0.06


def test_policy_action_mask_uses_target_frame_tags_and_report_steps(tmp_path):
    data_root, cache_root = tmp_path / "data", tmp_path / "cache"
    frames = 14
    steps = _write_episode(
        data_root / "hdf5/recovery/insert_hole/seed.hdf5", frames=frames,
        tags=["approach", "approach", "approach", "perturb"] + ["push"] * (frames - 4),
    )
    episode_id = "recovery/insert_hole/seed"
    row = {
        "episode_id": episode_id,
        "task": "insert_hole",
        "source": "recovery",
        "path": "hdf5/recovery/insert_hole/seed.hdf5",
        "split": "train",
        "frames": frames,
        "segments": [[0, frames]],
        "prompt": "insert",
        "policy_action_excluded_ranges": [{"start_exclusive": 6, "end_inclusive": 10}],
    }
    manifest = data_root / "manifests/shared.jsonl"
    write_jsonl(manifest, [row])
    cache_path = cache_root / f"{episode_id}.npz"
    cache_path.parent.mkdir(parents=True)
    np.savez(cache_path, tokens=np.zeros((frames, 8, 512), np.float16), raw_step=steps)
    stats = {key: {"q01": [0] * 8, "q99": [100] * 8} for key in ("state", "action")}

    dataset = ContactDataset(
        manifest, data_root, cache_root, stats,
        ContactConfig(k=2, chunk_size=8, belief_warmup_min=0, belief_warmup_max=0),
        policy_action_mask=True,
    )
    item = next(index for index, (_rid, _start, _stop, current) in enumerate(dataset.index) if current == 2)
    sample = dataset[(item, 5)]
    assert sample["task"] == "insert_hole"
    assert sample["action_mask"].all()
    # q[t+1] is the target; frame 3 is perturb and frames 4-5 are blocked_hold.
    np.testing.assert_array_equal(
        sample["policy_action_mask"][0], [False, False, False, True, True, True, True, True],
    )


def test_blocked_hold_report_interval_ends_at_next_phase_event(tmp_path):
    report = tmp_path / "diagnostics.json"
    report.write_text(json.dumps({"events": [
        {"event": "blocked_hold", "step": 814, "strategy_steps": 2},
        {"event": "retreat", "step": 818},
    ]}))
    assert _blocked_hold_ranges(report) == [{"start_exclusive": 814, "end_inclusive": 818}]
