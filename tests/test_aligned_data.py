import importlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from contact_belief_wam.aligned_data import (
    SHARED_TASKS,
    AlignedDataset,
    AlignedProcessor,
    AlignedTaskBalancedSampler,
    prepare_aligned_data,
)
from contact_belief_wam.config import ContactConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data"
REFERENCE_ROOT = WORKSPACE / "wave0-univtac-fast"
TOKENIZER = WORKSPACE / "models/pi05_base/big_vision/paligemma_tokenizer.model"
WM_STATS = WORKSPACE / "outputs/wave0-shared-8task-mb32-002/artifacts/wm_norm_stats.json"


def _balanced_dataset():
    tasks, sources = [], []
    for task in SHARED_TASKS:
        for source in (("original", "recovery") if task in {"insert_hole", "insert_HDMI"} else ("original",)):
            count = 20 if source == "original" else 10
            tasks.extend([task] * count)
            sources.extend([source] * count)
    return SimpleNamespace(tasks=tasks, sources=sources)


def _small_manifest():
    return [
        {"task": "insert_hole", "source": "original", "episode_id": f"original/insert_hole/{index}",
         "path": f"{index}.hdf5", "split": "train"}
        for index in range(20)
    ] + [
        {"task": "insert_hole", "source": "recovery", "episode_id": f"recovery/insert_hole/{index}",
         "path": f"{index}.hdf5", "split": "train"}
        for index in range(20, 30)
    ]


def test_aligned_sampler_has_exact_stage1_composition_and_replays_by_offset():
    dataset = _balanced_dataset()
    sampler = AlignedTaskBalancedSampler(dataset, 64, seed=29, rank=0, world_size=2,
                                         micro_batch=32, task_balanced=True)
    draws = list(sampler)
    assert draws == list(AlignedTaskBalancedSampler(
        dataset, 64, seed=29, rank=0, world_size=2, micro_batch=32, task_balanced=True
    ))
    assert draws[32:] == list(AlignedTaskBalancedSampler(
        dataset, 32, seed=29, rank=0, world_size=2, start_draw=32,
        micro_batch=32, task_balanced=True
    ))
    sampled_tasks = np.asarray(dataset.tasks)[[index for index, _ in draws[:32]]]
    sampled_sources = np.asarray(dataset.sources)[[index for index, _ in draws[:32]]]
    assert Counter(sampled_tasks) == Counter({task: 4 for task in SHARED_TASKS})
    for task in SHARED_TASKS:
        chosen = sampled_sources[sampled_tasks == task]
        expected = Counter({"original": 2, "recovery": 2}) if task in {"insert_hole", "insert_HDMI"} else Counter({"original": 4})
        assert Counter(chosen) == expected

    other_rank = list(AlignedTaskBalancedSampler(
        dataset, 32, seed=29, rank=1, world_size=2, micro_batch=32, task_balanced=True
    ))
    assert set(index for index, _ in draws[:32]).isdisjoint(index for index, _ in other_rank)


def test_aligned_sampler_stage2_is_fixed_half_sources_and_resume_exact():
    dataset = SimpleNamespace(
        tasks=["insert_hole"] * 90,
        sources=["original"] * 60 + ["recovery"] * 30,
    )
    sampler = AlignedTaskBalancedSampler(dataset, 64, seed=7, rank=0, world_size=2,
                                         micro_batch=32, task_balanced=False)
    draws = list(sampler)
    assert draws[32:] == list(AlignedTaskBalancedSampler(
        dataset, 32, seed=7, rank=0, world_size=2, start_draw=32,
        micro_batch=32, task_balanced=False
    ))
    for begin in (0, 32):
        selected = [index for index, _ in draws[begin:begin + 32]]
        assert Counter(np.asarray(dataset.sources)[selected]) == {"original": 16, "recovery": 16}


def test_world_action_roundtrip_uses_previous_interval_anchor():
    stats = {
        "state": {"q01": [-20.0] * 8, "q99": [20.0] * 8},
        "actions": {"q01": [-4.0] * 8, "q99": [4.0] * 8},
    }
    joints = np.stack([np.arange(8, dtype=np.float32) * 0.01 + index * 0.02
                       for index in range(60)])
    k, current = 10, 37
    anchor = joints[current - k]
    executed_targets = joints[current - k + 1:current + 1]
    encoded = AlignedDataset._world_actions(executed_targets, anchor, stats)[:, :8]
    decoded = AlignedProcessor(stats, None, "insert").absolute(encoded, anchor)
    np.testing.assert_allclose(decoded, executed_targets, atol=2e-6, rtol=1e-6)
    assert not np.allclose(anchor[:7], joints[current, :7])


def _load_reference_dataset_module():
    source = str(REFERENCE_ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    return importlib.import_module("wave0_univtac.data")


@pytest.mark.skipif(
    not all(path.exists() for path in (
        DATA_ROOT / "manifests/insert_hole.jsonl",
        DATA_ROOT / "hdf5/original/insert_hole/0.hdf5",
        DATA_ROOT / "cache/anytouch2/original/insert_hole/0.npz",
        REFERENCE_ROOT / "baseline/insert_hole/segments.json",
        WM_STATS,
        TOKENIZER,
    )),
    reason="Reference UniVTAC data assets are not installed",
)
def test_real_original_sample_matches_reference_processor_and_recurrent_anchor(tmp_path):
    pytest.importorskip("sentencepiece")
    reference_data = _load_reference_dataset_module()
    run = {
        "workspace": str(WORKSPACE),
        "data_root": str(DATA_ROOT),
        "tasks": list(SHARED_TASKS),
        "aligned_manifest_dir": str(tmp_path / "manifests/aligned"),
        "policy_manifest_dir": str(tmp_path / "manifests/policy_all_train"),
        "task_stats_dir": str(tmp_path / "stats/tasks"),
        "wm_stats": str(tmp_path / "stats/wm_norm_stats.json"),
        "cache_root": str(DATA_ROOT / "cache/anytouch2"),
        "tokenizer_path": str(TOKENIZER),
    }
    prepared = prepare_aligned_data(run)
    assert prepared["validation_episode_ids"] == [14, 20, 51, 60, 71, 74, 82, 86, 91, 92]
    assert prepared["summary"]["stage1_train_original"] == 720
    assert prepared["summary"]["stage1_train_recovery"] == 100
    assert prepared["summary"]["stage1_val_original"] == 80
    for task in ("insert_hole", "insert_HDMI"):
        assert prepared["summary"]["tasks"][task]["stage2_original"] == 100
        assert prepared["summary"]["tasks"][task]["stage2_recovery"] == 50

    config = ContactConfig(stage="stage2", k=10, chunk_size=50,
                           belief_warmup_min=0, belief_warmup_max=0)
    for task in ("insert_hole", "insert_HDMI"):
        aligned = AlignedDataset(
            prepared["policy_manifests"][task], DATA_ROOT, DATA_ROOT / "cache/anytouch2",
            prepared["task_stats"], prepared["wm_stats"], config,
            tokenizer_path=prepared["tokenizer_path"],
        )
        reference_run = {
            "baseline": REFERENCE_ROOT / "baseline" / task,
            "workspace": WORKSPACE,
            "task": task,
            "tokenizer": TOKENIZER,
        }
        reference = reference_data.UniVTACDataset(reference_run, episode_ids={0})
        try:
            aligned_index = next(index for index, entry in enumerate(aligned.index)
                                 if aligned.rows[entry[0]]["episode_id"] == f"original/{task}/0"
                                 and entry[3] == 37)
            reference_index = next(index for index, (segment, frame) in enumerate(reference.index)
                                   if reference.segments[segment]["source_episode_id"] == 0 and frame == 37)
            aligned_sample = aligned[aligned_index]
            reference_sample = reference[reference_index]
            np.testing.assert_allclose(aligned_sample["state"][0], reference_sample["observation.state"].numpy(),
                                       atol=1e-7, rtol=0)
            np.testing.assert_array_equal(aligned_sample["tokens"][0],
                                          reference_sample["observation.language.tokens"].numpy())
            np.testing.assert_array_equal(aligned_sample["token_mask"][0],
                                          reference_sample["observation.language.attention_mask"].numpy())
            np.testing.assert_allclose(aligned_sample["actions"][0], reference_sample["action"].numpy(),
                                       atol=1e-7, rtol=0)
            np.testing.assert_allclose(aligned_sample["image"][0],
                                       reference_sample[reference_data.IMAGE].numpy() * 2 - 1,
                                       atol=1e-6, rtol=0)
            np.testing.assert_allclose(aligned_sample["future_image"][0],
                                       reference_sample[reference_data.FUTURE].numpy() * 2 - 1,
                                       atol=1e-6, rtol=0)

            # At t=37, executed targets cover q[28:38] and are decoded from q[27].
            with h5py.File(DATA_ROOT / "hdf5/original" / task / "0.hdf5", "r") as handle:
                joints = np.asarray(handle["embodiment/joint"][:, :8], np.float32)
            decoded_executed = AlignedProcessor(prepared["wm_stats"], None, "insert").absolute(
                aligned_sample["executed_actions"][0], joints[27]
            )
            np.testing.assert_allclose(decoded_executed, joints[28:38], atol=2e-5, rtol=1e-6)
        finally:
            aligned.close()
            reference.close()

    # Recovery policy exclusions apply to blocked holds and tagged perturbations
    # at the target-frame granularity for either configured execution interval.
    for task in ("insert_hole", "insert_HDMI"):
        policy_rows = [json.loads(line) for line in Path(prepared["policy_manifests"][task]).read_text().splitlines()]
        recovery_row = next(row for row in policy_rows if row["source"] == "recovery"
                            and row.get("policy_action_excluded_ranges"))
        with h5py.File(DATA_ROOT / recovery_row["path"], "r") as handle:
            steps = np.asarray(handle["step"][:])
            raw_tags = handle["atom/tag"][:] if "atom/tag" in handle else None
        tags = (np.char.decode(raw_tags, "utf-8") if raw_tags is not None and raw_tags.dtype.kind == "S"
                else raw_tags.astype(str) if raw_tags is not None else None)
        interval = recovery_row["policy_action_excluded_ranges"][0]
        hold_target = int(np.flatnonzero((steps > int(interval["start_exclusive"])) &
                                         (steps <= int(interval["end_inclusive"])))[0])
        current = hold_target - 1
        segment_start, segment_stop = next((start, stop) for start, stop in recovery_row["segments"]
                                           if start <= current < stop - 1)
        assert current - 10 >= segment_start
        for k in (5, 10):
            config = ContactConfig(stage="stage2", k=k, chunk_size=50,
                                   belief_warmup_min=0, belief_warmup_max=0)
            aligned = AlignedDataset(
                prepared["policy_manifests"][task], DATA_ROOT, DATA_ROOT / "cache/anytouch2",
                prepared["task_stats"], prepared["wm_stats"], config,
                tokenizer_path=prepared["tokenizer_path"],
            )
            try:
                item = next(index for index, entry in enumerate(aligned.index)
                            if aligned.rows[entry[0]]["episode_id"] == recovery_row["episode_id"]
                            and entry[1] == segment_start and entry[2] == segment_stop
                            and entry[3] == current)
                sample = aligned[item]
                target_indices = np.minimum(current + 1 + np.arange(50), segment_stop - 1)
                target_steps = steps[target_indices]
                expected = np.ones(50, dtype=bool)
                if tags is not None:
                    expected &= tags[target_indices] != "perturb"
                for excluded in recovery_row["policy_action_excluded_ranges"]:
                    expected &= ~((target_steps > int(excluded["start_exclusive"])) &
                                  (target_steps <= int(excluded["end_inclusive"])))
                np.testing.assert_array_equal(sample["policy_action_mask"][0], expected)
                assert not expected.any() or not expected[np.flatnonzero(target_steps == steps[hold_target])].all()
                assert sample["future_mask"][0]
                with h5py.File(DATA_ROOT / recovery_row["path"], "r") as handle:
                    joints = np.asarray(handle["embodiment/joint"][:, :8], np.float32)
                decoded = AlignedProcessor(prepared["wm_stats"], None, "insert").absolute(
                    sample["executed_actions"][0], joints[current - k]
                )
                np.testing.assert_allclose(decoded, joints[current - k + 1:current + 1], atol=2e-5, rtol=1e-6)
            finally:
                aligned.close()

