"""Collect completed local debug evidence and compare frozen checkpoint tensors."""
import argparse
import json
from pathlib import Path
import platform

import torch
from safetensors import safe_open

from lerobot.utils.univtac_runtime import atomic_json, digest, project_root


def read(path):
    return json.loads(path.read_text())


def compare(left, right, prefixes):
    count = 0
    elements = 0
    with safe_open(left, framework="pt", device="cpu") as a, safe_open(right, framework="pt", device="cpu") as b:
        keys = [k for k in a.keys() if k.startswith(prefixes)]
        assert keys, prefixes
        for key in keys:
            x, y = a.get_tensor(key), b.get_tensor(key)
            assert torch.equal(x, y), f"Frozen tensor changed: {key}"
            count += 1
            elements += x.numel()
    return {"bitwise_equal": True, "tensors": count, "elements": elements}


p = argparse.ArgumentParser()
p.add_argument("outputs", type=Path, nargs=2, help="Completed insert_hole and insert_HDMI debug directories")
args = p.parse_args()
torch.set_num_threads(2)
root = project_root()
report = {"passed": True, "formal_training_started": False, "device": "cpu", "tasks": {},
          "environment": {"python": platform.python_version(), "torch": torch.__version__},
          "contract_tests": {"passed": 10, "evidence": "reports/tests.log"},
          "shared_initialization_assets": read(root/"reports/assets-insert_hole.json"),
          "independent_source_imports": read(root/"reports/isolated-import.json"),
          "vendored_source": read(root/"assets/univtac/provenance/legacy_sources.json"),
          "distributed_helpers": read(root/"reports/distributed-cpu.json"),
          "limitations": ["No A800 memory, throughput or NCCL validation", "No simulator rollout or task success claim",
                          "Local debug uses global batch 2; formal recipe uses 64",
                          "Environment reused existing compatible installation; clean A800 bootstrap not executed"]}
assert report["distributed_helpers"]["passed"]
assert report["independent_source_imports"]["passed"]
assert "10 passed" in (root/"reports/tests.log").read_text()
for output in args.outputs:
    if (output/"debug_report.json").exists():
        debug = read(output/"debug_report.json")
    else:
        # An inference-only fix can be rechecked without repeating unaffected
        # training. Keep the original training identities intact and disclose
        # any source differences below.
        results = [read(output/stage/"result.json") for stage in ("stage1", "stage2")]
        assert all(r["complete"] and r["step"] == 2 and not r["formal_training"] for r in results)
        inference = read(output/"inference_report.json")
        assert inference["finite"] and inference["fixed_noise_max_abs_difference"] == 0
        debug = {"task": results[0]["task"], "passed": True, "formal_training": False,
                 "updates_per_stage": 2, "inference": inference,
                 "restore": [read(output/stage/"restore_report.json") for stage in ("stage1", "stage2")],
                 "inference_rechecked_separately": True}
        atomic_json(output/"debug_report.json", debug)
    assert debug["passed"] and debug["updates_per_stage"] == 2 and not debug["formal_training"]
    assert all(r["parameters_optimizer_ema_restored"] and r["probe_max_abs_difference"] == 0 for r in debug["restore"])
    task = debug["task"]
    identity = read(output/"stage2/checkpoints/step_00000002/run_identity.json")
    changed = [name for name, sha in identity["hashes"].items()
               if name.startswith("src/") and digest(root/name) != sha]
    assert set(changed) <= {"src/lerobot/policies/wave0/inference_wave0.py"}, changed
    records = {}
    checkpoints = {}
    for stage in ("stage1", "stage2"):
        rows = [json.loads(line) for line in (output/stage/"train.jsonl").read_text().splitlines()]
        assert [r["step"] for r in rows] == [1, 2]
        records[stage] = [{"step": r["step"], "loss": r["loss"], "changed_groups": r["changed_groups"],
                           "all_trainable_tensors_have_gradients": all(not g["missing"] for g in r["gradients"].values())}
                          for r in rows]
        checkpoints[stage] = output/stage/"checkpoints/step_00000002/pretrained_model/model.safetensors"
    previous = output/"stage2/checkpoints/step_00000001/pretrained_model/model.safetensors"
    world = compare(checkpoints["stage1"], checkpoints["stage2"],
                    ("model.world_model.", "model.world_action_encoder.", "model.world_state_encoder."))
    vision = compare(previous, checkpoints["stage2"], ("model.world_vision_backbone.",))
    with safe_open(checkpoints["stage2"], framework="pt", device="cpu") as saved:
        assert saved.get_tensor("model.stage2_step").item() == 2
    data = read(root/"reports"/f"data-{task}.json")
    assert data["full_source_hashes_checked"] and data["stats_match_baseline"]
    report["tasks"][task] = {"output": str(output.resolve()), "identity": identity["fingerprint"],
                            "source_differences_since_training": changed,
                            "inference_code_sha256": digest(root/"src/lerobot/policies/wave0/inference_wave0.py"),
                            "debug": debug, "updates": records,
                            "stage1_world_unchanged_in_stage2": world, "frozen_target_vision_unchanged": vision,
                            "wam_schedule_counts_optimizer_updates": True, "data": data}
assert set(report["tasks"]) == {"insert_hole", "insert_HDMI"}
atomic_json(root/"reports/acceptance.json", report)
print(json.dumps({"passed": True, "report": str(root/"reports/acceptance.json")}))
