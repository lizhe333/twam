"""Validate and collect shared-workflow local evidence, without quality claims."""
import argparse
import json
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
from lerobot.utils.univtac_runtime import configure, atomic_json, digest
configure()
import torch
from safetensors import safe_open


def read(path):
    return json.loads(Path(path).read_text())


def compare(left, right, prefixes):
    elements, count = 0, 0
    with safe_open(left, framework="pt", device="cpu") as a, safe_open(right, framework="pt", device="cpu") as b:
        keys = [k for k in a.keys() if k.startswith(prefixes)]
        assert keys
        for key in keys:
            x, y = a.get_tensor(key), b.get_tensor(key)
            assert torch.equal(x, y), key
            elements += x.numel()
            count += 1
    return {"bitwise_equal": True, "tensors": count, "elements": elements}


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=Path)
p.add_argument("--tests-log", type=Path, required=True)
args = p.parse_args()
torch.set_num_threads(2)
out = args.output.resolve()
debug = read(out/"debug_report.json")
assert debug["passed"] and not debug["formal_training"]
data = read(out/"artifacts/data_report.json")
assert len(data["tasks"])==8 and all(t["full_source_hashes_checked"] for t in data["tasks"].values())
assert all(t["stats_match_baseline"] for t in data["tasks"].values())
teacher = Path(debug["shared_wm"]["selected_checkpoint"])
all_forwards = read(out/"all_task_forwards.json")
assert all_forwards["passed"] and len(all_forwards["tasks"])==8
completed_resume = read(out/"completed_resume.json")
assert completed_resume["skipped_completed"]
assert re.search(r"\b\d+ passed\b", args.tests_log.read_text())
assert not re.search(r"\b\d+ failed\b", args.tests_log.read_text())
restored = read(out/"shared_wm/stage1/restore_report.json")
assert restored["parameters_optimizer_ema_restored"] and restored["probe_max_abs_difference"]==0
evidence = {}
for task, result in debug["tasks"].items():
    assert result["train"]["step"]==2 and not result["train"]["formal_training"]
    assert result["restore"]["parameters_optimizer_ema_restored"] and result["restore"]["probe_max_abs_difference"]==0
    assert result["inference"]["finite"] and result["inference"]["fixed_noise_max_abs_difference"]==0
    stage = out/"tasks"/task/"stage2"
    current = stage/"checkpoints/step_00000002"
    before = stage/"checkpoints/step_00000001"
    ident = read(current/"run_identity.json")
    changed = [name for name, sha in ident["hashes"].items()
               if name.startswith("src/") and digest(ROOT/name)!=sha]
    # This adapter is never imported by training. Recheck its final EMA
    # inference separately while preserving the original training identity.
    assert set(changed) <= {"src/lerobot/policies/wave0/inference_wave0.py"}, changed
    assert ident["recipe"]["shared_teacher"] == read(teacher/"complete.json")
    assert digest(current/"wm_norm_stats.json")==digest(teacher/"wm_norm_stats.json")
    assert digest(current/"norm_stats.json")==digest(ROOT/"assets/univtac/baseline"/task/"norm_stats.json")
    current_model = current/"pretrained_model/model.safetensors"
    rows = [json.loads(line) for line in (stage/"train.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows]==[1, 2]
    assert all(all(not g["missing"] for g in row["gradients"].values()) for row in rows)
    with safe_open(current_model, framework="pt", device="cpu") as tensors:
        assert tensors.get_tensor("model.stage2_step").item()==2
    evidence[task] = {"identity": ident["fingerprint"], "updates": rows,
                     "source_differences_since_training": changed,
                     "inference_rechecked_after_adapter_fix": bool(changed),
                     "shared_teacher_bitwise_frozen": compare(teacher/"pretrained_model/model.safetensors", current_model,
                         ("model.world_model.", "model.world_state_encoder.", "model.world_action_encoder.")),
                     "target_vision_bitwise_frozen": compare(before/"pretrained_model/model.safetensors", current_model,
                         ("model.world_vision_backbone.",))}
report = {"passed": True, "formal_training_started": False, "simulator_rollout": False,
          "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
          "tests_log": str(args.tests_log.resolve()), "tests_output": args.tests_log.read_text(),
          "data": data, "debug": debug, "shared_restore": restored,
          "initialization_assets": read(ROOT/"reports/assets-shared_wm.json"),
          "all_task_real_forwards": all_forwards, "checkpoint_checks": evidence,
          "completed_resume_skips_model_loading": completed_resume["skipped_completed"],
          "distributed_helpers": read(ROOT/"reports/distributed-cpu.json"),
          "limitations": ["CUDA memory/throughput/NCCL untested", "No simulator success-rate measurement",
                          "Real Stage-2 optimizer updates cover insert_hole and insert_HDMI; other six have real forward checks",
                          "Debug uses Stage-1 global batch 16 and Stage-2 global batch 2 on CPU",
                          "Camera matching must be read from each checkpoint deployment contract; baseline snapshots retain historical provenance"]}
assert report["distributed_helpers"]["passed"]
atomic_json(ROOT/"reports/acceptance-shared.json", report)
print(json.dumps({"passed": True, "report": str(ROOT/"reports/acceptance-shared.json")}))
