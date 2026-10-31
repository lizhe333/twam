"""All eight real Stage-2 forwards from one debug checkpoint (no optimizer updates)."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
from lerobot.utils.univtac_runtime import configure, atomic_json
configure()
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from lerobot.configs.univtac import TASKS, load_config
from lerobot.datasets.univtac_dataset import UniVTACDataset
from lerobot.policies.wave0.loading_wave0 import create_policy
from lerobot.datasets.univtac_shared import install_world_normalization
from lerobot.scripts.lerobot_train import latest

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--threads", type=int, default=8)
args = p.parse_args()
torch.set_num_threads(args.threads)
output = args.output.resolve()
cfg = ROOT/"configs/univtac/shared_wm_8xa800.yaml"
teacher = json.loads((output/"shared_wm/stage1/result.json").read_text())["selected_checkpoint"]
checkpoint = latest(output/"tasks/insert_hole", "stage2")
run = load_config(cfg, output=output, task="insert_hole", stage1=teacher)
policy, _ = create_policy(run, "stage2", Path(teacher)/"pretrained_model", checkpoint/"pretrained_model")
policy.eval()
target = json.loads(Path(run["wm_norm"]).read_text())["norm_stats"]
records = {}
for task in TASKS:
    run = load_config(cfg, output=output, task=task, stage1=teacher)
    for handle in policy._world_norm_handles:
        handle.remove()
    policy._world_norm_handles = []
    install_world_normalization(policy, Path(run["baseline"])/"norm_stats.json", run["wm_norm"])
    ds = UniVTACDataset(run)
    samples = [ds[0], ds[1]]
    batch = default_collate(samples)
    original_action = batch["action"].clone()
    captured = {}

    def capture_state(module, inputs, result):
        captured["state"] = inputs[0].detach().clone()

    def capture_actions(module, inputs, result):
        captured["actions"] = inputs[0].detach().clone()
        captured["action_state"] = inputs[1].detach().clone()

    handles = [policy.model.world_state_encoder.register_forward_hook(capture_state),
               policy.model.world_action_encoder.register_forward_hook(capture_actions)]
    torch.manual_seed(123)
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        loss, metrics = policy(batch)
    assert torch.isfinite(loss)
    assert torch.equal(batch["action"], original_action)
    expected_state, expected_actions = [], []
    for index in (0, 1):
        segment, frame = ds.index[index]
        source = ds._file(ds.segments[segment])
        state = np.asarray(source["embodiment/joint"][frame, :8], np.float32)
        actions = np.asarray(source["embodiment/joint"][frame+1:frame+11, :8], np.float32)
        actions[:, :7] -= state[:7]
        for values, key, dest in ((state, "state", expected_state), (actions, "actions", expected_actions)):
            q = target[key]
            dest.append(((values-np.asarray(q["q01"]))/
                         (np.asarray(q["q99"])-np.asarray(q["q01"])+1e-6)*2-1).astype(np.float32))
    differences = {}
    for key, expected in (("state", expected_state), ("action_state", expected_state), ("actions", expected_actions)):
        got = captured[key][..., :8].float().numpy()
        reference = np.asarray(expected)
        np.testing.assert_allclose(got, reference, rtol=1e-4, atol=5e-4)
        differences[key] = float(np.abs(got-reference).max())
        assert not captured[key][..., 8:].count_nonzero()
    for handle in handles:
        handle.remove()
    ds.close()
    records[task] = {"finite_loss": float(loss), "world_input_max_abs_error": differences,
                     "policy_flow_targets_unchanged": True, "metrics": metrics}
    print(json.dumps({"task": task, **records[task]}), flush=True)
atomic_json(output/"all_task_forwards.json", {"passed": True, "tasks": records,
            "checkpoint": str(checkpoint), "separate_task_training": False, "optimizer_updates": 0})
