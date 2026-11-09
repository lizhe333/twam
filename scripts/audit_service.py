"""Actual encoder/service protocol check using recorded images, without simulation."""
import h5py
import numpy as np

from contact_belief_wam.paths import PROJECT, atomic_json, read_jsonl
from contact_belief_wam.preprocessing import decode_rgb, window_indices
from contact_belief_wam.protocol import Client
from contact_belief_wam.tactile import tactile_keys


def main():
    row = read_jsonl(PROJECT / "data/manifests/insert_hole.jsonl")[0]
    client = Client()
    try:
        episode = "recorded-input-protocol-check"
        client.call(op="reset", episode=episode)
        report = {"scope": "Recorded-input service check; not a model-driven simulator rollout", **client.metadata}
        executed = np.zeros((0, 8), np.float32)
        results = []
        with h5py.File(PROJECT / "data" / row["path"]) as f:
            for step in (0, client.metadata["k"]):
                frame = 24 + step
                window = np.stack([np.stack([decode_rgb(f[key][i]) for i in window_indices(frame)]) for key in tactile_keys(f)])
                views = ({"wrist_image": decode_rgb(f["observation/wrist/rgb"][frame])}
                         if client.metadata.get("camera_mode") == "head_wrist" else {})
                result = client.call(op="infer", episode=episode, step=step,
                                     image=decode_rgb(f["observation/head/rgb"][frame]),
                                     state=f["embodiment/joint"][frame, :8], tactile_window=window,
                                     executed_actions=executed, prompt=row["prompt"], **views)
                assert result["actions"].shape == (50, 8) and np.isfinite(result["actions"]).all()
                diagnostics = result["diagnostics"]
                diagnostics["belief_norm"] = diagnostics["belief_norm"].tolist()
                assert diagnostics["expectation_valid"] == (step > 0)
                assert diagnostics["target_step"] == step + client.metadata["k"]
                results.append(diagnostics)
                executed = result["actions"][:client.metadata["k"]]
        report["updates"] = results
        report["reset"] = client.call(op="reset", episode=episode)
        client.call(op="close", episode=episode)
        atomic_json(PROJECT / "reports/service_protocol_actual.json", report)
        print(report, flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
