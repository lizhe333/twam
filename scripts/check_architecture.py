"""Full-size architecture check while pretrained downloads are pending; zero optimizer updates.

This is supplementary evidence, not the pretrained real-batch/checkpoint debug report.
"""
from pathlib import Path
import argparse
import json
import time
import torch

from contact_belief_wam.config import ContactConfig
from contact_belief_wam.legacy.data import ContactDataset, collate_contact
from contact_belief_wam.model import ContactWAM
from contact_belief_wam.paths import PROJECT, atomic_json
from contact_belief_wam.preprocessing import vision_preprocess, PromptTokenizer
from contact_belief_wam.sequence import SequenceTrainingModel, prepare_batch, observation_at


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="stage1", choices=["stage1", "stage2"])
    parser.add_argument("--task", default="insert_hole")
    parser.add_argument("--mode", default="full")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    cfg = ContactConfig(stage=args.stage, mode=args.mode, k=args.k)
    data = PROJECT / "data"
    ds = ContactDataset(data / "manifests" / f"{args.task}.jsonl", data, data / "cache/anytouch2",
                        data / "stats" / f"{args.task}.quantiles.json", cfg)
    index = next(i for i, (rid, start, stop, frame) in enumerate(ds.index)
                 if (data / "cache/anytouch2" / (ds.rows[rid]["episode_id"] + ".npz")).exists()
                 and frame - start >= 4 * cfg.k and frame + cfg.k < stop)
    raw = collate_contact([ds[(index, 0)]])
    if args.stage == "stage1":
        batch = dict(raw)
        for key in ("image", "future_image"):
            images = raw[key].numpy()
            batch[key] = vision_preprocess(images.reshape(-1, *images.shape[2:])).reshape(1, 5, 3, 224, 224)
        # Stage 1 has no language tower and never reads these fields.
        batch["tokens"] = torch.zeros(1, 5, 1, dtype=torch.long)
        batch["token_mask"] = torch.ones(1, 5, 1, dtype=torch.bool)
    else:
        tokenizer = PromptTokenizer(PROJECT / "models/tokenizer")
        batch = prepare_batch(raw, tokenizer, "cpu")
    started = time.monotonic()
    print("constructing full-size random architecture", flush=True)
    policy = ContactWAM(cfg).train()
    if args.inference_only:
        if args.stage != "stage2":
            parser.error("--inference-only requires --stage stage2")
        policy.eval()
        obs = observation_at(batch, 4)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            result = policy.predict_action_chunk(obs, noise=torch.zeros(1, 50, 32))
        if not torch.isfinite(result.actions).all():
            raise FloatingPointError("Nonfinite inference actions")
        report = {"mode": args.mode, "k": cfg.k, "shape": list(result.actions.shape),
                  "expected_shape": list(result.expected_contact.shape) if result.expected_contact is not None else None,
                  "target_step": result.target_step.tolist(), "step": obs.step.tolist(),
                  "pretrained_weights_loaded": False, "optimizer_updates": 0,
                  "elapsed_seconds": time.monotonic() - started}
        atomic_json(PROJECT / "reports" / f"architecture_inference_{args.mode}_k{cfg.k}.json", report)
        print(json.dumps(report), flush=True)
        return
    print("forward", flush=True)
    wrapper = SequenceTrainingModel(policy)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = wrapper(batch)
    print("backward", float(output["loss"]), flush=True)
    output["loss"].backward()
    gradients = {name: float(p.grad.float().norm()) for name, p in policy.named_parameters()
                 if p.requires_grad and p.grad is not None}
    if not all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None):
        raise FloatingPointError("Nonfinite architecture-check gradient")
    result = {"stage": args.stage, "mode": args.mode, "pretrained_weights_loaded": False,
              "optimizer_updates": 0, "real_episode": raw["episode_id"],
              "sequence_mask": batch["sequence_mask"].tolist(), "loss": float(output["loss"]),
              "gradient_norms": gradients, "elapsed_seconds": time.monotonic() - started}
    atomic_json(PROJECT / "reports" / f"architecture_{args.stage}_{args.mode}.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "gradient_norms"}), flush=True)


if __name__ == "__main__":
    main()
