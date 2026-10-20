"""Actual pretrained recurrence/gradient audit, with zero optimizer updates."""
import argparse
from pathlib import Path
import torch
from contact_belief_wam.data import ContactDataset, collate_contact
from contact_belief_wam.model import ContactWAM
from contact_belief_wam.paths import PROJECT, atomic_json
from contact_belief_wam.preprocessing import PromptTokenizer
from contact_belief_wam.sequence import SequenceTrainingModel, prepare_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    path = PROJECT / "outputs/debug_local_base" / args.stage / "pretrained_model"
    policy = ContactWAM.from_pretrained(path, stage=args.stage).train()
    data = PROJECT / "data"
    dataset = ContactDataset(data / "manifests/insert_hole.jsonl", data, data / "cache/anytouch2",
                             data / "stats/insert_hole.quantiles.json", policy.config)
    index = next(i for i, (_, start, stop, frame) in enumerate(dataset.index)
                 if frame - start >= 4 * policy.config.k and frame + policy.config.k < stop)
    batch = prepare_batch(collate_contact([dataset[(index, 0)]]), PromptTokenizer(path / "tokenizer"), "cpu")
    updates = []
    handle = policy.register_forward_hook(lambda module, inputs, output: updates.append(output))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = SequenceTrainingModel(policy)(batch)
    assert len(updates) == 2
    memory_gradient, = torch.autograd.grad(updates[-1]["loss_per_sample"].mean(), updates[0]["belief"], retain_graph=True)
    result["loss"].backward()
    gradients = {name: float(value.grad.float().norm()) if value.grad is not None else None
                 for name, value in policy.named_parameters() if name.startswith("contact.") and value.requires_grad}
    missing = [name for name, norm in gradients.items() if norm is None]
    if missing:
        raise AssertionError(f"Trainable contact parameters missing gradients: {missing}")
    if not torch.isfinite(memory_gradient).all() or memory_gradient.abs().sum() == 0:
        raise AssertionError("Final loss does not reach the previous belief update")
    report = {"stage": args.stage, "optimizer_updates": 0, "real_episode": batch["episode_id"],
              "last_loss_to_previous_belief_grad_norm": float(memory_gradient.float().norm()),
              "contact_gradient_norms": gradients, "missing_contact_gradients": missing,
              "bf16_no_grad_warmup_cache_cleared": True}
    atomic_json(PROJECT / "reports" / f"gradient_audit_{args.stage}.json", report)
    handle.remove()
    print({k: v for k, v in report.items() if k != "contact_gradient_norms"}, flush=True)


if __name__ == "__main__":
    main()
