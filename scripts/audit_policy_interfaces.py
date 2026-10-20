"""Real exported-model feedback, ablation and k-resize checks; no optimization."""
import argparse
from dataclasses import replace
import torch
from contact_belief_wam.config import ContactConfig
from contact_belief_wam.data import ContactDataset, collate_contact
from contact_belief_wam.model import ContactWAM
from contact_belief_wam.paths import PROJECT, atomic_json
from contact_belief_wam.preprocessing import PromptTokenizer
from contact_belief_wam.sequence import prepare_batch, observation_at


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    checkpoint = PROJECT / "outputs/debug_local_base/stage2/pretrained_model"
    model = ContactWAM.from_pretrained(checkpoint, stage="inference", k=args.k).eval()
    cfg = model.config
    data = PROJECT / "data"
    ds = ContactDataset(data / "manifests/insert_hole.jsonl", data, data / "cache/anytouch2",
                        data / "stats/insert_hole.quantiles.json", cfg)
    i = next(i for i, (_, start, stop, frame) in enumerate(ds.index)
             if frame - start >= 4 * cfg.k and frame + cfg.k < stop)
    batch = prepare_batch(collate_contact([ds[(i, 0)]]), PromptTokenizer(checkpoint / "tokenizer"), "cpu")
    obs = observation_at(batch, 4)
    diff = lambda a, b: float((a.float() - b.float()).abs().max())
    report = {"k": cfg.k, "optimizer_updates": 0, "real_episode": batch["episode_id"]}
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        belief = model.update_belief(obs)
        perturbed = replace(obs, tactile=obs.tactile.flip(-1))
        report["tactile_to_belief_difference"] = diff(belief, model.update_belief(perturbed))
        latent, _, _ = model.encode_context(obs, belief)
        cfg.belief_to_latent_queries = False
        report["latent_query_switch_difference"] = diff(latent, model.encode_context(obs, belief)[0])
        cfg.belief_to_latent_queries = True
        current = model.encode_visual(obs.image)
        initial, corrected, contact = model.predict_future(current, obs.state, latent, belief, obs.tactile)
        cfg.belief_to_world_state = False
        report["world_state_switch_difference"] = diff(initial, model.predict_future(current, obs.state, latent, belief, obs.tactile)[0])
        cfg.belief_to_world_state = True
        model.contact.corrector.use_belief = False
        report["corrector_switch_difference"] = diff(corrected, model.contact.corrector(initial, belief, latent))
        model.contact.corrector.use_belief = True
        model.contact.predictor.use_belief = False
        report["predictor_switch_difference"] = diff(contact, model.contact.predictor(corrected, latent, obs.tactile, belief))
        model.contact.predictor.use_belief = True
        cfg.belief_to_expert = False
        off = model.policy_condition(obs, belief, current)[0]
        cfg.belief_to_expert = True
        on = model.policy_condition(obs, belief, current)[0]
        report["expert_condition_tokens_off_on"] = [off.shape[1], on.shape[1]]
        calls = []
        handle = model.core.world_action_encoder.register_forward_pre_hook(lambda module, values: calls.append(values[0].shape[1]))
        prediction = model.predict_action_chunk(obs, noise=torch.zeros(1, 50, 32))
        handle.remove()
        report.update(action_shape=list(prediction.actions.shape), actions_finite=bool(torch.isfinite(prediction.actions).all()),
                      expected_shape=list(prediction.expected_contact.shape), query_count=latent.shape[1],
                      final_action_prefix_lengths=calls, step=obs.step.tolist(), target_step=prediction.target_step.tolist())
        assert calls == [cfg.k] and latent.shape[1] == cfg.k
        assert torch.equal(prediction.target_step, obs.step + cfg.k)
        assert report["actions_finite"] and report["expert_condition_tokens_off_on"] == [4, 8]
        for key, value in report.items():
            if key.endswith("difference"):
                assert value > 0, (key, value)
    atomic_json(PROJECT / "reports" / f"policy_interfaces_k{cfg.k}.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
