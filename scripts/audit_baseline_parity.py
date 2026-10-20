"""Compare wrapper baseline modes to the vendored sampler on a real input."""
import torch

from contact_belief_wam.data import ContactDataset, collate_contact
from contact_belief_wam.model import ContactWAM, pad8
from contact_belief_wam.paths import PROJECT, atomic_json
from contact_belief_wam.preprocessing import PromptTokenizer
from contact_belief_wam.sequence import prepare_batch, observation_at


def main():
    torch.set_num_threads(6)
    checkpoint = PROJECT / "outputs/debug_local_base/stage2/pretrained_model"
    model = ContactWAM.from_pretrained(checkpoint, stage="inference").eval()
    data = PROJECT / "data"
    ds = ContactDataset(data / "manifests/insert_hole.jsonl", data, data / "cache/anytouch2",
                        data / "stats/insert_hole.quantiles.json", model.config)
    batch = prepare_batch(collate_contact([ds[(0, 0)]]), PromptTokenizer(checkpoint / "tokenizer"), "cpu")
    obs = observation_at(batch, 4)
    mask = [torch.ones(1, dtype=torch.bool)]
    report = {"optimizer_updates": 0, "real_episode": batch["episode_id"],
              "scope": "Same exported parameters and fixed noise; wrapper versus official sampler, not trained baseline quality."}
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        for mode in ("pi05", "wave0"):
            model.config.mode = mode
            model.core.config.world_model_stage = "inference" if mode == "wave0" else "disabled"
            noise = torch.zeros(1, 50, 32)
            kwargs = {"state": pad8(obs.state), "world_images": [obs.image]} if mode == "wave0" else {}
            official = model.core.sample_actions([obs.image], mask, obs.tokens, obs.token_mask, noise=noise.clone(), **kwargs)
            wrapped = model.predict_action_chunk(obs, noise=noise.clone()).actions
            delta = float((official[:, :, :8] - wrapped).abs().max())
            report[mode] = {"max_abs_difference": delta, "finite": bool(torch.isfinite(wrapped).all())}
            assert delta == 0 and report[mode]["finite"]
    atomic_json(PROJECT / "reports/baseline_sampler_parity.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
