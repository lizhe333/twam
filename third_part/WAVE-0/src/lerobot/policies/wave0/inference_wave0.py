from pathlib import Path
import json
import torch
from safetensors.torch import load_file

from lerobot.policies.wave0.loading_wave0 import BaselineWAVE0
from lerobot.datasets.univtac_dataset import Processor
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.wave0.configuration_wave0 import WAVE0Config


def inference_config(path):
    # Saved configs include the choice-registry discriminator, "type": "wave0".
    config = PreTrainedConfig.from_pretrained(str(path), local_files_only=True)
    if not isinstance(config, WAVE0Config):
        raise ValueError("Expected a WAVE-0 checkpoint configuration")
    config.world_model_stage = "inference"
    config.world_model_stage1_checkpoint_path = None
    config.world_model_base_model_path = None
    config.device = "cpu"
    return config


def install_inference_normalization(model, path):
    path = Path(path)
    if (path/"wm_norm_stats.json").exists():
        from lerobot.datasets.univtac_shared import install_world_normalization
        # Official inference construction omits this training-only module
        # entirely. Represent that omission explicitly for the shared hook API.
        if not hasattr(model.model, "world_action_encoder"):
            model.model.world_action_encoder = None
        install_world_normalization(model, path/"norm_stats.json", path/"wm_norm_stats.json")


class Policy:
    """Observation -> absolute 50x8 joint targets; caller executes first ten."""
    def __init__(self, checkpoint, device="cpu", use_ema=None):
        path = Path(checkpoint)
        self.metadata = json.loads((path/"deployment.json").read_text())
        if use_ema is None:
            use_ema = self.metadata.get("use_ema", True)
        if not isinstance(use_ema, bool):
            raise ValueError("use_ema must be a boolean")
        self.use_ema = use_ema
        if use_ema and not (path/"ema.safetensors").is_file():
            raise FileNotFoundError("EMA requested but checkpoint has no ema.safetensors")
        if json.loads((path/"complete.json").read_text())["stage"] != "stage2":
            raise ValueError("Inference requires a Stage-2 checkpoint")
        config = inference_config(path/"pretrained_model")
        self.model = BaselineWAVE0.from_pretrained(str(path/"pretrained_model"), config=config,
                                                   strict=True, local_files_only=True)
        install_inference_normalization(self.model, path)
        if use_ema:
            state = load_file(str(path/"ema.safetensors"))
            expected = self.model.state_dict()
            if any(k not in expected or v.shape != expected[k].shape for k,v in state.items()):
                raise ValueError("EMA is incompatible with inference model")
            self.model.load_state_dict(state, strict=False)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        # Upstream denoising switches the Expert to eager attention. Set it
        # before future-condition prefill too, so the first chunk uses the same
        # numerical path as subsequent chunks (rather than SDPA then eager).
        self.model.model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        self.processor = Processor(path/"norm_stats.json", path/"tokenizer.model", self.metadata["prompt"],
                                   self.metadata["camera"])
        if tuple(config.world_model_image_features) != self.processor.camera_keys:
            raise ValueError("Checkpoint camera features disagree with deployment metadata")

    @torch.inference_mode()
    def infer(self, head_rgb, state, noise=None, *, wrist_rgb=None):
        batch = {k:v[None].to(self.device) for k,v in self.processor.observation(head_rgb, state, wrist_rgb=wrist_rgb).items()}
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            actions = self.model.predict_action_chunk(batch, noise=noise)
        result = self.processor.absolute(actions[0].float().cpu().numpy(), state)
        if result.shape != (50,8) or not torch.isfinite(torch.from_numpy(result)).all():
            raise ValueError("Invalid absolute policy actions")
        return result
