from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class ContactConfig:
    aligned: bool = False
    policy_augmentation: bool = False
    policy_norm_stats: dict | None = None
    world_norm_stats: dict | None = None
    mode: str = "full"
    stage: str = "stage1"
    # Old checkpoints retain their original coupled contact predictor.
    independent_tactile: bool = False
    # Legacy checkpoints default to their original single-camera/trainable WM.
    camera_mode: str = "head_only"
    stage1_freeze_world: bool = False
    k: int = 10
    chunk_size: int = 50
    action_dim: int = 8
    padded_dim: int = 32
    width: int = 512
    heads: int = 8
    ffn: int = 2048
    belief_tokens: int = 4
    tactile_tokens: int = 8
    world_width: int = 1024
    world_depth: int = 12
    world_heads: int = 16
    world_ffn: int = 4096
    vision_width: int = 2048
    vision_pool: int = 4
    expert_width: int = 1024
    tokenizer_max_length: int = 200
    num_inference_steps: int = 10
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    attention_backend: str = "eager"  # Legacy checkpoints retain their execution path.
    belief_to_latent_queries: bool = True
    belief_to_world_state: bool = True
    belief_to_future_corrector: bool = True
    belief_to_contact_predictor: bool = True
    belief_to_expert: bool = True
    # Optional Stage-2 parameter freezes. The forward pass remains differentiable
    # with respect to module inputs, so policy losses can still shape upstream
    # trainable projections and the shared belief state.
    stage2_freeze_belief: bool = False
    stage2_freeze_contact_predictor: bool = False
    belief_warmup_min: int = 0
    belief_warmup_max: int = 4
    belief_bptt_steps: int = 2
    # Execution-only optimization: recurrence remains sequential and differentiable.
    batch_stage2_policy: bool = False
    counterfactual_history_prob: float = 0.0
    pi05_base: str = ""
    tokenizer_path: str = ""
    encoder_reference: str = "anytouch2-four-frames-gsmini-v1"

    def __post_init__(self):
        if self.camera_mode not in {"head_only", "head_wrist"}:
            raise ValueError(f"Unsupported camera mode: {self.camera_mode}")
        if self.attention_backend not in {"eager", "sdpa"}:
            raise ValueError(f"Unsupported attention backend: {self.attention_backend}")
        if self.mode not in {"pi05", "wave0", "direct_touch", "full"}:
            raise ValueError(f"Unknown mode: {self.mode}")
        if self.stage not in {"tactile", "stage1", "stage2", "inference"}:
            raise ValueError(f"Unknown stage: {self.stage}")
        if self.stage == "tactile" and (self.mode != "full" or not self.independent_tactile):
            raise ValueError("The tactile stage requires full mode and independent_tactile=True")
        if not 2 <= self.k <= self.chunk_size:
            raise ValueError("k must be between 2 and chunk_size")
        if not 0 <= self.belief_warmup_min <= self.belief_warmup_max:
            raise ValueError("Invalid belief warmup range")
        if self.belief_bptt_steps not in {1, 2}:
            raise ValueError("belief_bptt_steps must be 1 or 2")
        if self.action_dim != 8 or self.padded_dim != 32:
            raise ValueError("UniVTAC v1 uses eight absolute targets padded to 32")

    @property
    def world_only(self):
        return self.stage in {"tactile", "stage1"}

    @property
    def use_belief(self):
        return self.mode in {"direct_touch", "full"}

    @property
    def use_world(self):
        return self.mode in {"wave0", "full"}

    def to_dict(self):
        return {"type": "contact_wave0", **asdict(self)}

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value.pop("type", None)
        return cls(**value)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class RecoveryConfig:
    successful_episodes_per_task: int = 50
    min_blockages: int = 1
    max_blockages: int = 2
    lateral_min_mm: float = 2.0
    lateral_max_mm: float = 5.0
    lateral_direction: str = "uniform"
    angular_max_deg: float = 5.0
    angular_axis: int = 2
    advance_increment_mm: float = 0.5
    probe_physics_steps: int = 16
    alignment_waypoints: int = 6
    alignment_physics_steps: int = 24
    precontact_clearance_mm: float = 3.0
    final_insertion_attempts: int = 3
    approach_mm: float = 0.0
    retreat_min_mm: float = 2.0
    retreat_max_mm: float = 5.0
    retreat_mm: float | None = None  # Optional fixed-distance override.
    blocked_hold_steps_min: int = 1
    blocked_hold_steps_max: int = 2
    blocked_extra_press_mm: float = 0.2
    tactile_threshold_floor: float = 0.01
    tactile_threshold_std_scale: float = 3.0
    blocked_persistence_steps: int = 3
    progress_ratio_threshold: float = 0.2
    stable_window_steps: int = 8
    max_probe_mm: float = 15.0
    max_attempts: int = 1000
    seed: int = 2000000
