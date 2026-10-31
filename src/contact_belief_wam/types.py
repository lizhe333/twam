from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ContactObservation:
    image: Any
    state: Any
    tactile: Any
    tokens: Any = None
    token_mask: Any = None
    belief: Any = None
    expected_contact: Any = None
    expected_valid: Any = None
    executed_actions: Any = None
    executed_valid: Any = None
    belief_valid: Any = None
    step: Any = None
    # Metadata used for task-preserving Stage-1 ranking negatives. It is not a
    # model input and is intentionally absent from deployment visual encoding.
    task: Any = None
    world_state: Any = None
    wrist_image: Any = None


@dataclass
class ContactTargets:
    actions: Any
    action_mask: Any
    future_image: Any
    future_tactile: Any
    future_mask: Any
    # Optional per-policy-target mask, combined with action_mask for flow loss.
    # World-model losses continue to use all real actions.
    policy_action_mask: Any = None
    world_actions: Any = None
    future_wrist_image: Any = None


@dataclass
class ContactPrediction:
    actions: Any
    belief: Any
    expected_contact: Any
    target_step: Any
    diagnostics: dict
