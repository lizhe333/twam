"""Review-only label contracts for a bounded insert_hole perturbation preview."""
from __future__ import annotations

import numpy as np


def label_ranges(sources, phases, decisions, *, success, kind, perturb_start=None,
                 perturb_stop=None, takeover=None, teacher_events=()):
    """Separate executed context from BC; never infer perturbation eligibility."""
    n = len(sources)
    if not (len(phases) == len(decisions) == n):
        raise ValueError("Action metadata lengths differ")
    flags = np.zeros(n, bool)
    if success:
        if kind == "normal":
            flags[:] = np.asarray(sources) == "wave0"
        elif kind == "self_corrected":
            if perturb_start is None or perturb_stop is None:
                raise ValueError("Perturbed success needs its actual perturbation interval")
            flags[:perturb_start] = True
            flags[perturb_stop:] = np.asarray(sources[perturb_stop:]) == "wave0"
        elif kind == "oracle_recovery":
            if takeover is None or (perturb_stop is not None and takeover < perturb_stop):
                raise ValueError("Recovery must follow a completed perturbation when present")
            # The whole pre-Teacher branch remains context and WM-only. Normal
            # behavior is supervised by the paired, unperturbed success.
            for event in teacher_events:
                if event.get("bc_eligible") is True:
                    a, b = int(event["step"]), int(event["stop"])
                    if not takeover <= a < b <= n:
                        raise ValueError("Invalid Teacher interval")
                    flags[a:b] = True
        else:
            raise ValueError("Unknown preview kind")
    flags[np.asarray(sources) == "perturb"] = False
    ranges = []
    for i in range(n):
        if not flags[i]:
            continue
        # Split even successful WAVE-0 labels at every new feedback decision.
        if (ranges and ranges[-1]["stop"] == i and
                ranges[-1]["decision_step"] == int(decisions[i]) and
                ranges[-1]["phase"] == str(phases[i])):
            ranges[-1]["stop"] = i + 1
        else:
            if int(decisions[i]) != i:
                raise ValueError("Eligible interval does not start at its decision observation")
            ranges.append(dict(start=i, stop=i+1, decision_step=i,
                               phase=str(phases[i]), reason=kind))
    return ranges, flags


def validate_preview_labels(sources, ranges, perturb_start=None, perturb_stop=None):
    """Defense in depth for both direct labels and 50-action look-ahead windows."""
    from .takeover import policy_prefix
    n = len(sources)
    commands = np.arange(n * 8, dtype=np.float32).reshape(n, 8)
    perturb = np.asarray(sources) == "perturb"
    if perturb_start is not None:
        expected = np.zeros(n, bool)
        expected[perturb_start:perturb_stop] = True
        if not np.array_equal(perturb, expected):
            raise ValueError("Perturbation provenance disagrees with actual commands")
    for t in range(n):
        target, mask = policy_prefix(commands, t, ranges, 50)
        count = int(mask.sum())
        if np.any(perturb[t:t+count]):
            raise ValueError("Perturbation entered a policy target")
        changed = commands.copy()
        changed[t+count:] += 123456
        altered, altered_mask = policy_prefix(changed, t, ranges, 50)
        if not np.array_equal(target, altered) or not np.array_equal(mask, altered_mask):
            raise ValueError("Excluded future commands leaked into policy input")
    return dict(passed=True, anchors=n, perturb_actions=int(perturb.sum()),
                perturb_bc_labels=0, excluded_future_poison_checks=n)
