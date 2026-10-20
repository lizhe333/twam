import json

import pytest

from contact_belief_wam.cli import _verify_evaluation_results


def test_evaluation_summary_requires_every_requested_seed_to_complete(tmp_path):
    summary_path = tmp_path / "evaluation.json"
    summary_path.write_text(json.dumps({"episodes": [
        {"seed": 10, "complete_rollout": True},
        {"seed": 11, "complete_rollout": False},
    ]}))

    with pytest.raises(RuntimeError, match="11 \\(missing 1\\)"):
        _verify_evaluation_results(summary_path, [10, 11])


def test_evaluation_summary_accepts_complete_requested_seeds(tmp_path):
    summary_path = tmp_path / "evaluation.json"
    summary_path.write_text(json.dumps({"episodes": [
        {"seed": 10, "complete_rollout": True},
        {"seed": 11, "complete_rollout": True},
    ]}))

    _verify_evaluation_results(summary_path, [10, 11])
