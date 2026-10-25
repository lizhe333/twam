"""Regression for restoring CUDA RNG before the first CUDA operation."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from contact_belief_wam.trainer import rng_state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fresh_process_restore_survives_queued_manual_seed(tmp_path):
    torch.cuda.init()
    torch.manual_seed(917)
    state = rng_state()
    probe = torch.rand(4, device="cuda").cpu()
    checkpoint = tmp_path / "rng.pt"
    torch.save({"state": state, "probe": probe}, checkpoint)
    script = """
import sys, torch
from contact_belief_wam.trainer import restore_rng
saved = torch.load(sys.argv[1], weights_only=False)
assert not torch.cuda.is_initialized()
torch.manual_seed(42)
restore_rng(saved['state'])
torch.testing.assert_close(torch.rand(4, device='cuda').cpu(), saved['probe'], rtol=0, atol=0)
"""
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root / "vendor/wave0/src"),
                                        env.get("PYTHONPATH", "")])
    subprocess.run([sys.executable, "-c", script, str(checkpoint)], env=env, check=True,
                   capture_output=True, text=True, timeout=120)
