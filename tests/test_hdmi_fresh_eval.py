"""No-EMA checkpoint and serving selection for the completed HDMI fine-tune."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
import eval_hdmi_fresh as fresh


def test_raw_checkpoint_can_be_pinned_without_ema(tmp_path):
    runner = fresh.watch.load_runner(PROJECT.parent)
    write = runner.batch.write
    run = tmp_path / 'run/full'
    ck = run / 'tasks/insert_HDMI/stage2/checkpoints/step_00010000'
    source = ck / 'pretrained_model'
    source.mkdir(parents=True)
    for name in runner.ARTIFACTS:
        if name != 'ema.safetensors':
            (source / name).write_bytes(b'test')
    contract = dict(stage='stage2', task='insert_HDMI', variant='full', mode='full', steps=10000,
                    expert_only=False, data_mix='2:1', ema_enabled=False, finetune_use_ema=False,
                    stage1_checkpoint='wm5k')
    write(ck / 'training_contract.json', contract)
    write(ck / 'run_identity.json', dict(contract=contract, fingerprint='policy10k'))
    write(ck / 'complete.json', dict(stage='stage2', step=10000, training_contract=contract,
                                    identity='policy10k', ema_sha256=None,
                                    model_sha256=fresh.watch.digest(source / 'model.safetensors')))
    write(run / 'shared_wm/stage1/checkpoints/step_00005000/complete.json',
          dict(stage='stage1', step=5000, identity='wm5k'))
    write(source / 'config.json', dict(camera_mode='head_only', k=10, independent_tactile=True))
    deploy = dict(task='insert_HDMI', variant='full', mode='full', aligned=True, camera='head_only',
                  simulator_action='absolute_joint', action_chunk_length=50,
                  executed_action_prefix=10, use_ema=False, ema_file=None)
    write(source / 'deployment.json', deploy)
    job = fresh.discover(run, 'insert_HDMI', runner)
    fresh.snapshot(job, tmp_path / 'suite', runner)
    assert Path(job['checkpoint'], 'model.safetensors').is_file()
    assert not Path(job['checkpoint'], 'ema.safetensors').exists()
    deploy['use_ema'] = True
    write(source / 'deployment.json', deploy)
    with pytest.raises(ValueError, match='deployment'):
        fresh.discover(run, 'insert_HDMI', runner)


def test_no_ema_option_reaches_model_service(monkeypatch):
    from contact_belief_wam import cli, serve
    received = []
    monkeypatch.setattr(serve, 'serve', lambda **kwargs: received.append(kwargs))
    cli.main(['serve', '--checkpoint', '/tmp/fixture', '--no-ema'])
    assert received[-1]['use_ema'] is False
    cli.main(['serve', '--checkpoint', '/tmp/fixture'])
    assert 'use_ema' not in received[-1]  # Existing defaults remain unchanged.


def test_runtime_changes_only_ema_selection_in_private_native_bridge(tmp_path):
    runner = fresh.watch.load_runner(PROJECT.parent)
    for name in ['twam/src/contact_belief_wam/model.py', 'twam/inference/run_native.py',
                 'twam/vendor/wave0/src/model.py', 'twam/pyproject.toml', 'twam/scripts/cloud_train.sh',
                 'UniVTAC/scripts/eval_policy.py']:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fixture\n')
    (tmp_path / 'twam/vendor/transformers').mkdir()
    original = (PROJECT.parent / 'UniVTAC/policy/TactileWave0Wam/deploy_policy.py').read_text()
    bridge = tmp_path / 'UniVTAC/policy/TactileWave0Wam/deploy_policy.py'
    bridge.parent.mkdir(parents=True)
    bridge.write_text(original)
    runtime, record = fresh.aligned.pin_runtime(tmp_path, tmp_path / 'suite', runner.batch, use_ema=False)
    updated = (runtime / 'native_policy_original.py').read_text()
    assert record['use_ema'] is False and bridge.read_text() == original
    assert updated.count('use_ema=False') == 3 and "'--no-ema'" in updated
    assert fresh.aligned.verify_runtime(runtime) == record
    with pytest.raises(ValueError, match='EMA selection'):
        fresh.aligned.pin_runtime(tmp_path, tmp_path / 'suite', runner.batch, use_ema=True)
