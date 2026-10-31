"""Safety and result-integrity checks for the 4090 eight-task monitor."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
spec = importlib.util.spec_from_file_location('watch_4090_tactile_8task',
                                               PROJECT / 'scripts/watch_4090_tactile_8task.py')
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


@pytest.mark.parametrize('error_seed', [None, 1000001])
def test_environment_errors_are_not_reported_as_model_success_rate(tmp_path, error_seed):
    directory = tmp_path / 'result'
    directory.mkdir()
    rows = {str(seed): dict(result='success' if seed == 1000000 else 'failed',
        policy_execution=dict(input_alignment=monitor.aligned.PROFILE))
        for seed in range(1000000, 1000003)}
    if error_seed is not None:
        rows[str(error_seed)] = dict(result='error', failure_reason='pre_move failed')
    (directory / 'metadata.json').write_text(json.dumps(rows))
    result = dict(status='completed', native_result_dir=str(directory), successes=1,
                  completed_episodes=3, success_rate=1 / 3, exceptions=int(error_seed is not None))
    batch = SimpleNamespace(read=lambda path: json.loads(Path(path).read_text()),
                            write=lambda path, value: Path(path).write_text(json.dumps(value)))
    item = dict(job={})
    out = monitor.audited_finish(lambda *args: result, batch, item, tmp_path, 3, 0)
    assert out['formal_result'] is (error_seed is None)
    assert out['environment_error_seeds'] == ([] if error_seed is None else [error_seed])
    assert out['valid_model_episodes'] == (3 if error_seed is None else 2)
    assert out['success_rate'] == (1 / 3 if error_seed is None else None)
    assert item['job']['result'] == json.loads((directory / 'result.json').read_text())


def test_unaligned_policy_inputs_are_rejected(tmp_path):
    directory = tmp_path / 'result'
    directory.mkdir()
    (directory / 'metadata.json').write_text(json.dumps({'1000000': {
        'result': 'success', 'policy_execution': {'input_alignment': 'old'}}}))
    batch = SimpleNamespace(read=lambda path: json.loads(Path(path).read_text()),
                            write=lambda path, value: Path(path).write_text(json.dumps(value)))
    result = dict(status='completed', native_result_dir=str(directory))
    item = dict(job={}, work=tmp_path / 'attempt01')
    rejected = monitor.audited_finish(lambda *args: result, batch, item, tmp_path, 1, 0)
    assert rejected['status'] == 'incomplete_or_error'
    assert rejected['formal_result'] is False
    assert 'Unexpected model input alignment' in rejected['reason']
    assert Path(rejected['native_result_dir']).is_dir()
    assert not directory.exists()


def test_reference_requires_identical_checkpoint_and_all_fixed_seeds(tmp_path):
    suite = tmp_path / 'new-suite'
    suite.mkdir()
    old = tmp_path / 'outputs/eval_suites' / monitor.REFERENCE
    result_dir = old / 'results/insert_hole'
    result_dir.mkdir(parents=True)
    job = dict(task='insert_hole', model_sha256='model', ema_sha256='ema',
               training_identity='identity', step=10000, camera='head_only')
    prior = dict(job, result=dict(native_result_dir=str(result_dir)))
    (old / 'manifest.json').write_text(json.dumps(dict(episodes=50, seed_start=1000000,
        seed_end=1000049, camera='head_only', jobs=[prior])))
    (result_dir / 'result.json').write_text(json.dumps(dict(status='completed',
        completed_episodes=50, successes=20, exceptions=0)))
    metadata = {str(seed): dict(result='success' if seed < 1000020 else 'failed')
                for seed in range(1000000, 1000050)}
    (result_dir / 'metadata.json').write_text(json.dumps(metadata))
    batch = SimpleNamespace(read=lambda path: json.loads(Path(path).read_text()),
                            write=lambda path, value: Path(path).write_text(json.dumps(value)))
    monitor.pin_a800_reference(tmp_path, suite, [job], batch)
    pinned = json.loads((suite / 'a800_reference.json').read_text())
    assert pinned['tasks']['insert_hole']['successes'] == 20
    with pytest.raises(ValueError, match='checkpoint mismatch'):
        monitor.pin_a800_reference(tmp_path, suite, [dict(job, ema_sha256='different')], batch)
    del metadata['1000010']
    (result_dir / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='incomplete A800'):
        monitor.pin_a800_reference(tmp_path, suite, [job], batch)


def test_checkpoint_is_hardlinked_while_busy_and_hashed_before_launch(tmp_path, monkeypatch):
    source = tmp_path / 'train/pretrained_model'
    reference = tmp_path / 'old-checkpoint'
    source.mkdir(parents=True)
    reference.mkdir()
    artifacts = ('model.safetensors', 'ema.safetensors', 'config.json')
    for name in artifacts:
        (source / name).write_bytes(name.encode())
        os.link(source / name, reference / name)
    for name in ('complete.json', 'training_contract.json', 'run_identity.json'):
        (reference / name).write_text('{}')
    digest = lambda name: hashlib.sha256(name.encode()).hexdigest()
    job = dict(id='test-job', task='insert_hole', run='run', source=str(source), step=10000,
        camera='head_only', model_sha256=digest('model.safetensors'),
        ema_sha256=digest('ema.safetensors'), training_identity='identity')
    previous = dict(job, checkpoint=str(reference))
    pin = dict(job_id=job['id'], identity=job['training_identity'],
               model_sha256=job['model_sha256'], ema_sha256=job['ema_sha256'])
    (reference / 'pinned.json').write_text(json.dumps(pin))
    batch = SimpleNamespace(read=lambda path: json.loads(Path(path).read_text()),
                            write=lambda path, value: Path(path).write_text(json.dumps(value)))
    runner = SimpleNamespace(batch=batch, ARTIFACTS=artifacts)
    with monkeypatch.context() as temporary:
        temporary.setattr(monitor.watch, 'digest', lambda path: pytest.fail('Weight bytes read while GPU busy'))
        monitor.pin_existing_checkpoint(job, tmp_path / 'new', runner, {'insert_hole': previous})
    assert os.path.samefile(Path(job['checkpoint']) / 'ema.safetensors', source / 'ema.safetensors')
    monitor.verify_checkpoint(job)
    (source / 'ema.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum changed'):
        monitor.verify_checkpoint(job)
