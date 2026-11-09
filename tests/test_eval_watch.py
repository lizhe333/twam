"""Committed-checkpoint discovery and persistent native-evaluation scheduling."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/watch_eval_10k.py'
spec = importlib.util.spec_from_file_location('watch_eval_10k', SCRIPT)
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


@pytest.fixture
def runner():
    return watch.load_runner(watch.ROOT)


@pytest.fixture
def args(tmp_path):
    return SimpleNamespace(root=tmp_path, run_root=tmp_path / 'training/full',
        suite=tmp_path / 'eval_suites/test', gpu='0', episodes=2, poll_seconds=.000001,
        max_attempts=2, timeout=100, stall_timeout=100, scan_only=False,
        model_python=Path('/model/python'))


def checkpoint(args, runner, task='insert_hole', step=10000, partial=False):
    path = args.run_root / 'tasks' / task / 'stage2/checkpoints' / f'step_{step:08d}'
    if partial:
        path = path.with_name(path.name + '.partial')
    source = path / 'pretrained_model'
    source.mkdir(parents=True)
    for name in runner.ARTIFACTS:
        (source / name).write_bytes(b'fixture inference artifact')
    contract = dict(stage='stage2', task=task, variant='full', mode='full', steps=10000,
                    expert_only=True, data_mix='1:0')
    write = runner.batch.write
    write(path / 'training_contract.json', contract)
    write(path / 'run_identity.json', dict(fingerprint='fixture-identity', contract=contract))
    write(path / 'complete.json', dict(stage='stage2', step=step, identity='fixture-identity',
        training_contract=contract, model_sha256=watch.digest(source / 'model.safetensors'),
        ema_sha256=watch.digest(source / 'ema.safetensors')))
    write(source / 'config.json', dict(camera_mode='head_only', k=10, independent_tactile=True, belief_bptt_steps=2))
    write(source / 'deployment.json', dict(task=task, variant='full', mode='full', aligned=True,
        camera='head_only', simulator_action='absolute_joint', action_chunk_length=50,
        executed_action_prefix=10, use_ema=True, ema_file='ema.safetensors'))
    return path


def test_only_complete_10k_checkpoints_enter_the_queue(args, runner):
    checkpoint(args, runner, step=9000)
    checkpoint(args, runner, partial=True)
    manifest = watch.prepare(args, runner)
    watch.refresh(args, manifest, runner)
    assert manifest['jobs'] == []
    final = checkpoint(args, runner)
    (final / 'complete.json').unlink()
    assert watch.discover_task(args.run_root, 'insert_hole', runner) is None


def test_snapshot_deduplicates_survives_pruning_and_preserves_completed(args, runner):
    source = checkpoint(args, runner)
    (source / 'optimizer.rank00.pt').write_bytes(b'never pin optimizer')
    manifest = watch.prepare(args, runner)
    watch.refresh(args, manifest, runner)
    first = copy.deepcopy(manifest['jobs'][0])
    watch.refresh(args, manifest, runner)
    assert manifest['jobs'] == [first]
    pinned = Path(first['checkpoint'])
    assert not (pinned / 'optimizer.rank00.pt').exists()
    assert (pinned / 'model.safetensors').stat().st_ino == (source / 'pretrained_model/model.safetensors').stat().st_ino
    shutil.rmtree(source)
    assert (pinned / 'ema.safetensors').read_bytes() == b'fixture inference artifact'
    result = dict(status='completed', completed_episodes=2, successes=1,
                  evaluated_checkpoint=str(pinned), evaluation_protocol=runner.batch.EVAL_PROTOCOL)
    runner.batch.write(args.suite / 'results' / first['id'] / 'result.json', result)
    recovered = watch.prepare(args, runner)
    assert recovered['jobs'][0]['state'] == 'completed'
    watch.refresh(args, recovered, runner)
    assert len(recovered['jobs']) == 1 and recovered['jobs'][0]['result'] == result


@pytest.mark.parametrize('field', ['camera', 'identity', 'step', 'ema'])
def test_reject_wrong_or_corrupt_checkpoint(args, runner, field):
    source = checkpoint(args, runner)
    if field == 'camera':
        file = source / 'pretrained_model/deployment.json'
        values = runner.batch.read(file)
        values['camera'] = 'head_wrist'
        runner.batch.write(file, values)
    elif field in ('identity', 'step'):
        file = source / 'complete.json'
        values = runner.batch.read(file)
        values[field] = 'wrong' if field == 'identity' else 9000
        runner.batch.write(file, values)
    else:
        (source / 'pretrained_model/ema.safetensors').write_bytes(b'corrupted')
    manifest = watch.prepare(args, runner)
    with pytest.raises(ValueError):
        watch.refresh(args, manifest, runner)
    assert manifest['jobs'] == []
    assert not list((args.suite / 'checkpoints').glob('**/*.partial'))


def test_gpu_with_training_process_is_not_claimed_even_with_free_memory(args, runner, monkeypatch):
    args.suite.mkdir(parents=True)
    info = dict(uuid='GPU-test', name='A800', free_mib=60000, total_mib=80000)
    monkeypatch.setattr(runner.batch, 'gpu_info', lambda gpu: info)
    monkeypatch.setattr(watch.subprocess, 'check_output', lambda *a, **kw: '12345\n')
    lock, _, reason = watch.claim_gpu(args, runner)
    assert lock is None and 'existing compute processes' in reason
    monkeypatch.setattr(watch.subprocess, 'check_output', lambda *a, **kw: '')
    lock, actual, reason = watch.claim_gpu(args, runner)
    assert actual == info and reason is None
    # The same GPU cannot be claimed by a second evaluation queue.
    another, _, _ = watch.claim_gpu(args, runner)
    assert another is None
    lock.close()


def test_scan_only_never_queries_gpu_or_launches(args, runner, monkeypatch):
    args.scan_only = True
    monkeypatch.setattr(watch, 'claim_gpu', lambda *a: pytest.fail('GPU requested in scan-only mode'))
    monkeypatch.setattr(runner, 'launch', lambda *a: pytest.fail('Worker launched in scan-only mode'))
    manifest = watch.prepare(args, runner)
    assert watch.run_monitor(args, manifest, runner) == 0
    assert manifest['status'] == 'scanned'


def test_resume_keeps_selection_and_does_not_touch_foreign_live_worker(args, runner, monkeypatch):
    checkpoint(args, runner)
    manifest = watch.prepare(args, runner)
    watch.refresh(args, manifest, runner)
    manifest['jobs'][0].update(state='running', worker_host='a-different-host', pid=12, worker_start='token')
    runner.batch.save_summary(args.suite, manifest)
    monkeypatch.setattr(watch.os, 'killpg', lambda *a: pytest.fail('Stopped an unrelated process'))
    with pytest.raises(RuntimeError, match='resume the monitor on that host'):
        watch.prepare(args, runner)
    args.episodes = 50
    with pytest.raises(ValueError, match='another training run'):
        watch.prepare(args, runner)


@pytest.mark.parametrize('first_fails', [False, True])
def test_new_checkpoint_is_enqueued_during_eval_and_failures_are_bounded(args, runner, monkeypatch, first_fails):
    monkeypatch.setattr(watch, 'TASKS', ('insert_hole', 'insert_HDMI'))
    checkpoint(args, runner)
    manifest = watch.prepare(args, runner)
    commands, scans, preflights = [], [], []
    refresh = watch.refresh
    def scan(*values):
        refresh(*values)
        scans.append([(j['task'], j['state']) for j in values[1]['jobs']])
    monkeypatch.setattr(watch, 'refresh', scan)
    monkeypatch.setattr(watch, 'claim_gpu', lambda *a: (io.StringIO(), {'uuid': 'GPU-test'}, None))
    monkeypatch.setattr(runner.batch, 'preflight', lambda *a: preflights.append(a))
    monkeypatch.setattr(runner.batch, 'free_port', lambda start: start)
    monkeypatch.setattr(runner.batch, 'stop_group', lambda process: None)
    def sleep(_):
        if not (args.run_root / 'tasks/insert_HDMI').exists():
            checkpoint(args, runner, task='insert_HDMI')
    monkeypatch.setattr(watch.time, 'sleep', sleep)
    class Process:
        pid = 123456789
        def __init__(self, failed):
            self.failed, self.polls = failed, 0
        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else (1 if self.failed else 0)
    def popen(command, **options):
        commands.append(command)
        task = command[3]
        env = options['env']
        failed = first_fails and task == 'insert_hole'
        if not failed:
            native = Path(env['PI05_EVAL_OUTPUT_ROOT']) / 'episode-results'
            runner.batch.write(native / 'metadata.json', {'1000000': {'result': 'success'}, '1000001': {'result': 'failed'}})
            options['stdout'].write('Expert preflight passed;\n'
                f'Evaluation protocol: {runner.batch.EVAL_PROTOCOL}\n'
                f'Evaluation directory: {native}\nFinal Result: 1/2\n')
        else:
            options['stdout'].write('simulated infrastructure error\n')
        options['stdout'].flush()
        return Process(failed)
    monkeypatch.setattr(runner.subprocess, 'Popen', popen)
    assert watch.run_monitor(args, manifest, runner) == int(first_fails)
    assert any(rows == [('insert_hole', 'running'), ('insert_HDMI', 'pending')] for rows in scans)
    assert len(preflights) == 1
    assert len(commands) == (3 if first_fails else 2)
    assert all(c[c.index('--start_seed') + 1] == '1000000' for c in commands)
    assert [j['state'] for j in manifest['jobs']] == (['failed', 'completed'] if first_fails else ['completed', 'completed'])
    hdmi = manifest['jobs'][1]
    assert (args.suite / 'results' / hdmi['id'] / 'metadata.json').exists()
    attempts = [j['attempts'] for j in manifest['jobs']]
    recovered = watch.prepare(args, runner)
    assert [j['attempts'] for j in recovered['jobs']] == attempts
    assert watch.run_monitor(args, recovered, runner) == int(first_fails)
    assert len(commands) == sum(attempts)  # No repeated completed evaluations.
