#!/usr/bin/env python3
"""Wait for a local RTX 4090 and run the eight aligned Isaac 4.5 evaluations.

This supervisor does not stop or signal other GPU jobs. It pins the tactile
weights and inference code, then lets the existing fixed-seed queue run one
task at a time when the GPU lock, process list and free-memory checks permit.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil

import reeval_hdmi_aligned as aligned
import watch_eval_10k as watch


SUITE = 'twam-single-view'
REFERENCE = 'baidu-tactile-head-only-mb16-003-eval50-sm80'
CONTRACT = 'rtx4090-isaac45-tactile-head-only-eight-tasks-v1'


def external_sources(root):
    """Record the live task/bridge code that the pinned evaluator imports."""
    paths = list((root / 'UniVTAC/envs').rglob('*.py'))
    paths.extend((root / 'UniVTAC/instructions').glob('*.json'))
    paths.extend(root / name for name in (
        'UniVTAC/task_config/clean.yml',
        'UniVTAC/policy/Pi05Wam/deploy_policy.py',
        'pi05-univtac/src/pi05_univtac/sim_policy.py',
        'pi05-univtac/src/pi05_univtac/constants.py',
        'pi05-univtac/scripts/eval-worker.sh',
        'pi05-univtac/scripts/sim-env.sh',
        'pi05-univtac/scripts/env.sh',
    ))
    return {str(path.relative_to(root)): watch.digest(path) for path in sorted(paths)}


def pin_runtime(root, suite, batch):
    runtime = suite / 'runtime'
    if runtime.exists():
        record = aligned.verify_runtime(runtime)
        if record.get('monitor_contract') != CONTRACT:
            raise ValueError('This runtime belongs to a different evaluation contract')
        if external_sources(root) != record['external_sources']:
            raise ValueError('Simulator/task source changed since preparation')
        return runtime, record
    temporary = suite / 'runtime.partial'
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    project = root / 'twam'
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache', '.git')
    try:
        for name in ('src', 'inference', 'vendor/wave0', 'vendor/transformers'):
            shutil.copytree(project / name, temporary / name, ignore=ignore)
        for name in ('pyproject.toml', 'scripts/cloud_train.sh'):
            (temporary / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(project / name, temporary / name)
        shutil.copy2(root / 'UniVTAC/policy/TactileWave0Wam/deploy_policy.py',
                     temporary / 'native_policy_original.py')
        shutil.copy2(root / 'UniVTAC/scripts/eval_policy.py', temporary / 'native_eval.py')
        files = aligned.inventory(temporary)
        external = external_sources(root)
        fingerprint = hashlib.sha256(json.dumps(dict(files=files, external=external),
                                                sort_keys=True).encode()).hexdigest()
        record = dict(profile=aligned.PROFILE, monitor_contract=CONTRACT,
                      created_at=watch.now(), files=files, external_sources=external,
                      fingerprint=fingerprint)
        batch.write(temporary / 'runtime_manifest.json', record)
        temporary.rename(runtime)
        return runtime, record
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def pin_a800_reference(root, suite, jobs, batch):
    source = root / 'outputs/eval_suites' / REFERENCE
    manifest = batch.read(source / 'manifest.json')
    expected_seeds = {str(seed) for seed in range(1000000, 1000050)}
    if (manifest.get('episodes') != 50 or manifest.get('seed_start') != 1000000
            or manifest.get('seed_end') != 1000049 or manifest.get('camera') != 'head_only'):
        raise ValueError('The A800 reference does not have the expected seed/camera scope')
    reference = {}
    for job in jobs:
        previous, = [item for item in manifest['jobs'] if item['task'] == job['task']]
        for key in ('model_sha256', 'ema_sha256', 'training_identity', 'step', 'camera'):
            if previous.get(key) != job.get(key):
                raise ValueError(f'{job["task"]}: A800 checkpoint mismatch: {key}')
        directory = Path(previous['result']['native_result_dir'])
        result = batch.read(directory / 'result.json')
        metadata = batch.read(directory / 'metadata.json')
        outcomes = {seed: metadata[seed]['result'] for seed in expected_seeds if seed in metadata}
        if (result.get('status') != 'completed' or result.get('completed_episodes') != 50
                or set(outcomes) != expected_seeds):
            raise ValueError(f'{job["task"]}: incomplete A800 fixed-seed result')
        reference[job['task']] = dict(result_file=str(directory / 'result.json'),
            successes=result['successes'], environment_errors=result.get('exceptions', 0),
            outcomes=outcomes, ema_sha256=job['ema_sha256'])
    record = dict(source=str(source), simulator='Isaac Sim 4.5',
        input_alignment='legacy-raw-tactile-commanded-history',
        comparison_note='Same checkpoint and scene seed; 4090 uses aligned tactile and measured history, '
                        'so the two success rates do not isolate GPU differences.', tasks=reference)
    destination = suite / 'a800_reference.json'
    if destination.exists() and batch.read(destination) != record:
        raise ValueError('The pinned A800 reference changed')
    batch.write(destination, record)


def pin_existing_checkpoint(job, suite, runner, reference_jobs):
    """Hard-link the previously checked A800 checkpoint without rereading 230 GiB.

    Full hashes are checked again after the GPU is idle and just before each
    task is launched, avoiding a large read while the current GPU job is busy.
    """
    previous = reference_jobs[job['task']]
    for key in ('model_sha256', 'ema_sha256', 'training_identity', 'step', 'camera'):
        if previous.get(key) != job.get(key):
            raise ValueError(f'{job["task"]}: A800 checkpoint mismatch: {key}')
    reference = Path(previous['checkpoint'])
    old_pin = reference / 'pinned.json'
    expected = dict(job_id=job['id'], identity=job['training_identity'],
                    model_sha256=job['model_sha256'], ema_sha256=job['ema_sha256'])
    if runner.batch.read(old_pin) != expected:
        raise ValueError(f'{job["task"]}: missing previously verified checkpoint identity')
    pin_time = old_pin.stat().st_mtime_ns
    source = Path(job['source'])
    for name in ('model.safetensors', 'ema.safetensors'):
        old = reference / name
        if not old.is_file() or not os.path.samefile(old, source / name) or old.stat().st_mtime_ns > pin_time:
            raise ValueError(f'{job["task"]}: checkpoint changed since the A800 snapshot: {name}')
    target = suite / 'checkpoints' / job['task'] / job['run'] / '10000'
    if not target.exists():
        temporary = target.with_name('10000.partial')
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        try:
            for name in (*runner.ARTIFACTS, 'complete.json', 'training_contract.json', 'run_identity.json'):
                # These directories are on the shared workspace filesystem.
                # Do not copy large weights into another filesystem while another job is running.
                os.link(reference / name, temporary / name)
            runner.batch.write(temporary / 'pinned.json', expected)
            temporary.rename(target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    if runner.batch.read(target / 'pinned.json') != expected:
        raise ValueError(f'{job["task"]}: pinned checkpoint identity changed')
    for name in runner.ARTIFACTS:
        if not (target / name).is_file() or not (target / name).stat().st_size:
            raise ValueError(f'{job["task"]}: missing checkpoint artifact: {name}')
    for name in ('model.safetensors', 'ema.safetensors'):
        if not os.path.samefile(target / name, reference / name):
            raise ValueError(f'{job["task"]}: pinned weights no longer match A800: {name}')
    job['checkpoint'] = str(target)


def verify_checkpoint(job):
    """Verify the actual weight bytes while holding the idle GPU's lock."""
    checkpoint = Path(job['checkpoint'])
    for name, key in (('model.safetensors', 'model_sha256'), ('ema.safetensors', 'ema_sha256')):
        if watch.digest(checkpoint / name) != job[key]:
            raise ValueError(f'{job["task"]}: pinned checkpoint checksum changed: {name}')


def audited_finish(base_finish, batch, item, suite, episodes, code):
    result = base_finish(item, suite, episodes, code)
    if result['status'] != 'completed':
        return result
    directory = Path(result['native_result_dir'])
    metadata = batch.read(directory / 'metadata.json')
    rows = {str(seed): metadata[str(seed)] for seed in range(1000000, 1000000 + episodes)}
    bad = [seed for seed, row in rows.items() if row.get('result') != 'error'
           and row.get('policy_execution', {}).get('input_alignment') != aligned.PROFILE]
    if bad:
        # The common runner has already atomically published this attempt.
        # Move it back to the attempt directory and mark it invalid so a retry
        # can publish a fresh result without hiding the diagnostic artifacts.
        work = item['work']
        work.mkdir(parents=True, exist_ok=True)
        rejected = work / 'rejected-result'
        directory.rename(rejected)
        result.update(status='incomplete_or_error', formal_result=False, success_rate=None,
                      native_result_dir=str(rejected),
                      reason=f'Unexpected model input alignment for seeds: {bad}')
        batch.write(rejected / 'result.json', result)
        item['job'].update(state='incomplete_or_error', result=result,
                           log=str(rejected / 'eval.log'))
        return result
    errors = sorted(int(seed) for seed, row in rows.items() if row.get('result') == 'error')
    result.update(simulator='Isaac Sim 4.5', gpu_type='RTX 4090', input_alignment=aligned.PROFILE,
                  environment_error_seeds=errors, valid_model_episodes=episodes - len(errors),
                  formal_result=not errors)
    if errors:
        result['observed_rate_including_environment_errors'] = result.get('success_rate')
        result['success_rate'] = None
    batch.write(directory / 'result.json', result)
    item['job']['result'] = result
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=watch.ROOT)
    parser.add_argument('--suite', type=Path)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--poll-seconds', type=float, default=10)
    parser.add_argument('--idle-seconds', type=float, default=60)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0 or args.idle_seconds < 0 or not args.gpu.isdigit():
        parser.error('Use one GPU index, a positive poll interval, and nonnegative idle time')
    args.root = args.root.resolve()
    args.run_root = args.root / 'twam/runs' / watch.RUN / 'full'
    args.suite = (args.suite or args.root / 'outputs/eval_suites' / SUITE).resolve()
    if args.suite.is_relative_to(args.run_root):
        parser.error('Keep evaluation outputs outside the training run')
    manifest_path = args.suite / 'manifest.json'
    if manifest_path.exists():
        archived = json.loads(manifest_path.read_text())
        if (archived.get('archived') and archived.get('jobs')
                and all(job['state'] == 'completed' for job in archived['jobs'])):
            print(json.dumps(dict(status=archived['status'], suite=str(args.suite),
                                  archived=True, completed_tasks=len(archived['jobs'])), indent=2))
            return 0
    args.episodes, args.max_attempts = 50, 2
    args.timeout, args.stall_timeout = 43200, 1800
    args.scan_only = args.prepare_only
    args.model_python = args.root / 'contact-belief-wam/.venv/bin/python'
    sim_prefix = args.root / '.conda/univtac-sim'
    for executable in (args.model_python, sim_prefix / 'bin/python'):
        if not os.access(executable, os.X_OK):
            raise RuntimeError(f'Required Python environment unavailable: {executable}')
    # Never inherit the A800-specific native override or another Isaac install.
    for key in ('PYTHONHOME', 'PYTHONPATH', 'PI05_SIM_NATIVE_ROOT', 'LD_LIBRARY_PATH'):
        os.environ.pop(key, None)
    os.environ.update(TWAM_ROOT=str(args.root), SIM_PREFIX=str(sim_prefix),
        PI05_MODEL_LD_LIBRARY_PATH='',
        PI05_SIM_GRAPHICS_LIBRARY_PATH=str(args.root / '.cache/sim-graphics-runtime/lib'),
        PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        PATH=str(args.model_python.parent) + os.pathsep + os.environ.get('PATH', ''))
    args.suite.mkdir(parents=True, exist_ok=True)
    with (args.suite / 'queue.lock').open('a') as queue_lock:
        try:
            fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f'Queue already running: {args.suite}', flush=True)
            return 0
        runner = watch.load_runner(args.root)
        previous = runner.batch.read(args.root / 'outputs/eval_suites' / REFERENCE / 'manifest.json')
        reference_jobs = {job['task']: job for job in previous['jobs']}
        if set(reference_jobs) != set(watch.TASKS):
            raise ValueError('The A800 reference does not contain the expected eight tasks')
        watch.snapshot = lambda job, suite, current: pin_existing_checkpoint(
            job, suite, current, reference_jobs)
        runtime, record = pin_runtime(args.root, args.suite, runner.batch)
        manifest = watch.prepare(args, runner)
        if manifest.get('monitor_contract', CONTRACT) != CONTRACT:
            raise ValueError('This suite has a different monitoring contract')
        manifest.update(monitor_contract=CONTRACT, simulator='Isaac Sim 4.5',
            input_alignment=aligned.PROFILE, runtime=str(runtime),
            runtime_fingerprint=record['fingerprint'], gpu_type='RTX 4090',
            environment_errors='reported_separately_from_model_failures')
        watch.refresh(args, manifest, runner)
        if {job['task'] for job in manifest['jobs']} != set(watch.TASKS):
            raise ValueError('All eight committed tactile checkpoints are required')
        pin_a800_reference(args.root, args.suite, manifest['jobs'], runner.batch)
        runner.batch.save_summary(args.suite, manifest)
        original_finish = runner.batch.finish
        runner.batch.finish = lambda item, suite, episodes, code: audited_finish(
            original_finish, runner.batch, item, suite, episodes, code)
        def launch(*values, **kwargs):
            if external_sources(args.root) != record['external_sources']:
                raise RuntimeError('Simulator/task source changed; use a new suite')
            verify_checkpoint(values[0])
            return aligned.launch(*values, runner=runner)
        runner.launch = launch
        watch.claim_gpu = aligned.idle_gpu_guard(watch.claim_gpu, args.idle_seconds)
        if not args.scan_only:
            (args.suite / 'daemon.pid').write_text(f'{os.getpid()}\n')
        print(f'{watch.now()} WATCH 4090 eight aligned tactile tasks; '
              'seeds=1000000..1000049, 50 episodes/task', flush=True)
        code = watch.run_monitor(args, manifest, runner)
        if not args.scan_only and code == 0:
            errors = [job['task'] for job in manifest['jobs'] if not job.get('result', {}).get('formal_result')]
            if errors:
                watch.publish_state(args, manifest, runner, 'completed_with_environment_errors',
                                    tasks_with_environment_errors=errors)
                return 2
        return code


if __name__ == '__main__':
    raise SystemExit(main())
