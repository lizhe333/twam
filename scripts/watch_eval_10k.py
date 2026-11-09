#!/usr/bin/env python3
"""Watch committed TWAM 10k checkpoints and evaluate them on a separate GPU.

Uses the existing UniVTAC native fixed-seed runner and result validation. Only
inference artifacts are pinned; the training checkout and optimizer are untouched.
The monitor itself uses only the Python standard library and runs without CUDA.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
RUN = 'head-only-expert-only-8task-all-train-fast-mb16-003'
TASKS = ('insert_hole', 'insert_HDMI', 'insert_tube', 'pull_out_key',
         'grasp_classify', 'lift_bottle', 'lift_can', 'put_bottle_in_shelf')
SUITE = 'tactile-head-only-expert-mb16-003-10k-fixed-seeds'


def load_runner(root):
    path = root / 'pi05-univtac/scripts/eval-tactile-wave0-10k.py'
    spec = importlib.util.spec_from_file_location('twam_native_eval_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def discover_task(run_root, task, runner):
    checkpoint = run_root / 'tasks' / task / 'stage2/checkpoints/step_00010000'
    if not (checkpoint / 'complete.json').is_file():
        return None  # A log line, export directory, or .partial is not a commit.
    read = runner.batch.read
    complete = read(checkpoint / 'complete.json')
    contract = read(checkpoint / 'training_contract.json')
    identity = read(checkpoint / 'run_identity.json')
    if (complete.get('stage') != 'stage2' or complete.get('step') != 10000
            or contract != complete.get('training_contract')
            or identity.get('contract') != contract or identity.get('fingerprint') != complete.get('identity')
            or any(contract.get(k) != v for k, v in dict(stage='stage2', task=task, variant='full',
                mode='full', steps=10000, expert_only=True, data_mix='1:0').items())):
        raise ValueError(f'Invalid expert-only Stage-2 10k identity: {checkpoint}')
    source = checkpoint / 'pretrained_model'
    deployment, config = read(source / 'deployment.json'), read(source / 'config.json')
    expected = dict(task=task, variant='full', mode='full', aligned=True, camera='head_only',
                    simulator_action='absolute_joint', action_chunk_length=50,
                    executed_action_prefix=10, use_ema=True, ema_file='ema.safetensors')
    if any(deployment.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Invalid head-only tactile deployment: {source}')
    if (config.get('camera_mode') != 'head_only' or config.get('k') != 10
            or not config.get('independent_tactile') or config.get('belief_bptt_steps') != 2):
        raise ValueError(f'Invalid tactile model configuration: {source}')
    for name in runner.ARTIFACTS:
        if not (source / name).is_file() or (source / name).stat().st_size == 0:
            raise ValueError(f'Missing inference artifact: {source / name}')
    run = run_root.parent.name
    return dict(id=f'{task}__head-only-tactile__{run}__step10000', task=task, run=run,
                source=str(source), step=10000, camera='head_only', policy_family='tactile_wave0',
                variant='full', use_ema=True, state='pending', attempts=0,
                model_sha256=complete['model_sha256'], ema_sha256=complete['ema_sha256'],
                training_identity=complete['identity'], training_data_mix='1:0',
                backbone_precision='bfloat16', world_contact_precision='float32', tactile_device='cuda')


def snapshot(job, suite, runner):
    source = Path(job['source'])
    target = suite / 'checkpoints' / job['task'] / job['run'] / '10000'
    if not target.exists():
        temporary = target.with_name('10000.partial')
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        try:
            for name in runner.ARTIFACTS:
                runner.batch.link_or_copy(source / name, temporary / name)
            for name in ('complete.json', 'training_contract.json', 'run_identity.json'):
                runner.batch.link_or_copy(source.parent / name, temporary / name)
            for name, key in (('model.safetensors', 'model_sha256'), ('ema.safetensors', 'ema_sha256')):
                if digest(temporary / name) != job[key]:
                    raise ValueError(f'Checkpoint hash mismatch: {source / name}')
            runner.batch.write(temporary / 'pinned.json', dict(
                job_id=job['id'], identity=job['training_identity'],
                model_sha256=job['model_sha256'], ema_sha256=job['ema_sha256']))
            temporary.rename(target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    pinned = runner.batch.read(target / 'pinned.json')
    if any(pinned.get(key) != job[value] for key, value in (
            ('job_id', 'id'), ('identity', 'training_identity'),
            ('model_sha256', 'model_sha256'), ('ema_sha256', 'ema_sha256'))):
        raise ValueError(f'Existing pinned checkpoint has a different identity: {target}')
    for name in runner.ARTIFACTS:
        if not (target / name).is_file() or (target / name).stat().st_size == 0:
            raise ValueError(f'Incomplete pinned checkpoint: {target / name}')
    job['checkpoint'] = str(target)


def refresh(args, manifest, runner):
    known = {job['task'] for job in manifest['jobs']}
    for task in TASKS:
        if task in known:
            continue
        job = discover_task(args.run_root, task, runner)
        if job is not None:
            snapshot(job, args.suite, runner)
            manifest['jobs'].append(job)
            # Commit each insertion so a crash cannot lose earlier discoveries.
            runner.batch.save_summary(args.suite, manifest)
            print(f'{now()} ENQUEUE {job["task"]} step=10000', flush=True)


def process_start(pid):
    try:
        # comm may contain spaces; starttime is field 22 of proc(5).
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def recover_jobs(args, manifest, runner):
    for job in manifest['jobs']:
        result_file = args.suite / 'results' / job['id'] / 'result.json'
        if result_file.is_file():
            result = runner.batch.read(result_file)
            if (result.get('status') != 'completed' or result.get('completed_episodes') != args.episodes
                    or result.get('evaluation_protocol') != runner.batch.EVAL_PROTOCOL
                    or result.get('evaluated_checkpoint') != job.get('checkpoint')):
                raise ValueError(f'Existing evaluation result does not match this queue: {result_file}')
            job.update(state='completed', result=result)
            continue
        if job['state'] == 'completed':
            raise ValueError(f'Completed job has lost its published result: {result_file}')
        # A new monitor must not launch a duplicate while an old worker is alive.
        if job.get('state') == 'running':
            owner = job.get('worker_host')
            if owner != socket.gethostname():
                raise RuntimeError(f'Unfinished worker belongs to {owner}; resume the monitor on that host')
            pid, token = job.get('pid'), job.get('worker_start')
            if pid and token and process_start(pid) == token:
                if os.getpgid(pid) != pid:
                    raise RuntimeError(f'Cannot safely stop old worker process group: {pid}')
                os.killpg(pid, signal.SIGTERM)
                deadline = time.monotonic() + 8
                while process_start(pid) == token and time.monotonic() < deadline:
                    time.sleep(.1)
                if process_start(pid) == token:
                    os.killpg(pid, signal.SIGKILL)
            job['state'] = 'interrupted'
        if job['state'] not in ('failed',):
            job['state'] = 'pending' if job['attempts'] < args.max_attempts else 'failed'
        snapshot(job, args.suite, runner)


def prepare(args, runner):
    args.suite.mkdir(parents=True, exist_ok=True)
    selection = dict(training_root=str(args.run_root), tasks=list(TASKS), episodes=args.episodes,
                     seed_start=1000000, seed_end=1000000 + args.episodes - 1,
                     evaluation_protocol=runner.batch.EVAL_PROTOCOL, camera='head_only', step=10000)
    runtime_options = dict(skip_sim_check=getattr(args, 'skip_sim_check', False),
                           skip_driver_version_check=getattr(args, 'skip_driver_version_check', False))
    path = args.suite / 'manifest.json'
    if path.exists():
        manifest = runner.batch.read(path)
        if any(manifest.get(key) != value for key, value in selection.items()):
            raise ValueError('Existing queue uses another training run or evaluation protocol')
        previous_options = manifest.get('runtime_options', dict(skip_sim_check=False, skip_driver_version_check=False))
        if previous_options != runtime_options:
            raise ValueError('Existing queue uses different simulator startup options; choose a new suite')
        if not args.scan_only:
            recover_jobs(args, manifest, runner)
    else:
        manifest = dict(selection, schema_version=1, run=args.run_root.parent.name,
                        jobs=[], status='watching', created_at=now(), runtime_options=runtime_options)
    runner.batch.save_summary(args.suite, manifest)
    return manifest


def claim_gpu(args, runner):
    """Never infer idleness from free memory alone: training may still use it."""
    lock = None
    try:
        info = runner.batch.gpu_info(args.gpu)
        lock = (args.suite.parent / f'.{info["uuid"]}.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = runner.batch.gpu_info(args.gpu)
        processes = subprocess.check_output(['nvidia-smi', '-i', args.gpu,
            '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
        if processes:
            raise RuntimeError(f'GPU has existing compute processes: {processes}')
        if info['free_mib'] < 20 * 1024:
            raise RuntimeError(f'GPU has only {info["free_mib"]} MiB free; need 20480 MiB')
        return lock, info, None
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        if lock is not None:
            lock.close()
        return None, None, str(exc)


def publish_state(args, manifest, runner, status, **details):
    previous = (manifest.get('status'), manifest.get('wait_reason'))
    manifest.update(status=status, updated_at=now(), wait_reason=details.pop('wait_reason', None), **details)
    runner.batch.save_summary(args.suite, manifest)
    runner.batch.write(args.suite / 'daemon.json', dict(status=status, updated_at=now(),
        pid=os.getpid(), host=socket.gethostname(), gpu=args.gpu, training_root=str(args.run_root),
        waiting_tasks=[task for task in TASKS if task not in {j['task'] for j in manifest['jobs']}],
        wait_reason=manifest.get('wait_reason')))
    if previous != (status, manifest.get('wait_reason')):
        print(f'{now()} {status}: {manifest.get("wait_reason") or ""}', flush=True)


def run_monitor(args, manifest, runner):
    batch = runner.batch
    active, gpu_lock, compatibility_gpu = None, None, None
    last_scan, runtime_retry_at = -float('inf'), 0.
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        while True:
            if time.monotonic() - last_scan >= args.poll_seconds:
                refresh(args, manifest, runner)
                last_scan = time.monotonic()
            if args.scan_only:
                publish_state(args, manifest, runner, 'scanned')
                return 0
            if active is not None:
                text = active['log'].read_text(errors='replace')
                progress = re.findall(r'Total \d+/\d+\([^\n]*', text)
                current = progress[-1] if progress else ''
                if current != active['progress']:
                    active.update(progress=current, progress_time=time.monotonic())
                code = active['process'].poll()
                if code is None and (time.monotonic() - active['start'] > args.timeout
                        or time.monotonic() - active['progress_time'] > args.stall_timeout):
                    batch.stop_group(active['process'])
                    code = 124
                if code is not None:
                    result = batch.finish(active, args.suite, args.episodes, code)
                    job = active['job']
                    if result['status'] != 'completed':
                        job['state'] = 'pending' if job['attempts'] < args.max_attempts else 'failed'
                        job['last_error'] = result
                    active = None
                    gpu_lock.close()
                    gpu_lock = None
                    batch.save_summary(args.suite, manifest)
            pending = next((job for job in manifest['jobs'] if job['state'] == 'pending'), None)
            finished = len(manifest['jobs']) == len(TASKS) and all(
                job['state'] in ('completed', 'failed') for job in manifest['jobs'])
            if finished and active is None:
                failed = any(job['state'] == 'failed' for job in manifest['jobs'])
                publish_state(args, manifest, runner, 'finished_with_errors' if failed else 'completed')
                return int(failed)
            if active is None and pending is not None and time.monotonic() < runtime_retry_at:
                publish_state(args, manifest, runner, 'waiting_for_runtime',
                              wait_reason=manifest.get('wait_reason'))
            elif active is None and pending is not None:
                gpu_lock, info, reason = claim_gpu(args, runner)
                if gpu_lock is None:
                    publish_state(args, manifest, runner, 'waiting_for_gpu', wait_reason=reason)
                else:
                    try:
                        if compatibility_gpu != info['uuid']:
                            if getattr(args, 'skip_sim_check', False):
                                publish_state(args, manifest, runner, 'starting', gpu_info=info,
                                              simulator_check='skipped_by_request')
                            else:
                                publish_state(args, manifest, runner, 'checking_simulator', gpu_info=info)
                                check_options = ({'skip_driver_version_check': True}
                                                 if getattr(args, 'skip_driver_version_check', False) else {})
                                batch.preflight(args.root, args.suite, args.gpu, args.stall_timeout, **check_options)
                            compatibility_gpu = info['uuid']
                        port = batch.free_port(18880)
                        attempts = pending['attempts']
                        try:
                            launch_options = ({'skip_driver_version_check': True}
                                              if getattr(args, 'skip_driver_version_check', False) else {})
                            active = runner.launch(pending, args.root, args.suite, args.gpu,
                                                   args.episodes, port, args.model_python, **launch_options)
                        except Exception:
                            # launch() increments before Popen; no process means
                            # no evaluation attempt and only an empty work folder.
                            abandoned = args.suite / 'work' / pending['id'] / f'attempt{attempts + 1:02d}'
                            shutil.rmtree(abandoned, ignore_errors=True)
                            pending.update(state='pending', attempts=attempts)
                            raise
                        pending.update(worker_host=socket.gethostname(), worker_start=process_start(active['process'].pid))
                        publish_state(args, manifest, runner, 'evaluating', gpu_info=info)
                    except Exception as exc:
                        gpu_lock.close()
                        gpu_lock = None
                        if active is not None:
                            raise
                        runtime_retry_at = time.monotonic() + max(300, args.poll_seconds)
                        # Infrastructure failures keep the queue intact for a
                        # later GPU/environment recovery; do not count policy failure.
                        publish_state(args, manifest, runner, 'waiting_for_runtime',
                                      wait_reason=f'{type(exc).__name__}: {exc}')
            elif active is None:
                publish_state(args, manifest, runner, 'watching')
            else:
                publish_state(args, manifest, runner, 'evaluating')
            time.sleep(min(args.poll_seconds, 30))
    except BaseException as exc:
        if active is not None:
            batch.stop_group(active['process'])
            active['job']['state'] = 'interrupted'
        publish_state(args, manifest, runner, 'interrupted_or_error', wait_reason=f'{type(exc).__name__}: {exc}')
        if isinstance(exc, KeyboardInterrupt):
            return 130
        raise
    finally:
        if gpu_lock is not None:
            gpu_lock.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--run-root', type=Path, default=PROJECT / 'runs' / RUN / 'full')
    parser.add_argument('--suite', type=Path)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--max-attempts', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=43200)
    parser.add_argument('--stall-timeout', type=float, default=1800)
    parser.add_argument('--model-python', type=Path)
    parser.add_argument('--skip-driver-version-check', action='store_true',
                        help='Pass the opt-in Kit driver verification bypass to the simulator')
    parser.add_argument('--skip-sim-check', action='store_true',
                        help='Start formal evaluation without the standalone physics/RGB probe')
    parser.add_argument('--scan-only', action='store_true', help='Scan and pin ready weights once; never start a GPU process')
    args = parser.parse_args(argv)
    if (args.episodes < 1 or args.poll_seconds <= 0 or args.max_attempts < 1
            or args.timeout <= 0 or args.stall_timeout <= 0 or not args.gpu or ',' in args.gpu):
        parser.error('Expected positive limits and exactly one GPU')
    args.root, args.run_root = args.root.resolve(), args.run_root.resolve()
    if not re.fullmatch(r'[A-Za-z0-9._-]+', args.run_root.parent.name):
        parser.error('Unsupported training run name')
    args.suite = (args.suite or args.root / 'outputs/eval_suites' / SUITE).resolve()
    if args.suite.is_relative_to(args.run_root):
        parser.error('Keep evaluation output separate from the training run')
    args.model_python = (args.model_python or args.root / 'contact-belief-wam/.venv/bin/python').absolute()
    runner = load_runner(args.root)
    if not args.scan_only:
        for executable in (args.model_python, Path(os.environ.get('SIM_PREFIX', args.root / '.conda/univtac-sim')) / 'bin/python'):
            if not os.access(executable, os.X_OK):
                parser.error(f'Python environment unavailable: {executable}')
    os.environ['TWAM_ROOT'] = str(args.root)
    os.environ.setdefault('PI05_SIM_GRAPHICS_LIBRARY_PATH', str(args.root / '.cache/sim-graphics-runtime/lib'))
    # eval-worker.sh uses python3 before activating the simulator environment.
    os.environ['PATH'] = str(args.model_python.parent) + os.pathsep + os.environ.get('PATH', '')
    args.suite.mkdir(parents=True, exist_ok=True)
    with (args.suite / 'queue.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f'Queue already running: {args.suite}', flush=True)
            return 0
        manifest = prepare(args, runner)
        if not args.scan_only:
            (args.suite / 'daemon.pid').write_text(f'{os.getpid()}\n')
        print(f'{now()} WATCH {args.run_root}; eval GPU={args.gpu}; seeds=1000000..{1000000 + args.episodes - 1}', flush=True)
        return run_monitor(args, manifest, runner)


if __name__ == '__main__':
    raise SystemExit(main())
