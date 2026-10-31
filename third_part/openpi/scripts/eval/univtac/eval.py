#!/usr/bin/env python3
"""Evaluate existing full-finetuning checkpoints on one GPU, with bounded concurrency.

Default scope: all committed numeric checkpoints under pi05_univtac_*_full.
Includes saved intermediate checkpoints; excludes smoke, LoRA, base and deleted weights.
Snapshots inference weights before execution so training checkpoint pruning cannot race eval.
Isaac RGB compatibility is checked on the actual node, before any policy evaluation.
"""
import argparse
import csv
import errno
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time

TASKS = ('grasp_classify', 'insert_HDMI', 'insert_hole', 'insert_tube',
         'lift_bottle', 'lift_can', 'pull_out_key', 'put_bottle_in_shelf')
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
os.environ['OPENPI_ROOT'] = str(PROJECT)
EVAL_PROTOCOL = 'bounded-actions-fixed-seeds-v2'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def discover(root):
    jobs = []
    for task in TASKS:
        for run in sorted((root / 'outputs/checkpoints' / f'pi05_univtac_{task}_full').glob('*')):
            settings_path = run / 'univtac_config.json'
            if settings_path.is_file():
                settings = read(settings_path)
                task_value, training = settings.get('task'), settings
            elif (run / 'run_identity.json').is_file():
                legacy = read(run / 'run_identity.json')
                task_value, training = legacy.get('task'), legacy.get('training', {})
            else:
                continue
            if task_value != task or training.get('mode') != 'full':
                raise ValueError(f'Checkpoint task/mode mismatch: {run}')
            for ck in sorted(run.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
                if not ck.name.isdigit() or not (ck / '_CHECKPOINT_METADATA').is_file():
                    continue
                if not (ck / 'params/_METADATA').is_file():
                    continue
                if not (ck / f'assets/full/{task}/norm_stats.json').is_file():
                    raise ValueError(f'Missing normalization: {ck}')
                mode = training.get('camera_mode', 'head_wrist')
                if (ck / 'camera_contract.json').exists():
                    if read(ck / 'camera_contract.json')['camera_mode'] != mode:
                        raise ValueError(f'Camera contract mismatch: {ck}')
                elif mode != 'head_wrist':
                    raise ValueError(f'Missing camera contract: {ck}')
                label = f'{task}__{mode.replace("_", "-")}__{run.name}__step{ck.name}'
                if not re.fullmatch(r'[A-Za-z0-9._-]{1,195}', label):
                    raise ValueError(f'Unsupported checkpoint name: {label}')
                jobs.append(dict(id=label, task=task, run=run.name, step=int(ck.name),
                                 camera=mode, source=str(ck), state='pending', attempts=0))
    if not jobs:
        raise ValueError('No committed full-finetuning checkpoints found on this node.')
    return jobs


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError as exc:
        if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
            raise
        shutil.copy2(src, dst)
    return dst


def snapshot(job, suite):
    source = Path(job['source'])
    target = suite / 'checkpoints' / job['task'] / job['run'] / str(job['step'])
    temporary = target.with_name(target.name + '.partial')
    if temporary.exists():
        shutil.rmtree(temporary)
    if not target.exists():
        temporary.mkdir(parents=True)
        try:
            for name in ('params', 'assets'):
                shutil.copytree(source / name, temporary / name, copy_function=link_or_copy)
            for name in ('camera_contract.json', '_CHECKPOINT_METADATA'):
                if (source / name).exists():
                    shutil.copy2(source / name, temporary / name)
            for metadata in ('univtac_config.json', 'run_identity.json'):
                if (source.parent / metadata).is_file():
                    shutil.copy2(source.parent / metadata, target.parent / metadata)
            temporary.rename(target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    job['checkpoint'] = str(target)


def gpu_info(gpu):
    out = subprocess.check_output(['nvidia-smi', '-i', gpu,
        '--query-gpu=name,uuid,memory.total,memory.free', '--format=csv,noheader,nounits'],
        text=True, timeout=15).strip().splitlines()
    if len(out) != 1:
        raise ValueError('Select exactly one GPU, without MIG.')
    name, uuid, total, free = [x.strip() for x in out[0].split(',')]
    return dict(name=name, uuid=uuid, total_mib=int(total), free_mib=int(free))


def environment(root, work, gpu):
    env = dict(os.environ)
    env.pop('PYTHONHOME', None)
    env.update(TWAM_ROOT=str(root), CUDA_VISIBLE_DEVICES=gpu, JAX_PLATFORMS='cuda',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONUNBUFFERED='1',
               PYTHONUTF8='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', NO_PROXY='127.0.0.1,localhost',
               no_proxy='127.0.0.1,localhost', LIVESTREAM='0',
               OMNI_WORKSPACE_ROOT=str(work / 'omniverse'),
               OMNI_CONFIG_PATH=str(work / 'omniverse/config'),
               XDG_RUNTIME_DIR=str(work / 'xdg-runtime'),
               XDG_DATA_HOME=str(work / 'xdg-data'),
               XDG_CACHE_HOME=str(work / 'xdg-cache'),
               PI05_EVAL_OUTPUT_ROOT=str(work / 'native'),
               PI05_TRACE_DIR=str(work / 'initial-policy-inputs'),
               PI05_SERVER_LOG=str(work / 'pi05-server.log'))
    return env


def stop_group(process):
    # Kill descendants even when the launcher/Kit has already exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=8)


def preflight(root, suite, gpu, timeout, *, skip_driver_version_check=False):
    work = suite / 'work/compatibility'
    # Only this suite's disposable compatibility artifacts are cleared.
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    command = ['bash', str(root / 'pi05-univtac/scripts/eval-worker.sh'),
               str(root / 'pi05-univtac/scripts/check_sim_app.py'), '--output', str(work / 'probe')]
    if skip_driver_version_check:
        command.append('--skip-driver-version-check')
    with (suite / 'compatibility.log').open('w') as log:
        child = subprocess.Popen(command,
            env=environment(root, work, gpu), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        finally:
            stop_group(child)
    report = read(work / 'probe/report.json') if (work / 'probe/report.json').exists() else {}
    report['exit_code'] = code
    report['driver_version_check_disabled'] = skip_driver_version_check
    write(suite / 'compatibility.json', report)
    if code != 0 or report.get('passed') is not True:
        raise RuntimeError(f'GPU physics/RGB compatibility failed. See {suite / "compatibility.log"}. '
                           'A800 is not officially supported by Isaac Sim; use an RTX-capable node if rendering fails.')
    shutil.rmtree(work)


def result_from_log(text, code, expected):
    final = re.findall(r'Final Result: (\d+)/(\d+)', text)
    progress = re.findall(r'Total (\d+)/(\d+)\(', text)
    counts = final or progress
    success, completed = map(int, counts[-1]) if counts else (0, 0)
    errors = len(re.findall(r'Seed \d+ occurred exception:', text))
    fixed_seeds = f'Evaluation protocol: {EVAL_PROTOCOL}' in text
    attempts = completed if fixed_seeds else completed + errors
    passed = 'Expert preflight passed;' in text
    directories = re.findall(r'Evaluation directory: ([^\r\n]+)', text)
    valid = code == 0 and passed and bool(final) and completed == expected and success <= completed
    return dict(status='completed' if valid else 'incomplete_or_error',
                evaluation_protocol=EVAL_PROTOCOL if fixed_seeds else 'legacy',
                successes=success, completed_episodes=completed,
                failures=max(0, completed-success-(errors if fixed_seeds else 0)),
                attempted_episodes=attempts,
                policy_failures=len(re.findall(r'Seed \d+ Policy output rejected;', text)),
                exceptions=errors, success_rate=success/completed if completed else None,
                success_rate_including_exceptions=success/attempts if attempts else None,
                expert_preflight_passed=passed, exit_code=code,
                native_result_dir=directories[-1] if directories else None)


def warmed(text):
    return 'Expert preflight passed;' in text and bool(re.search(r'Total \d+/[1-9]\d*\(', text))


def can_launch(active, limit, free_mib, reserve_mib, worker_mib):
    return len(active) < limit and all(x['warm'] for x in active.values()) and free_mib >= reserve_mib + worker_mib


def free_port(start):
    for port in range(start, start + 1000):
        with socket.socket() as sock:
            try:
                sock.bind(('127.0.0.1', port))
            except OSError:
                continue
            return port
    raise RuntimeError('No free policy port')


def launch(job, root, suite, gpu, episodes, port, model_python):
    job['attempts'] += 1
    work = suite / 'work' / job['id'] / f'attempt{job["attempts"]:02d}'
    work.mkdir(parents=True)
    # JSON is valid YAML; no dependency and no modifications to shared deploy.yml.
    deploy = work / 'deploy.yml'
    write(deploy, dict(policy_name='Pi05Wam', seed=0, instruction_type='seen',
        pi05_host='127.0.0.1', pi05_port=port, pi05_device='cuda',
        pi05_python=str(model_python), pi05_startup_timeout=1800, pi05_request_timeout=1800,
        pi05_physics_steps_per_action=2, pi05_expert_preflight=True,
        pi05_eval_protocol=EVAL_PROTOCOL))
    env = environment(root, work, gpu)
    env['PI05_CHECKPOINT'] = job['checkpoint']
    command = ['bash', str(root / 'pi05-univtac/scripts/eval-worker.sh'), str(root / 'UniVTAC/scripts/eval_policy.py'),
               job['task'], 'clean', str(deploy), '--headless', '--livestream', '0',
               '--device', 'cuda:0', '--total_num', str(episodes), '--start_seed', '1000000',
               '--run_name', job['id']]
    logfile = work / 'eval.log'
    with logfile.open('w') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
    job.update(state='running', log=str(logfile), pid=child.pid)
    print(f'START {job["id"]} port={port}', flush=True)
    return dict(process=child, log=logfile, work=work, start=time.monotonic(), warm=False,
                progress_time=time.monotonic(), progress='', job=job)


def finish(item, suite, episodes, code):
    stop_group(item['process'])
    job, work = item['job'], item['work']
    result = result_from_log(item['log'].read_text(errors='replace'), code, episodes)
    if result['status'] == 'completed':
        native = Path(result['native_result_dir'] or '/')
        if not native.is_relative_to(work / 'native') or not (native / 'metadata.json').is_file():
            result.update(status='incomplete_or_error', reason='Missing or unexpected native output')
        else:
            # Cross-check the native artifact, not just the terminal summary.
            metadata = read(native / 'metadata.json')
            fixed_seeds = result['evaluation_protocol'] == EVAL_PROTOCOL
            outcomes = ('success', 'failed', 'error') if fixed_seeds else ('success', 'failed')
            rows = {k: r for k, r in metadata.items() if int(k) >= 1000000 and r.get('result') in outcomes}
            bad_seeds = fixed_seeds and {int(k) for k in rows} != set(range(1000000, 1000000+episodes))
            bad_errors = fixed_seeds and sum(r['result'] == 'error' for r in rows.values()) != result['exceptions']
            if (len(rows) != episodes or bad_seeds or bad_errors
                    or sum(r['result'] == 'success' for r in rows.values()) != result['successes']):
                result.update(status='incomplete_or_error', reason='Native metadata disagrees with final log')
            else:
                destination = suite / 'results' / job['id']
                destination.parent.mkdir(exist_ok=True)
                for name in ('initial-policy-inputs', 'eval.log', 'pi05-server.log', 'deploy.yml'):
                    if (work / name).exists():
                        shutil.move(str(work / name), str(native / name))
                result['native_result_dir'] = str(destination)
                result.update(task=job['task'], source_checkpoint=job['source'],
                              evaluated_checkpoint=job['checkpoint'], camera=job['camera'], step=job['step'])
                write(native / 'result.json', result)
                native.rename(destination)  # Publish the complete result atomically.
                job['log'] = str(destination / 'eval.log')
    # Incomplete videos/caches are not published as experimental results.
    if result['status'] == 'completed':
        shutil.rmtree(work)
    elif (work / 'native').exists():
        shutil.rmtree(work / 'native')
    job.update(state=result['status'], result=result)
    print(f'FINISH {job["id"]}: {result["status"]} {result["successes"]}/{result["completed_episodes"]}', flush=True)
    return result


def save_summary(suite, manifest):
    relocations = read(suite / 'result-relocations.json') if (suite / 'result-relocations.json').exists() else {}
    for job in manifest['jobs']:
        result = job.get('result', {})
        old = result.get('native_result_dir')
        if old in relocations:
            result['native_result_dir'] = relocations[old]
    write(suite / 'manifest.json', manifest)
    fields = ['task', 'run', 'step', 'camera', 'state', 'successes', 'completed_episodes',
              'success_rate', 'exceptions', 'success_rate_including_exceptions',
              'evaluation_protocol', 'policy_failures', 'result_dir']
    with (suite / 'summary.csv.tmp').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for job in manifest['jobs']:
            result = job.get('result', {})
            row = {k: job.get(k, result.get(k, '')) for k in fields}
            row['result_dir'] = result.get('native_result_dir', '') if job['state'] == 'completed' else ''
            writer.writerow(row)
    (suite / 'summary.csv.tmp').replace(suite / 'summary.csv')


def reuse_completed(root, job, episodes):
    """Reuse only a full native evaluation of this exact checkpoint, never smoke runs."""
    # Archived watcher records live with their evaluation suite. Keep the old
    # location as a fallback for a monitor that has not yet published its result.
    candidates = sorted((root / 'outputs/eval_suites').glob('*/results/*/result.json'))
    candidates.extend(sorted((root / 'outputs/watchers').glob('*/result.json')))
    for path in candidates:
        try:
            result = read(path)
            if (result.get('checkpoint') != job['source'] or result.get('status') != 'completed'
                    or result.get('completed_episodes') != episodes
                    or result.get('requested_episodes') != episodes
                    or result.get('expert_preflight_passed') is not True):
                continue
            native = Path(result.get('native_result_dir') or '/')
            metadata = read(native / 'metadata.json')
            rows = [v for k, v in metadata.items() if int(k) >= 1000000 and v.get('result') in ('success', 'failed')]
            if len(rows) != episodes or sum(v['result'] == 'success' for v in rows) != result['successes']:
                continue
            # An error-free legacy rollout used the same first N seeds and never
            # needed the newly allowed arm saturation. Its outcomes remain usable.
            seeds = {int(k) for k in metadata if int(k) >= 1000000}
            if result.get('exceptions', 0) or seeds != set(range(1000000, 1000000+episodes)):
                continue
            result['success_rate'] = result['native_success_rate']
            result['evaluation_protocol'] = 'legacy-no-errors-fixed-seeds-verified'
            job.update(state='completed', result=result, reused_result=str(path))
            return True
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return False


def refresh_checkpoints(root, suite, manifest, run_map, reuse):
    known = {j['id'] for j in manifest['jobs']}
    for job in discover(root):
        if job['id'] in known or (run_map and run_map.get(job['task']) != job['run']):
            continue
        if not (reuse and reuse_completed(root, job, manifest['episodes'])):
            snapshot(job, suite)
        manifest['jobs'].append(job)
        print(f'DISCOVER {job["id"]}: {job["state"]}', flush=True)


def final_checkpoints_evaluated(manifest, run_map):
    terminal = {'completed', 'incomplete_or_error'}
    return all(any(j['task'] == task and j['run'] == run and j['step'] == 10000
                   and j['state'] in terminal for j in manifest['jobs']) for task, run in run_map.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('TWAM_ROOT', PROJECT.parent)))
    parser.add_argument('--run', default='a800-all-existing')
    parser.add_argument('--parallel', type=int, default=2, choices=range(1, 5))
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    parser.add_argument('--model-python', type=Path)
    parser.add_argument('--worker-memory-gib', type=float, default=24)
    parser.add_argument('--reserve-gib', type=float, default=4)
    parser.add_argument('--timeout', type=int, default=21600, help='Maximum seconds per checkpoint')
    parser.add_argument('--stall-timeout', type=int, default=1800)
    parser.add_argument('--resume', action='store_true', help='Keep the original snapshot; skip completed results')
    parser.add_argument('--reuse-completed', action='store_true', help='Reuse matching complete native results')
    parser.add_argument('--watch-run-map', type=Path,
                        help='Continuously pin/evaluate new checkpoints of these task->run mappings, through step 10000')
    parser.add_argument('--list', action='store_true', help='Read-only inventory; no GPU or simulation required')
    args = parser.parse_args()
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,90}', args.run) or ',' in args.gpu
            or not args.gpu or args.episodes < 1 or args.worker_memory_gib <= 0 or args.reserve_gib < 0
            or args.timeout <= 0 or args.stall_timeout <= 0):
        parser.error('Invalid run name, single-GPU selector, episode count, memory budget or timeout')
    root = args.root.resolve()
    if args.list:
        print(json.dumps(discover(root), ensure_ascii=False, indent=2))
        return 0
    model_python = (args.model_python or root / '.venvs/openpi/bin/python').absolute()
    for executable in (model_python, root / '.conda/univtac-sim/bin/python'):
        if not os.access(executable, os.X_OK):
            raise RuntimeError(f'Python unavailable on this node: {executable}; mount/install the existing environments first.')
    info = gpu_info(args.gpu)
    suite = root / 'outputs/eval_suites' / args.run
    suite.mkdir(parents=True, exist_ok=True)
    # Also prevents two instances of this launcher from claiming the same GPU.
    with (suite.parent / f'.{info["uuid"]}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_suite(args, root, model_python, info, suite)


def run_suite(args, root, model_python, info, suite):
    run_map = read(args.watch_run_map) if args.watch_run_map else {}
    if args.watch_run_map and (not run_map or any(t not in TASKS or not isinstance(r, str) for t, r in run_map.items())):
        raise ValueError('Invalid task->training-run mapping')
    manifest_path = suite / 'manifest.json'
    if manifest_path.exists():
        if not args.resume:
            raise RuntimeError('Run exists; use --resume or a different --run name.')
        manifest = read(manifest_path)
        if manifest['episodes'] != args.episodes:
            raise RuntimeError('Resume must keep the original episode count.')
        if manifest.get('evaluation_protocol', 'legacy') != EVAL_PROTOCOL:
            raise RuntimeError('Evaluation protocol changed; use a separate suite to preserve legacy results.')
        for job in manifest['jobs']:
            committed = suite / 'results' / job['id'] / 'result.json'
            if committed.exists():
                result = read(committed)
                if result.get('status') == 'completed' and result.get('completed_episodes') == args.episodes:
                    job.update(state='completed', result=result)
            if job['state'] != 'completed':
                job['state'] = 'pending'
    else:
        jobs = [j for j in discover(root) if not run_map or run_map.get(j['task']) == j['run']]
        manifest = dict(episodes=args.episodes, jobs=jobs, status='preparing', evaluation_protocol=EVAL_PROTOCOL,
                        scope='All existing committed full-finetuning checkpoints; startup snapshot, not a training watcher.')
        write(manifest_path, manifest)
    for job in manifest['jobs']:
        if job['state'] != 'completed' and args.reuse_completed:
            reuse_completed(root, job, args.episodes)
        if job['state'] != 'completed':
            snapshot(job, suite)
    if run_map:
        manifest.update(watch_run_map=run_map, scope='Continuous evaluation of committed checkpoints through step 10000')
    manifest.update(gpu=info, parallel_limit=args.parallel, status='running')
    save_summary(suite, manifest)
    if all(j['state'] == 'completed' for j in manifest['jobs']) and (not run_map or final_checkpoints_evaluated(manifest, run_map)):
        manifest['status'] = 'completed'
        save_summary(suite, manifest)
        return 0
    print(f'GPU={info["name"]}, checkpoints={len(manifest["jobs"])}, max_parallel={args.parallel}', flush=True)
    if re.search(r'\b[AH](?:100|800)\b', info['name']):
        print('WARNING: Isaac Sim does not officially support this non-RTX GPU. Testing real RGB rendering first.', flush=True)
    active = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    previous_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    limit = args.parallel
    try:
        preflight(root, suite, args.gpu, args.stall_timeout)
        last_scan = 0
        while True:
            if run_map and time.monotonic() - last_scan >= 30:
                refresh_checkpoints(root, suite, manifest, run_map, args.reuse_completed)
                last_scan = time.monotonic()
            for key, item in list(active.items()):
                text = item['log'].read_text(errors='replace')
                item['warm'] = warmed(text)
                progress = re.findall(r'Total \d+/\d+\([^\n]*', text)
                current = progress[-1] if progress else ''
                if current != item['progress']:
                    item.update(progress=current, progress_time=time.monotonic())
                code = item['process'].poll()
                if code is None and (time.monotonic()-item['start'] > args.timeout
                        or time.monotonic()-item['progress_time'] > args.stall_timeout):
                    stop_group(item['process'])
                    code = 124
                if code is not None:
                    result = finish(item, suite, args.episodes, code)
                    del active[key]
                    if result['status'] != 'completed':
                        model_log = item['work'] / 'pi05-server.log'
                        error_text = text + (model_log.read_text(errors='replace') if model_log.exists() else '')
                        oom = bool(re.search(r'out of memory|RESOURCE_EXHAUSTED|CUDA_ERROR_OUT_OF_MEMORY', error_text, re.I))
                        if oom and limit > 1 and item['job']['attempts'] == 1:
                            limit = 1
                            item['job']['state'] = 'pending'
                            print('OOM: reducing concurrency to one and retrying this checkpoint once.', flush=True)
                        elif not item['warm']:
                            raise RuntimeError(f'Worker failed before valid startup: {key}. See {item["log"]}')
            pending = next((j for j in manifest['jobs'] if j['state'] == 'pending'), None)
            if pending is None and not active and (not run_map or final_checkpoints_evaluated(manifest, run_map)):
                break
            info = gpu_info(args.gpu)
            if pending and can_launch(active, limit, info['free_mib'],
                    args.reserve_gib*1024, args.worker_memory_gib*1024):
                port = free_port(18765 + manifest['jobs'].index(pending))
                active[pending['id']] = launch(pending, root, suite, args.gpu, args.episodes, port, model_python)
                manifest['status'] = 'running'
            elif pending and not active:
                if getattr(args, 'wait_for_vram', False):
                    manifest['status'] = 'waiting_for_gpu'
                else:
                    raise RuntimeError(f'Insufficient free VRAM: {info["free_mib"]} MiB; '
                                       'another job may occupy the GPU. Resume after releasing it.')
            manifest.update(gpu=info, effective_parallel_limit=limit)
            save_summary(suite, manifest)
            time.sleep(5)
        manifest['status'] = 'completed' if all(j['state'] == 'completed' for j in manifest['jobs']) else 'finished_with_errors'
    except BaseException as exc:
        manifest.update(status='interrupted_or_error', error=str(exc))
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        for item in active.values():
            stop_group(item['process'])
            item['job']['state'] = 'interrupted'
            if (item['work'] / 'native').exists():
                shutil.rmtree(item['work'] / 'native')
        save_summary(suite, manifest)
    print(f'{manifest["status"]}: {suite / "summary.csv"}', flush=True)
    return 0 if manifest['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
