#!/usr/bin/env python3
"""Four independent two-GPU JAX queues; default prints a plan without training."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get('TWAM_ROOT', PROJECT.parent)).resolve()
CONFIG = PROJECT / 'configs/univtac/head_wrist.yaml'
SOURCE = PROJECT / 'src'
RUN = 'univtac-2gpu-001'
GROUPS = (('grasp_classify', 'lift_bottle'), ('insert_HDMI', 'lift_can'),
          ('insert_hole', 'pull_out_key'), ('insert_tube', 'put_bottle_in_shelf'))


def make_plan(visible):
    import yaml
    settings = yaml.safe_load(CONFIG.read_text())
    if settings.get('fsdp_devices') != 2:
        raise ValueError('Parallel launcher requires a two-GPU recipe')
    cards = [card.strip() for card in visible.split(',')]
    if len(cards) != 8 or len(set(cards)) != 8 or any(not c.strip() for c in cards):
        raise ValueError('Exactly eight distinct visible A800 GPUs are required')
    groups = [{'worker': i, 'tasks': list(tasks), 'gpus': cards[2*i:2*i+2],
               'run': f'{RUN}-pair{i}'} for i, tasks in enumerate(GROUPS)]
    return {'run': RUN, 'camera_mode': settings['camera_mode'], 'config': str(CONFIG), 'batch_per_task': settings.get('batch_size', 64),
            'fsdp_devices_per_task': 2, 'parallel_tasks': 4, 'steps_per_task': settings.get('num_train_steps', 10000),
            'groups': groups,
            'run_map': {task: g['run'] for g in groups for task in g['tasks']},
            'progress': str(ROOT / 'outputs/queues/full' / RUN / 'report.json')}


def worker(index, resume):
    # Import only after the parent restricts this worker to its own GPU pair.
    sys.path.insert(0, str(SOURCE))
    from openpi.training.univtac_config import make_config
    from openpi.training.univtac_queue import run_queue
    tasks = GROUPS[index]
    for task in tasks:
        cfg = make_config(task, 'full', run=f'{RUN}-pair{index}', recipe=CONFIG)
        if cfg.fsdp_devices != 2:
            raise ValueError('This launcher assigns two GPUs per task; choose a two-GPU recipe')
    run_queue(tasks, 'full', f'{RUN}-pair{index}', resume=resume, recipe=CONFIG)


def failure_detail(directory, group):
    """Surface the trainer's actual exception, not only its wrapper's exit code."""
    child = directory.parent / group['run']
    log = directory / f"pair{group['worker']}.log"
    try:
        report = json.loads((child / 'report.json').read_text())
        task = report.get('active_task')
        if task in group['tasks']:
            candidate = child / 'logs' / f'{task}-train-10000.log'
            if candidate.is_file(): log = candidate
    except (OSError, ValueError):
        pass
    try:
        with log.open('rb') as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 8192))
            tail = '\n'.join(handle.read().decode(errors='replace').splitlines()[-20:])
    except OSError as exc:
        tail = f'Could not read log: {exc}'
    return f'Trainer log: {log}\n{tail}'


def launch(plan, resume):
    directory = Path(plan['progress']).parent
    report_path = directory / 'report.json'
    if directory.exists() and not resume:
        raise FileExistsError(f'{directory} exists; use explicit --resume')
    if resume and not report_path.is_file():
        raise FileNotFoundError('No prior eight-GPU queue to resume')
    if resume and json.loads(report_path.read_text()).get('plan') != plan:
        raise ValueError('GPU assignments or task plan changed')
    for group in plan['groups']:
        child = directory.parent / group['run']
        if not resume and child.exists():
            raise FileExistsError(f'Existing child queue: {child}')
        for task in group['tasks']:
            ckpt = ROOT / 'outputs/checkpoints' / f'pi05_univtac_{task}_full' / group['run']
            if ckpt.exists() and not (resume and (child / 'report.json').is_file()):
                raise FileExistsError(f'Untracked or existing checkpoint directory: {ckpt}')
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {'plan': plan, 'status': 'running', 'workers': [],
                  'started_at': datetime.now(timezone.utc).isoformat()}

        def write():
            status['updated_at'] = datetime.now(timezone.utc).isoformat()
            temp = report_path.with_suffix('.tmp')
            temp.write_text(json.dumps(status, indent=2) + '\n')
            temp.replace(report_path)

        (directory / 'run-map.json').write_text(json.dumps(plan['run_map'], indent=2) + '\n')
        processes, handles = [], []

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Received signal {signum}')

        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            write()
            for group in plan['groups']:
                log = directory / f"pair{group['worker']}.log"
                handle = log.open('a'); handles.append(handle)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(group['gpus']),
                           PYTHONPATH=str(SOURCE), TWAM_ROOT=str(ROOT), JAX_PLATFORMS='cuda',
                           XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONUNBUFFERED='1')
                command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(group['worker']), '--config', str(CONFIG), '--run', RUN]
                if resume and (directory.parent / group['run'] / 'report.json').is_file():
                    command.append('--resume')
                process = subprocess.Popen(command, env=env, stdout=handle,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(process)
                status['workers'].append({**group, 'pid': process.pid, 'state': 'running',
                                          'log': str(log), 'returncode': None})
                print(f"Started pair {group['worker']}: GPUs {group['gpus']}, tasks {group['tasks']}", flush=True)
            write()
            while True:
                codes = [p.poll() for p in processes]
                changed = False
                for row, code in zip(status['workers'], codes):
                    if code is not None and row['returncode'] is None:
                        row.update(returncode=code, state='done' if code == 0 else 'failed')
                        changed = True
                if changed: write()
                if any(code not in (None, 0) for code in codes):
                    failed = next(i for i, code in enumerate(codes) if code not in (None, 0))
                    detail = failure_detail(directory, plan['groups'][failed])
                    raise RuntimeError(f'A worker failed: exit codes {codes}\n{detail}')
                if all(code == 0 for code in codes): break
                time.sleep(2)
            status['status'] = 'completed'
        except BaseException as exc:
            status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            for process in processes:
                if status['status'] == 'failed' or process.poll() is None:
                    try: os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError: pass
            for process in processes:
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    process.wait()
            if status['status'] == 'failed':
                # A dead worker may have left a trainer child in its session.
                for process in processes:
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
            for row, process in zip(status['workers'], processes):
                row.update(returncode=process.returncode,
                           state='done' if process.returncode == 0 else 'failed')
            for handle in handles: handle.close()
            signal.signal(signal.SIGTERM, previous)
            write()


def main():
    global CONFIG, RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--run', default=RUN)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', type=int, choices=range(4))
    args = parser.parse_args()
    CONFIG, RUN = args.config.resolve(), args.run
    if not RUN or Path(RUN).name != RUN or RUN in ('.', '..'):
        parser.error('Use a simple run name without path separators')
    if args.worker is not None:
        worker(args.worker, args.resume)
        return
    plan = make_plan(os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7'))
    print(json.dumps(plan, indent=2), flush=True)
    if args.launch: launch(plan, args.resume)


if __name__ == '__main__':
    main()
