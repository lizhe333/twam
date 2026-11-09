#!/usr/bin/env python3
"""Re-evaluate visual WAVE-0 with the existing native 50-seed protocol.

Defaults to a read-only plan. --launch creates a separate resumable queue and
waits for the shared GPU lock; never waits for an unrelated historical pi05 run.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace

from common import TASKS, experiment, load, name, workspace


def selection(args):
    root = workspace()
    recipe = experiment(args.view)
    wave = load(root / 'pi05-univtac/scripts' / recipe['evaluator'], 'repro_wave_eval')
    wave.RUN = name(args.source_run or recipe['training_run'])
    requested = list(TASKS) if not args.tasks else list(dict.fromkeys(args.tasks))
    jobs = [j for j in wave.discover(root) if j['task'] in requested]
    missing = set(requested) - {j['task'] for j in jobs}
    if missing:
        raise FileNotFoundError(f'Missing complete 10k checkpoints for {sorted(missing)}')
    jobs.sort(key=lambda j: requested.index(j['task']))
    suite_name = name(args.suite or f'wave0-repro-{args.view}-10k-fixed-seeds')
    if suite_name in {experiment(v)['suite'] for v in ('single', 'dual')}:
        raise ValueError('Use a new suite; published historical results are read-only references')
    suite = root / 'outputs/eval_suites' / suite_name
    selected = dict(view=args.view, camera=recipe['camera'], run=wave.RUN, tasks=requested)
    return root, wave, jobs, suite, selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--view', choices=['single', 'dual'], required=True)
    p.add_argument('--source-run', help='Default: the historical 10k run for this view')
    p.add_argument('--tasks', nargs='+', choices=TASKS)
    p.add_argument('--suite', help='New evaluation directory name; same name resumes the same queue')
    p.add_argument('--model-python', type=Path)
    p.add_argument('--gpu', default='0')
    p.add_argument('--launch', action='store_true')
    args = p.parse_args()
    root, wave, jobs, suite, selected = selection(args)
    os.environ['WAVE0_ROOT'] = str(Path(__file__).resolve().parents[3])
    if not args.gpu or ',' in args.gpu:
        p.error('--gpu must select one GPU')
    print(json.dumps(dict(selection=selected, suite=str(suite), episodes=50,
                         seed_start=1000000, seed_end=1000049,
                         checkpoints=[j['source'] for j in jobs], launch=args.launch), indent=2), flush=True)
    if not args.launch:
        return 0
    model_python = Path(args.model_python or os.environ.get(
        'WAVE0_PYTHON', root / 'contact-belief-wam/.venv/bin/python')).absolute()
    for path in (model_python, root / '.conda/univtac-sim/bin/python'):
        if not os.access(path, os.X_OK):
            raise FileNotFoundError(f'Python environment unavailable: {path}')
    suite.mkdir(parents=True, exist_ok=True)
    batch = wave.batch
    with (suite / 'queue.lock').open('a') as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (suite / 'manifest.json').exists():
            manifest = batch.read(suite / 'manifest.json')
            if (manifest.get('selection') != selected or manifest.get('episodes') != 50
                    or manifest.get('evaluation_protocol') != batch.EVAL_PROTOCOL):
                raise ValueError('Queue selection/protocol changed; choose a new --suite')
        else:
            for job in jobs:
                wave.snapshot(job, suite)
            manifest = dict(selection=selected, jobs=jobs, episodes=50, status='prepared',
                            seed_start=1000000, seed_end=1000049,
                            evaluation_protocol=batch.EVAL_PROTOCOL, use_ema=True)
            batch.save_summary(suite, manifest)
        if all(j['state'] == 'completed' for j in manifest['jobs']):
            print('All selected tasks already completed.', flush=True)
            return 0
        os.environ['TWAM_ROOT'] = str(root)
        # eval-worker.sh performs a Python check before activating the sim env.
        os.environ['PATH'] = str(root / '.conda/univtac-sim/bin') + os.pathsep + os.environ.get('PATH', '')
        os.environ.setdefault('PI05_SIM_GRAPHICS_LIBRARY_PATH', str(root / '.cache/sim-graphics-runtime/lib'))
        common_queue = load(root / 'pi05-univtac/scripts/eval-dual-view-10k.py', 'repro_gpu_queue')
        info, gpu_lock = common_queue.wait_for_gpu(suite, manifest, args.gpu)
        with gpu_lock:
            batch.snapshot, batch.launch = wave.snapshot, wave.launch
            settings = SimpleNamespace(resume=True, episodes=50, parallel=1, gpu=args.gpu,
                watch_run_map=None, reuse_completed=False, timeout=43200,
                stall_timeout=1800, worker_memory_gib=18, reserve_gib=2, wait_for_vram=True)
            return batch.run_suite(settings, root, model_python, info, suite)


if __name__ == '__main__':
    raise SystemExit(main())
