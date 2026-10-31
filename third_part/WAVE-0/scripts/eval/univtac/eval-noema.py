#!/usr/bin/env python3
"""Evaluate no-EMA 10k policies with the unchanged native 50-seed protocol."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time
from types import SimpleNamespace

os.environ['WAVE0_ROOT'] = str(Path(__file__).resolve().parents[3])

ROOT = Path(os.environ.get('TWAM_ROOT', next(p for p in Path(__file__).resolve().parents if (p / 'UniVTAC').is_dir()))).resolve()
RUN = 'wave0-shared-8task-head-wrist-noema-001'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluator(root, run):
    wave = load(root/'pi05-univtac/scripts/eval-wave0-dual-view.py', 'noema_base_eval')
    wave.RUN = run
    wave.FILES = tuple(name for name in wave.FILES if name != 'ema.safetensors')
    return wave


def discover(wave, root, tasks):
    jobs = [j for j in wave.discover(root) if j['task'] in tasks]
    if set(tasks) != {j['task'] for j in jobs}:
        raise FileNotFoundError('Selected tasks do not all have complete no-EMA 10k checkpoints')
    for job in jobs:
        metadata = wave.batch.read(Path(job['source'])/'deployment.json')
        if metadata.get('use_ema') is not False:
            raise ValueError('No-EMA evaluation requires an explicit use_ema=false checkpoint')
        job['use_ema'] = False
    return sorted(jobs, key=lambda j: tasks.index(j['task']))


def launcher(batch):
    def launch(job, root, suite, gpu, episodes, port, model_python):
        job['attempts'] += 1
        work = suite/'work'/job['id']/f'attempt{job["attempts"]:02d}'
        work.mkdir(parents=True)
        deploy = work/'deploy.yml'
        batch.write(deploy, dict(policy_name='Wave0NoEMA', seed=0, instruction_type='seen',
            pi05_host='127.0.0.1', pi05_port=port, pi05_python=str(model_python),
            pi05_request_timeout=1800, pi05_startup_timeout=1800,
            pi05_physics_steps_per_action=2, pi05_expert_preflight=True,
            pi05_eval_protocol=batch.EVAL_PROTOCOL))
        env = batch.environment(root, work, gpu)
        env['WAVE0_CHECKPOINT'] = job['checkpoint']
        command = ['bash', str(root/'pi05-univtac/scripts/eval-worker.sh'),
                   str(root/'UniVTAC/scripts/eval_policy.py'), job['task'], 'clean', str(deploy),
                   '--headless', '--livestream', '0', '--device', 'cuda:0',
                   '--total_num', str(episodes), '--start_seed', '1000000', '--run_name', job['id']]
        logfile = work/'eval.log'
        with logfile.open('w') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        job.update(state='running', log=str(logfile), pid=child.pid)
        return dict(process=child, log=logfile, work=work, start=time.monotonic(), warm=False,
                    progress_time=time.monotonic(), progress='', job=job)
    return launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default=RUN)
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--suite', default='wave0-head-wrist-noema-001-10k-fixed-seeds')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--model-python', type=Path)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    for name in (args.run, args.suite):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,80}', name):
            parser.error('Invalid run or suite name')
    if not args.gpu or ',' in args.gpu:
        parser.error('--gpu must select one GPU')
    wave = evaluator(ROOT, args.run)
    batch = wave.batch
    tasks = list(dict.fromkeys(args.tasks or batch.TASKS))
    if not set(tasks) <= set(batch.TASKS):
        parser.error('Unknown task')
    jobs = discover(wave, ROOT, tasks)
    suite = ROOT/'outputs/eval_suites'/args.suite
    selection = dict(run=args.run, camera='head_wrist', tasks=tasks, use_ema=False)
    print(json.dumps(dict(selection=selection, suite=str(suite), episodes=50,
                         seed_start=1000000, seed_end=1000049, launch=args.launch,
                         checkpoints=[j['source'] for j in jobs]), indent=2), flush=True)
    if not args.launch:
        return
    model_python = Path(args.model_python or os.environ.get(
        'WAVE0_PYTHON', ROOT/'contact-belief-wam/.venv/bin/python')).absolute()
    for path in (model_python, ROOT/'.conda/univtac-sim/bin/python'):
        if not os.access(path, os.X_OK):
            raise FileNotFoundError('Python environment unavailable: ' + str(path))
    suite.mkdir(parents=True, exist_ok=True)
    with (suite/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (suite/'manifest.json').exists():
            manifest = batch.read(suite/'manifest.json')
            if (manifest.get('selection') != selection or manifest.get('episodes') != 50
                    or manifest.get('evaluation_protocol') != batch.EVAL_PROTOCOL):
                raise ValueError('Queue settings changed; choose a new suite name')
        else:
            for job in jobs:
                wave.snapshot(job, suite)
            manifest = dict(selection=selection, jobs=jobs, episodes=50, status='prepared',
                            seed_start=1000000, seed_end=1000049,
                            evaluation_protocol=batch.EVAL_PROTOCOL, use_ema=False)
            batch.save_summary(suite, manifest)
        if all(j['state'] == 'completed' for j in manifest['jobs']):
            print('All selected tasks already completed.')
            return
        os.environ['TWAM_ROOT'] = str(ROOT)
        os.environ['PATH'] = str(ROOT/'.conda/univtac-sim/bin') + os.pathsep + os.environ.get('PATH', '')
        os.environ.setdefault('PI05_SIM_GRAPHICS_LIBRARY_PATH', str(ROOT/'.cache/sim-graphics-runtime/lib'))
        common = load(ROOT/'pi05-univtac/scripts/eval-dual-view-10k.py', 'noema_gpu_queue')
        info, gpu_lock = common.wait_for_gpu(suite, manifest, args.gpu)
        with gpu_lock:
            batch.snapshot, batch.launch = wave.snapshot, launcher(batch)
            settings = SimpleNamespace(resume=True, episodes=50, parallel=1, gpu=args.gpu,
                watch_run_map=None, reuse_completed=False, timeout=43200, stall_timeout=1800,
                worker_memory_gib=18, reserve_gib=2, wait_for_vram=True)
            return batch.run_suite(settings, ROOT, model_python, info, suite)


if __name__ == '__main__':
    raise SystemExit(main())
