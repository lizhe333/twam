#!/usr/bin/env python3
"""Formal 50+50 episode A800 comparison, with frozen runtime and matched rendering."""
from __future__ import annotations

import argparse
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
import sys
import time

import watch_eval_10k as watch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))
from pair_contract import INPUT_PROFILE, PROFILE, RENDER_SETTINGS, RNG_PROFILE, kit_arguments

SUITE = 'baidu-lift-can-base-vs-tactile-aligned-optix32-v2'
TASK = 'lift_can'


def load_base(root):
    spec = importlib.util.spec_from_file_location('pair_base_runner', root / 'pi05-univtac/scripts/eval-wave0-checkpoints.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory(root):
    return {str(p.relative_to(root)): watch.digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts
            and p.name != 'runtime_manifest.json'}


def native_sources(root):
    # Shared simulator/task code must remain unchanged between the two runs.
    paths = list((root / 'UniVTAC/envs').rglob('*.py'))
    paths += [root / 'UniVTAC/task_config/clean.yml',
              root / 'pi05-univtac/src/pi05_univtac/sim_policy.py',
              root / 'pi05-univtac/src/pi05_univtac/constants.py']
    return {str(p.relative_to(root)): watch.digest(p) for p in sorted(paths)}


def verify_runtime(root, runtime):
    record = json.loads((runtime / 'runtime_manifest.json').read_text())
    if record['profile'] != PROFILE or inventory(runtime) != record['files']:
        raise ValueError('Pinned comparison runtime changed; use a new suite')
    if native_sources(root) != record['native_sources']:
        raise ValueError('Simulator/task source changed since preparation; use a new suite')
    return record


def freeze_native_evaluator(source):
    # Patch only the private frozen copy; shared UniVTAC and Pro 6K stay intact.
    replacements = {
        'app_launcher = AppLauncher(args_cli)':
            'from pair_contract import SIMULATION_APP_SETTINGS\n'
            'from pair_render import initialize_rendering, configure_task_render\n'
            'vars(args_cli).update(SIMULATION_APP_SETTINGS)\n'
            'app_launcher = AppLauncher(args_cli)',
        'simulation_app = app_launcher.app':
            'simulation_app = app_launcher.app\ninitialize_rendering()',
        '    env_cfg:BaseTaskCfg = task_module.TaskCfg()':
            '    env_cfg:BaseTaskCfg = task_module.TaskCfg()\n    configure_task_render(env_cfg)',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f'Native evaluator template changed: {old}')
        source = source.replace(old, new)
    compile(source, '<frozen-a800-native-eval>', 'exec')
    return source


def pin_runtime(root, suite, batch):
    runtime = suite / 'runtime'
    if runtime.exists():
        return runtime, verify_runtime(root, runtime)
    temp = suite / 'runtime.partial'
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir()
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache')
    try:
        project = root / 'twam'
        for folder in ('src', 'inference', 'vendor/wave0'):
            shutil.copytree(project / folder, temp / folder, ignore=ignore)
        for name in ('pyproject.toml', 'scripts/cloud_train.sh'):
            (temp / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(project / name, temp / name)
        (temp / 'vendor/transformers').symlink_to((project / 'vendor/transformers').resolve(), target_is_directory=True)
        for family, original in [('tactile', 'TactileWave0Wam'), ('base', 'Wave0Wam'), ('pi05', 'Pi05Wam')]:
            shutil.copy2(root / f'UniVTAC/policy/{original}/deploy_policy.py', temp / f'native_{family}.py')
        (temp / 'native_eval.py').write_text(freeze_native_evaluator(
            (root / 'UniVTAC/scripts/eval_policy.py').read_text()))
        base_root = temp / 'base_workspace/wave0/single-view'
        for folder in ('src', 'vendor/wave0', 'configs'):
            shutil.copytree(root / 'wave0/single-view' / folder, base_root / folder, ignore=ignore)
        (temp / 'base_workspace/openpi').symlink_to(root / 'openpi', target_is_directory=True)
        shutil.copy2(root / 'pi05-univtac/scripts/wave0-policy-server.py', temp / 'base_server_original.py')
        bootstrap = ("import os, sys\nfrom pathlib import Path\n"
                     "sys.path.insert(0, str(Path(os.environ['PAIR_RUNTIME']) / 'inference'))\n"
                     "from pair_servers import {function}\nraise SystemExit({function}())\n")
        (temp / 'src/contact_belief_wam/__main__.py').write_text(bootstrap.format(function='tactile_main'))
        server = temp / 'base_workspace/pi05-univtac/scripts/wave0-policy-server.py'
        server.parent.mkdir(parents=True)
        server.write_text(bootstrap.format(function='base_main'))
        files = inventory(temp)
        record = dict(profile=PROFILE, created_at=watch.now(), files=files,
                      native_sources=native_sources(root),
                      fingerprint=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
        batch.write(temp / 'runtime_manifest.json', record)
        temp.rename(runtime)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return runtime, record


def prepare(args, runner, base):
    batch = runner.batch
    path = args.suite / 'manifest.json'
    families = ['base', 'tactile'] if args.family == 'both' else [args.family]
    selection = dict(task=TASK, episodes=50, seed_start=1000000, seed_end=1000049,
                     camera='head_only', step=10000, use_ema=True,
                     evaluation_protocol=batch.EVAL_PROTOCOL, profile=PROFILE,
                     input_alignment=INPUT_PROFILE, rng_profile=RNG_PROFILE,
                     render_settings=RENDER_SETTINGS, parallel_limit=1, families=families)
    if path.exists():
        manifest = batch.read(path)
        if any(manifest.get(k) != v for k, v in selection.items()):
            raise ValueError('Suite has another comparison contract; use --suite with a new directory')
        if any(j['state'] == 'running' for j in manifest['jobs']):
            raise RuntimeError('Suite records an unfinished cloud worker. Use a new --suite after stopping that job.')
    else:
        jobs = []
        for family in families:
            job = (next((j for j in base.discover(args.root) if j['task'] == TASK), None)
                   if family == 'base' else
                   watch.discover_task(args.root / 'twam/runs' / watch.RUN / 'full', TASK, runner))
            if job is None:
                raise RuntimeError(f'Committed lift_can {family} Stage-2 10k checkpoint is required')
            job['comparison_family'] = family
            jobs.append(job)
        manifest = dict(selection, jobs=jobs, status='preparing', created_at=watch.now())
        batch.save_summary(args.suite, manifest)
    for job in manifest['jobs']:
        if job['comparison_family'] == 'base':
            base.snapshot(job, args.suite)
        else:
            watch.snapshot(job, args.suite, runner)
        c = Path(job['checkpoint'])
        names = ('pretrained_model/model.safetensors', 'ema.safetensors') if job['comparison_family'] == 'base' else ('model.safetensors', 'ema.safetensors')
        # Also verify prepared/resumed weights, including base weights without stored training hashes.
        hashes = {name: watch.digest(c / name) for name in names}
        if job.get('pinned_weight_hashes', hashes) != hashes:
            raise ValueError('Pinned model/EMA changed')
        job['pinned_weight_hashes'] = hashes
        job['asset_hashes'] = {n: watch.digest(c / n) for n in ('norm_stats.json', 'wm_norm_stats.json', 'tokenizer.model')}
    if len(manifest['jobs']) == 2 and manifest['jobs'][0]['asset_hashes'] != manifest['jobs'][1]['asset_hashes']:
        raise ValueError('The base and tactile normalization/tokenizer assets differ')
    runtime, record = pin_runtime(args.root, args.suite, batch)
    manifest.update(runtime=str(runtime), runtime_fingerprint=record['fingerprint'],
                    status='prepared', simulator_test_run=False,
                    render_validation='Settings checked in each formal episode; saved RGB must be reviewed on A800.')
    batch.save_summary(args.suite, manifest)
    return manifest


def launch(job, args, manifest, batch, gpu):
    verify_runtime(args.root, Path(manifest['runtime']))
    job['attempts'] += 1
    work = args.suite / 'work' / job['id'] / f'attempt{job["attempts"]:02d}'
    work.mkdir(parents=True)
    deploy = work / 'deploy.yml'
    batch.write(deploy, dict(policy_name='A800Compare', seed=0, instruction_type='seen',
        pi05_host='127.0.0.1', pi05_port=batch.free_port(18880),
        pi05_python=str(args.root / 'contact-belief-wam/.venv/bin/python'),
        pi05_startup_timeout=1800, pi05_request_timeout=1800,
        pi05_physics_steps_per_action=2, pi05_expert_preflight=True,
        pi05_eval_protocol=batch.EVAL_PROTOCOL))
    env = batch.environment(args.root, work, args.gpu)
    env.update(PAIR_RUNTIME=manifest['runtime'], PAIR_FAMILY=job['comparison_family'],
               PAIR_GPU_UUID=gpu['uuid'], PAIR_RUNTIME_FINGERPRINT=manifest['runtime_fingerprint'],
               CBWAM_ROOT=manifest['runtime'], PYTHONDONTWRITEBYTECODE='1',
               HF_HOME=str(args.root / '.cache/huggingface'),
               TORCH_HOME=str(args.root / '.cache/torch'), MPLCONFIGDIR=str(work / 'matplotlib'))
    env['WAVE0_CHECKPOINT' if job['comparison_family'] == 'base' else 'TACTILE_WAVE0_CHECKPOINT'] = job['checkpoint']
    env.pop('PI05_TRACE_DIR', None)  # The paired adapter records actual initial inputs for both models.
    command = ['bash', str(args.root / 'pi05-univtac/scripts/eval-worker.sh'),
        str(Path(manifest['runtime']) / 'inference/run_pair_native.py'), TASK, 'clean', str(deploy),
        '--headless', '--livestream', '0', '--device', 'cuda:0', '--total_num', '50',
        '--start_seed', '1000000', '--run_name', job['id'], kit_arguments()]
    logfile = work / 'eval.log'
    with logfile.open('w') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
    job.update(state='running', log=str(logfile), pid=child.pid, worker_host=socket.gethostname())
    print(f'{watch.now()} START {job["comparison_family"]}: {logfile}', flush=True)
    return dict(process=child, log=logfile, work=work, start=time.monotonic(), job=job)


def validate_episode_audits(metadata, family, manifest):
    rows = {k: v for k, v in metadata.items() if k.isdigit() and int(k) >= 1000000}
    if set(rows) != {str(s) for s in range(1000000, 1000050)}:
        raise ValueError('Missing/extra formal scene seeds')
    for seed, row in rows.items():
        if row.get('result') == 'error':
            continue  # Runtime errors remain in the 50-episode denominator.
        audit = row.get('policy_execution', {}).get('comparison', {})
        expected = dict(profile=PROFILE, family=family, scene_seed=int(seed),
                        gpu_uuid=manifest['gpu']['uuid'], render_settings=RENDER_SETTINGS,
                        runtime_fingerprint=manifest['runtime_fingerprint'], rng_profile=RNG_PROFILE,
                        input_alignment=INPUT_PROFILE if family == 'tactile' else 'base-head-only')
        if any(audit.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Unmatched render/input contract for seed {seed}')
    return rows


def comparison(suite, manifest, batch):
    if {j['comparison_family'] for j in manifest['jobs']} != {'base', 'tactile'}:
        return
    if not all(j['state'] == 'completed' for j in manifest['jobs']):
        return
    records = {}
    for job in manifest['jobs']:
        directory = Path(job['result']['native_result_dir'])
        records[job['comparison_family']] = validate_episode_audits(batch.read(directory / 'metadata.json'), job['comparison_family'], manifest)
    visual, touch = records['base'], records['tactile']
    for s in visual:
        if visual[s].get('result') != 'error' and touch[s].get('result') != 'error':
            if any(visual[s].get(k) != touch[s].get(k) for k in ('can_size', 'grasp_noise')):
                raise ValueError(f'Initial scene metadata differs for seed {s}')
    counts = {k: sum(v['result'] == 'success' for v in rows.values()) for k, rows in records.items()}
    batch.write(suite / 'comparison.json', dict(status='completed', task=TASK, episodes_per_model=50,
        profile=PROFILE, gpu=manifest['gpu'], successes=counts,
        success_rates={k: v / 50 for k, v in counts.items()},
        change_percentage_points=2 * (counts['tactile'] - counts['base']),
        gained_seeds=[int(s) for s in visual if touch[s]['result'] == 'success' and visual[s]['result'] != 'success'],
        lost_seeds=[int(s) for s in visual if touch[s]['result'] != 'success' and visual[s]['result'] == 'success'],
        note='Matched A800 PathTracing/OptiX evaluation. Review input-audit RGB; do not pool with older RTX realtime results.'))
    print(f'COMPARISON base={counts["base"]}/50 tactile={counts["tactile"]}/50', flush=True)


def publish(args, manifest, batch, status, **details):
    manifest.update(status=status, updated_at=watch.now(), **details)
    batch.save_summary(args.suite, manifest)
    batch.write(args.suite / 'daemon.json', dict(status=status, updated_at=manifest['updated_at'],
        host=socket.gethostname(), pid=os.getpid(), **details))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=watch.ROOT)
    parser.add_argument('--suite', type=Path)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--family', choices=('base', 'tactile', 'both'), default='both')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    suite_name = SUITE if args.family == 'both' else SUITE.replace('base-vs-tactile', args.family)
    args.suite = (args.suite or args.root / 'outputs/eval_suites' / suite_name).resolve()
    if not args.gpu or ',' in args.gpu:
        parser.error('Select exactly one GPU')
    args.suite.mkdir(parents=True, exist_ok=True)
    runner, base = watch.load_runner(args.root), load_base(args.root)
    batch = runner.batch
    with (args.suite / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f'Preparing formal comparison: {args.suite}', flush=True)
        manifest = prepare(args, runner, base)
        if args.prepare_only:
            print('Prepared checkpoints/runtime; no GPU or simulator started.', flush=True)
            return 0
        if all(j['state'] == 'completed' for j in manifest['jobs']):
            comparison(args.suite, manifest, batch)
            publish(args, manifest, batch, 'completed', active_family=None)
            return 0
        gpu_lock, active = None, None
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Signal {signum}')
        previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            while gpu_lock is None:
                gpu_lock, gpu, reason = watch.claim_gpu(args, runner)
                if gpu_lock is None:
                    publish(args, manifest, batch, 'waiting_for_gpu', wait_reason=reason)
                    time.sleep(30)
            if 'A800' not in gpu['name']:
                raise RuntimeError(f'This sm80 submission requires an A800, got {gpu["name"]}')
            if manifest.get('gpu', gpu)['uuid'] != gpu['uuid']:
                raise RuntimeError('A paired suite must run on the same GPU; use a new --suite')
            manifest['gpu'] = gpu
            for job in manifest['jobs']:
                if job['state'] == 'completed':
                    continue
                active = launch(job, args, manifest, batch, gpu)
                publish(args, manifest, batch, 'evaluating', active_family=job['comparison_family'])
                last_progress, last_at = '', time.monotonic()
                while active['process'].poll() is None:
                    content = active['log'].read_text(errors='replace')
                    found = re.findall(r'Total \d+/\d+\([^\n]*', content)
                    progress = found[-1] if found else ''
                    if progress != last_progress:
                        last_progress, last_at = progress, time.monotonic()
                        print(f'{watch.now()} {job["comparison_family"]}: {progress}', flush=True)
                    if time.monotonic() - last_at > 1800 or time.monotonic() - active['start'] > 86400:
                        raise TimeoutError('Formal worker stalled or exceeded 24 hours; logs retained')
                    publish(args, manifest, batch, 'evaluating', active_family=job['comparison_family'], progress=progress)
                    time.sleep(15)
                code = active['process'].returncode
                result = batch.result_from_log(active['log'].read_text(errors='replace'), code, 50)
                if result['status'] != 'completed':
                    raise RuntimeError(f'{job["comparison_family"]} did not finish 50 episodes; see {active["log"]}')
                native = Path(result['native_result_dir'] or '/')
                if not native.is_relative_to(active['work'] / 'native'):
                    raise ValueError('Unexpected result path')
                validate_episode_audits(batch.read(native / 'metadata.json'), job['comparison_family'], manifest)
                result = batch.finish(active, args.suite, 50, code)
                active = None
                if result['status'] != 'completed':
                    raise RuntimeError('Native result validation failed')
                publish(args, manifest, batch, 'evaluating')
            comparison(args.suite, manifest, batch)
            publish(args, manifest, batch, 'completed', active_family=None)
            return 0
        except BaseException as exc:
            if active is not None:
                batch.stop_group(active['process'])
                active['job']['state'] = 'interrupted_or_error'
            publish(args, manifest, batch, 'interrupted_or_error', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if gpu_lock is not None:
                gpu_lock.close()
            for s, handler in previous.items():
                signal.signal(s, handler)


if __name__ == '__main__':
    raise SystemExit(main())
