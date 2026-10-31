#!/usr/bin/env python3
"""Evaluate the completed HDMI recovery fine-tune against the saved 50-seed baseline."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import watch_eval_10k as watch

RUN = 'hdmi-head-only-recovery-finetune-001'
SUITE = 'hdmi-head-only-recovery-finetune-001-2k-4090-fixed-seeds'
BASELINE = 'tactile-head-only-expert-mb16-003-10k-fixed-seeds'
PROFILE = 'native-raw-tactile-command-history-v1'


def discover(run_root, task, runner):
    ck = run_root / 'tasks' / task / 'stage2/checkpoints/step_00002000'
    read = runner.batch.read
    complete, contract, identity = [read(ck / name) for name in
                                  ('complete.json', 'training_contract.json', 'run_identity.json')]
    expected = dict(stage='stage2', task='insert_HDMI', variant='full', mode='full',
                    steps=2000, expert_only=False, data_mix='2:1')
    if (complete.get('stage') != 'stage2' or complete.get('step') != 2000
            or complete.get('training_contract') != contract or identity.get('contract') != contract
            or identity.get('fingerprint') != complete.get('identity')
            or any(contract.get(k) != v for k, v in expected.items())):
        raise ValueError('Expected the committed HDMI recovery fine-tune at step 2000')
    source = ck / 'pretrained_model'
    deploy, cfg = read(source / 'deployment.json'), read(source / 'config.json')
    expected = dict(task=task, variant='full', mode='full', aligned=True, camera='head_only',
                    simulator_action='absolute_joint', action_chunk_length=50,
                    executed_action_prefix=10, use_ema=True, ema_file='ema.safetensors')
    if (any(deploy.get(k) != v for k, v in expected.items())
            or cfg.get('camera_mode') != 'head_only' or cfg.get('k') != 10
            or not cfg.get('independent_tactile')):
        raise ValueError('Unexpected fine-tuned deployment contract')
    return dict(id=f'{task}__head-only-tactile__{RUN}__step2000', task=task, run=RUN,
                source=str(source), step=2000, camera='head_only', policy_family='tactile_wave0',
                variant='full', use_ema=True, state='pending', attempts=0,
                model_sha256=complete['model_sha256'], ema_sha256=complete['ema_sha256'],
                training_identity=complete['identity'], training_data_mix='2:1')


def snapshot(job, suite, runner):
    source = Path(job['source'])
    target = suite / 'checkpoints' / job['task'] / job['run'] / str(job['step'])
    if not target.exists():
        temp = target.with_name(target.name + '.partial')
        if temp.exists():
            shutil.rmtree(temp)
        temp.mkdir(parents=True)
        try:
            for name in runner.ARTIFACTS:
                if name == 'ema.safetensors' and not job['use_ema']:
                    continue
                runner.batch.link_or_copy(source / name, temp / name)
            for name in ('complete.json', 'training_contract.json', 'run_identity.json'):
                runner.batch.link_or_copy(source.parent / name, temp / name)
            for name, key in [('model.safetensors', 'model_sha256'), ('ema.safetensors', 'ema_sha256')]:
                if key == 'ema_sha256' and not job['use_ema']:
                    continue
                if watch.digest(temp / name) != job[key]:
                    raise ValueError(f'Incomplete or changed weight file: {name}')
            temp.rename(target)
        except BaseException:
            shutil.rmtree(temp, ignore_errors=True)
            raise
    complete = runner.batch.read(target / 'complete.json')
    if complete['identity'] != job['training_identity']:
        raise ValueError('Pinned checkpoint identity changed')
    job['checkpoint'] = str(target)


def pin_runtime(root, suite, batch):
    runtime = suite / 'runtime'
    if runtime.exists():
        for name in ('pyproject.toml', 'scripts/cloud_train.sh'):
            if not (runtime / name).is_file():
                raise ValueError(f'Pinned runtime lacks project marker: {name}')
        record = batch.read(runtime / 'runtime_manifest.json')
        for name, digest in record['files'].items():
            if watch.digest(runtime / name) != digest:
                raise ValueError(f'Pinned runtime changed: {name}')
        return runtime, record
    temp = suite / 'runtime.partial'
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir()
    project = root / 'twam'
    try:
        ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache')
        for folder in ('src', 'inference', 'vendor/wave0'):
            shutil.copytree(project / folder, temp / folder, ignore=ignore)
        for name in ('pyproject.toml', 'scripts/cloud_train.sh'):
            (temp / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(project / name, temp / name)
        (temp / 'vendor/transformers').symlink_to((project / 'vendor/transformers').resolve(), target_is_directory=True)
        shutil.copy2(root / 'UniVTAC/policy/TactileWave0Wam/deploy_policy.py', temp / 'native_policy_original.py')
        shutil.copy2(root / 'UniVTAC/scripts/eval_policy.py', temp / 'native_eval.py')
        # Keep the original adapter's raw tactile pixels and actual command history.
        # The isolated bridge changes only its import location and audit metadata.
        (temp / 'inference/native_policy.py').write_text('''import os
from pathlib import Path
from contact_belief_wam import sim_eval
from policy.TactileWave0Wam import deploy_policy as native
native.PROJECT = Path(os.environ['CBWAM_ROOT']).resolve()
assert Path(sim_eval.__file__).resolve().is_relative_to(native.PROJECT / 'src')
class Policy(native.Policy):
    def record_episode(self, task, outcome):
        super().record_episode(task, outcome)
        task.metadata['policy_execution']['input_profile'] = 'native-raw-tactile-command-history-v1'
        task.metadata['policy_execution']['runtime_fingerprint'] = os.environ['HDMI_EVAL_RUNTIME']
''')
        p = temp / 'inference/run_native.py'
        p.write_text(p.read_text().replace('policy.TactileWave0WamAligned', 'policy.HdmiRecoveryEval'))
        files = {str(p.relative_to(temp)): watch.digest(p) for p in sorted(temp.rglob('*'))
                 if p.is_file() and not p.is_symlink()}
        record = dict(profile=PROFILE, files=files, created_at=watch.now(),
                      fingerprint=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
        batch.write(temp / 'runtime_manifest.json', record)
        temp.rename(runtime)
        return runtime, record
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def launch(job, root, suite, gpu, episodes, port, model_python, **kwargs):
    if '4090' not in RUNNER.batch.gpu_info(gpu)['name']:
        raise ValueError('This comparison requires the RTX 4090')
    runtime, record = pin_runtime(root, suite, RUNNER.batch)
    job['attempts'] += 1
    work = suite / 'work' / job['id'] / f'attempt{job["attempts"]:02d}'
    work.mkdir(parents=True)
    deploy = work / 'deploy.yml'
    RUNNER.batch.write(deploy, dict(policy_name='HdmiRecoveryEval', seed=0, instruction_type='seen',
        pi05_host='127.0.0.1', pi05_port=port, pi05_python=str(model_python),
        pi05_startup_timeout=1800, pi05_request_timeout=1800, pi05_physics_steps_per_action=2,
        pi05_expert_preflight=True, pi05_eval_protocol=RUNNER.batch.EVAL_PROTOCOL))
    env = RUNNER.batch.environment(root, work, gpu)
    env.update(TACTILE_WAVE0_CHECKPOINT=job['checkpoint'], CBWAM_ROOT=str(runtime),
               HDMI_EVAL_RUNTIME=record['fingerprint'], PYTHONDONTWRITEBYTECODE='1')
    env.pop('PI05_TRACE_DIR', None)
    cmd = ['bash', str(root / 'pi05-univtac/scripts/eval-worker.sh'),
           str(runtime / 'inference/run_native.py'), 'insert_HDMI', 'clean', str(deploy),
           '--headless', '--livestream', '0', '--device', 'cuda:0', '--total_num', str(episodes),
           '--start_seed', '1000000', '--run_name', job['id']]
    log = work / 'eval.log'
    with log.open('w') as stream:
        child = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    job.update(state='running', log=str(log), pid=child.pid)
    print(f'{watch.now()} START {job["id"]}', flush=True)
    return dict(process=child, log=log, work=work, start=time.monotonic(), warm=False,
                progress_time=time.monotonic(), progress='', job=job)


def comparison(suite, manifest, batch):
    job = manifest['jobs'][0]
    if job['state'] != 'completed':
        return
    ref = batch.read(suite / 'baseline.json')
    current = batch.read(Path(job['result']['native_result_dir']) / 'metadata.json')
    seeds = set(str(s) for s in range(1000000, 1000050))
    if not seeds.issubset(current):
        raise ValueError('Missing comparison seeds')
    for s in seeds:
        if current[s]['result'] == 'error':
            continue
        execution = current[s]['policy_execution']
        if (execution['input_profile'] != PROFILE or execution['camera'] != 'head_only'
                or execution['chunk_size'] != 50 or execution['execute_steps'] != 10
                or execution['physical_steps_per_action'] != 2 or execution['physics_call_limit'] != 1200
                or not execution['use_ema']):
            raise ValueError(f'Unexpected episode protocol at {s}')
    before = ref['result']['successes']
    after = job['result']['successes']
    report = dict(status='completed', task='insert_HDMI', episodes=50,
        baseline_successes=before, finetuned_successes=after,
        baseline_success_rate=before / 50, finetuned_success_rate=after / 50,
        change_percentage_points=2 * (after - before),
        gained_seeds=sorted(int(s) for s in seeds if current[s]['result'] == 'success' and ref['outcomes'][s] != 'success'),
        lost_seeds=sorted(int(s) for s in seeds if current[s]['result'] != 'success' and ref['outcomes'][s] == 'success'),
        baseline_result=ref['result_file'], finetuned_result=job['result']['native_result_dir'],
        note='Same RTX 4090, fixed scene seeds, raw tactile/command history and execution budget. '
             'Historical baseline; policy noise is not paired. Success-rate change alone does not identify the causal module.')
    batch.write(suite / 'comparison.json', report)
    print(f'COMPARISON baseline={before}/50 finetuned={after}/50', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan-only', action='store_true')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    args.root = watch.ROOT
    args.run_root = args.root / 'twam/runs' / RUN / 'full'
    args.suite = args.root / 'outputs/eval_suites' / SUITE
    args.episodes, args.poll_seconds, args.max_attempts = 50, 15, 3
    args.timeout, args.stall_timeout = 43200, 1800
    # The same GPU/simulator is already running successful collection; native
    # expert preflight still runs under the unchanged formal evaluation protocol.
    args.skip_sim_check, args.skip_driver_version_check = True, False
    args.model_python = args.root / 'contact-belief-wam/.venv/bin/python'
    args.suite.mkdir(parents=True, exist_ok=True)
    os.environ.update(TWAM_ROOT=str(args.root),
        PI05_SIM_GRAPHICS_LIBRARY_PATH=str(args.root / '.cache/sim-graphics-runtime/lib'))
    os.environ['PATH'] = str(args.model_python.parent) + os.pathsep + os.environ.get('PATH', '')
    global RUNNER
    RUNNER = watch.load_runner(args.root)
    batch = RUNNER.batch
    watch.TASKS, watch.discover_task, watch.snapshot = ('insert_HDMI',), discover, snapshot
    RUNNER.launch = launch
    with (args.suite / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reference_root = args.root / 'outputs/eval_suites' / BASELINE
        old = next(j for j in batch.read(reference_root / 'manifest.json')['jobs'] if j['task'] == 'insert_HDMI')
        result_file = reference_root / 'results' / old['id'] / 'result.json'
        result = batch.read(result_file)
        meta = batch.read(result_file.parent / 'metadata.json')
        if result['status'] != 'completed' or result['completed_episodes'] != 50:
            raise ValueError('Expected the completed 50-seed baseline')
        ref = dict(result_file=str(result_file), result=result,
                   outcomes={str(s): meta[str(s)]['result'] for s in range(1000000, 1000050)})
        batch.write(args.suite / 'baseline.json', ref)
        path = args.suite / 'manifest.json'
        manifest = batch.read(path) if path.exists() else dict(
            training_root=str(args.run_root), tasks=['insert_HDMI'], episodes=50, seed_start=1000000,
            seed_end=1000049, evaluation_protocol=batch.EVAL_PROTOCOL, camera='head_only', step=2000,
            schema_version=1, run=RUN, jobs=[], status='prepared', created_at=watch.now(), input_profile=PROFILE)
        selection = dict(training_root=str(args.run_root), tasks=['insert_HDMI'], episodes=50,
                         seed_start=1000000, seed_end=1000049, step=2000, input_profile=PROFILE)
        if any(manifest.get(k) != v for k, v in selection.items()):
            raise ValueError('Existing evaluation suite uses a different selection')
        if not args.scan_only:
            watch.recover_jobs(args, manifest, RUNNER)
        if not manifest['jobs']:
            job = discover(args.run_root, 'insert_HDMI', RUNNER)
            snapshot(job, args.suite, RUNNER)
            manifest['jobs'].append(job)
            print(f'{watch.now()} ENQUEUE insert_HDMI step=2000', flush=True)
        runtime, record = pin_runtime(args.root, args.suite, batch)
        manifest.update(runtime=str(runtime), runtime_fingerprint=record['fingerprint'])
        batch.save_summary(args.suite, manifest)
        code = watch.run_monitor(args, manifest, RUNNER)
        comparison(args.suite, manifest, batch)
        return code


if __name__ == '__main__':
    raise SystemExit(main())
