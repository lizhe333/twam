#!/usr/bin/env python3
"""Evaluate the completed 5k+10k HDMI raw-weight recovery model on RTX 4090."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path

import watch_eval_10k as watch
import reeval_hdmi_aligned as aligned
from eval_hdmi_finetune import snapshot

RUN = 'hdmi-head-only-recovery-finetune-003-fresh-noema-s1-5k-s2-10k'
SUITE = RUN + '-4090-aligned-eval50'


def discover(run_root, task, runner):
    ck = run_root / 'tasks' / task / 'stage2/checkpoints/step_00010000'
    read = runner.batch.read
    complete, contract, identity = [read(ck / name) for name in
                                   ('complete.json', 'training_contract.json', 'run_identity.json')]
    expected = dict(stage='stage2', task='insert_HDMI', variant='full', mode='full',
                    steps=10000, expert_only=False, data_mix='2:1',
                    ema_enabled=False, finetune_use_ema=False)
    if (complete.get('stage') != 'stage2' or complete.get('step') != 10000
            or complete.get('training_contract') != contract or identity.get('contract') != contract
            or identity.get('fingerprint') != complete.get('identity')
            or any(contract.get(k) != v for k, v in expected.items())
            or complete.get('ema_sha256') is not None):
        raise ValueError('Expected the completed 10k mixed-data HDMI checkpoint without EMA')
    teacher = read(run_root / 'shared_wm/stage1/checkpoints/step_00005000/complete.json')
    if (teacher.get('stage') != 'stage1' or teacher.get('step') != 5000
            or teacher.get('identity') != contract.get('stage1_checkpoint')):
        raise ValueError('Stage 2 did not inherit the completed 5k Stage 1')
    source = ck / 'pretrained_model'
    deploy, cfg = read(source / 'deployment.json'), read(source / 'config.json')
    expected = dict(task=task, variant='full', mode='full', aligned=True, camera='head_only',
                    simulator_action='absolute_joint', action_chunk_length=50,
                    executed_action_prefix=10, use_ema=False, ema_file=None)
    if (any(deploy.get(k) != v for k, v in expected.items())
            or cfg.get('camera_mode') != 'head_only' or cfg.get('k') != 10
            or not cfg.get('independent_tactile')):
        raise ValueError('Unexpected raw-weight deployment contract')
    return dict(id=f'{task}__head-only-tactile__{RUN}__step10000', task=task, run=RUN,
                source=str(source), step=10000, camera='head_only', policy_family='tactile_wave0',
                variant='full', use_ema=False, state='pending', attempts=0,
                model_sha256=complete['model_sha256'], ema_sha256=None,
                training_identity=complete['identity'], training_data_mix='2:1',
                input_alignment=aligned.PROFILE, backbone_precision='bfloat16',
                world_contact_precision='float32', tactile_device='cuda')


def pin_reference(root, suite, batch):
    directory = root / 'outputs/eval_suites/twam-single-view/results/insert_HDMI'
    result, meta = batch.read(directory / 'result.json'), batch.read(directory / 'metadata.json')
    seeds = [str(s) for s in range(1000000, 1000050)]
    if (result.get('status') != 'completed' or result.get('completed_episodes') != 50
            or result.get('evaluation_protocol') != batch.EVAL_PROTOCOL
            or result.get('input_alignment') != aligned.PROFILE
            or any(meta[s]['policy_execution'].get('input_alignment') != aligned.PROFILE for s in seeds)):
        raise ValueError('Expected the completed 50-seed training-aligned historical baseline')
    record = dict(result_file=str(directory / 'result.json'), result=result,
                  use_ema=True, outcomes={s: meta[s]['result'] for s in seeds})
    path = suite / 'comparison_reference.json'
    if path.exists() and batch.read(path) != record:
        raise ValueError('Pinned historical comparison changed')
    batch.write(path, record)


def compare(suite, manifest, batch):
    job = manifest['jobs'][0]
    if job['state'] != 'completed':
        return
    reference = batch.read(suite / 'comparison_reference.json')
    result = job['result']
    meta = batch.read(Path(result['native_result_dir']) / 'metadata.json')
    for seed in reference['outcomes']:
        row = meta[seed]
        if row['result'] == 'error':
            continue
        execution = row['policy_execution']
        expected = dict(input_alignment=aligned.PROFILE, camera='head_only', use_ema=False,
                        chunk_size=50, execute_steps=10, physical_steps_per_action=2, physics_call_limit=1200)
        if any(execution.get(k) != v for k, v in expected.items()) or execution['inference']['use_ema'] is not False:
            raise ValueError(f'Unexpected execution contract for seed {seed}')
    before, after = reference['result']['successes'], result['successes']
    batch.write(suite / 'comparison.json', dict(
        status='completed', task='insert_HDMI', episodes=50, input_alignment=aligned.PROFILE,
        baseline_successes=before, finetuned_successes=after,
        baseline_success_rate=before / 50, finetuned_success_rate=after / 50,
        change_percentage_points=2 * (after - before),
        baseline_use_ema=True, finetuned_use_ema=False,
        gained_seeds=sorted(int(s) for s, v in reference['outcomes'].items() if v != 'success' and meta[s]['result'] == 'success'),
        lost_seeds=sorted(int(s) for s, v in reference['outcomes'].items() if v == 'success' and meta[s]['result'] != 'success'),
        exceptions=result['exceptions'], policy_failures=result.get('policy_failures', 0),
        baseline_result=reference['result_file'], finetuned_result=result['native_result_dir'],
        note='Same RTX 4090, scene seeds, aligned input preprocessing and action budget. '
             'Historical baseline uses EMA; current model uses ordinary weights. Policy noise is not paired. '
             'This compares deployed variants, not an isolated training-length or EMA ablation.'))
    print(f'COMPARISON historical EMA={before}/50 new raw={after}/50', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan-only', action='store_true')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    args.root = watch.ROOT
    args.run_root = args.root / 'twam/runs' / RUN / 'full'
    args.suite = args.root / 'outputs/eval_suites' / SUITE
    args.episodes, args.poll_seconds, args.max_attempts = 50, 15, 2
    args.timeout, args.stall_timeout = 43200, 1800
    args.skip_sim_check, args.skip_driver_version_check = True, False
    args.model_python = args.root / 'contact-belief-wam/.venv/bin/python'
    os.environ.update(TWAM_ROOT=str(args.root),
                      PI05_SIM_GRAPHICS_LIBRARY_PATH=str(args.root / '.cache/sim-graphics-runtime/lib'))
    os.environ['PATH'] = str(args.model_python.parent) + os.pathsep + os.environ.get('PATH', '')
    args.suite.mkdir(parents=True, exist_ok=True)
    with (args.suite / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = watch.load_runner(args.root)
        watch.TASKS, watch.discover_task, watch.snapshot = ('insert_HDMI',), discover, snapshot
        runtime, record = aligned.pin_runtime(args.root, args.suite, runner.batch, use_ema=False)
        manifest = watch.prepare(args, runner)
        manifest.update(use_ema=False, input_alignment=aligned.PROFILE,
                        runtime=str(runtime), runtime_fingerprint=record['fingerprint'])
        watch.refresh(args, manifest, runner)
        pin_reference(args.root, args.suite, runner.batch)
        runner.batch.save_summary(args.suite, manifest)
        runner.launch = lambda *values: aligned.launch(*values, runner=runner)
        watch.claim_gpu = aligned.idle_gpu_guard(watch.claim_gpu, 0)
        if not args.scan_only:
            (args.suite / 'daemon.pid').write_text(f'{os.getpid()}\n')
        code = watch.run_monitor(args, manifest, runner)
        if not args.scan_only:
            compare(args.suite, manifest, runner.batch)
        return code


if __name__ == '__main__':
    raise SystemExit(main())
