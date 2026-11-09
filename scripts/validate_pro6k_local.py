#!/usr/bin/env python3
"""Run complete local validation episodes using the formal Isaac 5.1 worker."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import signal
import subprocess
import time

import reeval_pro6k as queue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=queue.watch.ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-suite', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', choices=queue.watch.TASKS, default=list(queue.watch.TASKS))
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.env_home = args.root / '.cache' / queue.prepare_env.ENV_NAME
    args.suite = args.output.resolve()
    args.suite.mkdir(parents=True, exist_ok=True)
    ready = queue.prepare_env.verify_ready(args.env_home)
    if ready.get('native_architectures') != list(queue.prepare_env.NATIVE_ARCHITECTURES):
        raise RuntimeError('Finish the Ada/Blackwell native build before local validation')
    batch = queue.watch.load_runner(args.root).batch
    source = batch.read(args.checkpoint_suite / 'manifest.json')
    if source['profile'] != queue.PROFILE:
        raise ValueError('Checkpoint suite has a different profile')
    gpu = batch.gpu_info(str(args.gpu))
    gpu['index'] = args.gpu
    with (args.root / 'outputs/eval_suites' / f'.{gpu["uuid"]}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        processes = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
            '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        if processes:
            raise RuntimeError(f'Local GPU is still in use: {processes}')
        runtime, record = queue.pin_runtime(args.root, args.suite, batch)
        manifest = dict(runtime=str(runtime), runtime_fingerprint=record['fingerprint'])
        report = dict(purpose='local development validation', formal_result=False, gpu=gpu,
                      runtime_fingerprint=record['fingerprint'], episodes_per_task=1, tasks={})
        report_path = args.suite / 'validation.json'
        batch.write(report_path, report)
        for task in args.tasks:
            original, = [j for j in source['jobs'] if j['task'] == task]
            job = dict(original, id=task, attempts=0, state='pending')
            pinned = batch.read(Path(job['checkpoint']) / 'pinned.json')
            if pinned['identity'] != job['training_identity'] or pinned['ema_sha256'] != job['ema_sha256']:
                raise ValueError(f'Checkpoint identity mismatch for {task}')
            item = queue.launch(job, args, manifest, batch, gpu, set(), validation=True)
            def interrupted(signum, frame):
                raise KeyboardInterrupt(f'Signal {signum}')
            previous = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
            try:
                deadline, progress = time.monotonic() + 3600, ''
                while item['process'].poll() is None:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f'Validation exceeded one hour: {task}')
                    lines = item['log'].read_text(errors='replace').splitlines()
                    current = next((line for line in reversed(lines)
                                    if re.search(r'(?:Total|Final Result:)\s+\d+/\d+', line)), '')
                    if current and current != progress:
                        print(current, flush=True)
                        progress = current
                    time.sleep(5)
                native = item['work'] / 'native'
                if item['process'].returncode != 0:
                    raise RuntimeError(f'{task} exited {item["process"].returncode}; see {item["log"]}')
                metadata = batch.read(native / 'metadata.json')
                row = metadata['1000000']
                execution = row.get('policy_execution', {})
                feedback = row.get('tactile_feedback', {})
                if (row['result'] not in ('success', 'failed') or row.get('failure_kind')
                        or execution.get('physical_action_calls', 0) <= 0):
                    raise RuntimeError(f'{task} did not complete a real policy episode: {row.get("failure_reason", row["result"])}')
                if execution.get('transfer', {}).get('runtime_fingerprint') != record['fingerprint']:
                    raise RuntimeError(f'{task} has no valid RGB/render/input audit')
                if not feedback.get('actions') or not feedback.get('updates'):
                    raise RuntimeError(f'{task} has no recorded tactile feedback')
                report['tasks'][task] = dict(status='passed', episode_result=row['result'],
                    physical_action_calls=execution['physical_action_calls'],
                    tactile_updates=len(feedback['updates']), log=str(item['log']),
                    rgb=str(native / 'input-audit/1000000-head.png'))
                print(f'LOCAL VALIDATION PASSED {task}: {report["tasks"][task]}', flush=True)
            except BaseException as exc:
                report['tasks'][task] = dict(status='error', reason=str(exc), log=str(item['log']))
                raise
            finally:
                batch.stop_group(item['process'])
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
                batch.write(report_path, report)


if __name__ == '__main__':
    main()
