#!/usr/bin/env python3
"""Eight tactile-only, 50-episode transfer jobs on RTX PRO 6000 GPUs."""
from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
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
import prepare_pro6k as prepare_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))
from pro6k_contract import (INPUT_PROFILE, PROFILE, UNIVTAC_COMMIT, RESET_TIMEOUT_SECONDS, CONTACT_RESET_PATCH,
                           WORKER_ROUTING, kit_arguments, validate_audits)

SUITE = 'baidu-tactile-8task-pro6k-isaac51-transfer-v8'
GIB = 1024 ** 3


def inventory(path):
    return {str(p.relative_to(path)): watch.digest(p) for p in sorted(path.rglob('*'))
            if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts
            and p.name != 'runtime_manifest.json'}


def pin_runtime(root, suite, batch):
    runtime = suite / 'runtime'
    if runtime.exists():
        record = batch.read(runtime / 'runtime_manifest.json')
        if record['profile'] != PROFILE or inventory(runtime) != record['files']:
            raise RuntimeError('Frozen inference runtime changed; use a new suite')
        return runtime, record
    temporary = suite / 'runtime.partial'
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache', '.git')
    project = root / 'twam'
    try:
        # Preserve which native and graphics libraries this suite was prepared
        # against, including repairs that do not change model source files.
        shutil.copy2(root / '.cache' / prepare_env.ENV_NAME / 'ready.json',
                     temporary / 'prepared_environment.json')
        for name in ('src', 'inference', 'vendor/wave0', 'vendor/transformers'):
            shutil.copytree(project / name, temporary / name, ignore=ignore)
        for name in ('pyproject.toml', 'scripts/cloud_train.sh'):
            (temporary / name).parent.mkdir(exist_ok=True, parents=True)
            shutil.copy2(project / name, temporary / name)
        shutil.copytree(root / 'pi05-univtac/src/pi05_univtac', temporary / 'support/pi05_univtac', ignore=ignore)
        for family, folder in [('tactile', 'TactileWave0Wam'), ('pi05', 'Pi05Wam')]:
            shutil.copy2(root / f'UniVTAC/policy/{folder}/deploy_policy.py', temporary / f'native_{family}.py')
        native = temporary / 'native_tactile.py'
        text = native.read_text()
        substitutions = {
            "'--device', 'cuda', '--tactile-device', 'cuda'":
                "'--device', os.environ['PRO6K_DEVICE'], '--tactile-device', os.environ['PRO6K_DEVICE']",
            "tactile_device='cuda'": "tactile_device=os.environ['PRO6K_DEVICE']",
        }
        for old, new in substitutions.items():
            if text.count(old) != 1:
                raise RuntimeError(f'Native policy template changed: {old}')
            text = text.replace(old, new)
        native.write_text(text)
        (temporary / 'src/contact_belief_wam/__main__.py').write_text(
            "import os, sys\nfrom pathlib import Path\nimport torch\n"
            "sys.path.insert(0, str(Path(os.environ['PRO6K_RUNTIME']) / 'inference'))\n"
            "torch.cuda.set_device(os.environ['PRO6K_DEVICE'])\n"
            "from pair_servers import tactile_main\nraise SystemExit(tactile_main())\n")
        # Reuse the exact fixed-seed counting/rejection loop without importing
        # the Isaac 4.5 app entry or its expert-preflight/data dependencies.
        tree = ast.parse((root / 'UniVTAC/scripts/eval_policy.py').read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('log', 'eval_policy')]
        if {node.name for node in functions} != {'log', 'eval_policy'}:
            raise RuntimeError('Native fixed-seed loop not found')
        (temporary / 'inference/fixed_seed_loop.py').write_text(
            'from shutil import ExecError\nimport time, json, traceback\nfrom pathlib import Path\n'
            'from typing import Literal\n' + '\n\n'.join(ast.unparse(node) for node in functions) + '\n')
        files = inventory(temporary)
        record = dict(profile=PROFILE, files=files, created_at=watch.now(),
            fingerprint=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
        batch.write(temporary / 'runtime_manifest.json', record)
        temporary.rename(runtime)
        return runtime, record
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def resources(proc=Path('/proc'), cgroup=Path('/sys/fs/cgroup')):
    """Respect the container's cgroup limit, not the host's advertised RAM/CPU."""
    values = {k: int(v.split()[0]) * 1024 for k, v in
              (line.split(':', 1) for line in (proc / 'meminfo').read_text().splitlines())}
    total, available = values['MemTotal'], values['MemAvailable']
    cpus = len(os.sched_getaffinity(0))
    candidates = [(cgroup / 'memory.max', cgroup / 'memory.current'),
                  (cgroup / 'memory/memory.limit_in_bytes', cgroup / 'memory/memory.usage_in_bytes')]
    # Usually /sys/fs/cgroup is the container's namespace root. Also account
    # for hosts exposing the full cgroup hierarchy.
    if (proc / 'self/cgroup').is_file():
        for line in (proc / 'self/cgroup').read_text().splitlines():
            _, controllers, name = line.split(':', 2)
            directory = cgroup / name.lstrip('/')
            if not controllers:
                candidates.append((directory / 'memory.max', directory / 'memory.current'))
            elif 'memory' in controllers.split(','):
                directory = cgroup / 'memory' / name.lstrip('/')
                candidates.append((directory / 'memory.limit_in_bytes', directory / 'memory.usage_in_bytes'))
    for limit_path, used_path in candidates:
        if limit_path.is_file() and used_path.is_file():
            limit = limit_path.read_text().strip()
            if limit != 'max':
                cap, used = int(limit), int(used_path.read_text())
                stat = limit_path.parent / 'memory.stat'
                reclaimable = 0
                if stat.is_file():
                    stats = dict(line.split() for line in stat.read_text().splitlines())
                    reclaimable = int(stats.get('total_inactive_file', stats.get('inactive_file', '0')))
                total, available = min(total, cap), min(available, max(0, cap - used + reclaimable))
    if (cgroup / 'cpu.max').is_file():
        quota, period = (cgroup / 'cpu.max').read_text().split()
        if quota != 'max':
            cpus = min(cpus, int(quota) / int(period))
    elif (cgroup / 'cpu/cpu.cfs_quota_us').is_file():
        quota = int((cgroup / 'cpu/cpu.cfs_quota_us').read_text())
        if quota > 0:
            cpus = min(cpus, quota / int((cgroup / 'cpu/cpu.cfs_period_us').read_text()))
    return dict(total_bytes=total, available_bytes=available, cpus=cpus)


def capacity(available, gpu_count, per_gpu):
    # 48 GiB per worker plus 16 GiB for the scheduler/shared runtime; admit
    # only with 48 GiB still free to absorb model+EMA loading peaks.
    return max(0, min(gpu_count * per_gpu,
                      int(max(0, available['total_bytes'] - 16 * GIB) // (48 * GIB)),
                      int(available['cpus'] // 8)))


def validate_dedicated_resources(gpus, host, per_gpu):
    """Fail before launch if eight dedicated workers cannot be admitted."""
    count = len(watch.TASKS)
    if len(gpus) != count or per_gpu != 1:
        raise RuntimeError(f'Dedicated task mode requires {count} GPUs in one instance and --per-gpu 1')
    if capacity(host, len(gpus), per_gpu) < count or host['available_bytes'] < count * 48 * GIB:
        raise RuntimeError(f'Eight simultaneous workers require at least 64 CPU cores, '
                           f'400 GiB total RAM and 384 GiB available RAM; allocation: {host}')
    insufficient = [str(g['index']) for g in gpus if g['free_mib'] < 26 * 1024]
    if insufficient:
        raise RuntimeError(f'Each dedicated GPU needs at least 26 GiB free; insufficient GPUs: {insufficient}')


def assign_dedicated_tasks(manifest, indices):
    jobs = manifest['jobs']
    if [job['task'] for job in jobs] != list(watch.TASKS) or len(indices) != len(jobs):
        raise RuntimeError('Dedicated GPU assignment requires the eight expected tasks and GPU indices')
    assignment = {str(index): job['task'] for index, job in zip(indices, jobs)}
    previous = manifest.get('dedicated_task_gpus')
    if previous is not None and previous != assignment:
        raise RuntimeError('This suite has a different fixed GPU assignment; use a new suite')
    manifest['dedicated_task_gpus'] = assignment
    return assignment


def select_pending_job(pending, gpu, assignment=None):
    if assignment is None:
        return pending[0] if pending else None
    task = assignment[str(gpu['index'])]
    return next((job for job in pending if job['task'] == task), None)


def can_start(active, gpu, host, *, limit, per_gpu):
    same_gpu = [a for a in active.values() if a['job']['gpu_uuid'] == gpu['uuid']]
    return (len(active) < limit and len(same_gpu) < per_gpu
            and all(a['warm'] for a in same_gpu)
            and gpu['free_mib'] >= 26 * 1024
            and host['available_bytes'] >= 48 * GIB)


def gpu_info(index):
    fields = 'index,name,uuid,memory.total,memory.free,driver_version,compute_cap'
    try:
        result = subprocess.check_output(['nvidia-smi', '-i', str(index), f'--query-gpu={fields}',
                                         '--format=csv,noheader,nounits'], text=True, timeout=20)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f'Cannot access GPU {index}; check the GPUs allocated to this instance') from exc
    rows = result.strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError('Select one physical GPU per index')
    i, name, uuid, total, free, driver, capability = [x.strip() for x in rows[0].split(',')]
    # Baidu exposes the user's Pro 6K allocation with an opaque product name.
    # Accept that observed alias, while still checking the hardware properties.
    supported_name = ('RTX PRO 6000' in name and 'Blackwell' in name) or name == 'NVIDIA RPBZZZ6'
    if not supported_name or capability != '12.0':
        raise RuntimeError(f'Expected RTX PRO 6000 Blackwell, got {name} / {capability}')
    # Allow reserved framebuffer memory on a 96 GiB card, but reject smaller
    # devices even if their product name and compute capability match.
    if int(total) < 90 * 1024:
        raise RuntimeError(f'Expected a 96 GB-class Pro 6K GPU (at least 90 GiB reported), got {name} / {total} MiB')
    if int(driver.split('.')[0]) < 580:
        raise RuntimeError(f'Isaac 5.1 submission requires an R580-or-newer driver, found {driver}')
    return dict(index=int(i), name=name, uuid=uuid, total_mib=int(total), free_mib=int(free), driver=driver)


def validate_allocation(gpus, visibility=None):
    """Removing CUDA remapping must never escape the job's GPU allocation."""
    visibility = os.environ.get('CUDA_VISIBLE_DEVICES') if visibility is None else visibility
    if visibility is None:
        return
    allowed = {x.strip() for x in visibility.split(',')}
    for gpu in gpus:
        if str(gpu['index']) not in allowed and gpu['uuid'] not in allowed:
            raise RuntimeError(f'GPU {gpu["index"]} is outside CUDA_VISIBLE_DEVICES={visibility}; '
                               'use one instance exposing both allocated GPUs with matching indices')


def prepare(args, runner):
    batch = runner.batch
    path = args.suite / 'manifest.json'
    contract = dict(profile=PROFILE, tasks=list(watch.TASKS), model_family='tactile_only',
        camera='head_only', episodes=50, step=10000, use_ema=True,
        training_simulator='4.5', evaluation_simulator='5.1',
        evaluation_type='cross-version-transfer', comparable_to_isaac45=False,
        input_alignment=INPUT_PROFILE, univtac_commit=UNIVTAC_COMMIT,
        worker_routing=WORKER_ROUTING, reset_timeout_seconds=RESET_TIMEOUT_SECONDS,
        contact_reset_patch=CONTACT_RESET_PATCH,
        environment_errors='isolate_task_without_formal_result')
    if path.exists():
        manifest = batch.read(path)
        if any(manifest.get(k) != v for k, v in contract.items()):
            raise ValueError('Suite has another evaluation contract')
    else:
        jobs = [watch.discover_task(args.root / 'twam/runs' / watch.RUN / 'full', task, runner)
                for task in watch.TASKS]
        if any(job is None for job in jobs):
            raise RuntimeError('All eight committed tactile 10k checkpoints are required')
        manifest = dict(contract, jobs=jobs, created_at=watch.now(), status='preparing')
        batch.save_summary(args.suite, manifest)
    for job in manifest['jobs']:
        print(f'Pinning checkpoint: {job["task"]}', flush=True)
        watch.snapshot(job, args.suite, runner)
        if job['state'] == 'running':
            # A cloud OOM can kill the scheduler without its finally block.
            # Never assume an unobserved remote worker has exited.
            raise RuntimeError('Suite has an unfinished worker; stop the previous job and use a new --suite')
    runtime, record = pin_runtime(args.root, args.suite, batch)
    manifest.update(runtime=str(runtime), runtime_fingerprint=record['fingerprint'],
                    status='prepared', max_per_gpu=args.per_gpu)
    batch.save_summary(args.suite, manifest)
    return manifest


def worker_env(args, manifest, work, gpu):
    env = dict(os.environ)
    for key in ('PYTHONHOME', 'PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'PI05_SIM_NATIVE_ROOT',
                'LD_LIBRARY_PATH', 'CONDA_PREFIX', 'CONDA_DEFAULT_ENV'):
        env.pop(key, None)
    prefix = args.env_home / 'sim'
    env.update(TWAM_ROOT=str(args.root), CBWAM_ROOT=manifest['runtime'],
        PRO6K_RUNTIME=manifest['runtime'], PRO6K_UNIVTAC=str(args.env_home / 'UniVTAC'),
        # USDRT SelectPrims supports only the process-local cuda:0. Isolate
        # each worker before CUDA is imported, including its policy children.
        CUDA_VISIBLE_DEVICES=str(gpu['index']), PRO6K_DEVICE='cuda:0',
        PRO6K_RENDER_GPU=str(gpu['index']), PRO6K_GPU_UUID=gpu['uuid'],
        PRO6K_RUNTIME_FINGERPRINT=manifest['runtime_fingerprint'],
        CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_HOME=str(prefix), CUDA_PATH=str(prefix),
        PI05_MODEL_LD_LIBRARY_PATH='', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=os.pathsep.join([manifest['runtime'] + '/support',
            manifest['runtime'] + '/src', manifest['runtime'] + '/inference',
            str(args.root / 'openpi/packages/openpi-client/src')]),
        PATH=str(prefix / 'bin') + os.pathsep + env.get('PATH', ''),
        LD_LIBRARY_PATH=os.pathsep.join([str(prefix / 'lib'),
            str(prefix / 'lib/python3.11/site-packages/torch/lib'),
            str(args.env_home / prepare_env.GRAPHICS_DIR / 'lib')]),
        OMNI_KIT_ACCEPT_EULA='YES', LIVESTREAM='0',
        OMNI_CONFIG_PATH=str(work / 'omniverse/config'),
        OMNI_WORKSPACE_ROOT=str(work / 'omniverse'),
        XDG_RUNTIME_DIR=str(work / 'xdg-runtime'), XDG_DATA_HOME=str(work / 'xdg-data'),
        XDG_CACHE_HOME=str(work / 'xdg-cache'), MPLCONFIGDIR=str(work / 'matplotlib'),
        TORCH_EXTENSIONS_DIR=str(work / 'torch-extensions'),
        CUDA_CACHE_PATH=str(work / 'cuda-cache'), TRITON_CACHE_DIR=str(work / 'triton-cache'),
        HF_HOME=str(args.root / '.cache/huggingface'), TORCH_HOME=str(args.root / '.cache/torch'),
        PI05_SERVER_LOG=str(work / 'policy-server.log'),
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        PXR_WORK_THREAD_LIMIT='4', TOKENIZERS_PARALLELISM='false',
        NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
    if Path('/etc/vulkan/icd.d/nvidia_icd.json').exists() and not env.get('VK_ICD_FILENAMES'):
        env['VK_ICD_FILENAMES'] = str(args.root / 'pi05-univtac/scripts/nvidia-egl-icd.json')
    for directory in ('config', 'data', 'cache', 'logs'):
        (work / 'omniverse' / directory).mkdir(parents=True, exist_ok=True)
    (work / 'xdg-runtime').mkdir(mode=0o700, exist_ok=True)
    (work / 'omniverse/config/omniverse.toml').write_text('[paths]\n' + '\n'.join(
        f'{key}_root = {json.dumps(str(work / "omniverse" / folder))}'
        for key, folder in [('data', 'data'), ('cache', 'cache'), ('logs', 'logs')]) + '\n')
    return env


def launch(job, args, manifest, batch, gpu, used_ports, *, validation=False):
    job['attempts'] += 1
    work = args.suite / 'work' / job['id'] / f'attempt{job["attempts"]:02d}'
    work.mkdir(parents=True)
    port = 18880
    while port in used_ports:
        port += 1
    port = batch.free_port(port)
    while port in used_ports:
        port = batch.free_port(port + 1)
    deployment = dict(task_name=job['task'], pi05_host='127.0.0.1', pi05_port=port,
        pi05_python=str(args.env_home / 'model/bin/python'), pi05_startup_timeout=1800,
        pi05_request_timeout=1800, pi05_eval_protocol=batch.EVAL_PROTOCOL)
    batch.write(work / 'deploy.json', deployment)
    env = worker_env(args, manifest, work, gpu)
    env['TACTILE_WAVE0_CHECKPOINT'] = job['checkpoint']
    command = [str(args.env_home / 'sim/bin/python'), '-u',
        str(Path(manifest['runtime']) / 'inference/run_pro6k_native.py'),
        '--task', job['task'], '--deploy', str(work / 'deploy.json'), '--output', str(work / 'native'),
        '--headless', '--livestream', '0', '--device', 'cuda:0', kit_arguments(gpu['index'])]
    if validation:
        command.insert(-1, '--validation-episode')
    log = work / 'eval.log'
    with log.open('w') as stream:
        child = subprocess.Popen(command, cwd=args.env_home / 'UniVTAC', env=env,
            stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    job.update(state='running', gpu_index=gpu['index'], gpu_uuid=gpu['uuid'], pid=child.pid,
               worker_host=socket.gethostname(), log=str(log), started_at=watch.now())
    print(f'{watch.now()} START {job["task"]} GPU {gpu["index"]} port {port}', flush=True)
    return dict(process=child, job=job, work=work, log=log, port=port, warm=False,
                started=time.monotonic(), progress='', last_progress=time.monotonic())


def finish(item, args, manifest, batch):
    batch.stop_group(item['process'])
    job, work = item['job'], item['work']
    text = item['log'].read_text(errors='replace')
    final = re.findall(r'Final Result: (\d+)/(\d+)', text)
    if item['process'].returncode != 0 or not final or int(final[-1][1]) != 50:
        errors = '\n'.join(line for line in text.splitlines()
                           if ('[error]' in line.lower() or 'Assertion' in line)
                           and '[carb.crashreporter' not in line)
        errors = '\n'.join(errors.splitlines()[:10])
        frames = '\n'.join(line for line in text.splitlines()
                            if '[Fatal]' in line and re.search(r'\b00[0-7]:', line))
        tail = '\n'.join(text.splitlines()[-20:])
        raise RuntimeError(f'{job["task"]} did not finish all 50 episodes '
                           f'(exit code {item["process"].returncode}); see {item["log"]}\n'
                           f'Worker errors before crash report:\n{errors or "(see log tail)"}\n'
                           f'Native crash frames:\n{frames or "(none)"}\n'
                           f'Worker log tail:\n{tail}')
    rows = validate_audits(batch.read(work / 'native/metadata.json'), job, manifest['runtime_fingerprint'])
    successes = sum(row['result'] == 'success' for row in rows.values())
    if successes != int(final[-1][0]):
        raise RuntimeError('Native success counts disagree with the log')
    destination = args.suite / 'results' / job['id']
    destination.parent.mkdir(exist_ok=True)
    result = dict(status='completed', task=job['task'], successes=successes, completed_episodes=50,
        success_rate=successes / 50, success_rate_including_exceptions=successes / 50,
        exceptions=sum(row['result'] == 'error' for row in rows.values()),
        evaluation_protocol=batch.EVAL_PROTOCOL, profile=PROFILE,
        contact_reset_patch=CONTACT_RESET_PATCH,
        training_simulator='4.5', evaluation_simulator='5.1', comparable_to_isaac45=False,
        expert_preflight_performed=False, gpu_uuid=job['gpu_uuid'], camera='head_only', step=10000,
        evaluated_checkpoint=job['checkpoint'], native_result_dir=str(destination),
        runtime_fingerprint=manifest['runtime_fingerprint'])
    batch.write(work / 'native/result.json', result)
    for name in ('eval.log', 'policy-server.log', 'deploy.json', 'gpu-binding.json'):
        shutil.move(str(work / name), str(work / 'native' / name))
    (work / 'native').rename(destination)
    shutil.rmtree(work)
    job.update(state='completed', result=result, log=str(destination / 'eval.log'))
    print(f'{watch.now()} FINISH {job["task"]}: {successes}/50', flush=True)


def publish(args, manifest, batch, status, **details):
    manifest.update(status=status, updated_at=watch.now(), **details)
    batch.save_summary(args.suite, manifest)
    batch.write(args.suite / 'daemon.json', dict(status=status, updated_at=manifest['updated_at'],
        host=socket.gethostname(), pid=os.getpid(), **details))


def check_worker_errors(text, task, log):
    # This native error can leave the worker alive with no Python traceback.
    # Do not pay for a hung worker until the one-hour episode watchdog fires.
    for line in text.splitlines():
        if ('GPUs other than cuda:0 are not currently supported' in line
                or 'Init Residual is nan' in line or 'Init Residual is inf' in line):
            raise RuntimeError(f'{task}: {line.strip()}; see {log}')


def fail_worker(item, error, batch):
    """Preserve diagnostics and isolate one failed process group from its peers."""
    batch.stop_group(item['process'])
    job = item['job']
    failure = dict(status='failed', task=job['task'], error=f'{type(error).__name__}: {error}',
        exit_code=item['process'].returncode, log=str(item['log']), finished_at=watch.now(),
        formal_result=False)
    metadata_path = item['work'] / 'native/metadata.json'
    if metadata_path.is_file():
        try:
            rows = batch.read(metadata_path)
            failure['completed_seeds'] = sorted(int(k) for k, row in rows.items()
                if k.isdigit() and 1000000 <= int(k) <= 1000049
                and row.get('result') in ('success', 'failed'))
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            failure['metadata_error'] = str(exc)
    batch.write(item['work'] / 'failure.json', failure)
    job.update(state='failed', failure=failure)
    print(f'{watch.now()} FAILED {job["task"]}: {error}\n'
          'Other tasks continue; no formal success rate produced for this task.', flush=True)


def poll_workers(active, args, manifest, batch):
    """A task failure must not cancel independent workers or pending tasks."""
    for identifier, item in list(active.items()):
        try:
            content = item['log'].read_text(errors='replace')
            check_worker_errors(content, item['job']['task'], item['log'])
            progress = re.findall(r'Total \d+/\d+\([^\n]*', content)
            if progress and progress[-1] != item['progress']:
                audited = any((item['work'] / 'native/input-audit').glob('*.json'))
                item.update(progress=progress[-1], warm=audited, last_progress=time.monotonic())
                item['job']['progress'] = progress[-1]
                print(f'{watch.now()} {item["job"]["task"]}: {progress[-1]}', flush=True)
            if item['process'].poll() is not None:
                finish(item, args, manifest, batch)
                del active[identifier]
            elif time.monotonic() - item['last_progress'] > 3600:
                raise TimeoutError(f'{identifier} has made no episode progress for one hour; logs retained')
        except Exception as exc:
            fail_worker(item, exc, batch)
            del active[identifier]


def pending_jobs(manifest, active):
    return [j for j in manifest['jobs']
            if j['state'] not in ('completed', 'failed') and j['id'] not in active]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=watch.ROOT)
    parser.add_argument('--suite', type=Path)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--per-gpu', type=int, default=4)
    parser.add_argument('--dedicated-task-per-gpu', action='store_true',
                        help='Assign each of the eight tasks to its own GPU')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--check-hardware', action='store_true')
    args = parser.parse_args(argv)
    indices = args.gpus.split(',')
    if len(indices) != len(set(indices)) or not all(s.isdigit() for s in indices) or not 1 <= args.per_gpu <= 4:
        parser.error('Use unique GPU indices and --per-gpu from 1 to 4')
    if args.dedicated_task_per_gpu and (len(indices) != len(watch.TASKS) or args.per_gpu != 1):
        parser.error('Dedicated task mode requires eight GPU indices and --per-gpu 1')
    if args.check_hardware:
        gpus, host = [gpu_info(i) for i in indices], resources()
        validate_allocation(gpus)
        limit = capacity(host, len(gpus), args.per_gpu)
        if args.dedicated_task_per_gpu:
            validate_dedicated_resources(gpus, host, args.per_gpu)
        if limit < 1:
            raise RuntimeError(f'Need at least 64 GiB container RAM and 8 CPU cores: {host}')
        print(json.dumps(dict(gpus=gpus, resources=host, admitted_workers=limit), indent=2), flush=True)
        return 0
    args.root = args.root.resolve()
    args.env_home = args.root / '.cache' / prepare_env.ENV_NAME
    args.suite = (args.suite or args.root / 'outputs/eval_suites' / SUITE).resolve()
    args.suite.mkdir(parents=True, exist_ok=True)
    runner = watch.load_runner(args.root)
    batch = runner.batch
    with (args.suite / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare_env.require_contact_reset(prepare_env.verify_ready(args.env_home))
        manifest = prepare(args, runner)
        assignment = None
        if args.dedicated_task_per_gpu:
            assignment = assign_dedicated_tasks(manifest, indices)
            batch.save_summary(args.suite, manifest)
        elif manifest.get('dedicated_task_gpus') is not None:
            raise RuntimeError('This suite has dedicated GPU assignments; retain --dedicated-task-per-gpu')
        if args.prepare_only:
            print('All eight tactile checkpoints/runtime prepared; no model or simulator started.', flush=True)
            return 0
        if all(j['state'] == 'completed' for j in manifest['jobs']):
            publish(args, manifest, batch, 'completed', active_tasks=[])
            return 0
        ready = batch.read(args.env_home / 'ready.json')
        if ready['profile'] != PROFILE:
            raise RuntimeError('Prepare the Isaac 5.1 Blackwell environment first')
        gpus = [gpu_info(index) for index in indices]
        validate_allocation(gpus)
        host = resources()
        limit = capacity(host, len(gpus), args.per_gpu)
        if assignment is not None:
            validate_dedicated_resources(gpus, host, args.per_gpu)
            print('Fixed task assignment: ' + ', '.join(
                f'GPU {index}={task}' for index, task in assignment.items()), flush=True)
        if limit < 1:
            raise RuntimeError(f'Need at least 64 GiB container RAM and 8 CPU cores: {host}')
        print(f'Resource admission: at most {limit} workers, {args.per_gpu}/GPU; RAM {host["total_bytes"]/GIB:.1f} GiB, CPU {host["cpus"]}', flush=True)
        gpu_locks, active = [], {}
        previous_handlers = {}
        def stop(signum, frame):
            raise KeyboardInterrupt(f'Signal {signum}')
        try:
            for gpu in gpus:
                gpu_lock = (args.suite.parent / f'.{gpu["uuid"]}.lock').open('a')
                gpu_locks.append(gpu_lock)
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                existing = subprocess.check_output(['nvidia-smi', '-i', str(gpu['index']),
                    '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
                if existing:
                    raise RuntimeError(f'GPU {gpu["index"]} already has compute jobs: {existing}')
            manifest.update(gpus=gpus, resource_limit=limit, resources_at_start=host)
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[sig] = signal.signal(sig, stop)
            while True:
                poll_workers(active, args, manifest, batch)
                pending = pending_jobs(manifest, active)
                if not pending and not active:
                    failed = [j['task'] for j in manifest['jobs'] if j['state'] == 'failed']
                    publish(args, manifest, batch, 'completed_with_errors' if failed else 'completed',
                            active_tasks=[], failed_tasks=failed)
                    return 1 if failed else 0
                current_host = resources()
                for gpu in sorted(gpus, key=lambda g: sum(a['job']['gpu_uuid'] == g['uuid'] for a in active.values())):
                    fresh = gpu_info(gpu['index'])
                    if fresh['uuid'] != gpu['uuid']:
                        raise RuntimeError('GPU identity changed during evaluation')
                    job = select_pending_job(pending, fresh, assignment)
                    if job is not None and can_start(active, fresh, current_host, limit=limit, per_gpu=args.per_gpu):
                        pending.remove(job)
                        item = launch(job, args, manifest, batch, gpu,
                                      {a['port'] for a in active.values()})
                        active[item['job']['id']] = item
                        # Reserve loading RAM immediately, before the OS sees
                        # the new model process's allocations.
                        current_host['available_bytes'] -= 48 * GIB
                publish(args, manifest, batch, 'evaluating' if active else 'waiting_for_memory',
                    active_tasks=[a['job']['task'] for a in active.values()], resource_limit=limit)
                time.sleep(15)
        except BaseException as exc:
            for item in active.values():
                batch.stop_group(item['process'])
                item['job']['state'] = 'interrupted_or_error'
            publish(args, manifest, batch, 'interrupted_or_error', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            for gpu_lock in gpu_locks:
                gpu_lock.close()
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
