"""CPU-only checks for transfer scope, resources, GPU routing and result integrity."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / 'scripts'), str(PROJECT / 'inference')]
import reeval_pro6k as queue
import prepare_pro6k as setup
from pro6k_contract import (PROFILE, INPUT_PROFILE, RENDER_SETTINGS, RESET_TIMEOUT_SECONDS,
                           configure_reset, stop_on_environment_error, validate_audits, validate_timing)


def test_runtime_check_never_installs_when_environment_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prepare_pro6k.py', '--root', str(tmp_path), '--check-only'])
    monkeypatch.setattr(setup, 'install', lambda *args: pytest.fail('Must not install in check-only mode'))
    monkeypatch.setattr(setup, 'prepare_source', lambda *args, **kwargs: pytest.fail('Must not prepare sources'))
    with pytest.raises(RuntimeError, match='development machine first'):
        setup.main()
    assert list(tmp_path.iterdir()) == []


def test_ready_check_rejects_changed_native_library(tmp_path):
    source = tmp_path / 'UniVTAC'
    source.mkdir()
    native, executable = (tmp_path / n for n in ('native.so', 'python'))
    graphics = [tmp_path / setup.GRAPHICS_DIR / 'lib' / name for name in setup.GRAPHICS_SONAMES]
    graphics[0].parent.mkdir(parents=True)
    for p in (native, *graphics, executable):
        p.write_bytes(b'fixture')
    setup.write(tmp_path / 'source.json', dict(commit=setup.UNIVTAC_COMMIT, files={}, missing_lfs=[]))
    setup.write(tmp_path / 'ready.json', dict(profile=PROFILE, univtac_commit=setup.UNIVTAC_COMMIT,
        libraries={str(native): setup.digest(native)}, graphics_libraries={str(p): setup.digest(p) for p in graphics},
        simulator_python=str(executable), model_python=str(executable), source=str(source)))
    setup.verify_ready(tmp_path)
    native.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='library is missing or changed'):
        setup.verify_ready(tmp_path)


def test_partial_graphics_manifest_is_not_ready(tmp_path):
    setup.write(tmp_path / 'ready.json', dict(profile=PROFILE, univtac_commit=setup.UNIVTAC_COMMIT,
        libraries={'native.so': 'old'}, graphics_libraries={'libGLdispatch.so.0': 'old'}))
    with pytest.raises(RuntimeError, match='missing the matching GLVND stack'):
        setup.verify_ready(tmp_path)


@pytest.mark.parametrize('mode', [[], ['--repair-contact-reset']])
def test_installer_refuses_to_build_inside_aihc_job(tmp_path, monkeypatch, mode):
    monkeypatch.setenv('AIHC_JOB_NAME', 'job-pro6k')
    monkeypatch.setattr(sys, 'argv', ['prepare_pro6k.py', '--root', str(tmp_path), *mode])
    with pytest.raises(RuntimeError, match='development machine, not inside'):
        setup.main()
    assert list(tmp_path.iterdir()) == []


def test_formal_queue_rejects_unpatched_or_mismatched_contact_backend(tmp_path, monkeypatch):
    source = tmp_path / setup.CONTACT_SOURCE
    source.parent.mkdir(parents=True)
    source.write_text('verified contact update source')
    expected = setup.digest(source)
    monkeypatch.setattr(setup, 'CONTACT_PATCHED_SHA256', expected)
    record = dict(source=str(tmp_path), libraries={'backend.so': 'verified-binary'})
    with pytest.raises(RuntimeError, match='development machine first'):
        setup.require_contact_reset(record)
    record['contact_reset_patch'] = dict(id=setup.CONTACT_RESET_PATCH,
        source_sha256=expected, binary='backend.so', binary_sha256='old-binary')
    with pytest.raises(RuntimeError, match='development machine first'):
        setup.require_contact_reset(record)
    record['contact_reset_patch']['binary_sha256'] = 'verified-binary'
    setup.require_contact_reset(record)
    source.write_text('changed after compilation')
    with pytest.raises(RuntimeError, match='no longer matches'):
        setup.require_contact_reset(record)


def test_contact_repair_rejects_unknown_native_source_before_build(tmp_path, monkeypatch):
    source = tmp_path / 'UniVTAC' / setup.CONTACT_SOURCE
    source.parent.mkdir(parents=True)
    source.write_text('a different upstream implementation')
    monkeypatch.setattr(setup, 'verify_ready', lambda home: dict(source=str(tmp_path / 'UniVTAC')))
    monkeypatch.setattr(setup, 'run', lambda *args, **kwargs: pytest.fail('Must reject before building'))
    with pytest.raises(RuntimeError, match='Unknown UIPC source revision'):
        setup.repair_contact_reset(tmp_path, 2)
    assert source.read_text() == 'a different upstream implementation'


@pytest.mark.parametrize('name', ['NVIDIA RPBZZZ6', 'NVIDIA RTX PRO 6000 Blackwell Server Edition'])
def test_gpu_admission_accepts_baidu_alias_and_standard_name(monkeypatch, name):
    monkeypatch.setattr(queue.subprocess, 'check_output', lambda *args, **kwargs:
                        f'1, {name}, GPU-one, 97887, 97000, 580.95.05, 12.0\n')
    gpu = queue.gpu_info(1)
    assert gpu['name'] == name
    assert gpu['index'] == 1
    assert gpu['uuid'] == 'GPU-one'
    assert gpu['total_mib'] == 97887


@pytest.mark.parametrize('name,capability,memory,driver,error', [
    ('NVIDIA RPBZZZ6', '8.0', 97887, '580.95.05', 'Expected RTX PRO'),
    ('NVIDIA RPBZZZ6', '12.0', 49152, '580.95.05', '96 GB-class'),
    ('NVIDIA RPBZZZ6', '12.0', 97887, '575.64.03', 'R580-or-newer'),
    ('NVIDIA unknown GPU', '12.0', 97887, '580.95.05', 'Expected RTX PRO'),
])
def test_baidu_alias_does_not_bypass_hardware_checks(monkeypatch, name, capability, memory, driver, error):
    monkeypatch.setattr(queue.subprocess, 'check_output', lambda *args, **kwargs:
                        f'0, {name}, GPU-one, {memory}, 47000, {driver}, {capability}\n')
    with pytest.raises(RuntimeError, match=error):
        queue.gpu_info(0)


def test_container_memory_and_cpu_limits_override_large_host(tmp_path, monkeypatch):
    proc, cg = tmp_path / 'proc', tmp_path / 'cgroup'
    proc.mkdir(); cg.mkdir()
    (proc / 'meminfo').write_text('MemTotal: 1073741824 kB\nMemAvailable: 900000000 kB\n')
    (cg / 'memory.max').write_text(str(64 * queue.GIB))
    (cg / 'memory.current').write_text(str(20 * queue.GIB))
    (cg / 'cpu.max').write_text('400000 100000')
    monkeypatch.setattr(queue.os, 'sched_getaffinity', lambda pid: set(range(128)))
    result = queue.resources(proc, cg)
    assert result == dict(total_bytes=64 * queue.GIB, available_bytes=44 * queue.GIB, cpus=4)
    assert queue.capacity(result, 2, 4) == 0


def test_eight_workers_require_ram_and_cpu_as_well_as_two_gpus():
    assert queue.capacity(dict(total_bytes=512 * queue.GIB, cpus=64), 2, 4) == 8
    assert queue.capacity(dict(total_bytes=256 * queue.GIB, cpus=64), 2, 4) == 5
    assert queue.capacity(dict(total_bytes=512 * queue.GIB, cpus=16), 2, 4) == 2


@pytest.mark.parametrize('problem', ['gpu_count', 'cpu', 'total_ram', 'available_ram', 'gpu_memory'])
def test_dedicated_eight_gpu_preflight_requires_capacity_for_all_tasks(problem):
    gpus = [dict(index=i, free_mib=97000) for i in range(8)]
    host = dict(total_bytes=512 * queue.GIB, available_bytes=512 * queue.GIB, cpus=64)
    queue.validate_dedicated_resources(gpus, host, per_gpu=1)
    if problem == 'gpu_count':
        gpus.pop()
    elif problem == 'cpu':
        host['cpus'] = 63
    elif problem == 'total_ram':
        host['total_bytes'] = 399 * queue.GIB
    elif problem == 'available_ram':
        host['available_bytes'] = 383 * queue.GIB
    else:
        gpus[3]['free_mib'] = 25 * 1024
    with pytest.raises(RuntimeError):
        queue.validate_dedicated_resources(gpus, host, per_gpu=1)


@pytest.mark.parametrize('fail_one', [False, True])
def test_dedicated_queue_runs_one_fixed_task_on_each_gpu(tmp_path, monkeypatch, fail_one):
    jobs = [dict(id=task, task=task, state='pending') for task in queue.watch.TASKS]
    manifest = dict(jobs=jobs)
    batch = SimpleNamespace(read=lambda path: dict(profile=queue.PROFILE),
                            save_summary=lambda suite, value: None)
    host = dict(total_bytes=512 * queue.GIB, available_bytes=512 * queue.GIB, cpus=64)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(queue.watch, 'load_runner', lambda root: SimpleNamespace(batch=batch))
    monkeypatch.setattr(queue.prepare_env, 'verify_ready', lambda home: {})
    monkeypatch.setattr(queue.prepare_env, 'require_contact_reset', lambda ready: None)
    monkeypatch.setattr(queue, 'prepare', lambda args, runner: manifest)
    monkeypatch.setattr(queue, 'resources', lambda: dict(host))
    monkeypatch.setattr(queue, 'gpu_info', lambda index: dict(index=int(index), uuid=f'GPU-{index}', free_mib=97000))
    monkeypatch.setattr(queue.subprocess, 'check_output', lambda *args, **kwargs: '')
    monkeypatch.setattr(queue.time, 'sleep', lambda seconds: None)
    # Reordering pending jobs must not change their assigned GPUs.
    original_pending = queue.pending_jobs
    monkeypatch.setattr(queue, 'pending_jobs', lambda current, active: list(reversed(original_pending(current, active))))
    launched, statuses = [], []
    def launch(job, args, current, batch, gpu, used_ports):
        launched.append((job['task'], gpu['index']))
        job.update(state='running', gpu_uuid=gpu['uuid'])
        return dict(job=job, port=18880 + gpu['index'], warm=False)
    def poll(active, args, current, batch):
        for item in active.values():
            item['job']['state'] = ('failed' if fail_one and item['job']['task'] == queue.watch.TASKS[2]
                                    else 'completed')
        active.clear()
    monkeypatch.setattr(queue, 'launch', launch)
    monkeypatch.setattr(queue, 'poll_workers', poll)
    monkeypatch.setattr(queue, 'publish', lambda args, current, batch, status, **kw: statuses.append((status, kw)))
    result = queue.main(['--root', str(tmp_path), '--gpus', '0,1,2,3,4,5,6,7',
                         '--per-gpu', '1', '--dedicated-task-per-gpu'])
    assert launched == list(zip(queue.watch.TASKS, range(8)))
    assert manifest['dedicated_task_gpus'] == {str(i): task for i, task in enumerate(queue.watch.TASKS)}
    assert result == (1 if fail_one else 0)
    assert statuses[-1][0] == ('completed_with_errors' if fail_one else 'completed')
    assert statuses[-1][1]['failed_tasks'] == ([queue.watch.TASKS[2]] if fail_one else [])


def test_eight_gpu_submission_uses_same_allocation_for_preflight_and_eval(tmp_path, monkeypatch):
    python = tmp_path / 'contact-belief-wam/.venv/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$PRO6K_TEST_LOG"\n')
    python.chmod(0o755)
    log = tmp_path / 'calls.log'
    monkeypatch.setenv('TWAM_ROOT', str(tmp_path))
    monkeypatch.setenv('PRO6K_TEST_LOG', str(log))
    script = PROJECT / 'scripts/baidu-eval-head-only-pro6k-8gpu.sh'
    subprocess.run(['bash', str(script), '--suite', str(tmp_path / 'suite')], check=True)
    calls = log.read_text().splitlines()
    assert len(calls) == 3
    assert '--check-only' in calls[0]
    assert '--check-hardware' in calls[1]
    for call in calls[1:]:
        assert '--gpus 0,1,2,3,4,5,6,7' in call
        assert '--per-gpu 1 --dedicated-task-per-gpu' in call
        assert f'--suite {tmp_path / "suite"}' in call


def test_admission_waits_for_real_first_episode_and_memory():
    gpu = dict(uuid='GPU-one', free_mib=96000)
    host = dict(available_bytes=256 * queue.GIB)
    active = {'a': dict(job=dict(gpu_uuid='GPU-one'), warm=False)}
    assert not queue.can_start(active, gpu, host, limit=8, per_gpu=4)
    assert queue.can_start(active, dict(gpu, uuid='GPU-two'), host, limit=8, per_gpu=4)
    active['a']['warm'] = True
    assert queue.can_start(active, gpu, host, limit=8, per_gpu=4)
    assert not queue.can_start(active, dict(gpu, free_mib=25000), host, limit=8, per_gpu=4)
    assert not queue.can_start(active, gpu, dict(available_bytes=47 * queue.GIB), limit=8, per_gpu=4)


def test_physics_decimation_cannot_silently_double_action_duration():
    task = SimpleNamespace(cfg=SimpleNamespace(decimation=1, sim=SimpleNamespace(dt=1 / 120)))
    validate_timing(task)
    task.cfg.decimation = 2
    with pytest.raises(RuntimeError, match='decimation=1'):
        validate_timing(task)


def test_second_gpu_worker_exposes_only_its_gpu_as_logical_zero(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('PI05_SIM_NATIVE_ROOT', '/old/a800')
    args = SimpleNamespace(root=PROJECT.parent, env_home=tmp_path / 'env')
    manifest = dict(runtime=str(tmp_path / 'runtime'), runtime_fingerprint='frozen')
    env = queue.worker_env(args, manifest, tmp_path / 'work', dict(index=1, uuid='GPU-one'))
    assert env['CUDA_VISIBLE_DEVICES'] == '1'
    assert 'PI05_SIM_NATIVE_ROOT' not in env
    assert env['PRO6K_DEVICE'] == 'cuda:0'
    assert env['PRO6K_RENDER_GPU'] == '1'
    assert env['PRO6K_GPU_UUID'] == 'GPU-one'
    assert env['PI05_MODEL_LD_LIBRARY_PATH'] == ''
    assert 'work' in env['TORCH_EXTENSIONS_DIR']


def test_cuda_routing_never_escapes_an_existing_gpu_allocation():
    gpus = [dict(index=0, uuid='GPU-zero'), dict(index=1, uuid='GPU-one')]
    queue.validate_allocation(gpus, '0,1')
    queue.validate_allocation(gpus, 'GPU-zero,GPU-one')
    with pytest.raises(RuntimeError, match='outside CUDA_VISIBLE_DEVICES'):
        queue.validate_allocation(gpus, '0')


def test_launch_is_tactile_only_and_ports_do_not_collide(tmp_path, monkeypatch):
    batch = queue.watch.load_runner(PROJECT.parent).batch
    monkeypatch.setattr(batch, 'free_port', lambda port: port)
    commands = []
    def popen(command, **kwargs):
        commands.append((command, kwargs['env']))
        return SimpleNamespace(pid=42)
    monkeypatch.setattr(queue.subprocess, 'Popen', popen)
    args = SimpleNamespace(root=PROJECT.parent, env_home=tmp_path / 'env', suite=tmp_path / 'suite')
    manifest = dict(runtime=str(tmp_path / 'runtime'), runtime_fingerprint='frozen')
    used = set()
    for i, task in enumerate(queue.watch.TASKS):
        job = dict(task=task, id=task, checkpoint='/tactile/weights', attempts=0)
        item = queue.launch(job, args, manifest, batch, dict(index=i % 2, uuid=f'GPU-{i % 2}'), used)
        assert item['port'] not in used
        used.add(item['port'])
    assert len(commands) == 8
    for i, (command, env) in enumerate(commands):
        assert command[command.index('--device') + 1] == 'cuda:0'
        assert env['CUDA_VISIBLE_DEVICES'] == str(i % 2)
        assert env['PRO6K_DEVICE'] == 'cuda:0'
        assert env['PRO6K_RENDER_GPU'] == str(i % 2)
        assert f'--/renderer/activeGpu={i % 2}' in command[-1]
        assert '--/physics/cudaDevice=0' in command[-1]
        assert '--/renderer/multiGpu/enabled=false' in command[-1]
        assert env['TACTILE_WAVE0_CHECKPOINT'] == '/tactile/weights'
        assert 'WAVE0_CHECKPOINT' not in env


def test_transfer_audit_rejects_old_environment_and_missing_seeds():
    job = dict(task='lift_can', gpu_uuid='GPU-one')
    rows = {str(s): dict(result='success', policy_execution=dict(transfer=dict(
        profile=PROFILE, input_alignment=INPUT_PROFILE, task='lift_can', scene_seed=s,
        contact_reset_patch=setup.CONTACT_RESET_PATCH,
        gpu_uuid='GPU-one', runtime_fingerprint='frozen', render_settings=RENDER_SETTINGS,
        training_simulator='4.5', evaluation_simulator='5.1'))) for s in range(1000000, 1000050)}
    assert len(validate_audits(rows, job, 'frozen')) == 50
    valid = rows['1000000']
    rows['1000000'] = dict(result='error')
    with pytest.raises(ValueError, match='formal result is invalid'):
        validate_audits(rows, job, 'frozen')
    rows['1000000'] = valid
    rows['1000001']['policy_execution']['transfer']['evaluation_simulator'] = '4.5'
    with pytest.raises(ValueError, match='Invalid transfer'):
        validate_audits(rows, job, 'frozen')
    del rows['1000002']
    with pytest.raises(ValueError, match='all 50'):
        validate_audits(rows, job, 'frozen')


def test_reset_wait_changes_wall_clock_only():
    cfg = SimpleNamespace(reset_time_limit=120.0, step_lim=300, decimation=1,
                          sim=SimpleNamespace(dt=1 / 120))
    configure_reset(cfg)
    assert cfg.reset_time_limit == RESET_TIMEOUT_SECONDS == 600.0
    assert (cfg.step_lim, cfg.decimation, cfg.sim.dt) == (300, 1, 1 / 120)


@pytest.mark.parametrize('environment_error', [True, False])
def test_native_loop_aborts_environment_errors_but_keeps_model_failures(tmp_path, environment_error):
    # Execute the upstream fixed-seed loop, not a second implementation of it.
    tree = ast.parse((PROJECT.parent / 'UniVTAC/scripts/eval_policy.py').read_text())
    function, = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'eval_policy']
    namespace = {}
    exec('import time, json, traceback\nfrom pathlib import Path\nfrom typing import Literal', namespace)
    logs, saved, visited = [], [], []
    namespace['log'] = logs.append
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<native-loop>', 'exec'), namespace)
    task = SimpleNamespace(save_root=tmp_path / 'native', metadata={}, cfg=SimpleNamespace(step_lim=1),
                           take_action_cnt=0, plan_success=True, step_count=0)
    def reset(seed, **kwargs):
        task.cfg.seed = seed
        task.metadata = {}
        visited.append(seed)
        if environment_error:
            raise RuntimeError('Timeout: reset exceed time limit')
    def clean_cache(mean_steps=0.0, result=None):
        saved.append(dict(task.metadata, result=result, seed=task.cfg.seed))
    class InvalidOutput(RuntimeError):
        policy_output_error = True
    def invalid_output(*args):
        raise InvalidOutput('invalid action')
    task.reset, task.clean_cache = reset, clean_cache
    task._get_observations = lambda: {}
    policy = SimpleNamespace(fixed_seed_evaluation=True, evaluation_protocol='fixed',
                             reset=lambda: None, eval=invalid_output)
    stop_on_environment_error(task)
    arguments = (task, policy, False, 1000000, 1000049, 50, dict(seen=['test']))
    if environment_error:
        with pytest.raises(RuntimeError, match='no formal success rate produced'):
            namespace['eval_policy'](*arguments)
        assert visited == [1000000]
        assert saved[0]['result'] == 'error'
        assert saved[0]['failure_kind'] == 'environment_or_runtime'
        assert not any('Total ' in line for line in logs)
    else:
        result = namespace['eval_policy'](*arguments)
        assert result == dict(succ_num=0, test_num=50)
        assert visited == list(range(1000000, 1000050))
        assert all(row['result'] == 'failed' for row in saved)


def test_scheduler_detects_usdrt_hang_without_waiting_for_worker_exit():
    with pytest.raises(RuntimeError, match='insert_HDMI.*GPUs other than cuda:0'):
        queue.check_worker_errors('[Error] GPUs other than cuda:0 are not currently supported',
                                  'insert_HDMI', '/worker/eval.log')


@pytest.mark.parametrize('kind', ['native_abort', 'nonzero_exit', 'stalled', 'invalid_result'])
def test_failed_worker_preserves_other_gpu_and_pending_tasks(tmp_path, monkeypatch, kind):
    batch = queue.watch.load_runner(PROJECT.parent).batch
    stopped = []
    monkeypatch.setattr(batch, 'stop_group', lambda process: stopped.append(process))
    now = queue.time.monotonic()

    def worker(task, gpu):
        work = tmp_path / task
        (work / 'native').mkdir(parents=True)
        log = work / 'eval.log'
        log.write_text('Setting seed: 1000002\n')
        process = SimpleNamespace(returncode=None)
        process.poll = lambda: process.returncode
        return dict(job=dict(id=task, task=task, state='running', gpu_index=gpu),
            process=process, work=work, log=log, progress='', warm=True, last_progress=now)

    broken, healthy = worker('pull_out_key', 1), worker('insert_hole', 0)
    batch.write(broken['work'] / 'native/metadata.json', {
        '1000000': dict(result='failed'), '1000001': dict(result='success'),
        '1000002': dict(result='error')})
    if kind == 'native_abort':
        broken['log'].write_text('[error] [cuda] Assertion failed. Init Residual is nan, norm(r) = -nan\n')
    elif kind == 'nonzero_exit':
        broken['process'].returncode = -6
    elif kind == 'stalled':
        broken['last_progress'] = now - 3601
    else:
        broken['process'].returncode = 0
        broken['log'].write_text('Final Result: 1/50\n')
        monkeypatch.setattr(queue, 'finish', lambda *args: (_ for _ in ()).throw(
            ValueError('Missing fixed seeds')))
    waiting = dict(id='lift_can', task='lift_can', state='pending')
    manifest = dict(jobs=[broken['job'], healthy['job'], waiting], runtime_fingerprint='frozen')
    active = dict(pull_out_key=broken, insert_hole=healthy)
    queue.poll_workers(active, SimpleNamespace(suite=tmp_path), manifest, batch)
    assert list(active) == ['insert_hole']
    assert healthy['job']['state'] == 'running'
    assert healthy['process'] not in stopped
    assert broken['job']['state'] == 'failed'
    failure = batch.read(broken['work'] / 'failure.json')
    assert failure['formal_result'] is False
    assert failure['completed_seeds'] == [1000000, 1000001]
    assert 'success_rate' not in failure
    assert queue.pending_jobs(manifest, active) == [waiting]


def test_user_interrupt_is_not_swallowed_as_a_worker_failure(tmp_path, monkeypatch):
    log = tmp_path / 'eval.log'
    log.write_text('')
    item = dict(log=log, job=dict(task='lift_can'))
    def interrupted(*args):
        raise KeyboardInterrupt('stop requested')
    monkeypatch.setattr(queue, 'check_worker_errors', interrupted)
    with pytest.raises(KeyboardInterrupt, match='stop requested'):
        queue.poll_workers({'lift_can': item}, None, {}, None)
