"""CPU-only regression checks; never launch a model, GPU job, or simulator."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / 'inference'), str(PROJECT / 'scripts')]
import pair_contract as contract
import pair_servers
import pair_render
import reeval_lift_can_a800 as queue


class Settings(dict):
    def set(self, key, value):
        self[key] = value


def test_real_simulation_app_reset_keeps_fixed_defaults_without_starting_kit(monkeypatch):
    source = PROJECT.parent / '.conda/univtac-sim/lib/python3.10/site-packages/isaacsim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py'
    if not source.is_file():
        pytest.skip('Installed Isaac Sim 4.5 source is unavailable')
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'SimulationApp')
    nodes = [n for n in cls.body if
             (isinstance(n, ast.FunctionDef) and n.name in ('_set_render_settings', 'reset_render_settings')) or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DEFAULT_LAUNCHER_CONFIG' for t in n.targets))]
    # Execute only the installed settings methods, with an in-memory carb
    # replacement; no SimulationApp constructor, Kit imports or GPU calls.
    tree = ast.Module(body=[ast.ClassDef(name='SettingsApp', bases=[], keywords=[], body=nodes, decorator_list=[])], type_ignores=[])
    monkeypatch.setitem(sys.modules, '_a800_cpu_test.utils', SimpleNamespace(
        set_carb_setting=lambda settings, key, value: settings.set(key, value)))
    scope = {'__package__': '_a800_cpu_test', '__name__': '_a800_cpu_test.app'}
    exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), scope)
    app = scope['SettingsApp']()
    app.config = dict(app.DEFAULT_LAUNCHER_CONFIG)
    app._carb_settings = Settings()
    app.reset_render_settings()
    app._carb_settings.update(contract.RENDER_SETTINGS)
    app.reset_render_settings()
    assert app._carb_settings['/rtx/post/aa/op'] == 3  # Reproduces the old failure.
    app.config.update(contract.SIMULATION_APP_SETTINGS)
    contract.apply_render_settings(app._carb_settings)
    for _ in range(3):
        app.reset_render_settings()
        assert contract.validate_render(app._carb_settings) == contract.RENDER_SETTINGS
        assert app._carb_settings['/rtx-defaults/post/aa/op'] == 0


def test_render_settings_are_restored_before_pixels_and_mid_render_drift_is_rejected():
    settings = Settings(contract.persistent_render_settings())
    rendered = []
    class Simulator:
        def render(self, *, drift=False):
            rendered.append(dict(settings))
            if drift:
                settings['/rtx/post/aa/op'] = 3
            return 'frame'
    pair_render.install_render_guard(Simulator, settings)
    guard = Simulator.render
    pair_render.install_render_guard(Simulator, settings)
    assert Simulator.render is guard
    settings['/rtx/post/aa/op'] = 3
    settings['/rtx-defaults/post/aa/op'] = 3
    sim = Simulator()
    assert sim.render() == 'frame'
    assert rendered[-1]['/rtx/post/aa/op'] == 0
    assert rendered[-1]['/rtx-defaults/post/aa/op'] == 0
    with pytest.raises(RuntimeError, match='renderer settings changed'):
        sim.render(drift=True)


def test_frozen_evaluator_configures_launch_and_task_before_creation():
    original = (PROJECT.parent / 'UniVTAC/scripts/eval_policy.py').read_text()
    frozen = queue.freeze_native_evaluator(original)
    assert frozen.index('vars(args_cli).update') < frozen.index('app_launcher = AppLauncher')
    assert frozen.index('initialize_rendering()') < frozen.index('task_module = importlib.import_module')
    assert frozen.index('configure_task_render(env_cfg)') < frozen.index('task:BaseTask = task_module.Task')
    with pytest.raises(RuntimeError, match='template changed'):
        queue.freeze_native_evaluator(original.replace('AppLauncher(args_cli)', 'AppLauncher(other_args)'))
    cfg = SimpleNamespace(sim=SimpleNamespace(render=SimpleNamespace()))
    pair_render.configure_task_render(cfg)
    assert cfg.sim.render.antialiasing_mode is None
    assert cfg.sim.render.rendering_mode is None
    assert cfg.sim.render.carb_settings == contract.RENDER_SETTINGS


def test_tactile_only_selection_never_requires_base_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'weights'
    checkpoint.mkdir()
    for name in ('model.safetensors', 'ema.safetensors', 'norm_stats.json', 'wm_norm_stats.json', 'tokenizer.model'):
        (checkpoint / name).write_text('fixture')
    job = dict(checkpoint=str(checkpoint), task='lift_can', state='pending')
    monkeypatch.setattr(queue.watch, 'discover_task', lambda *args: job)
    monkeypatch.setattr(queue.watch, 'snapshot', lambda *args: None)
    monkeypatch.setattr(queue, 'pin_runtime', lambda *args: (tmp_path / 'runtime', dict(fingerprint='frozen')))
    args = SimpleNamespace(root=tmp_path, suite=tmp_path / 'suite', family='tactile')
    batch = SimpleNamespace(EVAL_PROTOCOL='test', save_summary=lambda *args: None)
    base = SimpleNamespace(discover=lambda *args: pytest.fail('Base must not be discovered'))
    manifest = queue.prepare(args, SimpleNamespace(batch=batch), base)
    assert manifest['families'] == ['tactile']
    assert manifest['jobs'] == [job]
    assert job['comparison_family'] == 'tactile'
    queue.comparison(args.suite, manifest, batch)


def test_renderer_no_dlss_and_detects_runtime_override():
    assert contract.RENDER_SETTINGS['/rtx/rendermode'] == 'PathTracing'
    assert contract.RENDER_SETTINGS['/rtx/pathtracing/optixDenoiser/enabled'] is True
    assert contract.RENDER_SETTINGS['/rtx/pathtracing/optixDenoiser/AOV'] is False
    assert contract.validate_render(contract.RENDER_SETTINGS) == contract.RENDER_SETTINGS
    changed = dict(contract.RENDER_SETTINGS, **{'/rtx/rendermode': 'RaytracedLighting'})
    with pytest.raises(RuntimeError, match='renderer settings changed'):
        contract.validate_render(changed)
    assert '--/rtx/pathtracing/optixDenoiser/enabled=true' in contract.kit_arguments()


def test_sampling_seed_is_tied_to_scene_and_replan_not_previous_outcome(monkeypatch):
    import torch
    calls = []
    monkeypatch.setattr(torch, 'manual_seed', calls.append)
    pair_servers.seed_sampler(1000003, 0)
    pair_servers.seed_sampler(1000001, 20)
    pair_servers.seed_sampler(1000003, 0)
    assert calls[0] == calls[2] != calls[1]
    for scene, step in [(0, 0), (1000000, 1), (1000000, -10)]:
        with pytest.raises(ValueError):
            contract.inference_seed(scene, step)


@pytest.fixture
def batch():
    return queue.watch.load_runner(PROJECT.parent).batch


def test_both_launches_use_identical_render_seeds_budget_and_separate_checkpoints(tmp_path, monkeypatch, batch):
    commands = []
    monkeypatch.setattr(queue, 'verify_runtime', lambda *args: {})
    monkeypatch.setattr(batch, 'free_port', lambda port: port)
    def popen(command, **kwargs):
        commands.append((command, kwargs['env']))
        return SimpleNamespace(pid=55)
    monkeypatch.setattr(queue.subprocess, 'Popen', popen)
    args = SimpleNamespace(root=tmp_path, suite=tmp_path / 'suite', gpu='0')
    manifest = dict(runtime=str(tmp_path / 'runtime'), runtime_fingerprint='frozen')
    for family in ['base', 'tactile']:
        job = dict(id=family, task='lift_can', comparison_family=family, checkpoint=f'/weights/{family}', attempts=0)
        item = queue.launch(job, args, manifest, batch, dict(uuid='GPU-test'))
        assert json.loads((item['work'] / 'deploy.yml').read_text())['policy_name'] == 'A800Compare'
        assert job['state'] == 'running'
    for command, env in commands:
        assert command[command.index('--total_num') + 1] == '50'
        assert command[command.index('--start_seed') + 1] == '1000000'
        assert command[-1] == contract.kit_arguments()
        assert env['PAIR_GPU_UUID'] == 'GPU-test'
        assert 'PI05_TRACE_DIR' not in env
    assert commands[0][1]['WAVE0_CHECKPOINT'] == '/weights/base'
    assert commands[1][1]['TACTILE_WAVE0_CHECKPOINT'] == '/weights/tactile'


def rows(family, gpu='GPU-test'):
    return {str(s): dict(result='success', can_size=5, grasp_noise=[0, 0, 0],
        policy_execution=dict(comparison=dict(profile=contract.PROFILE, family=family, scene_seed=s,
            gpu_uuid=gpu, render_settings=contract.RENDER_SETTINGS, rng_profile=contract.RNG_PROFILE,
            runtime_fingerprint='frozen', input_alignment=contract.INPUT_PROFILE if family == 'tactile' else 'base-head-only')))
        for s in range(1000000, 1000050)}


def test_old_adapter_other_gpu_or_incomplete_seeds_cannot_be_published():
    manifest = dict(gpu=dict(uuid='GPU-test'), runtime_fingerprint='frozen')
    data = rows('tactile')
    assert len(queue.validate_episode_audits(data, 'tactile', manifest)) == 50
    data['1000000']['policy_execution']['comparison']['input_alignment'] = 'raw-target-history'
    with pytest.raises(ValueError, match='Unmatched'):
        queue.validate_episode_audits(data, 'tactile', manifest)
    with pytest.raises(ValueError, match='Unmatched'):
        queue.validate_episode_audits(rows('base', 'different-gpu'), 'base', manifest)
    data = rows('base'); del data['1000001']
    with pytest.raises(ValueError, match='Missing/extra'):
        queue.validate_episode_audits(data, 'base', manifest)


def test_comparison_uses_same_fifty_seeds_and_counts_errors_as_failures(tmp_path, batch):
    jobs = []
    for family in ['base', 'tactile']:
        directory = tmp_path / family; directory.mkdir()
        data = rows(family)
        if family == 'tactile':
            data['1000001'] = dict(result='error')
            data['1000002']['result'] = 'failed'
        batch.write(directory / 'metadata.json', data)
        jobs.append(dict(state='completed', comparison_family=family, result=dict(native_result_dir=str(directory))))
    manifest = dict(jobs=jobs, gpu=dict(uuid='GPU-test'), runtime_fingerprint='frozen')
    queue.comparison(tmp_path, manifest, batch)
    result = batch.read(tmp_path / 'comparison.json')
    assert result['successes'] == dict(base=50, tactile=48)
    assert result['change_percentage_points'] == -4
    assert result['lost_seeds'] == [1000001, 1000002]
    data = rows('base'); data['1000003']['can_size'] = 6
    batch.write(tmp_path / 'base/metadata.json', data)
    with pytest.raises(ValueError, match='Initial scene'):
        queue.comparison(tmp_path, manifest, batch)
