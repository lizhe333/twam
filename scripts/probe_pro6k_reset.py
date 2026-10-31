"""Replay saved actions to diagnose a reset without loading a policy model.

This is a development probe. It does not produce formal evaluation results.
"""
import argparse
import json
import os
from pathlib import Path
import sys

RUNTIME = Path(os.environ['PRO6K_RUNTIME'])
SOURCE = Path(os.environ['PRO6K_UNIVTAC'])
sys.path[:0] = [str(RUNTIME / 'src'), str(RUNTIME / 'inference'), str(SOURCE)]


def main():
    os.chdir(SOURCE)
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--seeds', default='1000000,1000001,1000002')
    parser.add_argument('--backend-dir', type=Path)
    parser.add_argument('--reset-only', action='store_true')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    from pro6k_gpu import configure_worker_gpu, worker_app_launcher
    configure_worker_gpu(args, torch.cuda, os.environ)
    app = worker_app_launcher(AppLauncher, os.environ)(args).app
    task = None
    events = []

    def record(event, **data):
        item = dict(event=event, **data)
        events.append(item)
        (args.output / 'probe.json').write_text(json.dumps(dict(
            formal_result=False, backend=str(args.backend_dir), events=events), indent=2) + '\n')
        print(json.dumps(item), flush=True)

    try:
        if args.backend_dir:
            import uipc
            config = uipc.default_config()
            config['module_dir'] = str(args.backend_dir.resolve())
            uipc.init(config)
        from envs.utils.env_parser import build_task_env_cfg, load_task_config
        from pro6k_contract import configure_reset, validate_timing, RENDER_SETTINGS
        import carb.settings
        config, path = load_task_config('clean', ['env_settings.frequencies.eval=120'])
        module, cfg, _, _ = build_task_env_cfg('pull_out_key', config, path.stem, 'eval',
                                              device=args.device, save_dir=args.output / 'native')
        cfg.seed = 0
        configure_reset(cfg)
        cfg.sim.enable_scene_query_support = True
        cfg.sim.render.rendering_mode = None
        cfg.sim.render.enable_dl_denoiser = True
        cfg.sim.render.enable_dlssg = False
        cfg.sim.render.dlss_mode = 2
        settings = carb.settings.get_settings()
        for key, value in RENDER_SETTINGS.items():
            settings.set(key, value)
        task = module.Task(cfg, mode='eval')
        validate_timing(task)
        original_step = task._step
        def traced_step(*args, **kwargs):
            if task.step_count == 0:
                record('first_physics_step', seed=task.cfg.seed, frame=task.uipc_sim.world.frame())
            return original_step(*args, **kwargs)
        task._step = traced_step
        reference = json.loads(args.reference.read_text())
        instructions = json.loads((SOURCE / 'instructions/pull_out_key.json').read_text())['seen']
        for seed in map(int, args.seeds.split(',')):
            record('reset_start', seed=seed)
            task.reset(seed=seed, instructions=instructions)
            if not task.plan_success:
                raise RuntimeError(f'pre_move failed at seed {seed}')
            record('reset_completed', seed=seed, steps=task.step_count)
            row = reference.get(str(seed))
            if row and not args.reset_only:
                task.cfg.step_lim = row['policy_execution']['physics_call_limit']
                task.mean_steps = task.cfg.step_lim
                for item in row['tactile_feedback']['actions']:
                    value = torch.as_tensor(item['action'], dtype=torch.float32,
                                            device=task._robot_manager.device)
                    success = False
                    for _ in range(2):
                        okay, success = task.take_action(value, action_type='qpos', force=True)
                        if not okay or not task.plan_success:
                            raise RuntimeError('Replay action rejected')
                        if success:
                            break
                    task._get_observations()
                    if success or task.check_early_stop():
                        break
                record('replay_completed', seed=seed, steps=task.step_count,
                       success=bool(task.check_success()))
            task.clean_cache(result='probe')
        record('completed')
    finally:
        if task is not None:
            task.close()
        app.close()


if __name__ == '__main__':
    main()
