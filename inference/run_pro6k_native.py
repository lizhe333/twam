"""Launch Isaac 5.1 with the frozen fixed-seed loop and a tactile-only policy."""
import argparse
import importlib.abc
import importlib.util
import importlib.machinery
import json
import os
from pathlib import Path
import sys

RUNTIME = Path(os.environ['PRO6K_RUNTIME'])
SOURCE = Path(os.environ['PRO6K_UNIVTAC'])
sys.path[:0] = [str(RUNTIME / 'src'), str(RUNTIME / 'inference'), str(SOURCE)]
from pro6k_contract import (RENDER_SETTINGS, RESET_TIMEOUT_SECONDS, WORKER_ROUTING,
                           configure_reset, stop_on_environment_error, validate_timing)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('policy.TactileWave0Wam', 'policy.Pi05Wam'):
            return importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
        names = {'policy.TactileWave0Wam.deploy_policy': 'native_tactile.py',
                 'policy.Pi05Wam.deploy_policy': 'native_pi05.py'}
        if fullname in names:
            return importlib.util.spec_from_file_location(fullname, RUNTIME / names[fullname])


def main():
    os.chdir(SOURCE)
    sys.meta_path.insert(0, Finder())
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--deploy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation-episode', action='store_true',
                        help='Run one complete development-validation episode, not a formal evaluation')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    import torch
    from pro6k_gpu import configure_worker_gpu, worker_app_launcher
    configure_worker_gpu(args, torch.cuda, os.environ)
    (args.output.parent / 'gpu-binding.json').write_text(os.environ['PRO6K_GPU_BINDING'] + '\n')
    app = worker_app_launcher(AppLauncher, os.environ)(args).app
    task = policy = None
    try:
        from envs.utils.env_parser import build_task_env_cfg, load_task_config
        import carb.settings
        # Keep the support package pinned when the old bridge prepends its
        # original checkout path later.
        import pi05_univtac
        from pro6k_policy import Policy
        import fixed_seed_loop as loop
        config, path = load_task_config('clean', ['env_settings.frequencies.eval=120'])
        module, cfg, timing, _ = build_task_env_cfg(args.task, config, path.stem, 'eval',
                                                   device=args.device, save_dir=args.output)
        # Keep 120 physical ticks / second, with two explicit calls per model action.
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
        deployment = json.loads(args.deploy.read_text())
        deployment['task_name'] = args.task
        policy = Policy(deployment)
        task = module.Task(cfg, mode='eval')
        stop_on_environment_error(task)
        validate_timing(task)
        loop.log_path = task.save_root / 'log.log'
        loop.args_cli = argparse.Namespace(print_only=False)
        loop.log(f'Evaluation directory: {task.save_root.resolve()}')
        loop.log('Transfer evaluation: training=Isaac4.5 evaluation=Isaac5.1; not an in-domain comparison')
        if args.validation_episode:
            (task.save_root / 'validation-only.json').write_text(json.dumps(
                dict(purpose='development validation', formal_result=False, episodes=1)) + '\n')
        (task.save_root / 'resolved_config.json').write_text(json.dumps(
            dict(task=args.task, physical_hz=120, strategy_hz=60, action_repeats=2,
                 decimation=cfg.decimation, simulator='5.1', training_simulator='4.5',
                 reset_timeout_seconds=RESET_TIMEOUT_SECONDS, worker_routing=WORKER_ROUTING), indent=2))
        instructions = json.loads((SOURCE / 'instructions' / f'{args.task}.json').read_text())
        episodes = 1 if args.validation_episode else 50
        result = loop.eval_policy(task, policy, False, 1000000, 1000000 + episodes - 1, episodes, instructions, 'seen')
        loop.log(f"Final Result: {result['succ_num']}/{result['test_num']} success.")
    finally:
        try:
            if policy is not None:
                policy.close()
        finally:
            try:
                if task is not None:
                    task.close()
            finally:
                app.close()


if __name__ == '__main__':
    main()
