"""Seed both policy samplers per scene and replan; never seed simulator RNG here."""
import importlib.util
import os
from pathlib import Path

from pair_contract import inference_seed


def seed_sampler(scene, step):
    import torch
    torch.manual_seed(inference_seed(scene, step))


def tactile_main():
    from contact_belief_wam import serve
    original = serve.AlignedPolicyService.infer

    def infer(self, message):
        scene = int(message['episode'].rsplit(':', 1)[-1])
        seed_sampler(scene, message['step'])
        return original(self, message)

    serve.AlignedPolicyService.infer = infer
    from contact_belief_wam.cli import main
    return main()


def base_main():
    runtime = Path(os.environ['PAIR_RUNTIME'])
    spec = importlib.util.spec_from_file_location('paired_base_server', runtime / 'base_server_original.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.dispatch_observation

    def dispatch(policy, observation):
        seed_sampler(observation['pair_scene_seed'], observation['pair_policy_step'])
        return original(policy, observation)

    module.dispatch_observation = dispatch
    return module.main()
