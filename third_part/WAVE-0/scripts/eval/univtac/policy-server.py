#!/usr/bin/env python3
"""Serve WAVE-0 using the checkpoint's explicit EMA/raw-weight contract."""
import argparse
import json
from pathlib import Path
import traceback


def prepare_inference_policy(policy, precision='bfloat16', device='cuda'):
    """Convert the loaded policy on CPU before allocating GPU weights.

    Use the upstream PI0.5 mixed-precision policy, including its FP32 norm
    and vision-embedding exceptions. World-model weights and normalization
    remain FP32. This changes only evaluation storage, never checkpoint files.
    """
    import torch
    if precision not in ('bfloat16', 'float32'):
        raise ValueError(f'Unsupported inference precision: {precision}')
    if any(p.device.type != 'cpu' for p in policy.model.parameters()):
        raise ValueError('Load the policy on CPU before inference conversion')
    before = sum(p.numel()*p.element_size() for p in policy.model.parameters())
    policy.model.model.paligemma_with_expert.to_bfloat16_for_selected_params(precision)
    after = sum(p.numel()*p.element_size() for p in policy.model.parameters())
    policy.device = torch.device(device)
    policy.model.to(policy.device).eval()
    return dict(backbone_precision=precision, world_model_precision='float32',
                parameter_bytes_before=before, parameter_bytes_after=after)


def infer_observation(policy, observation):
    import numpy as np
    if observation.get('task') != policy.metadata['task']:
        raise ValueError('Observation task does not match checkpoint')
    if observation.get('prompt') != policy.metadata['prompt']:
        raise ValueError('Observation instruction does not match checkpoint')
    image = np.asarray(observation['image'])
    state = np.asarray(observation['state'], dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError('Expected HWC uint8 head RGB')
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError('Expected finite current eight-dimensional joint state')
    # Only current observations enter inference; future/GT targets remain local.
    kwargs = {}
    if policy.metadata['camera'] == 'head_wrist':
        wrist = np.asarray(observation['wrist_image'])
        if wrist.ndim != 3 or wrist.shape[-1] != 3 or wrist.dtype != np.uint8:
            raise ValueError('Expected HWC uint8 current wrist RGB')
        kwargs['wrist_rgb'] = wrist
    capture = getattr(policy, 'latent_capture', None)
    if capture is not None:
        capture.latest = None
    result = {'actions': policy.infer(image, state, **kwargs)}
    if capture is not None:
        if capture.latest is None:
            raise RuntimeError('No latent prediction captured')
        result['latent_diagnostics'] = capture.latest
    return result


def handler_for(policy, metadata):
    from openpi_client import msgpack_numpy
    def handle(connection):
        packer = msgpack_numpy.Packer()
        connection.send(packer.pack(metadata))
        for message in connection:
            try:
                connection.send(packer.pack(dispatch_observation(policy, msgpack_numpy.unpackb(message))))
            except Exception:
                traceback.print_exc()
                connection.send(traceback.format_exc())
                return
    return handle


def dispatch_observation(policy, observation):
    store = getattr(policy, 'verification_rng', None)
    command = observation.get('verification_command')
    if command is not None and store is None:
        raise ValueError('Verification RNG RPC is disabled on this policy server')
    if store is None:
        return infer_observation(policy, observation)
    with store.inference_lock:
        if command == 'capture':
            token, metadata = store.capture()
            return dict(token=token, metadata=metadata)
        if command == 'restore':
            return dict(restored=True, metadata=store.restore(observation['token']))
        if command == 'check_same_observation':
            token, _ = store.capture()
            try:
                return store.verify_same_observation(policy, observation['observation'], token, tolerance=1e-5)
            finally:
                store.discard(token)
        if command is not None:
            raise ValueError(f'Unknown verification command: {command}')
        return infer_observation(policy, observation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--backbone-precision', choices=['bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--record-latents', action='store_true', help='Return read-only latent diagnostics to the simulator')
    parser.add_argument('--verification-rng', action='store_true', help='Enable isolated contact-verification RNG snapshots')
    args = parser.parse_args()
    from lerobot.utils.univtac_runtime import configure
    configure()
    import torch
    from lerobot.policies.wave0.inference_wave0 import Policy
    from websockets.sync.server import serve
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    policy = Policy(args.checkpoint, device='cpu')
    precision = prepare_inference_policy(policy, args.backbone_precision, device=args.device)
    if args.record_latents:
        import sys
        sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / 'UniVTAC').is_dir())/'pi05-univtac/src'))
        from pi05_univtac.wave_trace import LatentCapture
        policy.latent_capture = LatentCapture(policy.model.model)
    torch.manual_seed(0)
    if args.verification_rng:
        import sys
        sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / 'UniVTAC').is_dir())/'pi05-univtac/src'))
        from pi05_univtac.contact_policy_rng import TorchRNGTokenStore
        policy.verification_rng = TorchRNGTokenStore()
    metadata = dict(tasks=[policy.metadata['task']], action_shape=[50, 8],
                    checkpoint=str(args.checkpoint.resolve()), model_mode='full',
                    policy_family='wave0', camera_mode=policy.metadata['camera'], use_ema=policy.use_ema,
                    inference_precision=precision, record_latents=args.record_latents,
                    inference_device=args.device, cpu_threads=args.cpu_threads,
                    verification_rng=args.verification_rng)
    print(json.dumps(metadata), flush=True)
    with serve(handler_for(policy, metadata), args.host, args.port,
               compression=None, max_size=None, ping_interval=None) as server:
        server.serve_forever()


if __name__ == '__main__':
    main()
