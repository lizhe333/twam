"""Route isolated CUDA workers separately from the physical graphics device."""
import json


def worker_app_launcher(base_launcher, environ):
    class WorkerAppLauncher(base_launcher):
        def _resolve_device_settings(self, launcher_args):
            super()._resolve_device_settings(launcher_args)
            # Isaac Lab 2.3 otherwise copies the logical CUDA index into the
            # graphics index. Vulkan does not follow CUDA_VISIBLE_DEVICES.
            launcher_args.update(active_gpu=int(environ['PRO6K_RENDER_GPU']),
                                 physics_gpu=0, multi_gpu=False)
    return WorkerAppLauncher


def configure_worker_gpu(args, cuda, environ):
    # Invalid/unavailable CUDA devices still produce their native CUDA error.
    # Management UUIDs are bookkeeping labels, not a cross-API requirement.
    cuda.set_device(args.device)
    environ['PRO6K_DEVICE'] = args.device
    record = dict(cuda_device=args.device, management_uuid=environ.get('PRO6K_GPU_UUID'),
                  cuda_visible_devices=environ.get('CUDA_VISIBLE_DEVICES'),
                  render_gpu=environ.get('PRO6K_RENDER_GPU'))
    try:
        record['cuda_uuid'] = str(cuda.get_device_properties(args.device).uuid)
    except Exception as exc:
        record['identity_note'] = f'{type(exc).__name__}: {exc}'
    environ['PRO6K_GPU_BINDING'] = json.dumps(record)
    print(f'GPU assignment (diagnostic only): {environ["PRO6K_GPU_BINDING"]}', flush=True)
    return record
