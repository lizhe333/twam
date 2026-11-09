"""Regressions for cloud UUID aliases and ordinary CUDA ordinal selection."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / name)
               for name in ('inference', 'scripts')]
from pro6k_gpu import configure_worker_gpu, worker_app_launcher


@pytest.mark.parametrize('index,cloud_uuid,cuda_uuid', [
    (0, 'GPU-81764817-8ff4-40db-9386-2dbef5ace833', '4596116d-bf3e-dac0-4617-ac822ee65962'),
    (1, 'GPU-12901efc-ded7-2c70-3d49-c94c93877cb8', '87096ab5-5617-b27b-72c1-85b05cdcbed9'),
])
def test_cloud_uuid_mismatch_does_not_block_requested_device(index, cloud_uuid, cuda_uuid):
    selected = []
    cuda = SimpleNamespace(set_device=selected.append,
                           get_device_properties=lambda device: SimpleNamespace(uuid=cuda_uuid))
    args = SimpleNamespace(device='cuda:0')
    env = dict(PRO6K_GPU_UUID=cloud_uuid, CUDA_VISIBLE_DEVICES=str(index), PRO6K_RENDER_GPU=str(index))
    record = configure_worker_gpu(args, cuda, env)
    assert selected == ['cuda:0']
    assert env['PRO6K_DEVICE'] == 'cuda:0'
    assert record['render_gpu'] == str(index)
    assert record['cuda_visible_devices'] == str(index)
    assert record['management_uuid'] == cloud_uuid
    assert record['cuda_uuid'] == cuda_uuid
    assert json.loads(env['PRO6K_GPU_BINDING']) == record


@pytest.mark.parametrize('physical_index', [0, 1])
def test_isaaclab_default_does_not_reset_renderer_to_physical_zero(physical_index):
    class Launcher:
        def _resolve_device_settings(self, args):
            args.update(active_gpu=0, physics_gpu=0)
    launcher = worker_app_launcher(Launcher, dict(PRO6K_RENDER_GPU=str(physical_index)))()
    args = dict(device='cuda:0')
    launcher._resolve_device_settings(args)
    assert args == dict(device='cuda:0', active_gpu=physical_index, physics_gpu=0, multi_gpu=False)


def test_missing_uuid_metadata_does_not_prevent_startup():
    selected = []
    cuda = SimpleNamespace(set_device=selected.append,
                           get_device_properties=lambda device: SimpleNamespace())
    record = configure_worker_gpu(SimpleNamespace(device='cuda:1'), cuda, {})
    assert selected == ['cuda:1']
    assert 'identity_note' in record


def test_actual_cuda_device_failure_is_not_hidden():
    def unavailable(device):
        raise RuntimeError('invalid device ordinal')
    cuda = SimpleNamespace(set_device=unavailable)
    with pytest.raises(RuntimeError, match='invalid device ordinal'):
        configure_worker_gpu(SimpleNamespace(device='cuda:7'), cuda, {})


def test_worker_failure_surfaces_original_error_in_cloud_console(tmp_path):
    import reeval_pro6k
    log = tmp_path / 'eval.log'
    log.write_text('Traceback:\nRuntimeError: invalid device ordinal\n')
    batch = SimpleNamespace(stop_group=lambda process: None)
    item = dict(process=SimpleNamespace(returncode=1), job=dict(task='insert_hole'),
                work=tmp_path, log=log)
    with pytest.raises(RuntimeError, match='invalid device ordinal') as exc:
        reeval_pro6k.finish(item, None, None, batch)
    assert 'exit code 1' in str(exc.value)


def test_native_crash_keeps_first_error_before_long_driver_dump(tmp_path):
    import reeval_pro6k
    log = tmp_path / 'eval.log'
    error = '[Error] [carb.glinterop.plugin] Could not get required EGL context creation functions'
    log.write_text(error + '\n' + 'NVIDIA crash diagnostics\n' * 200)
    batch = SimpleNamespace(stop_group=lambda process: None)
    item = dict(process=SimpleNamespace(returncode=-11), job=dict(task='insert_hole'),
                work=tmp_path, log=log)
    with pytest.raises(RuntimeError) as exc:
        reeval_pro6k.finish(item, None, None, batch)
    assert error in str(exc.value)
    assert 'exit code -11' in str(exc.value)
