"""Explicit contract for Isaac 4.5-trained policies transferred to Isaac 5.1."""
PROFILE = 'tactile-head-only-isaac51-pro6k-transfer-v1'
UNIVTAC_COMMIT = 'd541e5568227ca3b66104d294f63c80acad7c52c'
INPUT_PROFILE = 'jpeg-measured-history-v1'
WORKER_ROUTING = 'isolated-cuda0-physical-render-v1'
CONTACT_RESET_PATCH = 'refresh-contact-before-friction-v1'
RESET_TIMEOUT_SECONDS = 600.0
RENDER_SETTINGS = {
    '/rtx/rendermode': 'RaytracedLighting',
    '/rtx-transient/dldenoiser/enabled': True,
    '/rtx-transient/dlssg/enabled': False,
    '/rtx/post/aa/op': 3,
    '/rtx/post/dlss/execMode': 2,
    '/rtx/post/motionblur/enabled': False,
}


def validate_render(settings):
    actual = {k: settings.get(k) for k in RENDER_SETTINGS}
    if actual != RENDER_SETTINGS:
        raise RuntimeError(f'Unexpected transfer renderer settings: {actual}')
    return actual


def kit_arguments(gpu_index):
    settings = dict(RENDER_SETTINGS, **{
        '/renderer/activeGpu': int(gpu_index),
        '/physics/cudaDevice': 0,
        '/renderer/multiGpu/enabled': False,
        '/rtx/multiGpu/enabled': False,
        '/plugins/carb.tasking.plugin/threadCount': 4,
    })
    return '--kit_args=' + ' '.join(
        f'--{k}={str(v).lower() if isinstance(v, bool) else v}' for k, v in settings.items())


def validate_timing(task):
    if task.cfg.decimation != 1 or abs(task.cfg.sim.dt - 1 / 120) > 1e-9:
        raise RuntimeError('Policy repeats each action twice: require decimation=1 and physics=120 Hz')


def configure_reset(cfg):
    # This is an elapsed wall-clock timeout during reset, not an action budget
    # or simulation-time limit. Shader and CUDA cold starts can exceed 120 s.
    cfg.reset_time_limit = RESET_TIMEOUT_SECONDS


def stop_on_environment_error(task):
    """Keep the failed seed's diagnostics, then stop instead of reporting 0/50."""
    original = task.clean_cache

    def clean_cache(mean_steps=0.0, result=None):
        value = original(mean_steps=mean_steps, result=result)
        if result == 'error':
            reason = task.metadata.get('failure_reason', 'unknown environment/runtime error')
            raise RuntimeError(f'Evaluation aborted at seed {task.cfg.seed}: {reason}; '
                               'diagnostics saved, no formal success rate produced')
        return value

    task.clean_cache = clean_cache


def validate_audits(metadata, job, fingerprint):
    rows = {k: v for k, v in metadata.items() if k.isdigit() and int(k) >= 1000000}
    if set(rows) != {str(s) for s in range(1000000, 1000050)}:
        raise ValueError('Expected all 50 fixed seeds; failures/errors must not be filtered')
    for seed, row in rows.items():
        if row.get('result') not in ('success', 'failed', 'error'):
            raise ValueError(f'Invalid outcome for {seed}')
        if row['result'] == 'error':
            raise ValueError(f'Environment/runtime error at seed {seed}; formal result is invalid')
        audit = row.get('policy_execution', {}).get('transfer', {})
        expected = dict(profile=PROFILE, input_alignment=INPUT_PROFILE,
                        contact_reset_patch=CONTACT_RESET_PATCH,
                        training_simulator='4.5', evaluation_simulator='5.1',
                        task=job['task'], scene_seed=int(seed), gpu_uuid=job['gpu_uuid'],
                        runtime_fingerprint=fingerprint, render_settings=RENDER_SETTINGS)
        if any(audit.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Invalid transfer input audit for {seed}')
    return rows
