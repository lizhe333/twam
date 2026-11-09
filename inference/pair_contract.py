"""Inference-only contract for the A800 base/tactile comparison (no Kit imports)."""
PROFILE = 'a800-pathtracing-optix32-aligned-v2'
INPUT_PROFILE = 'jpeg-measured-history-v1'
RNG_PROFILE = 'scene-seed-policy-step-v1'

# These settings exist in the installed Isaac Sim 4.5 / Kit 106 renderer.
# A800 supports OptiX, but not DLSS Ray Reconstruction. Use 32 fresh samples
# per rendered frame instead of depending on temporal accumulation/DLSS.
RENDER_SETTINGS = {
    '/rtx/rendermode': 'PathTracing',
    '/rtx/pathtracing/spp': 32,
    '/rtx/pathtracing/totalSpp': 32,
    '/rtx/pathtracing/adaptiveSampling/enabled': False,
    '/rtx/pathtracing/maxBounces': 4,
    '/rtx/pathtracing/optixDenoiser/enabled': True,
    '/rtx/pathtracing/optixDenoiser/temporalMode/enabled': False,
    '/rtx/pathtracing/optixDenoiser/blendFactor': 0.0,
    # Do not denoise depth or other sensor ground-truth AOVs.
    '/rtx/pathtracing/optixDenoiser/AOV': False,
    '/rtx/post/aa/op': 0,
    '/rtx-transient/dlssg/enabled': False,
    '/rtx-transient/dldenoiser/enabled': False,
    '/rtx/post/motionblur/enabled': False,
}


SIMULATION_APP_SETTINGS = {
    'renderer': RENDER_SETTINGS['/rtx/rendermode'],
    'anti_aliasing': RENDER_SETTINGS['/rtx/post/aa/op'],
    'samples_per_pixel_per_frame': RENDER_SETTINGS['/rtx/pathtracing/spp'],
    'denoiser': RENDER_SETTINGS['/rtx/pathtracing/optixDenoiser/enabled'],
    'max_bounces': RENDER_SETTINGS['/rtx/pathtracing/maxBounces'],
}


def persistent_render_settings():
    # Kit restores these defaults when render products/stages are initialized.
    defaults = {key.replace('/rtx', '/rtx-defaults', 1): value
                for key, value in RENDER_SETTINGS.items()}
    return {**defaults, **RENDER_SETTINGS}


def apply_render_settings(settings):
    for key, value in persistent_render_settings().items():
        if settings.get(key) != value:
            settings.set(key, value)
    return validate_render(settings)


def kit_arguments():
    settings = {'/rtx/verifyDriverVersion/enabled': False, **persistent_render_settings()}
    def value(v):
        return str(v).lower() if isinstance(v, bool) else str(v)
    return '--kit_args=' + ' '.join(f'--{k}={value(v)}' for k, v in settings.items())


def inference_seed(scene_seed, policy_step):
    scene_seed, policy_step = int(scene_seed), int(policy_step)
    if not 1000000 <= scene_seed <= 1000049 or not 0 <= policy_step < 10000 or policy_step % 10:
        raise ValueError('Expected a formal scene seed and a ten-action replan boundary')
    return scene_seed * 10000 + policy_step


def validate_render(settings):
    actual = {key: settings.get(key) for key in RENDER_SETTINGS}
    wrong = {key: (RENDER_SETTINGS[key], value) for key, value in actual.items()
             if value != RENDER_SETTINGS[key]}
    if wrong:
        raise RuntimeError(f'Comparison renderer settings changed: {wrong}')
    return actual
