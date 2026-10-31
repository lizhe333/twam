"""Keep the A800 render contract in effect before camera frames are produced."""
from functools import wraps

from pair_contract import RENDER_SETTINGS, apply_render_settings, validate_render


def configure_task_render(cfg):
    cfg.sim.render.rendering_mode = None
    # Setting this to "Off" calls Replicator's realtime-renderer helper and
    # would switch out of PathTracing. Set the numeric AA value through carb.
    cfg.sim.render.antialiasing_mode = None
    cfg.sim.render.enable_dlssg = False
    cfg.sim.render.enable_dl_denoiser = False
    cfg.sim.render.carb_settings = dict(RENDER_SETTINGS)


def install_render_guard(simulation_type, settings):
    original = simulation_type.render
    if getattr(original, '_a800_render_guard', False):
        return

    @wraps(original)
    def render(self, *args, **kwargs):
        # Apply BEFORE rendering, so an already-produced image is never
        # relabeled by changing settings only when the policy receives it.
        apply_render_settings(settings)
        result = original(self, *args, **kwargs)
        # An override during rendering still invalidates the frame.
        validate_render(settings)
        return result

    render._a800_render_guard = True
    simulation_type.render = render


def initialize_rendering():
    # Only called by the frozen native evaluator after SimulationApp starts.
    import carb.settings
    from isaaclab.sim import SimulationContext

    settings = carb.settings.get_settings()
    apply_render_settings(settings)
    install_render_guard(SimulationContext, settings)
