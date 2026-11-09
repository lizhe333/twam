"""Same-seed renderer experiment, using existing commands and no collection.

No policy server or recovery Teacher is started. Short continuations compare
restored physical states with rendering history retained versus AA disabled.
The existing dataset, manifests, acceptance thresholds, and quota are untouched.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import traceback

import h5py
import numpy as np

from .paths import WORKSPACE, atomic_json, sha256


IMAGE_KEYS = ("head_rgb", "wrist_rgb", "left_rgb_marker", "right_rgb_marker")


def image_errors(a, b):
    return {key: dict(mae=float(np.abs(a[key].astype(float) - b[key]).mean()),
                      maximum=float(np.abs(a[key].astype(float) - b[key]).max()))
            for key in IMAGE_KEYS}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    os.chdir(WORKSPACE / "UniVTAC")
    sys.path[:0] = [str(WORKSPACE / "UniVTAC"), str(WORKSPACE / "pi05-univtac/src")]
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(argv)
    args.output, args.reference = args.output.resolve(), args.reference.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists():
        raise ValueError("Refusing to overwrite an existing native diagnostic")
    import subprocess
    uuid = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    lock = (WORKSPACE / "outputs/eval_suites" / f".{uuid}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.headless, args.enable_cameras, args.livestream = True, True, 0
    app = AppLauncher(args).app
    task = None
    report = dict(purpose="same_state_pixel_diagnosis", diagnostic_only=True, seed=3000001,
                  collection_enabled=False, training_enabled=False, accepted_samples=0,
                  reference=str(args.reference), reference_sha256=sha256(args.reference),
                  source_sha256=sha256(__file__), physical_strategy_actions=0)

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    try:
        import carb
        import torch
        import yaml
        from pi05_univtac.contact_restore import capture_task_state, restore_task_state
        from pi05_univtac.contact_trace import snapshot_observation
        from pi05_univtac.sim_policy import execute_action
        from pi05_univtac.constants import prompt
        from .takeover import compare_replay, relative_grasp_pose
        settings = carb.settings.get_settings()
        setting_keys = ("/rtx/rendermode", "/rtx/post/aa/op", "/rtx/post/dlss/execMode",
                        "/rtx-transient/dlssg/enabled", "/rtx-transient/dldenoiser/enabled",
                        "/rtx/indirectDiffuse/enabled", "/rtx/directLighting/enabled",
                        "/rtx/reflections/enabled", "/rtx/ambientOcclusion/enabled")
        module = importlib.import_module("envs.insert_HDMI")
        cfg = module.TaskCfg()
        clean = yaml.safe_load((WORKSPACE / "UniVTAC/task_config/clean.yml").read_text())
        cfg.save_dir, cfg.decimation = args.output / "native", 1
        cfg.save_frequency, cfg.video_frequency, cfg.render_frequency = 2, 2, 0
        cfg.tactile_sensor_type, cfg.obs_data_type = "gsmini", clean["observations"]
        cfg.random_texture = clean.get("random_texture", False)
        cfg.step_lim, cfg.reset_time_limit, cfg.sim.device = 1200, 300, args.device or "cuda:0"
        task = module.Task(cfg, mode="eval")
        task.reset(seed=3000001, instructions=[prompt("insert_HDMI")])
        if not task.plan_success:
            raise RuntimeError("Native pre-move failed")
        report["render_settings"] = {key: settings.get(key) for key in setting_keys}
        atomic_json(args.output / "progress.json", report | {"phase": "initialized"})
        with h5py.File(args.reference, "r") as f:
            commands = f["actions/clipped_actual_target"][:70]

        def row():
            value = snapshot_observation(task, task._get_observations())
            value = {key: np.asarray(v).copy() for key, v in value.items()}
            value["object_in_gripper_pose_world"] = relative_grasp_pose(value)
            for name in ("prism", "slot"):
                value[f"{name}_vertices_world"] = np.asarray(getattr(task, name).vertices, np.float32).copy()
            value.update(physics_step=np.asarray(task.step_count), action_count=np.asarray(task.take_action_cnt),
                         uipc_frame=np.asarray(task.uipc_sim.world.frame()))
            return value

        def step(command):
            success, substeps = execute_action(task, command)
            report["physical_strategy_actions"] += 1
            if success or substeps != 2:
                raise RuntimeError("Diagnostic reached success/partial action unexpectedly")

        def save(name, rows):
            np.savez_compressed(args.output / f"{name}.npz", **{k: np.stack([r[k] for r in rows]) for k in rows[0]})

        def continuation(name):
            rows = [row()]
            for command in commands[40:50]:
                step(command)
                rows.append(row())
            save(name, rows)
            atomic_json(args.output / "progress.json", report | {"phase": name})
            return rows

        def static_render(name, count=24):
            rows = [row()]
            for _ in range(count):
                task._update_render()
                rows.append(row())
            invariants = {}
            for key in ("physics_step", "action_count", "uipc_frame", "joint_q",
                        "prism_vertices_world", "slot_vertices_world"):
                maximum = max(float(np.abs(r[key].astype(float) - rows[0][key]).max()) for r in rows)
                invariants[key] = maximum
                if maximum > (1e-6 if key == "joint_q" else 0):
                    raise RuntimeError(f"Render-only operation changed physical state: {key}={maximum}")
            save(name, rows)
            report[name] = dict(physical_invariant_errors=invariants,
                image_differences_from_cached=[image_errors(rows[0], r) for r in rows],
                final_consecutive_difference=image_errors(rows[-2], rows[-1]))
            return rows

        for command in commands[:40]:
            step(command)
        # Camera data is lazy. The actual Runner records o_t before snapshotting;
        # do the same here, so the saved cache/dirty flags belong to this frame.
        # Snapshotting an unread camera would make restored o_t fetch the later
        # RTX output, introducing a separate cache-order error into this test.
        row()
        original = capture_task_state(task)
        expected = continuation("default_direct")
        for command in commands[50:70]:
            step(command)
        restore_task_state(task, original)
        actual = continuation("default_restored")
        report["default_replay"] = compare_replay(expected, actual, commands[40:50], commands[40:50])
        report["default_step_pixels"] = [image_errors(a, b) for a, b in zip(expected, actual)]
        restore_task_state(task, original)
        static_render("default_render_only")

        # One isolated renderer intervention; physics, actions and snapshot APIs
        # remain unchanged. Restore the mode before leaving the diagnostic.
        aa = settings.get("/rtx/post/aa/op")
        settings.set_int("/rtx/post/aa/op", 0)
        report["aa_off_value"] = settings.get("/rtx/post/aa/op")
        static_render("aa_off_render_only")
        off_snapshot = capture_task_state(task)
        expected_off = continuation("aa_off_direct")
        for command in commands[50:70]:
            step(command)
        restore_task_state(task, off_snapshot)
        actual_off = continuation("aa_off_restored")
        report["aa_off_replay"] = compare_replay(expected_off, actual_off, commands[40:50], commands[40:50])
        report["aa_off_step_pixels"] = [image_errors(a, b) for a, b in zip(expected_off, actual_off)]

        denoiser_key = "/rtx-transient/dldenoiser/enabled"
        denoiser = settings.get(denoiser_key)
        settings.set_bool(denoiser_key, False)
        restore_task_state(task, off_snapshot)
        static_render("aa_and_denoiser_off_render_only")
        clean_snapshot = capture_task_state(task)
        expected_clean = continuation("aa_and_denoiser_off_direct")
        for command in commands[50:70]:
            step(command)
        restore_task_state(task, clean_snapshot)
        actual_clean = continuation("aa_and_denoiser_off_restored")
        report["aa_and_denoiser_off_replay"] = compare_replay(
            expected_clean, actual_clean, commands[40:50], commands[40:50])
        report["aa_and_denoiser_off_step_pixels"] = [image_errors(a, b) for a, b in zip(expected_clean, actual_clean)]

        # A separate render-only test distinguishes RGB marker noise from real
        # contact changes. Restore RNG after each sample, so each synthesis uses
        # the same Torch noise. No actions, physics, or student data are created.
        restore_task_state(task, clean_snapshot)
        for _ in range(8):
            task._update_render()
        cpu_rng = torch.random.get_rng_state().clone()
        cuda_rng = [x.clone() for x in torch.cuda.get_rng_state_all()]
        fixed = []
        for _ in range(12):
            torch.random.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
            task._update_render()
            fixed.append(row())
        save("fixed_rng_render_only", fixed)
        report["fixed_rng_render_only"] = dict(
            differences_from_first=[image_errors(fixed[0], r) for r in fixed],
            physical_invariant_errors={k:max(float(np.max(np.abs(r[k].astype(float)-fixed[0][k]))) for r in fixed)
                for k in ("physics_step", "action_count", "uipc_frame", "joint_q", "prism_vertices_world", "slot_vertices_world")},
            render_settings={key:settings.get(key) for key in setting_keys})
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        settings.set_bool(denoiser_key, denoiser)
        settings.set_int("/rtx/post/aa/op", aa)
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        atomic_json(args.output / "result.json", report)
        if task is not None:
            task.close()
        app.close()
        lock.close()
    print(json.dumps({k: report.get(k) for k in ("status", "error", "physical_strategy_actions", "render_settings")}), flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
