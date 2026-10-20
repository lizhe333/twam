"""Recovery expert extensions; imported inside the existing UniVTAC simulator process."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .config import RecoveryConfig
from .paths import atomic_json


def cpu(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def insertion_progress(displacement, axis, entrance_normal):
    """Do not mistake sliding across a tilted entrance for insertion progress."""
    displacement, axis, entrance_normal = map(np.asarray, (displacement, axis, entrance_normal))
    axial = float(displacement @ axis)
    incidence = float(axis @ entrance_normal)
    if abs(incidence) < 1e-6:
        return axial
    depth = float(displacement @ entrance_normal) / incidence
    return min(axial, depth)


class TactileBlockageDetector:
    def __init__(self, cfg: RecoveryConfig, baseline):
        self.cfg = cfg
        self.reference = np.mean(baseline, axis=0).astype(np.float32)
        stable = [self.raw_score(x) for x in baseline]
        self.baseline_mean = float(np.mean(stable))
        self.threshold = max(cfg.tactile_threshold_floor,
                             float(cfg.tactile_threshold_std_scale * np.std(stable)))
        self.recent = []

    def raw_score(self, images):
        return float(np.abs(np.asarray(images, np.float32) - self.reference).mean(axis=(1, 2, 3)).max() / 255)

    def score(self, images):
        return max(0.0, self.raw_score(images) - self.baseline_mean)

    def observe(self, images):
        score = self.score(images)
        self.recent.append(score)
        self.recent = self.recent[-self.cfg.blocked_persistence_steps:]
        return score

    def blocked(self, actual_progress, requested_progress):
        sustained = (len(self.recent) == self.cfg.blocked_persistence_steps and
                     min(self.recent) >= self.threshold)
        return sustained and actual_progress < requested_progress * self.cfg.progress_ratio_threshold


class RecoveryController:
    def __init__(self, task, task_name, cfg: RecoveryConfig):
        self.task, self.task_name, self.cfg = task, task_name, cfg
        self.events, self.trace, self.phase = [], [], "approach"
        self.detector = None
        self.blockages = 0
        self.original_step = task._step
        self.original_save = task.save_observations
        task._step = self.step
        task.save_observations = self.save_observation

    def close(self):
        self.task._step = self.original_step
        self.task.save_observations = self.original_save

    @staticmethod
    def touch(observation):
        values = observation["tactile"]
        return np.stack([cpu(values.get(f"{side}_tactile", values.get(f"{side}_gsmini"))["rgb_marker"])
                         for side in ("left", "right")])

    def save_observation(self, obs):
        # Match upstream downloaded HDF5 names without changing live sensor names.
        obs = dict(obs)
        obs["tactile"] = {key.replace("_tactile", "_gsmini"): value for key, value in obs["tactile"].items()}
        self.original_save(obs)

    def step(self, is_save=True):
        # The stock expert has unsaved settling delays. During recovery those
        # frames belong to the same sequence and must remain available as targets.
        is_save = bool(is_save or not self.task.in_pre_move)
        self.original_step(is_save=is_save)
        task = self.task
        if not task.in_pre_move and is_save and task.step_count % 2 == 0:
            obs = task._get_observations()
            score = self.detector.observe(self.touch(obs)) if self.detector else 0.0
            self.trace.append({"step": int(task.step_count), "phase": self.phase, "touch_score": score,
                               "threshold": self.detector.threshold if self.detector else None,
                               "prism_position": task.prism.get_pose().p.tolist(),
                               "prism_pose": task.prism.get_pose().tolist(),
                               "in_hand_pose": task.prism.get_pose().rebase(
                                   task._robot_manager.get_gripper_center_pose()).tolist(),
                               "joint": cpu(obs["embodiment"]["joint"])[:8].tolist()})

    def event(self, phase, **values):
        self.phase = phase
        self.events.append({"event": phase, "step": int(self.task.step_count), **values})
        print(json.dumps(self.events[-1]), flush=True)

    def move(self, actions, phase, **kwargs):
        self.phase = phase
        okay = self.task.move(actions, tag=phase, is_save=True, delay=False, **kwargs)
        if not okay or not self.task.plan_success:
            raise RuntimeError(f"Recovery motion planner failed during {phase}")

    def probe(self, displacement, physics_steps=None):
        """Execute one tiny planned translation without the planner's long minimum duration.

        cuRobo emits about 104 physical steps even for a 0.5 mm displacement.
        Resample only these local probes; approach, retreat and insertion retain
        the task's native trajectories. All physical steps still render/save.
        """
        task = self.task
        action = task.atom.move_by_displacement(*displacement.tolist())[0]
        self.execute_local_pose(action.target_pose, physics_steps or self.cfg.probe_physics_steps, "push")

    def execute_local_pose(self, target_pose, physics_steps, phase):
        import torch
        task = self.task
        arm = task._robot_manager.plan_arm(target_pose, time_dilation_factor=.3)
        if arm["status"] == "Fail":
            raise RuntimeError(f"Recovery motion planner failed during {phase}")
        count = min(arm["num_steps"], max(2, physics_steps))
        indices = torch.linspace(0, arm["num_steps"] - 1, count, device=arm["position"].device).round().long()
        position = arm["position"][indices]
        velocity = torch.zeros_like(position)
        velocity[1:] = (position[1:] - position[:-1]) / task.cfg.sim.dt
        task.atom_id += 1
        task.atom_tag = phase
        task.take_dense_action({"arm": {**arm, "num_steps": count, "position": position,
                                        "velocity": velocity}, "gripper": None}, is_save=True)
        task._update_render()

    def align_above_hole(self):
        """Orient the part before rim contact, while its tip is still clear."""
        from scipy.spatial.transform import Rotation, Slerp
        task = self.task
        start = task.prism.get_pose().clone()
        aligned = task.target_pose.add_bias([0, 0, self.cfg.precontact_clearance_mm / 1000])
        self.event("align_above_hole", clearance_mm=self.cfg.precontact_clearance_mm)
        rotations = Rotation.from_quat(np.stack([start.q, aligned.q])[:, [1, 2, 3, 0]])
        slerp = Slerp([0., 1.], rotations)
        for fraction in np.linspace(0, 1, self.cfg.alignment_waypoints + 1)[1:]:
            waypoint = start.clone()
            waypoint.p = start.p * (1 - fraction) + aligned.p * fraction
            waypoint.q = slerp(fraction).as_quat()[[3, 0, 1, 2]]
            target_ee = task.atom.place_actor(task.prism, target_pose=waypoint,
                                              pre_dis=0, dis=0, is_open=False)[0].target_pose
            self.execute_local_pose(target_ee, self.cfg.alignment_physics_steps, "align_above_hole")

    def final_insertion(self):
        task = self.task
        # The part is already aligned after local retreat/correction. There is
        # no second approach or rotation while the part is engaged with the rim.
        target = (task.target_pose.add_bias([0, 0, -.055]) if self.task_name == "insert_hole"
                  else task.target_pose)
        self.event("insert")
        for attempt in range(self.cfg.final_insertion_attempts):
            # The compliant fingers can leave the object short of a planned
            # hand endpoint. Recompute from the observed object pose, keeping
            # the same insertion goal and never replaying the approach.
            self.move(task.atom.place_actor(task.prism, target_pose=target, pre_dis=0, dis=0, is_open=False),
                      "insert", time_dilation_factor=.5)
            if task.check_success() or task.check_early_stop():
                break
            self.event("insert_continue", attempt=attempt + 1,
                       remaining_mm=float(np.linalg.norm(target.p - task.prism.get_pose().p) * 1000))

    def calibrate_contact_baseline(self):
        self.detector = None
        baseline = []
        self.event("contact_baseline")
        for _ in range(self.cfg.stable_window_steps):
            self.task.delay(2, is_save=True)
            baseline.append(self.touch(self.task._get_observations()))
        self.detector = TactileBlockageDetector(self.cfg, baseline)

    def run(self):
        task, cfg = self.task, self.cfg
        if self.task_name == "insert_hole":
            self.align_above_hole()
        reference_frame = (task.target_pose if self.task_name == "insert_hole" else task.hole_pose).clone()
        axis = -reference_frame.R[:, 2]
        entrance_normal = task.hole_pose.R[:, 2]
        desired_blockages = int(task.rng.integers(cfg.min_blockages, cfg.max_blockages + 1))
        self.event("approach", distance_mm=cfg.approach_mm)
        if cfg.approach_mm:
            self.move(task.atom.move_by_displacement(*(axis * cfg.approach_mm / 1000).tolist()),
                      "approach", time_dilation_factor=.5)
        remaining_lateral, remaining_rotation = None, None
        for attempt in range(desired_blockages):
            radius = task.rng.uniform(cfg.lateral_min_mm, cfg.lateral_max_mm) / 1000
            angle = task.rng.uniform(-np.pi, np.pi)
            if cfg.lateral_direction == "transverse":
                # The tilted tube readily self-centers in its tilt plane. Draw
                # both transverse sides with a random +/-30 degree spread so
                # collection spends more attempts on actual blocked contacts.
                angle = task.rng.choice([-np.pi / 2, np.pi / 2]) + task.rng.uniform(-np.pi / 6, np.pi / 6)
            elif cfg.lateral_direction != "uniform":
                raise ValueError(f"Unknown lateral_direction: {cfg.lateral_direction}")
            lateral = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
            rotation = np.zeros(3)
            rotation[cfg.angular_axis] = np.deg2rad(task.rng.uniform(-cfg.angular_max_deg, cfg.angular_max_deg))
            if remaining_lateral is not None:
                lateral, rotation = remaining_lateral, remaining_rotation
            # Keep the current insertion depth. Only lateral position and angle
            # are perturbed; a second blockage starts near the first recovery.
            current = task.prism.get_pose().clone()
            offset = reference_frame.p - current.p
            aligned = current.add_bias(offset - axis * float(offset @ axis), coord="world")
            aligned.q = reference_frame.q.copy()
            perturbed = aligned.add_bias(lateral, coord=reference_frame).add_rotation(rotation)
            self.event("perturb", attempt=attempt, lateral_m=lateral.tolist(), rotation_rad=rotation.tolist())
            self.move(task.atom.place_actor(task.prism, target_pose=perturbed, pre_dis=0, dis=0, is_open=False),
                      "perturb", time_dilation_factor=.3)
            self.calibrate_contact_baseline()
            self.event("push", attempt=attempt)
            blocked = False
            delta = cfg.advance_increment_mm / 1000
            for probe in range(int(np.ceil(cfg.max_probe_mm / cfg.advance_increment_mm))):
                before = task.prism.get_pose().p.copy()
                displacement = axis * delta
                self.probe(displacement)
                after = task.prism.get_pose().p.copy()
                progress = insertion_progress(after - before, axis, entrance_normal)
                if self.detector.blocked(progress, delta):
                    self.blockages += 1
                    self.event("blocked", attempt=attempt, progress_m=progress,
                               axial_progress_m=float((after - before) @ axis),
                               entrance_depth_m=float(-(after - before) @ entrance_normal),
                               requested_m=delta, touch_score=self.detector.recent[-1],
                               threshold=self.detector.threshold)
                    blocked = True
                    break
                if task.check_early_stop() or task.check_success():
                    break
            if not blocked:
                if self.blockages and not task.check_early_stop():
                    # A partial correction can already be sufficient. Preserve
                    # the observed single recovery instead of requiring a second
                    # blockage just because this seed sampled two attempts.
                    self.event("retry_aligned", attempt=attempt)
                    break
                raise RuntimeError("Perturbed attempt did not produce a measured blockage")
            hold_steps = int(task.rng.integers(cfg.blocked_hold_steps_min, cfg.blocked_hold_steps_max + 1))
            self.event("blocked_hold", strategy_steps=hold_steps, extra_press_mm=cfg.blocked_extra_press_mm)
            self.probe(axis * cfg.blocked_extra_press_mm / 1000, physics_steps=2 * hold_steps)
            retreat_mm = cfg.retreat_mm if cfg.retreat_mm is not None else task.rng.uniform(cfg.retreat_min_mm, cfg.retreat_max_mm)
            self.event("retreat", attempt=attempt, distance_mm=float(retreat_mm))
            displacement = -axis * retreat_mm / 1000
            self.move(task.atom.move_by_displacement(*displacement.tolist()), "retreat", time_dilation_factor=.3)
            residual = float(task.rng.uniform(.3, .6)) if attempt + 1 < desired_blockages else 0.0
            remaining_lateral, remaining_rotation = lateral * residual, rotation * residual
            self.event("correct", attempt=attempt, residual_fraction=residual)
            current = task.prism.get_pose().clone()
            offset = reference_frame.p - current.p
            corrected = current.add_bias(offset - axis * float(offset @ axis), coord="world")
            corrected.q = reference_frame.q.copy()
            corrected = corrected.add_bias(remaining_lateral, coord=reference_frame).add_rotation(remaining_rotation)
            self.move(task.atom.place_actor(task.prism, target_pose=corrected, pre_dis=0, dis=0, is_open=False),
                      "correct", time_dilation_factor=.3)
        self.final_insertion()
        task.delay(20, is_save=True)
        success = bool(task.plan_success and task.check_success() and not task.check_early_stop())
        self.event("success" if success else "failed", blockages=self.blockages)
        return success

    def report(self):
        cfg = {**asdict(self.cfg), "seed": int(self.task.cfg.seed)}
        return {"task": self.task_name, "seed": int(self.task.cfg.seed), "config": cfg,
                "hole_pose": self.task.hole_pose.tolist(), "target_pose": self.task.target_pose.tolist(),
                "blockages": self.blockages, "events": self.events, "trace": self.trace,
                "signal": "rgb_marker_change_above_recent_precontact_baseline", "force_available": False,
                "controller_version": "local-recovery-v6"}


def plot_recovery(report_path, output=None):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    report = json.loads(Path(report_path).read_text())
    trace = report["trace"]
    if not trace:
        return
    times = (np.array([r["step"] for r in trace]) - trace[0]["step"]) / 120
    has_grasp = all("in_hand_pose" in row for row in trace)
    fig, axes = plt.subplots(3 if has_grasp else 2, 1, figsize=(11, 8 if has_grasp else 6), sharex=True)
    axes[0].plot(times, [r["touch_score"] for r in trace], label="tactile change")
    axes[0].plot(times, [r["threshold"] or np.nan for r in trace], "--", label="blockage threshold")
    position = np.asarray([r["prism_position"] for r in trace])
    for dim, name in enumerate("xyz"):
        axes[1].plot(times, (position[:, dim] - position[0, dim]) * 1000, label=name)
    event_groups = []
    for event in report["events"]:
        if event["event"] in {"blocked", "blocked_hold", "retreat", "correct", "insert", "success"}:
            t = (event["step"] - trace[0]["step"]) / 120
            axes[0].axvline(t, alpha=.3)
            label = "hold" if event["event"] == "blocked_hold" else event["event"]
            if event_groups and t - event_groups[-1][0] < .15:
                event_groups[-1][1].append(label)
            else:
                event_groups.append((t, [label]))
    for index, (t, labels) in enumerate(event_groups):
        axes[0].annotate("\n".join(labels), xy=(t, .98), xycoords=("data", "axes fraction"),
                         xytext=(-5 if t > times[-1] * .9 else 5, -10 - 20 * (index % 2)),
                         textcoords="offset points", va="top", fontsize=8,
                         ha="right" if t > times[-1] * .9 else "left",
                         bbox={"facecolor": "white", "edgecolor": "none", "alpha": .75},
                         arrowprops={"arrowstyle": "-", "alpha": .4})
    axes[0].set_ylabel("normalized RGB difference")
    axes[1].set_ylabel("object displacement (mm)")
    if has_grasp:
        from scipy.spatial.transform import Rotation
        hand = np.asarray([r["in_hand_pose"] for r in trace])
        axes[2].plot(times, np.linalg.norm(hand[:, :3] - hand[0, :3], axis=1) * 1000,
                     alpha=.5, label="object tip motion in gripper (mm)")
        rotations = Rotation.from_quat(hand[:, [4, 5, 6, 3]]).as_matrix()
        # Track the SAME material point at the initial grasp center. For a long
        # tube, tip motion alone conflates small pivoting with translational slip.
        anchor = -rotations[0].T @ hand[0, :3]
        grasp_point = hand[:, :3] + np.einsum("nij,j->ni", rotations, anchor)
        axes[2].plot(times, np.linalg.norm(grasp_point - grasp_point[0], axis=1) * 1000,
                     label="grasp-point translation (mm)")
        dot = np.abs((hand[:, 3:] * hand[0, 3:]).sum(1))
        angle = 2 * np.rad2deg(np.arccos(np.clip(dot, 0, 1)))
        axes[2].plot(times, angle, label="in-hand rotation from start (deg)")
        axes[2].set_ylabel("grasp change")
    axes[-1].set_xlabel("time (s)")
    for ax in axes:
        ax.legend(loc="upper left")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output or Path(report_path).with_suffix(".png"), dpi=140)
    plt.close(fig)
