"""Retry one existing preview's blocked state, never collect another seed.

Replay recorded physical commands to reconstruct the perturbation context,
record that replay as a new immutable prefix, then restore its own live blocked
snapshot. The replay is explicitly not claimed to be a new WAVE-0 rollout.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import sys
import traceback
from types import SimpleNamespace

import numpy as np
from .paths import WORKSPACE, atomic_json
from .sim_oracle_batch import BatchRunner, batch_trace
from .sim_hole_preview import finish, restore_branch
from .sim_takeover import TeacherStopped


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--trial-result",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    os.chdir(WORKSPACE/"UniVTAC")
    sys.path[:0]=[str(WORKSPACE/"UniVTAC"),str(WORKSPACE/"pi05-univtac/src")]
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p);args=p.parse_args(argv)
    cfgdata=json.loads(args.config.read_text());prior=json.loads(args.trial_result.read_text())
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/"result.json").exists():raise RuntimeError("Retry output already exists")
    if cfgdata.get("training_enabled") is not False or prior["success"]:
        raise ValueError("Only one failed review branch may be retried, without training")
    checkpoint=Path(cfgdata["checkpoint"])
    args.headless,args.enable_cameras,args.livestream=True,True,0
    app=AppLauncher(args).app
    def interrupted(signum,frame):raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM,interrupted)
    task=None;runners=[]
    result=dict(status="running",seed=cfgdata["seed"],task="insert_hole",checkpoint=str(checkpoint),
                training_enabled=False,branches=[],source_trial=str(args.trial_result.resolve()),
                provenance="recorded_WAVE0_and_perturb_commands_replayed_then_same_instance_takeover")
    try:
        import h5py,yaml
        from envs.insert_hole import Task,TaskCfg
        from pi05_univtac.constants import prompt
        from pi05_univtac.contact_restore import capture_task_state
        from pi05_univtac.contact_recovery import RecoveryPlanError
        from .oracle_batch_teacher import rotation
        from .hole_preview_teacher import InContactHoleOracle
        native=TaskCfg();clean=yaml.safe_load((WORKSPACE/"UniVTAC/task_config/clean.yml").read_text())
        native.save_dir,native.decimation=root/"native",1
        native.save_frequency,native.video_frequency,native.render_frequency=2,2,0
        native.tactile_sensor_type,native.obs_data_type="gsmini",clean["observations"]
        native.random_texture=clean.get("random_texture",False)
        native.step_lim,native.reset_time_limit=634,300
        native.sim.device=args.device or "cuda:0"
        task=Task(native,mode="eval");task.reset(seed=int(cfgdata["seed"]),instructions=[prompt("insert_hole")])
        if not task.plan_success:raise RuntimeError("Native initialization failed")
        cfg=SimpleNamespace(budget=317,execute_steps=10,detector=cfgdata.get("detector",{}))
        context=dict(source="wave0",phase="normal",decision_step=0,policy_eligible=False)
        baseline=BatchRunner(task,batch_trace(root/"replayed_failure/trace",checkpoint,context,"insert_hole"),
                             cfg,rotation(task.target_pose.tolist()).apply([0,0,-1]),context)
        runners.append(baseline)
        snapshot=detector=saved_step=None
        perturb=deepcopy(prior["perturbation"])
        before_perturb=None
        with h5py.File(prior["trajectory"],"r") as original:
            source=original["actions/source"].asstr()[:];phases=original["actions/phase"].asstr()[:]
            decisions=original["actions/decision_step"][:]
            for i,command in enumerate(original["actions/clipped_actual_target"]):
                if baseline.step==perturb["start"]:
                    before_perturb=baseline.row()["prism_pose_world"].copy()
                baseline.execute(command,source=str(source[i]),phase=str(phases[i]),decision_step=int(decisions[i]))
                if baseline.step==perturb["stop"]:
                    after=baseline.row()["prism_pose_world"].copy()
                    perturb.update(before_object_pose=before_perturb.tolist(),after_object_pose=after.tolist(),
                        actual_object_delta_world_mm=((after[:3]-before_perturb[:3])*1000).tolist())
                candidate=baseline.detector.first_confirmed
                if candidate is None:
                    snapshot=detector=saved_step=None
                elif candidate>=perturb["stop"]+10 and candidate!=saved_step and not baseline.success:
                    snapshot,detector,saved_step=capture_task_state(task),deepcopy(baseline.detector),candidate
                if baseline.success or task.check_early_stop():break
        replay=finish(baseline,"self_corrected",root/"replayed_failure",perturb=perturb,
                      reason="recorded_failed_rollout_replay",source_trial=str(args.trial_result.resolve()),
                      provenance=result["provenance"])
        result["branches"].append(replay)
        if baseline.success or snapshot is None:
            raise RuntimeError("Recorded replay did not preserve a sustained blocked candidate")
        recovered,checked=restore_branch(baseline,snapshot,detector,saved_step,root/"recovery",checkpoint)
        runners.append(recovered)
        teacher_config=dict(cfgdata["teacher"],rotation_tolerance_deg=2.)
        teacher=InContactHoleOracle(recovered,"insert_hole",teacher_config)
        reason=""
        try:teacher.run()
        except (TeacherStopped,RecoveryPlanError) as exc:reason=str(exc)
        c=detector.candidate
        timing=dict(t_contact=c["t_contact"],t_tactile=c["t_tactile"],t_decision=saved_step,
                    t_recovery=saved_step,tactile_window_indices=c["input_tactile_indices"])
        recovery=finish(recovered,"oracle_recovery",root/"recovery",perturb=perturb,takeover=saved_step,
            events=teacher.events,timing=timing,physical_restore=checked,reason=reason,
            perturbed_reference=replay["trajectory"],teacher_config=teacher.config,
            teacher_revision=teacher.revision,provenance=result["provenance"],source_trial=str(args.trial_result.resolve()))
        result["branches"].append(recovery);result["status"]="completed"
    except BaseException as exc:
        result.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "error",
                      error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
    finally:
        for runner in runners:
            if runner.trace.file is not None:runner.trace.finish(task,result["status"],result.get("error",""))
        atomic_json(root/"result.json",result)
        if task is not None:task.close()
        app.close()
    return int(result["status"]!="completed")


if __name__=="__main__":raise SystemExit(main())
