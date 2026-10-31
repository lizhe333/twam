import numpy as np
import pytest

from contact_belief_wam.hole_preview import label_ranges, validate_preview_labels


def metadata():
    return (["wave0"]*20 + ["perturb"]*10 + ["wave0"]*30,
            ["normal"]*20 + ["perturb"]*10 + ["wave0_after_perturb"]*30,
            np.repeat(np.arange(0, 60, 10), 10))


def test_perturbation_never_enters_labels_or_fifty_action_input():
    sources, phases, decisions = metadata()
    ranges, flags = label_ranges(sources, phases, decisions, success=True,
        kind="self_corrected", perturb_start=20, perturb_stop=30)
    assert [(r["start"], r["stop"]) for r in ranges] == [(0,10),(10,20),(30,40),(40,50),(50,60)]
    assert not flags[20:30].any()
    assert flags.sum() == 50
    assert validate_preview_labels(sources, ranges, 20, 30)["perturb_bc_labels"] == 0


def test_failed_rollout_is_context_only_and_recovery_excludes_stalled_prefix():
    sources, phases, decisions = metadata()
    ranges, flags = label_ranges(sources, phases, decisions, success=False,
        kind="self_corrected", perturb_start=20, perturb_stop=30)
    assert ranges == [] and not flags.any()
    sources[40:] = ["oracle"]*20
    phases[40:50], phases[50:] = ["oracle_unload"]*10, ["oracle_insert"]*10
    events = [dict(step=40,stop=50,bc_eligible=True),dict(step=50,stop=60,bc_eligible=False)]
    ranges, flags = label_ranges(sources, phases, decisions, success=True,
        kind="oracle_recovery", perturb_start=20, perturb_stop=30, takeover=40,teacher_events=events)
    assert flags.sum() == 10 and not flags[:40].any() and not flags[50:].any()
    validate_preview_labels(sources, ranges, 20, 30)


def test_mislabeled_perturbation_and_mid_decision_interval_rejected():
    sources, phases, decisions = metadata()
    with pytest.raises(ValueError, match="Perturbation entered"):
        validate_preview_labels(sources,[dict(start=0,stop=60)])
    with pytest.raises(ValueError, match="decision observation"):
        label_ranges(sources, phases, decisions, success=True, kind="oracle_recovery",
            perturb_start=20,perturb_stop=30,takeover=40,
            teacher_events=[dict(step=42,stop=50,bc_eligible=True)])


def test_preview_export_uses_actual_loader_and_keeps_perturb_context(tmp_path, monkeypatch):
    import json
    import h5py
    from test_oracle_batch_export import make_trace, Encoder
    from contact_belief_wam import hole_preview_export as preview
    from contact_belief_wam import oracle_batch_export
    checkpoint=tmp_path / "checkpoint"; checkpoint.mkdir()
    (checkpoint/"deployment.json").write_text(json.dumps(dict(camera="head_wrist",prompt="insert tube")))
    stats={key:dict(q01=[-2.]*8,q99=[3.]*8) for key in ("state","actions")}
    for name in ("norm_stats.json","wm_norm_stats.json"):
        (checkpoint/name).write_text(json.dumps(stats))
    trace=make_trace(tmp_path/"trace.h5","insert_hole",checkpoint,success=True,recovery=False)
    sources=["wave0"]*4+["perturb"]*4+["wave0"]*4
    with h5py.File(trace,"r+") as f:
        phases=["normal"]*4+["perturb"]*4+["wave0_after_perturb"]*4
        f["actions/phase"][:]=phases
        f.create_dataset("actions/source",data=sources,dtype=h5py.string_dtype())
        poses=np.zeros((13,7));poses[:,3]=1
        f["observations/target_pose_world"]=poses
        poses[8,1]=.001
        f["observations/prism_pose_world"]=poses
        ranges,_=label_ranges(sources,phases,f["actions/decision_step"][:],success=True,
            kind="self_corrected",perturb_start=4,perturb_stop=8)
    result=tmp_path/"result.json"
    result.write_text(json.dumps(dict(kind="self_corrected",success=True,trajectory=str(trace),
        strategy_actions=12,demonstrator="wave0",policy_supervision_ranges=ranges,
        perturbation=dict(start=4,stop=8,offset_target_frame_mm=[0,1],
                          actual_object_delta_world_mm=[0,1,0]))))
    monkeypatch.setattr(oracle_batch_export,"_video",lambda *a,**k:None)
    monkeypatch.setattr(preview,"render_video",lambda *a,**k:None)
    row=preview.export_sample(result,tmp_path/"review",seed=7,checkpoint=checkpoint,encoder=Encoder())
    checked=json.loads((__import__('pathlib').Path(row["review_page"]).parent/"validation.json").read_text())
    assert checked["loader"]["history_continuous"]
    assert checked["labels"]["perturb_bc_labels"]==0
    assert checked["loader"]["supervised_anchors"]==8
    assert row["collection_origin"]=="controlled_robot_perturbation"
    assert row["training_enabled"] is False


def test_contact_alignment_remains_bounded_and_preserves_measured_grasp():
    from test_oracle_batch_teacher import Runner, relative_pose, rotation
    from contact_belief_wam.hole_preview_teacher import InContactHoleOracle
    runner=Runner("insert_hole",pitch=30.);runner.contact=.0003
    before=runner.row();grasp=relative_pose(before["ee_pose_world"],before["prism_pose_world"])
    teacher=InContactHoleOracle(runner,"insert_hole",dict(rotation_tolerance_deg=2.))
    teacher.unload();teacher.orient()
    assert teacher.events[0]["object_progress_mm"]>0
    assert teacher.rotation_travel_deg<=30
    assert all(e["actual_actions"]==2 for e in teacher.events[1:])
    after=runner.row()
    assert np.allclose(relative_pose(after["ee_pose_world"],after["prism_pose_world"]),grasp)
    assert np.rad2deg((rotation(teacher.target)*rotation(after["prism_pose_world"]).inv()).magnitude())<=2.00001


def test_contact_teacher_does_not_rotate_when_unloading_has_no_progress():
    from test_oracle_batch_teacher import Runner
    from contact_belief_wam.hole_preview_teacher import InContactHoleOracle
    from contact_belief_wam.sim_takeover import TeacherStopped
    runner=Runner("insert_hole",pitch=30.)
    def stuck(target,**kwargs):runner.step+=kwargs["duration"]
    runner.move=stuck
    teacher=InContactHoleOracle(runner,"insert_hole")
    with pytest.raises(TeacherStopped,match="no_object_progress"):teacher.run()
    assert len(teacher.events)==1 and not teacher.events[0]["bc_eligible"]
