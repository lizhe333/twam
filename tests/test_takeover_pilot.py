"""Causal takeover, data leakage, replay and restart invariants without Isaac."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from contact_belief_wam.takeover import (
    BlockageDetector, PilotLedger, TakeoverConfig, compare_replay,
    feedback_decision, policy_prefix, seed_overlap, supervision_ranges, tactile_indices,
    finalize_supervision,
)
from contact_belief_wam.takeover_report import (point_triangle_distance, temporal_audit,
                                               precontact_evidence, progress_proxy_audit)

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE/"pi05-univtac/src"))


def detector_run(contact_at=8, move=False, spike=False):
    detector = BlockageDetector()
    for t in range(21):
        level = 40 if (t >= contact_at and (not spike or t == contact_at)) else 0
        z = .2-t*.0001 if move else .2
        detector.observe(np.full((2,4,4,3), level, np.uint8), [0,0,z],
                         None if t==0 else [0,0,z-.0005])
    return detector


def test_confirms_first_or_second_chunk_not_three_chunks():
    assert detector_run(contact_at=8).first_confirmed == 10
    assert detector_run(contact_at=10).first_confirmed == 20


def test_normal_contact_and_spikes_do_not_trigger():
    assert detector_run(contact_at=8, move=True).first_confirmed is None
    assert detector_run(contact_at=8, spike=True).first_confirmed is None


def test_future_observations_cannot_change_earlier_decision():
    d = detector_run()
    first = dict(d.events[0])
    for t in range(21,41):
        d.observe(np.full((2,4,4,3),255,np.uint8), [0,0,.2], [0,0,.1995])
    assert d.events[0] == first and d.first_confirmed == 10


def test_feedback_is_progress_conditioned_not_touch_amplitude_alone():
    common = dict(requested_mm=.5,touch_before=.1,touch_after=.2,threshold=.01,grip_stable=True)
    assert feedback_decision(achieved_mm=.4,**common)[0] == "continue"
    assert feedback_decision(achieved_mm=.02,**common)[0] == "withdraw"
    common.update(touch_after=.001)
    assert feedback_decision(achieved_mm=.02,**common)[0] == "repeat"
    assert feedback_decision(achieved_mm=.02,repeated=True,**common)[0] == "withdraw"


def test_flow_input_never_contains_excluded_future_actions():
    commands = np.arange(160*8,dtype=np.float32).reshape(160,8)
    ranges = [dict(start=110,stop=120)]
    targets, mask = policy_prefix(commands,110,ranges)
    modified = commands.copy();modified[120:] = 999999
    after, after_mask = policy_prefix(modified,110,ranges)
    np.testing.assert_array_equal(targets,after)
    np.testing.assert_array_equal(mask,after_mask)
    assert mask.sum() == 10
    before, mask = policy_prefix(commands,80,ranges)
    assert not mask.any() and not before.any()
    # The original context and actual action history were never modified.
    assert commands[100,0] == 800
    np.testing.assert_array_equal(tactile_indices(110),[104,106,108,110])


def test_no_label_crosses_a_feedback_decision_or_exploration():
    meta = [dict(decision_step=t//10*10,phase="insert",policy_eligible=True) for t in range(30)]
    for r in meta[10:20]:r["policy_eligible"] = False
    ranges = supervision_ranges(meta,recovery_success=True)
    assert [(r["start"],r["stop"]) for r in ranges] == [(0,10),(20,30)]
    assert supervision_ranges(meta) == []


def test_success_and_failure_flags_agree_with_training_ranges():
    meta = [dict(decision_step=0, phase="natural", policy_eligible=False) for _ in range(10)]
    assert finalize_supervision(meta, normal_success=True) == [dict(start=0,stop=10,reason="normal_success")]
    assert all(row["policy_eligible"] for row in meta)
    assert finalize_supervision(meta) == []
    assert not any(row["policy_eligible"] for row in meta)


def test_restart_does_not_refund_interrupted_attempt_or_remove_review_stop(tmp_path):
    c = TakeoverConfig(); identity = {"checkpoint":"frozen"}
    first = PilotLedger(tmp_path,c,identity)
    first.reserve("natural",3000000);first.close()
    resumed = PilotLedger(tmp_path,c,identity)
    assert resumed.counts()["natural"] == 1
    assert resumed.state["attempts"][0]["status"] == "interrupted"
    with pytest.raises(RuntimeError,match="consumed"):
        resumed.reserve("natural",3000000)
    resumed.pause_for_review("preview_ready");resumed.close()
    again = PilotLedger(tmp_path,c,identity)
    with pytest.raises(RuntimeError,match="review"):
        again.reserve("natural",3000001)
    again.close()
    with pytest.raises(ValueError,match="identity"):
        PilotLedger(tmp_path,replace(c,natural_limit=19),identity)


def test_takeover_quota_survives_restart(tmp_path):
    c=TakeoverConfig(); ledger=PilotLedger(tmp_path,c,{})
    for seed in range(3000000,3000010):
        row=ledger.reserve("recovery",seed);ledger.finish(row,dict(status="failed"))
    ledger.close();ledger=PilotLedger(tmp_path,c,{})
    with pytest.raises(RuntimeError,match="quota"):
        ledger.reserve("recovery",3000010)
    ledger.close()


def observation():
    row={name:np.array([0,0,0,1,0,0,0],float) for name in
         ("ee_pose_world","gripper_pose_world","prism_pose_world","slot_pose_world")}
    row["joint_q"]=np.zeros(8)
    for side in ("left","right"):
        row[f"{side}_marker"]=np.zeros((2,4,2))
        row[f"{side}_depth"]=np.ones((4,4))
        row[f"{side}_rgb_marker"]=np.zeros((4,4,3),np.uint8)
    for name in ("head_rgb","wrist_rgb"):row[name]=np.zeros((4,4,3),np.uint8)
    return row


def test_relative_grasp_rejects_individually_tolerable_world_pose_changes():
    a,b=observation(),observation()
    b["gripper_pose_world"][0]=.00004
    b["prism_pose_world"][0]=-.00004
    report=compare_replay([a],[b],[],[])
    assert not report["passed"]
    assert any("object_in_gripper" in x for x in report["violations"])


def test_replay_rejects_tactile_mismatch():
    a,b=observation(),observation();b["left_rgb_marker"][:]=10
    assert not compare_replay([a],[b],[],[])["passed"]


def test_point_triangle_distance_face_edge_and_degenerate():
    triangle=np.array([[[0,0,0],[1,0,0],[0,1,0]]],float)
    assert point_triangle_distance([[.2,.2,2]],triangle)==pytest.approx(2)
    assert point_triangle_distance([[2,0,0]],triangle)==pytest.approx(1)
    assert point_triangle_distance([[0,0,1]],np.zeros((1,3,3)))==pytest.approx(1)


def test_timing_rejects_signal_after_action_and_requires_current_latent():
    result=dict(takeover_step=20,detector=dict(scores=[0.]*12+[.1]*19,tactile_threshold=.01),
                action_metadata=[dict(source="replay")]*20+[dict(source="teacher")]*10)
    dist=np.r_[np.ones(10)*.005,np.zeros(21)]
    delta=np.r_[np.zeros(12),np.ones(19)*.1]
    assert temporal_audit(result,dist,delta,.01)["passed"]
    delta[20]=0
    assert not temporal_audit(result,dist,delta,.01)["passed"]
    assert not temporal_audit(result,np.ones(31),delta,.01)["passed"]


def test_seed_ranges_and_nested_manifests(tmp_path):
    import json
    (tmp_path/"manifest.json").write_text(json.dumps(dict(seed_start=2999998,episodes=5)))
    audit=seed_overlap([tmp_path],range(3000000,3000020))
    assert audit["conflicts"][0]["seeds"] == [3000000,3000001,3000002]


def test_seed_audit_covers_numeric_episode_paths_and_unlisted_raw_data(tmp_path):
    import json,h5py
    (tmp_path/"train.jsonl").write_text(json.dumps(dict(path="hdf5/recovery/insert_HDMI/3000002.hdf5"))+"\n")
    with h5py.File(tmp_path/"3000003.hdf5","w"):
        pass
    audit=seed_overlap([tmp_path],range(3000000,3000020))
    assert {s for c in audit['conflicts'] for s in c['seeds']} == {3000002,3000003}
    assert audit['hdf_episodes_checked']==1


def test_existing_wave0_server_passes_wrist_to_inference_and_refuses_missing_view():
    path=WORKSPACE/"wave0/stage1-sources/dual-view/scripts/policy-server.py"
    spec=importlib.util.spec_from_file_location("pilot_wave_server",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    received=[]
    policy=SimpleNamespace(metadata=dict(camera="head_wrist",task="insert_HDMI",prompt="insert"),
                           infer=lambda image,state,**kwargs:received.append((image,kwargs)) or np.zeros((50,8)))
    msg=dict(task="insert_HDMI",prompt="insert",image=np.ones((4,4,3),np.uint8),state=np.zeros(8))
    with pytest.raises(KeyError):module.infer_observation(policy,msg)
    msg["wrist_image"]=np.full((4,4,3),99,np.uint8)
    module.infer_observation(policy,msg)
    np.testing.assert_array_equal(received[0][1]["wrist_rgb"],msg["wrist_image"])


def test_recorded_controller_preserves_actions_rng_and_replan_cadence():
    sys.path.insert(0,str(WORKSPACE/"UniVTAC"))
    from pi05_univtac.sim_policy import ChunkController
    from policy.Wave0Wam.deploy_policy import RecordedController
    task=SimpleNamespace(_robot_manager=SimpleNamespace(
        robot=SimpleNamespace(data=SimpleNamespace(joint_pos_limits=np.array([[[-1.,1.]]*7]))),
        _arm_ids=np.arange(7),gripper_max_qpos=.04))
    obs=dict(embodiment=dict(joint=np.zeros(8)),observation={
        "head":dict(rgb=np.zeros((4,4,3),np.uint8)),"wrist":dict(rgb=np.ones((4,4,3),np.uint8))})
    runs=[]
    for cls in (ChunkController,RecordedController):
        np.random.seed(971)
        calls=[]
        def infer(message):
            calls.append(message)
            actions=np.random.uniform(-.01,.01,(50,8)).astype(np.float32)
            actions[:,7]=.02
            return dict(actions=actions)
        controller=cls(SimpleNamespace(infer=infer),"insert_HDMI",arm_clip_tolerance=.02)
        actions=[controller.next_action(task,obs) for _ in range(21)]
        runs.append((np.stack(actions),np.random.get_state(),len(calls)))
    np.testing.assert_array_equal(runs[0][0],runs[1][0])
    np.testing.assert_array_equal(runs[0][1][1],runs[1][1][1])
    assert runs[0][1][2:] == runs[1][1][2:]
    assert runs[0][2] == runs[1][2] == 3


def test_real_hdf_recorder_multiple_controller_strings_and_n_plus_one(tmp_path,monkeypatch):
    import h5py
    from contact_belief_wam.sim_takeover import make_trace
    from pi05_univtac import contact_trace
    row=observation();row["root_pose_world"]=np.array([0,0,0,1,0,0,0.])
    monkeypatch.setattr(contact_trace,"snapshot_observation",lambda task,obs:row)
    context=dict(source="wave0",phase="natural",decision_step=0,policy_eligible=False)
    trace=make_trace(tmp_path,"checkpoint",context)
    trace._observation_defaults=lambda task,obs:dict(row,physics_step=task.step_count,observation_valid=True)
    trace.fk_fn=lambda *args:np.zeros(3)
    task=SimpleNamespace(cfg=SimpleNamespace(seed=3000000),step_count=0,take_action_cnt=0,_get_observations=lambda:{})
    trace.append_observation(task,{})
    for i in range(3):
        task.step_count+=2;task.take_action_cnt+=2
        if i==2:context.update(source="teacher",phase="feedback_withdraw")
        trace.append_action(task,{},raw_target=np.zeros(8),clipped_target=np.zeros(8),
            chunk_index=0,replan=i==0,physics_step_begin=2*i,action_count_begin=2*i,success=True)
    trace.finish(task,"failed")
    with h5py.File(trace.path,'r') as f:
        assert len(f['observations/joint_q'])==4
        assert len(f['actions/clipped_actual_target'])==3
        assert f['actions/source'][:].tolist()==[b'wave0',b'wave0',b'teacher']
        assert f['actions/phase'][-1]==b'feedback_withdraw'


def test_teacher_does_not_score_free_space_after_unloading_as_recovery():
    from contact_belief_wam.sim_takeover import Runner
    runner=object.__new__(Runner)
    runner.cfg=TakeoverConfig();runner.trace=SimpleNamespace(action_count=20)
    runner.success=False;runner.teacher_events=[];runner.metadata=[]
    runner.detector=SimpleNamespace(scores=[.05],threshold=.01)
    pose=np.array([0.,0.,.2,1.,0.,0.,0.]);steps=[]
    runner.row=lambda:dict(ee_pose_world=pose.copy(),observed_fk_world=pose[:3].copy(),joint_q=np.r_[np.zeros(7),.02])
    def move(target,*,phase,eligible,duration=10,gripper=None):
        steps.append(dict(phase=phase,start=pose.copy(),target=target.copy()))
        pose[:]=target;runner.trace.action_count+=duration
        if phase=='probe':runner.success=True
    runner.move=move
    runner.run_teacher(3000001)
    probe=next(row for row in steps if row['phase']=='probe')
    assert probe['start'][2]==pytest.approx(.2)
    assert probe['target'][2]==pytest.approx(.1995)
    assert [row['phase'] for row in steps][:3]==['unload','exploration','exploration_approach']


def test_already_changed_grasp_is_not_new_contact_evidence():
    images=np.zeros((31,4,4,3),np.uint8);images[8:]=200
    z=np.ones((31,8,4),np.float32);z[8:,:,0]=10
    handle={'observations':{f'{s}_rgb_marker':images for s in ('left','right')}}
    result=dict(takeover_step=30,detector=dict(scores=[0.]*8+[.4]*23,tactile_threshold=.01),
                action_metadata=[dict(source='replay')]*30+[dict(source='teacher')])
    distance=np.r_[np.ones(20)*.005,np.zeros(11)]
    def audit():
        evidence=precontact_evidence(handle,z,20,30)
        return temporal_audit(result,distance,np.array(evidence['latent_changes']),evidence['latent_threshold'],
                              raw_scores=evidence['raw_changes'],raw_threshold=evidence['raw_threshold'])
    assert not audit()['passed']
    images[20:]=240;z[20:,:,0]=1;z[20:,:,1]=10
    assert audit()['passed']


def test_audit_exposes_arm_tracking_despite_stalled_object_and_slip():
    ee=np.zeros((21,3));ee[:,2]=.2-np.arange(21)*.0001
    obj=np.zeros((21,7));obj[:,2]=.02
    slot=np.zeros_like(obj)
    grasp=np.zeros((21,7));grasp[:,2]=.01-np.arange(21)*.0001
    events=[dict(step=t,progress_ratio=1.,tactile_sustained=True,status="normal") for t in (10,20)]
    handle={'observations':dict(observed_fk_world=ee,prism_pose_world=obj,
                               slot_pose_world=slot,object_in_gripper_pose_world=grasp)}
    result=dict(detector=dict(events=events))
    check=progress_proxy_audit(handle,result,np.zeros(21))
    assert not check['usable_for_this_failure'] and check['audit_only']
    assert not check['online_ground_truth_used']
    assert check['contradictions'][0]['ee_progress_mm'] == pytest.approx(1.)
    # Real object progress with the same normal-contact signal is not rejected.
    obj[:,2]=.02-np.arange(21)*.0001
    grasp[:,2]=.01
    assert progress_proxy_audit(handle,result,np.zeros(21))['usable_for_this_failure']
