import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from contact_belief_wam.mixed_recovery import MixedLedger, rank_trials, stage_for_distance, load_config


def settings():
    return dict(tasks={"insert_HDMI": dict(seed_start=6000000, natural_limit=6, perturb_limit=5,
        recovery_limit=3, max_perturb_per_seed=4, normal_target=2, self_corrected_target=2,
        recovery_target=3, perturb_amplitudes_mm=[.5,1.,2.])})


def ledger_at(path, config=None):
    return MixedLedger(path, config or settings(), dict(checkpoints={}, source_hashes={}))


def row(seed, kind, *, imported=False, origin="controlled_robot_perturbation"):
    return dict(task="insert_HDMI", seed=seed, kind=kind, trace_sha256=f"{seed}-{kind}",
                imported_existing=imported, collection_origin=origin)


def test_real_config_uses_successful_teacher_and_review_only_protocol():
    cfg = load_config(Path(__file__).parents[1]/"configs/recovery_mixed_90.yaml")
    assert not cfg["training_enabled"]
    assert cfg["tasks"]["insert_hole"]["teacher_revision"] == "hole-bounded-contact-alignment-v1"
    assert cfg["tasks"]["insert_hole"]["teacher"]["rotation_tolerance_deg"] == 2.
    assert sum(t["normal_target"]+t["self_corrected_target"]+t["recovery_target"] for t in cfg["tasks"].values()) == 90


def test_reservations_survive_restart_and_failed_branch_is_not_refunded(tmp_path):
    ledger = ledger_at(tmp_path)
    ledger.reserve_seed("insert_HDMI",6000000)
    trial = rank_trials(settings(),ledger.snapshot(),"insert_HDMI",6000000)[0]
    for i in range(4):
        t = dict(trial, direction=str(i))
        assert ledger.reserve_branch("insert_HDMI",6000000,"perturb",trial=t)
    assert ledger.reserve_branch("insert_HDMI",6000000,"perturb",trial=dict(trial,direction="extra")) is None
    assert ledger.reserve_branch("insert_HDMI",6000000,"oracle")
    assert ledger.reserve_branch("insert_HDMI",6000000,"oracle") is None
    ledger.finish_seed("insert_HDMI",6000000,dict(status="interrupted"))
    resumed = MixedLedger(tmp_path)
    with pytest.raises(ValueError,match="already consumed"):
        resumed.reserve_seed("insert_HDMI",6000000)
    c = resumed.counts(resumed.snapshot(),"insert_HDMI")
    assert (c["natural"],c["perturb"],c["recovery"]) == (1,4,1)
    resumed.reserve_seed("insert_HDMI",6000001)
    assert resumed.reserve_branch("insert_HDMI",6000001,"perturb",trial=trial)
    assert resumed.reserve_branch("insert_HDMI",6000001,"perturb",trial=dict(trial,direction="last")) is None
    assert resumed.next_seed() is None
    assert resumed.snapshot()["task_status"]["insert_HDMI"] == "allowance_exhausted"


def test_resume_accepts_checkpoint_directory_alias_but_not_changed_weights_or_config(tmp_path):
    actual, alias = tmp_path / 'organized', tmp_path / 'old-name'
    actual.mkdir()
    alias.symlink_to(actual, target_is_directory=True)
    cfg = settings()
    cfg['tasks']['insert_HDMI']['checkpoint'] = str(alias)
    identity = dict(checkpoints=dict(insert_HDMI=dict(checkpoint=str(alias),
        hashes={str(alias / 'ema.safetensors'): 'same-weights'}, deployment=dict(use_ema=True))),
        source_hashes={})
    root = tmp_path / 'campaign'
    original = MixedLedger(root, cfg, identity).snapshot()
    normalized_config = json.loads(json.dumps(cfg))
    normalized_config['tasks']['insert_HDMI']['checkpoint'] = str(actual)
    normalized_identity = dict(checkpoints=dict(insert_HDMI=dict(checkpoint=str(actual),
        hashes={str(actual / 'ema.safetensors'): 'same-weights'}, deployment=dict(use_ema=True))),
        source_hashes={})
    resumed = MixedLedger(root, normalized_config, normalized_identity).snapshot()
    assert resumed['config'] == original['config']
    assert resumed['identity'] == original['identity']
    assert resumed['revisions'][-1] == normalized_identity
    normalized_identity['checkpoints']['insert_HDMI']['hashes'][str(actual / 'ema.safetensors')] = 'changed'
    with pytest.raises(ValueError, match='Checkpoint identity changed'):
        MixedLedger(root, normalized_config, normalized_identity)
    normalized_config['tasks']['insert_HDMI']['natural_limit'] += 1
    with pytest.raises(ValueError, match='configuration changed'):
        MixedLedger(root, normalized_config, identity)


def test_duplicate_import_cannot_inflate_counts_and_gate_requires_new_samples(tmp_path):
    ledger = ledger_at(tmp_path)
    imported = row(1,"normal",imported=True,origin="unperturbed")
    ledger.record_export(imported);ledger.record_export(imported)
    assert ledger.counts(ledger.snapshot(),"insert_HDMI")["success"]["normal"] == 1
    assert ledger.counts(ledger.snapshot(),"insert_HDMI")["natural"] == 0
    changed = dict(imported,trace_sha256="different")
    with pytest.raises(ValueError,match="cannot replace"):
        ledger.record_export(changed)
    ledger.record_export(row(6000000,"normal",origin="unperturbed"))
    ledger.record_export(row(6000000,"self_corrected"))
    for seed in (6000000,6000001):ledger.record_export(row(seed,"oracle_recovery",origin="natural_failure"))
    assert ledger.snapshot()["task_status"]["insert_HDMI"] == "validating"
    ledger.record_export(row(6000002,"oracle_recovery"))
    assert ledger.snapshot()["task_status"]["insert_HDMI"] == "collecting"


def test_scheduling_varies_cells_and_does_not_change_numpy_rng(tmp_path):
    ledger=ledger_at(tmp_path);ledger.reserve_seed("insert_HDMI",6000000)
    np.random.seed(117);before=np.random.get_state()
    first=rank_trials(settings(),ledger.snapshot(),"insert_HDMI",6000000)[0]
    ledger.reserve_branch("insert_HDMI",6000000,"perturb",trial=first)
    second=rank_trials(settings(),ledger.snapshot(),"insert_HDMI",6000000)[0]
    after=np.random.get_state()
    assert first != second and first["stage"] != second["stage"]
    assert np.array_equal(before[1],after[1]) and before[2:] == after[2:]
    assert stage_for_distance(.0005) is None
    assert stage_for_distance(.001) == "near"
    assert stage_for_distance(.002) == "approach"
    assert stage_for_distance(.007) is None


def test_natural_recovery_labels_need_no_synthetic_perturbation():
    from contact_belief_wam.hole_preview import label_ranges,validate_preview_labels
    sources=["wave0"]*10+["oracle"]*4
    phases=["normal"]*10+["oracle_unload"]*4
    decisions=[0]*10+[10]*4
    ranges,flags=label_ranges(sources,phases,decisions,success=True,kind="oracle_recovery",
        takeover=10,teacher_events=[dict(step=10,stop=14,bc_eligible=True)])
    assert not flags[:10].any() and flags[10:].all()
    assert validate_preview_labels(sources,ranges)["perturb_bc_labels"] == 0


@pytest.mark.parametrize("task",["insert_HDMI","insert_hole"])
def test_task_specific_self_correction_uses_measured_object_motion(tmp_path,task):
    from contact_belief_wam.mixed_recovery_export import self_correction_check
    trace=tmp_path/"trace.h5"
    poses=np.zeros((31,7));poses[:,3]=1
    with h5py.File(trace,"w") as f:
        f["observations/target_pose_world"]=poses
        f["observations/slot_pose_world"]=poses
        obj=poses.copy();obj[20,1]=.002;obj[30,1]=.0002
        f["observations/prism_pose_world"]=obj
    result=dict(success=True,strategy_actions=30,trajectory=str(trace),perturbation=dict(
        start=10,stop=20,offset_target_frame_mm=[0,2],actual_object_delta_world_mm=[0,1.8,0]))
    assert self_correction_check(result,task)["passed"]
    result["perturbation"]["actual_object_delta_world_mm"]=[0,.01,0]
    assert not self_correction_check(result,task)["passed"]


def test_hdmi_export_uses_real_loader_and_keeps_normal_decision_boundaries(tmp_path,monkeypatch):
    from test_oracle_batch_export import make_trace,Encoder
    from contact_belief_wam import hole_preview_export as preview,oracle_batch_export
    from contact_belief_wam.hole_preview import label_ranges
    cp=tmp_path/"checkpoint";cp.mkdir()
    (cp/"deployment.json").write_text(json.dumps(dict(camera="head_wrist",prompt="insert HDMI")))
    stats={key:dict(q01=[-1.]*8,q99=[2.]*8) for key in ("state","actions")}
    for name in ("norm_stats.json","wm_norm_stats.json"):(cp/name).write_text(json.dumps(stats))
    trace=make_trace(tmp_path/"trace.h5","insert_HDMI",cp,success=True,recovery=False)
    with h5py.File(trace,"r+") as f:
        f.create_dataset("actions/source",data=["wave0"]*12,dtype=h5py.string_dtype())
        ranges,_=label_ranges(["wave0"]*12,f["actions/phase"].asstr()[:],f["actions/decision_step"][:],
                              success=True,kind="normal")
    result=tmp_path/"result.json";result.write_text(json.dumps(dict(task="insert_HDMI",kind="normal",
        success=True,trajectory=str(trace),strategy_actions=12,demonstrator="wave0",policy_supervision_ranges=ranges)))
    monkeypatch.setattr(oracle_batch_export,"_video",lambda *a,**k:None)
    monkeypatch.setattr(preview,"render_video",lambda *a,**k:None)
    row=preview.export_sample(result,tmp_path/"out",seed=1,checkpoint=cp,encoder=Encoder())
    assert row["task"] == "insert_HDMI" and row["group_id"] == "insert_HDMI/1"
    validation=json.loads((Path(row["review_page"]).parent/"validation.json").read_text())
    assert validation["loader"]["history_continuous"]
    assert validation["loader"]["excluded_future_poison_checks"] >= 4
    assert len(row["policy_supervision_ranges"]) == 4
