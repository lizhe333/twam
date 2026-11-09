from __future__ import annotations

import copy
import json
import multiprocessing
from pathlib import Path

import pytest

from contact_belief_wam.oracle_batch_campaign import (
    BatchLedger, cleanup_finished_attempt, live_owned_workers, load_config, refresh_index_safely, seed_audit,
)


def config(root, *, natural_limit=4, recovery_limit=2, target=2):
    return dict(output_root=str(root), tasks={
        "insert_HDMI": dict(seed_start=4000000, natural_limit=natural_limit, recovery_limit=recovery_limit,
                            recovery_target=target, normal_target=2, budget=600),
        "insert_hole": dict(seed_start=5000000, natural_limit=natural_limit, recovery_limit=recovery_limit,
                            recovery_target=target, normal_target=2, budget=317),
    })


def make_ledger(tmp_path, **kwargs):
    root = tmp_path / "campaign"
    return BatchLedger(root, config(root, **kwargs), dict(checkpoint="immutable", source="immutable"))


def failed(ledger, task="insert_HDMI", seed=4000000):
    ledger.reserve_natural(task, seed, ledger.root / task / str(seed))
    ledger.record_natural(task, seed, dict(success=False, status="failed", takeover_step=40))


def _child_reserve(root, task, seed, queue):
    try:
        BatchLedger(root).reserve_recovery(task, seed)
        queue.put("reserved")
    except Exception as exc:
        queue.put(type(exc).__name__)


def test_child_can_reserve_and_duplicate_is_atomic(tmp_path):
    ledger = make_ledger(tmp_path)
    failed(ledger)
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    children = [ctx.Process(target=_child_reserve, args=(ledger.root, "insert_HDMI", 4000000, queue)) for _ in range(2)]
    for child in children:
        child.start()
    for child in children:
        child.join(timeout=15)
        assert not child.is_alive(), "Short transactions must not leave a parent-held lock"
        assert child.exitcode == 0
    assert sorted([queue.get(timeout=2), queue.get(timeout=2)]) == ["RuntimeError", "reserved"]
    assert ledger.counts("insert_HDMI")["recovery"] == 1


def test_crash_keeps_both_reservations_and_cannot_retry(tmp_path):
    ledger = make_ledger(tmp_path)
    failed(ledger)
    ledger.reserve_recovery("insert_HDMI", 4000000)
    reopened = BatchLedger(ledger.root)
    assert reopened.snapshot()["attempts"][0]["status"] == "running"  # Worker open does not mark parent interrupted.
    reopened.recover_interrupted()
    assert reopened.counts("insert_HDMI")["natural"] == reopened.counts("insert_HDMI")["recovery"] == 1
    assert reopened.snapshot()["attempts"][0]["status"] == "interrupted"
    with pytest.raises(RuntimeError, match="consumed"):
        reopened.reserve_natural("insert_HDMI", 4000000, ledger.root / "again")
    with pytest.raises(RuntimeError):
        reopened.reserve_recovery("insert_HDMI", 4000000)
    assert reopened.next_attempt() == ("insert_hole", 5000000)


def test_identity_and_budget_changes_do_not_reset_quotas(tmp_path):
    ledger = make_ledger(tmp_path)
    failed(ledger)
    state = ledger.snapshot()
    changed = copy.deepcopy(state["config"])
    changed["tasks"]["insert_HDMI"]["natural_limit"] += 1
    with pytest.raises(ValueError, match="identity/config"):
        BatchLedger(ledger.root, changed, state["identity"])
    with pytest.raises(ValueError, match="identity/config"):
        BatchLedger(ledger.root, state["config"], dict(checkpoint="new"))
    assert BatchLedger(ledger.root).counts("insert_HDMI")["natural"] == 1


def test_source_fix_is_append_only_without_refunding_attempts(tmp_path):
    root = tmp_path / "campaign"
    cfg = config(root)
    first = dict(checkpoints={"ema": "unchanged"}, source_hashes={"worker.py": "old"})
    ledger = BatchLedger(root, cfg, first)
    failed(ledger)
    changed = dict(checkpoints={"ema": "unchanged"}, source_hashes={"worker.py": "fixed"})
    resumed = BatchLedger(root, cfg, changed)
    assert resumed.counts("insert_HDMI")["natural"] == 1
    assert len(resumed.snapshot()["source_revisions"]) == 2
    assert resumed.snapshot()["source_revisions"][-1]["changed_paths"] == ["worker.py"]
    assert resumed.snapshot()["attempts"][0]["source_revision"] == 0
    resumed.reserve_natural("insert_hole", 5000000, root / "hole")
    assert resumed.snapshot()["attempts"][1]["source_revision"] == 1


def missed_ledger(tmp_path, **kwargs):
    ledger = make_ledger(tmp_path, **kwargs)
    directory = ledger.root / "insert_HDMI" / "attempts" / "4000000"
    ledger.reserve_natural("insert_HDMI", 4000000, directory)
    natural = dict(success=False, status="failed", takeover_step=None,
                   detector=dict(events=[dict(status="confirmed", step=60)]))
    ledger.record_natural("insert_HDMI", 4000000, natural)
    ledger.finish_attempt("insert_HDMI", 4000000, dict(status="completed", natural=natural,
        recovery_status="no_persistent_contact_candidate"))
    return ledger, directory


def test_explicit_missed_revisit_preserves_original_and_charges_rollout(tmp_path):
    ledger, directory = missed_ledger(tmp_path)
    original = ledger.snapshot()["attempts"][0]
    with pytest.raises(RuntimeError, match="source correction"):
        ledger.reserve_missed_revisit("insert_HDMI", 4000000, directory / "revisit-01")
    state = ledger.snapshot()
    identity = dict(state["identity"], source_hashes={"detector": "fixed"})
    ledger = BatchLedger(ledger.root, state["config"], identity)
    row = ledger.reserve_missed_revisit("insert_HDMI", 4000000, directory / "revisit-01")
    assert row["prior_attempts"] == [original]
    assert row["status"] == "running" and "natural" not in row
    assert ledger.counts("insert_HDMI")["natural"] == 2
    assert ledger.counts("insert_HDMI")["recovery"] == 0
    ledger.record_natural("insert_HDMI", 4000000, dict(success=False, status="failed", takeover_step=60))
    ledger.reserve_recovery("insert_HDMI", 4000000)
    ledger.finish_attempt("insert_HDMI", 4000000, dict(status="completed", recovery=dict(success=True)))
    ledger.record_export("insert_HDMI", 4000000, "recovery", dict(suitable_for_imitation=True))
    reopened = BatchLedger(ledger.root)
    assert reopened.counts("insert_HDMI") == dict(natural=2, recovery=1, normal_success=0, recovery_success=1)
    with pytest.raises(RuntimeError, match="unretried"):
        reopened.reserve_missed_revisit("insert_HDMI", 4000000, directory / "again")
    with pytest.raises(RuntimeError, match="consumed"):
        reopened.reserve_natural("insert_HDMI", 4000000, directory / "again")


def test_missed_revisit_cannot_exceed_existing_allowance(tmp_path):
    ledger, directory = missed_ledger(tmp_path, natural_limit=1)
    state = ledger.snapshot()
    ledger = BatchLedger(ledger.root, state["config"], dict(state["identity"], source_hashes={"detector": "fixed"}))
    with pytest.raises(RuntimeError, match="allowance"):
        ledger.reserve_missed_revisit("insert_HDMI", 4000000, directory / "revisit-01")
    assert ledger.counts("insert_HDMI")["natural"] == 1


def test_normal_export_cap_does_not_consume_recovery_target(tmp_path):
    ledger = make_ledger(tmp_path)
    for seed in (3000002, 3000003):
        ledger.record_export("insert_HDMI", seed, "normal", dict(suitable_for_imitation=True), import_path=tmp_path / str(seed))
    with pytest.raises(RuntimeError, match="target"):
        ledger.record_export("insert_HDMI", 3000004, "normal", dict(suitable_for_imitation=True), import_path=tmp_path / "3000004")
    assert ledger.counts("insert_HDMI")["normal_success"] == 2
    assert ledger.counts("insert_HDMI")["recovery_success"] == 0
    assert ledger.next_attempt() == ("insert_HDMI", 4000000)


def test_recovery_requires_completed_natural_failure_and_boundary(tmp_path):
    ledger = make_ledger(tmp_path)
    ledger.reserve_natural("insert_HDMI", 4000000, ledger.root / "a")
    with pytest.raises(RuntimeError, match="completed natural"):
        ledger.reserve_recovery("insert_HDMI", 4000000)
    ledger.record_natural("insert_HDMI", 4000000, dict(success=True, status="success", takeover_step=40))
    with pytest.raises(RuntimeError, match="completed natural"):
        ledger.reserve_recovery("insert_HDMI", 4000000)
    ledger.reserve_natural("insert_HDMI", 4000001, ledger.root / "b")
    ledger.record_natural("insert_HDMI", 4000001, dict(success=False, status="failed", takeover_step=41))
    with pytest.raises(ValueError, match="boundary"):
        ledger.reserve_recovery("insert_HDMI", 4000001)
    assert ledger.counts("insert_HDMI")["recovery"] == 0


def test_allowances_stop_task_without_resetting_on_reopen(tmp_path):
    ledger = make_ledger(tmp_path, natural_limit=4, recovery_limit=1)
    failed(ledger)
    ledger.reserve_recovery("insert_HDMI", 4000000)
    ledger.finish_attempt("insert_HDMI", 4000000, dict(status="failed", recovery=dict(success=False)))
    ledger = BatchLedger(ledger.root)
    with pytest.raises(RuntimeError, match="allowance"):
        ledger.reserve_natural("insert_HDMI", 4000001, ledger.root / "b")
    assert ledger.next_attempt() == ("insert_hole", 5000000)


def test_import_counts_once_but_does_not_consume_rollout(tmp_path):
    ledger = make_ledger(tmp_path)
    row = dict(kind="recovery", suitable_for_imitation=True, episode_id="insert_HDMI_3000001")
    ledger.record_export("insert_HDMI", 3000001, "recovery", row, import_path=tmp_path / "old/result.json")
    ledger.record_export("insert_HDMI", 3000001, "recovery", row, import_path=tmp_path / "old/result.json")
    assert ledger.counts("insert_HDMI") == dict(natural=0, recovery=0, normal_success=0, recovery_success=1)
    with pytest.raises(RuntimeError, match="Import source"):
        ledger.record_export("insert_HDMI", 3000001, "recovery", row, import_path=tmp_path / "other/result.json")


def test_target_counts_only_accepted_exports_and_no_future_normal_rollouts(tmp_path):
    ledger = make_ledger(tmp_path, target=1)
    failed(ledger)
    ledger.reserve_recovery("insert_HDMI", 4000000)
    ledger.finish_attempt("insert_HDMI", 4000000, dict(status="success", recovery=dict(success=True)))
    assert ledger.counts("insert_HDMI")["recovery_success"] == 0
    ledger.record_export("insert_HDMI", 4000000, "recovery", dict(suitable_for_imitation=True))
    assert ledger.counts("insert_HDMI")["recovery_success"] == 1
    assert ledger.next_attempt() == ("insert_hole", 5000000)
    assert ledger.snapshot()["milestones"]["insert_HDMI"]["first_three_checked"]


def test_unreserved_success_cannot_be_counted(tmp_path):
    ledger = make_ledger(tmp_path)
    failed(ledger)
    with pytest.raises(RuntimeError, match="Unreserved"):
        ledger.finish_attempt("insert_HDMI", 4000000, dict(status="success", recovery=dict(success=True)))
    assert ledger.snapshot()["attempts"][0]["status"] == "running"


def test_seed_audit_reads_explicit_ranges_and_excludes_own_output(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "campaign.json").write_text(json.dumps(dict(seed=4000000)))
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "train.jsonl").write_text(json.dumps(dict(seed=4000002)) + "\n")
    (datasets / "deployment.json").write_text(json.dumps(dict(evaluation_seeds=[4999999, 5000001])))
    result = seed_audit(config(root), [root, datasets])
    assert {seed for row in result["conflicts"] for seed in row["seeds"]} == {4000002, 5000000, 5000001}
    assert result["manifests_checked"] == 2


def test_seed_audit_skips_frozen_code_fixtures_but_checks_actual_results(tmp_path):
    suite = tmp_path / 'eval_suite'
    fixture = suite / 'runtime/vendor/tests/sample.json'
    fixture.parent.mkdir(parents=True)
    (suite / 'runtime/runtime_manifest.json').write_text('{"files": {}}')
    fixture.write_text('{"seed": 4000000}\n{"seed": 4000001}\n')
    result_file = suite / 'results/episode.json'
    result_file.parent.mkdir()
    result_file.write_text('{"seed": 4000002}')
    audit = seed_audit(config(tmp_path / 'campaign'), [suite])
    assert audit['manifests_checked'] == 1
    assert audit['conflicts'] == [dict(path=str(result_file), seeds=[4000002])]
    result_file.write_text('broken actual result metadata')
    with pytest.raises(json.JSONDecodeError):
        seed_audit(config(tmp_path / 'campaign'), [suite])


def test_config_enforces_task_specific_budgets_and_no_training(tmp_path):
    import yaml
    cfg = config(tmp_path / "out")
    for task in cfg["tasks"].values():
        task["checkpoint"] = str(tmp_path / "checkpoint")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert load_config(path)["tasks"]["insert_hole"]["budget"] == 317
    cfg["tasks"]["insert_hole"]["budget"] = 600
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="budget"):
        load_config(path)
    cfg["tasks"]["insert_hole"]["budget"] = 317
    cfg["training_enabled"] = True
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="training"):
        load_config(path)


def completed_recovery_before_parent_exit(ledger):
    failed(ledger)
    ledger.reserve_recovery("insert_HDMI", 4000000)
    row = ledger.snapshot()["attempts"][0]
    directory = Path(row["directory"])
    directory.mkdir(parents=True)
    result = dict(task="insert_HDMI", seed=4000000, status="completed", natural=row["natural"],
                  recovery=dict(success=True, takeover_step=40, strategy_actions=198))
    (directory / "result.json").write_text(json.dumps(result))
    return directory, result


def fake_process(proc_root, directory, *, pid=42, owned=True, record=True, state="S"):
    process = proc_root / str(pid)
    process.mkdir(parents=True)
    command = ["python", "-m", "contact_belief_wam.sim_oracle_batch", "--output", str(directory)] if owned else ["python", "unrelated.py"]
    (process / "cmdline").write_bytes("\0".join(command).encode() + b"\0")
    (process / "stat").write_text(f"{pid} (python worker) {state} 1 1 1")
    if record:
        (directory / "worker.process.json").write_text(json.dumps(dict(pid=pid, command=command)))
    return process


def test_resume_recovers_completed_child_result_without_repeating_or_refunding(tmp_path):
    ledger = make_ledger(tmp_path)
    directory, result = completed_recovery_before_parent_exit(ledger)
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    before = ledger.counts()
    assert ledger.recover_interrupted(proc_root=proc_root) == [dict(task="insert_HDMI", seed=4000000)]
    row = ledger.snapshot()["attempts"][0]
    assert row["status"] == "completed"
    assert row["result"] == result
    assert row["resume_events"][0]["previous_status"] == "running"
    assert ledger.counts() == before
    assert ledger.recover_interrupted(proc_root=proc_root) == []
    assert ledger.next_attempt() == ("insert_hole", 5000000)


@pytest.mark.parametrize("field,value", [("task", "insert_hole"), ("seed", 4000001)])
def test_resume_rejects_result_from_different_reservation_without_mutation(tmp_path, field, value):
    ledger = make_ledger(tmp_path)
    directory, result = completed_recovery_before_parent_exit(ledger)
    result[field] = value
    (directory / "result.json").write_text(json.dumps(result))
    before = ledger.path.read_bytes()
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    with pytest.raises(ValueError, match="task/seed"):
        ledger.recover_interrupted(proc_root=proc_root)
    assert ledger.path.read_bytes() == before


def test_resume_rejects_unreserved_recovery_even_when_recovery_failed(tmp_path):
    ledger = make_ledger(tmp_path)
    failed(ledger)
    row = ledger.snapshot()["attempts"][0]
    directory = Path(row["directory"])
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(json.dumps(dict(task=row["task"], seed=row["seed"], status="completed",
        natural=row["natural"], recovery=dict(success=False, takeover_step=40))))
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    with pytest.raises(RuntimeError, match="unreserved"):
        ledger.recover_interrupted(proc_root=proc_root)
    assert ledger.counts("insert_HDMI")["recovery"] == 0
    assert ledger.snapshot()["attempts"][0]["status"] == "running"


@pytest.mark.parametrize("record_pid", [True, False])
def test_live_orphan_blocks_resume_and_cleanup_even_with_completed_result(tmp_path, record_pid):
    ledger = make_ledger(tmp_path)
    directory, result = completed_recovery_before_parent_exit(ledger)
    proc_root = tmp_path / "proc"
    fake_process(proc_root, directory, record=record_pid)
    before = ledger.path.read_bytes()
    dump = directory / "native/scene/dump"
    dump.mkdir(parents=True)
    (dump / "q.42").write_text("private simulation state")
    with pytest.raises(RuntimeError, match="still alive"):
        ledger.recover_interrupted(proc_root=proc_root)
    assert ledger.path.read_bytes() == before
    with pytest.raises(RuntimeError, match="still alive"):
        cleanup_finished_attempt(directory, result=result, proc_root=proc_root)
    assert (dump / "q.42").exists()


def test_reused_pid_does_not_block_finished_dump_cleanup(tmp_path):
    ledger = make_ledger(tmp_path)
    directory, result = completed_recovery_before_parent_exit(ledger)
    proc_root = tmp_path / "proc"
    fake_process(proc_root, directory, owned=False)
    assert live_owned_workers(directory, proc_root=proc_root) == []
    ledger.recover_interrupted(proc_root=proc_root)
    dump = directory / "native/scene/dump"
    dump.mkdir(parents=True)
    (dump / "q.42").write_text("private simulation state")
    (directory / "trajectory.h5").write_text("formal data must be preserved")
    receipt = cleanup_finished_attempt(directory, result=result, proc_root=proc_root)
    assert receipt["status"] == "removed"
    assert not dump.exists()
    assert (directory / "trajectory.h5").read_text() == "formal data must be preserved"
    assert cleanup_finished_attempt(directory, result=result, proc_root=proc_root) == receipt


def test_failed_dashboard_refresh_preserves_collection_exception(tmp_path, monkeypatch, capsys):
    from contact_belief_wam import oracle_batch_export
    ledger = make_ledger(tmp_path)
    def broken_index(*args):
        raise OSError("report filesystem unavailable")
    monkeypatch.setattr(oracle_batch_export, "rebuild_index", broken_index)
    original = RuntimeError("native worker failed")
    with pytest.raises(RuntimeError, match="native worker failed"):
        try:
            raise original
        except RuntimeError:
            ledger.set_status("error", error=str(original))
            refresh_index_safely(ledger)
            raise
    assert ledger.snapshot()["status"] == "error"
    assert "report filesystem unavailable" in capsys.readouterr().err


def test_live_dashboard_receives_running_attempt_immediately(tmp_path, monkeypatch):
    from contact_belief_wam import oracle_batch_export
    ledger = make_ledger(tmp_path)
    ledger.reserve_natural("insert_HDMI", 4000000, ledger.root / "attempt")
    seen = []
    monkeypatch.setattr(oracle_batch_export, "rebuild_index", lambda root, state: seen.append(state))
    refresh_index_safely(ledger)
    assert seen[0]["status"] == "collecting"
    assert seen[0]["attempts"][0]["status"] == "running"
