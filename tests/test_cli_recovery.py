import json

import pytest

import contact_belief_wam.cli as cli
from contact_belief_wam.cli import _verify_recovery_collection


def _write_collection(path, attempts):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"attempts": attempts}))


def _eligible(seed):
    return {"seed": seed, "success": True, "blockages": 1}


def test_prepare_recovery_rejects_zero_exit_without_collection_report(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "configure", lambda: None)
    monkeypatch.setattr(cli, "PROJECT", tmp_path / "project")
    monkeypatch.setattr(cli.subprocess, "run", lambda *_args, **_kwargs: None)

    data_root = tmp_path / "data"
    with pytest.raises(RuntimeError, match=r"0/1 qualifying episodes; missing or unreadable collection.json"):
        cli.main([
            "prepare", "recovery", "--task", "insert_hole",
            "--data-root", str(data_root), "--episodes", "1",
        ])


def test_recovery_collection_rejects_fewer_eligible_hdf5s_than_requested(tmp_path):
    report_path = tmp_path / "collection.json"
    hdf5_root = tmp_path / "hdf5"
    hdf5_root.mkdir()
    (hdf5_root / "10.hdf5").touch()
    _write_collection(report_path, [
        _eligible(10),
        {"seed": 11, "success": False, "blockages": 2},
        {"seed": 12, "success": True, "blockages": 0},
        {"seed": 13, "success": True, "blockages": 1, "excluded_from_training": True},
    ])

    with pytest.raises(RuntimeError, match=r"1/2 qualifying episodes with HDF5"):
        _verify_recovery_collection(report_path, hdf5_root, 2)


def test_recovery_collection_does_not_count_eligible_record_without_hdf5(tmp_path):
    report_path = tmp_path / "collection.json"
    _write_collection(report_path, [_eligible(20)])

    with pytest.raises(RuntimeError, match=r"0/1 qualifying episodes with HDF5.*1 missing HDF5 file"):
        _verify_recovery_collection(report_path, tmp_path / "hdf5", 1)


def test_recovery_collection_accepts_requested_episodes_with_hdf5(tmp_path):
    report_path = tmp_path / "collection.json"
    hdf5_root = tmp_path / "hdf5"
    hdf5_root.mkdir()
    (hdf5_root / "30.hdf5").touch()
    (hdf5_root / "31.hdf5").touch()
    _write_collection(report_path, [
        _eligible(30), _eligible(31),
        {"seed": 32, "success": True, "blockages": 0},
        {"seed": 33, "success": True, "blockages": 1, "excluded_from_training": True},
        _eligible(34),  # HDF5 absent: does not satisfy the requested count.
    ])

    assert _verify_recovery_collection(report_path, hdf5_root, 2) == 2
