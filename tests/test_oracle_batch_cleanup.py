import json
import os
from pathlib import Path

import pytest

from contact_belief_wam.oracle_batch_cleanup import cleanup_completed_native_dump


def attempt(tmp_path, status="completed"):
    root = tmp_path / "attempt"
    root.mkdir()
    (root / "result.json").write_text(json.dumps(dict(status=status)))
    files = {
        "native/scene/dump/cuda/finite_element/finite_element_method.cu/q.5": b"physics",
        "native/scene/dump/cuda/global_geometry/global_vertex_manager.cu/positions.85": b"vertices",
        "native/scene/dump/common/sim_engine.cpp/state.85.json": b'{"frame":85}',
        "native/scene/systems.json": b"systems",
        "native/video/42.mp4": b"video",
        "natural/trace/trajectory.h5": b"raw-hdf",
        "snapshot.json": b"audit",
        "cache/tactile.npz": b"features",
        "worker.log": b"log",
    }
    for name, contents in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    return root, files


def test_only_owned_physics_dumps_removed_with_receipt(tmp_path):
    root, files = attempt(tmp_path)
    receipt = cleanup_completed_native_dump(root, child_completed=True)
    assert receipt["status"] == "removed"
    assert receipt["removed_files"] == 3
    assert receipt["removed_bytes"] == sum(len(data) for name, data in files.items() if "/dump/" in name)
    assert receipt["frame_ids"] == [5, 85]
    assert not (root / "native/scene/dump").exists()
    for name, contents in files.items():
        if "/dump/" not in name:
            assert (root / name).read_bytes() == contents
    assert json.loads((root / "native_dump_cleanup.json").read_text()) == receipt


@pytest.mark.parametrize("confirmation", [False, None, 1, "yes"])
def test_requires_explicit_waited_child(tmp_path, confirmation):
    root, _ = attempt(tmp_path)
    with pytest.raises(ValueError, match="completed child"):
        cleanup_completed_native_dump(root, child_completed=confirmation)
    assert (root / "native/scene/dump").is_dir()


@pytest.mark.parametrize("status", ["running", "interrupted", "error", "failed"])
def test_noncompleted_attempt_remains_untouched(tmp_path, status):
    root, _ = attempt(tmp_path, status)
    with pytest.raises(ValueError, match="completed attempt"):
        cleanup_completed_native_dump(root, child_completed=True)
    assert (root / "native/scene/dump").is_dir()


def test_idempotent_cleanup_preserves_original_counts(tmp_path):
    root, _ = attempt(tmp_path)
    first = cleanup_completed_native_dump(root, child_completed=True)
    assert cleanup_completed_native_dump(root, child_completed=True) == first


def test_missing_dump_gets_zero_receipt_without_creating_native_tree(tmp_path):
    root = tmp_path / "attempt"
    root.mkdir()
    (root / "result.json").write_text('{"status":"completed"}')
    result = cleanup_completed_native_dump(root, child_completed=True)
    assert result["status"] == "absent" and result["removed_files"] == 0
    assert not (root / "native").exists()


@pytest.mark.parametrize("link_kind", ["inside_file", "inside_dir", "root", "ancestor", "receipt"])
def test_symlinks_never_redirect_or_enable_removal(tmp_path, link_kind):
    root, files = attempt(tmp_path)
    other = tmp_path / "outside"
    other.mkdir()
    victim = other / "q.5"
    victim.write_bytes(b"outside data")
    target = root
    if link_kind == "inside_file":
        (root / "native/scene/dump/q.9").symlink_to(victim)
    elif link_kind == "inside_dir":
        (root / "native/scene/dump/linked").symlink_to(other, target_is_directory=True)
    elif link_kind == "root":
        target = tmp_path / "alias"
        target.symlink_to(root, target_is_directory=True)
    elif link_kind == "ancestor":
        native = root / "native"
        native.rename(root / "real_native")
        native.symlink_to(root / "real_native", target_is_directory=True)
    else:
        (root / "native_dump_cleanup.json").symlink_to(victim)
    with pytest.raises(ValueError, match="symlink"):
        cleanup_completed_native_dump(target, child_completed=True)
    assert victim.read_bytes() == b"outside data"
    for name, contents in files.items():
        assert (root / name).read_bytes() == contents


def test_unknown_raw_data_inside_dump_prevents_all_deletion(tmp_path):
    root, files = attempt(tmp_path)
    (root / "native/scene/dump/trajectory.h5").write_bytes(b"raw")
    with pytest.raises(ValueError, match="Unrecognized"):
        cleanup_completed_native_dump(root, child_completed=True)
    assert not (root / "native_dump_cleanup.json").exists()
    for name, contents in files.items():
        assert (root / name).read_bytes() == contents


def test_special_file_is_rejected_without_touching_other_files(tmp_path):
    root, files = attempt(tmp_path)
    os.mkfifo(root / "native/scene/dump/q.99")
    with pytest.raises(ValueError, match="special file"):
        cleanup_completed_native_dump(root, child_completed=True)
    for name, contents in files.items():
        assert (root / name).read_bytes() == contents


def test_supplied_result_cannot_override_running_disk_result(tmp_path):
    root, _ = attempt(tmp_path, "running")
    with pytest.raises(ValueError, match="completed attempt"):
        cleanup_completed_native_dump(root, child_completed=True, result={"status": "completed"})
