"""Remove completed attempts' private UIPC dumps, never recorded dataset data.

The caller must have waited for the worker to exit. A result file alone is not
proof: the worker writes that result before closing its simulator. Cleanup is
limited to recognized transient files under this attempt's native/scene/dump.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import stat

from .paths import atomic_json


_DUMP_FILE = re.compile(r"(?:q|q_v|q_prev|positions|prev_positions)\.(\d+)$|state\.(\d+)\.json$")
_OPEN_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _no_symlink_path(path):
    """Check even ancestors, so a redirected native directory is not accepted."""
    for item in reversed((path,) + tuple(path.parents)):
        if item.is_symlink():
            raise ValueError(f"Refusing symlink in native cleanup path: {item}")


def _identity(info):
    return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns


def _inventory(fd, prefix=""):
    files, directories = [], []
    with os.scandir(fd) as entries:
        for entry in entries:
            info = entry.stat(follow_symlinks=False)
            relative = prefix + entry.name
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f"Refusing symlink in physics dump: {relative}")
            if stat.S_ISDIR(info.st_mode):
                child = os.open(entry.name, _OPEN_DIR, dir_fd=fd)
                try:
                    nested_files, nested_dirs = _inventory(child, relative + "/")
                finally:
                    os.close(child)
                directories.append(dict(path=relative, identity=_identity(info)))
                files.extend(nested_files)
                directories.extend(nested_dirs)
            elif stat.S_ISREG(info.st_mode):
                match = _DUMP_FILE.fullmatch(entry.name)
                if match is None:
                    raise ValueError(f"Unrecognized file in physics dump; nothing removed: {relative}")
                files.append(dict(path=relative, identity=_identity(info), bytes=info.st_size,
                                  frame=int(next(group for group in match.groups() if group is not None))))
            else:
                raise ValueError(f"Refusing special file in physics dump: {relative}")
    return files, directories


def _open_relative_directory(root_fd, relative):
    fd = os.dup(root_fd)
    try:
        for component in Path(relative).parts:
            child = os.open(component, _OPEN_DIR, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _check_identity(fd, name, expected):
    if _identity(os.stat(name, dir_fd=fd, follow_symlinks=False)) != tuple(expected):
        raise RuntimeError(f"Physics dump changed after worker completion: {name}")


def cleanup_completed_native_dump(attempt, *, child_completed, result=None):
    """Clean an exited, completed worker's dump and write an auditable receipt.

    ``child_completed=True`` must mean subprocess.wait()/run_child returned,
    not merely that result.json exists. The caller also owns any live-process
    identity check for historical attempts. This function refuses symlinks and
    unknown file types/names before removing anything, then uses directory FDs
    and no-follow opens so descendants cannot redirect removal outside dump/.
    """
    if child_completed is not True:
        raise ValueError("Native dump cleanup requires explicit completed child confirmation")
    attempt = Path(attempt).expanduser().absolute()
    _no_symlink_path(attempt)
    if not attempt.is_dir():
        raise ValueError("Attempt directory does not exist")
    result_path = attempt / "result.json"
    _no_symlink_path(result_path)
    disk_result = json.loads(result_path.read_text())
    if disk_result.get("status") != "completed" or (result is not None and result.get("status") != "completed"):
        raise ValueError("Only a completed attempt may discard its physics dumps")
    dump = attempt / "native" / "scene" / "dump"
    receipt_path = attempt / "native_dump_cleanup.json"
    _no_symlink_path(dump)
    _no_symlink_path(receipt_path)
    if dump.resolve() != attempt.resolve() / "native" / "scene" / "dump":
        raise ValueError("Physics dump is outside the owned attempt")
    if not dump.exists():
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("status") in ("removed", "absent") and receipt.get("dump_path") == str(dump):
                return receipt
        receipt = dict(status="absent", dump_path=str(dump), child_completed=True,
                       result_status="completed", removed_files=0, removed_bytes=0, frame_ids=[],
                       completed_at=datetime.now(timezone.utc).isoformat())
        atomic_json(receipt_path, receipt)
        return receipt
    attempt_fd = os.open(attempt, _OPEN_DIR)
    native_fd = scene_fd = dump_fd = None
    try:
        native_fd = os.open("native", _OPEN_DIR, dir_fd=attempt_fd)
        scene_fd = os.open("scene", _OPEN_DIR, dir_fd=native_fd)
        dump_fd = os.open("dump", _OPEN_DIR, dir_fd=scene_fd)
        dump_identity = _identity(os.fstat(dump_fd))
        files, directories = _inventory(dump_fd)
        receipt = dict(status="planned", dump_path=str(dump), child_completed=True,
                       result_status="completed", removed_files=0, removed_bytes=0,
                       planned_files=len(files), planned_bytes=sum(item["bytes"] for item in files),
                       frame_ids=sorted({item["frame"] for item in files}),
                       reason="same_instance_UIPC_dumps_not_needed_after_worker_exit",
                       started_at=datetime.now(timezone.utc).isoformat())
        atomic_json(receipt_path, receipt)
        for item in files:
            relative = Path(item["path"])
            parent_fd = _open_relative_directory(dump_fd, relative.parent)
            try:
                _check_identity(parent_fd, relative.name, item["identity"])
                os.unlink(relative.name, dir_fd=parent_fd)
            finally:
                os.close(parent_fd)
            receipt["removed_files"] += 1
            receipt["removed_bytes"] += item["bytes"]
        for item in sorted(directories, key=lambda x: len(Path(x["path"]).parts), reverse=True):
            relative = Path(item["path"])
            parent_fd = _open_relative_directory(dump_fd, relative.parent)
            try:
                # Directory mtimes/sizes change as its files are removed; only
                # its inode, device and directory type must still match.
                current = _identity(os.stat(relative.name, dir_fd=parent_fd, follow_symlinks=False))
                if current[:3] != tuple(item["identity"][:3]):
                    raise RuntimeError("Physics dump directory changed during cleanup")
                os.rmdir(relative.name, dir_fd=parent_fd)
            finally:
                os.close(parent_fd)
        current = _identity(os.stat("dump", dir_fd=scene_fd, follow_symlinks=False))
        if current[:3] != dump_identity[:3]:
            raise RuntimeError("Physics dump root changed during cleanup")
        os.rmdir("dump", dir_fd=scene_fd)
        receipt.update(status="removed", completed_at=datetime.now(timezone.utc).isoformat())
        atomic_json(receipt_path, receipt)
        return receipt
    finally:
        for fd in (dump_fd, scene_fd, native_fd, attempt_fd):
            if fd is not None:
                os.close(fd)
