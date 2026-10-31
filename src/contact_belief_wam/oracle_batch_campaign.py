"""Bounded, resumable privileged-oracle collection; this module never trains.

Worker API: open ``BatchLedger(root)``, record_natural(task, seed, result), then
reserve_recovery(task, seed) *before* restoring or executing a recovery branch.
The parent reserves the natural attempt and finishes/exports the combined result.
Every reservation commits under a short file lock; a child never waits for a
parent-held ledger lock. Reservations survive crashes and are never refunded.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import sys
import tempfile

from .paths import PROJECT, WORKSPACE, atomic_json
from .takeover import fingerprint
from .takeover_collect import gpu_info, run_child

VERSION = "oracle-campaign-v2"
TASK_BUDGETS = {"insert_HDMI": 600, "insert_hole": 317}
DEFAULT_CONFIG = PROJECT / "configs/recovery_oracle_50.yaml"


def now():
    return datetime.now(timezone.utc).isoformat()


def durable_json(path, value):
    """Atomic replace and fsync both file and directory before allowing actions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
            name = handle.name
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def live_owned_workers(directory, *, proc_root=Path("/proc")):
    """Find a still-running native worker without confusing a reused PID.

    A completed result is written before simulator shutdown and is not evidence
    of process exit. The worker retains its ``-m ... --output`` arguments when
    sim.sh execs Python. If the supervisor died before recording the child PID,
    scan those exact arguments to cover the Popen/process-file write gap.
    """
    directory, proc_root = Path(directory).resolve(), Path(proc_root)
    process_path = directory / "worker.process.json"
    if process_path.exists():
        process = json.loads(process_path.read_text())
        pid = process.get("pid")
        if type(pid) is not int or pid <= 0:
            raise ValueError(f"Invalid worker PID record: {process_path}")
        candidates = [proc_root / str(pid)]
    else:
        candidates = [p for p in proc_root.iterdir() if p.name.isdigit()]
    live = []
    for process_dir in candidates:
        try:
            command = process_dir.joinpath("cmdline").read_bytes().decode("utf-8", errors="replace").split("\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as exc:
            raise RuntimeError(f"Cannot verify worker liveness at {process_dir}") from exc
        if "contact_belief_wam.sim_oracle_batch" not in command or "--output" not in command:
            continue
        index = command.index("--output")
        if index + 1 >= len(command) or Path(command[index + 1]).resolve() != directory:
            continue
        try:
            # Linux stat has a parenthesized comm field, which can contain spaces.
            state = process_dir.joinpath("stat").read_text().rsplit(")", 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            continue
        if state in {"Z", "X"}:
            continue
        live.append(dict(pid=int(process_dir.name), directory=str(directory)))
    return live


def assert_worker_exited(directory, *, proc_root=Path("/proc")):
    live = live_owned_workers(directory, proc_root=proc_root)
    if live:
        raise RuntimeError(f"Owned native worker is still alive; cannot resume or clean this attempt: {live}")


def cleanup_finished_attempt(directory, *, result=None, proc_root=Path("/proc")):
    """Clean only private native dumps after independently checking worker exit."""
    from .oracle_batch_cleanup import cleanup_completed_native_dump
    directory = Path(directory)
    assert_worker_exited(directory, proc_root=proc_root)
    path = directory / "result.json"
    if not path.exists():
        return None
    result = result or json.loads(path.read_text())
    if result.get("status") != "completed":
        return None
    return cleanup_completed_native_dump(directory, child_completed=True, result=result)


def load_config(path, output=None):
    import yaml
    from .oracle_batch_teacher import DEFAULTS as teacher_defaults
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or not isinstance(config.get("tasks"), dict):
        raise ValueError("Campaign requires a tasks mapping")
    config.setdefault("policy_device", "cuda")
    config.setdefault("device", "cuda:0")
    config.setdefault("port", 18891)
    config.setdefault("model_python", str(WORKSPACE / "contact-belief-wam/.venv/bin/python"))
    config.setdefault("chunk_size", 50)
    config.setdefault("execute_steps", 10)
    config.setdefault("physical_steps_per_action", 2)
    config.setdefault("training_enabled", False)
    detector_defaults = dict(min_command_mm=.2, progress_ratio=.2, tactile_persistence_observations=3, resumed_progress_mm=.04)
    detector = config.setdefault("detector", {})
    if set(detector) - detector_defaults.keys():
        raise ValueError("Unknown stall-detector setting")
    config["detector"] = dict(detector_defaults, **detector)
    config["output_root"] = str(Path(output or config["output_root"]).expanduser().resolve())
    config["model_python"] = str(Path(config["model_python"]).expanduser().absolute())
    if config["training_enabled"] is not False:
        raise ValueError("Collection cannot enable training")
    if (config["chunk_size"], config["execute_steps"], config["physical_steps_per_action"]) != (50, 10, 2):
        raise ValueError("Approved execution protocol is 50/10/2")
    if config["policy_device"] not in {"cuda", "cpu"} or config["device"] != "cuda:0":
        raise ValueError("Campaign supports one serial cuda:0 simulation GPU")
    if not 1 <= int(config["port"]) <= 65535:
        raise ValueError("Invalid policy port")
    seed_ranges = []
    for task, cfg in config["tasks"].items():
        if task not in TASK_BUDGETS or cfg.get("budget") != TASK_BUDGETS[task]:
            raise ValueError(f"Task/budget mismatch for {task}")
        for key in ("recovery_target", "normal_target", "natural_limit", "recovery_limit", "seed_start"):
            if type(cfg.get(key)) is not int or cfg[key] < (0 if key == "normal_target" else 1):
                raise ValueError(f"Invalid {task}.{key}")
        if cfg["recovery_limit"] > cfg["natural_limit"]:
            raise ValueError("At most one recovery branch may be attempted per natural seed")
        cfg["checkpoint"] = str(Path(cfg["checkpoint"]).expanduser().resolve())
        if set(cfg.get("teacher", {})) - teacher_defaults[task].keys():
            raise ValueError(f"Unknown {task} Teacher setting")
        cfg["teacher"] = dict(teacher_defaults[task], **cfg.get("teacher", {}))
        cfg["import_recovery"] = [str(Path(p).expanduser().resolve()) for p in cfg.get("import_recovery", [])]
        cfg["import_normal"] = [str(Path(p).expanduser().resolve()) for p in cfg.get("import_normal", [])]
        seeds = set(range(cfg["seed_start"], cfg["seed_start"] + cfg["natural_limit"]))
        if any(seeds & previous for previous in seed_ranges):
            raise ValueError("Task seed ranges overlap")
        seed_ranges.append(seeds)
    if not config["tasks"]:
        raise ValueError("Empty task set")
    return config


def checkpoint_identity(checkpoint, task, budget):
    checkpoint = Path(checkpoint).resolve()
    deployment = json.loads((checkpoint / "deployment.json").read_text())
    complete = json.loads((checkpoint / "complete.json").read_text())
    expected = dict(task=task, camera="head_wrist", chunk_size=50, execute_steps=10,
                    physical_steps_per_action=2, policy_step_budget=budget, use_ema=True,
                    strategy_hz=60, physics_hz=120)
    bad = {key: (deployment.get(key), value) for key, value in expected.items() if deployment.get(key) != value}
    if bad or complete.get("step") != 10000 or complete.get("stage") != "stage2":
        raise ValueError(f"Checkpoint violates {task} dual-view 10k EMA contract: {bad}")
    names = ("deployment.json", "complete.json", "ema.safetensors", "norm_stats.json",
             "wm_norm_stats.json", "pretrained_model/config.json")
    return dict(checkpoint=str(checkpoint), hashes=fingerprint([checkpoint / n for n in names]), deployment=expected)


def _seed_values(value, candidates):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"seed", "episode_seed", "start_seed", "seed_start"} and type(item) is int:
                count = value.get("episodes", value.get("total_num", value.get("natural_limit", 1)))
                count = count if type(count) is int and count > 0 else 1
                yield from (s for s in candidates if item <= s < item + count)
            elif key in {"seeds", "evaluation_seeds"} and isinstance(item, list):
                if key == "evaluation_seeds" and len(item) == 2 and all(type(s) is int for s in item):
                    yield from (s for s in candidates if item[0] <= s <= item[1])
                else:
                    yield from (s for s in item if type(s) is int and s in candidates)
            elif key in {"path", "episode_id"} and isinstance(item, str) and Path(item).stem.isdigit():
                seed = int(Path(item).stem)
                if seed in candidates:
                    yield seed
            yield from _seed_values(item, candidates)
    elif isinstance(value, list):
        for item in value:
            yield from _seed_values(item, candidates)


def _seed_metadata_paths(base):
    if base.is_file():
        yield base
        return
    for directory, folders, files in os.walk(base):
        # Evaluation suites also contain frozen Python source and third-party
        # fixtures (some named .json contain JSONL). These are not seed records.
        # Prune only declared code snapshots; keep validating result metadata.
        if 'runtime_manifest.json' in files:
            folders.clear()
            continue
        yield from (Path(directory) / name for name in files)


def seed_audit(config, roots=None):
    """Inspect dataset metadata, never checkpoint weights; exclude this campaign."""
    if roots is None:
        roots = [WORKSPACE / "contact-belief-wam/data/manifests", PROJECT / "data/manifests",
                 WORKSPACE / "contact-belief-wam/data/hdf5", PROJECT / "data/hdf5",
                 WORKSPACE / "outputs/eval_suites"]
        roots += list((PROJECT / "data").glob("recovery*"))
    output = Path(config["output_root"]).resolve()
    candidates = {s for cfg in config["tasks"].values()
                  for s in range(cfg["seed_start"], cfg["seed_start"] + cfg["natural_limit"])}
    checked, hdf_checked, conflicts, seen = 0, 0, [], set()
    for base in roots:
        base = Path(base)
        if not base.exists():
            continue
        paths = _seed_metadata_paths(base)
        for path in paths:
            if not path.is_file() or path.suffix not in {".json", ".jsonl", ".h5", ".hdf5"}:
                continue
            resolved = path.resolve()
            if output == resolved or output in resolved.parents or resolved in seen:
                continue
            seen.add(resolved)
            found = set()
            if path.suffix in {".h5", ".hdf5"}:
                hdf_checked += 1
                if path.stem.isdigit():
                    found = {int(path.stem)} & candidates
                else:
                    import h5py
                    with h5py.File(path, "r") as handle:
                        found = {int(handle.attrs[k]) for k in ("seed", "episode_seed") if k in handle.attrs} & candidates
                    # Trace directory conventions provide seeds even for legacy HDF without attrs.
                    for component in path.parts:
                        if component.startswith("seed-") and component[5:].isdigit():
                            found |= {int(component[5:])} & candidates
            else:
                # Large per-step logs are not seed registries; all HDF traces are checked separately.
                if path.stat().st_size > 32_000_000:
                    continue
                checked += 1
                rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip()) if path.suffix == ".jsonl" else [json.loads(path.read_text())]
                found = {s for row in rows for s in _seed_values(row, candidates)}
            if found:
                conflicts.append(dict(path=str(resolved), seeds=sorted(found)))
    if not checked and not hdf_checked:
        raise ValueError("No dataset/evaluation metadata was available for overlap audit")
    return dict(manifests_checked=checked, hdf_episodes_checked=hdf_checked, conflicts=conflicts,
                excluded_campaign=str(output))


class BatchLedger:
    """Transactional ledger shared by the supervisor and its live worker."""
    def __init__(self, root, config=None, identity=None):
        self.root = Path(root).resolve()
        self.path = self.root / "campaign.json"
        if config is None and not self.path.exists():
            raise FileNotFoundError(self.path)
        self.root.mkdir(parents=True, exist_ok=True)
        with self._transaction() as state:
            if not state:
                if config is None or identity is None:
                    raise ValueError("New ledger requires config and identity")
                state.update(version=VERSION, config=config, identity=identity, created_at=now(),
                             status="ready", attempts=[], imports=[], milestones={}, last_task=None,
                             source_revisions=[dict(revision=0, recorded_at=now(), hashes=identity.get("source_hashes", {}))])
            else:
                old_identity = {k: v for k, v in state["identity"].items() if k != "source_hashes"}
                new_identity = {k: v for k, v in (identity or state["identity"]).items() if k != "source_hashes"}
                if state.get("version") != VERSION or (config is not None and state["config"] != config) or old_identity != new_identity:
                    raise ValueError("Campaign identity/config changed; use the existing settings without resetting quotas")
                if identity is not None and state["identity"].get("source_hashes") != identity.get("source_hashes"):
                    previous = state["identity"].get("source_hashes", {})
                    current = identity.get("source_hashes", {})
                    revisions = state.setdefault("source_revisions", [])
                    revisions.append(dict(revision=len(revisions), recorded_at=now(), hashes=current,
                                          changed_paths=sorted(k for k in previous.keys() | current.keys() if previous.get(k) != current.get(k)),
                                          preserves_config_checkpoints_and_consumed_quotas=True))
                    state["identity"] = identity

    @contextmanager
    def _transaction(self):
        with (self.root / "campaign.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                state = json.loads(self.path.read_text()) if self.path.exists() else {}
                before = json.dumps(state, sort_keys=True)
                yield state
                if json.dumps(state, sort_keys=True) != before:
                    state["updated_at"] = now()
                    durable_json(self.path, state)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def snapshot(self):
        with self._transaction() as state:
            return json.loads(json.dumps(state))

    @staticmethod
    def _counts(state, task):
        attempts = [r for r in state["attempts"] if r["task"] == task]
        reservations = [old for r in attempts for old in r.get("prior_attempts", [])] + attempts
        exports = [e for r in attempts + [r for r in state["imports"] if r["task"] == task]
                   for e in r.get("exports", []) if e.get("accepted")]
        return dict(natural=len(reservations), recovery=sum("recovery_reserved_at" in r for r in reservations),
                    normal_success=sum(e["kind"] == "normal" for e in exports),
                    recovery_success=sum(e["kind"] == "recovery" for e in exports))

    def counts(self, task=None):
        state = self.snapshot()
        return self._counts(state, task) if task else {t: self._counts(state, t) for t in state["config"]["tasks"]}

    @staticmethod
    def _row(state, task, seed):
        rows = [r for r in state["attempts"] if r["task"] == task and r["seed"] == seed]
        if len(rows) != 1:
            raise ValueError(f"Expected one reserved attempt: {task}/{seed}")
        return rows[0]

    def reserve_natural(self, task, seed, directory):
        with self._transaction() as state:
            cfg, counts = state["config"]["tasks"][task], self._counts(state, task)
            if not cfg["seed_start"] <= seed < cfg["seed_start"] + cfg["natural_limit"]:
                raise ValueError("Seed outside the campaign reservation")
            if counts["natural"] >= cfg["natural_limit"] or counts["recovery"] >= cfg["recovery_limit"] or counts["recovery_success"] >= cfg["recovery_target"]:
                raise RuntimeError("Task target reached or attempt allowance exhausted")
            if any(r["task"] == task and r["seed"] == seed for r in state["attempts"] + state["imports"]):
                raise RuntimeError("Seed already consumed; interrupted attempts cannot be retried")
            row = dict(task=task, seed=seed, directory=str(Path(directory).resolve()), status="running",
                       natural_reserved_at=now(), exports=[], source_revision=len(state.get("source_revisions", [])) - 1)
            state["attempts"].append(row)
            state.update(status="collecting", last_task=task)
            return dict(row)

    def record_natural(self, task, seed, result):
        with self._transaction() as state:
            row = self._row(state, task, seed)
            if row["status"] != "running" or "natural" in row:
                raise RuntimeError("Natural outcome already recorded or attempt is terminal")
            if type(result.get("success")) is not bool:
                raise ValueError("Natural result requires an explicit success boolean")
            if result.get("status") in {"running", "interrupted", "error"}:
                raise ValueError("Natural rollout must complete before recording its outcome")
            row["natural"] = json.loads(json.dumps(result))
            row["natural_finished_at"] = now()
            return dict(row)

    def reserve_missed_revisit(self, task, seed, directory):
        """Explicit, once-only revisit of a detector miss after a source fix.

        Never selected by next_attempt(). Preserve the original outcome and
        charge a second natural rollout. Already attempted recoveries, successful
        baselines and interrupted jobs cannot enter this repair path.
        """
        with self._transaction() as state:
            row = self._row(state, task, seed)
            cfg, counts = state["config"]["tasks"][task], self._counts(state, task)
            natural, result = row.get("natural", {}), row.get("result", {})
            if (row["status"] != "completed" or natural.get("success") is not False or
                    result.get("recovery_status") != "no_persistent_contact_candidate" or
                    row.get("prior_attempts") or row.get("exports") or "recovery_reserved_at" in row):
                raise RuntimeError("Only an unretried, completed detector miss can be explicitly revisited")
            if not any(e.get("status") == "confirmed" for e in natural.get("detector", {}).get("events", [])):
                raise RuntimeError("Missed revisit requires an observed causal contact candidate")
            revision = len(state.get("source_revisions", [])) - 1
            if revision <= row.get("source_revision", 0):
                raise RuntimeError("Revisit requires a recorded source correction")
            if (counts["natural"] >= cfg["natural_limit"] or counts["recovery"] >= cfg["recovery_limit"] or
                    counts["recovery_success"] >= cfg["recovery_target"]):
                raise RuntimeError("Task target reached or attempt allowance exhausted")
            directory = Path(directory).resolve()
            expected = Path(row["directory"]).resolve() / "revisit-01"
            if directory != expected or directory.exists():
                raise RuntimeError("Revisit must use a new child directory without overwriting the original")
            assert_worker_exited(row["directory"])
            original = json.loads(json.dumps(row))
            row.clear()
            row.update(task=task, seed=seed, directory=str(directory), status="running", exports=[],
                       natural_reserved_at=now(), source_revision=revision, prior_attempts=[original],
                       revisit_reason="explicitly_requested_recovery_of_missed_contact_candidate")
            state.update(status="collecting", last_task=task)
            return dict(row)

    def reserve_recovery(self, task, seed):
        with self._transaction() as state:
            row = self._row(state, task, seed)
            cfg, counts = state["config"]["tasks"][task], self._counts(state, task)
            natural = row.get("natural", {})
            candidate = natural.get("takeover_step", (natural.get("detector") or {}).get("first_confirmed"))
            if row["status"] != "running" or natural.get("success") is not False or candidate is None:
                raise RuntimeError("Recovery requires a completed natural failure and saved candidate")
            if not isinstance(candidate, int) or candidate < 0 or candidate >= cfg["budget"] or candidate % 10:
                raise ValueError("Takeover must be an in-budget decision boundary")
            if "recovery_reserved_at" in row:
                raise RuntimeError("Recovery reservation already consumed")
            if counts["recovery"] >= cfg["recovery_limit"] or counts["recovery_success"] >= cfg["recovery_target"]:
                raise RuntimeError("Recovery target/allowance exhausted")
            row.update(recovery_reserved_at=now(), takeover_step=candidate)
            return dict(row)

    def finish_attempt(self, task, seed, result):
        with self._transaction() as state:
            row = self._row(state, task, seed)
            if row["status"] != "running":
                if row.get("result") == result:
                    return dict(row)
                raise RuntimeError("Attempt is already terminal")
            if result.get("status") == "running":
                raise ValueError("Cannot finish with running status")
            if (result.get("recovery") or {}).get("success") and "recovery_reserved_at" not in row:
                raise RuntimeError("Unreserved recovery cannot be accepted")
            row.update(status=result.get("status", "completed"), finished_at=now(), result=result)
            return dict(row)

    def record_export(self, task, seed, kind, export_result, *, import_path=None):
        if kind not in {"normal", "recovery"}:
            raise ValueError("Unknown sample kind")
        with self._transaction() as state:
            if import_path is None:
                row = self._row(state, task, seed)
                if row["status"] == "running":
                    raise RuntimeError("Cannot export an unfinished attempt")
            else:
                rows = [r for r in state["imports"] if r["task"] == task and r["seed"] == seed]
                if any(r["task"] == task and r["seed"] == seed for r in state["attempts"]):
                    raise RuntimeError("Import conflicts with a consumed seed")
                row = rows[0] if rows else dict(task=task, seed=seed, source=str(Path(import_path).resolve()), exports=[])
                if not rows:
                    state["imports"].append(row)
                elif row["source"] != str(Path(import_path).resolve()):
                    raise RuntimeError("Import source changed")
            old = [e for e in row["exports"] if e["kind"] == kind]
            accepted = bool(export_result.get("suitable_for_imitation", export_result.get("accepted", False)))
            entry = dict(kind=kind, accepted=accepted, data=export_result)
            if old:
                if old[0] == entry:
                    return dict(row)
                raise RuntimeError("Sample export already recorded with different content")
            counts, cfg = self._counts(state, task), state["config"]["tasks"][task]
            if accepted and counts[kind + "_success"] >= cfg[("normal" if kind == "normal" else "recovery") + "_target"]:
                raise RuntimeError("Sample target already reached")
            row["exports"].append(entry)
            counts = self._counts(state, task)
            if counts["recovery_success"] >= min(3, cfg["recovery_target"]):
                state["milestones"].setdefault(task, dict(first_three_checked=True, passed_at=now()))
            return dict(row)

    def recover_interrupted(self, *, proc_root=Path("/proc")):
        """Reconcile saved completion after supervisor death; never rerun a seed.

        Supervisor only, after obtaining its exclusive process lock. A SIGKILL
        can leave its new-session child alive, so check every owned worker before
        changing any attempt status, regardless of whether result.json exists.
        """
        with self._transaction() as state:
            for row in state["attempts"]:
                assert_worker_exited(row["directory"], proc_root=proc_root)
            reconciled = []
            for row in state["attempts"]:
                if row["status"] not in {"running", "interrupted"}:
                    continue
                path = Path(row["directory"]) / "result.json"
                result = json.loads(path.read_text()) if path.exists() else None
                if result is not None and result.get("status") == "completed":
                    if result.get("task") != row["task"] or result.get("seed") != row["seed"]:
                        raise ValueError(f"Completed result task/seed does not match reservation: {path}")
                    if result.get("natural") != row.get("natural") or type((result.get("natural") or {}).get("success")) is not bool:
                        raise ValueError(f"Completed result differs from the recorded natural outcome: {path}")
                    recovery = result.get("recovery")
                    if recovery is not None:
                        if "recovery_reserved_at" not in row:
                            raise RuntimeError(f"Completed result contains an unreserved recovery: {path}")
                        if type(recovery.get("success")) is not bool or recovery.get("takeover_step") != row["takeover_step"]:
                            raise ValueError(f"Completed recovery does not match its takeover reservation: {path}")
                    elif "recovery_reserved_at" in row:
                        raise ValueError(f"Completed result omits its reserved recovery branch: {path}")
                    row.setdefault("resume_events", []).append(dict(previous_status=row["status"],
                        previous_error=row.pop("error", None), reconciled_at=now(), result=str(path),
                        consumed_quotas_preserved=True))
                    row.update(status="completed", finished_at=now(), result=result)
                    reconciled.append(dict(task=row["task"], seed=row["seed"]))
                elif row["status"] == "running":
                    row.update(status="interrupted", finished_at=now(), error="Previous supervisor exited before completion; reservations retained")
            return reconciled

    def set_status(self, status, **details):
        with self._transaction() as state:
            state.update(status=status, **details)

    def next_attempt(self):
        state = self.snapshot()
        tasks = list(state["config"]["tasks"])
        last = state.get("last_task")
        if last in tasks:
            index = tasks.index(last) + 1
            tasks = tasks[index:] + tasks[:index]
        for task in tasks:
            cfg, counts = state["config"]["tasks"][task], self._counts(state, task)
            if counts["recovery_success"] >= cfg["recovery_target"] or counts["natural"] >= cfg["natural_limit"] or counts["recovery"] >= cfg["recovery_limit"]:
                continue
            used = {r["seed"] for r in state["attempts"] + state["imports"] if r["task"] == task}
            for seed in range(cfg["seed_start"], cfg["seed_start"] + cfg["natural_limit"]):
                if seed not in used:
                    return task, seed
        return None


def preflight(config):
    if not Path(config["model_python"]).is_file():
        raise FileNotFoundError(config["model_python"])
    checkpoints = {task: checkpoint_identity(cfg["checkpoint"], task, cfg["budget"]) for task, cfg in config["tasks"].items()}
    audit = seed_audit(config)
    if audit["conflicts"]:
        raise ValueError(f"Reserved seeds overlap existing data: {audit['conflicts']}")
    for task, cfg in config["tasks"].items():
        for field in ("import_recovery", "import_normal"):
            for path in cfg.get(field, []):
                result = json.loads(Path(path).read_text())
                if not result.get("success") or result.get("task", task) != task:
                    raise ValueError(f"Invalid existing successful sample: {path}")
    names = ("oracle_batch_campaign.py", "sim_oracle_batch.py", "oracle_batch_teacher.py", "oracle_batch_detector.py", "oracle_batch_geometry.py",
             "oracle_batch_export.py", "oracle_batch_cleanup.py", "takeover_dataset.py", "data.py", "sim_takeover.py", "takeover.py")
    sources = [Path(__file__).with_name(n) for n in names if Path(__file__).with_name(n).exists()]
    sources.extend(WORKSPACE / "pi05-univtac/src/pi05_univtac" / n for n in
                   ("contact_restore.py", "contact_trace.py", "contact_recovery.py", "sim_policy.py"))
    sources.extend(WORKSPACE / "UniVTAC" / n for n in
                   ("envs/insert_HDMI.py", "envs/insert_hole.py", "policy/Wave0Wam/deploy_policy.py"))
    return dict(checkpoints=checkpoints, seed_audit=audit, source_hashes=fingerprint(sources), training_enabled=False)


def _export_rows(ledger, task, seed, rows, import_path=None):
    for row in rows:
        kind = "normal" if row.get("kind", row.get("source")) in {"normal", "natural", "normal_success", "wave0"} else "recovery"
        ledger.record_export(task, seed, kind, row, import_path=import_path)


def refresh_index_safely(ledger):
    """Keep the live dashboard current without hiding a collection exception."""
    try:
        from .oracle_batch_export import rebuild_index
        rebuild_index(ledger.root, ledger.snapshot())
    except Exception as exc:
        print(f"Campaign index refresh failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Read-only checkpoint, configuration and seed audit")
    parser.add_argument("--max-attempts", type=int, help="Optional additional natural-attempt cap for bounded validation")
    parser.add_argument("--revisit-missed", action="append", default=[], metavar="TASK:SEED",
                        help="Explicitly revisit an unattempted detector miss once, retaining both reservations")
    args = parser.parse_args(argv)
    if args.max_attempts is not None and args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    config = load_config(args.config, args.output)
    revisits = []
    for value in args.revisit_missed:
        try:
            task, seed = value.rsplit(":", 1)
            seed = int(seed)
        except ValueError:
            parser.error("--revisit-missed must be TASK:SEED")
        if task not in config["tasks"] or (task, seed) in revisits:
            parser.error("Unknown task or duplicate --revisit-missed")
        revisits.append((task, seed))
    checked = preflight(config)
    root = Path(config["output_root"])
    identity = dict(checkpoints=checked["checkpoints"], source_hashes=checked["source_hashes"])
    if args.dry_run:
        if (root / "campaign.json").exists():
            previous = json.loads((root / "campaign.json").read_text())
            if previous["config"] != config or ({k: v for k, v in previous["identity"].items() if k != "source_hashes"}
                                                != {k: v for k, v in identity.items() if k != "source_hashes"}):
                raise ValueError("Existing campaign identity differs from requested run")
        print(json.dumps(dict(status="preflight_passed", config=config, **checked), ensure_ascii=False), flush=True)
        return 0
    root.mkdir(parents=True, exist_ok=True)
    with (root / "supervisor.lock").open("a") as supervisor_lock:
        fcntl.flock(supervisor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = BatchLedger(root, config, identity)
        ledger.recover_interrupted()
        for previous in ledger.snapshot()["attempts"]:
            if previous["status"] == "completed":
                cleanup_finished_attempt(previous["directory"], result=previous.get("result"))
        durable_json(root / "effective_config.json", config)
        durable_json(root / "preflight.json", checked)
        gpu_lock = None
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}")
        old_handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            uuid, free = gpu_info()
            gpu_path = WORKSPACE / "outputs/eval_suites" / f".{uuid}.lock"
            gpu_path.parent.mkdir(parents=True, exist_ok=True)
            gpu_lock = gpu_path.open("a")
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if free < 20 * 1024:
                raise RuntimeError(f"GPU occupied: {free} MiB free")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", config["port"]))
            from .oracle_batch_export import export_attempt, import_existing, rebuild_index
            ledger.set_status("collecting", resumed_at=now(), error=None)
            durable_json(root / "daemon.json", dict(pid=os.getpid(), running=True, gpu_uuid=uuid, started_at=now()))
            for task, cfg in config["tasks"].items():
                for field in ("import_recovery", "import_normal"):
                    for path in cfg.get(field, []):
                        source = json.loads(Path(path).read_text())
                        if any(r["task"] == task and r["seed"] == source["seed"] for r in ledger.snapshot()["imports"]):
                            continue
                        rows = import_existing(Path(path), root)
                        _export_rows(ledger, task, source["seed"], rows, import_path=path)
            # Export a completed result after an interrupted parent, without rerunning a seed.
            for row in ledger.snapshot()["attempts"]:
                path = Path(row["directory"]) / "result.json"
                if row["status"] == "completed" and not row.get("exports") and path.exists():
                    result = json.loads(path.read_text())
                    if result.get("status") != "completed" or result.get("task") != row["task"] or result.get("seed") != row["seed"]:
                        raise ValueError(f"Saved completed result no longer matches its ledger reservation: {path}")
                    normal_needed = ledger.counts(row["task"])["normal_success"] < config["tasks"][row["task"]]["normal_target"]
                    if ((result.get("natural") or {}).get("success") and normal_needed) or (result.get("recovery") or {}).get("success"):
                        _export_rows(ledger, row["task"], row["seed"], export_attempt(path, root))
            rebuild_index(root, ledger.snapshot())
            env = dict(os.environ, CBWAM_ROOT=str(PROJECT), TWAM_ROOT=str(WORKSPACE),
                PYTHONPATH=os.pathsep.join([str(PROJECT / "src"), str(WORKSPACE / "pi05-univtac/src"), str(WORKSPACE / "openpi/packages/openpi-client/src")]),
                PI05_SIM_GRAPHICS_LIBRARY_PATH=str(WORKSPACE / ".cache/sim-graphics-runtime/lib"),
                OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", HF_HUB_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", PYTHONUTF8="1")
            started = 0
            while (next_item := revisits[0] if revisits else ledger.next_attempt()) is not None:
                if (root / "STOP_AFTER_CURRENT").exists():
                    ledger.set_status("stopped", stop_reason="requested_stop_after_current")
                    break
                if args.max_attempts is not None and started >= args.max_attempts:
                    ledger.set_status("stopped", stop_reason="requested_attempt_cap")
                    break
                task, seed = next_item
                directory = root / task / "attempts" / str(seed)
                if revisits:
                    directory = directory / "revisit-01"
                    ledger.reserve_missed_revisit(task, seed, directory)
                    revisits.pop(0)
                else:
                    if directory.exists():
                        raise RuntimeError(f"Unledgered directory already exists; refusing overwrite: {directory}")
                    ledger.reserve_natural(task, seed, directory)
                refresh_index_safely(ledger)
                started += 1
                command = ["bash", str(PROJECT / "scripts/sim.sh"), "-m", "contact_belief_wam.sim_oracle_batch",
                           "--config", str(root / "effective_config.json"), "--campaign-root", str(root),
                           "--task", task, "--seed", str(seed), "--output", str(directory),
                           "--headless", "--device", config["device"]]
                code = run_child(command, directory / "worker.log", env)
                path = directory / "result.json"
                result = json.loads(path.read_text()) if path.exists() else dict(status="error", error=f"worker_exit_{code}_without_result")
                ledger.finish_attempt(task, seed, result)
                if code or result.get("status") in {"error", "interrupted"}:
                    raise RuntimeError(f"Worker failed at {task}/{seed}: {result.get('error', code)}")
                if result.get("status") == "completed":
                    cleanup_finished_attempt(directory, result=result)
                normal_needed = ledger.counts(task)["normal_success"] < config["tasks"][task]["normal_target"]
                rows = export_attempt(path, root) if (not (result.get("natural") or {}).get("success") or normal_needed) else []
                _export_rows(ledger, task, seed, rows)
                rebuild_index(root, ledger.snapshot())
                print(json.dumps(dict(task=task, seed=seed, status=result.get("status"), counts=ledger.counts()), ensure_ascii=False), flush=True)
            else:
                counts = ledger.counts()
                done = all(counts[t]["recovery_success"] >= cfg["recovery_target"] for t, cfg in config["tasks"].items())
                ledger.set_status("complete" if done else "allowance_exhausted", stop_reason="target_reached" if done else "attempt_allowance_exhausted")
            rebuild_index(root, ledger.snapshot())
            return 0
        except BaseException as exc:
            ledger.set_status("interrupted" if isinstance(exc, KeyboardInterrupt) else "error", error=f"{type(exc).__name__}: {exc}")
            refresh_index_safely(ledger)
            raise
        finally:
            durable_json(root / "daemon.json", dict(pid=os.getpid(), running=False, status=ledger.snapshot()["status"], stopped_at=now()))
            refresh_index_safely(ledger)
            if gpu_lock is not None:
                gpu_lock.close()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
