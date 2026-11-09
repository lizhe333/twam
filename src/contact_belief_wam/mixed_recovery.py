"""Durable quotas and pure scheduling for the approved mixed recovery dataset."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
from pathlib import Path

from .paths import PROJECT
from .oracle_batch_campaign import durable_json, now

KINDS = ("normal", "self_corrected", "oracle_recovery")
BUDGETS = {"insert_HDMI": 600, "insert_hole": 317}
STAGES = {"approach": (2., 6.), "near": (.5, 2.)}
DIRECTIONS = {"x_plus": (1., 0.), "x_minus": (-1., 0.), "y_plus": (0., 1.), "y_minus": (0., -1.)}
VERSION = "mixed-recovery-v3"


def load_config(path):
    import yaml
    from .oracle_batch_campaign import load_config as load_base
    original = yaml.safe_load(Path(path).read_text())
    config = load_base(path)
    for task, cfg in config["tasks"].items():
        for key in ("self_corrected_target", "perturb_limit", "max_perturb_per_seed"):
            if type(cfg.get(key)) is not int or cfg[key] < 1:
                raise ValueError(f"Invalid {task}.{key}")
        if cfg["max_perturb_per_seed"] > 4:
            raise ValueError("At most four perturbations per seed")
        allowed = [.5, 1., 2.] if task == "insert_HDMI" else [.75, 1.5, 3., 4.]
        if cfg.get("perturb_amplitudes_mm") != allowed:
            raise ValueError("Approved perturbation amplitudes changed")
        expected = "hdmi-oracle-v2" if task == "insert_HDMI" else "hole-bounded-contact-alignment-v1"
        if cfg.get("teacher_revision") != expected:
            raise ValueError("Wrong approved Teacher revision")
        if task == "insert_hole" and cfg["teacher"]["rotation_tolerance_deg"] != 2.:
            raise ValueError("Use the successful preview's two-degree alignment tolerance")
    config["import_manifests"] = [str(Path(p).resolve()) for p in original.get("import_manifests", [])]
    return config


def targets(config, task):
    cfg = config["tasks"][task]
    return dict(normal=cfg["normal_target"], self_corrected=cfg["self_corrected_target"],
                oracle_recovery=cfg["recovery_target"])


def stage_for_distance(distance_m):
    mm = float(distance_m)*1000
    if .5 < mm < 2.:
        return "near"
    if 2. <= mm <= 6.:
        return "approach"
    return None


def rank_trials(config, state, task, seed):
    """Balance attempted cells without consuming policy/simulation RNG."""
    cfg = config["tasks"][task]
    used = [b["trial"] for a in state["attempts"] if a["task"] == task
            for b in a["branches"] if b["type"] == "perturb"]
    cells = Counter((x["stage"], x["direction"], x["amplitude_mm"]) for x in used)
    margins = [Counter(x[k] for x in used) for k in ("stage", "direction", "amplitude_mm")]
    trials = []
    for stage in STAGES:
        for direction, vector in DIRECTIONS.items():
            for amplitude in cfg["perturb_amplitudes_mm"]:
                trials.append(dict(stage=stage, direction=direction, amplitude_mm=amplitude,
                                   offset_mm=[v*amplitude for v in vector], duration=10))
    # Rotate deterministic ties across scene seeds, using no global RNG.
    shift = (seed-cfg["seed_start"]) % len(trials)
    trials = trials[shift:] + trials[:shift]
    order = {id(t): i for i, t in enumerate(trials)}
    def score(t):
        values = (t["stage"], t["direction"], t["amplitude_mm"])
        return (cells[values], sum(c[v] for c, v in zip(margins, values)), order[id(t)])
    return sorted(trials, key=score)


def _canonical_checkpoint_config(config):
    result = deepcopy(config)
    for task in result.get('tasks', {}).values():
        if 'checkpoint' in task:
            task['checkpoint'] = str(Path(task['checkpoint']).resolve())
    return result


def _canonical_checkpoint_identities(checkpoints):
    result = deepcopy(checkpoints)
    for identity in result.values():
        identity['checkpoint'] = str(Path(identity['checkpoint']).resolve())
        identity['hashes'] = {str(Path(path).resolve()): digest
                              for path, digest in identity['hashes'].items()}
    return result


class MixedLedger:
    def __init__(self, root, config=None, identity=None):
        self.root = Path(root).resolve()
        self.path = self.root / "campaign.json"
        if config is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            with self._transaction(create=True) as state:
                if not state:
                    state.update(version=VERSION, config=config, identity=identity, created_at=now(),
                                 attempts=[], imports=[], exports=[], revisions=[identity],
                                 task_status={t: "validating" for t in config["tasks"]}, status="ready")
                elif (state["version"] != VERSION or
                      _canonical_checkpoint_config(state["config"]) != _canonical_checkpoint_config(config)):
                    raise ValueError("Existing mixed campaign configuration changed")
                elif identity is not None:
                    # Directory aliases may change when the workspace is
                    # organized. Still require identical files and hashes.
                    if (_canonical_checkpoint_identities(state["identity"]["checkpoints"]) !=
                            _canonical_checkpoint_identities(identity["checkpoints"])):
                        raise ValueError("Checkpoint identity changed")
                    if state["revisions"][-1] != identity:
                        state["revisions"].append(identity)
        elif not self.path.exists():
            raise ValueError("Mixed ledger does not exist")

    @contextmanager
    def _transaction(self, create=False):
        with (self.root / "campaign.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else {}
            if not state and not create:
                raise ValueError("Missing campaign state")
            yield state
            state["updated_at"] = now()
            durable_json(self.path, state)

    def snapshot(self):
        return json.loads(self.path.read_text())

    @staticmethod
    def counts(state, task):
        attempts = [a for a in state["attempts"] if a["task"] == task]
        branches = [b for a in attempts for b in a["branches"]]
        rows = [r for r in state["exports"] if r["task"] == task]
        return dict(natural=len(attempts), perturb=sum(b["type"] == "perturb" for b in branches),
                    recovery=sum(b["type"] == "oracle" for b in branches),
                    success={k: sum(r["kind"] == k for r in rows) for k in KINDS},
                    new_success={k: sum(r["kind"] == k and not r.get("imported_existing") for r in rows) for k in KINDS},
                    new_perturbed_recovery=sum(r["kind"] == "oracle_recovery" and
                        r.get("collection_origin") == "controlled_robot_perturbation" and
                        not r.get("imported_existing") for r in rows))

    @staticmethod
    def _attempt(state, task, seed):
        return next(a for a in state["attempts"] if a["task"] == task and a["seed"] == seed)

    def needs(self, task):
        s = self.snapshot()
        c = self.counts(s, task)["success"]
        return {k: c[k] < v for k, v in targets(s["config"], task).items()}

    def reserve_seed(self, task, seed):
        with self._transaction() as s:
            cfg = s["config"]["tasks"][task]
            if s["task_status"][task] not in {"validating", "collecting"}:
                raise ValueError("Task is stopped")
            if not cfg["seed_start"] <= seed < cfg["seed_start"]+cfg["natural_limit"]:
                raise ValueError("Seed outside reserved range")
            if any(a["task"] == task and a["seed"] == seed for a in s["attempts"]):
                raise ValueError("Seed already consumed")
            if self.counts(s, task)["natural"] >= cfg["natural_limit"]:
                raise ValueError("Natural allowance exhausted")
            directory = self.root/task/"attempts"/str(seed)
            s["attempts"].append(dict(task=task, seed=seed, directory=str(directory), status="running",
                reserved_at=now(), branches=[], result_paths=[], revision=len(s["revisions"])-1))
        return directory

    def reserve_branch(self, task, seed, branch_type, *, trial=None):
        if branch_type not in {"perturb", "oracle"}:
            raise ValueError("Unknown branch type")
        with self._transaction() as s:
            a = self._attempt(s, task, seed)
            if a["status"] != "running":
                raise ValueError("Seed is no longer running")
            cfg = s["config"]["tasks"][task]
            count = self.counts(s, task)
            if branch_type == "oracle":
                if count["recovery"] >= cfg["recovery_limit"] or any(b["type"] == "oracle" for b in a["branches"]):
                    return None
            else:
                previous = [b for b in a["branches"] if b["type"] == "perturb"]
                if count["perturb"] >= cfg["perturb_limit"] or len(previous) >= cfg["max_perturb_per_seed"]:
                    return None
                if trial is None or any(b["trial"] == trial for b in previous):
                    raise ValueError("Missing or repeated perturbation")
            identifier = f"{branch_type}-{sum(b['type'] == branch_type for b in a['branches'])+1:02d}"
            a["branches"].append(dict(id=identifier, type=branch_type, trial=trial,
                                      status="reserved", reserved_at=now()))
        return identifier

    def record_result(self, task, seed, path, branch_id=None):
        path = str(Path(path).resolve())
        with self._transaction() as s:
            a = self._attempt(s, task, seed)
            if path not in a["result_paths"]:
                a["result_paths"].append(path)
            if branch_id:
                b = next(b for b in a["branches"] if b["id"] == branch_id)
                b.update(status="completed", result_path=path)

    def finish_seed(self, task, seed, result):
        with self._transaction() as s:
            a = self._attempt(s, task, seed)
            a.update(status=result["status"], result=result, finished_at=now())
            for b in a["branches"]:
                if b["status"] == "reserved":
                    b["status"] = "interrupted"
            if result["status"] == "error":
                s["task_status"][task] = "technical_error"

    def record_export(self, row):
        key = (row["task"], row["seed"], row["kind"])
        with self._transaction() as s:
            existing = next((r for r in s["exports"] if (r["task"], r["seed"], r["kind"]) == key), None)
            if existing:
                if existing["trace_sha256"] != row["trace_sha256"]:
                    raise ValueError("Duplicate seed/kind cannot replace an accepted sample")
                return
            count = self.counts(s, row["task"])["success"][row["kind"]]
            if count >= targets(s["config"], row["task"])[row["kind"]]:
                raise ValueError("Success quota already reached")
            s["exports"].append(deepcopy(row))
            c = self.counts(s, row["task"])
            if (s["task_status"][row["task"]] == "validating" and c["new_success"]["normal"] >= 1
                    and c["new_success"]["self_corrected"] >= 1
                    and c["new_success"]["oracle_recovery"] >= 3 and c["new_perturbed_recovery"] >= 1):
                s["task_status"][row["task"]] = "collecting"

    def next_seed(self):
        with self._transaction() as s:
            order = list(s["config"]["tasks"])
            if s.get("last_task") in order:
                pos = (order.index(s["last_task"])+1) % len(order)
                order = order[pos:]+order[:pos]
            for task in order:
                if s["task_status"][task] not in {"validating", "collecting"}:
                    continue
                cfg, c = s["config"]["tasks"][task], self.counts(s, task)
                need = {k: c["success"][k] < v for k, v in targets(s["config"], task).items()}
                if not any(need.values()):
                    s["task_status"][task] = "complete"
                    continue
                if (c["natural"] >= cfg["natural_limit"] or
                    (need["oracle_recovery"] and c["recovery"] >= cfg["recovery_limit"]) or
                    (need["self_corrected"] and c["perturb"] >= cfg["perturb_limit"])):
                    s["task_status"][task] = "allowance_exhausted"
                    continue
                used = {a["seed"] for a in s["attempts"] if a["task"] == task}
                seed = next(i for i in range(cfg["seed_start"], cfg["seed_start"]+cfg["natural_limit"]) if i not in used)
                s["last_task"] = task
                return task, seed
        return None

    def set_status(self, status, **extra):
        with self._transaction() as s:
            s.update(status=status, **extra)

    def stop_task(self, task, reason):
        with self._transaction() as s:
            s["task_status"][task] = "technical_error"
            s.setdefault("task_errors", {})[task] = reason

    def record_import(self, source, status, **extra):
        with self._transaction() as s:
            s["imports"].append(dict(source=str(source), status=status, **extra))
