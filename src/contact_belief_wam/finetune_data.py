"""Freeze HDMI mixed data into a separate fine-tuning dataset."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .paths import atomic_json, read_jsonl, sha256


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_source(run):
    from .config import ContactConfig

    root = Path(run["finetune_from"])
    required = ["config.json", "model.safetensors", "norm_stats.json", "wm_norm_stats.json", "deployment.json"]
    if run["finetune_use_ema"]:
        required.append("ema.safetensors")
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Missing fine-tuning source file: {root / name}")
    cfg = ContactConfig.load(root / "config.json")
    deployment = json.loads((root / "deployment.json").read_text())
    if (cfg.stage != "stage2" or cfg.mode != "full" or not cfg.aligned or not cfg.independent_tactile
            or cfg.camera_mode != "head_only" or cfg.k != 10 or cfg.chunk_size != 50
            or deployment.get("task") != "insert_HDMI"):
        raise ValueError("Expected the head-only HDMI Stage-2 TWAM, k=10 / chunk=50")
    for field, filename in (("policy_norm_stats", "norm_stats.json"), ("world_norm_stats", "wm_norm_stats.json")):
        norm = json.loads((root / filename).read_text())
        if getattr(cfg, field) != norm.get("norm_stats", norm):
            raise ValueError(f"Source checkpoint normalization disagrees with {filename}")
    return {"directory": str(root), "config_sha256": sha256(root / "config.json"),
            "files": {name: (root / name).stat().st_size for name in required},
            "camera": cfg.camera_mode, "use_ema": run["finetune_use_ema"]}


def split_groups(rows, *, seed, recovery_groups=5, control_groups=2):
    if recovery_groups == control_groups == 0:
        return []
    groups = {}
    for row in rows:
        groups.setdefault(row["group_id"], set()).add(row["kind"])
    ordered = sorted(groups, key=lambda group: digest([seed, group]))
    recoveries = [group for group in ordered if "oracle_recovery" in groups[group]]
    controls = [group for group in ordered if "oracle_recovery" not in groups[group]]
    if not 0 < recovery_groups < len(recoveries) or not 0 <= control_groups < len(controls):
        raise ValueError("Not enough independent groups for the configured train/validation split")
    return sorted(recoveries[:recovery_groups] + controls[:control_groups])


def _write_rows(path, rows):
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows))
    temporary.replace(path)


def prepare_model_cache(row, root):
    """Validate raw evidence once, then materialize exactly the model's RGB input."""
    import numpy as np
    from .takeover_dataset import load_episode

    root = Path(root)
    metadata = root / "metadata.json"
    if metadata.exists():
        meta = json.loads(metadata.read_text())
        if meta["trace_sha256"] != row["trace_sha256"] or meta["cameras"] != ["head"]:
            raise ValueError("Existing model-input cache belongs to different data")
    else:
        episode = load_episode(row["path"], row, row["feature_cache_root"], cameras=("head",))
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".model-input-", dir=root.parent) as tmp:
            tmp = Path(tmp)
            arrays = {"joints": episode["joints"], "commands": episode["commands"], "head": episode["images"]["head"]}
            for name, array in arrays.items():
                np.save(tmp / f"{name}.npy", array, allow_pickle=False)
            meta = dict(schema="takeover-model-input-v1", trace_sha256=row["trace_sha256"],
                        cameras=["head"], frames=row["frames"],
                        files={p.name: sha256(p) for p in tmp.glob("*.npy")})
            atomic_json(tmp / "metadata.json", meta)
            root.mkdir(exist_ok=True)
            for p in tmp.iterdir():
                p.replace(root / p.name)
    checksums = {str(root / name): expected for name, expected in meta["files"].items()}
    checksums[str(metadata)] = sha256(metadata)
    for path, expected in checksums.items():
        if sha256(path) != expected:
            raise ValueError(f"Model-input cache changed: {path}")
    return checksums


def write_submission_lists(run, report):
    """Separate new upload inputs from dependencies already used by the cloud run."""
    from .paths import PROJECT

    root = Path(run["finetune_data_dir"])
    upload = set(report["prepared_sha256"]) | {str(root / "prepared.json"), run["config"]}
    runtime = set(upload)
    for row in read_jsonl(report["shared_manifest"]):
        cache = Path(row.get("feature_cache_root", run["tactile_cache"])) / row["episode_id"]
        candidates = [cache.with_name(cache.name + suffix) for suffix in (".npz", ".tokens.npy", ".tokens.json")]
        for path in candidates:
            if path.is_file():
                runtime.add(str(path))
                if row.get("data_format"):
                    upload.add(str(path))
        if not row.get("model_input_cache"):
            runtime.add(row["path"])
    code = list((PROJECT / "src/contact_belief_wam").rglob("*.py"))
    code += [PROJECT / path for path in ("pyproject.toml", "scripts/cloud_train.sh", "scripts/train.sh", "scripts/finetune_hdmi.sh",
                                        "scripts/finetune_hdmi_continue.sh", "scripts/finetune_hdmi_fresh.sh",
                                        "docs/hdmi_recovery_finetune.md")]
    upload.update(str(p) for p in code if p.is_file())
    runtime.update(upload)
    runtime.update(str(Path(run["finetune_from"]) / name) for name in report["source_checkpoint"]["files"])
    runtime.add(run["tokenizer_path"])
    runtime.update(str(Path(run["pi05_base"]) / name) for name in ("model.safetensors", "config.json"))
    (root / "training_files.txt").write_text("".join(path + "\n" for path in sorted(runtime)))
    workspace = Path(run["workspace"])
    (root / "upload_files.txt").write_text("".join(str(Path(path).relative_to(workspace)) + "\n" for path in sorted(upload)))


def prepare_finetune_data(run):
    task = "insert_HDMI"
    source = verify_source(run)
    root = Path(run["finetune_data_dir"])
    root.mkdir(parents=True, exist_ok=True)
    prepared = root / "prepared.json"
    settings = run.get("finetune_validation", {"recovery_groups": 0, "control_groups": 0})
    preparation_recipe = dict(source=source, validation=settings, split_seed=run["seed"],
                              original_cache=run["tactile_cache"], recovery_cache=run["recovery_cache"],
                              data_root=run["data_root"])
    if run.get("finetune_prepared_from"):
        preparation_recipe["prepared_from"] = run["finetune_prepared_from"]
        if not prepared.exists():
            # Preserve the exact previous manifests, labels and model-input caches.
            # Cloud continuation needs no live collection ledger or raw review videos.
            previous = json.loads((Path(run["finetune_prepared_from"]) / "prepared.json").read_text())
            for key in ("validation", "split_seed", "original_cache", "recovery_cache", "data_root"):
                if previous["input_identity"].get(key) != preparation_recipe[key]:
                    raise ValueError(f"Frozen continuation data differs in {key}")
            for path, expected in previous["prepared_sha256"].items():
                if sha256(path) != expected:
                    raise ValueError(f"Prepared fine-tuning data changed: {path}")
            for old, name in ((previous["task_stats"][task], "norm_stats.json"),
                              (previous["wm_stats"], "wm_norm_stats.json")):
                if json.loads(Path(old).read_text()) != json.loads((Path(run["finetune_from"]) / name).read_text()):
                    raise ValueError("Continuation checkpoint normalization differs from frozen data")
            report = dict(previous, input_identity={**previous["input_identity"], **preparation_recipe},
                          source_checkpoint=source, training_started=False)
            atomic_json(prepared, report)
    if prepared.exists():
        # The frozen manifests are authoritative on resume/cloud submission.
        # Do not require the live collector's ledger or raw diagnostic videos.
        report = json.loads(prepared.read_text())
        if any(report["input_identity"].get(key) != value for key, value in preparation_recipe.items()):
            raise ValueError("Fine-tuning preparation settings changed; choose a new finetune_data_dir")
        for path, expected in report["prepared_sha256"].items():
            if sha256(path) != expected:
                raise ValueError(f"Prepared fine-tuning data changed: {path}")
        for row in read_jsonl(report["shared_manifest"]):
            cache = Path(row.get("feature_cache_root", run["tactile_cache"])) / (row["episode_id"] + ".npz")
            if not cache.is_file() or (not row.get("model_input_cache") and not Path(row["path"]).is_file()):
                raise FileNotFoundError(f"Missing training episode/cache: {row['episode_id']}")
        write_submission_lists(run, report)
        return report
    originals = read_jsonl(run["original_manifest"])
    if len(originals) != 100 or any(r["task"] != task or r["source"] != "original" for r in originals):
        raise ValueError("Expected the existing 100 HDMI Original expert trajectories")
    mixed = sorted((r for r in read_jsonl(run["recovery_manifest"]) if r["task"] == task),
                   key=lambda r: r["episode_id"])
    counts = dict(Counter(r["kind"] for r in mixed))
    if counts != {"oracle_recovery": 25, "normal": 10, "self_corrected": 10}:
        raise ValueError(f"Expected the completed 45 HDMI samples; found {counts}")
    ids = [(r["group_id"], r["kind"]) for r in mixed]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate seed/category in the HDMI review data")
    if any(r.get("training_enabled") is not False or not r.get("suitable_for_imitation") for r in mixed):
        raise ValueError("Source must be immutable, technically validated review data")
    for key, filename in (("task_stats", "norm_stats.json"), ("wm_stats", "wm_norm_stats.json")):
        expected = json.loads((Path(run["finetune_from"]) / filename).read_text())
        for path in {r[key] for r in mixed}:
            if json.loads(Path(path).read_text()) != expected:
                raise ValueError(f"Recovery data normalization differs from the source TWAM: {path}")
    identity = dict(format_version=2, original_sha256=sha256(run["original_manifest"]), mixed_rows_sha256=digest(mixed),
                    **preparation_recipe)
    held = set(split_groups(mixed, seed=run["seed"], **settings))
    if bool(held) != bool(run["stage1"]["validate_every"]):
        raise ValueError("Validation groups and Stage-1 validation setting must agree")
    rows, required, cached = [], set(), {}
    for original in originals:
        row = dict(original)
        row.update(path=str((Path(run["data_root"]) / row["path"]).resolve()), split="train",
                   group_id=row["episode_id"], kind="original_expert")
        rows.append(row)
    for original in mixed:
        row = dict(original)
        row.update(split="val" if row["group_id"] in held else "train", training_enabled=True,
                   training_approved=True, feature_cache_root=run["recovery_cache"],
                   preparation="hdmi-two-stage-finetune-v1")
        model_cache = root / "model_inputs" / row["episode_id"]
        cached.update(prepare_model_cache(row, model_cache))
        row["model_input_cache"] = str(model_cache)
        rows.append(row)
    for row in rows:
        cache = Path(row.get("feature_cache_root", run["tactile_cache"])) / (row["episode_id"] + ".npz")
        for path in (Path(row["path"]), cache):
            if not path.is_file():
                raise FileNotFoundError(f"Upload the training dependency: {path}")
            required.add(str(path))
    train_rows = [r for r in rows if r["split"] == "train"]
    assert not {r["group_id"] for r in train_rows}.intersection(held)
    shared, policy, validation = root / "stage1.jsonl", root / "stage2_train.jsonl", root / "validation.jsonl"
    for path, selected in ((shared, rows), (policy, train_rows), (validation, [r for r in rows if r["split"] == "val"])):
        _write_rows(path, selected)
    stats, wm_stats = root / "norm_stats.json", root / "wm_norm_stats.json"
    for name, path in (("norm_stats.json", stats), ("wm_norm_stats.json", wm_stats)):
        shutil.copy2(Path(run["finetune_from"]) / name, path)
    files = [shared, policy, validation, stats, wm_stats]
    required.update(str(Path(run["finetune_from"]) / name) for name in source["files"])
    required.update(map(str, files))
    required.update(cached)
    required.update(run[key] for key in ("tokenizer_path", "original_manifest", "recovery_manifest", "config"))
    required.update(r[key] for r in mixed for key in ("task_stats", "wm_stats"))
    for name in ("model.safetensors", "config.json"):
        required.add(str(Path(run["pi05_base"]) / name))
    (root / "required_files.txt").write_text("".join(path + "\n" for path in sorted(required)))
    report = dict(input_identity=identity, source_checkpoint=source, shared_manifest=str(shared),
                  policy_manifests={task: str(policy)}, task_stats={task: str(stats)}, wm_stats=str(wm_stats),
                  data_root=run["data_root"], cache_root=run["tactile_cache"], tokenizer_path=run["tokenizer_path"],
                  episode_fingerprint=digest(rows), policy_fingerprints={task: digest(train_rows)},
                  validation_episode_ids=[r["episode_id"] for r in rows if r["split"] == "val"],
                  validation_groups=sorted(held), counts={split: dict(Counter(r["kind"] for r in rows if r["split"] == split))
                                                        for split in ("train", "val")},
                  prepared_sha256={**{str(p): sha256(p) for p in files}, **cached},
                  review_source_unchanged=True, training_started=False)
    atomic_json(prepared, report)
    write_submission_lists(run, report)
    return report


def check_loaders(run):
    """CPU-only checks of real recorded samples at supervision boundaries."""
    import numpy as np
    from .train import stage_run
    from .trainer import _load_dataset

    checks = []
    for stage in (("stage2",) if run.get("finetune_stage2_only") else ("stage1", "stage2")):
        dataset = _load_dataset(stage_run(run, stage, "insert_HDMI" if stage == "stage2" else None), stage)
        try:
            indexed = {}
            for index, (rid, _, _, frame) in enumerate(dataset.index):
                indexed.setdefault(rid, {})[frame] = index
            for rid, row in enumerate(dataset.rows):
                available = indexed[rid]
                anchors = {min(available), max(available)}
                for interval in row.get("policy_supervision_ranges", []):
                    anchors.update(t for t in (interval["start"] - 1, interval["start"], interval["stop"] - 1)
                                   if t in available)
                poison_checks = 0
                for frame in sorted(anchors):
                    sample = dataset[available[frame]]
                    if "wrist_image" in sample or any(k in sample for k in ("ground_truth", "prism_pose", "slot_pose", "controller", "kind")):
                        raise ValueError("Unexpected camera or privileged information in student input")
                    if not np.isfinite(sample["actions"]).all() or not np.isfinite(sample["tactile"]).all():
                        raise ValueError("Non-finite training tensor")
                    if stage == "stage2" and not sample["policy_action_mask"][-1].any():
                        raise ValueError("Stage-2 sampler included an unsupervised anchor")
                    # Alter future commands beyond the allowed prefix; current
                    # flow targets must not change. WM dynamics may change.
                    if row.get("data_format") and stage == "stage2":
                        end = next(r["stop"] for r in row["policy_supervision_ranges"] if r["start"] <= frame < r["stop"])
                        commands = dataset._load(rid)["commands"]
                        saved = commands[end:].copy()
                        try:
                            commands[end:] += 123
                            changed = dataset[available[frame]]
                            np.testing.assert_array_equal(sample["actions"], changed["actions"])
                            np.testing.assert_array_equal(sample["executed_actions"], changed["executed_actions"])
                            poison_checks += 1
                        finally:
                            commands[end:] = saved
                checks.append(dict(stage=stage, episode_id=row["episode_id"], anchors=len(anchors),
                                   excluded_future_checks=poison_checks))
        finally:
            dataset.close()
    return dict(passed=True, camera="head_only", checks=checks, training_started=False)


def main():
    from .train import resolve_config, _prepare
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/finetune_insert_HDMI_4xa800.yaml")
    parser.add_argument("--check-loaders", action="store_true")
    args = parser.parse_args()
    run = resolve_config(args.config)
    report = _prepare(run)
    if args.check_loaders:
        checks = check_loaders(run)
        atomic_json(Path(run["finetune_data_dir"]) / "loader_checks.json", checks)
        report = {"counts": report["counts"], "passed": checks["passed"], "episodes_checked": len(checks["checks"])}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
