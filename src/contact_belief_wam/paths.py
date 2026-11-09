from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def _is_project_root(path: Path) -> bool:
    try:
        return ((path / "pyproject.toml").is_file() and
                (path / "src/contact_belief_wam").is_dir() and
                (path / "scripts/cloud_train.sh").is_file())
    except OSError:
        # Automatic discovery scans unrelated ancestors and siblings too; an
        # inaccessible candidate must not prevent checking the remaining ones.
        return False


def _parents(start: Path):
    yield start
    yield from start.parents


def discover_project_root(module_file=__file__, cwd=None, prefix=None, env=None) -> Path:
    """Find the checkout root for source runs and wheel installs without trusting site-packages depth."""
    env = os.environ if env is None else env
    explicit = env.get("CBWAM_ROOT")
    if explicit:
        try:
            root = Path(explicit).expanduser().resolve()
        except OSError as exc:
            raise RuntimeError(f"Could not access explicit CBWAM_ROOT={explicit}: {exc}") from exc
        if not _is_project_root(root):
            raise RuntimeError(
                f"CBWAM_ROOT={root} is not accessible or is not a Contact-Belief WAM project root; expected "
                "pyproject.toml, src/contact_belief_wam/, and scripts/cloud_train.sh"
            )
        return root

    module_input = Path(module_file).expanduser()
    module_path = module_input.resolve()
    cwd_path = Path(cwd or Path.cwd()).expanduser().resolve()
    prefix_input = Path(prefix or sys.prefix).expanduser()
    prefix_path = prefix_input.resolve()

    # A source checkout or a project-local .venv wheel install resolves through
    # an ancestor with the project markers. Validate every candidate so that
    # site-packages/lib/pythonX.Y can never be mistaken for the checkout.
    for start in (module_path.parent, module_input.absolute().parent, cwd_path,
                  prefix_path, prefix_input.absolute()):
        for candidate in _parents(start):
            if _is_project_root(candidate):
                return candidate

    # Also support a separate environment placed beside the checkout, such as
    # /workspace/.venv and /workspace/contact-belief-wam.
    neighbor_candidates = set()
    neighbor_bases = {prefix_path.parent, prefix_path.parent.parent,
                      prefix_input.absolute().parent, prefix_input.absolute().parent.parent}
    for base in neighbor_bases:
        try:
            if not base.is_dir():
                continue
            for candidate in base.iterdir():
                try:
                    if not candidate.is_dir():
                        continue
                    resolved = candidate.resolve()
                except OSError:
                    continue
                if _is_project_root(resolved):
                    neighbor_candidates.add(resolved)
        except OSError:
            continue
    if len(neighbor_candidates) == 1:
        return next(iter(neighbor_candidates))
    if len(neighbor_candidates) > 1:
        choices = ", ".join(sorted(map(str, neighbor_candidates)))
        raise RuntimeError(f"Multiple Contact-Belief WAM project roots found beside Python environment: {choices}; set CBWAM_ROOT")

    raise RuntimeError(
        "Could not locate the Contact-Belief WAM project root. Set CBWAM_ROOT to the checkout directory "
        "(it must contain pyproject.toml, src/contact_belief_wam/, and scripts/cloud_train.sh), "
        "or run from a project checkout."
    )


PROJECT = discover_project_root()
WORKSPACE = Path(os.environ.get("TWAM_ROOT", PROJECT.parent)).resolve()


def configure():
    vendor = PROJECT / "vendor/wave0/src"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    defaults = {"HF_HOME": PROJECT / ".cache/huggingface", "TORCH_HOME": PROJECT / ".cache/torch",
                "MPLCONFIGDIR": PROJECT / ".cache/matplotlib", "TOKENIZERS_PARALLELISM": "false"}
    for key, value in defaults.items():
        os.environ.setdefault(key, str(value))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        tmp = Path(f.name)
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        tmp = Path(f.name)
    tmp.replace(path)
