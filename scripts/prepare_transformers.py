#!/usr/bin/env python3
"""Materialize the pinned WAVE-0 Transformers dependency, without installing it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCE = json.loads((ROOT / "vendor/SOURCE.json").read_text())["transformers"]


def prepare(archive=None):
    target = ROOT / "vendor/transformers"
    marker = target / ".twam-source.json"
    if target.exists():
        if marker.is_file() and json.loads(marker.read_text()) == SOURCE:
            print(f"Pinned Transformers already prepared: {target}")
            return
        raise FileExistsError(f"Refusing to replace existing {target}; inspect or move it first")
    with tempfile.TemporaryDirectory(prefix="twam-transformers-", dir=ROOT / "vendor") as tmp:
        tmp = Path(tmp)
        if archive is None:
            archive = tmp / "source.tar.gz"
            with urllib.request.urlopen(SOURCE["archive_url"], timeout=120) as response, archive.open("wb") as out:
                shutil.copyfileobj(response, out)
        archive = Path(archive).resolve()
        digest = hashlib.sha256()
        with archive.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != SOURCE["archive_sha256"]:
            raise ValueError("Transformers archive SHA-256 does not match vendor/SOURCE.json")
        extracted = tmp / "source"
        extracted.mkdir()
        # Only regular source files and directories; reject paths escaping the
        # staging directory. This also works on Python 3.10 before filter='data'.
        with tarfile.open(archive) as tar:
            for member in tar:
                relative = Path(member.name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe archive path: {member.name}")
                destination = extracted / relative
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as source, destination.open("wb") as out:
                        shutil.copyfileobj(source, out)
        roots = list(extracted.iterdir())
        if len(roots) != 1 or not (roots[0] / "setup.py").is_file():
            raise ValueError("Unexpected Transformers source layout")
        roots[0].rename(target)
        marker.write_text(json.dumps(SOURCE, indent=2) + "\n")
    print(f"Prepared Transformers {SOURCE['revision']} at {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a downloaded archive instead of network access")
    prepare(parser.parse_args().archive)
