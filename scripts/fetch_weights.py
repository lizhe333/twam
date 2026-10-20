#!/usr/bin/env python3
"""Fetch public WAVE-0 weights and the matching public PaliGemma tokenizer.

Downloads are pinned to immutable Hugging Face revisions. Xet is disabled so
downloads use normal HTTP. Old partial Xet files are removed before the first
HTTP attempt because they may be sparse or non-contiguous and cannot be resumed
as an HTTP prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import struct
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
LOG_PATH = ROOT / "reports" / "downloads.log"

os.environ["HF_HUB_DISABLE_XET"] = "1"
# Stay on huggingface_hub's standard HTTP path.  The optional Rust
# hf_transfer backend currently fails against this repo's Xet bridge endpoint
# in the development network and does not retain a resumable local prefix.
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "600"

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

REPOSITORIES = {
    "pi05_base": {
        "repo_id": "lerobot/pi05_base",
        "revision": "22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0",
        "destination": MODELS / "pi05_base",
    },
    "wave0_robotwin": {
        "repo_id": "KiyotakaWang/WAVE-0-RoboTwin",
        "revision": "d0568d0329b8824bd496bb893c3b6a702ed38189",
        "destination": MODELS / "wave0_robotwin",
    },
}
TOKENIZER_URL = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
WORLD_COMPONENTS = ("model.world_model.", "model.world_action_encoder.", "model.world_state_encoder.")


def configure_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )


def remove_old_incomplete_files(destination: Path) -> int:
    cache = destination / ".cache" / "huggingface" / "download"
    partials = list(cache.rglob("*.incomplete")) if cache.exists() else []
    for partial in partials:
        size = partial.stat().st_size
        logging.warning("Removing stale partial before HTTP download: %s (%d bytes)", partial, size)
        partial.unlink()
    return len(partials)


def prepare_http_download(destination: Path, revision: str) -> None:
    marker = destination / ".http-download.json"
    expected = {"revision": revision, "method": "huggingface_hub regular HTTP; Xet disabled"}
    if marker.is_file():
        try:
            if json.loads(marker.read_text(encoding="utf-8")) == expected:
                logging.info("Resuming this script's regular HTTP partials in %s", destination)
                return
        except (OSError, json.JSONDecodeError):
            pass
    remove_old_incomplete_files(destination)
    marker.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_snapshot(name: str, spec: dict[str, object], api: HfApi) -> None:
    repo_id = str(spec["repo_id"])
    revision = str(spec["revision"])
    destination = Path(spec["destination"])
    destination.mkdir(parents=True, exist_ok=True)

    # Only reuse partials marked by this script. Old unmarked Xet partials may
    # be sparse/non-contiguous, so they must be discarded before switching to HTTP.
    prepare_http_download(destination, revision)
    for attempt in range(1, 13):
        try:
            info = api.model_info(repo_id=repo_id, revision=revision, files_metadata=True)
            if info.sha != revision:
                raise RuntimeError(f"Resolved {repo_id}@{revision} to unexpected revision {info.sha}")
            expected = {item.rfilename: item for item in info.siblings}
            logging.info("Fetching %s at %s (%d files)", repo_id, revision, len(expected))
            for filename, item in sorted(expected.items()):
                if item.size is not None:
                    logging.info("  %s: %d bytes", filename, item.size)

            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=str(destination),
                max_workers=4,
                etag_timeout=600,
                force_download=False,
            )

            files = []
            for filename, item in sorted(expected.items()):
                path = destination / filename
                if not path.is_file():
                    raise FileNotFoundError(f"Missing downloaded model file: {path}")
                actual_size = path.stat().st_size
                if item.size is not None and actual_size != item.size:
                    raise IOError(f"Size mismatch for {path}: {actual_size} != {item.size}")
                remote_sha256 = getattr(getattr(item, "lfs", None), "sha256", None)
                actual_sha256 = sha256_file(path)
                if remote_sha256 and actual_sha256 != remote_sha256:
                    raise IOError(f"SHA-256 mismatch for {path}: {actual_sha256} != {remote_sha256}")
                files.append(
                    {
                        "path": filename,
                        "size": actual_size,
                        "sha256": actual_sha256,
                        "hf_lfs_sha256": remote_sha256,
                        "hf_blob_id": item.blob_id,
                    }
                )

            manifest = {
                "repo_id": repo_id,
                "revision": revision,
                "source": f"https://huggingface.co/{repo_id}/tree/{revision}",
                "download_method": "huggingface_hub HTTP with HF_HUB_DISABLE_XET=1",
                "files": files,
            }
            (destination / "DOWNLOAD_SOURCE.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            (destination / ".http-download.json").unlink(missing_ok=True)
            logging.info("Finished %s; verified %d files by size and recorded SHA-256.", name, len(files))
            return
        except Exception:
            logging.exception("Attempt %d/12 failed downloading %s", attempt, repo_id)
            if attempt == 12:
                raise
            time.sleep(min(300, 5 * 2 ** (attempt - 1)))


def fetch_world_subset(api: HfApi) -> None:
    """Range-fetch the contiguous world-model tensors from the official checkpoint.

    The RoboTwin checkpoint stores the world model, action encoder, and state
    encoder in one contiguous region of model.safetensors.  Fetching this
    region avoids downloading the unrelated policy and vision tower weights.
    """
    import requests

    spec = REPOSITORIES["wave0_robotwin"]
    repo_id = str(spec["repo_id"])
    revision = str(spec["revision"])
    destination = MODELS / "wave0_robotwin_world"
    destination.mkdir(parents=True, exist_ok=True)
    partial_path = destination / ".world-tensors.partial"
    chunk_dir = destination / ".world-tensor-chunks"
    marker_path = destination / ".partial-source.json"
    expected_marker = {"repo_id": repo_id, "revision": revision, "file": "model.safetensors"}
    if marker_path.exists():
        try:
            if json.loads(marker_path.read_text(encoding="utf-8")) != expected_marker:
                partial_path.unlink(missing_ok=True)
                shutil.rmtree(chunk_dir, ignore_errors=True)
        except (OSError, json.JSONDecodeError):
            partial_path.unlink(missing_ok=True)
            shutil.rmtree(chunk_dir, ignore_errors=True)
    else:
        # Unmarked partials may have come from an Xet sparse download.
        partial_path.unlink(missing_ok=True)
        shutil.rmtree(chunk_dir, ignore_errors=True)
    marker_path.write_text(json.dumps(expected_marker, indent=2) + "\n", encoding="utf-8")

    info = api.model_info(repo_id=repo_id, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError(f"Resolved {repo_id}@{revision} to unexpected revision {info.sha}")
    item = next((entry for entry in info.siblings if entry.rfilename == "model.safetensors"), None)
    if item is None or item.size is None:
        raise RuntimeError("Pinned RoboTwin repository has no sized model.safetensors file")
    remote_sha256 = getattr(getattr(item, "lfs", None), "sha256", None)
    if not remote_sha256:
        raise RuntimeError("Hugging Face did not provide an LFS SHA-256 for model.safetensors")

    # Get the safetensors header using a small, verified HTTP Range response.
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/model.safetensors"
    prefix = None
    for attempt in range(1, 13):
        try:
            response = requests.get(
                url,
                headers={"Range": "bytes=0-7", "Accept-Encoding": "identity"},
                timeout=(30, 600),
                stream=True,
            )
            response.raise_for_status()
            if response.status_code != 206 or response.headers.get("Content-Range", "").split("/")[0] != "bytes 0-7":
                raise IOError(f"Server did not honor the safetensors header Range request: {response.status_code}")
            prefix = response.content
            response.close()
            if len(prefix) != 8:
                raise IOError(f"Expected 8 header-prefix bytes, received {len(prefix)}")
            break
        except Exception:
            logging.exception("Attempt %d/12 failed reading the RoboTwin safetensors header prefix", attempt)
            if attempt == 12:
                raise
            time.sleep(min(120, 5 * 2 ** (attempt - 1)))
    header_length = struct.unpack("<Q", prefix)[0]
    if header_length <= 0 or header_length > 64 * 1024 * 1024:
        raise IOError(f"Unreasonable safetensors header length: {header_length}")
    header_start, header_end = 0, 8 + header_length - 1
    header_response = None
    for attempt in range(1, 13):
        try:
            response = requests.get(
                url,
                headers={"Range": f"bytes={header_start}-{header_end}", "Accept-Encoding": "identity"},
                timeout=(30, 600),
                stream=True,
            )
            response.raise_for_status()
            expected_range = f"bytes {header_start}-{header_end}/{item.size}"
            if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                raise IOError(f"Unexpected header response range {response.headers.get('Content-Range')!r}")
            header_response = response.content
            response.close()
            if len(header_response) != header_end - header_start + 1:
                raise IOError(f"Header response length mismatch: {len(header_response)}")
            break
        except Exception:
            logging.exception("Attempt %d/12 failed reading the RoboTwin safetensors header", attempt)
            if attempt == 12:
                raise
            time.sleep(min(120, 5 * 2 ** (attempt - 1)))

    header = json.loads(header_response[8:])
    tensors = {key: value for key, value in header.items() if key.startswith(WORLD_COMPONENTS)}
    if not tensors:
        raise RuntimeError("No world model/action/state tensors found in pinned checkpoint header")
    ordered = sorted((value["data_offsets"][0], value["data_offsets"][1], key) for key, value in tensors.items())
    payload_start = ordered[0][0]
    payload_end = ordered[-1][1]
    previous_end = payload_start
    for start, end, key in ordered:
        if start != previous_end:
            raise RuntimeError(f"Selected world tensors are not contiguous before {key}: {previous_end} != {start}")
        previous_end = end
    payload_size = payload_end - payload_start
    source_start = 8 + header_length + payload_start
    source_end = source_start + payload_size - 1
    if source_end >= item.size:
        raise RuntimeError(f"World tensor byte range exceeds checkpoint size: {source_end} >= {item.size}")
    logging.info(
        "Fetching %d world tensors (%d bytes) from official checkpoint range %d-%d",
        len(tensors), payload_size, source_start, source_end,
    )

    chunk_dir.mkdir(parents=True, exist_ok=True)
    if partial_path.exists():
        # Preserve any verified-prefix bytes from an earlier single-range attempt.
        prefix_path = chunk_dir / "00000.part"
        if not prefix_path.exists() and partial_path.stat().st_size <= 64 * 1024 * 1024:
            os.replace(partial_path, prefix_path)
        else:
            partial_path.unlink(missing_ok=True)
    chunk_size = 64 * 1024 * 1024
    chunk_count = (payload_size + chunk_size - 1) // chunk_size

    def download_chunk(index: int) -> int:
        chunk_start = source_start + index * chunk_size
        chunk_end = min(source_end, chunk_start + chunk_size - 1)
        expected_chunk_size = chunk_end - chunk_start + 1
        final_chunk = chunk_dir / f"{index:05d}.bin"
        partial_chunk = chunk_dir / f"{index:05d}.part"
        if final_chunk.is_file() and final_chunk.stat().st_size == expected_chunk_size:
            return expected_chunk_size
        if final_chunk.exists():
            final_chunk.unlink()
        if partial_chunk.exists() and partial_chunk.stat().st_size > expected_chunk_size:
            partial_chunk.unlink()
        for attempt in range(1, 13):
            done = partial_chunk.stat().st_size if partial_chunk.exists() else 0
            if done == expected_chunk_size:
                os.replace(partial_chunk, final_chunk)
                return expected_chunk_size
            request_start = chunk_start + done
            try:
                response = requests.get(
                    url,
                    headers={"Range": f"bytes={request_start}-{chunk_end}", "Accept-Encoding": "identity"},
                    timeout=(30, 600),
                    stream=True,
                )
                response.raise_for_status()
                expected_range = f"bytes {request_start}-{chunk_end}/{item.size}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                    raise IOError(f"Unexpected data response range {response.headers.get('Content-Range')!r}")
                received = 0
                with partial_chunk.open("ab") as output:
                    for data in response.iter_content(2 * 1024 * 1024):
                        if data:
                            output.write(data)
                            received += len(data)
                response.close()
                if received != expected_chunk_size - done:
                    raise IOError(f"Downloaded chunk tail {received} != expected {expected_chunk_size - done}")
                os.replace(partial_chunk, final_chunk)
                return expected_chunk_size
            except Exception:
                logging.exception(
                    "World tensor HTTP range failed at chunk %d offset %d (attempt %d/12)",
                    index, done, attempt,
                )
                if attempt == 12:
                    raise
                time.sleep(min(120, 5 * 2 ** (attempt - 1)))
        raise RuntimeError(f"Unable to download world tensor chunk {index}")

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as executor:
        for completed, _ in enumerate(executor.map(download_chunk, range(chunk_count)), start=1):
            logging.info("World tensor range chunks completed: %d/%d", completed, chunk_count)

    # Rebuild a valid safetensors file with only selected keys and rebased offsets.
    out_header = {}
    if "__metadata__" in header:
        out_header["__metadata__"] = header["__metadata__"]
    for key, value in tensors.items():
        out_header[key] = {
            "dtype": value["dtype"],
            "shape": value["shape"],
            "data_offsets": [value["data_offsets"][0] - payload_start, value["data_offsets"][1] - payload_start],
        }
    encoded_header = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
    encoded_header += b" " * ((8 - len(encoded_header) % 8) % 8)
    final_path = destination / "model.safetensors"
    assembling = destination / ".model.safetensors.assembling"
    with assembling.open("wb") as output:
        output.write(struct.pack("<Q", len(encoded_header)))
        output.write(encoded_header)
        for index in range(chunk_count):
            chunk_path = chunk_dir / f"{index:05d}.bin"
            expected_chunk_size = min(chunk_size, payload_size - index * chunk_size)
            if not chunk_path.is_file() or chunk_path.stat().st_size != expected_chunk_size:
                raise IOError(f"World tensor chunk {index} is missing or has the wrong size")
            with chunk_path.open("rb") as source:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
    os.replace(assembling, final_path)
    manifest = {
        "repo_id": repo_id,
        "revision": revision,
        "source": f"https://huggingface.co/{repo_id}/blob/{revision}/model.safetensors",
        "download_method": "official HTTPS HTTP Range requests; TLS verification enabled",
        "original_file_size": item.size,
        "original_hf_lfs_sha256": remote_sha256,
        "source_range": [source_start, source_end],
        "source_range_size": payload_size,
        "tensor_count": len(tensors),
        "tensor_prefixes": list(WORLD_COMPONENTS),
        "output": final_path.name,
        "output_size": final_path.stat().st_size,
        "output_sha256": sha256_file(final_path),
        "note": "Subset of official checkpoint; original full-file SHA is recorded but cannot be recomputed from a subset.",
    }
    (destination / "DOWNLOAD_SOURCE.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(chunk_dir)
    partial_path.unlink(missing_ok=True)
    marker_path.unlink(missing_ok=True)
    try:
        from safetensors import safe_open

        with safe_open(str(final_path), framework="pt", device="cpu") as checkpoint:
            output_keys = set(checkpoint.keys())
        if output_keys != set(tensors):
            raise RuntimeError("Rebuilt safetensors key set does not match selected source tensors")
    except ImportError:
        logging.warning("safetensors is not installed in this helper environment; structural output checks passed")
    logging.info("Wrote verified-format world tensor subset to %s", final_path)


def fetch_pi05_base(api: HfApi) -> None:
    """Fetch the pinned PI0.5 checkpoint as resumable HTTP ranges and verify LFS SHA-256."""
    import requests
    from concurrent.futures import ThreadPoolExecutor

    spec = REPOSITORIES["pi05_base"]
    repo_id = str(spec["repo_id"])
    revision = str(spec["revision"])
    destination = Path(spec["destination"])
    destination.mkdir(parents=True, exist_ok=True)
    info = api.model_info(repo_id=repo_id, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError(f"Resolved {repo_id}@{revision} to unexpected revision {info.sha}")
    item = next((entry for entry in info.siblings if entry.rfilename == "model.safetensors"), None)
    if item is None or item.size is None:
        raise RuntimeError("Pinned PI05 repository has no sized model.safetensors file")
    remote_sha256 = getattr(getattr(item, "lfs", None), "sha256", None)
    if not remote_sha256 or len(remote_sha256) != 64:
        raise RuntimeError("Hugging Face did not provide a valid LFS SHA-256 for model.safetensors")

    # Fetch the small immutable config and pre/postprocessor files separately.
    small_files = [entry.rfilename for entry in info.siblings if entry.rfilename != "model.safetensors"]
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(destination),
        allow_patterns=small_files,
        max_workers=4,
        etag_timeout=600,
        force_download=False,
    )
    for filename in small_files:
        path = destination / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned PI05 metadata file: {path}")

    final_path = destination / "model.safetensors"
    if final_path.is_file() and final_path.stat().st_size == item.size:
        actual = sha256_file(final_path)
        if actual == remote_sha256:
            logging.info("Existing PI05 checkpoint has the expected size and official LFS SHA-256.")
            return
        raise IOError(f"Existing checkpoint SHA-256 mismatch: {actual} != {remote_sha256}")

    # Drop the previous single-stream helper partial. It was not written with
    # our range-chunk marker, and a clean chunk set is easier to audit/resume.
    remove_old_incomplete_files(destination)
    (destination / ".http-download.json").unlink(missing_ok=True)
    chunk_dir = destination / ".range-parts"
    marker_path = destination / ".range-download.json"
    marker = {
        "repo_id": repo_id,
        "revision": revision,
        "file": "model.safetensors",
        "size": item.size,
        "sha256": remote_sha256,
    }
    if marker_path.exists():
        try:
            if json.loads(marker_path.read_text(encoding="utf-8")) != marker:
                shutil.rmtree(chunk_dir, ignore_errors=True)
        except (OSError, json.JSONDecodeError):
            shutil.rmtree(chunk_dir, ignore_errors=True)
    elif chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")

    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/model.safetensors"
    chunk_size = 64 * 1024 * 1024
    chunk_count = (item.size + chunk_size - 1) // chunk_size

    def download_chunk(index: int) -> int:
        start = index * chunk_size
        end = min(item.size - 1, start + chunk_size - 1)
        expected_size = end - start + 1
        final_chunk = chunk_dir / f"{index:05d}.bin"
        partial_chunk = chunk_dir / f"{index:05d}.part"
        if final_chunk.is_file() and final_chunk.stat().st_size == expected_size:
            return expected_size
        if final_chunk.exists():
            final_chunk.unlink()
        if partial_chunk.exists() and partial_chunk.stat().st_size > expected_size:
            partial_chunk.unlink()
        for attempt in range(1, 13):
            done = partial_chunk.stat().st_size if partial_chunk.exists() else 0
            if done == expected_size:
                os.replace(partial_chunk, final_chunk)
                return expected_size
            request_start = start + done
            try:
                response = requests.get(
                    url,
                    headers={"Range": f"bytes={request_start}-{end}", "Accept-Encoding": "identity"},
                    timeout=(30, 600),
                    stream=True,
                )
                response.raise_for_status()
                expected_range = f"bytes {request_start}-{end}/{item.size}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                    raise IOError(f"Unexpected PI05 response range {response.headers.get('Content-Range')!r}")
                received = 0
                with partial_chunk.open("ab") as output:
                    for data in response.iter_content(2 * 1024 * 1024):
                        if data:
                            output.write(data)
                            received += len(data)
                response.close()
                if received != expected_size - done:
                    raise IOError(f"Downloaded PI05 chunk tail {received} != expected {expected_size - done}")
                os.replace(partial_chunk, final_chunk)
                return expected_size
            except Exception:
                logging.exception(
                    "PI05 HTTP range failed at chunk %d offset %d (attempt %d/12)",
                    index, done, attempt,
                )
                if attempt == 12:
                    raise
                time.sleep(min(120, 5 * 2 ** (attempt - 1)))
        raise RuntimeError(f"Unable to download PI05 checkpoint chunk {index}")

    logging.info(
        "Fetching PI05 checkpoint from official pinned revision in %d resumable 64 MiB ranges (%d bytes total).",
        chunk_count, item.size,
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        for completed, _ in enumerate(executor.map(download_chunk, range(chunk_count)), start=1):
            logging.info("PI05 range chunks completed: %d/%d", completed, chunk_count)

    assembling = destination / ".model.safetensors.assembling"
    assembling.unlink(missing_ok=True)
    digest = hashlib.sha256()
    with assembling.open("wb") as output:
        for index in range(chunk_count):
            chunk_path = chunk_dir / f"{index:05d}.bin"
            expected_size = min(chunk_size, item.size - index * chunk_size)
            if not chunk_path.is_file() or chunk_path.stat().st_size != expected_size:
                raise IOError(f"PI05 checkpoint chunk {index} is missing or has the wrong size")
            with chunk_path.open("rb") as source:
                while data := source.read(16 * 1024 * 1024):
                    output.write(data)
                    digest.update(data)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != remote_sha256:
        mismatch = {
            **marker,
            "assembled_sha256": actual_sha256,
            "verified": False,
        }
        (destination / "HASH_MISMATCH.json").write_text(json.dumps(mismatch, indent=2) + "\n", encoding="utf-8")
        assembling.unlink(missing_ok=True)
        raise IOError(f"PI05 checkpoint SHA-256 mismatch: {actual_sha256} != {remote_sha256}; refusing to publish")

    os.replace(assembling, final_path)
    (destination / "DOWNLOAD_SOURCE.json").write_text(
        json.dumps(
            {
                "repo_id": repo_id,
                "revision": revision,
                "source": f"https://huggingface.co/{repo_id}/blob/{revision}/model.safetensors",
                "download_method": "official HTTPS 64 MiB HTTP Range requests; TLS verification enabled",
                "size": item.size,
                "hf_lfs_sha256": remote_sha256,
                "sha256": actual_sha256,
                "verified": True,
                "range_workers": 8,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (destination / "HASH_MISMATCH.json").unlink(missing_ok=True)
    shutil.rmtree(chunk_dir)
    marker_path.unlink(missing_ok=True)
    logging.info("Downloaded and verified PI05 base checkpoint (%d bytes, SHA-256 %s).", item.size, actual_sha256)


def download_tokenizer() -> None:
    destination = MODELS / "tokenizer"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "paligemma_tokenizer.model"
    temporary = destination / "paligemma_tokenizer.model.download"

    from sentencepiece import SentencePieceProcessor
    from transformers import GemmaTokenizer

    if not target.is_file():
        for attempt in range(1, 13):
            try:
                temporary.unlink(missing_ok=True)
                request = urllib.request.Request(TOKENIZER_URL, headers={"User-Agent": "WAVE-0-public-checkpoint-fetch/1"})
                with urllib.request.urlopen(request, timeout=600) as response, temporary.open("wb") as output:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                if temporary.stat().st_size < 100_000:
                    raise IOError(f"Tokenizer download is unexpectedly small: {temporary.stat().st_size} bytes")
                processor = SentencePieceProcessor()
                if not processor.Load(str(temporary)):
                    raise IOError("Downloaded tokenizer.model could not be loaded by SentencePiece")
                os.replace(temporary, target)
                break
            except Exception:
                logging.exception("Attempt %d/12 failed downloading the public PaliGemma tokenizer", attempt)
                temporary.unlink(missing_ok=True)
                if attempt == 12:
                    raise
                time.sleep(min(300, 5 * 2 ** (attempt - 1)))

    processor = SentencePieceProcessor(model_file=str(target))
    tokenizer = GemmaTokenizer(vocab_file=str(target), add_bos_token=True, add_eos_token=False)
    tokenizer.save_pretrained(destination)

    prompt = "Task: put the bottle on the shelf, State: 0 127 255;\nAction: "
    sp_tokens = processor.encode(prompt, out_type=int, add_bos=True)
    hf_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    compatible = sp_tokens == hf_tokens
    report = {
        "source_url": TOKENIZER_URL,
        "model_file": str(target.relative_to(ROOT)),
        "size": target.stat().st_size,
        "sha256": sha256_file(target),
        "sentencepiece_vocab_size": processor.get_piece_size(),
        "tokenizer_class": tokenizer.__class__.__name__,
        "prompt_token_ids_match_sentencepiece": compatible,
        "prompt": prompt,
        "sentencepiece_token_ids": sp_tokens,
        "transformers_token_ids": hf_tokens,
    }
    report_path = ROOT / "reports" / "download-tokenizer-compatibility.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if compatible:
        logging.info("PaliGemma tokenizer saved; prompt IDs match SentencePiece exactly.")
    else:
        logging.warning("PaliGemma tokenizers differ for the sample prompt; see %s", report_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["pi05_base", "wave0_robotwin", "wave0_world", "tokenizer"],
        help="Fetch one asset; default fetches PI05, the RoboTwin world subset, and tokenizer.",
    )
    args = parser.parse_args()
    configure_logging()
    api = HfApi()

    if args.only in (None, "pi05_base"):
        fetch_pi05_base(api)
    if args.only == "wave0_robotwin":
        fetch_snapshot("wave0_robotwin", REPOSITORIES["wave0_robotwin"], api)
    if args.only in (None, "wave0_world"):
        fetch_world_subset(api)
    if args.only in (None, "tokenizer"):
        download_tokenizer()


if __name__ == "__main__":
    main()
