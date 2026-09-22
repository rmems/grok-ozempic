#!/usr/bin/env python3
"""Validate and load versioned, order-preserving Grok-1 text input windows.

This is deliberately separate from ``grok1_block0_experiment.token_ids``.  It
does not tokenize text: an operator must freeze the tokenizer output and all
provenance before this module will expose a window to an experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "grok1-ordered-inputs-v1"
INPUT_MODE = "ordered_text"
PRODUCTION_TOKENS = 8192
SEED = 20260806


class InputManifestError(ValueError):
    """The input artifact is unsafe or does not match its manifest."""


@dataclass(frozen=True)
class OrderedWindow:
    window_id: str
    partition: str
    document_id: str
    token_ids: np.ndarray


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_order_sha256(ids: np.ndarray) -> str:
    """Hash token count and ordered, canonical little-endian int64 values."""
    values = np.asarray(ids, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(values.size.to_bytes(8, "little"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def token_content_sha256(ids: np.ndarray) -> str:
    """Order-independent multiset hash (duplicates remain significant)."""
    return token_order_sha256(np.sort(np.asarray(ids, dtype=np.int64)))


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputManifestError(f"{name} must be an object")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputManifestError(f"{name} must be a non-empty string")
    return value


def _relative_file(root: Path, value: Any, name: str) -> Path:
    rel = Path(_text(value, name))
    if rel.is_absolute() or ".." in rel.parts:
        raise InputManifestError(f"{name} must be a portable relative path")
    path = root / rel
    if not path.is_file():
        raise InputManifestError(f"{name} does not exist: {rel}")
    return path


def _verify_artifact(root: Path, spec: dict[str, Any], name: str) -> Path:
    path = _relative_file(root, spec.get("path"), f"{name}.path")
    expected = _text(spec.get("sha256"), f"{name}.sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise InputManifestError(f"{name} digest mismatch: expected {expected}, got {actual}")
    return path


def _load_ids(path: Path, vocab_size: int, window_id: str) -> np.ndarray:
    try:
        ids = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise InputManifestError(f"{window_id}: cannot load token array: {exc}") from exc
    if ids.ndim != 1:
        raise InputManifestError(f"{window_id}: token array must be one-dimensional")
    if ids.dtype.kind not in "iu":
        raise InputManifestError(f"{window_id}: token dtype must be integer, got {ids.dtype}")
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= vocab_size):
        raise InputManifestError(
            f"{window_id}: token IDs must be in [0, {vocab_size}), "
            f"got [{int(ids.min())}, {int(ids.max())}]"
        )
    # Return an owned, read-only canonical array: callers cannot mutate evidence.
    result = np.array(ids, dtype=np.int64, copy=True)
    result.flags.writeable = False
    return result


def load_ordered_windows(
    manifest_path: Path, *, allow_fixture: bool = False
) -> tuple[dict[str, Any], tuple[OrderedWindow, ...]]:
    """Fail closed while validating provenance, splits, hashes, and token IDs.

    ``allow_fixture`` permits short, explicitly labelled synthetic manifests for
    tests. It must never be used to qualify a production comparison.
    """
    manifest_path = Path(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputManifestError(f"cannot read manifest: {exc}") from exc
    manifest = _object(manifest, "manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise InputManifestError(f"schema_version must be {SCHEMA_VERSION!r}")
    if manifest.get("input_mode") != INPUT_MODE:
        raise InputManifestError("input_mode must be 'ordered_text'; mixed/sampled evidence is rejected")
    if manifest.get("selection_seed") != SEED:
        raise InputManifestError(f"selection_seed must be {SEED}")

    fixture = manifest.get("fixture") is True
    if fixture and not allow_fixture:
        raise InputManifestError("fixture manifests are not production experiment inputs")
    if not fixture and manifest.get("fixture") is not False:
        raise InputManifestError("fixture must explicitly be false for production inputs")

    root = manifest_path.resolve().parent
    tokenizer = _object(manifest.get("tokenizer"), "tokenizer")
    for field in ("identity", "revision", "special_tokens_policy"):
        _text(tokenizer.get(field), f"tokenizer.{field}")
    vocab_size = tokenizer.get("vocab_size")
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0:
        raise InputManifestError("tokenizer.vocab_size must be a positive integer")
    _verify_artifact(
        root,
        _object(tokenizer.get("model_file"), "tokenizer.model_file"),
        "tokenizer.model_file",
    )

    source = _object(manifest.get("source"), "source")
    for field in ("identity", "revision", "license", "normalization_version"):
        _text(source.get(field), f"source.{field}")
    _verify_artifact(root, _object(source.get("artifact"), "source.artifact"), "source.artifact")

    raw_windows = manifest.get("windows")
    if not isinstance(raw_windows, list) or not raw_windows:
        raise InputManifestError("windows must be a non-empty array")
    windows: list[OrderedWindow] = []
    ids_seen: set[str] = set()
    content_seen: dict[str, str] = {}
    ranges: list[tuple[str, int, int, str, str]] = []
    expected_count = manifest.get("window_tokens")
    if not fixture and expected_count != PRODUCTION_TOKENS:
        raise InputManifestError(f"window_tokens must be {PRODUCTION_TOKENS}")
    if not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count <= 0:
        raise InputManifestError("window_tokens must be a positive integer")

    for index, raw in enumerate(raw_windows):
        spec = _object(raw, f"windows[{index}]")
        window_id = _text(spec.get("id"), f"windows[{index}].id")
        if window_id in ids_seen:
            raise InputManifestError(f"duplicate window id: {window_id}")
        ids_seen.add(window_id)
        partition = spec.get("partition")
        if partition not in {"calibration", "held_out"}:
            raise InputManifestError(f"{window_id}: partition must be calibration or held_out")
        document_id = _text(spec.get("document_id"), f"{window_id}.document_id")
        source_range = _object(spec.get("source_range"), f"{window_id}.source_range")
        start, end = source_range.get("start"), source_range.get("end")
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, end)) or not 0 <= start < end:
            raise InputManifestError(f"{window_id}: source_range must contain integers 0 <= start < end")
        if spec.get("sequence_boundary") != "independent":
            raise InputManifestError(f"{window_id}: sequence_boundary must be 'independent'")
        token_path = _relative_file(root, spec.get("token_path"), f"{window_id}.token_path")
        ids = _load_ids(token_path, vocab_size, window_id)
        if ids.size != expected_count or spec.get("token_count") != ids.size:
            raise InputManifestError(f"{window_id}: expected exactly {expected_count} tokens")
        file_hash = sha256_file(token_path)
        order_hash, content_hash = token_order_sha256(ids), token_content_sha256(ids)
        for field, actual in (("token_file_sha256", file_hash), ("order_sha256", order_hash), ("content_sha256", content_hash)):
            if spec.get(field) != actual:
                raise InputManifestError(f"{window_id}: {field} mismatch")
        if content_hash in content_seen:
            raise InputManifestError(f"duplicate token content in {window_id} and {content_seen[content_hash]}")
        content_seen[content_hash] = window_id
        for old_doc, old_start, old_end, old_partition, old_id in ranges:
            if document_id == old_doc and max(start, old_start) < min(end, old_end):
                detail = "split leakage" if partition != old_partition else "overlapping source ranges"
                raise InputManifestError(f"{detail}: {window_id} overlaps {old_id}")
        ranges.append((document_id, start, end, partition, window_id))
        windows.append(OrderedWindow(window_id, partition, document_id, ids))

    counts = {p: sum(w.partition == p for w in windows) for p in ("calibration", "held_out")}
    if counts != {"calibration": 1, "held_out": 2}:
        raise InputManifestError("v1 requires exactly one calibration and two held-out unique windows")
    return manifest, tuple(windows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--allow-fixture", action="store_true", help="accept labelled unit-test fixtures")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest, windows = load_ordered_windows(args.manifest, allow_fixture=args.allow_fixture)
    print(json.dumps({"schema_version": manifest["schema_version"], "windows": [w.window_id for w in windows]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
