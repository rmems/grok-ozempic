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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "grok1-ordered-inputs-v1"
INPUT_MODE = "ordered_text"
PRODUCTION_TOKENS = 8192
SEED = 20260806
INT64_MAX = int(np.iinfo(np.int64).max)
_WINDOW_HASH_FIELDS = (
    "token_file_sha256",
    "order_sha256",
    "content_sha256",
)


class InputManifestError(ValueError):
    """The input artifact is unsafe or does not match its manifest."""


@dataclass(frozen=True)
class OrderedWindow:
    window_id: str
    partition: str
    document_id: str
    token_ids: np.ndarray = field(hash=False)


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


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InputManifestError(f"{name} must be a positive integer")
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_texts(obj: dict[str, Any], fields: tuple[str, ...], prefix: str) -> None:
    for name in fields:
        _text(obj.get(name), f"{prefix}.{name}")


def _relative_file(root: Path, value: Any, name: str) -> Path:
    rel = Path(_text(value, name))
    if rel.is_absolute() or ".." in rel.parts:
        raise InputManifestError(f"{name} must be a portable relative path")
    path = root / rel
    if not path.is_file():
        raise InputManifestError(f"{name} does not exist: {rel}")
    resolved = path.resolve()
    base = root.resolve()
    if base not in resolved.parents and resolved != base:
        raise InputManifestError(f"{name} must stay inside the manifest directory")
    return resolved


def _verify_artifact(root: Path, spec: dict[str, Any], name: str) -> Path:
    path = _relative_file(root, spec.get("path"), f"{name}.path")
    expected = _text(spec.get("sha256"), f"{name}.sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise InputManifestError(f"{name} digest mismatch: expected {expected}, got {actual}")
    return path


def _require_id_range(ids: np.ndarray, vocab_size: int, window_id: str) -> None:
    if not ids.size:
        return
    lo = int(ids.min())
    hi = int(ids.max())
    if hi > INT64_MAX:
        raise InputManifestError(f"{window_id}: token IDs must fit in int64")
    if lo < 0 or hi >= vocab_size:
        raise InputManifestError(
            f"{window_id}: token IDs must be in [0, {vocab_size}), got [{lo}, {hi}]"
        )


def _load_ids(path: Path, vocab_size: int, window_id: str, expected_count: int) -> np.ndarray:
    if path.suffix != ".npy":
        raise InputManifestError(f"{window_id}: token path must use .npy")
    try:
        ids = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise InputManifestError(f"{window_id}: cannot load token array: {exc}") from exc
    if not isinstance(ids, np.ndarray):
        raise InputManifestError(f"{window_id}: token file must contain a .npy array")
    if ids.ndim != 1:
        raise InputManifestError(f"{window_id}: token array must be one-dimensional")
    if ids.size != expected_count:
        raise InputManifestError(f"{window_id}: expected exactly {expected_count} tokens")
    if ids.dtype.kind not in "iu":
        raise InputManifestError(f"{window_id}: token dtype must be integer, got {ids.dtype}")
    _require_id_range(ids, vocab_size, window_id)
    # Return an owned, read-only canonical array: callers cannot mutate evidence.
    result = np.array(ids, dtype=np.int64, copy=True)
    result.flags.writeable = False
    return result


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputManifestError(f"cannot read manifest: {exc}") from exc
    return _object(manifest, "manifest")


def _require_header(manifest: dict[str, Any], *, allow_fixture: bool) -> bool:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise InputManifestError(f"schema_version must be {SCHEMA_VERSION!r}")
    if manifest.get("input_mode") != INPUT_MODE:
        raise InputManifestError(
            "input_mode must be 'ordered_text'; mixed/sampled evidence is rejected"
        )
    if manifest.get("selection_seed") != SEED:
        raise InputManifestError(f"selection_seed must be {SEED}")
    fixture = manifest.get("fixture") is True
    if fixture and not allow_fixture:
        raise InputManifestError("fixture manifests are not production experiment inputs")
    if not fixture and manifest.get("fixture") is not False:
        raise InputManifestError("fixture must explicitly be false for production inputs")
    return fixture


def _require_tokenizer(manifest: dict[str, Any], root: Path) -> int:
    tokenizer = _object(manifest.get("tokenizer"), "tokenizer")
    _require_texts(tokenizer, ("identity", "revision", "special_tokens_policy"), "tokenizer")
    vocab_size = _positive_int(tokenizer.get("vocab_size"), "tokenizer.vocab_size")
    _verify_artifact(
        root,
        _object(tokenizer.get("model_file"), "tokenizer.model_file"),
        "tokenizer.model_file",
    )
    return vocab_size


def _require_source(manifest: dict[str, Any], root: Path) -> None:
    source = _object(manifest.get("source"), "source")
    _require_texts(source, ("identity", "revision", "license", "normalization_version"), "source")
    _verify_artifact(root, _object(source.get("artifact"), "source.artifact"), "source.artifact")


def _window_token_count(manifest: dict[str, Any], fixture: bool) -> int:
    expected_count = manifest.get("window_tokens")
    if not fixture and expected_count != PRODUCTION_TOKENS:
        raise InputManifestError(f"window_tokens must be {PRODUCTION_TOKENS}")
    return _positive_int(expected_count, "window_tokens")


def _source_range(spec: dict[str, Any], window_id: str) -> tuple[int, int]:
    source_range = _object(spec.get("source_range"), f"{window_id}.source_range")
    start, end = source_range.get("start"), source_range.get("end")
    if not _is_int(start) or not _is_int(end) or not 0 <= start < end:
        raise InputManifestError(
            f"{window_id}: source_range must contain integers 0 <= start < end"
        )
    return start, end


def _record_range(
    ranges: list[tuple[str, int, int, str, str]],
    document_id: str,
    start: int,
    end: int,
    partition: str,
    window_id: str,
) -> None:
    for old_doc, old_start, old_end, old_partition, old_id in ranges:
        if document_id == old_doc and max(start, old_start) < min(end, old_end):
            detail = "split leakage" if partition != old_partition else "overlapping source ranges"
            raise InputManifestError(f"{detail}: {window_id} overlaps {old_id}")
    ranges.append((document_id, start, end, partition, window_id))


def _verify_window_digests(
    spec: dict[str, Any],
    token_path: Path,
    ids: np.ndarray,
    window_id: str,
    content_seen: dict[str, str],
) -> None:
    actuals = {
        "token_file_sha256": sha256_file(token_path),
        "order_sha256": token_order_sha256(ids),
        "content_sha256": token_content_sha256(ids),
    }
    for name in _WINDOW_HASH_FIELDS:
        if spec.get(name) != actuals[name]:
            raise InputManifestError(f"{window_id}: {name} mismatch")
    content_hash = actuals["content_sha256"]
    if content_hash in content_seen:
        raise InputManifestError(
            f"duplicate token content in {window_id} and {content_seen[content_hash]}"
        )
    content_seen[content_hash] = window_id


def _parse_window_header(
    raw: Any, index: int, ids_seen: set[str]
) -> tuple[dict[str, Any], str, str, str, int, int]:
    spec = _object(raw, f"windows[{index}]")
    window_id = _text(spec.get("id"), f"windows[{index}].id")
    if window_id in ids_seen:
        raise InputManifestError(f"duplicate window id: {window_id}")
    ids_seen.add(window_id)
    partition = spec.get("partition")
    if partition not in {"calibration", "held_out"}:
        raise InputManifestError(f"{window_id}: partition must be calibration or held_out")
    document_id = _text(spec.get("document_id"), f"{window_id}.document_id")
    start, end = _source_range(spec, window_id)
    if spec.get("sequence_boundary") != "independent":
        raise InputManifestError(f"{window_id}: sequence_boundary must be 'independent'")
    return spec, window_id, partition, document_id, start, end


def _load_one_window(
    raw: Any,
    index: int,
    root: Path,
    vocab_size: int,
    expected_count: int,
    ids_seen: set[str],
    content_seen: dict[str, str],
    ranges: list[tuple[str, int, int, str, str]],
) -> OrderedWindow:
    spec, window_id, partition, document_id, start, end = _parse_window_header(raw, index, ids_seen)
    token_path = _relative_file(root, spec.get("token_path"), f"{window_id}.token_path")
    ids = _load_ids(token_path, vocab_size, window_id, expected_count)
    if spec.get("token_count") != ids.size:
        raise InputManifestError(f"{window_id}: expected exactly {expected_count} tokens")
    _verify_window_digests(spec, token_path, ids, window_id, content_seen)
    _record_range(ranges, document_id, start, end, partition, window_id)
    return OrderedWindow(window_id, partition, document_id, ids)


def _load_windows(
    manifest: dict[str, Any], root: Path, vocab_size: int, expected_count: int
) -> tuple[OrderedWindow, ...]:
    raw_windows = manifest.get("windows")
    if not isinstance(raw_windows, list) or not raw_windows:
        raise InputManifestError("windows must be a non-empty array")
    windows: list[OrderedWindow] = []
    ids_seen: set[str] = set()
    content_seen: dict[str, str] = {}
    ranges: list[tuple[str, int, int, str, str]] = []
    for index, raw in enumerate(raw_windows):
        windows.append(
            _load_one_window(
                raw, index, root, vocab_size, expected_count, ids_seen, content_seen, ranges
            )
        )
    return tuple(windows)


def _require_v1_partitions(windows: tuple[OrderedWindow, ...]) -> None:
    counts = {p: sum(w.partition == p for w in windows) for p in ("calibration", "held_out")}
    if counts != {"calibration": 1, "held_out": 2}:
        raise InputManifestError(
            "v1 requires exactly one calibration and two held-out unique windows"
        )


def load_ordered_windows(
    manifest_path: Path, *, allow_fixture: bool = False
) -> tuple[dict[str, Any], tuple[OrderedWindow, ...]]:
    """Fail closed while validating provenance, splits, hashes, and token IDs.

    ``allow_fixture`` permits short, explicitly labelled synthetic manifests for
    tests. It must never be used to qualify a production comparison.
    """
    manifest_path = Path(manifest_path)
    manifest = _read_manifest(manifest_path)
    fixture = _require_header(manifest, allow_fixture=allow_fixture)
    root = manifest_path.resolve().parent
    vocab_size = _require_tokenizer(manifest, root)
    _require_source(manifest, root)
    expected_count = _window_token_count(manifest, fixture)
    windows = _load_windows(manifest, root, vocab_size, expected_count)
    _require_v1_partitions(windows)
    return manifest, windows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--allow-fixture", action="store_true", help="accept labelled unit-test fixtures"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest, windows = load_ordered_windows(args.manifest, allow_fixture=args.allow_fixture)
    print(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "windows": [w.window_id for w in windows],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
