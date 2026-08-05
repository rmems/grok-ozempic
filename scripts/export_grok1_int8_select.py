#!/usr/bin/env python3
"""Manifest selection + structural stem helpers for int8 export."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from export_grok1_int8_scan import ExportError

PRESERVE_KINDS = ("router", "block_norm", "final_norm")
ATTENTION_KINDS = ("attn_proj_i8.narrow", "attn_proj_i8.model_width")
EXPERT_KINDS = ("moe_expert.gate", "moe_expert.up", "moe_expert.down")

MODES = {
    "attention_only": ATTENTION_KINDS + PRESERVE_KINDS,
    "expert_only": EXPERT_KINDS + PRESERVE_KINDS,
    "attention_plus_expert": ATTENTION_KINDS + EXPERT_KINDS + PRESERVE_KINDS,
    "preserve_only": PRESERVE_KINDS,
}


def structural_stem(name: str) -> str:
    """``block_000.slot_04.attn_proj_i8.model_width`` -> filename stem.

    Rejects path separators, Windows drive letters, and parent-reference
    segments so a malformed conversion manifest cannot write outside
    ``--output-dir``.

    ``__`` is the reserved encoding of ``.`` and is rejected inside a name:
    without that guard ``a.b`` and ``a__b`` both map to ``a__b``, so two
    distinct manifest entries would silently overwrite each other's ``.npy``
    and feed the wrong tensor to the pack. Rejecting keeps the encoding
    injective, which is what ``npy_stem_to_tensor_name`` assumes on the Rust
    side.
    """
    if any(sep in name for sep in "/\\"):
        raise ExportError(f"structural name contains path separator: {name!r}")
    if ":" in name:
        raise ExportError(f"structural name contains colon (drive-relative path): {name!r}")
    if "__" in name:
        raise ExportError(
            f"structural name contains the reserved '__' separator: {name!r}; "
            "'__' encodes '.' in npy stems, so this name is not round-trippable"
        )
    parts = name.split(".")
    if any(part in ("", "..") for part in parts):
        raise ExportError(
            f"structural name has empty or parent-reference segment: {name!r}"
        )
    return name.replace(".", "__")


def _safe_out_path(output_dir: Path, name: str) -> Path:
    """Resolve the final output path and enforce it stays under ``output_dir``.

    The structural name is sanitized above, but a literal ``..`` or absolute
    prefix could still be introduced by platform-specific path handling.
    """
    out = (output_dir / f"{structural_stem(name)}.npy").resolve()
    base = output_dir.resolve()
    if base not in out.parents and out != base:
        raise ExportError(
            f"resolved output path {out} escapes --output-dir {base} for name {name!r}"
        )
    return out


def load_manifest(path: Path) -> list[dict]:
    """Read an xai-dissect conversion manifest, failing closed on bad input.

    Unreadable or malformed manifests surface as ``ExportError`` (clean
    ``error:`` + exit 2) rather than an ``OSError``/``JSONDecodeError``
    traceback, so a drifted run3 input is reported the same way as every other
    rejected layout.
    """
    try:
        with open(path, "rb") as fh:
            doc = json.load(fh)
    except OSError as exc:
        raise ExportError(f"{path}: cannot read conversion manifest: {exc}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        # Invalid UTF-8 raises UnicodeDecodeError, not JSONDecodeError.
        raise ExportError(f"{path}: conversion manifest is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise ExportError(
            f"{path}: conversion manifest root is {type(doc).__name__}, expected an object"
        )
    tensors = doc.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ExportError(f"{path}: no `tensors` array (not an xai-dissect conversion manifest?)")
    _validate_entries(path, tensors)
    return tensors


# Required keys and the type each must carry, checked declaratively so adding a
# key does not add a branch. `kind` and `block` drive whole-block selection, so a
# wrongly typed one would silently change which tensors get exported: a numeric
# `kind` matches no mode, and a stringy `block` never equals `--block`, so the
# selector would quietly under-pick instead of failing.
_REQUIRED_TENSOR_KEYS: dict[str, tuple[type, ...]] = {
    "structural_name": (str,),
    "source_shard_path": (str,),
    "shape": (list, tuple),
    "kind": (str,),
}
# `block` is int for block tensors and null for model-level ones (the embedding),
# so it is checked separately rather than in the table above.
_BLOCK_TYPES: tuple[type, ...] = (int,)


def _validate_entries(path: Path, tensors: list) -> None:
    """Reject entries the exporter would otherwise fail on far from the cause."""
    seen: set[str] = set()
    for i, t in enumerate(tensors):
        _validate_entry(path, i, t, seen)


def _validate_entry(path: Path, i: int, t: object, seen: set[str]) -> None:
    if not isinstance(t, dict):
        raise ExportError(f"{path}: tensors[{i}] is {type(t).__name__}, expected an object")
    _require_keys(path, i, t)
    name = t["structural_name"]
    if name in seen:
        raise ExportError(f"{path}: duplicate structural_name {name!r} at tensors[{i}]")
    seen.add(name)
    _validate_block(path, i, t)
    _validate_shape(path, i, t["shape"])


def _validate_block(path: Path, i: int, t: dict) -> None:
    """`block` must be an int (block tensor) or null (model-level, e.g. embedding)."""
    if "block" not in t:
        raise ExportError(f"{path}: tensors[{i}] missing required key: block")
    block = t["block"]
    if block is None:
        return
    if not isinstance(block, _BLOCK_TYPES) or isinstance(block, bool):
        raise ExportError(
            f"{path}: tensors[{i}].block is {type(block).__name__}, expected int or null"
        )


def _require_keys(path: Path, i: int, t: dict) -> None:
    missing = [k for k in _REQUIRED_TENSOR_KEYS if k not in t]
    if missing:
        raise ExportError(
            f"{path}: tensors[{i}] missing required key(s): {', '.join(missing)}"
        )
    for key, want in _REQUIRED_TENSOR_KEYS.items():
        if not isinstance(t[key], want):
            names = " or ".join(w.__name__ for w in want)
            raise ExportError(
                f"{path}: tensors[{i}].{key} is {type(t[key]).__name__}, expected {names}"
            )


def _validate_shape(path: Path, i: int, shape: object) -> None:
    """Require genuine ints, not int-coercible values.

    ``int("3")`` and ``int(3.7)`` both succeed, so coercing would accept a
    manifest whose shape disagrees with the shard while claiming to have
    validated it. ``bool`` is an ``int`` subclass and is excluded explicitly.
    """
    bad = [
        d for d in shape  # type: ignore[union-attr]
        if not isinstance(d, int) or isinstance(d, bool)
    ]
    if bad:
        raise ExportError(
            f"{path}: tensors[{i}].shape {shape!r} is not a list of ints "
            f"(offending value {bad[0]!r})"
        )


def select_tensors(
    tensors: list[dict],
    *,
    block: int | None,
    mode: str,
    names: list[str],
) -> list[dict]:
    """Pick explicit structural names, or every tensor of one block in ``mode``."""
    if names:
        return _select_by_names(tensors, names)
    return _select_by_block(tensors, block, mode)


def _select_by_names(tensors: list[dict], names: list[str]) -> list[dict]:
    _reject_repeated(names)
    by_name = {t["structural_name"]: t for t in tensors}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ExportError(f"structural names not in manifest: {', '.join(missing)}")
    return [by_name[n] for n in names]


def _reject_repeated(names: list[str]) -> None:
    """A name given twice would export the same tensor twice under one stem."""
    dups = sorted({n for n, c in Counter(names).items() if c > 1})
    if dups:
        raise ExportError(f"duplicate --structural-name entries: {', '.join(dups)}")


def _select_by_block(tensors: list[dict], block: int | None, mode: str) -> list[dict]:
    kinds = MODES[mode]
    picked = [t for t in tensors if t.get("block") == block and t.get("kind") in kinds]
    if not picked:
        raise ExportError(f"no tensors for block {block} in mode {mode}")
    return picked


