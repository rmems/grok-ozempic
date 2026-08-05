#!/usr/bin/env python3
"""Manifest selection + structural stem helpers for int8 export."""
from __future__ import annotations

import json
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
    """
    if any(sep in name for sep in "/\\"):
        raise ExportError(f"structural name contains path separator: {name!r}")
    if ":" in name:
        raise ExportError(f"structural name contains colon (drive-relative path): {name!r}")
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
    with open(path, "rb") as fh:
        doc = json.load(fh)
    tensors = doc.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ExportError(f"{path}: no `tensors` array (not an xai-dissect conversion manifest?)")
    return tensors


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
    by_name = {t["structural_name"]: t for t in tensors}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ExportError(f"structural names not in manifest: {', '.join(missing)}")
    return [by_name[n] for n in names]


def _select_by_block(tensors: list[dict], block: int | None, mode: str) -> list[dict]:
    kinds = MODES[mode]
    picked = [t for t in tensors if t.get("block") == block and t.get("kind") in kinds]
    if not picked:
        raise ExportError(f"no tensors for block {block} in mode {mode}")
    return picked


