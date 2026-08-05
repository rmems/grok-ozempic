#!/usr/bin/env python3
r"""
Export Grok-1 official ckpt-0 shards to f32 .npy (int8 x bf16 dequant).

Split modules: export_grok1_int8_scan / _dequant / _select.
This file is the CLI entrypoint and test re-export surface (GH #53 / RM-222).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_grok1_int8_dequant import (
    DEFAULT_CHUNK_MIB,
    export_tensor,
    grouping as grouping,
    npy_header,
)
from export_grok1_int8_scan import (
    ArraySpec as ArraySpec,
    ExportError,
    _HeaderState as _HeaderState,
    _PAYLOAD_READ_LIMIT as _PAYLOAD_READ_LIMIT,
    _StopAtPayload as _StopAtPayload,
    _Unknown as _Unknown,
    scan_shard,
    split_quantized as split_quantized,
)
from export_grok1_int8_select import (
    ATTENTION_KINDS as ATTENTION_KINDS,
    EXPERT_KINDS as EXPERT_KINDS,
    MODES,
    PRESERVE_KINDS,
    load_manifest,
    select_tensors,
    structural_stem,
    _safe_out_path,
)

# Public re-exports for `import export_grok1_int8_npy as exp` unit tests.
__all__ = [
    "ATTENTION_KINDS",
    "ArraySpec",
    "DEFAULT_CHUNK_MIB",
    "EXPERT_KINDS",
    "ExportError",
    "MODES",
    "PRESERVE_KINDS",
    "export_tensor",
    "grouping",
    "load_manifest",
    "npy_header",
    "scan_shard",
    "select_tensors",
    "split_quantized",
    "structural_stem",
    "_HeaderState",
    "_PAYLOAD_READ_LIMIT",
    "_StopAtPayload",
    "_Unknown",
    "_safe_out_path",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dequantize Grok-1 int8 ckpt-0 shards to f32 .npy (GH #53 / RM-222)"
    )
    p.add_argument(
        "--conversion-manifest", type=Path, help="xai-dissect run3 conversion-manifest.json"
    )
    p.add_argument("--block", type=int, help="pilot block index (with --mode)")
    p.add_argument("--mode", choices=sorted(MODES), default="attention_only")
    p.add_argument(
        "--structural-name",
        action="append",
        default=[],
        dest="names",
        help=(
            "debug/repair: export one tensor by structural name "
            "(repeatable; overrides --block/--mode; not the pilot whole-block path)"
        ),
    )
    p.add_argument("--output-dir", type=Path, help="destination directory for .npy files")
    p.add_argument("--chunk-mib", type=int, default=DEFAULT_CHUNK_MIB, help="write chunk size")
    p.add_argument("--dry-run", action="store_true", help="report plan and sizes, write nothing")
    p.add_argument("--inspect", type=Path, help="print raw array layout of one shard and exit")
    return p


def run_exports(picked: list[dict], output_dir: Path, chunk_mib: int, dry_run: bool) -> int:
    """Export each selected tensor; return total f32 bytes planned or written."""
    total = 0
    stems: dict[str, str] = {}
    for t in sorted(picked, key=lambda x: x["structural_name"]):
        name = t["structural_name"]
        shard = Path(t["source_shard_path"])
        if not shard.exists():
            raise ExportError(f"{name}: shard missing: {shard}")
        out = _safe_out_path(output_dir, name)
        # Two distinct structural names can collide after __-encoding (e.g.
        # ``a.b`` and ``a__b``); structural_stem rejects ``__`` in names, but
        # preflight the selected set too so a malformed manifest cannot silently
        # overwrite one tensor with another.
        stem = out.stem
        if stem in stems:
            raise ExportError(
                f"duplicate output stem {stem!r} for {stems[stem]!r} and {name!r}"
            )
        stems[stem] = name
        # expect_shape cross-checks the manifest against the shard before writing.
        info = export_tensor(
            shard, out, chunk_mib=chunk_mib, dry_run=dry_run, expect_shape=t["shape"]
        )
        total += info["out_bytes"]
        groups = info["scale_groups"]
        detail = f"groups={groups}" if groups is not None else "passthrough"
        print(
            f"{'plan' if dry_run else 'wrote'} {name}  {info['source_dtype']:>13}  "
            f"{tuple(info['shape'])}  {detail}  {info['out_bytes'] / 1024**3:.3f} GiB "
            f"-> {out.name}"
        )
    return total


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.inspect:
        for spec in scan_shard(args.inspect):
            print(
                f"{spec.descr:>9} {spec.shape!s:>24}  "
                f"offset={spec.offset}  bytes={spec.nbytes}"
            )
        return 0

    if not args.conversion_manifest:
        parser.error("--conversion-manifest is required (or use --inspect)")
    if not args.output_dir:
        parser.error("--output-dir is required")
    if args.block is None and not args.names:
        parser.error("need --block (with --mode) or at least one --structural-name")
    if args.chunk_mib <= 0:
        parser.error("--chunk-mib must be a positive integer")

    picked = select_tensors(
        load_manifest(args.conversion_manifest),
        block=args.block,
        mode=args.mode,
        names=args.names,
    )
    total = run_exports(picked, Path(args.output_dir), args.chunk_mib, args.dry_run)
    print(f"{len(picked)} tensor(s), {total / 1024**3:.3f} GiB f32 total -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
