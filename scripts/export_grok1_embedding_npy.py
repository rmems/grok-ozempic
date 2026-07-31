#!/usr/bin/env python3
"""Export a Grok-1 official pickle shard payload to a stream-compatible .npy file.

Official ``xai-org/grok-1`` ckpt-0 shards are JAX/pickle frames. ``grok-ozempic``
``stream::run_quantization`` only accepts ``*.safetensors`` or flat ``*.npy``
(NpyDir). This script bridges **embedding-first** for the first GOZ1 experiment
(GitHub #37 / Linear RM-189).

Default target: ``tensor00000_000`` — f32 token embedding ``(131072, 6144)``.

Filename stem uses ``__`` for ``.`` so ``npy_stem_to_tensor_name`` recovers the
logical name (e.g. ``embedding__slot_00__token_embedding.npy`` →
``embedding.slot_00.token_embedding``).

Usage::

    python3 scripts/export_grok1_embedding_npy.py \\
      --shard /home/raulmc/.models/xai-grok-1/ckpt-0/tensor00000_000 \\
      --output-dir /home/raulmc/.models/xai-grok-1/export-npy
"""

from __future__ import annotations

import argparse
import mmap as mmap_mod
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

# Known layout for tensor00000_000 from xai-dissect / observed-grok1-ckpt0 docs.
DEFAULT_SHAPE = (131072, 6144)
DEFAULT_OFFSET = 0x97  # 151 decimal; confirmed by xai-dissect dissect
DEFAULT_DTYPE = "f32"
DEFAULT_STEM = "embedding__slot_00__token_embedding"
DEFAULT_SHARD = (
    Path.home() / ".models" / "xai-grok-1" / "ckpt-0" / "tensor00000_000"
)
DEFAULT_OUTPUT_DIR = Path.home() / ".models" / "xai-grok-1" / "export-npy"

ITEMSIZE = {"f32": 4, "f16": 2, "bf16": 2}


def expected_nbytes(shape: tuple[int, ...], dtype: str) -> int:
    n = 1
    for d in shape:
        n *= d
    return n * ITEMSIZE[dtype]


def parse_dissect_table(text: str) -> tuple[int, int, tuple[int, ...], str] | None:
    """Parse first data row from ``xai-dissect dissect`` pretty table or plain text."""
    # Pretty table: │ 0   ┆ tensor ┆ f32   ┆ (131072, 6144) ┆ 0x97   ┆ 3221225472 │
    row = re.search(
        r"│\s*\d+\s*┆[^│]*┆\s*(\w+)\s*┆\s*\(([^)]+)\)\s*┆\s*(0x[0-9a-fA-F]+|\d+)\s*┆\s*(\d+)\s*│",
        text,
    )
    if row:
        dtype = row.group(1).lower()
        shape = tuple(int(x.strip()) for x in row.group(2).split(",") if x.strip())
        off_s = row.group(3)
        offset = int(off_s, 16) if off_s.lower().startswith("0x") else int(off_s)
        nbytes = int(row.group(4))
        return offset, nbytes, shape, dtype

    # Plain: offset=151  nbytes=3221225472  dtype=f32  shape=(131072, 6144)
    m = re.search(
        r"offset\s*=\s*(0x[0-9a-fA-F]+|\d+).*?nbytes\s*=\s*(\d+).*?"
        r"dtype\s*=\s*(\w+).*?shape\s*=\s*\(([^)]+)\)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return None
    off_s = m.group(1)
    offset = int(off_s, 16) if off_s.lower().startswith("0x") else int(off_s)
    nbytes = int(m.group(2))
    dtype = m.group(3).lower()
    shape = tuple(int(x.strip()) for x in m.group(4).split(",") if x.strip())
    return offset, nbytes, shape, dtype


def try_dissect(shard: Path, xai_dissect: str | None) -> tuple[int, int, tuple[int, ...], str] | None:
    binary = xai_dissect or shutil.which("xai-dissect")
    candidates = []
    if binary:
        candidates.append(binary)
    candidates.append(
        str(Path.home() / "rmems" / "xai-dissect" / "target" / "release" / "xai-dissect")
    )
    for cand in candidates:
        if not cand or not os.path.isfile(cand) or not os.access(cand, os.X_OK):
            continue
        # dissect takes the checkpoint directory and filters by prefix
        ckpt_dir = str(shard.parent)
        prefix = shard.name  # tensor00000_000
        try:
            proc = subprocess.run(
                [cand, "dissect", ckpt_dir, "--limit", "1", "--prefix", prefix],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        parsed = parse_dissect_table(text)
        if parsed:
            return parsed
    return None


def write_npy_f32(path: Path, shape: tuple[int, ...], payload: memoryview) -> None:
    """Write little-endian C-order float32 .npy (NumPy v1.0 header)."""
    if expected_nbytes(shape, "f32") != len(payload):
        raise SystemExit(
            f"payload length {len(payload)} != expected "
            f"{expected_nbytes(shape, 'f32')} for shape {shape}"
        )
    if len(shape) == 1:
        shape_str = f"({shape[0]},)"
    else:
        shape_str = "(" + ", ".join(str(d) for d in shape) + ")"
    dict_str = f"{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_str}, }}"
    magic = b"\x93NUMPY"
    preamble_len = 6 + 1 + 1 + 2
    raw_header_len = len(dict_str)
    total_unpadded = preamble_len + raw_header_len
    pad = (64 - (total_unpadded % 64)) % 64
    header_len = raw_header_len + pad
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(magic)
        f.write(bytes([1, 0]))
        f.write(struct.pack("<H", header_len))
        f.write(dict_str.encode("ascii"))
        f.write(b" " * pad)
        # stream payload in chunks to avoid an extra full copy
        view = payload
        chunk = 64 * 1024 * 1024
        for i in range(0, len(view), chunk):
            f.write(view[i : i + chunk])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shard",
        type=Path,
        default=DEFAULT_SHARD,
        help=f"path to pickle shard (default: {DEFAULT_SHARD})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for .npy (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--stem",
        default=DEFAULT_STEM,
        help=f"output filename stem without .npy (default: {DEFAULT_STEM})",
    )
    p.add_argument("--offset", type=lambda s: int(s, 0), default=None, help="payload byte offset")
    p.add_argument(
        "--shape",
        default=None,
        help="comma-separated shape, e.g. 131072,6144",
    )
    p.add_argument("--dtype", default=None, choices=sorted(ITEMSIZE.keys()))
    p.add_argument(
        "--xai-dissect",
        default=None,
        help="path to xai-dissect binary (optional; auto-detected)",
    )
    p.add_argument(
        "--no-dissect",
        action="store_true",
        help="skip xai-dissect; use defaults / explicit flags only",
    )
    args = p.parse_args()

    shard: Path = args.shard.expanduser().resolve()
    if not shard.is_file():
        print(f"error: shard not found: {shard}", file=sys.stderr)
        return 1

    offset = args.offset
    shape: tuple[int, ...] | None = None
    if args.shape:
        shape = tuple(int(x.strip()) for x in args.shape.split(",") if x.strip())
    dtype = args.dtype
    nbytes: int | None = None

    if not args.no_dissect and (offset is None or shape is None or dtype is None):
        parsed = try_dissect(shard, args.xai_dissect)
        if parsed:
            po, pn, ps, pd = parsed
            offset = offset if offset is not None else po
            nbytes = pn
            shape = shape if shape is not None else ps
            dtype = dtype if dtype is not None else pd
            print(f"xai-dissect: offset={offset} nbytes={nbytes} dtype={dtype} shape={shape}")

    # Embedding defaults
    if offset is None:
        offset = DEFAULT_OFFSET
    if shape is None:
        shape = DEFAULT_SHAPE
    if dtype is None:
        dtype = DEFAULT_DTYPE

    if dtype != "f32":
        print(
            f"error: only f32 export is implemented (got dtype={dtype})",
            file=sys.stderr,
        )
        return 1

    exp = expected_nbytes(shape, dtype)
    if nbytes is not None and nbytes != exp:
        print(
            f"error: dissect nbytes={nbytes} != shape*itemsize={exp}",
            file=sys.stderr,
        )
        return 1

    file_size = shard.stat().st_size
    if offset + exp > file_size:
        print(
            f"error: offset+payload ({offset}+{exp}) exceeds file size {file_size}",
            file=sys.stderr,
        )
        return 1

    out_path = args.output_dir.expanduser().resolve() / f"{args.stem}.npy"

    with shard.open("rb") as f:
        with mmap_mod.mmap(f.fileno(), 0, access=mmap_mod.ACCESS_READ) as mm:
            # Slice as memoryview only inside this block; release before mmap closes.
            payload = memoryview(mm)[offset : offset + exp]
            write_npy_f32(out_path, shape, payload)
            del payload

    logical = args.stem.replace("__", ".")
    print(f"wrote {out_path}")
    print(f"  logical tensor name (for stream): {logical}")
    print(f"  shape={shape} dtype={dtype} nbytes={exp}")
    print(f"  source offset={offset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
