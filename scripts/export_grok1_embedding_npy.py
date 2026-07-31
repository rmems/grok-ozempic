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
      --shard ~/.models/xai-grok-1/ckpt-0/tensor00000_000 \\
      --output-dir ~/.models/xai-grok-1/export-npy
"""

from __future__ import annotations

import argparse
import mmap as mmap_mod
import os
import re
import shutil
import struct
import subprocess  # nosec B404 — only invokes fixed basename xai-dissect via PATH
import sys
import tempfile
from pathlib import Path

# Known layout for tensor00000_000 from xai-dissect / observed-grok1-ckpt0 docs.
DEFAULT_SHAPE = (131072, 6144)
DEFAULT_OFFSET = 0x97  # 151 decimal; confirmed by xai-dissect dissect
DEFAULT_DTYPE = "f32"
DEFAULT_STEM = "embedding__slot_00__token_embedding"
DEFAULT_SHARD = Path.home() / ".models" / "xai-grok-1" / "ckpt-0" / "tensor00000_000"
DEFAULT_OUTPUT_DIR = Path.home() / ".models" / "xai-grok-1" / "export-npy"
DISSECT_BIN = "xai-dissect"

ITEMSIZE = {"f32": 4, "f16": 2, "bf16": 2}
Layout = tuple[int, int, tuple[int, ...], str]


def expected_nbytes(shape: tuple[int, ...], dtype: str) -> int:
    if dtype not in ITEMSIZE:
        raise ValueError(
            f"unsupported dtype: {dtype!r}; supported: {sorted(ITEMSIZE)}"
        )
    n = 1
    for d in shape:
        n *= d
    return n * ITEMSIZE[dtype]


def _parse_hex_or_int(s: str) -> int:
    return int(s, 16) if s.lower().startswith("0x") else int(s)


def _parse_shape_csv(inner: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in inner.split(",") if x.strip())


def _parse_pretty_dissect_row(text: str) -> Layout | None:
    """Parse first data row from ``xai-dissect dissect`` pretty table."""
    # │ 0   ┆ tensor ┆ f32   ┆ (131072, 6144) ┆ 0x97   ┆ 3221225472 │
    row = re.search(
        r"│\s*\d+\s*┆[^│]*┆\s*(\w+)\s*┆\s*\(([^)]+)\)\s*┆"
        r"\s*(0x[0-9a-fA-F]+|\d+)\s*┆\s*(\d+)\s*│",
        text,
    )
    if not row:
        return None
    dtype = row.group(1).lower()
    shape = _parse_shape_csv(row.group(2))
    offset = _parse_hex_or_int(row.group(3))
    nbytes = int(row.group(4))
    return offset, nbytes, shape, dtype


def _parse_plain_dissect(text: str) -> Layout | None:
    """Parse plain ``offset=… nbytes=… dtype=… shape=(…)`` text."""
    m = re.search(
        r"offset\s*=\s*(0x[0-9a-fA-F]+|\d+).*?nbytes\s*=\s*(\d+).*?"
        r"dtype\s*=\s*(\w+).*?shape\s*=\s*\(([^)]+)\)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return None
    offset = _parse_hex_or_int(m.group(1))
    nbytes = int(m.group(2))
    dtype = m.group(3).lower()
    shape = _parse_shape_csv(m.group(4))
    return offset, nbytes, shape, dtype


def parse_dissect_table(text: str) -> Layout | None:
    """Parse first data row from ``xai-dissect dissect`` pretty table or plain text."""
    return _parse_pretty_dissect_row(text) or _parse_plain_dissect(text)


def _is_safe_dissect_binary(path: Path) -> bool:
    """Accept only an executable whose basename is exactly ``xai-dissect``."""
    if path.name != DISSECT_BIN:
        return False
    return path.is_file() and os.access(path, os.X_OK)


def _dissect_search_dirs(explicit: str | None) -> list[Path]:
    """Directories to put on PATH so we can exec the fixed ``xai-dissect`` name."""
    dirs: list[Path] = []
    if explicit:
        p = Path(explicit).expanduser()
        if _is_safe_dissect_binary(p):
            dirs.append(p.parent.resolve())
    which = shutil.which(DISSECT_BIN)
    if which:
        dirs.append(Path(which).resolve().parent)
    # Portable optional install locations (no username hardcoding).
    for rel in (
        Path.home() / "rmems" / "xai-dissect" / "target" / "release",
        Path.home() / "xai-dissect" / "target" / "release",
    ):
        cand = rel / DISSECT_BIN
        if _is_safe_dissect_binary(cand):
            dirs.append(rel.resolve())
    # Deduplicate while preserving order.
    seen: set[Path] = set()
    out: list[Path] = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _run_dissect_in_dir(bin_dir: Path, shard: Path) -> Layout | None:
    """Run fixed-name ``xai-dissect`` with PATH restricted to ``bin_dir`` (+ system)."""
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    try:
        # Fixed basename + static flags; PATH resolve + basename check above.
        proc = subprocess.run(  # noqa: S603  # nosec B603
            [
                DISSECT_BIN,
                "dissect",
                str(shard.parent),
                "--limit",
                "1",
                "--prefix",
                shard.name,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return parse_dissect_table(text)


def try_dissect(shard: Path, xai_dissect: str | None) -> Layout | None:
    for bin_dir in _dissect_search_dirs(xai_dissect):
        parsed = _run_dissect_in_dir(bin_dir, shard)
        if parsed:
            return parsed
    return None


def write_npy_f32(path: Path, shape: tuple[int, ...], payload: memoryview) -> None:
    """Write little-endian C-order float32 .npy (NumPy v1.0 header).

    Header dict ends with ``\\n``, then space-padded so (preamble + header_len)
    is a multiple of 64. Write is atomic via temp file + os.replace.
    """
    try:
        exp = expected_nbytes(shape, "f32")
    except ValueError as e:
        raise SystemExit(str(e)) from e
    if exp != len(payload):
        raise SystemExit(
            f"payload length {len(payload)} != expected {exp} for shape {shape}"
        )
    if len(shape) == 1:
        shape_str = f"({shape[0]},)"
    else:
        shape_str = "(" + ", ".join(str(d) for d in shape) + ")"
    # NPY v1.0: dict + spaces so (preamble+header) % 64 == 0, terminated by \n.
    dict_str = f"{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_str}, }}"
    magic = b"\x93NUMPY"
    preamble_len = 6 + 1 + 1 + 2
    raw_header_len = len(dict_str) + 1
    total_unpadded = preamble_len + raw_header_len
    pad = (64 - (total_unpadded % 64)) % 64
    header_len = raw_header_len + pad
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(magic)
            f.write(bytes([1, 0]))
            f.write(struct.pack("<H", header_len))
            f.write(dict_str.encode("ascii"))
            f.write(b" " * pad)
            f.write(b"\n")
            chunk = 64 * 1024 * 1024
            for i in range(0, len(payload), chunk):
                f.write(payload[i : i + chunk])
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _apply_dissect_defaults(
    offset: int | None,
    shape: tuple[int, ...] | None,
    dtype: str | None,
    parsed: Layout,
) -> tuple[int | None, tuple[int, ...] | None, str | None, int]:
    po, pn, ps, pd = parsed
    return (
        offset if offset is not None else po,
        shape if shape is not None else ps,
        dtype if dtype is not None else pd,
        pn,
    )


def resolve_layout(
    shard: Path,
    args: argparse.Namespace,
) -> tuple[int, tuple[int, ...], str, int | None]:
    offset = args.offset
    shape: tuple[int, ...] | None = None
    if args.shape:
        shape = _parse_shape_csv(args.shape)
    dtype = args.dtype
    nbytes: int | None = None

    need_dissect = not args.no_dissect and (
        offset is None or shape is None or dtype is None
    )
    if need_dissect:
        parsed = try_dissect(shard, args.xai_dissect)
        if parsed:
            offset, shape, dtype, nbytes = _apply_dissect_defaults(
                offset, shape, dtype, parsed
            )
            print(
                f"xai-dissect: offset={offset} nbytes={nbytes} "
                f"dtype={dtype} shape={shape}"
            )

    return (
        DEFAULT_OFFSET if offset is None else offset,
        DEFAULT_SHAPE if shape is None else shape,
        DEFAULT_DTYPE if dtype is None else dtype,
        nbytes,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shard",
        type=Path,
        default=DEFAULT_SHARD,
        help=(
            "path to pickle shard "
            "(default: ~/.models/xai-grok-1/ckpt-0/tensor00000_000)"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for .npy (default: ~/.models/xai-grok-1/export-npy)",
    )
    p.add_argument(
        "--stem",
        default=DEFAULT_STEM,
        help=f"output filename stem without .npy (default: {DEFAULT_STEM})",
    )
    p.add_argument(
        "--offset",
        type=lambda s: int(s, 0),
        default=None,
        help="payload byte offset",
    )
    p.add_argument(
        "--shape",
        default=None,
        help="comma-separated shape, e.g. 131072,6144",
    )
    p.add_argument("--dtype", default=None, choices=sorted(ITEMSIZE.keys()))
    p.add_argument(
        "--xai-dissect",
        default=None,
        help="path to xai-dissect binary (optional; auto-detected via PATH)",
    )
    p.add_argument(
        "--no-dissect",
        action="store_true",
        help="skip xai-dissect; use defaults / explicit flags only",
    )
    return p


def _validate_export(
    dtype: str, shape: tuple[int, ...], nbytes: int | None, offset: int, file_size: int
) -> int | None:
    """Return expected payload size, or print error and return None."""
    if dtype != "f32":
        print(
            f"error: only f32 export is implemented (got dtype={dtype})",
            file=sys.stderr,
        )
        return None
    try:
        exp = expected_nbytes(shape, dtype)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return None
    if nbytes is not None and nbytes != exp:
        print(
            f"error: dissect nbytes={nbytes} != shape*itemsize={exp}",
            file=sys.stderr,
        )
        return None
    if offset < 0:
        print(f"error: offset must be >= 0 (got {offset})", file=sys.stderr)
        return None
    if offset + exp > file_size:
        print(
            f"error: offset+payload ({offset}+{exp}) exceeds file size {file_size}",
            file=sys.stderr,
        )
        return None
    return exp


def export_embedding(args: argparse.Namespace) -> int:
    shard: Path = args.shard.expanduser().resolve()
    if not shard.is_file():
        print(f"error: shard not found: {shard}", file=sys.stderr)
        return 1

    offset, shape, dtype, nbytes = resolve_layout(shard, args)
    exp = _validate_export(dtype, shape, nbytes, offset, shard.stat().st_size)
    if exp is None:
        return 1

    out_path = args.output_dir.expanduser().resolve() / f"{args.stem}.npy"
    with (
        shard.open("rb") as f,
        mmap_mod.mmap(f.fileno(), 0, access=mmap_mod.ACCESS_READ) as mm,
    ):
        payload = memoryview(mm)[offset : offset + exp]
        try:
            write_npy_f32(out_path, shape, payload)
        finally:
            del payload

    logical = args.stem.replace("__", ".")
    print(f"wrote {out_path}")
    print(f"  logical tensor name (for stream): {logical}")
    print(f"  shape={shape} dtype={dtype} nbytes={exp}")
    print(f"  source offset={offset}")
    return 0


def main() -> int:
    return export_embedding(build_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
