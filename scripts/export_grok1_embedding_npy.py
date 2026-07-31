#!/usr/bin/env python3
r"""
Export a Grok-1 official pickle shard payload to a stream-compatible .npy file.

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
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path

# Known layout for tensor00000_000 from xai-dissect / observed-grok1-ckpt0 docs.
DEFAULT_SHAPE = (131072, 6144)
DEFAULT_OFFSET = 0x97  # 151 decimal; confirmed by xai-dissect dissect
DEFAULT_DTYPE = "f32"
DEFAULT_STEM = "embedding__slot_00__token_embedding"
DEFAULT_SHARD_NAME = "tensor00000_000"
DEFAULT_SHARD = Path.home() / ".models" / "xai-grok-1" / "ckpt-0" / DEFAULT_SHARD_NAME
DEFAULT_OUTPUT_DIR = Path.home() / ".models" / "xai-grok-1" / "export-npy"
DISSECT_BIN = "xai-dissect"

ITEMSIZE = {"f32": 4, "f16": 2, "bf16": 2}
Layout = tuple[int, int, tuple[int, ...], str]


class LayoutError(ValueError):

    """Invalid layout / CLI input for export."""


def expected_nbytes(shape: tuple[int, ...], dtype: str) -> int:
    """Return payload size for shape and dtype, or raise LayoutError."""
    if dtype not in ITEMSIZE:
        raise LayoutError(
            f"unsupported dtype: {dtype!r}; supported: {sorted(ITEMSIZE)}"
        )
    n = 1
    for d in shape:
        n *= d
    return n * ITEMSIZE[dtype]


def _parse_hex_or_int(s: str) -> int:
    return int(s, 16) if s.lower().startswith("0x") else int(s)


def _parse_one_dim(p: str) -> int:
    if not (p.isascii() and p.isdigit()):
        raise LayoutError(f"invalid shape component {p!r}: not an integer")
    try:
        d = int(p, 10)
    except ValueError as e:
        raise LayoutError(f"invalid shape component {p!r}: not an integer") from e
    if d <= 0:
        raise LayoutError(f"invalid shape component {d}: must be > 0")
    return d


def _parse_shape_csv(inner: str) -> tuple[int, ...]:
    """Parse comma-separated positive dimensions; empty or non-int is an error."""
    parts = [x.strip() for x in inner.split(",")]
    if any(not p for p in parts):
        raise LayoutError("shape must not contain empty dimensions")
    return tuple(_parse_one_dim(p) for p in parts)


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


def _append_unique(dirs: list[Path], seen: set[Path], d: Path) -> None:
    if d not in seen:
        seen.add(d)
        dirs.append(d)


def _optional_install_dirs() -> list[Path]:
    """Portable optional install locations (no username hardcoding)."""
    found: list[Path] = []
    for rel in (
        Path.home() / "rmems" / "xai-dissect" / "target" / "release",
        Path.home() / "xai-dissect" / "target" / "release",
    ):
        if _is_safe_dissect_binary(rel / DISSECT_BIN):
            found.append(rel.resolve())
    return found


def _validate_explicit_dissect_path(path_str: str) -> Path:
    """
    Validate --xai-dissect path is a usable binary; raise LayoutError if not.

    Returns the resolved parent directory of the validated binary.
    """
    p = Path(path_str).expanduser()
    if not _is_safe_dissect_binary(p):
        raise LayoutError(
            f"--xai-dissect path is not a usable {DISSECT_BIN!r} binary: {p}"
        )
    return p


def _dissect_search_dirs(explicit: str | None) -> list[Path]:
    """
    Directories to put on PATH so we can exec the fixed ``xai-dissect`` name.

    If ``explicit`` is set, it must resolve to a safe binary; otherwise raises
    LayoutError (never silently ignored).
    """
    if explicit:
        p = _validate_explicit_dissect_path(explicit)
        return [p.parent.resolve()]

    out: list[Path] = []
    seen: set[Path] = set()
    which = shutil.which(DISSECT_BIN)
    if which:
        _append_unique(out, seen, Path(which).resolve().parent)
    for d in _optional_install_dirs():
        _append_unique(out, seen, d)
    return out


def _dissect_cmd(binary: Path, shard: Path) -> list[str]:
    return [
        str(binary),
        "dissect",
        str(shard.parent),
        "--limit",
        "1",
        "--prefix",
        shard.name,
    ]


def _invoke_dissect(cmd: list[str]) -> str | None:
    """Run dissect; return combined output on success, else None."""
    try:
        # Absolute path to basename-checked binary; static argv, no shell.
        proc = subprocess.run(  # nosec B603  # noqa: S603
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def _run_dissect_in_dir(bin_dir: Path, shard: Path) -> Layout | None:
    """Run fixed-name ``xai-dissect`` via absolute path under ``bin_dir``."""
    binary = bin_dir / DISSECT_BIN
    if not _is_safe_dissect_binary(binary):
        return None
    text = _invoke_dissect(_dissect_cmd(binary, shard))
    if text is None:
        return None
    return parse_dissect_table(text)


def try_dissect(shard: Path, xai_dissect: str | None) -> Layout | None:
    for bin_dir in _dissect_search_dirs(xai_dissect):
        parsed = _run_dissect_in_dir(bin_dir, shard)
        if parsed:
            return parsed
    return None


def _npy_shape_str(shape: tuple[int, ...]) -> str:
    if len(shape) == 1:
        return f"({shape[0]},)"
    return "(" + ", ".join(str(d) for d in shape) + ")"


def _write_npy_body(f, dict_str: str, pad: int, payload: memoryview) -> None:
    magic = b"\x93NUMPY"
    header_len = len(dict_str) + 1 + pad
    f.write(magic)
    f.write(bytes([1, 0]))
    f.write(struct.pack("<H", header_len))
    f.write(dict_str.encode("ascii"))
    f.write(b" " * pad)
    f.write(b"\n")
    chunk = 64 * 1024 * 1024
    for i in range(0, len(payload), chunk):
        f.write(payload[i : i + chunk])


def write_npy_f32(path: Path, shape: tuple[int, ...], payload: memoryview) -> None:
    r"""
    Write little-endian C-order float32 .npy (NumPy v1.0 header).

    Header is space-padded so (preamble + header_len) is a multiple of 64, then
    terminated by ``\n`` (pad-then-newline). Write is atomic via temp + os.replace.
    """
    exp = expected_nbytes(shape, "f32")
    if exp != len(payload):
        raise LayoutError(
            f"payload length {len(payload)} != expected {exp} for shape {shape}"
        )
    shape_str = _npy_shape_str(shape)
    dict_str = f"{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_str}, }}"
    preamble_len = 6 + 1 + 1 + 2
    raw_header_len = len(dict_str) + 1
    total_unpadded = preamble_len + raw_header_len
    pad = (64 - (total_unpadded % 64)) % 64
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            _write_npy_body(f, dict_str, pad, payload)
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


def _can_use_embedding_defaults(shard: Path) -> bool:
    return shard.name == DEFAULT_SHARD_NAME


def _cli_layout(
    args: argparse.Namespace,
) -> tuple[int | None, tuple[int, ...] | None, str | None]:
    shape: tuple[int, ...] | None = None
    if args.shape is not None:
        shape = _parse_shape_csv(args.shape)
    return args.offset, shape, args.dtype


def _try_dissect_fill(
    shard: Path,
    args: argparse.Namespace,
    offset: int | None,
    shape: tuple[int, ...] | None,
    dtype: str | None,
) -> tuple[int | None, tuple[int, ...] | None, str | None, int | None]:
    parsed = try_dissect(shard, args.xai_dissect)
    if not parsed:
        print(
            "warning: xai-dissect unavailable or failed; "
            "falling back only if layout is fully specified or "
            f"shard is {DEFAULT_SHARD_NAME}",
            file=sys.stderr,
        )
        return offset, shape, dtype, None
    offset, shape, dtype, nbytes = _apply_dissect_defaults(
        offset, shape, dtype, parsed
    )
    print(
        f"xai-dissect: offset={offset} nbytes={nbytes} "
        f"dtype={dtype} shape={shape}"
    )
    return offset, shape, dtype, nbytes


def _apply_embedding_defaults(
    offset: int | None,
    shape: tuple[int, ...] | None,
    dtype: str | None,
) -> tuple[int, tuple[int, ...], str]:
    if offset is None:
        offset = DEFAULT_OFFSET
    if shape is None:
        shape = DEFAULT_SHAPE
    if dtype is None:
        dtype = DEFAULT_DTYPE
    print(
        f"warning: using hard-coded layout for {DEFAULT_SHARD_NAME}: "
        f"offset={offset} shape={shape} dtype={dtype}",
        file=sys.stderr,
    )
    return offset, shape, dtype


def _missing_layout_error(shard: Path, offset, shape, dtype) -> LayoutError:
    missing = [
        n
        for n, v in (
            ("--offset", offset),
            ("--shape", shape),
            ("--dtype", dtype),
        )
        if v is None
    ]
    return LayoutError(
        f"cannot resolve layout for shard {shard.name!r}: missing "
        f"{', '.join(missing)}. Run with working xai-dissect, pass full "
        f"--offset/--shape/--dtype, or use --no-dissect only with a "
        f"complete layout (defaults apply only to {DEFAULT_SHARD_NAME})"
    )


def _layout_complete(
    offset: int | None, shape: tuple[int, ...] | None, dtype: str | None
) -> bool:
    return offset is not None and shape is not None and dtype is not None


def _fill_missing_layout(
    shard: Path,
    offset: int | None,
    shape: tuple[int, ...] | None,
    dtype: str | None,
) -> tuple[int, tuple[int, ...], str]:
    if _layout_complete(offset, shape, dtype):
        # Narrowed by _layout_complete.
        return offset, shape, dtype  # type: ignore[return-value]
    if _can_use_embedding_defaults(shard):
        return _apply_embedding_defaults(offset, shape, dtype)
    raise _missing_layout_error(shard, offset, shape, dtype)


def resolve_layout(
    shard: Path,
    args: argparse.Namespace,
) -> tuple[int, tuple[int, ...], str, int | None]:
    """
    Resolve offset/shape/dtype/nbytes for export.

    Prefer xai-dissect when layout is incomplete (unless --no-dissect). Hard-coded
    embedding defaults only for shard basename tensor00000_000.
    """
    if args.xai_dissect:
        _validate_explicit_dissect_path(args.xai_dissect)
    offset, shape, dtype = _cli_layout(args)
    nbytes: int | None = None
    if not args.no_dissect and not _layout_complete(offset, shape, dtype):
        offset, shape, dtype, nbytes = _try_dissect_fill(
            shard, args, offset, shape, dtype
        )
    offset, shape, dtype = _fill_missing_layout(shard, offset, shape, dtype)
    return offset, shape, dtype, nbytes


def _stem_is_single_segment(stem: str) -> bool:
    p = Path(stem)
    return (not p.is_absolute()) and len(p.parts) == 1 and p.name == stem


def validate_stem(stem: str) -> str:
    """Reject path separators / absolute paths so stem cannot escape output-dir."""
    if not stem or stem in (".", "..") or stem.endswith(".npy"):
        raise LayoutError("--stem must be a non-empty filename stem without .npy")
    if not _stem_is_single_segment(stem):
        raise LayoutError(
            f"--stem must be a single path segment without separators (got {stem!r})"
        )
    return stem


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
        help="path to xai-dissect binary (must be named xai-dissect; optional)",
    )
    p.add_argument(
        "--no-dissect",
        action="store_true",
        help=(
            "skip xai-dissect; require full --offset/--shape/--dtype unless "
            f"shard is {DEFAULT_SHARD_NAME}"
        ),
    )
    return p


def _check_payload_bounds(
    exp: int, nbytes: int | None, offset: int, file_size: int
) -> bool:
    if nbytes is not None and nbytes != exp:
        print(
            f"error: dissect nbytes={nbytes} != shape*itemsize={exp}",
            file=sys.stderr,
        )
        return False
    if offset < 0:
        print(f"error: offset must be >= 0 (got {offset})", file=sys.stderr)
        return False
    if offset + exp > file_size:
        print(
            f"error: offset+payload ({offset}+{exp}) exceeds file size {file_size}",
            file=sys.stderr,
        )
        return False
    return True


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
    except LayoutError as e:
        print(f"error: {e}", file=sys.stderr)
        return None
    if not _check_payload_bounds(exp, nbytes, offset, file_size):
        return None
    return exp


def _export_payload(
    shard: Path, out_path: Path, offset: int, exp: int, shape: tuple[int, ...]
) -> None:
    with (
        shard.open("rb") as f,
        mmap_mod.mmap(f.fileno(), 0, access=mmap_mod.ACCESS_READ) as mm,
    ):
        payload = memoryview(mm)[offset : offset + exp]
        try:
            write_npy_f32(out_path, shape, payload)
        finally:
            del payload


def export_embedding(args: argparse.Namespace) -> int:
    try:
        stem = validate_stem(args.stem)
        shard: Path = args.shard.expanduser().resolve()
        if not shard.is_file():
            print(f"error: shard not found: {shard}", file=sys.stderr)
            return 1

        offset, shape, dtype, nbytes = resolve_layout(shard, args)
        exp = _validate_export(dtype, shape, nbytes, offset, shard.stat().st_size)
        if exp is None:
            return 1

        out_path = args.output_dir.expanduser().resolve() / f"{stem}.npy"
        _export_payload(shard, out_path, offset, exp, shape)

        logical = stem.replace("__", ".")
        print(f"wrote {out_path}")
        print(f"  logical tensor name (for stream): {logical}")
        print(f"  shape={shape} dtype={dtype} nbytes={exp}")
        print(f"  source offset={offset}")
        return 0
    except LayoutError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    return export_embedding(build_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
