#!/usr/bin/env python3
"""
Trit histogram for GOZ1 packed checkpoints (GH #51 / Linear RM-201).

Reads a ``.goz1`` file produced by ``quantize-goz1`` / ``run_quantization``
and reports, per ``TENSOR_TERNARY`` tensor, the exact counts of 0 / +1 / -1
trits — i.e. the measured sparsity of the GIF ternary gate. FP16/preserve
tensors are listed but not histogrammed.

Layout mirrors ``src/core/weight_pack.rs`` / ``weight_pack_read.rs``:

* magic ``GOZ1`` (LE u32), version u32, tensor_count u64, meta_count u64
* metadata entries: key (u64 len + utf8), vtype u32 (0=u32, 1=str), value
* tensor table: name (u64 len + utf8), ndim u32, dims u64*ndim,
  tensor_type u32 (0=f16, 1=ternary), data_offset u64 (relative)
* header padded to DATA_ALIGNMENT=32; each blob padded to 32 too
* ternary payload: 2 bits per trit, 4 per byte, LSB-first;
  0b00=0, 0b01=+1, 0b10=-1; final byte zero-padded

Stdlib-only (project convention for scripts/). Usage::

    python3 scripts/goz1_trit_histogram.py PACK.goz1 [--json]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

DATA_ALIGNMENT = 32
META_U32 = 0
META_STR = 1
TENSOR_F16 = 0
TENSOR_TERNARY = 1
MAGIC = b"GOZ1"
CHUNK = 64 * 1024 * 1024

# Per byte value: counts of (zero, +1, -1) among its four 2-bit trits.
# 0b11 never appears in packs written by encode_trit; count it as "invalid".
_LUT: list[tuple[int, int, int, int]] = []
for b in range(256):
    z = p = m = bad = 0
    for shift in (0, 2, 4, 6):
        t = (b >> shift) & 0b11
        if t == 0b00:
            z += 1
        elif t == 0b01:
            p += 1
        elif t == 0b10:
            m += 1
        else:
            bad += 1
    _LUT.append((z, p, m, bad))


class Goz1Error(ValueError):
    """Malformed GOZ1 input."""


def _read_exact(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise Goz1Error(f"unexpected EOF (wanted {n} bytes, got {len(b)})")
    return b


def _read_u32(f) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _read_u64(f) -> int:
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _read_str(f) -> str:
    n = _read_u64(f)
    return _read_exact(f, n).decode("utf-8")


def _align_up(n: int, align: int) -> int:
    return (n + align - 1) // align * align


def read_header(f) -> tuple[int, dict[str, object], list[dict], int]:
    """Parse header; return (version, metadata, tensor infos, data_start)."""
    if _read_exact(f, 4) != MAGIC:
        raise Goz1Error("bad magic (expected GOZ1)")
    version = _read_u32(f)
    tensor_count = _read_u64(f)
    meta_count = _read_u64(f)

    metadata: dict[str, object] = {}
    for _ in range(meta_count):
        key = _read_str(f)
        vtype = _read_u32(f)
        if vtype == META_U32:
            metadata[key] = _read_u32(f)
        elif vtype == META_STR:
            metadata[key] = _read_str(f)
        else:
            raise Goz1Error(f"unsupported metadata value type {vtype}")

    tensors: list[dict] = []
    for _ in range(tensor_count):
        name = _read_str(f)
        ndim = _read_u32(f)
        shape = [_read_u64(f) for _ in range(ndim)]
        tensor_type = _read_u32(f)
        data_offset = _read_u64(f)
        tensors.append(
            {
                "name": name,
                "shape": shape,
                "tensor_type": tensor_type,
                "data_offset": data_offset,
            }
        )

    data_start = _align_up(f.tell(), DATA_ALIGNMENT)
    return version, metadata, tensors, data_start


def _num_elements(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def histogram_ternary(f, abs_offset: int, n_elements: int) -> dict[str, int]:
    """Count trits for one ternary blob of ``n_elements`` values."""
    nbytes = (n_elements + 3) // 4
    f.seek(abs_offset)
    zeros = pos = neg = invalid = 0
    remaining = nbytes
    last_byte: int | None = None
    while remaining > 0:
        chunk = _read_exact(f, min(CHUNK, remaining))
        remaining -= len(chunk)
        if remaining == 0 and n_elements % 4 != 0:
            last_byte = chunk[-1]
            chunk = chunk[:-1]
        for value, count in ((v, chunk.count(v)) for v in set(chunk)):
            z, p, m, bad = _LUT[value]
            zeros += z * count
            pos += p * count
            neg += m * count
            invalid += bad * count
    if last_byte is not None:
        for i in range(n_elements % 4):
            t = (last_byte >> (2 * i)) & 0b11
            if t == 0b00:
                zeros += 1
            elif t == 0b01:
                pos += 1
            elif t == 0b10:
                neg += 1
            else:
                invalid += 1
    return {"zeros": zeros, "pos": pos, "neg": neg, "invalid": invalid}


def analyze(path: Path) -> dict:
    with path.open("rb") as f:
        version, metadata, tensors, data_start = read_header(f)
        out_tensors = []
        for t in tensors:
            n = _num_elements(t["shape"])
            entry: dict[str, object] = {
                "name": t["name"],
                "shape": t["shape"],
                "num_elements": n,
                "type": "ternary" if t["tensor_type"] == TENSOR_TERNARY else "f16",
            }
            if t["tensor_type"] == TENSOR_TERNARY:
                hist = histogram_ternary(f, data_start + t["data_offset"], n)
                total = hist["zeros"] + hist["pos"] + hist["neg"] + hist["invalid"]
                if total != n:
                    raise Goz1Error(
                        f"trit count {total} != num_elements {n} for {t['name']}"
                    )
                entry["histogram"] = hist
                entry["sparsity"] = hist["zeros"] / n if n else 1.0
            out_tensors.append(entry)
    return {
        "file": str(path),
        "file_size": path.stat().st_size,
        "version": version,
        "metadata": metadata,
        "tensors": out_tensors,
    }


def _print_human(result: dict) -> None:
    print(f"{result['file']}  ({result['file_size']} bytes, version {result['version']})")
    gif = result["metadata"].get("oz.gif_threshold")
    if gif is not None:
        print(f"  oz.gif_threshold = {gif}")
    for t in result["tensors"]:
        shape = "x".join(str(d) for d in t["shape"])
        if t["type"] != "ternary":
            print(f"  {t['name']}  [{shape}]  f16/preserve (not histogrammed)")
            continue
        h = t["histogram"]
        n = t["num_elements"]
        print(f"  {t['name']}  [{shape}]  n={n}")
        print(
            f"    zeros={h['zeros']} ({100.0 * h['zeros'] / n:.4f}%)  "
            f"+1={h['pos']} ({100.0 * h['pos'] / n:.4f}%)  "
            f"-1={h['neg']} ({100.0 * h['neg'] / n:.4f}%)"
            + (f"  INVALID={h['invalid']}" if h["invalid"] else "")
        )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pack", type=Path, help="path to .goz1 file")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args(argv)
    try:
        result = analyze(args.pack)
    except (OSError, Goz1Error) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.json:
        json.dump(result, sys.stdout, indent=2)
        print()
    else:
        _print_human(result)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
