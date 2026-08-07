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
  tensor_type u32 (0=f16, 1=ternary), data_offset u64 (relative),
  **and in version 2 (GH #65) a scale f32** — the per-tensor reconstruction
  scale ``α*``, appended to the row so ``value = scale × payload`` can be
  evaluated from the pack alone. Version 1 rows stop after ``data_offset``;
  unknown versions are rejected rather than parsed as a compatible prefix.
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
from collections import Counter
from pathlib import Path

DATA_ALIGNMENT = 32
META_U32 = 0
META_STR = 1
TENSOR_F16 = 0
TENSOR_TERNARY = 1
MAGIC = b"GOZ1"
# Highest container version this parser understands, and the first one carrying
# a per-tensor scale. Mirrors OZ1_VERSION / OZ1_VERSION_SCALED in Rust.
OZ1_VERSION_MAX = 3
OZ1_VERSION_SCALED = 2
# First version carrying the applied per-tensor thresholds (GH #66).
OZ1_VERSION_THRESHOLDS = 3
ROW_SENTINEL = 0x5CA1E021
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


def _read_f32(f) -> float:
    return struct.unpack("<f", _read_exact(f, 4))[0]


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
    if version == 0 or version > OZ1_VERSION_MAX:
        raise Goz1Error(
            f"unsupported container version {version} "
            f"(this parser understands 1..={OZ1_VERSION_MAX})"
        )
    has_scales = version >= OZ1_VERSION_SCALED
    has_thresholds = version >= OZ1_VERSION_THRESHOLDS
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
        # ``None`` on v1, which has no scale field at all. Kept distinct from a
        # numeric 0.0 so a consumer can tell "this format stores no scale, use
        # the legacy oracle path" from "this tensor's scale really is zero"
        # (which happens legitimately when no trit fires).
        # Each version appends to the row, so an older layout is this same code
        # reading fewer fields. ``None`` (not 0.0) marks a field the layout does
        # not have, so a consumer can tell "the pack cannot say" from "the value
        # really is zero" -- both occur legitimately.
        scale = _read_f32(f) if has_scales else None
        gif_threshold = _read_f32(f) if has_thresholds else None
        threshold_abs = _read_f32(f) if has_thresholds else None
        if has_scales:
            sentinel = _read_u32(f)
            if sentinel != ROW_SENTINEL:
                raise Goz1Error(
                    f"tensor {name!r} row sentinel is {hex(sentinel)}, expected "
                    f"{hex(ROW_SENTINEL)}; the row width this parser expects for "
                    f"version {version} does not match the file"
                )

        if scale is not None:
            import math

            if tensor_type == TENSOR_TERNARY:
                if not math.isfinite(scale) or scale < 0.0:
                    raise Goz1Error(f"ternary tensor {name!r} has non-finite or negative scale {scale}")
            elif tensor_type == TENSOR_F16:
                if not math.isfinite(scale) or scale != 1.0:
                    raise Goz1Error(f"fp16 tensor {name!r} has invalid scale {scale}; expected 1.0")
        tensors.append(
            {
                "name": name,
                "shape": shape,
                "tensor_type": tensor_type,
                "data_offset": data_offset,
                "scale": scale,
                "gif_threshold": gif_threshold,
                "threshold_abs": threshold_abs,
            }
        )

    data_start = _align_up(f.tell(), DATA_ALIGNMENT)
    return version, metadata, tensors, data_start


def _num_elements(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _apply_lut_counts(
    chunk: bytes, zeros: int, pos: int, neg: int, invalid: int
) -> tuple[int, int, int, int]:
    """Accumulate trit totals for full payload bytes via the 4-trit LUT."""
    for value, count in Counter(chunk).items():
        z, p, m, bad = _LUT[value]
        zeros += z * count
        pos += p * count
        neg += m * count
        invalid += bad * count
    return zeros, pos, neg, invalid


def _count_trailing_trits(last_byte: int, n_trits: int) -> tuple[int, int, int, int]:
    """Decode the final partial byte (1–3 LSB-first trits); ignore pad bits."""
    zeros = pos = neg = invalid = 0
    for i in range(n_trits):
        t = (last_byte >> (2 * i)) & 0b11
        if t == 0b00:
            zeros += 1
        elif t == 0b01:
            pos += 1
        elif t == 0b10:
            neg += 1
        else:
            invalid += 1
    return zeros, pos, neg, invalid


def histogram_ternary(f, abs_offset: int, n_elements: int) -> dict[str, int]:
    """Count trits for one ternary blob of ``n_elements`` values."""
    nbytes = (n_elements + 3) // 4
    f.seek(abs_offset)
    zeros = pos = neg = invalid = 0
    remaining = nbytes
    last_byte: int | None = None
    pad = n_elements % 4
    while remaining > 0:
        chunk = _read_exact(f, min(CHUNK, remaining))
        remaining -= len(chunk)
        if remaining == 0 and pad != 0:
            last_byte = chunk[-1]
            chunk = chunk[:-1]
        zeros, pos, neg, invalid = _apply_lut_counts(chunk, zeros, pos, neg, invalid)
    if last_byte is not None:
        z, p, m, bad = _count_trailing_trits(last_byte, pad)
        zeros += z
        pos += p
        neg += m
        invalid += bad
    return {"zeros": zeros, "pos": pos, "neg": neg, "invalid": invalid}


def _type_name(tensor_type: int, name: str) -> str:
    if tensor_type == TENSOR_TERNARY:
        return "ternary"
    if tensor_type == TENSOR_F16:
        return "f16"
    raise Goz1Error(
        f"unsupported tensor_type {tensor_type} for {name} "
        f"(expected {TENSOR_F16}=f16 or {TENSOR_TERNARY}=ternary)"
    )


def _payload_nbytes(tensor_type: int, n_elements: int, name: str) -> int:
    if tensor_type == TENSOR_TERNARY:
        return (n_elements + 3) // 4
    if tensor_type == TENSOR_F16:
        return n_elements * 2
    raise Goz1Error(
        f"unsupported tensor_type {tensor_type} for {name} "
        f"(expected {TENSOR_F16}=f16 or {TENSOR_TERNARY}=ternary)"
    )


def _validate_tensor_layout(
    tensors: list[dict], data_start: int, file_size: int
) -> None:
    """Match weight_pack_read::verify_pack_file cumulative offsets + bounds."""
    expected_rel = 0
    for i, t in enumerate(tensors):
        name = t["name"]
        off = t["data_offset"]
        if off != expected_rel:
            raise Goz1Error(
                f"tensor {i} ({name}) data_offset {off} != expected cumulative {expected_rel}"
            )
        n = _num_elements(t["shape"])
        nbytes = _payload_nbytes(t["tensor_type"], n, name)
        abs_off = data_start + off
        end = abs_off + nbytes
        if end > file_size:
            raise Goz1Error(
                f"tensor {i} ({name}) blob end {end} exceeds file size {file_size}"
            )
        expected_rel = _align_up(expected_rel + nbytes, DATA_ALIGNMENT)
    if data_start + expected_rel > file_size:
        raise Goz1Error(
            f"declared tensor payloads need {expected_rel} bytes past data start; "
            f"file has {file_size - data_start}"
        )


def _analyze_ternary(f, abs_offset: int, name: str, n: int) -> dict[str, object]:
    hist = histogram_ternary(f, abs_offset, n)
    total = hist["zeros"] + hist["pos"] + hist["neg"] + hist["invalid"]
    if total != n:
        raise Goz1Error(f"trit count {total} != num_elements {n} for {name}")
    return {
        "histogram": hist,
        "sparsity": hist["zeros"] / n if n else 1.0,
    }


def analyze(path: Path) -> dict:
    file_size = path.stat().st_size
    with path.open("rb") as f:
        version, metadata, tensors, data_start = read_header(f)
        _validate_tensor_layout(tensors, data_start, file_size)
        out_tensors = []
        for t in tensors:
            n = _num_elements(t["shape"])
            entry: dict[str, object] = {
                "name": t["name"],
                "shape": t["shape"],
                "num_elements": n,
                "type": _type_name(t["tensor_type"], t["name"]),
                # Present as None on packs whose version predates the field;
                # see read_header.
                "scale": t["scale"],
                "gif_threshold": t["gif_threshold"],
                "threshold_abs": t["threshold_abs"],
            }
            if t["tensor_type"] == TENSOR_TERNARY:
                entry.update(
                    _analyze_ternary(f, data_start + t["data_offset"], t["name"], n)
                )
            out_tensors.append(entry)
    return {
        "file": str(path),
        "file_size": file_size,
        "version": version,
        "metadata": metadata,
        "tensors": out_tensors,
    }


def _print_ternary_human(t: dict) -> None:
    shape = "x".join(str(d) for d in t["shape"])
    h = t["histogram"]
    n = t["num_elements"]
    print(f"  {t['name']}  [{shape}]  n={n}")
    if n == 0:
        inv = f"  INVALID={h['invalid']}" if h["invalid"] else ""
        print(f"    empty tensor (zeros=0, +1=0, -1=0); sparsity=1.0 by convention{inv}")
        return
    inv = f"  INVALID={h['invalid']}" if h["invalid"] else ""
    print(
        f"    zeros={h['zeros']} ({100.0 * h['zeros'] / n:.4f}%)  "
        f"+1={h['pos']} ({100.0 * h['pos'] / n:.4f}%)  "
        f"-1={h['neg']} ({100.0 * h['neg'] / n:.4f}%){inv}"
    )


def _print_human(result: dict) -> None:
    print(f"{result['file']}  ({result['file_size']} bytes, version {result['version']})")
    gif = result["metadata"].get("oz.gif_threshold")
    if gif is not None:
        print(f"  oz.gif_threshold = {gif}")
    for t in result["tensors"]:
        if t["type"] != "ternary":
            shape = "x".join(str(d) for d in t["shape"])
            print(f"  {t['name']}  [{shape}]  f16/preserve (not histogrammed)")
            continue
        _print_ternary_human(t)


def _json_out_conflicts_with_pack(json_out: Path, pack_path: Path) -> bool:
    """True if --json-out would overwrite the input pack (path or same inode)."""
    out_path = json_out.expanduser()
    out_resolved = out_path.resolve()
    if out_resolved == pack_path:
        return True
    if not out_path.exists():
        return False
    try:
        return out_resolved.samefile(pack_path)
    except OSError:
        return False


def _emit_result(result: dict, *, as_json: bool, json_out: Path | None) -> None:
    if json_out is not None:
        json_out.expanduser().write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    if as_json:
        json.dump(result, sys.stdout, indent=2)
        print()
    else:
        _print_human(result)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pack", type=Path, help="path to .goz1 file")
    ap.add_argument("--json", action="store_true", help="emit JSON on stdout instead of text")
    ap.add_argument(
        "--json-out",
        type=Path,
        help="write JSON to this path (still prints human text unless --json)",
    )
    args = ap.parse_args(argv)
    try:
        pack_path = args.pack.expanduser().resolve()
        if args.json_out is not None and _json_out_conflicts_with_pack(
            args.json_out, pack_path
        ):
            raise Goz1Error(
                f"--json-out {args.json_out} resolves to the input pack; "
                "refusing to overwrite the GOZ1 artifact"
            )
        result = analyze(pack_path)
        # Write failures (missing parent dir, permissions, full disk) stay in
        # the same controlled error path as analyze().
        _emit_result(result, as_json=args.json, json_out=args.json_out)
    except (OSError, Goz1Error) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
