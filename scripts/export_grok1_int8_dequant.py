#!/usr/bin/env python3
"""Dequant / npy write path for Grok-1 int8 export."""
from __future__ import annotations

from pathlib import Path

from export_grok1_int8_scan import ArraySpec, ExportError, scan_shard, split_quantized

DEFAULT_CHUNK_MIB = 256

def grouping(weight: ArraySpec, scales: ArraySpec) -> tuple[tuple[int, ...], int, int, int]:
    """Validate the grouped-scale layout; return ``(lead, K, N, G)``.

    ``weight`` is ``(*lead, K, N)`` and ``scales`` is ``(*lead, G, N)`` with
    ``K % G == 0``. Any deviation is fatal: silently guessing a broadcast would
    produce a plausible-looking but wrong tensor.
    """
    if len(weight.shape) < 2 or len(scales.shape) != len(weight.shape):
        raise ExportError(
            f"rank mismatch: weight {weight.shape} vs scales {scales.shape}"
        )
    lead, k, n = weight.shape[:-2], weight.shape[-2], weight.shape[-1]
    if scales.shape[:-2] != lead:
        raise ExportError(
            f"leading dims differ: weight {weight.shape} vs scales {scales.shape}"
        )
    if scales.shape[-1] != n:
        raise ExportError(
            f"scales last dim {scales.shape[-1]} != weight output dim {n}"
        )
    g = scales.shape[-2]
    if g == 0 or k % g != 0:
        raise ExportError(f"contracting dim {k} not divisible by scale groups {g}")
    return lead, k, n, g


def _chunk_rows(chunk_mib: int, bytes_per_row: int) -> int:
    """Row count for a target chunk size; reject non-positive inputs."""
    if chunk_mib <= 0:
        raise ExportError(f"--chunk-mib must be a positive integer, got {chunk_mib}")
    if bytes_per_row <= 0:
        raise ExportError(f"invalid bytes_per_row {bytes_per_row}")
    return max(1, (chunk_mib * 1024 * 1024) // bytes_per_row)


def npy_header(shape: tuple[int, ...], descr: str = "<f4") -> bytes:
    """Build a v1.0 ``.npy`` header padded so the payload starts 64-byte aligned."""
    shape_txt = "(" + "".join(f"{d}," for d in shape) + ")"
    body = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_txt}, }}"
    prefix = len(b"\x93NUMPY") + 2 + 2
    pad = -(prefix + len(body) + 1) % 64
    body = body + " " * pad + "\n"
    return b"\x93NUMPY" + bytes([1, 0]) + len(body).to_bytes(2, "little") + body.encode("latin1")


def export_tensor(
    shard: Path,
    out_path: Path,
    *,
    chunk_mib: int = DEFAULT_CHUNK_MIB,
    dry_run: bool = False,
    expect_shape: tuple[int, ...] | list[int] | None = None,
) -> dict:
    """Dequantize (or pass through) one shard into an f32 ``.npy``.

    Returns a summary dict; with ``dry_run`` nothing is written. ``expect_shape``
    is checked **before** any bytes are written, so a manifest/shard disagreement
    never leaves a multi-GiB wrong file behind for a later pack to consume.
    """
    import numpy as np

    weight, scales = split_quantized(scan_shard(shard))
    if expect_shape is not None and tuple(expect_shape) != weight.shape:
        raise ExportError(
            f"{shard.name}: manifest shape {tuple(expect_shape)} != shard shape "
            f"{weight.shape}; refusing to write {out_path.name}"
        )
    info: dict = {
        "shard": str(shard),
        "output": str(out_path),
        "shape": list(weight.shape),
        "source_dtype": "f32" if scales is None else "int8 x bf16",
        "out_bytes": weight.numel * 4,
        "scale_groups": None,
    }
    if scales is not None:
        _lead, k, _n, g = grouping(weight, scales)
        info["scale_groups"] = g
        info["group_rows"] = k // g
    if dry_run:
        return info

    if scales is None:
        src = np.memmap(
            shard, dtype="<f4", mode="r", offset=weight.offset, shape=weight.shape
        )
        _write_npy(out_path, weight.shape, src, chunk_mib=chunk_mib)
    else:
        _write_npy(
            out_path,
            weight.shape,
            _dequant_chunks(shard, weight, scales, chunk_mib),
            chunk_mib=chunk_mib,
            streaming=True,
        )
    return info


def _dequant_chunks(shard: Path, weight: ArraySpec, scales: ArraySpec, chunk_mib: int):
    """Yield f32 row-blocks of ``weight * scales``, never holding the whole tensor.

    Each block stays inside one scale group, so a single broadcast row applies.
    """
    import numpy as np

    lead, k, n, g = grouping(weight, scales)
    w = np.memmap(shard, dtype=np.int8, mode="r", offset=weight.offset, shape=weight.shape)
    s_raw = np.memmap(shard, dtype="<u2", mode="r", offset=scales.offset, shape=scales.shape)
    # bfloat16 -> f32 is an exact 16-bit left shift of the bit pattern.
    s = (s_raw.astype(np.uint32) << 16).view(np.float32)

    lead_n = 1
    for d in lead:
        lead_n *= d
    wf = w.reshape(lead_n, k, n)
    sf = s.reshape(lead_n, g, n)
    group_rows = k // g
    rows_per_chunk = _chunk_rows(chunk_mib, n * 4)

    for li in range(lead_n):
        for gi in range(g):
            base = gi * group_rows
            scale_row = sf[li, gi]
            for r0 in range(0, group_rows, rows_per_chunk):
                r1 = min(r0 + rows_per_chunk, group_rows)
                chunk = wf[li, base + r0 : base + r1].astype(np.float32)
                chunk *= scale_row
                yield chunk


def _write_npy(out_path, shape, source, *, chunk_mib: int, streaming: bool = False) -> None:
    """Write an f32 ``.npy`` atomically from an array or an iterable of chunks."""
    import numpy as np

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".partial")
    try:
        with open(tmp, "wb") as fh:
            fh.write(npy_header(tuple(shape)))
            if streaming:
                fh.writelines(
                    np.ascontiguousarray(chunk, dtype="<f4").tobytes() for chunk in source
                )
            else:
                rows = _chunk_rows(chunk_mib, max(1, source[0].nbytes))
                fh.writelines(
                    np.ascontiguousarray(source[r0 : r0 + rows], dtype="<f4").tobytes()
                    for r0 in range(0, len(source), rows)
                )
        tmp.replace(out_path)
    finally:
        tmp.unlink(missing_ok=True)


