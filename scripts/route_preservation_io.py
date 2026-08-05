#!/usr/bin/env python3
"""GOZ1 pack I/O for route-preservation metrics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from goz1_trit_histogram import (  # noqa: E402
    DATA_ALIGNMENT,
    TENSOR_F16,
    TENSOR_TERNARY,
    _align_up,
    _num_elements,
    _payload_nbytes,
    read_header,
)

# Re-export tensor type constants for callers.
__all__ = [
    "MetricsError",
    "TENSOR_F16",
    "TENSOR_TERNARY",
    "load_pack_index",
    "read_trits",
    "read_f16",
]

# Trit code -> value, matching quantizer.rs encode_trit (0b00=0, 0b01=+1, 0b10=-1).
_TRIT_LUT = np.zeros((256, 4), dtype=np.int8)
_INVALID_LUT = np.zeros((256, 4), dtype=bool)
for _b in range(256):
    for _s in range(4):
        _code = (_b >> (2 * _s)) & 0b11
        _TRIT_LUT[_b, _s] = 1 if _code == 0b01 else (-1 if _code == 0b10 else 0)
        _INVALID_LUT[_b, _s] = _code == 0b11


class MetricsError(RuntimeError):
    """Pack/npy mismatch that makes a comparison meaningless."""


def load_pack_index(pack: Path) -> tuple[dict, dict[str, dict]]:
    """Return ``(metadata, {name: entry})`` with absolute payload offsets."""
    with pack.open("rb") as f:
        _version, metadata, tensors, data_start = read_header(f)
    index: dict[str, dict] = {}
    rel = 0
    for t in tensors:
        n = _num_elements(t["shape"])
        nbytes = _payload_nbytes(t["tensor_type"], n, t["name"])
        if t["data_offset"] != rel:
            raise MetricsError(
                f"{t['name']}: data_offset {t['data_offset']} != cumulative {rel}"
            )
        index[t["name"]] = {
            **t,
            "numel": n,
            "nbytes": nbytes,
            "abs_offset": data_start + rel,
        }
        rel = _align_up(rel + nbytes, DATA_ALIGNMENT)
    return metadata, index


def read_trits(pack: Path, entry: dict, start: int, count: int) -> np.ndarray:
    """Decode ``count`` trits starting at flat index ``start``."""
    byte0, skip = divmod(start, 4)
    nbytes = (skip + count + 3) // 4
    with pack.open("rb") as f:
        f.seek(entry["abs_offset"] + byte0)
        raw = f.read(nbytes)
    if len(raw) != nbytes:
        raise MetricsError(
            f"{entry['name']}: truncated pack -- wanted {nbytes} bytes at payload "
            f"offset {byte0} (flat trit {start}), got {len(raw)}"
        )
    buf = np.frombuffer(raw, dtype=np.uint8)
    if _INVALID_LUT[buf].any():
        raise MetricsError(
            f"{entry['name']}: invalid 0b11 trit code near flat index {start} -- corrupt pack"
        )
    return _TRIT_LUT[buf].reshape(-1)[skip : skip + count]


def read_f16(pack: Path, entry: dict) -> np.ndarray:
    """Read a preserve-tier fp16 payload; short reads are MetricsError (exit 2)."""
    with pack.open("rb") as f:
        f.seek(entry["abs_offset"])
        raw = f.read(entry["nbytes"])
    if len(raw) != entry["nbytes"]:
        raise MetricsError(
            f"{entry['name']}: truncated preserve payload -- wanted {entry['nbytes']} "
            f"bytes, got {len(raw)}"
        )
    return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(entry["shape"])
