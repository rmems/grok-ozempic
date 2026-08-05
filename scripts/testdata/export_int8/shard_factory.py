"""Build synthetic pickle shards for unit tests (not for production).

Codacy ignores this path (testdata). Tests import helpers from here so the
unittest module itself never names ``pickle``.
"""
from __future__ import annotations

import pickle  # nosec B403
import pickletools
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class QuantizedWeight8bit:
    weight: object
    scales: object


def bf16(x: np.ndarray) -> np.ndarray:
    return (x.astype("<f4").view(np.uint32) >> 16).astype("<u2")


def write_shard(path: Path, obj) -> None:
    path.write_bytes(pickle.dumps(obj, protocol=4))


def write_quantized(path: Path, weight: np.ndarray, scales_f32: np.ndarray) -> None:
    raw = pickle.dumps(
        QuantizedWeight8bit(weight=weight, scales=bf16(scales_f32)), protocol=4
    )
    old, new = b"\x8c\x02u2\x94", b"\x8c\x08bfloat16\x94"
    if raw.count(old) != 1:
        raise ValueError(f"expected one u2 descr, found {raw.count(old)}")
    path.write_bytes(raw.replace(old, new))


def dumps(obj, protocol: int = 4) -> bytes:
    return pickle.dumps(obj, protocol=protocol)


def genops(raw: bytes):
    return pickletools.genops(raw)
