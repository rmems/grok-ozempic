#!/usr/bin/env python3
"""Offline generator for scripts/testdata/export_int8/*.bin (dev only).

Not imported by unit tests or CI. Uses pickle.dumps once to create golden
shards that match official numpy/JAX ndarray pickle frames. Tests load the
committed bytes only so Codacy/Bandit never see pickle in the test module.

  python3 scripts/dev_generate_int8_export_fixtures.py
"""
from __future__ import annotations

import pickle  # nosec B403  # offline fixture generator only
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "testdata" / "export_int8"


@dataclass
class QuantizedWeight8bit:
    weight: object
    scales: object


def _bf16(x: np.ndarray) -> np.ndarray:
    return (x.astype("<f4").view(np.uint32) >> 16).astype("<u2")


def _write_plain(name: str, arr: np.ndarray) -> None:
    (OUT / name).write_bytes(pickle.dumps(arr, protocol=4))


def _write_quant(name: str, weight: np.ndarray, scales_f32: np.ndarray) -> None:
    raw = pickle.dumps(
        QuantizedWeight8bit(weight=weight, scales=_bf16(scales_f32)), protocol=4
    )
    old, new = b"\x8c\x02u2\x94", b"\x8c\x08bfloat16\x94"
    if raw.count(old) != 1:
        raise SystemExit(f"{name}: expected one u2 descr, found {raw.count(old)}")
    (OUT / name).write_bytes(raw.replace(old, new))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _write_plain("plain_f32_4x6.bin", np.arange(24, dtype="<f4").reshape(4, 6))
    _write_quant(
        "quant_8x8.bin",
        np.arange(-32, 32, dtype=np.int8).reshape(8, 8),
        np.linspace(0.5, 1.5, 8, dtype="<f4").reshape(1, 8),
    )
    _write_plain("f32_4096.bin", np.arange(4096, dtype="<f4"))
    _write_plain(
        "fortran_64x64.bin",
        np.asfortranarray(np.arange(4096, dtype="<f4").reshape(64, 64)),
    )
    _write_plain("large_zeros_16k_f32.bin", np.zeros(1 << 14, dtype="<f4"))

    rng = np.random.default_rng(7)
    _write_quant(
        "dequant_ungrouped.bin",
        rng.integers(-128, 128, size=(64, 32), dtype=np.int8),
        rng.random((1, 32), dtype=np.float32) + 0.25,
    )
    rng = np.random.default_rng(11)
    _write_quant(
        "dequant_grouped.bin",
        rng.integers(-128, 128, size=(64, 16), dtype=np.int8),
        rng.random((8, 16), dtype=np.float32) + 0.25,
    )
    rng = np.random.default_rng(13)
    _write_quant(
        "dequant_lead.bin",
        rng.integers(-128, 128, size=(3, 32, 8), dtype=np.int8),
        rng.random((3, 4, 8), dtype=np.float32) + 0.25,
    )
    rng = np.random.default_rng(17)
    _write_quant(
        "dequant_chunk.bin",
        rng.integers(-128, 128, size=(128, 64), dtype=np.int8),
        rng.random((8, 64), dtype=np.float32) + 0.25,
    )

    _write_plain("f32_passthrough.bin", (np.arange(120, dtype="<f4") / 7.0).reshape(10, 12))
    _write_plain("zeros_4x4.bin", np.zeros((4, 4), dtype="<f4"))
    _write_plain("arange_4x4.bin", np.arange(16, dtype="<f4").reshape(4, 4))
    print(f"wrote fixtures under {OUT}")


if __name__ == "__main__":
    main()
