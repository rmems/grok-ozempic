"""Build synthetic pickle shards for unit tests (not for production).

Codacy ignores this path (testdata). Tests import helpers from here so the
unittest module itself never names ``pickle``.
"""
from __future__ import annotations

import pickle  # nosec B403
import pickletools  # nosec B403
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
    """Emit a shard framed exactly as the official ckpt-0 frames bfloat16 scales.

    numpy cannot name a ``bfloat16`` dtype without ``ml_dtypes``, so the array is
    pickled as ``<u2`` and the descriptor is rewritten into the opcodes that push
    ``ml_dtypes.bfloat16`` -- same 2-byte itemsize, same payload, the framing
    JAX/ml_dtypes actually produces. Fixtures therefore exercise the real code
    path rather than a synthetic bare-string spelling the scanner rejects.
    """
    write_quantized_global_dtype(path, weight, scales_f32)


def write_quantized_string_dtype(
    path: Path, weight: np.ndarray, scales_f32: np.ndarray, descr: str = "bfloat16"
) -> None:
    """Emit a shard naming the scales dtype as a bare descriptor string.

    Not a framing numpy or ml_dtypes ever produces for bfloat16 -- it exists so
    tests can confirm the scanner refuses to take a 2-byte element size from a
    loose string in the stream.
    """
    raw = pickle.dumps(
        QuantizedWeight8bit(weight=weight, scales=bf16(scales_f32)), protocol=4
    )
    old = b"\x8c\x02u2\x94"
    if raw.count(old) != 1:
        raise ValueError(f"expected one u2 descr, found {raw.count(old)}")
    text = descr.encode("utf-8")
    # SHORT_BINUNICODE carries a single length byte, so >255 would raise an opaque
    # "bytes must be in range" from bytes([...]) instead of naming the cause.
    if len(text) > 255:
        raise ValueError(f"descr {descr!r} too long for SHORT_BINUNICODE ({len(text)} bytes)")
    path.write_bytes(raw.replace(old, b"\x8c" + bytes([len(text)]) + text + b"\x94"))


def write_quantized_global_dtype(
    path: Path,
    weight: np.ndarray,
    scales_f32: np.ndarray,
    module: str = "ml_dtypes",
    name: str = "bfloat16",
) -> None:
    """Emit a shard naming bfloat16 via ``STACK_GLOBAL``, as ckpt-0 really does.

    :func:`write_quantized` rewrites the descriptor into a plain string, but the
    official checkpoint reaches ``numpy.dtype`` through
    ``STACK_GLOBAL ml_dtypes bfloat16``. The scanner must recover the descriptor
    from that framing too, so this builds it explicitly: the ``<u2`` descriptor
    string is replaced by the opcodes that push ``ml_dtypes.bfloat16``.
    """
    raw = pickle.dumps(
        QuantizedWeight8bit(weight=weight, scales=bf16(scales_f32)), protocol=4
    )
    old = b"\x8c\x02u2\x94"  # SHORT_BINUNICODE 'u2' + MEMOIZE
    if raw.count(old) != 1:
        raise ValueError(
            f"expected exactly one u2 dtype descriptor, found {raw.count(old)}; "
            f"numpy {np.__version__} may frame dtypes differently -- see "
            "scripts/testdata/export_int8/README.md"
        )
    # SHORT_BINUNICODE <module> + MEMOIZE, <name> + MEMOIZE, STACK_GLOBAL, MEMOIZE.
    # `module`/`name` are parameterized so tests can build an *untrusted* global
    # and confirm the scanner rejects it instead of trusting the name alone.
    def short_binunicode(text: str) -> bytes:
        raw_text = text.encode("utf-8")
        if len(raw_text) > 255:
            raise ValueError(f"{text!r} too long for SHORT_BINUNICODE")
        return b"\x8c" + bytes([len(raw_text)]) + raw_text + b"\x94"

    new = short_binunicode(module) + short_binunicode(name) + b"\x93\x94"
    path.write_bytes(raw.replace(old, new))


def dumps(obj, protocol: int = 4) -> bytes:
    return pickle.dumps(obj, protocol=protocol)


def genops(raw: bytes):
    return pickletools.genops(raw)
