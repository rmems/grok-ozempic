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
    """Emit a shard whose scales carry the ``bfloat16`` dtype descriptor.

    numpy cannot name a ``bfloat16`` dtype without ``ml_dtypes``, so the array
    is pickled as ``<u2`` and the descriptor bytes are rewritten in place --
    same 2-byte itemsize, same payload, the framing JAX/ml_dtypes produces.

    The substitution is pinned to how numpy frames a short dtype string under
    pickle protocol 4 (``SHORT_BINUNICODE`` + memoize), as produced by **numpy
    2.5.1 / CPython 3.14**. A byte-order prefix (``<u2``), a different memo
    layout, or a newer default protocol would change those bytes; the count
    assertion below turns that into a loud generation-time failure rather than
    a silently malformed shard. See this directory's README before relaxing it.
    """
    raw = pickle.dumps(
        QuantizedWeight8bit(weight=weight, scales=bf16(scales_f32)), protocol=4
    )
    old, new = b"\x8c\x02u2\x94", b"\x8c\x08bfloat16\x94"
    if raw.count(old) != 1:
        raise ValueError(
            f"expected exactly one u2 dtype descriptor, found {raw.count(old)}; "
            f"numpy {np.__version__} may frame dtypes differently -- see "
            "scripts/testdata/export_int8/README.md"
        )
    path.write_bytes(raw.replace(old, new))


def write_quantized_global_dtype(
    path: Path, weight: np.ndarray, scales_f32: np.ndarray
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
    # SHORT_BINUNICODE 'ml_dtypes' + MEMOIZE, 'bfloat16' + MEMOIZE, STACK_GLOBAL, MEMOIZE
    new = (
        b"\x8c\x09ml_dtypes\x94"
        b"\x8c\x08bfloat16\x94"
        b"\x93\x94"
    )
    path.write_bytes(raw.replace(old, new))


def dumps(obj, protocol: int = 4) -> bytes:
    return pickle.dumps(obj, protocol=protocol)


def genops(raw: bytes):
    return pickletools.genops(raw)
