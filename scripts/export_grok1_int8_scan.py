#!/usr/bin/env python3
"""Pickle-frame ndarray scanner for Grok-1 int8 export (no unpickle).

Walks opcode streams with ``pickletools.genops`` only — never imports
STACK_GLOBAL targets or executes checkpoint-controlled data.
"""
from __future__ import annotations

import io
import mmap as mmap_mod

# pickletools only *decodes* opcodes; nothing here unpickles.
import pickletools  # nosec B403
import struct
from dataclasses import dataclass
from pathlib import Path

# Window big enough for any numpy `_reconstruct` header; payloads are skipped.
_SCAN_WINDOW = 1 << 16
# Reads at or above this are payloads, not opcode arguments.
_PAYLOAD_READ_LIMIT = 4096
_MIN_PAYLOAD_BYTES = 16
_PAYLOAD_OPS = {"BINBYTES": 5, "BINBYTES8": 9, "SHORT_BINBYTES": 2}

_INT_DESCR = {"i1"}
_F32_DESCR = {"f4"}
_BF16_DESCR = {"bfloat16"}
_KNOWN_DESCR = _INT_DESCR | _F32_DESCR | _BF16_DESCR


class ExportError(RuntimeError):
    """Layout, manifest or CLI problem that makes an export unsafe."""


@dataclass(frozen=True)
class ArraySpec:
    """One ndarray recovered from a pickle frame."""

    shape: tuple[int, ...]
    descr: str
    offset: int
    nbytes: int
    fortran_order: bool = False

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def itemsize(self) -> int:
        return {"i1": 1, "bfloat16": 2, "f4": 4}[self.descr]

    def validate(self) -> None:
        want = self.numel * self.itemsize
        if want != self.nbytes:
            raise ExportError(
                f"payload size mismatch for shape {self.shape} descr {self.descr!r}: "
                f"expected {want} bytes, found {self.nbytes}"
            )


class _PayloadBoundary(Exception):
    """Raised when the opcode stream reaches an array payload."""

    def __init__(self, offset: int, nbytes: int) -> None:
        super().__init__(offset, nbytes)
        self.offset = offset
        self.nbytes = nbytes


class _StopAtPayload(io.BytesIO):
    """Byte stream that reports large reads instead of materializing them."""

    def read(self, size: int | None = -1) -> bytes:  # type: ignore[override]
        if size is not None and size >= _PAYLOAD_READ_LIMIT:
            raise _PayloadBoundary(self.tell(), size)
        return super().read(size)


class _HeaderState:
    """ndarray header facts accumulated as opcodes stream past."""

    def __init__(self) -> None:
        self.ints: list[int] = []
        self.shape: tuple[int, ...] | None = None
        self.descr: str | None = None
        self.fortran: bool | None = None

    def feed(self, opname: str, arg) -> None:
        if opname in ("BININT", "BININT1", "BININT2"):
            self.ints.append(int(arg))
        elif opname in ("TUPLE1", "TUPLE2", "TUPLE3"):
            k = int(opname[-1])
            if len(self.ints) >= k:
                self.shape = tuple(self.ints[-k:])
            self.ints = []
        elif opname == "SHORT_BINUNICODE" and arg in _KNOWN_DESCR:
            self.descr = str(arg)
        elif opname in ("NEWFALSE", "NEWTRUE"):
            self.fortran = opname == "NEWTRUE"

    def spec(self, name: str, offset: int, nbytes: int) -> ArraySpec:
        if self.shape is None or self.descr is None:
            raise ExportError(
                f"{name}: payload at byte {offset} has no preceding "
                "shape/dtype -- unrecognized pickle layout"
            )
        spec = ArraySpec(self.shape, self.descr, offset, nbytes, bool(self.fortran))
        spec.validate()
        return spec


def _payload_at(
    state: _HeaderState, opname: str, arg, pos: int, base: int, name: str
) -> ArraySpec | None:
    header = _PAYLOAD_OPS.get(opname)
    if header is None:
        state.feed(opname, arg)
        return None
    nbytes = len(arg) if isinstance(arg, bytes) else int(arg)
    if nbytes < _MIN_PAYLOAD_BYTES:
        return None
    return state.spec(name, base + pos + header, nbytes)


def _scan_window(window: bytes, base: int, name: str) -> ArraySpec | None:
    state = _HeaderState()
    stream = _StopAtPayload(window)
    try:
        for op, arg, pos in pickletools.genops(stream):
            found = _payload_at(state, op.name, arg, pos, base, name)
            if found is not None:
                return found
            if op.name == "STOP":
                return None
    except _PayloadBoundary as boundary:
        return state.spec(name, base + boundary.offset, boundary.nbytes)
    except (ValueError, IndexError, AssertionError, struct.error, EOFError):
        return None
    return None


def _reject_unsupported(spec: ArraySpec, size: int, name: str) -> None:
    if spec.offset + spec.nbytes > size:
        raise ExportError(
            f"{name}: {spec.descr} payload of {spec.nbytes} bytes at {spec.offset} "
            f"runs past end of file ({size}) -- truncated shard"
        )
    if spec.fortran_order:
        raise ExportError(
            f"{name}: array {spec.shape} is Fortran-ordered; this exporter writes "
            "C-order .npy and will not silently transpose"
        )


def _append_specs_from_mmap(mm, size: int, name: str) -> list[ArraySpec]:
    """Walk one mmap, collecting every ndarray payload."""
    specs: list[ArraySpec] = []
    base = 0
    while base < size:
        spec = _scan_window(mm[base : min(size, base + _SCAN_WINDOW)], base, name)
        if spec is None:
            break
        _reject_unsupported(spec, size, name)
        specs.append(spec)
        base = spec.offset + spec.nbytes
    return specs


def scan_shard(path: Path) -> list[ArraySpec]:
    """Recover every ndarray in a pickle shard without unpickling it."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ExportError(f"{path}: cannot stat shard: {exc}") from exc
    if size == 0:
        raise ExportError(f"{path.name}: empty file, no ndarray payload found")

    with open(path, "rb") as fh:
        mm = mmap_mod.mmap(fh.fileno(), 0, access=mmap_mod.ACCESS_READ)
        try:
            specs = _append_specs_from_mmap(mm, size, path.name)
        finally:
            mm.close()
    if not specs:
        raise ExportError(f"{path.name}: no ndarray payload found")
    return specs


def split_quantized(specs: list[ArraySpec]) -> tuple[ArraySpec, ArraySpec | None]:
    """Return ``(weight, scales)``; ``scales`` is ``None`` for plain f32 tensors."""
    if len(specs) == 1:
        w = specs[0]
        if w.descr not in _F32_DESCR:
            raise ExportError(
                f"single-array shard has dtype {w.descr!r}; expected f32 passthrough"
            )
        return w, None
    if len(specs) == 2:
        w, s = specs
        if w.descr not in _INT_DESCR or s.descr not in _BF16_DESCR:
            raise ExportError(
                f"expected int8 weight + bf16 scales, found {w.descr!r} + {s.descr!r}"
            )
        return w, s
    raise ExportError(f"expected 1 or 2 arrays in shard, found {len(specs)}")
