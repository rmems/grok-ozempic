#!/usr/bin/env python3
r"""
Export Grok-1 official ``ckpt-0`` shards to stream-compatible f32 ``.npy`` files,
**dequantizing** ``QuantizedWeight8bit`` (int8 weight x bfloat16 scales) payloads.

Why this exists (GH #53 / RM-222, beads ``goz-dus0``)
----------------------------------------------------
``scripts/export_grok1_embedding_npy.py`` is stdlib-only and f32-only: it copies a
raw float payload out of a pickle frame. That covers the token embedding, norms
and routers, but the official ``xai-org/grok-1`` release ships **every attention
projection and MoE expert as int8** ``__main__.QuantizedWeight8bit``. The
``grok-ozempic`` float stream rejects int8 (``SourceDtype::Other``), so a bounded
block pilot has nothing to pack until those tensors are dequantized to f32.

Observed on-disk layout (parsed, never unpickled -- see "Safety" below)::

    __main__.QuantizedWeight8bit
      .weight  ndarray int8   ( *lead, K, N )
      .scales  ndarray bf16   ( *lead, G, N )      K % G == 0

``scales`` are **grouped along the contracting axis** ``K``: group ``g`` covers
rows ``[g*K/G, (g+1)*K/G)``. ``G`` is the tensor-parallel shard count of the axis
in the original 8-way checkpoint (``G = 8`` when ``K`` was sharded, ``G = 1``
when the output axis ``N`` was sharded instead). Dequantization is therefore::

    w_f32 = weight.reshape(*lead, G, K // G, N) * scales[..., :, None, :]

Verified against ckpt-0 block 0:

===========================  ==================  ==================  =====
structural name              weight shape        scales shape        G
===========================  ==================  ==================  =====
slot_00.moe_expert.gate      (8, 6144, 32768)    (8, 1, 32768)       1
slot_01.moe_expert.down      (8, 32768, 6144)    (8, 8, 6144)        8
slot_02.moe_expert.up        (8, 6144, 32768)    (8, 1, 32768)       1
slot_03.attn_proj_i8.narrow  (6144, 1024)        (1, 1024)           1
slot_04.attn_proj_i8.m_width (6144, 6144)        (8, 6144)           8
===========================  ==================  ==================  =====

Plain (non-quantized) f32 tensors -- ``block_norm``, ``router``,
``token_embedding`` -- are passed through byte-for-byte, so one invocation can
materialize a whole block (preserve tier included) for a GOZ1 pack.

Safety
------
The shard is **never unpickled**. ``pickletools.genops`` walks the opcode stream
to recover each ndarray's shape, dtype and payload byte offset; no
``STACK_GLOBAL`` target is ever imported or called. Payloads are then read
through ``numpy.memmap`` at those offsets.

Scanning is memory-bounded. ``genops`` would otherwise materialize each
``BINBYTES`` argument -- 1.6 GB for one expert tensor -- so the scanner hands it
a stream that refuses reads at or above ``_PAYLOAD_READ_LIMIT`` and reports the
offset/length instead. Scanning then resumes in a fresh window just past the
payload. No opcode in these shards legitimately reads that much (the largest is
``SHORT_BINUNICODE`` at 255 bytes), so the interception is unambiguous.

Dependencies
------------
Unlike ``export_grok1_embedding_npy.py`` (stdlib-only, byte-copy), dequantization
is real arithmetic over up to 1.6e9 elements, so this script **requires numpy**
(already a CI dependency of ``.github/workflows/python-scripts.yml``).

Output is written in bounded chunks and no tensor is ever fully materialized.
Peak RSS is dominated by ``mmap`` page residency over the source shard rather
than by ``--chunk-mib``: exporting all of block 0 (18.3 GiB of f32 out, largest
source payload 1.6 GB) measured **2.03 GiB** peak.

Filename stems use ``__`` for ``.`` so ``npy_stem_to_tensor_name`` recovers the
V2 structural logical name, e.g.
``block_000__slot_04__attn_proj_i8__model_width.npy`` ->
``block_000.slot_04.attn_proj_i8.model_width``.

Usage::

    # whole pilot block, attention + experts + preserve tier
    python3 scripts/export_grok1_int8_npy.py \
      --conversion-manifest "$RUN3/conversion-manifest.json" \
      --block 0 --mode attention_plus_expert \
      --output-dir ~/.models/xai-grok-1/export-npy/block000-attn-expert

    # single tensor by structural name
    python3 scripts/export_grok1_int8_npy.py \
      --conversion-manifest "$RUN3/conversion-manifest.json" \
      --structural-name block_000.slot_04.attn_proj_i8.model_width \
      --output-dir /tmp/one

    # inspect the raw layout of any shard (no export)
    python3 scripts/export_grok1_int8_npy.py --inspect ~/.models/xai-grok-1/ckpt-0/tensor00006_000
"""

from __future__ import annotations

import argparse
import io
import json
import mmap as mmap_mod

# pickletools only *decodes* opcodes; nothing here unpickles, imports a
# STACK_GLOBAL target, or executes checkpoint-controlled data. See "Safety".
import pickletools  # nosec B403
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CHUNK_MIB = 256

# Window big enough for any numpy `_reconstruct` header; payloads are skipped.
_SCAN_WINDOW = 1 << 16
# Reads at or above this are payloads, not opcode arguments (max legitimate
# opcode read in these shards is SHORT_BINUNICODE at 255 bytes).
_PAYLOAD_READ_LIMIT = 4096
# Below this a "payload" is ndarray reconstruct scaffolding (b"b"), not data.
_MIN_PAYLOAD_BYTES = 16
# Payload-carrying opcodes -> bytes of opcode+length header before the data.
_PAYLOAD_OPS = {"BINBYTES": 5, "BINBYTES8": 9, "SHORT_BINBYTES": 2}

# numpy dtype descriptors we accept from the pickle stream.
_INT_DESCR = {"i1"}
_F32_DESCR = {"f4"}
_BF16_DESCR = {"bfloat16"}
_KNOWN_DESCR = _INT_DESCR | _F32_DESCR | _BF16_DESCR

# Preserve tier per run3 quant-plan `keep_fp32`; never ternary, never dequantized.
PRESERVE_KINDS = ("router", "block_norm", "final_norm")
ATTENTION_KINDS = ("attn_proj_i8.narrow", "attn_proj_i8.model_width")
EXPERT_KINDS = ("moe_expert.gate", "moe_expert.up", "moe_expert.down")

MODES = {
    "attention_only": ATTENTION_KINDS + PRESERVE_KINDS,
    "expert_only": EXPERT_KINDS + PRESERVE_KINDS,
    "attention_plus_expert": ATTENTION_KINDS + EXPERT_KINDS + PRESERVE_KINDS,
    "preserve_only": PRESERVE_KINDS,
}


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


class _Descr:
    """A dtype descriptor recovered from a pickle `numpy.dtype` reduce."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


class _DtypeBuilder:
    """Placeholder for `numpy.dtype` as a callable in a `REDUCE` frame."""


class _Global:
    """A `STACK_GLOBAL` callable that has not been reduced yet."""

    __slots__ = ("module", "name")

    def __init__(self, module: str, name: str) -> None:
        self.module = module
        self.name = name


class _Unknown:
    """Any other pickle object we do not need to model precisely."""

    __slots__ = ("tag",)

    def __init__(self, tag: str) -> None:
        self.tag = tag


class _HeaderState:
    """ndarray header facts accumulated as opcodes stream past.

    Maintains a minimal pickle value stack and memo so `BINGET`-referenced dtype
    strings (and memoized `numpy.dtype` objects) survive across scanner windows
    and can supply the descriptor for later arrays in the same shard.
    """

    _MARK = object()

    def __init__(self) -> None:
        self.stack: list = []
        self.memo: dict[int, object] = {}
        self.memo_next = 0
        self.ints: list[int] = []
        self.shape: tuple[int, ...] | None = None
        self.descr: str | None = None
        self.fortran: bool | None = None

    def _push(self, value: object) -> None:
        self.stack.append(value)

    def _pop(self) -> object:
        return self.stack.pop()

    def _pop_to_mark(self) -> list[object]:
        """Pop items until the most recent MARK, discarding the mark."""
        items: list[object] = []
        while self.stack:
            v = self.stack.pop()
            if v is self._MARK:
                break
            items.append(v)
        items.reverse()
        return items

    def _maybe_set_shape(self, items: list[object]) -> None:
        if items and all(isinstance(x, int) for x in items):
            self.shape = tuple(items)
            self.ints = []

    def _set_descr(self, value: object) -> None:
        """Extract a known dtype descriptor from a value if possible."""
        if isinstance(value, _Descr):
            self.descr = value.text
            return
        if isinstance(value, str) and value in _KNOWN_DESCR:
            self.descr = value
            return
        if isinstance(value, bytes):
            try:
                s = value.decode("ascii")
            except UnicodeDecodeError:
                return
            if s in _KNOWN_DESCR:
                self.descr = s

    def _memoize_top(self, idx: int) -> None:
        if self.stack:
            self.memo[idx] = self.stack[-1]

    # --- per-opcode handlers (dispatch keeps `feed` shallow for maintainers) ---

    def _op_FRAME(self, arg) -> None:
        return

    def _op_PROTO(self, arg) -> None:
        return

    def _push_str(self, arg) -> None:
        s = str(arg)
        self._push(s)
        self._set_descr(s)

    _op_SHORT_BINUNICODE = _push_str
    _op_BINUNICODE = _push_str
    _op_UNICODE = _push_str

    def _push_bytes(self, arg) -> None:
        self._push(arg)

    _op_SHORT_BINSTRING = _push_bytes
    _op_BINSTRING = _push_bytes
    _op_SHORT_BINBYTES = _push_bytes
    _op_BINBYTES = _push_bytes
    _op_BINBYTES8 = _push_bytes

    def _push_int(self, arg) -> None:
        i = int(arg)
        self._push(i)
        self.ints.append(i)

    _op_BININT = _push_int
    _op_BININT1 = _push_int
    _op_BININT2 = _push_int
    _op_LONG_BININT = _push_int
    _op_INT = _push_int
    _op_LONG = _push_int

    def _push_float(self, arg) -> None:
        self._push(float(arg))

    _op_FLOAT = _push_float
    _op_BINFLOAT = _push_float

    def _op_NONE(self, arg) -> None:
        self._push(None)

    def _op_NEWTRUE(self, arg) -> None:
        self._push(True)
        self.fortran = True

    def _op_NEWFALSE(self, arg) -> None:
        self._push(False)
        self.fortran = False

    def _op_EMPTY_TUPLE(self, arg) -> None:
        self._push(())

    def _op_EMPTY_LIST(self, arg) -> None:
        self._push([])

    def _op_EMPTY_DICT(self, arg) -> None:
        self._push({})

    def _op_EMPTY_SET(self, arg) -> None:
        self._push(set())

    def _op_MARK(self, arg) -> None:
        self._push(self._MARK)

    def _make_tuple(self, k: int) -> None:
        items = [self._pop() for _ in range(k)]
        items.reverse()
        self._push(tuple(items))
        self._maybe_set_shape(items)
        if items and all(isinstance(x, int) for x in items):
            self.ints = []

    def _op_TUPLE1(self, arg) -> None:
        self._make_tuple(1)

    def _op_TUPLE2(self, arg) -> None:
        self._make_tuple(2)

    def _op_TUPLE3(self, arg) -> None:
        self._make_tuple(3)

    def _op_TUPLE(self, arg) -> None:
        items = self._pop_to_mark()
        self._push(tuple(items))
        self._maybe_set_shape(items)

    def _op_LIST(self, arg) -> None:
        self._push(self._pop_to_mark())

    def _op_DICT(self, arg) -> None:
        items = self._pop_to_mark()
        d: dict[object, object] = {}
        for i in range(0, len(items), 2):
            d[items[i]] = items[i + 1]
        self._push(d)

    def _op_SETITEM(self, arg) -> None:
        v = self._pop()
        k = self._pop()
        d = self._pop()
        if isinstance(d, dict):
            d[k] = v
        self._push(d)

    def _op_APPEND(self, arg) -> None:
        v = self._pop()
        lst = self._pop()
        if isinstance(lst, list):
            lst.append(v)
        self._push(lst)

    def _op_SETITEMS(self, arg) -> None:
        pairs = self._pop_to_mark()
        d = self._pop()
        if isinstance(d, dict):
            for i in range(0, len(pairs), 2):
                d[pairs[i]] = pairs[i + 1]
        self._push(d)

    def _op_APPENDS(self, arg) -> None:
        vals = self._pop_to_mark()
        lst = self._pop()
        if isinstance(lst, list):
            lst.extend(vals)
        self._push(lst)

    def _op_ADDITEMS(self, arg) -> None:
        vals = self._pop_to_mark()
        s = self._pop()
        if isinstance(s, set):
            s.update(vals)
        self._push(s)

    def _op_STACK_GLOBAL(self, arg) -> None:
        name = self._pop()
        module = self._pop()
        if (
            module in ("numpy", "numpy.core.multiarray", "numpy._core.multiarray")
            and name == "dtype"
        ):
            self._push(_DtypeBuilder())
        else:
            self._push(_Global(str(module), str(name)))

    def _op_REDUCE(self, _) -> None:
        args = self._pop()
        func = self._pop()
        if isinstance(func, _DtypeBuilder) and isinstance(args, tuple) and args:
            raw = args[0]
            if isinstance(raw, bytes):
                try:
                    text = raw.decode("ascii")
                except UnicodeDecodeError:
                    text = None
            else:
                text = str(raw)
            if text is not None:
                d = _Descr(text)
                self._push(d)
                self._set_descr(d)
                return
        self._push(_Unknown("reduce"))

    def _op_BUILD(self, arg) -> None:
        self._pop()  # state
        self._pop()  # instance
        self._push(_Unknown("object"))

    def _op_NEWOBJ(self, arg) -> None:
        self._pop()  # args
        self._pop()  # cls
        self._push(_Unknown("newobj"))

    def _op_NEWOBJ_EX(self, arg) -> None:
        self._pop()  # kwargs
        self._pop()  # args
        self._pop()  # cls
        self._push(_Unknown("newobj"))

    def _binget(self, arg) -> None:
        idx = int(arg)
        v = self.memo.get(idx, _Unknown("missing"))
        self._push(v)
        self._set_descr(v)

    _op_GET = _binget
    _op_BINGET = _binget
    _op_LONG_BINGET = _binget

    def _memoize_arg(self, arg) -> None:
        idx = int(arg)
        self._memoize_top(idx)
        self.memo_next = max(self.memo_next, idx + 1)

    _op_PUT = _memoize_arg
    _op_BINPUT = _memoize_arg
    _op_LONG_BINPUT = _memoize_arg

    def _op_MEMOIZE(self, arg) -> None:
        self._memoize_top(self.memo_next)
        self.memo_next += 1

    def _op_POP(self, _) -> None:
        if self.stack:
            self._pop()

    def _op_POP_MARK(self, _) -> None:
        self._pop_to_mark()

    def _op_DUP(self, arg) -> None:
        if self.stack:
            self._push(self.stack[-1])

    def _push_unknown_persid(self, arg) -> None:
        self._push(_Unknown("persid"))

    _op_PERSID = _push_unknown_persid
    _op_BINPERSID = _push_unknown_persid

    def feed(self, opname: str, arg) -> None:
        handler = getattr(self, f"_op_{opname}", None)
        if handler is not None:
            handler(arg)

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
    """Return a spec if this opcode carries array data; otherwise record its facts.

    Small ``BINBYTES`` are ndarray reconstruct scaffolding (``b"b"``), not data.
    """
    header = _PAYLOAD_OPS.get(opname)
    if header is None:
        state.feed(opname, arg)
        return None
    nbytes = len(arg) if isinstance(arg, bytes) else int(arg)
    if nbytes < _MIN_PAYLOAD_BYTES:
        state.feed(opname, arg)
        return None
    return state.spec(name, base + pos + header, nbytes)


def _scan_window(
    window: bytes, base: int, name: str, state: _HeaderState
) -> ArraySpec | None:
    """Return the first ndarray payload in ``window``, or ``None`` if there is none.

    ``base`` is the window's absolute offset in the file; returned offsets are
    absolute. A truncated window (the payload lies past its end) still yields a
    complete spec, because the payload is intercepted by size before it is read.
    """
    stream = _StopAtPayload(window)
    try:
        for op, arg, pos in pickletools.genops(stream):
            found = _payload_at(state, op.name, arg, pos, base, name)
            if found is not None:
                return found
            if op.name == "STOP":
                return None
    except _PayloadBoundary as boundary:
        # tell() sits just past the length field, i.e. at the payload itself.
        return state.spec(name, base + boundary.offset, boundary.nbytes)
    return None


def scan_shard(path: Path) -> list[ArraySpec]:
    """Recover every ndarray in a pickle shard without unpickling it.

    Scans a bounded window, records the payload's offset and length, then jumps
    the window past the payload -- so a 1.6 GB expert tensor is never read into
    memory here. Raises on unknown dtypes or Fortran order so an unrecognized
    layout fails loudly instead of exporting garbage.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ExportError(f"{path}: cannot stat shard: {exc}") from exc
    if size == 0:
        raise ExportError(f"{path.name}: empty file, no ndarray payload found")

    specs: list[ArraySpec] = []
    state = _HeaderState()
    with open(path, "rb") as fh:
        mm = mmap_mod.mmap(fh.fileno(), 0, access=mmap_mod.ACCESS_READ)
        try:
            base = 0
            while base < size:
                try:
                    spec = _scan_window(
                        mm[base : min(size, base + _SCAN_WINDOW)], base, path.name, state
                    )
                except (ValueError, IndexError, AssertionError, struct.error, EOFError) as exc:
                    if specs:
                        raise ExportError(
                            f"{path.name}: corrupted pickle after {len(specs)} array(s): {exc}"
                        ) from exc
                    break
                if spec is None:
                    break
                _reject_unsupported(spec, size, path.name)
                specs.append(spec)
                base = spec.offset + spec.nbytes
        finally:
            mm.close()
    if not specs:
        raise ExportError(f"{path.name}: no ndarray payload found")
    return specs


def _reject_unsupported(spec: ArraySpec, size: int, name: str) -> None:
    """Fail on layouts this exporter cannot represent faithfully."""
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


def structural_stem(name: str) -> str:
    """``block_000.slot_04.attn_proj_i8.model_width`` -> filename stem.

    Rejects path separators, Windows drive letters, and parent-reference
    segments so a malformed conversion manifest cannot write outside
    ``--output-dir``.
    """
    if any(sep in name for sep in "/\\"):
        raise ExportError(f"structural name contains path separator: {name!r}")
    if ":" in name:
        raise ExportError(f"structural name contains colon (drive-relative path): {name!r}")
    parts = name.split(".")
    if any(part in ("", "..") for part in parts):
        raise ExportError(
            f"structural name has empty or parent-reference segment: {name!r}"
        )
    return name.replace(".", "__")


def _safe_out_path(output_dir: Path, name: str) -> Path:
    """Resolve the final output path and enforce it stays under ``output_dir``.

    The structural name is sanitized above, but a literal ``..`` or absolute
    prefix could still be introduced by platform-specific path handling.
    """
    out = (output_dir / f"{structural_stem(name)}.npy").resolve()
    base = output_dir.resolve()
    if base not in out.parents and out != base:
        raise ExportError(
            f"resolved output path {out} escapes --output-dir {base} for name {name!r}"
        )
    return out


def load_manifest(path: Path) -> list[dict]:
    with open(path, "rb") as fh:
        doc = json.load(fh)
    tensors = doc.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ExportError(f"{path}: no `tensors` array (not an xai-dissect conversion manifest?)")
    return tensors


def select_tensors(
    tensors: list[dict],
    *,
    block: int | None,
    mode: str,
    names: list[str],
) -> list[dict]:
    """Pick explicit structural names, or every tensor of one block in ``mode``."""
    if names:
        return _select_by_names(tensors, names)
    return _select_by_block(tensors, block, mode)


def _select_by_names(tensors: list[dict], names: list[str]) -> list[dict]:
    by_name = {t["structural_name"]: t for t in tensors}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ExportError(f"structural names not in manifest: {', '.join(missing)}")
    return [by_name[n] for n in names]


def _select_by_block(tensors: list[dict], block: int | None, mode: str) -> list[dict]:
    kinds = MODES[mode]
    picked = [t for t in tensors if t.get("block") == block and t.get("kind") in kinds]
    if not picked:
        raise ExportError(f"no tensors for block {block} in mode {mode}")
    return picked


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dequantize Grok-1 int8 ckpt-0 shards to f32 .npy (GH #53 / RM-222)"
    )
    p.add_argument(
        "--conversion-manifest", type=Path, help="xai-dissect run3 conversion-manifest.json"
    )
    p.add_argument("--block", type=int, help="pilot block index (with --mode)")
    p.add_argument("--mode", choices=sorted(MODES), default="attention_only")
    p.add_argument(
        "--structural-name",
        action="append",
        default=[],
        dest="names",
        help="export one tensor by structural name (repeatable; overrides --block/--mode)",
    )
    p.add_argument("--output-dir", type=Path, help="destination directory for .npy files")
    p.add_argument("--chunk-mib", type=int, default=DEFAULT_CHUNK_MIB, help="write chunk size")
    p.add_argument("--dry-run", action="store_true", help="report plan and sizes, write nothing")
    p.add_argument("--inspect", type=Path, help="print raw array layout of one shard and exit")
    return p


def run_exports(picked: list[dict], output_dir: Path, chunk_mib: int, dry_run: bool) -> int:
    """Export each selected tensor; return total f32 bytes planned or written."""
    total = 0
    for t in sorted(picked, key=lambda x: x["structural_name"]):
        name = t["structural_name"]
        shard = Path(t["source_shard_path"])
        if not shard.exists():
            raise ExportError(f"{name}: shard missing: {shard}")
        out = _safe_out_path(output_dir, name)
        # expect_shape cross-checks the manifest against the shard before writing.
        info = export_tensor(
            shard, out, chunk_mib=chunk_mib, dry_run=dry_run, expect_shape=t["shape"]
        )
        total += info["out_bytes"]
        groups = info["scale_groups"]
        detail = f"groups={groups}" if groups is not None else "passthrough"
        print(
            f"{'plan' if dry_run else 'wrote'} {name}  {info['source_dtype']:>13}  "
            f"{tuple(info['shape'])}  {detail}  {info['out_bytes'] / 1024**3:.3f} GiB "
            f"-> {out.name}"
        )
    return total


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.inspect:
        for spec in scan_shard(args.inspect):
            print(
                f"{spec.descr:>9} {spec.shape!s:>24}  "
                f"offset={spec.offset}  bytes={spec.nbytes}"
            )
        return 0

    if not args.conversion_manifest:
        parser.error("--conversion-manifest is required (or use --inspect)")
    if not args.output_dir:
        parser.error("--output-dir is required")
    if args.block is None and not args.names:
        parser.error("need --block (with --mode) or at least one --structural-name")
    if args.chunk_mib <= 0:
        parser.error("--chunk-mib must be a positive integer")

    picked = select_tensors(
        load_manifest(args.conversion_manifest),
        block=args.block,
        mode=args.mode,
        names=args.names,
    )
    total = run_exports(picked, Path(args.output_dir), args.chunk_mib, args.dry_run)
    print(f"{len(picked)} tensor(s), {total / 1024**3:.3f} GiB f32 total -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
