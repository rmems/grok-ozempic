#!/usr/bin/env python3
"""Pickle-frame ndarray scanner for Grok-1 int8 export (never unpickles).

Walks opcode streams with pickletools.genops only. Includes a minimal value
stack/memo so BINGET-referenced dtypes survive payload jumps (PR #57 review).
"""
from __future__ import annotations

import io
import mmap as mmap_mod

# pickletools only *decodes* opcodes; nothing here unpickles.
import pickletools  # nosec B403
import struct
from dataclasses import dataclass
from pathlib import Path

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


