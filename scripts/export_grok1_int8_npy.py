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

Dependencies
------------
Unlike ``export_grok1_embedding_npy.py`` (stdlib-only, byte-copy), dequantization
is real arithmetic over up to 1.6e9 elements, so this script **requires numpy**
(already a CI dependency of ``.github/workflows/python-scripts.yml``). Output is
written in bounded chunks; peak RSS stays near ``--chunk-mib`` regardless of
tensor size.

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
import json
import mmap as mmap_mod
import pickletools
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CHUNK_MIB = 256

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


def scan_shard(path: Path) -> list[ArraySpec]:
    """Recover every ndarray in a pickle shard without unpickling it.

    Walks the opcode stream and pairs the most recent shape tuple + dtype
    descriptor with each large ``BINBYTES*`` payload. Raises on unknown dtypes so
    an unrecognized layout fails loudly instead of exporting garbage.
    """
    specs: list[ArraySpec] = []
    with open(path, "rb") as fh:
        mm = mmap_mod.mmap(fh.fileno(), 0, access=mmap_mod.ACCESS_READ)
        try:
            ints: list[int] = []
            shape: tuple[int, ...] | None = None
            descr: str | None = None
            for op, arg, pos in pickletools.genops(mm):
                name = op.name
                if name in ("BININT", "BININT1", "BININT2"):
                    ints.append(int(arg))
                elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
                    k = int(name[-1])
                    if len(ints) >= k:
                        shape = tuple(ints[-k:])
                    ints = []
                elif name == "SHORT_BINUNICODE" and arg in _KNOWN_DESCR:
                    descr = str(arg)
                elif name in ("BINBYTES", "BINBYTES8", "SHORT_BINBYTES"):
                    nbytes = len(arg) if isinstance(arg, bytes) else int(arg)
                    # Small BINBYTES carry ndarray reconstruct scaffolding (b"b"),
                    # not payloads; a real payload is always >= 16 bytes here.
                    if nbytes < 16:
                        continue
                    if shape is None or descr is None:
                        raise ExportError(
                            f"{path.name}: payload at byte {pos} has no preceding "
                            "shape/dtype -- unrecognized pickle layout"
                        )
                    header = {"BINBYTES": 5, "BINBYTES8": 9, "SHORT_BINBYTES": 2}[name]
                    spec = ArraySpec(shape, descr, pos + header, nbytes)
                    spec.validate()
                    specs.append(spec)
                elif name == "STOP":
                    break
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
) -> dict:
    """Dequantize (or pass through) one shard into an f32 ``.npy``.

    Returns a summary dict; with ``dry_run`` nothing is written.
    """
    import numpy as np

    specs = scan_shard(shard)
    weight, scales = split_quantized(specs)
    info: dict = {
        "shard": str(shard),
        "output": str(out_path),
        "shape": list(weight.shape),
        "source_dtype": "f32" if scales is None else "int8 x bf16",
        "out_bytes": weight.numel * 4,
    }

    if scales is None:
        info["scale_groups"] = None
        if dry_run:
            return info
        src = np.memmap(
            shard, dtype="<f4", mode="r", offset=weight.offset, shape=weight.shape
        )
        _write_npy(out_path, weight.shape, src, chunk_mib=chunk_mib)
        return info

    lead, k, n, g = grouping(weight, scales)
    info["scale_groups"] = g
    info["group_rows"] = k // g
    if dry_run:
        return info

    w = np.memmap(shard, dtype=np.int8, mode="r", offset=weight.offset, shape=weight.shape)
    s_raw = np.memmap(
        shard, dtype="<u2", mode="r", offset=scales.offset, shape=scales.shape
    )
    # bfloat16 -> f32 is an exact 16-bit left shift of the bit pattern.
    s = (s_raw.astype(np.uint32) << 16).view(np.float32)

    lead_n = 1
    for d in lead:
        lead_n *= d
    wf = w.reshape(lead_n, k, n)
    sf = s.reshape(lead_n, g, n)
    group_rows = k // g

    rows_per_chunk = max(1, (chunk_mib * 1024 * 1024) // (n * 4))

    def blocks():
        for li in range(lead_n):
            for gi in range(g):
                base = gi * group_rows
                scale_row = sf[li, gi]
                for r0 in range(0, group_rows, rows_per_chunk):
                    r1 = min(r0 + rows_per_chunk, group_rows)
                    chunk = wf[li, base + r0 : base + r1].astype(np.float32)
                    chunk *= scale_row
                    yield chunk

    _write_npy(out_path, weight.shape, blocks(), chunk_mib=chunk_mib, streaming=True)
    return info


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
                for chunk in source:
                    fh.write(np.ascontiguousarray(chunk, dtype="<f4").tobytes())
            else:
                rows = max(1, (chunk_mib * 1024 * 1024) // max(1, source[0].nbytes))
                for r0 in range(0, len(source), rows):
                    fh.write(
                        np.ascontiguousarray(source[r0 : r0 + rows], dtype="<f4").tobytes()
                    )
        tmp.replace(out_path)
    finally:
        tmp.unlink(missing_ok=True)


def structural_stem(name: str) -> str:
    """``block_000.slot_04.attn_proj_i8.model_width`` -> filename stem."""
    return name.replace(".", "__")


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
    if names:
        by_name = {t["structural_name"]: t for t in tensors}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise ExportError(f"structural names not in manifest: {', '.join(missing)}")
        return [by_name[n] for n in names]
    kinds = MODES[mode]
    picked = [
        t for t in tensors if t.get("block") == block and t.get("kind") in kinds
    ]
    if not picked:
        raise ExportError(f"no tensors for block {block} in mode {mode}")
    return picked


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Dequantize Grok-1 int8 ckpt-0 shards to f32 .npy (GH #53 / RM-222)"
    )
    p.add_argument("--conversion-manifest", type=Path, help="xai-dissect run3 conversion-manifest.json")
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
    args = p.parse_args(argv)

    if args.inspect:
        for spec in scan_shard(args.inspect):
            print(f"{spec.descr:>9} {str(spec.shape):>24}  offset={spec.offset}  bytes={spec.nbytes}")
        return 0

    if not args.conversion_manifest:
        p.error("--conversion-manifest is required (or use --inspect)")
    if not args.output_dir:
        p.error("--output-dir is required")
    if args.block is None and not args.names:
        p.error("need --block (with --mode) or at least one --structural-name")

    tensors = load_manifest(args.conversion_manifest)
    picked = select_tensors(tensors, block=args.block, mode=args.mode, names=args.names)

    total = 0
    for t in sorted(picked, key=lambda x: x["structural_name"]):
        name = t["structural_name"]
        shard = Path(t["source_shard_path"])
        if not shard.exists():
            raise ExportError(f"{name}: shard missing: {shard}")
        out = Path(args.output_dir) / f"{structural_stem(name)}.npy"
        info = export_tensor(shard, out, chunk_mib=args.chunk_mib, dry_run=args.dry_run)
        # Cross-check the manifest against what we actually parsed out of the shard.
        if list(t["shape"]) != info["shape"]:
            raise ExportError(
                f"{name}: manifest shape {t['shape']} != shard shape {info['shape']}"
            )
        total += info["out_bytes"]
        groups = info["scale_groups"]
        detail = f"groups={groups}" if groups is not None else "passthrough"
        verb = "plan" if args.dry_run else "wrote"
        print(
            f"{verb} {name}  {info['source_dtype']:>13}  {tuple(info['shape'])}  "
            f"{detail}  {info['out_bytes'] / 1024**3:.3f} GiB  -> {out.name}"
        )
    print(f"{len(picked)} tensor(s), {total / 1024**3:.3f} GiB f32 total -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
