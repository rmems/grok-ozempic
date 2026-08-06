#!/usr/bin/env python3
"""Weight sources for the Grok-1 block-0 forward experiment (GH #61 / RM-249).

Three interchangeable views of one block's weights:

* :class:`NpyWeights`     -- the dequantized f32 reference (ground truth)
* :class:`F16Weights`     -- reference round-tripped through fp16 (control)
* :class:`PackWeights`    -- reconstructed from a GOZ1 pack

Memory is the binding constraint: block 0's three expert tensors are 6.44 GiB
each as f32, so nothing here materializes a whole expert tensor. Callers ask for
one expert at a time and the reference path hands back a ``memmap`` slice.

**Ternary reconstruction uses an oracle scale.** A GOZ1 pack stores only trits:
``quantize_f32`` computes ``rms`` and ``threshold`` per tensor but the container
persists neither, so a pack cannot be dequantized from its own contents. This
module therefore derives the least-squares optimal ``alpha`` from the *original*
weights, which is the most favourable scale any runtime could pick. Reported
degradation is consequently a lower bound on the damage, never an artefact of a
badly chosen scale.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grok1_block_forward import ForwardError, resolve_roles  # noqa: E402
from route_preservation_io import (  # noqa: E402
    TENSOR_F16,
    TENSOR_TERNARY,
    MetricsError,
    load_pack_index,
    read_f16,
    read_trits,
)

__all__ = [
    "ALPHA_CACHE_SUFFIX",
    "ATTENTION_ROLES",
    "EXPERT_ROLES",
    "F16Weights",
    "MixedWeights",
    "NpyWeights",
    "PackWeights",
    "TernaryScale",
    "alpha_for",
    "stem_of",
]

# The two ternary tiers, for attributing damage to one or the other.
ATTENTION_ROLES = frozenset({"query", "key", "value", "attn_out"})
EXPERT_ROLES = frozenset({"expert_gelu", "expert_value", "expert_down"})

ALPHA_CACHE_SUFFIX = ".oracle-alpha.json"
# 64 Mi elements per streaming step: 256 MiB of f32 plus 64 MiB of trits.
_ALPHA_CHUNK = 1 << 26


def stem_of(structural_name: str) -> str:
    """``block_000.slot_11.router`` -> ``block_000__slot_11__router`` (npy stem)."""
    return structural_name.replace(".", "__")


@dataclass(frozen=True)
class TernaryScale:
    """Least-squares optimal single scale for a ternary tensor."""

    alpha: float
    fired: int
    total: int

    @property
    def sparsity(self) -> float:
        return 1.0 - (self.fired / self.total) if self.total else 0.0


def _accumulate_alpha(flat_npy: np.ndarray, pack: Path, entry: dict) -> tuple[float, int]:
    """Stream ``(sum|w| over fired, fired count)`` for one ternary tensor."""
    total = int(entry["numel"])
    s1f, fired = 0.0, 0
    for start in range(0, total, _ALPHA_CHUNK):
        count = min(_ALPHA_CHUNK, total - start)
        trits = read_trits(pack, entry, start, count)
        block = np.asarray(flat_npy[start : start + count], dtype=np.float32)
        mask = trits != 0
        s1f += float(np.abs(block[mask], dtype=np.float32).sum(dtype=np.float64))
        fired += int(mask.sum())
    return s1f, fired


def alpha_for(npy_path: Path, pack: Path, entry: dict) -> TernaryScale:
    """Least-squares optimal ``alpha = sum(|w| fired) / count(fired)``.

    Minimizes ``||w - alpha*t||^2`` for the pack's own trit pattern, so it is the
    best scale a runtime could possibly use for this tensor.
    """
    flat = np.load(npy_path, mmap_mode="r").reshape(-1)
    if flat.size != int(entry["numel"]):
        raise ForwardError(
            f"{entry['name']}: npy has {flat.size} elements, pack expects {entry['numel']}"
        )
    s1f, fired = _accumulate_alpha(flat, pack, entry)
    return TernaryScale(
        alpha=(s1f / fired) if fired else 0.0, fired=fired, total=int(entry["numel"])
    )


class NpyWeights:
    """The dequantized f32 reference block, read via ``memmap``."""

    label = "fp32_reference"

    def __init__(self, npy_dir: Path, names: list[str]) -> None:
        self._paths = {n: npy_dir / f"{stem_of(n)}.npy" for n in names}
        missing = sorted(n for n, p in self._paths.items() if not p.exists())
        if missing:
            raise ForwardError(f"{npy_dir}: missing exported npy for {missing}")
        self._shapes = {
            n: tuple(int(d) for d in np.load(p, mmap_mode="r").shape)
            for n, p in self._paths.items()
        }
        self.roles = resolve_roles(self._shapes)

    def shapes(self) -> dict[str, tuple[int, ...]]:
        return dict(self._shapes)

    def _array(self, role: str) -> np.ndarray:
        return np.load(self._paths[self.roles[role]], mmap_mode="r")

    def vector(self, role: str) -> np.ndarray:
        """A norm gain or router matrix, small enough to load fully."""
        return np.asarray(self._array(role), dtype=np.float32)

    def matrix(self, role: str) -> np.ndarray:
        """An attention projection (at most 151 MiB)."""
        return np.asarray(self._array(role), dtype=np.float32)

    def expert(self, role: str, index: int) -> np.ndarray:
        """One expert's slice, left as a ``memmap`` view to bound memory."""
        return self._array(role)[index]


class F16Weights(NpyWeights):
    """Reference weights round-tripped through fp16.

    The control baseline: if the harness itself introduced meaningful routing
    drift, this would show it, because fp16 preserves ~3 decimal digits and must
    not move top-1/top-2 selection appreciably.
    """

    label = "fp16_roundtrip"

    @staticmethod
    def _cast(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=np.float16).astype(np.float32)

    def vector(self, role: str) -> np.ndarray:
        return self._cast(super().vector(role))

    def matrix(self, role: str) -> np.ndarray:
        return self._cast(super().matrix(role))

    def expert(self, role: str, index: int) -> np.ndarray:
        return self._cast(super().expert(role, index))


class PackWeights:
    """A block reconstructed from a GOZ1 pack, using oracle ternary scales."""

    label = "goz1_pack"

    def __init__(
        self,
        pack: Path,
        npy_dir: Path,
        *,
        alpha_cache: Path | None = None,
        partial: bool = False,
    ) -> None:
        self.pack = pack
        self._npy_dir = npy_dir
        _metadata, self._index = load_pack_index(pack)
        self.metadata = _metadata
        self._shapes = {n: tuple(int(d) for d in e["shape"]) for n, e in self._index.items()}
        # A tier-limited pack (attention only, say) is valid when the caller
        # supplies the remaining roles from a reference source via MixedWeights.
        self.roles = resolve_roles(self._shapes, require_complete=not partial)
        self._cache_path = alpha_cache or pack.with_suffix(pack.suffix + ALPHA_CACHE_SUFFIX)
        self._scales: dict[str, TernaryScale] = self._load_cache()

    def shapes(self) -> dict[str, tuple[int, ...]]:
        return dict(self._shapes)

    def _fingerprint(self) -> dict:
        """Identify the exact inputs an oracle alpha was derived from.

        The cache path defaults to the pack filename, so rebuilding a pack at the
        same path -- exactly what a tau sweep does -- would otherwise silently
        reuse a scale computed for different trits. Alpha also depends on the
        reference npy directory, since that supplies the magnitudes.
        """
        stat = self.pack.stat()
        return {
            "pack_size": stat.st_size,
            "pack_mtime_ns": stat.st_mtime_ns,
            "npy_dir": str(self._npy_dir.resolve()),
            "tensors": sorted(self._index),
        }

    def _load_cache(self) -> dict[str, TernaryScale]:
        """Load cached scales, discarding them if the inputs have changed."""
        if not self._cache_path.exists():
            return {}
        try:
            raw = json.loads(self._cache_path.read_text())
        except json.JSONDecodeError:
            return {}
        # Legacy flat caches carry no fingerprint and cannot be validated.
        if raw.get("fingerprint") != self._fingerprint():
            return {}
        return {n: TernaryScale(**v) for n, v in raw.get("scales", {}).items()}

    def _save_cache(self) -> None:
        payload = {
            "fingerprint": self._fingerprint(),
            "scales": {n: vars(s) for n, s in sorted(self._scales.items())},
        }
        self._cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def scale(self, name: str) -> TernaryScale:
        """Oracle scale for a ternary tensor, computed once and cached on disk."""
        if name not in self._scales:
            npy = self._npy_dir / f"{stem_of(name)}.npy"
            if not npy.exists():
                raise ForwardError(
                    f"{name}: oracle alpha needs the reference npy at {npy}; a GOZ1 pack "
                    "stores no per-tensor scale, so it cannot be dequantized alone"
                )
            self._scales[name] = alpha_for(npy, self.pack, self._index[name])
            self._save_cache()
        return self._scales[name]

    def scales(self) -> dict[str, TernaryScale]:
        return dict(self._scales)

    def _entry(self, role: str) -> dict:
        return self._index[self.roles[role]]

    def _read(self, role: str, start: int, count: int, shape: tuple[int, ...]) -> np.ndarray:
        """Reconstruct ``alpha * trits`` (ternary) or read fp16, for one slice."""
        entry = self._entry(role)
        kind = int(entry["tensor_type"])
        if kind == TENSOR_F16:
            # Slice by flat range so a preserve-tier expert stays correct rather
            # than silently reshaping the whole tensor into one expert's shape.
            return read_f16(self.pack, entry).reshape(-1)[start : start + count].reshape(shape)
        if kind != TENSOR_TERNARY:
            raise MetricsError(f"{entry['name']}: unsupported tensor_type {kind}")
        alpha = np.float32(self.scale(entry["name"]).alpha)
        trits = read_trits(self.pack, entry, start, count)
        return (trits.astype(np.float32) * alpha).reshape(shape)

    def vector(self, role: str) -> np.ndarray:
        entry = self._entry(role)
        shape = tuple(int(d) for d in entry["shape"])
        return self._read(role, 0, int(entry["numel"]), shape)

    def matrix(self, role: str) -> np.ndarray:
        return self.vector(role)

    def expert(self, role: str, index: int) -> np.ndarray:
        """Decode a single expert slice; never the whole 6.44 GiB tensor."""
        entry = self._entry(role)
        lead, rows, cols = (int(d) for d in entry["shape"])
        if not 0 <= index < lead:
            raise ForwardError(f"{entry['name']}: expert {index} out of range 0..{lead - 1}")
        per = rows * cols
        return self._read(role, index * per, per, (rows, cols))


class MixedWeights:
    """Draw some roles from one source and the rest from another.

    Used to attribute damage: routing within a single block is computed from
    ``rmsnorm(h0 + attn_out)``, which is entirely upstream of the experts, so a
    pack whose experts alone are ternary must reproduce the reference routing
    exactly. Running that as a baseline turns the argument into a measurement.
    """

    def __init__(self, primary, fallback, roles: frozenset[str], label: str) -> None:
        shared = set(primary.roles) & set(fallback.roles)
        disagree = sorted(r for r in shared if primary.roles[r] != fallback.roles[r])
        if disagree:
            raise ForwardError(f"mixed sources disagree on slot/role mapping for {disagree}")
        uncovered = sorted(roles - set(primary.roles))
        if uncovered:
            raise ForwardError(f"primary source lacks assigned roles {uncovered}")
        unresolved = sorted(set(fallback.roles) - set(primary.roles) - (set(fallback.roles) - roles))
        if unresolved:
            raise ForwardError(f"no source for roles {unresolved}")
        self._primary = primary
        self._fallback = fallback
        self._roles = roles
        self.label = label
        self.roles = {**primary.roles, **fallback.roles}

    def _pick(self, role: str):
        return self._primary if role in self._roles else self._fallback

    def vector(self, role: str) -> np.ndarray:
        return self._pick(role).vector(role)

    def matrix(self, role: str) -> np.ndarray:
        return self._pick(role).matrix(role)

    def expert(self, role: str, index: int) -> np.ndarray:
        return self._pick(role).expert(role, index)
