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

import hashlib
import json
import shutil
import subprocess  # nosec B404
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grok1_block_forward import ForwardError, resolve_roles  # noqa: E402
from route_preservation_io import (  # noqa: E402
    TENSOR_F16,
    TENSOR_TERNARY,
    MetricsError,
    load_pack_index,
    read_f16_slice,
    read_trits,
)

__all__ = [
    "ALPHA_CACHE_SUFFIX",
    "WeightSource",
    "PRESERVED_ROLES",
    "ATTENTION_ROLES",
    "EXPERT_ROLES",
    "F16Weights",
    "MixedWeights",
    "NpyWeights",
    "PackWeights",
    "TernaryScale",
    "alpha_for",
    "implementation_commit",
    "sha256_file",
    "stem_of",
]

# The two ternary tiers, for attributing damage to one or the other.
ATTENTION_ROLES = frozenset({"query", "key", "value", "attn_out"})
# Must never be quantized: the router and all four block norms.
PRESERVED_ROLES = frozenset(
    {"router", "norm_pre_attn", "norm_post_attn", "norm_pre_moe", "norm_post_moe"}
)
EXPERT_ROLES = frozenset({"expert_gelu", "expert_value", "expert_down"})

ALPHA_CACHE_SUFFIX = ".oracle-alpha.json"
# 64 Mi elements per streaming step: 256 MiB of f32 plus 64 MiB of trits.
_ALPHA_CHUNK = 1 << 26


class WeightSource(Protocol):
    """One block's weights, however they are backed.

    :class:`NpyWeights`, :class:`F16Weights`, :class:`PackWeights` and
    :class:`MixedWeights` are structurally interchangeable; declaring the shape
    explicitly is what lets a reader (or a type checker) see that
    ``forward_block`` works against any of them, rather than inferring it from
    call sites.
    """

    label: str
    roles: dict[str, str]

    # The concrete classes below list this Protocol as a base explicitly. That is
    # not required for structural typing, but it states the contract at the class
    # and gives checkers/IDEs a nominal edge to follow rather than re-deriving the
    # shape from every call site.

    def vector(self, role: str) -> np.ndarray:
        """A norm gain or the router matrix."""
        ...

    def matrix(self, role: str) -> np.ndarray:
        """An attention projection."""
        ...

    def expert(self, role: str, index: int) -> np.ndarray:
        """One expert's slice of a stacked expert tensor."""
        ...


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    """Streamed SHA-256 of a file, for identifying exact bytes in provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_commit(repo_root: Path | None = None) -> dict[str, str | bool | None]:
    """Record the commit these results were produced by, per RM-249 provenance.

    Returns ``{"commit": <sha or None>, "dirty": <bool or None>}``, where
    ``dirty`` covers the *implementation* (``scripts/``, ``src/``) and not the
    report tree. Writing an artifact necessarily modifies ``reports/``, so a
    whole-tree check would report every run as dirty and the flag would carry no
    information about whether the producing code was modified.

    Never raises: provenance is best-effort, and a run outside a git checkout
    must still write its metrics rather than abort.
    """
    root = repo_root or Path(__file__).resolve().parent.parent
    git = shutil.which("git")
    if git is None:
        return {"commit": None, "dirty": None}
    # Both argv lists are fully literal apart from ``root``, which derives from
    # ``__file__`` and never from user input, and ``git`` is resolved to an
    # absolute path above. No shell is involved (shell=False is the default).
    # Every scanner flags any subprocess call, so each annotation below names one
    # tool's own rule: nosec for Bandit, noqa for ruff, nosemgrep for Semgrep.
    try:
        # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit.dangerous-subprocess-use-audit
        sha = subprocess.run(  # nosec B603  # noqa: S603
            [git, "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
        # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit.dangerous-subprocess-use-audit
        status = subprocess.run(  # nosec B603  # noqa: S603
            [
                git, "-C", str(root), "status", "--porcelain",
                "--untracked-files=no", "--", "scripts", "src",
            ],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": sha, "dirty": bool(status)}


def stem_of(structural_name: str) -> str:
    """``block_000.slot_11.router`` -> ``block_000__slot_11__router`` (npy stem)."""
    return structural_name.replace(".", "__")


@dataclass(frozen=True)
class TernaryScale:
    """Least-squares optimal single scale for a ternary tensor."""

    alpha: float
    fired: int
    total: int
    # Fired positions where sign(trit) != sign(w). Zero for a correctly built
    # pack, since the quantizer assigns +1 above +tau and -1 below -tau.
    sign_mismatches: int = 0

    @property
    def sparsity(self) -> float:
        return 1.0 - (self.fired / self.total) if self.total else 0.0


def _accumulate_alpha(flat_npy: np.ndarray, pack: Path, entry: dict) -> tuple[float, int, int]:
    """Stream ``(sum(w*t) over fired, fired count, sign mismatches)``.

    The signed product is the actual least-squares numerator (see
    :func:`alpha_for`); summing ``|w|`` instead would silently paper over a
    sign disagreement between the pack and the reference weights.
    """
    total = int(entry["numel"])
    swt, fired, mismatched = 0.0, 0, 0
    for start in range(0, total, _ALPHA_CHUNK):
        count = min(_ALPHA_CHUNK, total - start)
        trits = read_trits(pack, entry, start, count)
        block = np.asarray(flat_npy[start : start + count], dtype=np.float32)
        mask = trits != 0
        products = block[mask] * trits[mask].astype(np.float32)
        swt += float(products.sum(dtype=np.float64))
        fired += int(mask.sum())
        mismatched += int((products < 0).sum())
    return swt, fired, mismatched


def alpha_for(npy_path: Path, pack: Path, entry: dict) -> TernaryScale:
    """Least-squares optimal ``alpha = sum(w*t) / count(fired)``.

    Minimizes ``||w - alpha*t||^2`` over ``alpha`` for the pack's own trit
    pattern (``sum(t^2) == count(fired)`` because every trit is 0 or +/-1), so it
    is the best scale a runtime could possibly use for this tensor.

    Equals ``sum(|w|)/count(fired)`` exactly when ``sign(t) == sign(w)``
    everywhere, which holds for a correctly built pack. Using the signed product
    means a pack whose trits disagree in sign yields a *smaller* alpha and a
    nonzero ``sign_mismatches`` instead of an inflated scale that would quietly
    falsify this module's "best case" claim.
    """
    flat = np.load(npy_path, mmap_mode="r").reshape(-1)
    if flat.size != int(entry["numel"]):
        raise ForwardError(
            f"{entry['name']}: npy has {flat.size} elements, pack expects {entry['numel']}"
        )
    swt, fired, mismatched = _accumulate_alpha(flat, pack, entry)
    return TernaryScale(
        alpha=(swt / fired) if fired else 0.0,
        fired=fired,
        total=int(entry["numel"]),
        sign_mismatches=mismatched,
    )


class NpyWeights(WeightSource):
    """The dequantized f32 reference block, read via ``memmap``."""

    label = "fp32_reference"

    def __init__(
        self, npy_dir: Path, names: list[str], *, expect_block: str | None = None
    ) -> None:
        self._paths = {n: npy_dir / f"{stem_of(n)}.npy" for n in names}
        missing = sorted(n for n, p in self._paths.items() if not p.exists())
        if missing:
            raise ForwardError(f"{npy_dir}: missing exported npy for {missing}")
        self._shapes = {
            n: tuple(int(d) for d in np.load(p, mmap_mode="r").shape)
            for n, p in self._paths.items()
        }
        self.roles = resolve_roles(self._shapes, expect_block=expect_block)

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


class PackWeights(WeightSource):
    """A block reconstructed from a GOZ1 pack, using oracle ternary scales."""

    label = "goz1_pack"

    def __init__(
        self,
        pack: Path,
        npy_dir: Path,
        *,
        alpha_cache: Path | None = None,
        partial: bool = False,
        expect_block: str | None = None,
    ) -> None:
        self.pack = pack
        self._npy_dir = npy_dir
        _metadata, self._index = load_pack_index(pack)
        self.metadata = _metadata
        self._shapes = {n: tuple(int(d) for d in e["shape"]) for n, e in self._index.items()}
        # A tier-limited pack (attention only, say) is valid when the caller
        # supplies the remaining roles from a reference source via MixedWeights.
        self.roles = resolve_roles(
            self._shapes, require_complete=not partial, expect_block=expect_block
        )
        self._cache_path = alpha_cache or pack.with_suffix(pack.suffix + ALPHA_CACHE_SUFFIX)
        self._fp: dict | None = None
        self._scales: dict[str, TernaryScale] = self._load_cache()

    def shapes(self) -> dict[str, tuple[int, ...]]:
        return dict(self._shapes)

    def _fingerprint(self) -> dict:
        """Identify the exact inputs an oracle alpha was derived from.

        The cache path defaults to the pack filename, so rebuilding a pack at the
        same path -- exactly what a tau sweep does -- would otherwise silently
        reuse a scale computed for different trits. Alpha also depends on the
        reference npy directory, since that supplies the magnitudes.

        The pack is identified by content hash rather than mtime: a rebuild that
        happens to preserve size and timestamp, or a restored-from-backup file,
        would defeat a stat-only discriminator. Computed once per instance.
        """
        if self._fp is None:
            self._fp = self._compute_fingerprint()
        return self._fp

    def _compute_fingerprint(self) -> dict:
        stat = self.pack.stat()
        return {
            "pack_size": stat.st_size,
            "pack_sha256": sha256_file(self.pack),
            "npy_dir": str(self._npy_dir.resolve()),
            "tensors": sorted(self._index),
            # alpha is derived from the reference magnitudes, so re-exporting an
            # npy in place must invalidate the cache too -- the pack is unchanged
            # in that case, so pack stat alone would happily reuse a stale scale.
            "reference_npy": self._reference_stats(),
        }

    def _reference_stats(self) -> dict[str, list[int]]:
        """``{stem: [size, mtime_ns]}`` for each reference npy backing a tensor."""
        stats: dict[str, list[int]] = {}
        for name in sorted(self._index):
            npy = self._npy_dir / f"{stem_of(name)}.npy"
            if npy.exists():
                st = npy.stat()
                stats[npy.name] = [st.st_size, st.st_mtime_ns]
        return stats

    def _load_cache(self) -> dict[str, TernaryScale]:
        """Load cached scales, discarding them if the inputs have changed."""
        if not self._cache_path.exists():
            return {}
        try:
            raw = json.loads(self._cache_path.read_text())
        except json.JSONDecodeError as exc:
            # A cache is derived data, so recomputing is the right recovery --
            # but do it loudly, since a corrupt file may point at a real problem.
            print(
                f"warning: {self._cache_path.name} is unreadable ({exc}); "
                "recomputing oracle alphas",
                file=sys.stderr,
            )
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

    def tensor_type(self, role: str) -> int:
        """GOZ1 tensor type for a role (``TENSOR_F16`` or ``TENSOR_TERNARY``)."""
        return int(self._entry(role)["tensor_type"])

    def require_preserved(self, roles: Iterable[str]) -> None:
        """Fail closed unless every named role is stored at preserve precision.

        The entire experiment rests on the router and all four norms being
        preserved: if a pack ternarized one of them, the resulting drift would be
        misattributed to the attention or expert tier. Asserting it against the
        pack itself is cheap and turns a silent misattribution into an error.
        """
        quantized = sorted(
            f"{role} ({self.roles[role]})"
            for role in roles
            if role in self.roles and self.tensor_type(role) != TENSOR_F16
        )
        if quantized:
            raise ForwardError(
                f"{self.pack.name}: these must be preserved but are quantized in the "
                f"pack: {quantized}. Routing drift would be misattributed."
            )

    def _read(self, role: str, start: int, count: int, shape: tuple[int, ...]) -> np.ndarray:
        """Reconstruct ``alpha * trits`` (ternary) or read fp16, for one slice."""
        entry = self._entry(role)
        kind = int(entry["tensor_type"])
        if kind == TENSOR_F16:
            # Seek to the slice rather than decoding the whole tensor: a
            # preserve-tier expert would otherwise cost 3.2 GiB to read one of
            # eight slices.
            return read_f16_slice(self.pack, entry, start, count).reshape(shape)
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


def _require_agreeing_mapping(primary: WeightSource, fallback: WeightSource) -> None:
    """Both sources must agree on what each shared role's tensor is called."""
    shared = set(primary.roles) & set(fallback.roles)
    disagree = sorted(r for r in shared if primary.roles[r] != fallback.roles[r])
    if disagree:
        raise ForwardError(f"mixed sources disagree on slot/role mapping for {disagree}")


def _require_primary_covers(primary: WeightSource, roles: frozenset[str]) -> None:
    """The primary must supply every role it has been assigned."""
    uncovered = sorted(roles - set(primary.roles))
    if uncovered:
        raise ForwardError(f"primary source lacks assigned roles {uncovered}")


def _require_all_served(
    primary: WeightSource, fallback: WeightSource, roles: frozenset[str]
) -> None:
    """Every claimed role must be served by whichever source ``_pick`` routes to.

    A role claimed by the primary but *outside* ``roles`` goes to the fallback, so
    if the fallback lacks it nothing serves it -- previously that surfaced as a
    bare KeyError deep inside forward_block, after the expensive reference pass
    had already run.
    """
    unserved = sorted(
        role
        for role in set(primary.roles) | set(fallback.roles)
        if role not in (primary.roles if role in roles else fallback.roles)
    )
    if unserved:
        raise ForwardError(f"no source serves roles {unserved}")


def _validate_mix(primary: WeightSource, fallback: WeightSource, roles: frozenset[str]) -> None:
    """Reject a mix that would fail later, or silently measure the wrong tensor."""
    _require_agreeing_mapping(primary, fallback)
    _require_primary_covers(primary, roles)
    _require_all_served(primary, fallback, roles)


class MixedWeights(WeightSource):
    """Draw some roles from one source and the rest from another.

    Used to attribute damage: routing within a single block is computed from
    ``rmsnorm(h0 + attn_out)``, which is entirely upstream of the experts, so a
    pack whose experts alone are ternary must reproduce the reference routing
    exactly. Running that as a baseline turns the argument into a measurement.
    """

    def __init__(
        self,
        primary: WeightSource,
        fallback: WeightSource,
        roles: frozenset[str],
        label: str,
    ) -> None:
        _validate_mix(primary, fallback, roles)
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
