#!/usr/bin/env python3
"""Helpers for GH #68 multi-block residual fidelity (kept Lizard-clean)."""
from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from grok1_block_forward import ForwardError
from grok1_block_weights import (
    EXPERT_ROLES,
    PRESERVED_ROLES,
    F16Weights,
    MixedWeights,
    NpyWeights,
    PackWeights,
)
from route_preservation_io import TENSOR_TERNARY, read_trits


class LegacyOracleError(ForwardError):
    """Raised when a pack falls back to legacy oracle α (not v3 pack-only)."""

AGENT_LINE = (
    "Grok Build: Grok 4.5 (xAI) · Model: grok-4.5 · Issue: #68 / Linear RM-255 · "
    "beads goz-vvgm5z"
)

# #73 / RM-362 — Grok Build product line; model string holds Grok-4.5 (high).
REMEDY_AGENT_LINE = (
    "Grok Build: Grok 4.5 (xAI) · Model: Grok-4.5 (high) · Issue: #73 / Linear RM-362"
)

REMEDY_V2_AGENT_LINE = (
    "OpenAI Codex: GPT-5.6 Sol (xhigh) · Issue: #75 / Linear RM-462 · beads goz-rvk"
)

REMEDY_V3_AGENT_LINE = (
    "Grok Build: Grok 4.5 (high) (xAI) · Issue: #80 / Linear RM-468 · beads goz-d603r4"
)

# Cited from reports/grok-1-expert-precision-remedy-v2/ (PR #76 option 2).
BASELINE_76_DENSER = {
    "source": "reports/grok-1-expert-precision-remedy-v2/ (PR #76 / #75 / RM-462)",
    "tokens": 2048,
    "token_seed": 2026 * 10_000 + 806,
    "blocks": [0, 1, 2, 3],
    "top_k": 2,
    "decision": 2,
    "block_output_cosine": [0.963572, 0.963211, 0.939684, 0.913853],
    "residual_in_drift": [0.0, 0.277351, 0.279321, 0.348475],
    "router_top1": [1.0, 0.887695, 0.729004, 0.615723],
    "router_top2": [1.0, 0.680664, 0.592773, 0.408691],
    "chain_exit_residual_drift": 0.4370545723375058,
    "arm_label": "expert_periodic_hp_123",
}

BASELINE_76_CEILING = {
    "source": "reports/grok-1-expert-precision-remedy-v2/hp-ceiling/ (PR #76)",
    "tokens": 2048,
    "token_seed": 2026 * 10_000 + 806,
    "blocks": [0, 1, 2, 3],
    "top_k": 2,
    "block_output_cosine": [1.0, 0.999997, 0.999998, 0.999984],
    "router_top1": [1.0, 1.0, 1.0, 1.0],
    "chain_exit_residual_drift": 0.005690267932494616,
    "arm_label": "expert_hp_ceiling",
    "viable": True,
}

V3_PRIMARY_ARM = "expert_int4"
V3_SECONDARY_ARM = "expert_int4_123"
INT4_QMAX = 7

# Cited from reports/grok-1-expert-only-multiblock/ (PR #72); bit-identical
# settings: tokens=2048, seed=20260806, blocks=0..3, pack-only v3 experts.
BASELINE_72 = {
    "source": "reports/grok-1-expert-only-multiblock/ (PR #72 / #68 / RM-255)",
    "tokens": 2048,
    "token_seed": 2026 * 10_000 + 806,
    "blocks": [0, 1, 2, 3],
    "top_k": 2,
    "decision": 3,
    "block_output_cosine": [0.963572, 0.945765, 0.884956, 0.839144],
    "residual_in_drift": [0.0, 0.277351, 0.341935, 0.498308],
    "router_top1": [1.0, 0.887695, 0.666504, 0.528320],
    "router_top2": [1.0, 0.680664, 0.548828, 0.289551],
    "chain_exit_residual_drift": 0.6538863846584987,
    "arm_label": "goz1_expert_ternary_only",
    # Pack SHAs from the #72 decision run (multiblock-68 packs).
    "pack_sha256": {
        0: "eb28728b6b66454753bc67072b105e701d99f589ca20f4edf174fecd06e0e1c4",
        1: "dd66181d6ce42ed47a6a257feaf606d20c4b0b23f471004ae373b21c1398c17f",
        2: "e61641e19735293e6802c33d69dda6f83507480fc955141f358eb0df31da8560",
        3: "9db504ff9ee08a2523f74e3f842228296458fc524a1ccf406de46c53d9e18302",
    },
    "pack_names": {
        0: "block_000-attention_plus_expert.goz1",
        1: "block_001-attention_plus_expert.goz1",
        2: "block_002-attention_plus_expert.goz1",
        3: "block_003-attention_plus_expert.goz1",
    },
}

# Cited from reports/grok-1-expert-precision-remedy/metrics.json (PR #74).
# The #74 Arm C run used the same samples and packs as #72.
BASELINE_74 = {
    "source": "reports/grok-1-expert-precision-remedy/ (PR #74 / #73 / RM-362)",
    "tokens": 2048,
    "token_seed": 2026 * 10_000 + 806,
    "blocks": [0, 1, 2, 3],
    "top_k": 2,
    "decision": 2,
    "block_output_cosine": [0.963572, 0.963211, 0.905360, 0.882308],
    "residual_in_drift": [0.0, 0.277351, 0.279321, 0.449088],
    "router_top1": [1.0, 0.887695, 0.729004, 0.547852],
    "router_top2": [1.0, 0.680664, 0.592773, 0.317383],
    "expert_load_js_bits": [0.0, 0.005559, 0.008548, 0.025243],
    "chain_exit_residual_drift": 0.5292121735722244,
    "arm_label": "expert_periodic_hp_n2",
    "pack_sha256": dict(BASELINE_72["pack_sha256"]),
    "pack_names": dict(BASELINE_72["pack_names"]),
}

BASELINE_64 = {
    "source": "reports/grok-1-full-block-forward/results.md (PR #64 / #61 / RM-249)",
    # Cite the published report only — do not attribute this PR's agent line to Fable.
    "tokens": 2048,
    "expert_only": {
        "block_output_cosine": 0.963572,
        "residual_stream_cosine": 1.0,
        "residual_drift_relative_norm": 0.0,
        "router_top1_agreement": 1.0,
        "router_top2_set_agreement": 1.0,
        "expert_load_js_bits": 0.0,
        "moe_output_cosine": 0.773483,
    },
}

V2_PRIMARY_ARM = "expert_periodic_hp_123"
V2_STACKED_ARM = "expert_periodic_hp_n2_plus_channel_alpha"
V2_CEILING_ARM = "expert_hp_ceiling"
V2_SECONDARY_ARMS = frozenset({V2_STACKED_ARM, V2_CEILING_ARM})


def parse_blocks(text: str) -> list[int]:
    """Parse ``0,1,2,3`` into a contiguous ascending chain."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ForwardError("--blocks must list at least one block index")
    blocks = [int(p) for p in parts]
    _assert_block_chain(blocks)
    return blocks


def parse_hp_blocks(text: str) -> set[int]:
    """Parse an explicit, unordered HP block set such as ``1,2,3``."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ForwardError("--hp-blocks must list at least one block index")
    try:
        blocks = {int(p) for p in parts}
    except ValueError as exc:
        raise ForwardError(f"--hp-blocks must contain integers, got {text!r}") from exc
    if any(b < 0 for b in blocks):
        raise ForwardError(f"negative HP block index in {sorted(blocks)}")
    return blocks


def _assert_block_chain(blocks: list[int]) -> None:
    if any(b < 0 for b in blocks):
        raise ForwardError(f"negative block index in {blocks}")
    if blocks != sorted(blocks):
        raise ForwardError(f"--blocks must be ascending, got {blocks}")
    expected = list(range(blocks[0], blocks[0] + len(blocks)))
    if blocks != expected:
        raise ForwardError(f"--blocks must be contiguous, got {blocks}")


def stem_of_inverse(path: Path) -> str:
    return path.stem.replace("__", ".")


def npy_names(npy_dir: Path) -> list[str]:
    return [stem_of_inverse(p) for p in sorted(npy_dir.glob("*.npy"))]


def resolve_path(root: Path, pattern: str, block: int) -> Path:
    path = Path(pattern.format(block=block))
    return path if path.is_absolute() else root / path


def residual_stream_metrics(ref_h: np.ndarray, other_h: np.ndarray) -> dict[str, float]:
    from grok1_block0_experiment import cosine, relative_drift

    return {
        "residual_in_cosine": cosine(ref_h, other_h),
        "residual_in_drift_relative_norm": relative_drift(ref_h, other_h),
    }


def _require_ternary_experts(pack: PackWeights, names: list[str]) -> None:
    present = set(pack.tensor_names())
    missing = [n for n in names if n not in present]
    if missing:
        raise ForwardError(f"{pack.pack.name}: missing expert tensors {missing}")
    wrong = [
        n for n in names if int(pack.tensor_entry(n)["tensor_type"]) != TENSOR_TERNARY
    ]
    if wrong:
        raise ForwardError(f"{pack.pack.name}: experts must be ternary: {wrong}")


def _assert_no_legacy(pack: PackWeights, sources: dict[str, str]) -> None:
    legacy = sorted(n for n, s in sources.items() if s == "legacy_oracle")
    if legacy:
        raise LegacyOracleError(
            f"{pack.pack.name}: legacy_oracle for {legacy}; rebuild GOZ1 v3"
        )


def _assert_pack_v3_scales(pack: PackWeights, names: list[str], sources: dict[str, str]) -> None:
    _assert_no_legacy(pack, sources)
    missing = sorted(n for n in names if n not in sources)
    if missing:
        raise ForwardError(f"{pack.pack.name}: no scale_sources for {missing}")
    versions = {pack.container_version(n) for n in names}
    if versions != {3}:
        raise ForwardError(f"{pack.pack.name}: expected GOZ1 v3, got {versions}")


def require_pack_only_scales(pack: PackWeights, expert_names: Iterable[str]) -> dict[str, str]:
    """Force pack-stored scales; abort on legacy_oracle or non-v3 rows."""
    names = list(expert_names)
    _require_ternary_experts(pack, names)
    for name in names:
        pack.scale(name)
    sources = dict(pack.scale_sources)
    _assert_pack_v3_scales(pack, names, sources)
    return sources


def _expert_names(reference: NpyWeights) -> list[str]:
    return [reference.roles[r] for r in sorted(EXPERT_ROLES) if r in reference.roles]


def _require_pack_experts(pack: PackWeights, pack_path: Path) -> None:
    for role in EXPERT_ROLES:
        if role not in pack.roles:
            raise ForwardError(f"{pack_path.name}: missing expert role {role!r}")


def periodic_hp_blocks(blocks: list[int], period: int = 2) -> set[int]:
    """Blocks that use high-precision experts under arm C.

    For period=2 on chain 0,1,2,3 → HP on {1,3}, ternary on {0,2}.
    Prose: \"every 2nd block\" means odd indices in the chain, not \"HP starting at 0\".
    """
    if period < 1:
        raise ForwardError(f"hp period must be >= 1, got {period}")
    return {b for i, b in enumerate(blocks) if (i % period) == (period - 1)}


def channel_alpha_dequant(weights: np.ndarray, trits: np.ndarray) -> np.ndarray:
    """Per-output-channel LS scale: α_n = Σ_k w·t / Σ_k t² along all but last axis."""
    if weights.shape != trits.shape:
        raise ForwardError(
            f"channel-α shape mismatch: weights {weights.shape} vs trits {trits.shape}"
        )
    w = weights.astype(np.float64, copy=False)
    t = trits.astype(np.float64, copy=False)
    if w.ndim <= 1:
        den = float((t * t).sum())
        alpha = (float((w * t).sum()) / den) if den > 0 else 0.0
        return trits.astype(np.float32) * np.float32(alpha)
    axes = tuple(range(w.ndim - 1))
    num = (w * t).sum(axis=axes)
    den = (t * t).sum(axis=axes)
    alpha = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return trits.astype(np.float32) * alpha.astype(np.float32)


def int4_absmax_dequant(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-channel absmax INT4 quantize-dequant (research side-table).

    ``q ∈ [-INT4_QMAX, INT4_QMAX]``; scale along all axes except the last.
    Returns ``(dequant_f32, scale_f32)`` with scale shape ``weights.shape[-1:]``.
    """
    w = np.asarray(weights, dtype=np.float32)
    if w.size == 0:
        raise ForwardError("int4 dequant: empty weight tensor")
    if not np.isfinite(w).all():
        raise ForwardError("int4 dequant: non-finite source weights (NaN/Inf)")
    if w.ndim == 1:
        amax = float(np.max(np.abs(w)))
        if not math.isfinite(amax):
            raise ForwardError("int4 dequant: non-finite absmax")
        scale = np.float32(max(amax / float(INT4_QMAX), 1e-12))
        q = np.clip(np.rint(w / scale), -INT4_QMAX, INT4_QMAX)
        deq = (q * scale).astype(np.float32)
        if not np.isfinite(deq).all():
            raise ForwardError("int4 dequant: non-finite dequantized weights")
        return deq, np.asarray([scale], dtype=np.float32)
    axes = tuple(range(w.ndim - 1))
    amax = np.max(np.abs(w), axis=axes)
    if not np.isfinite(amax).all():
        raise ForwardError("int4 dequant: non-finite absmax")
    scale = np.maximum(amax / float(INT4_QMAX), 1e-12).astype(np.float32)
    q = np.clip(np.rint(w / scale), -INT4_QMAX, INT4_QMAX)
    deq = (q * scale).astype(np.float32)
    if not np.isfinite(deq).all() or not np.isfinite(scale).all():
        raise ForwardError("int4 dequant: non-finite dequantized weights or scales")
    return deq, scale


class Int4SideExperts:
    """Expert primary: f32 reference → per-channel INT4 side-table dequant.

    Scale provenance tag is ``research_int4_side`` — never claim pack_v2.
    Side-table is formed online from the same npy used as the FP reference so
    the run is deterministic and pack-honest (no silent oracle from GOZ1).
    """

    label = "research_int4_side"

    def __init__(self, reference: NpyWeights) -> None:
        self._ref = reference
        self.roles = {r: reference.roles[r] for r in EXPERT_ROLES if r in reference.roles}
        missing = sorted(EXPERT_ROLES - set(self.roles))
        if missing:
            raise ForwardError(f"int4 primary lacks expert roles {missing}")
        self.scale_sources = {self.roles[r]: "research_int4_side" for r in self.roles}

    def vector(self, role: str) -> np.ndarray:
        w = np.asarray(self._ref.vector(role), dtype=np.float32)
        deq, _ = int4_absmax_dequant(w)
        return deq

    def matrix(self, role: str) -> np.ndarray:
        return self.vector(role)

    def expert(self, role: str, index: int) -> np.ndarray:
        w = np.asarray(self._ref.expert(role, index), dtype=np.float32)
        deq, _ = int4_absmax_dequant(w)
        return deq


class ChannelAlphaExperts:
    """Expert primary source: pack trits × research per-channel α (not pack scalar).

    Scale provenance tag is ``research_per_channel_side`` — never claim pack_v2.
    """

    label = "research_per_channel_side"

    def __init__(self, pack: PackWeights, reference: NpyWeights) -> None:
        self._pack = pack
        self._ref = reference
        self.roles = {r: pack.roles[r] for r in EXPERT_ROLES if r in pack.roles}
        missing = sorted(EXPERT_ROLES - set(self.roles))
        if missing:
            raise ForwardError(f"channel-α primary lacks expert roles {missing}")
        self.scale_sources = {
            self.roles[r]: "research_per_channel_side" for r in self.roles
        }

    def _trit_slice(self, role: str, start: int, count: int, shape: tuple[int, ...]) -> np.ndarray:
        entry = self._pack.tensor_entry(self.roles[role])
        if int(entry["tensor_type"]) != TENSOR_TERNARY:
            raise ForwardError(f"{entry['name']}: channel-α requires ternary payload")
        trits = read_trits(self._pack.pack, entry, start, count)
        return trits.astype(np.float32).reshape(shape)

    def vector(self, role: str) -> np.ndarray:
        entry = self._pack.tensor_entry(self.roles[role])
        shape = tuple(int(d) for d in entry["shape"])
        trits = self._trit_slice(role, 0, int(entry["numel"]), shape)
        w = np.asarray(self._ref.vector(role), dtype=np.float32)
        return channel_alpha_dequant(w, trits)

    def matrix(self, role: str) -> np.ndarray:
        return self.vector(role)

    def expert(self, role: str, index: int) -> np.ndarray:
        entry = self._pack.tensor_entry(self.roles[role])
        lead, rows, cols = (int(d) for d in entry["shape"])
        if not 0 <= index < lead:
            raise ForwardError(f"{entry['name']}: expert {index} out of range 0..{lead - 1}")
        per = rows * cols
        trits = self._trit_slice(role, index * per, per, (rows, cols))
        w = np.asarray(self._ref.expert(role, index), dtype=np.float32)
        return channel_alpha_dequant(w, trits)


def _expert_primary(
    mode: str,
    pack: PackWeights,
    reference: NpyWeights,
    control: F16Weights | None,
    *,
    block: int,
    hp_blocks: set[int],
    hp_period: int = 2,
    hp_label: str | None = None,
) -> tuple[object, str]:
    """Return (primary WeightSource for experts, arm label)."""
    if mode == "ternary":
        return pack, "goz1_expert_ternary_only"
    if mode == "periodic_hp":
        if block in hp_blocks:
            if control is None:
                raise ForwardError(
                    f"block {block}: periodic HP needs FP16 expert source "
                    "(do not --skip-fp16-control on decision runs)"
                )
            schedule = hp_label or f"n{int(hp_period)}"
            return control, f"expert_periodic_hp_{schedule}"
        return pack, "goz1_expert_ternary_only"
    if mode == "periodic_hp_plus_channel_alpha":
        if block in hp_blocks:
            if control is None:
                raise ForwardError(
                    f"block {block}: stacked HP+channel-alpha needs FP16 expert source "
                    "(do not --skip-fp16-control on decision runs)"
                )
            schedule = hp_label or f"n{int(hp_period)}"
            return control, f"expert_periodic_hp_{schedule}_plus_channel_alpha"
        return ChannelAlphaExperts(pack, reference), "research_per_channel_side"
    if mode == "all_hp":
        if control is None:
            raise ForwardError(
                f"block {block}: HP expert ceiling needs FP16 expert source "
                "(do not --skip-fp16-control on decision runs)"
            )
        return control, "expert_hp_ceiling"
    if mode == "channel_alpha":
        return ChannelAlphaExperts(pack, reference), "research_per_channel_side"
    if mode == "int4":
        if block in hp_blocks:
            if control is None:
                raise ForwardError(
                    f"block {block}: int4+HP needs FP16 expert source "
                    "(do not --skip-fp16-control on decision runs)"
                )
            schedule = hp_label or f"n{int(hp_period)}"
            return control, f"expert_int4_{schedule}"
        return Int4SideExperts(reference), "research_int4_side"
    raise ForwardError(f"unknown expert mode {mode!r}")


def _applied_expert_scale_sources(
    pack: PackWeights,
    primary: object,
    reference: NpyWeights,
    *,
    expert_mode: str,
    block: int,
    hp_blocks: set[int],
) -> dict[str, str]:
    """Map expert tensors to the scale source actually used for the pilot."""
    applied = dict(pack.scale_sources)
    if hasattr(primary, "scale_sources"):
        applied.update(dict(primary.scale_sources))
    elif expert_mode == "all_hp" or (
        expert_mode in ("periodic_hp", "periodic_hp_plus_channel_alpha", "int4")
        and block in hp_blocks
    ):
        for name in _expert_names(reference):
            applied[name] = "fp16_control"
    return applied


def load_block_sources(
    block: int,
    npy_dir: Path,
    pack_path: Path,
    *,
    require_fp16: bool,
    expert_mode: str = "ternary",
    hp_blocks: set[int] | None = None,
    hp_period: int = 2,
    hp_label: str | None = None,
) -> tuple[NpyWeights, PackWeights, MixedWeights, F16Weights | None]:
    """Load reference, pack, expert mix (arm-aware), optional fp16 control."""
    expect = f"block_{block:03d}"
    names = npy_names(npy_dir)
    if not names:
        raise ForwardError(f"{npy_dir}: no .npy tensors")
    hp = hp_blocks or set()
    reference = NpyWeights(npy_dir, names, expect_block=expect)
    pack = PackWeights(pack_path, npy_dir, partial=True, expect_block=expect)
    _require_pack_experts(pack, pack_path)
    pack.require_preserved([r for r in PRESERVED_ROLES if r in pack.roles])
    require_pack_only_scales(pack, _expert_names(reference))
    control = F16Weights(npy_dir, names, expect_block=expect) if require_fp16 else None
    primary, label = _expert_primary(
        expert_mode,
        pack,
        reference,
        control,
        block=block,
        hp_blocks=hp,
        hp_period=hp_period,
        hp_label=hp_label,
    )
    mixed = MixedWeights(primary, reference, frozenset(EXPERT_ROLES), label)
    mixed.applied_scale_sources = _applied_expert_scale_sources(  # type: ignore[attr-defined]
        pack, primary, reference, expert_mode=expert_mode, block=block, hp_blocks=hp
    )
    return reference, pack, mixed, control


def _host_safe_name(path: Path) -> str:
    """Basename only — do not commit home-directory paths into reports."""
    return Path(path).name


def pack_provenance_row(
    block: int,
    pack_path: Path,
    npy_dir: Path,
    pack: PackWeights,
    *,
    applied_scale_sources: dict[str, str] | None = None,
) -> dict:
    """Machine-readable pack provenance for one block."""
    names = pack.tensor_names()
    versions = {pack.container_version(n) for n in names}
    versions.discard(None)
    return {
        "block": block,
        "pack": _host_safe_name(pack_path),
        "pack_sha256": pack.pack_sha256(),
        "pack_bytes": pack_path.stat().st_size,
        "npy_dir": _host_safe_name(npy_dir),
        "container_versions": sorted(versions),
        "scale_sources": dict(applied_scale_sources or pack.scale_sources),
        "pack_scale_sources": dict(pack.scale_sources),
        "ternary_scales": {
            n: {"alpha": s.alpha, "sparsity": s.sparsity, "fired": s.fired, "total": s.total}
            for n, s in sorted(pack.scales().items())
        },
        "gif_thresholds": {
            n: {
                "gif_threshold": pack.tensor_entry(n).get("gif_threshold"),
                "threshold_abs": pack.tensor_entry(n).get("threshold_abs"),
            }
            for n in names
            if int(pack.tensor_entry(n).get("tensor_type", -1)) == TENSOR_TERNARY
        },
        "pack_metadata": {
            k: pack.metadata.get(k)
            for k in (
                "oz.quantization_version",
                "oz.gif_threshold",
                "oz.gif_threshold_authority",
                "oz.gif_threshold_scope",
            )
            if k in pack.metadata
        },
    }


def _topk_set_agreement(row: dict) -> float:
    """Router top-k set agreement; prefers explicit topk key over legacy top2."""
    e = row["expert_only"]
    return float(
        e.get(
            "router_topk_set_agreement",
            e.get("router_top2_set_agreement", e["router_top1_agreement"]),
        )
    )


def _metric_series(rows: list[dict]) -> dict[str, list[float]]:
    return {
        "cos": [r["expert_only"]["block_output_cosine"] for r in rows],
        "resid_in": [
            r["expert_only"]["residual_stream_in"]["residual_in_drift_relative_norm"] for r in rows
        ],
        "top1": [r["expert_only"]["router_top1_agreement"] for r in rows],
        "topk": [_topk_set_agreement(r) for r in rows],
        "js": [r["expert_only"]["expert_load_js_bits"] for r in rows],
        "out_drift": [r["expert_only"]["block_output_drift_relative_norm"] for r in rows],
    }


def _growth_ratios(later: list[float]) -> list[float]:
    out = []
    for i in range(len(later) - 1):
        out.append(later[i + 1] / later[i] if later[i] > 1e-12 else float("inf"))
    return out


def _is_saturating(later: list[float], finite: list[float]) -> bool:
    return bool(finite) and all(r < 1.15 for r in finite) and later[-1] < later[0] * 1.5 + 1e-6


def _is_runaway(later: list[float], finite: list[float]) -> bool:
    return any(r > 1.8 for r in finite) or later[-1] > later[0] * 3


def _label_from_later(later: list[float]) -> str:
    if len(later) < 2:
        return "unknown"
    finite = [r for r in _growth_ratios(later) if np.isfinite(r)]
    if _is_saturating(later, finite):
        return "sublinear_or_saturating"
    if _is_runaway(later, finite):
        return "superlinear_or_runaway"
    return "roughly_linear"


def _compounding_label(resid_in: list[float], end_drift: float | None = None) -> str:
    """Label residual growth; include end_drift so the final hop is counted."""
    series = list(resid_in)
    if end_drift is not None:
        series.append(float(end_drift))
    later = series[1:] if len(series) > 1 else []
    return _label_from_later(later)


def _chain_exit_metrics(chain: dict) -> dict | None:
    """Post-chain residual stream (residual into a virtual next block)."""
    end = chain.get("end_of_chain")
    if not isinstance(end, dict):
        return None
    # Prefer new keys; accept legacy misnamed key for older metrics.json.
    for key in ("expert_only_chain_exit", "expert_only_end_residual_in"):
        val = end.get(key)
        if isinstance(val, dict):
            return val
    return None


def _safe_float(raw: object) -> float | None:
    """Finite float from a real int/float only (reject bool, str, None)."""
    # bool is a subclass of int; reject JSON true/false and numeric strings.
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    val = float(raw)
    if not math.isfinite(val):
        return None
    return val


def _safe_int(raw: object, default: int | None = None) -> int | None:
    """Real non-bool int only — no float truncation or string coercion.

    Missing/malformed values return ``default`` (default ``None``) so callers
    can reject them instead of inventing baseline look-alikes (e.g. top_k→2).
    """
    if raw is None or isinstance(raw, bool) or not isinstance(raw, int):
        return default
    return raw


def _exit_metric_from_map(exit_m: dict) -> float | None:
    """First finite exit drift from preferred then legacy key; skip null/bad values."""
    for key in ("residual_drift_relative_norm", "residual_in_drift_relative_norm"):
        if exit_m.get(key) is None:
            continue
        val = _safe_float(exit_m[key])
        if val is not None:
            return val
    return None


def _exit_drift(chain: dict, out_drift: list[float]) -> float | None:
    """Drift after the final block (includes last hop). Prefer chain_exit keys."""
    exit_m = _chain_exit_metrics(chain)
    if exit_m is not None:
        val = _exit_metric_from_map(exit_m)
        if val is not None:
            return val
    return out_drift[-1] if out_drift else None


def _decision_payload(decision: int, text: str, rationale: list[str], compounding: str) -> dict:
    return {
        "decision": decision,
        "decision_text": text,
        "rationale": rationale,
        "compounding": compounding,
    }


def _topk_label(top_k: int) -> str:
    return f"router_top{top_k}_set_agreement" if top_k == 2 else f"router_topk_set_agreement_k{top_k}"


def _build_rationale(
    m: dict[str, list[float]],
    compounding: str,
    end_cos: float,
    last_resid_in: float | None,
    exit_drift: float | None,
    top_k: int,
) -> list[str]:
    baseline = BASELINE_64["expert_only"]["block_output_cosine"]
    topk_key = _topk_label(top_k)
    return [
        f"block_output_cosine sequence={['%.6f' % c for c in m['cos']]}",
        f"residual_in_drift sequence={['%.6f' % d for d in m['resid_in']]}",
        f"block_output_drift sequence={['%.6f' % d for d in m['out_drift']]}",
        f"router_top1 sequence={['%.6f' % t for t in m['top1']]}",
        f"{topk_key} sequence={['%.6f' % t for t in m['topk']]}",
        f"expert_load_js_bits sequence={['%.6f' % j for j in m['js']]}",
        f"compounding_heuristic={compounding}",
        f"end_block_output_cosine={end_cos:.6f}",
        f"last_block_residual_in_drift={last_resid_in}",
        f"chain_exit_residual_drift={exit_drift}",
        f"#64 block0 expert-only block_output_cosine baseline={baseline}",
    ]


def _fp16_gate(chain: dict, rows: list[dict], rationale: list[str], compounding: str) -> dict | None:
    if chain.get("skip_fp16_control"):
        return _decision_payload(
            4,
            "Inconclusive — FP16 control skipped; decision-quality run requires it.",
            rationale + ["fp16_control_skipped"],
            compounding,
        )
    fp16_rows = [r["fp16_control"] for r in rows if r.get("fp16_control") is not None]
    if not fp16_rows:
        return _decision_payload(
            4,
            "Inconclusive — FP16 control missing; decision-quality run requires it.",
            rationale + ["fp16_control_absent"],
            compounding,
        )
    fp16_cos = [r["block_output_cosine"] for r in fp16_rows]
    if min(fp16_cos) < 0.99:
        return _decision_payload(
            4,
            "Inconclusive — FP16 control block-output cosine fell below 0.99.",
            rationale + [f"fp16_block_output_cosine={fp16_cos}"],
            compounding,
        )
    return None


def _is_option_3(
    end_cos: float, min_top1: float, min_topk: float, compounding: str, exit_drift: float | None
) -> bool:
    return (
        end_cos < 0.85
        or min_top1 < 0.90
        or min_topk < 0.90
        or compounding == "superlinear_or_runaway"
        or (exit_drift is not None and exit_drift > 0.5)
    )


def _is_option_1(
    end_cos: float,
    min_top1: float,
    min_topk: float,
    compounding: str,
    exit_drift: float | None,
    cos: list[float],
) -> bool:
    return (
        end_cos >= 0.93
        and min_top1 >= 0.98
        and min_topk >= 0.98
        and compounding in ("sublinear_or_saturating", "unknown", "roughly_linear")
        and (exit_drift is None or exit_drift < 0.25)
        and min(cos) >= 0.92
    )


def decide(chain: dict) -> dict:
    """Pick exactly one of #68's four decision outputs."""
    rows = chain.get("per_block") or []
    if not rows:
        return _decision_payload(
            4,
            "Inconclusive — no blocks measured (activation-supply / harness gap).",
            ["empty per_block"],
            "unknown",
        )
    top_k = int(chain.get("top_k") or 2)
    m = _metric_series(rows)
    last_resid_in = m["resid_in"][-1] if m["resid_in"] else None
    exit_drift = _exit_drift(chain, m["out_drift"])
    # Compounding includes chain-exit drift so the final block hop is counted.
    compounding = _compounding_label(m["resid_in"], exit_drift)
    end_cos = m["cos"][-1]
    min_top1, min_topk = min(m["top1"]), min(m["topk"])
    rationale = _build_rationale(m, compounding, end_cos, last_resid_in, exit_drift, top_k)
    fp16_hit = _fp16_gate(chain, rows, rationale, compounding)
    if fp16_hit is not None:
        return fp16_hit
    if _is_option_3(end_cos, min_top1, min_topk, compounding, exit_drift):
        return _decision_payload(
            3,
            "Expert tier needs higher precision than single-scale ternary for multi-block "
            "(material residual / routing degradation across the chain).",
            rationale,
            compounding,
        )
    if _is_option_1(end_cos, min_top1, min_topk, compounding, exit_drift, m["cos"]):
        return _decision_payload(
            1,
            "Expert-only ternary remains viable for multi-block "
            "(drift bounded / non-compounding on the measured chain).",
            rationale,
            compounding,
        )
    return _decision_payload(
        2,
        "Needs a correction mechanism (e.g. residual feedback, scale refresh, "
        "or occasional higher-precision expert block) for multi-block expert ternary.",
        rationale,
        compounding,
    )


def _fmt_commit(impl: object) -> str:
    if isinstance(impl, dict):
        c = impl.get("commit") or "unknown"
        return f"{c}{' (dirty)' if impl.get('dirty') else ''}"
    return str(impl)


def _why_not_others(decision: int) -> str:
    selected = {
        1: "selected — drift bounded / non-compounding on the measured chain.",
        2: "selected — intermediate degradation; correction mechanism indicated.",
        3: "selected — residual-driven multi-block collapse; raise expert precision.",
        4: "selected — unresolved gap or missing/failed control.",
    }
    rejected = {
        1: "rejected — residual and/or routing degrade beyond bounded thresholds.",
        2: "rejected as primary — evidence favors higher expert precision first.",
        3: "rejected — chain stayed within bounded multi-block viability bands.",
        4: "rejected — architecture, pack v3 scales, and FP16 control resolved.",
    }
    lines = []
    for i in (1, 2, 3, 4):
        tag = "selected" if i == decision else "not chosen"
        body = selected[i] if i == decision else rejected[i]
        lines.append(f"- **Option {i} ({tag}):** {body}")
    return "\n".join(lines)


def _metrics_table(rows: list[dict], top_k: int) -> list[str]:
    topk_hdr = "top-2" if top_k == 2 else f"top-{top_k}"
    lines = [
        f"| block | block_out cos | resid_in drift | top-1 | {topk_hdr} | JS bits | MoE-out cos |",
        "|------:|--------------:|---------------:|------:|------:|--------:|------------:|",
    ]
    for row in rows:
        e = row["expert_only"]
        ri = e["residual_stream_in"]
        tk = e.get("router_topk_set_agreement", e.get("router_top2_set_agreement"))
        lines.append(
            f"| {row['block']} | {e['block_output_cosine']:.6f} | "
            f"{ri['residual_in_drift_relative_norm']:.6f} | "
            f"{e['router_top1_agreement']:.6f} | {tk:.6f} | "
            f"{e['expert_load_js_bits']:.6f} | {e['moe_output_cosine']:.6f} |"
        )
    return lines


def _fp16_table(rows: list[dict], top_k: int, heading: str = "### FP16 control") -> list[str]:
    if not any(r.get("fp16_control") for r in rows):
        return []
    topk_hdr = "top-2" if top_k == 2 else f"top-{top_k}"
    lines = [
        "",
        heading,
        "",
        f"| block | block_out cos | top-1 | {topk_hdr} |",
        "|------:|--------------:|------:|------:|",
    ]
    for row in rows:
        f = row.get("fp16_control")
        if not f:
            continue
        tk = f.get("router_topk_set_agreement", f.get("router_top2_set_agreement"))
        lines.append(
            f"| {row['block']} | {f['block_output_cosine']:.6f} | "
            f"{f['router_top1_agreement']:.6f} | {tk:.6f} |"
        )
    return lines


def _report_header(payload: dict, dec: int, decision_text: str) -> list[str]:
    return [
        "# Expert-only ternary multi-block residual fidelity",
        "",
        f"**Agent:** {AGENT_LINE}",
        "**Design:** Grok Build super-research · **Baseline (#64):** PR #64 / #61",
        "**Issue:** GH [#68](https://github.com/rmems/grok-ozempic/issues/68) / RM-255",
        "**Predecessor:** PR [#64](https://github.com/rmems/grok-ozempic/pull/64) / #61",
        f"**Implementation commit:** `{_fmt_commit(payload['provenance'].get('implementation'))}`",
        "",
        "## Decision",
        "",
        f"**Option {dec} — {decision_text}**",
        "",
        "Rationale:",
        "",
    ]


def _block0_baseline_note(rows: list[dict]) -> str:
    baseline = BASELINE_64["expert_only"]["block_output_cosine"]
    if not rows:
        return f"Source: `reports/grok-1-full-block-forward/` (PR #64). Baseline cosine {baseline}."
    measured = float(rows[0]["expert_only"]["block_output_cosine"])
    delta = abs(measured - baseline)
    if delta <= 1e-5:
        return (
            f"Source: `reports/grok-1-full-block-forward/` (PR #64). "
            f"Block-0 expert-only cosine **{measured:.6f}** matches baseline **{baseline}**."
        )
    return (
        f"Source: `reports/grok-1-full-block-forward/` (PR #64). "
        f"Block-0 expert-only cosine **{measured:.6f}** vs baseline **{baseline}** "
        f"(Δ={delta:.6f}; not an automatic match claim)."
    )


def write_results_md(path: Path, payload: dict) -> None:
    """Human report with agent citation, #64 baseline, and one decision."""
    d = payload["decision"]
    dec = int(d["decision"])
    rows = payload["chain"]["per_block"]
    top_k = int(payload["chain"].get("top_k") or 2)
    lines = _report_header(payload, dec, d["decision_text"])
    for r in d.get("rationale", []):
        lines.append(f"- `{r}`")
    note = (payload.get("provenance") or {}).get("metrics_note")
    if note:
        lines += ["", f"**Metrics note:** {note}", ""]
    lines += ["", "### Why not the other options", "", _why_not_others(dec), ""]
    lines += [
        "## #64 baseline (block 0 only — cite, not re-proved)",
        "",
        _block0_baseline_note(rows),
        "",
        "## Method",
        "",
        "- Sequential chain with paired residual trajectories.",
        "- Experts ternary (v3 pack-only); attention/routers/norms f32.",
        f"- Tokens: {payload['chain']['tokens']}, seed {payload['chain']['token_seed']}, top_k={top_k}.",
        "",
        "## Per-block metrics (expert-only vs FP reference)",
        "",
    ]
    lines += _metrics_table(rows, top_k)
    lines += _fp16_table(rows, top_k)
    lines += [
        "",
        "## Provenance",
        "",
        "See `metrics.json` for pack SHA-256, scales, τ, and `scale_sources` (`pack_v2`).",
        "Chain exit residual metrics live under "
        "`chain.end_of_chain.expert_only_chain_exit` "
        "(post-final-block residual stream, not residual-in to the last block).",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def _chain_blocks(chain: dict) -> list[int]:
    blocks = chain.get("blocks")
    if blocks is not None:
        return list(blocks)
    return [r["block"] for r in (chain.get("per_block") or [])]


def _schedule_match_72(chain: dict) -> bool:
    """Blocks/tokens/seed/top_k match the #72 decision run."""
    b72 = BASELINE_72
    # No baseline-shaped defaults: missing/malformed fields must not match #72.
    tokens = _safe_int(chain.get("tokens"))
    seed = _safe_int(chain.get("token_seed"))
    top_k = _safe_int(chain.get("top_k"))
    if tokens is None or seed is None or top_k is None:
        return False
    try:
        blocks = _chain_blocks(chain)
    except (TypeError, KeyError, AttributeError):
        return False
    observed = (blocks, tokens, seed, top_k)
    expected = (
        list(b72["blocks"]),
        int(b72["tokens"]),
        int(b72["token_seed"]),
        int(b72["top_k"]),
    )
    return observed == expected


def _canonical_block_id(value: object) -> int | None:
    """Accept only a real int block id (reject bool, float, and numeric strings)."""
    # bool is a subclass of int; reject it before the int check.
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _pack_identity_match_72(chain: dict) -> bool:
    """When pack_provenance is present, require #72 pack SHA-256 identity."""
    packs = chain.get("pack_provenance")
    if not packs:
        # Unit tests / dry metrics without packs: schedule settings only.
        return True
    if not isinstance(packs, list):
        return False
    expected = BASELINE_72["pack_sha256"]
    observed: dict[int, object] = {}
    for row in packs:
        if not isinstance(row, dict):
            return False
        block = _canonical_block_id(row.get("block"))
        if block is None:
            return False
        observed[block] = row.get("pack_sha256")
    return observed == expected


def settings_match_72(chain: dict) -> bool:
    """True when chain settings are bit-comparable to the cited #72 baseline.

    Compares blocks/tokens/seed/top_k, and pack SHA-256 when provenance is present.
    """
    return _schedule_match_72(chain) and _pack_identity_match_72(chain)


def settings_mismatch_reason(chain: dict) -> str | None:
    """Machine-readable why a chain is not comparable to #72, or None if ok."""
    if not _schedule_match_72(chain):
        return "settings_not_comparable_to_72"
    if not _pack_identity_match_72(chain):
        return "pack_identity_not_comparable_to_72"
    return None


def _improved_vs_72(m: dict[str, list[float]], exit_drift: float | None) -> bool:
    """True if remedy clearly beats cited #72 single-scale ternary baseline.

    Caller must ensure :func:`settings_match_72` first — otherwise final-block
    indices and sample size are not comparable to the fixed #72 vectors.
    """
    b72 = BASELINE_72
    if not m["top1"] or not m["cos"]:
        return False
    if len(m["top1"]) != len(b72["router_top1"]) or len(m["cos"]) != len(b72["block_output_cosine"]):
        return False
    top1_gain = m["top1"][-1] - b72["router_top1"][-1]
    cos_gain = m["cos"][-1] - b72["block_output_cosine"][-1]
    drift_gain = 0.0
    if exit_drift is not None:
        drift_gain = b72["chain_exit_residual_drift"] - exit_drift
    return top1_gain >= 0.05 or cos_gain >= 0.03 or drift_gain >= 0.08


def _remedy_option_1(
    end_cos: float,
    min_top1: float,
    min_topk: float,
    exit_drift: float | None,
    cos: list[float],
) -> bool:
    """#73 success shape: multi-block viability with documented remedy."""
    return (
        end_cos >= 0.93
        and min_top1 >= 0.95
        and min_topk >= 0.90
        and (exit_drift is None or exit_drift < 0.25)
        and min(cos) >= 0.90
    )


def _remedy_why_not(decision: int) -> str:
    selected = {
        1: "selected — remedy restores multi-block viability vs #72 collapse.",
        2: "selected — clear help vs #72 but residual/routing still need more.",
        3: "selected — tested remedies fail to stop residual-driven collapse.",
        4: "selected — inconclusive (control/pack/honesty gap).",
    }
    rejected = {
        1: "rejected — routing/residual still outside viability bands.",
        2: "rejected as primary — either fully viable or no clear help.",
        3: "rejected — remedy improved enough to prefer option 1 or 2.",
        4: "rejected — FP16 control and pack-honest path resolved.",
    }
    lines = []
    for i in (1, 2, 3, 4):
        tag = "selected" if i == decision else "not chosen"
        body = selected[i] if i == decision else rejected[i]
        lines.append(f"- **Option {i} ({tag}):** {body}")
    return "\n".join(lines)


def _remedy_rationale(chain: dict, m: dict, compounding: str, exit_drift: float | None) -> list[str]:
    top_k = int(chain.get("top_k") or 2)
    last_resid_in = m["resid_in"][-1] if m["resid_in"] else None
    rationale = _build_rationale(
        m, compounding, m["cos"][-1], last_resid_in, exit_drift, top_k
    )
    rationale.append(
        f"#72 baseline chain_exit_drift={BASELINE_72['chain_exit_residual_drift']:.6f} "
        f"b3_top1={BASELINE_72['router_top1'][-1]:.6f} "
        f"b3_cos={BASELINE_72['block_output_cosine'][-1]:.6f} (cited, not re-run)"
    )
    arm = chain.get("arm_label") or chain.get("expert_mode")
    if arm:
        rationale.append(f"arm={arm}")
    schedule = chain.get("hp_schedule_prose")
    if schedule:
        rationale.append(f"hp_schedule={schedule}")
    return rationale


def _remedy_from_metrics(
    chain: dict,
    m: dict[str, list[float]],
    exit_drift: float | None,
    rationale: list[str],
    compounding: str,
) -> dict:
    # All policy options (1–3) require #72-comparable settings; otherwise option 4.
    mismatch = settings_mismatch_reason(chain)
    if mismatch is not None:
        if mismatch == "pack_identity_not_comparable_to_72":
            text = (
                "Inconclusive — pack SHA-256 differs from cited #72 packs "
                "(schedule settings match, but weights are not the same packs)."
            )
        else:
            text = (
                "Inconclusive — chain settings differ from cited #72 baseline "
                "(blocks/tokens/seed/top_k); cannot claim option 1–3 vs that baseline."
            )
        return _decision_payload(4, text, rationale + [mismatch], compounding)
    end_cos = m["cos"][-1]
    min_top1, min_topk = min(m["top1"]), min(m["topk"])
    if _remedy_option_1(end_cos, min_top1, min_topk, exit_drift, m["cos"]):
        return _decision_payload(
            1,
            "Remedy restores multi-block viability "
            "(mostly-ternary experts with the documented remedy is policy-ready).",
            rationale,
            compounding,
        )
    if _improved_vs_72(m, exit_drift):
        return _decision_payload(
            2,
            "Remedy helps but needs further correction "
            "(clear improvement vs #72; residual still compounds).",
            rationale,
            compounding,
        )
    return _decision_payload(
        3,
        "Experts need still-higher / full higher precision "
        "(tested remedies fail to stop residual-driven routing collapse).",
        rationale,
        compounding,
    )


def decide_remedy(chain: dict) -> dict:
    """Pick exactly one of #73's four decision outputs (vs cited #72 baseline)."""
    rows = chain.get("per_block") or []
    if not rows:
        return _decision_payload(
            4,
            "Inconclusive — no blocks measured (activation-supply / harness gap).",
            ["empty per_block"],
            "unknown",
        )
    m = _metric_series(rows)
    exit_drift = _exit_drift(chain, m["out_drift"])
    compounding = _compounding_label(m["resid_in"], exit_drift)
    rationale = _remedy_rationale(chain, m, compounding, exit_drift)
    fp16_hit = _fp16_gate(chain, rows, rationale, compounding)
    if fp16_hit is not None:
        return fp16_hit
    return _remedy_from_metrics(chain, m, exit_drift, rationale, compounding)


def _v2_per_block_errors(chain: dict, label: str) -> list[str]:
    """Validate per_block row count, block IDs, and required nested metric fields."""
    rows = chain.get("per_block")
    if not isinstance(rows, list):
        return [f"{label}:per_block_missing_or_empty"]
    if not rows:
        return [f"{label}:per_block_missing_or_empty"]
    expected_blocks = list(BASELINE_72["blocks"])
    if len(rows) != len(expected_blocks):
        return [f"{label}:per_block_count={len(rows)} expected={len(expected_blocks)}"]
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            return [f"{label}:per_block[{index}]:not_a_mapping"]
    blocks: list[int] = []
    for index, row in enumerate(rows):
        block = _canonical_block_id(row.get("block"))
        if block is None:
            return [f"{label}:per_block[{index}]:invalid_block_id"]
        blocks.append(block)
    if sorted(blocks) != expected_blocks:
        return [f"{label}:per_block_blocks={sorted(blocks)} expected={expected_blocks}"]
    for row in rows:
        try:
            expert = row.get("expert_only")
            if not isinstance(expert, dict):
                return [f"{label}:block_{row.get('block')}:missing_expert_only"]
            _ = float(expert["block_output_cosine"])
            _ = float(expert["residual_stream_in"]["residual_in_drift_relative_norm"])
            _ = float(expert["router_top1_agreement"])
            _ = float(expert.get("router_topk_set_agreement", expert.get("router_top2_set_agreement", expert["router_top1_agreement"])))
            _ = float(expert["expert_load_js_bits"])
            _ = float(expert["block_output_drift_relative_norm"])
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            return [f"{label}:block_{row.get('block')}:malformed_expert_only:{exc}"]
    return []


def _malformed_exit_object(chain: dict) -> bool:
    """True when end_of_chain exit is present but yields no finite drift."""
    exit_m = _chain_exit_metrics(chain)
    if exit_m is None:
        return False
    return _exit_metric_from_map(exit_m) is None


def _v2_chain_summary(chain: dict) -> dict:
    rows = chain.get("per_block") or []
    if not rows:
        return {"arm_label": chain.get("arm_label"), "empty": True}
    if _malformed_exit_object(chain):
        return {"arm_label": chain.get("arm_label"), "empty": True, "malformed": True}
    try:
        metrics = _metric_series(rows)
        exit_drift = _exit_drift(chain, metrics["out_drift"])
        compounding = _compounding_label(metrics["resid_in"], exit_drift)
        viable = _remedy_option_1(
            metrics["cos"][-1],
            min(metrics["top1"]),
            min(metrics["topk"]),
            exit_drift,
            metrics["cos"],
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return {"arm_label": chain.get("arm_label"), "empty": True, "malformed": True}
    return {
        "arm_label": chain.get("arm_label"),
        "block_output_cosine": metrics["cos"],
        "residual_in_drift": metrics["resid_in"],
        "router_top1": metrics["top1"],
        "router_top2": metrics["topk"],
        "expert_load_js_bits": metrics["js"],
        "chain_exit_residual_drift": exit_drift,
        "compounding": compounding,
        "viable": viable,
    }


def _v2_expected_schedule(label: str) -> tuple[list[int], list[int], str]:
    schedules = {
        V2_PRIMARY_ARM: ([1, 2, 3], [], "periodic_hp"),
        V2_STACKED_ARM: ([1, 3], [0, 2], "periodic_hp_plus_channel_alpha"),
        V2_CEILING_ARM: ([0, 1, 2, 3], [], "all_hp"),
    }
    return schedules[label]


def _v2_controls(chain: dict) -> tuple[list[dict], str | None]:
    if chain.get("skip_fp16_control"):
        return [], "fp16_control_skipped"
    rows = chain.get("per_block") or []
    if not isinstance(rows, list) or not rows:
        return [], "fp16_control_absent"
    if any(not isinstance(row, dict) for row in rows):
        return [], "fp16_control_malformed"
    controls = [row.get("fp16_control") for row in rows]
    if any(control is None for control in controls):
        return [], "fp16_control_absent"
    return controls, None


def _v2_control_error(chain: dict) -> str | None:
    controls, error = _v2_controls(chain)
    if error is not None:
        return error
    cosines: list[float] = []
    for control in controls:
        try:
            if not isinstance(control, dict) or "block_output_cosine" not in control:
                return "fp16_control_malformed"
            cosines.append(float(control["block_output_cosine"]))
        except (TypeError, ValueError, KeyError):
            return "fp16_control_malformed"
    if not cosines:
        return "fp16_control_malformed"
    if min(cosines) < 0.99:
        return f"fp16_control_cosine_below_0.99:{cosines}"
    return None


def _v2_applied_source(block: int, hp_blocks: list[int], channel_blocks: list[int]) -> str:
    if block in channel_blocks:
        return "research_per_channel_side"
    return "fp16_control" if block in hp_blocks else "pack_v2"


def _v2_pack_row_errors(
    row: object,
    label: str,
    hp_blocks: list[int],
    channel_blocks: list[int],
    *,
    index: int | None = None,
) -> list[str]:
    row_tag = f"pack_provenance[{index}]" if index is not None else "pack_provenance"
    if not isinstance(row, dict):
        return [f"{label}:{row_tag}:not_a_mapping"]
    block = _canonical_block_id(row.get("block"))
    if block is None:
        return [f"{label}:{row_tag}:invalid_block_id"]
    errors: list[str] = []
    versions = row.get("container_versions")
    if not isinstance(versions, list):
        return [f"{label}:block_{block}:container_versions_not_list"]
    try:
        version_set = set(versions)
    except TypeError:
        return [f"{label}:block_{block}:container_versions_unhashable"]
    if version_set != {3}:
        errors.append(f"{label}:block_{block}:container_not_v3")
    pack_scale_sources = row.get("pack_scale_sources")
    if not isinstance(pack_scale_sources, dict):
        return [f"{label}:block_{block}:pack_scale_sources_not_mapping"]
    try:
        pack_sources = set(pack_scale_sources.values())
    except TypeError:
        return [f"{label}:block_{block}:pack_scale_sources_unhashable"]
    if pack_sources != {"pack_v2"}:
        errors.append(f"{label}:block_{block}:pack_scale_source_not_pack_v2")
    scale_sources = row.get("scale_sources")
    if not isinstance(scale_sources, dict):
        return [f"{label}:block_{block}:scale_sources_not_mapping"]
    try:
        applied = set(scale_sources.values())
    except TypeError:
        return [f"{label}:block_{block}:scale_sources_unhashable"]
    expected = _v2_applied_source(block, hp_blocks, channel_blocks)
    if applied != {expected}:
        errors.append(
            f"{label}:block_{block}:applied_scale_sources={sorted(applied)} "
            f"expected={expected}"
        )
    return errors


def _v2_scale_source_errors(chain: dict, label: str) -> list[str]:
    hp_blocks, channel_blocks, _ = _v2_expected_schedule(label)
    packs = chain.get("pack_provenance")
    if not isinstance(packs, list):
        return [f"{label}:missing_pack_provenance"]
    if len(packs) != len(BASELINE_72["blocks"]):
        return [f"{label}:missing_pack_provenance"]
    errors: list[str] = []
    for index, row in enumerate(packs):
        errors.extend(
            _v2_pack_row_errors(row, label, hp_blocks, channel_blocks, index=index)
        )
    return errors


def _v2_schedule_errors(chain: dict, label: str) -> list[str]:
    hp_blocks, channel_blocks, mode = _v2_expected_schedule(label)
    checks = [
        ("hp_blocks", list(chain.get("hp_blocks") or []), hp_blocks),
        ("channel_alpha_blocks", list(chain.get("channel_alpha_blocks") or []), channel_blocks),
        ("expert_mode", chain.get("expert_mode"), mode),
    ]
    return [
        f"{label}:{field}={observed!r} expected={expected!r}"
        for field, observed, expected in checks
        if observed != expected
    ]


def _v2_chain_errors(chain: dict, expected_label: str) -> list[str]:
    label = chain.get("arm_label")
    if label != expected_label:
        return [f"arm_label={label!r} expected={expected_label!r}"]
    errors: list[str] = []
    per_block_errors = _v2_per_block_errors(chain, expected_label)
    errors.extend(per_block_errors)
    mismatch = settings_mismatch_reason(chain)
    if mismatch is not None:
        errors.append(f"{label}:{mismatch}")
    errors.extend(_v2_schedule_errors(chain, expected_label))
    # Skip control probes when per_block is already structurally invalid; those
    # rows cannot yield a meaningful fp16 gate and may not be mappings.
    if not per_block_errors:
        control_error = _v2_control_error(chain)
        if control_error is not None:
            errors.append(f"{label}:{control_error}")
    errors.extend(_v2_scale_source_errors(chain, expected_label))
    return errors


_V2_SECONDARY_EVIDENCE_ROLE = "secondary; no independent decision"


def _v2_secondary_provenance_errors(
    payload: dict,
    label: str,
    expected_implementation: dict,
) -> list[str]:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return [f"{label}:secondary evidence missing provenance object"]
    errors = []
    if provenance.get("evidence_role") != _V2_SECONDARY_EVIDENCE_ROLE:
        errors.append(f"{label}:secondary evidence role is not locked")
    if provenance.get("implementation") != expected_implementation:
        errors.append(f"{label}:secondary implementation differs from primary")
    return errors


def _v2_secondary_map(
    payloads: list[dict],
    errors: list[str],
    expected_implementation: dict,
) -> dict[str, dict]:
    mapped: dict[str, dict] = {}
    for payload in payloads:
        if "decision" in payload:
            errors.append("secondary evidence contains a decision")
        chain = payload.get("chain")
        if not isinstance(chain, dict):
            errors.append("secondary evidence missing chain object")
            continue
        label = chain.get("arm_label")
        if label in mapped:
            errors.append(f"duplicate secondary arm {label!r}")
            continue
        if label not in V2_SECONDARY_ARMS:
            errors.append(f"unexpected secondary arm {label!r}")
            continue
        errors.extend(
            _v2_secondary_provenance_errors(payload, label, expected_implementation)
        )
        mapped[label] = payload
    missing = sorted(V2_SECONDARY_ARMS - set(mapped))
    if missing:
        errors.append(f"missing secondary arms {missing}")
    return mapped


def assemble_remedy_v2_comparison(
    primary_chain: dict,
    secondary_payloads: list[dict],
    *,
    primary_provenance: dict,
    load_errors: list[str] | None = None,
) -> dict:
    """Validate #75 evidence and return a decision-ready comparison payload."""
    errors = list(load_errors or [])
    implementation = primary_provenance.get("implementation")
    if not isinstance(implementation, dict):
        errors.append("primary evidence missing implementation provenance")
        implementation = {}
    secondary = _v2_secondary_map(secondary_payloads, errors, implementation)
    errors.extend(_v2_chain_errors(primary_chain, V2_PRIMARY_ARM))
    for label, payload in secondary.items():
        errors.extend(_v2_chain_errors(payload["chain"], label))
    summaries = {V2_PRIMARY_ARM: _v2_chain_summary(primary_chain)}
    summaries.update(
        {label: _v2_chain_summary(payload["chain"]) for label, payload in secondary.items()}
    )
    return {
        "primary_arm": V2_PRIMARY_ARM,
        "secondary_arms": secondary,
        "summaries": summaries,
        "validation_errors": errors,
    }


def _v2_improved_vs_74(summary: dict) -> bool:
    if summary.get("empty"):
        return False
    top1_gain = summary["router_top1"][-1] - BASELINE_74["router_top1"][-1]
    cos_gain = summary["block_output_cosine"][-1] - BASELINE_74["block_output_cosine"][-1]
    exit_drift = summary.get("chain_exit_residual_drift")
    drift_gain = 0.0 if exit_drift is None else BASELINE_74["chain_exit_residual_drift"] - exit_drift
    return top1_gain >= 0.05 or cos_gain >= 0.03 or drift_gain >= 0.08


def _v2_candidate_rank(summary: dict) -> tuple:
    if summary.get("empty"):
        return (False, float("-inf"), float("-inf"), float("-inf"))
    drift = summary.get("chain_exit_residual_drift")
    return (
        bool(summary["viable"]),
        float(summary["router_top1"][-1]),
        float(summary["block_output_cosine"][-1]),
        -float(drift if drift is not None else float("inf")),
    )


def _v2_best_candidate(summaries: dict) -> tuple[str, dict]:
    labels = (V2_PRIMARY_ARM, V2_STACKED_ARM)
    best_label = max(labels, key=lambda label: _v2_candidate_rank(summaries[label]))
    return best_label, summaries[best_label]


def _v2_decision_rationale(
    best_label: str,
    best: dict,
    ceiling: dict,
    any_improved: bool,
    ceiling_viable: bool,
) -> list[str]:
    return [
        f"best_mostly_ternary_arm={best_label}",
        f"best_b3_top1={best['router_top1'][-1]:.6f}",
        f"best_b3_cos={best['block_output_cosine'][-1]:.6f}",
        f"best_chain_exit_drift={best['chain_exit_residual_drift']}",
        f"#74_b3_top1={BASELINE_74['router_top1'][-1]:.6f} (cited, not re-run)",
        f"#74_b3_cos={BASELINE_74['block_output_cosine'][-1]:.6f} (cited, not re-run)",
        (
            f"#74_chain_exit_drift={BASELINE_74['chain_exit_residual_drift']:.6f} "
            "(cited, not re-run)"
        ),
        f"hp_ceiling_viable={ceiling['viable']}",
        f"any_mostly_ternary_improved_vs_74={any_improved}",
        f"locked_option_2_requires_improvement_and_ceiling={any_improved and ceiling_viable}",
    ]


def _v2_outcome(
    best: dict,
    *,
    any_improved: bool,
    ceiling_viable: bool,
) -> tuple[int, str]:
    if best["viable"]:
        return 1, "A stronger expert remedy restores multi-block viability for the measured chain."
    if any_improved and ceiling_viable:
        return 2, "Stronger remedies help, but full-HP experts or another correction remain required."
    if not any_improved and not ceiling_viable:
        return 3, "Even denser, stacked, and HP-ceiling expert remedies fail under the current policy."
    return 4, "Inconclusive — measured outcomes do not satisfy a locked decision branch."


def _v3_improved_vs_denser(summary: dict) -> bool:
    if summary.get("empty"):
        return False
    top1_gain = summary["router_top1"][-1] - BASELINE_76_DENSER["router_top1"][-1]
    cos_gain = summary["block_output_cosine"][-1] - BASELINE_76_DENSER["block_output_cosine"][-1]
    exit_drift = summary.get("chain_exit_residual_drift")
    denser_exit = BASELINE_76_DENSER["chain_exit_residual_drift"]
    drift_gain = 0.0 if exit_drift is None else denser_exit - float(exit_drift)
    return top1_gain >= 0.05 or cos_gain >= 0.03 or drift_gain >= 0.08


# #80 locked schedule: P0 all-INT4 primary + P1 denser-HP secondary (blocks 1,2,3).
_V3_SECONDARY_ARMS = frozenset({V3_SECONDARY_ARM})
_V3_SECONDARY_EVIDENCE_ROLE = "secondary; no independent decision"


def _v3_expected_schedule(label: str) -> tuple[list[int], list[int], str]:
    """Return (hp_blocks, int4_blocks, expert_mode) for a locked #80 arm label."""
    schedules = {
        V3_PRIMARY_ARM: ([], list(BASELINE_72["blocks"]), "int4"),
        V3_SECONDARY_ARM: ([1, 2, 3], [0], "int4"),
    }
    return schedules[label]


def _v3_applied_source(block: int, hp_blocks: list[int]) -> str:
    return "fp16_control" if block in hp_blocks else "research_int4_side"


def _canonical_block_list(raw: object) -> list[int] | None:
    """List of canonical block ids, or None if raw is not a list (incl. missing)."""
    if not isinstance(raw, list):
        return None
    out: list[int] = []
    for item in raw:
        block = _canonical_block_id(item)
        if block is None:
            return None
        out.append(block)
    return out


_V3_SCHEDULE_FIELDS = (
    "hp_blocks",
    "int4_blocks",
    "ternary_blocks",
    "channel_alpha_blocks",
)


def _v3_schedule_errors(chain: dict, label: str) -> list[str]:
    hp_blocks, int4_blocks, mode = _v3_expected_schedule(label)
    # Missing fields are not the same as explicit empty lists.
    for field in _V3_SCHEDULE_FIELDS:
        if field not in chain:
            return [f"{label}:{field}_missing"]
    observed = {field: _canonical_block_list(chain.get(field)) for field in _V3_SCHEDULE_FIELDS}
    for field, value in observed.items():
        if value is None:
            return [f"{label}:{field}_not_block_list"]
    # INT4 arms never use channel-α; require empty list for provenance honesty.
    checks = [
        ("hp_blocks", observed["hp_blocks"], hp_blocks),
        ("int4_blocks", observed["int4_blocks"], int4_blocks),
        ("ternary_blocks", observed["ternary_blocks"], []),
        ("channel_alpha_blocks", observed["channel_alpha_blocks"], []),
        ("expert_mode", chain.get("expert_mode"), mode),
    ]
    return [
        f"{label}:{field}={obs!r} expected={exp!r}"
        for field, obs, exp in checks
        if obs != exp
    ]


def _value_set(mapping: object) -> set | None:
    """Values of a mapping as a set, or None if not a mapping / unhashable values."""
    if not isinstance(mapping, dict):
        return None
    try:
        return set(mapping.values())
    except TypeError:
        return None


def _version_set(versions: object) -> set | None:
    if not isinstance(versions, list):
        return None
    try:
        return set(versions)
    except TypeError:
        return None


_STRUCTURAL_EXPERT_SUFFIXES = frozenset({"gate", "down", "up"})


def structural_expert_scale_map(block: int, source: str) -> dict[str, str]:
    """Three structural expert tensor names for pack/applied scale provenance."""
    return {
        f"block_{block:03d}.slot_00.moe_expert.gate": source,
        f"block_{block:03d}.slot_01.moe_expert.down": source,
        f"block_{block:03d}.slot_02.moe_expert.up": source,
    }


def _expert_tensor_keys_ok(keys: set[str], block: int) -> bool:
    """Full structural gate/up/down set for block (no aggregate fixture sentinel)."""
    if len(keys) != 3 or not all(isinstance(k, str) for k in keys):
        return False
    prefix = f"block_{block:03d}."
    suffixes: set[str] = set()
    for key in keys:
        if "moe_expert." not in key or not key.startswith(prefix):
            return False
        suffixes.add(key.rsplit(".", 1)[-1])
    return suffixes == _STRUCTURAL_EXPERT_SUFFIXES


def _v3_pack_row_shape(
    row: object, label: str, index: int
) -> tuple[int, set, dict, dict] | list[str]:
    """Return (block, versions, pack_map, scale_map) or a hard-error list."""
    row_tag = f"pack_provenance[{index}]"
    if not isinstance(row, dict):
        return [f"{label}:{row_tag}:not_a_mapping"]
    block = _canonical_block_id(row.get("block"))
    if block is None:
        return [f"{label}:{row_tag}:invalid_block_id"]
    versions = _version_set(row.get("container_versions"))
    if versions is None:
        return [f"{label}:block_{block}:container_versions_not_list"]
    pack_map = row.get("pack_scale_sources")
    scale_map = row.get("scale_sources")
    if not isinstance(pack_map, dict):
        return [f"{label}:block_{block}:pack_scale_sources_not_mapping"]
    if not isinstance(scale_map, dict):
        return [f"{label}:block_{block}:scale_sources_not_mapping"]
    return block, versions, pack_map, scale_map


def _v3_scale_value_errors(
    label: str, block: int, pack_map: dict, scale_map: dict, expected: str
) -> list[str]:
    """Value-level pack/applied source checks after keys are known good."""
    try:
        pack_sources = set(pack_map.values())
        applied = set(scale_map.values())
    except TypeError:
        return [f"{label}:block_{block}:scale_sources_unhashable"]
    errors: list[str] = []
    if pack_sources != {"pack_v2"}:
        errors.append(f"{label}:block_{block}:pack_scale_source_not_pack_v2")
    if not all(isinstance(value, str) for value in applied):
        errors.append(f"{label}:block_{block}:applied_scale_sources_not_strings")
    elif applied != {expected}:
        errors.append(
            f"{label}:block_{block}:applied_scale_sources={sorted(applied)} "
            f"expected={expected}"
        )
    return errors


def _v3_pack_row_errors(row: object, label: str, hp_blocks: list[int], *, index: int) -> list[str]:
    """Pack-honest row checks for #80 (v3 container + pack_v2 + INT4/HP applied)."""
    shaped = _v3_pack_row_shape(row, label, index)
    if isinstance(shaped, list):
        return shaped
    block, versions, pack_map, scale_map = shaped
    errors: list[str] = []
    if versions != {3}:
        errors.append(f"{label}:block_{block}:container_not_v3")
    pack_keys = set(pack_map.keys())
    scale_keys = set(scale_map.keys())
    if not _expert_tensor_keys_ok(pack_keys, block) or not _expert_tensor_keys_ok(
        scale_keys, block
    ):
        return errors + [f"{label}:block_{block}:scale_source_keys_not_expert"]
    if pack_keys != scale_keys:
        return errors + [f"{label}:block_{block}:scale_source_key_mismatch"]
    expected = _v3_applied_source(block, hp_blocks)
    errors.extend(_v3_scale_value_errors(label, block, pack_map, scale_map, expected))
    return errors


def _v3_scale_source_errors(chain: dict, label: str) -> list[str]:
    hp_blocks, _, _ = _v3_expected_schedule(label)
    packs = chain.get("pack_provenance")
    if not isinstance(packs, list) or len(packs) != len(BASELINE_72["blocks"]):
        return [f"{label}:missing_pack_provenance"]
    errors: list[str] = []
    for index, row in enumerate(packs):
        errors.extend(_v3_pack_row_errors(row, label, hp_blocks, index=index))
    return errors


def _finite_number(value: object) -> bool:
    """True for real int/float (not bool) that is finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _v3_chain_exit_error(chain: dict, label: str) -> str | None:
    """Require a real post-chain residual metric (no silent block-output fallback)."""
    end = chain.get("end_of_chain")
    if end is not None and not isinstance(end, dict):
        return f"{label}:end_of_chain_not_mapping"
    exit_m = _chain_exit_metrics(chain)
    if exit_m is None:
        return f"{label}:missing_chain_exit"
    val = _exit_metric_from_map(exit_m)
    if val is None:
        # Distinguish empty keys vs present-but-unparseable.
        if (
            exit_m.get("residual_drift_relative_norm") is None
            and exit_m.get("residual_in_drift_relative_norm") is None
        ):
            return f"{label}:chain_exit_missing_drift"
        return f"{label}:chain_exit_malformed"
    if val < 0.0:
        return f"{label}:chain_exit_out_of_domain"
    return None


def _v3_control_error(chain: dict) -> str | None:
    """FP16 control present, ≥0.99 cosine, finite, and in cosine domain."""
    error = _v2_control_error(chain)
    if error is not None:
        return error
    controls, _ = _v2_controls(chain)
    cos_hi = 1.0 + 1e-9
    for control in controls:
        if not isinstance(control, dict):
            return "fp16_control_malformed"
        cos = _safe_float(control.get("block_output_cosine"))
        if cos is None:
            return "fp16_control_non_finite"
        if cos < -1.0 or cos > cos_hi:
            return "fp16_control_out_of_domain"
    return None


def _v3_block_order_error(chain: dict, label: str) -> str | None:
    """Require per_block order [0,1,2,3] (not merely the same multiset)."""
    rows = chain.get("per_block")
    if not isinstance(rows, list) or not rows:
        return None
    blocks = [_canonical_block_id(row.get("block")) for row in rows if isinstance(row, dict)]
    expected = list(BASELINE_72["blocks"])
    if blocks != expected:
        return f"{label}:per_block_order={blocks} expected={expected}"
    return None


def _explicit_topk_raw(expert: dict) -> object:
    """Top-k set agreement without falling back to top-1 (locked top_k=2)."""
    if "router_topk_set_agreement" in expert:
        return expert["router_topk_set_agreement"]
    if "router_top2_set_agreement" in expert:
        return expert["router_top2_set_agreement"]
    return None


def _in_metric_domain(raw: object, lo: float, hi: float | None) -> bool:
    """True when raw is a real number inside [lo, hi] (hi=None → unbounded above)."""
    val = _safe_float(raw)
    if val is None or val < lo:
        return False
    return hi is None or val <= hi


def _v3_metric_domain_error(row: dict, label: str) -> str | None:
    """Reject non-finite or out-of-domain metrics that feed viable / ranking."""
    expert = row.get("expert_only")
    if not isinstance(expert, dict):
        return None
    residual = expert.get("residual_stream_in")
    if not isinstance(residual, dict):
        residual = {}
    topk = _explicit_topk_raw(expert)
    if topk is None:
        return f"{label}:block_{row.get('block')}:missing_top2_agreement"
    # Cosine upper bound allows tiny float noise above 1.0 (seen in-repo as 1+2e-16).
    cos_hi = 1.0 + 1e-9
    checks = (
        (expert.get("block_output_cosine"), -1.0, cos_hi),
        (residual.get("residual_in_drift_relative_norm"), 0.0, None),
        (expert.get("router_top1_agreement"), 0.0, 1.0),
        (topk, 0.0, 1.0),
        (expert.get("expert_load_js_bits"), 0.0, None),
        (expert.get("block_output_drift_relative_norm"), 0.0, None),
    )
    for raw, lo, hi in checks:
        if not _in_metric_domain(raw, lo, hi):
            return f"{label}:block_{row.get('block')}:metric_out_of_domain"
    return None


def _append_optional(errors: list[str], message: str | None) -> None:
    if message is not None:
        errors.append(message)


def _v3_finite_chain_errors(chain: dict, label: str) -> list[str]:
    for row in chain.get("per_block") or []:
        if isinstance(row, dict):
            message = _v3_metric_domain_error(row, label)
            if message is not None:
                return [message]
    return []


def _v3_chain_errors(chain: dict, expected_label: str) -> list[str]:
    label = chain.get("arm_label")
    if label != expected_label:
        return [f"arm_label={label!r} expected={expected_label!r}"]
    errors: list[str] = []
    per_block_errors = _v2_per_block_errors(chain, expected_label)
    errors.extend(per_block_errors)
    # Settings/pack probes subscript per_block rows — skip when structure is invalid.
    if per_block_errors:
        return errors
    _append_optional(errors, _v3_block_order_error(chain, expected_label))
    errors.extend(_v3_finite_chain_errors(chain, expected_label))
    control_error = _v3_control_error(chain)
    if control_error is not None:
        errors.append(f"{label}:{control_error}")
    mismatch = settings_mismatch_reason(chain)
    if mismatch is not None:
        errors.append(f"{label}:{mismatch}")
    errors.extend(_v3_schedule_errors(chain, expected_label))
    _append_optional(errors, _v3_chain_exit_error(chain, expected_label))
    errors.extend(_v3_scale_source_errors(chain, expected_label))
    return errors


def _v3_secondary_provenance_errors(
    payload: dict, label: str, expected_implementation: dict
) -> list[str]:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return [f"{label}:secondary evidence missing provenance object"]
    errors: list[str] = []
    if provenance.get("evidence_role") != _V3_SECONDARY_EVIDENCE_ROLE:
        errors.append(f"{label}:secondary evidence role is not locked")
    if provenance.get("implementation") != expected_implementation:
        errors.append(f"{label}:secondary implementation differs from primary")
    return errors


def _v3_secondary_entry(
    payload: object,
    mapped: dict[str, dict],
    expected_implementation: dict,
) -> tuple[str | None, dict | None, list[str]]:
    """Return (label, payload, errors) for one secondary evidence item."""
    if not isinstance(payload, dict):
        return None, None, ["secondary evidence must be a JSON object"]
    errors: list[str] = []
    if "decision" in payload:
        errors.append("secondary evidence contains a decision")
    chain = payload.get("chain")
    if not isinstance(chain, dict):
        return None, None, errors + ["secondary evidence missing chain object"]
    label = chain.get("arm_label")
    if not isinstance(label, str):
        return None, None, errors + ["secondary arm_label must be a string"]
    if label in mapped:
        return None, None, errors + [f"duplicate secondary arm {label!r}"]
    if label not in _V3_SECONDARY_ARMS:
        return None, None, errors + [f"unexpected secondary arm {label!r}"]
    errors.extend(
        _v3_secondary_provenance_errors(payload, label, expected_implementation)
    )
    return label, payload, errors


def _v3_secondary_map(
    payloads: list[dict],
    errors: list[str],
    expected_implementation: dict,
) -> dict[str, dict]:
    mapped: dict[str, dict] = {}
    for payload in payloads:
        label, accepted, item_errors = _v3_secondary_entry(
            payload, mapped, expected_implementation
        )
        errors.extend(item_errors)
        if label is not None and accepted is not None:
            mapped[label] = accepted
    missing = sorted(_V3_SECONDARY_ARMS - set(mapped))
    if missing:
        errors.append(f"missing secondary arms {missing}")
    return mapped


def _v3_cited_summaries() -> dict[str, dict]:
    return {
        "cited_denser_76": {
            "arm_label": BASELINE_76_DENSER["arm_label"],
            "block_output_cosine": list(BASELINE_76_DENSER["block_output_cosine"]),
            "residual_in_drift": list(BASELINE_76_DENSER["residual_in_drift"]),
            "router_top1": list(BASELINE_76_DENSER["router_top1"]),
            "router_top2": list(BASELINE_76_DENSER["router_top2"]),
            "chain_exit_residual_drift": BASELINE_76_DENSER["chain_exit_residual_drift"],
            "compounding": "cited",
            "viable": False,
            "cited": True,
        },
        "cited_ceiling_76": {
            "arm_label": BASELINE_76_CEILING["arm_label"],
            "block_output_cosine": list(BASELINE_76_CEILING["block_output_cosine"]),
            "router_top1": list(BASELINE_76_CEILING["router_top1"]),
            "chain_exit_residual_drift": BASELINE_76_CEILING["chain_exit_residual_drift"],
            "compounding": "cited",
            "viable": True,
            "cited": True,
        },
    }


def assemble_remedy_v3_comparison(
    primary_chain: dict,
    secondary_payloads: list[dict],
    *,
    primary_provenance: dict,
    load_errors: list[str] | None = None,
) -> dict:
    """Validate #80 evidence and return a decision-ready comparison payload."""
    errors = list(load_errors or [])
    implementation = (primary_provenance or {}).get("implementation")
    if not isinstance(implementation, dict):
        errors.append("primary evidence missing implementation provenance")
        implementation = {}
    secondary = _v3_secondary_map(secondary_payloads, errors, implementation)
    errors.extend(_v3_chain_errors(primary_chain, V3_PRIMARY_ARM))
    for label, payload in secondary.items():
        errors.extend(_v3_chain_errors(payload["chain"], label))
    summaries = {V3_PRIMARY_ARM: _v2_chain_summary(primary_chain)}
    summaries.update(_v3_cited_summaries())
    summaries.update(
        {label: _v2_chain_summary(payload["chain"]) for label, payload in secondary.items()}
    )
    return {
        "primary_arm": V3_PRIMARY_ARM,
        "secondary_arms": secondary,
        "summaries": summaries,
        "validation_errors": errors,
        "baselines_cited": {
            "denser": BASELINE_76_DENSER,
            "ceiling": BASELINE_76_CEILING,
            "baseline_74": BASELINE_74,
            "baseline_72": BASELINE_72,
        },
        "primary_provenance": {"implementation": implementation},
    }


def _finite_series(series: object) -> bool:
    """True when series is a non-empty list of finite numbers (every element)."""
    if not isinstance(series, list) or not series:
        return False
    return all(_finite_number(value) for value in series)


def _v3_middle_complete(summary: dict) -> bool:
    """Complete middle-ground arm: finite viability inputs on every block."""
    if not isinstance(summary, dict) or "viable" not in summary:
        return False
    return (
        _finite_series(summary.get("router_top1"))
        and _finite_series(summary.get("router_top2"))
        and _finite_series(summary.get("block_output_cosine"))
        and _finite_number(summary.get("chain_exit_residual_drift"))
    )


def _v3_middle_labels(summaries: dict) -> list[str]:
    return [
        label
        for label, summary in summaries.items()
        if str(label).startswith("expert_int4")
        and not summary.get("cited")
        and _v3_middle_complete(summary)
    ]


def _v3_outcome(
    best: dict,
    *,
    any_improved: bool,
    ceiling_viable: bool,
) -> tuple[int, str]:
    if best.get("viable"):
        return (
            1,
            "Middle-ground INT4 experts restore multi-block viability for the measured chain.",
        )
    if any_improved and ceiling_viable:
        return (
            2,
            "INT4 middle-ground helps vs denser ternary, but full-HP experts or another "
            "correction remain required.",
        )
    if not any_improved:
        return (
            3,
            "INT4 middle-ground fails to beat denser ternary meaningfully — gap is not "
            "payload-width alone.",
        )
    return 4, "Inconclusive — measured outcomes do not satisfy a locked decision branch."


def _v3_decision_rationale(
    best_label: str,
    best: dict,
    *,
    any_improved: bool,
    ceiling_viable: bool,
) -> list[str]:
    top1 = best["router_top1"]
    cos = best["block_output_cosine"]
    return [
        f"best_middle_ground_arm={best_label}",
        f"best_b3_top1={float(top1[-1]):.6f}",
        f"best_b3_cos={float(cos[-1]):.6f}",
        f"best_chain_exit_drift={best.get('chain_exit_residual_drift')}",
        f"#76_denser_b3_top1={BASELINE_76_DENSER['router_top1'][-1]:.6f} (cited)",
        f"#76_denser_b3_cos={BASELINE_76_DENSER['block_output_cosine'][-1]:.6f} (cited)",
        f"#76_denser_exit_drift={BASELINE_76_DENSER['chain_exit_residual_drift']:.6f} (cited)",
        f"#76_ceiling_viable={ceiling_viable} (cited)",
        f"any_middle_ground_improved_vs_denser={any_improved}",
        f"codec=research_int4_side per-output-channel absmax qmax={INT4_QMAX}",
    ]


def decide_remedy_v3(comparison: dict) -> dict:
    """Pick exactly one #80 decision across INT4 middle-ground arms vs #76 bounds."""
    errors = list(comparison.get("validation_errors") or [])
    if errors:
        return _decision_payload(
            4,
            "Inconclusive — required multi-arm evidence is incomplete or incomparable.",
            errors,
            "unknown",
        )
    summaries = comparison["summaries"]
    middle_labels = _v3_middle_labels(summaries)
    if not middle_labels:
        return _decision_payload(
            4,
            "Inconclusive — no complete INT4 middle-ground arm summaries present.",
            ["no_int4_summaries"],
            "unknown",
        )
    best_label = max(middle_labels, key=lambda label: _v2_candidate_rank(summaries[label]))
    best = summaries[best_label]
    ceiling = summaries.get("cited_ceiling_76") or {"viable": True}
    any_improved = any(_v3_improved_vs_denser(summaries[label]) for label in middle_labels)
    ceiling_viable = bool(ceiling.get("viable", True))
    decision, text = _v3_outcome(
        best, any_improved=any_improved, ceiling_viable=ceiling_viable
    )
    rationale = _v3_decision_rationale(
        best_label, best, any_improved=any_improved, ceiling_viable=ceiling_viable
    )
    payload = _decision_payload(decision, text, rationale, best.get("compounding", "unknown"))
    payload["best_remedy_arm"] = best_label
    return payload


def decide_remedy_v2(comparison: dict) -> dict:
    """Pick exactly one #75 decision across denser, stacked, and ceiling arms."""
    errors = list(comparison.get("validation_errors") or [])
    if errors:
        return _decision_payload(
            4,
            "Inconclusive — required multi-arm evidence is incomplete or incomparable.",
            errors,
            "unknown",
        )
    summaries = comparison["summaries"]
    best_label, best = _v2_best_candidate(summaries)
    ceiling = summaries[V2_CEILING_ARM]
    any_improved = any(
        _v2_improved_vs_74(summaries[label])
        for label in (V2_PRIMARY_ARM, V2_STACKED_ARM)
    )
    ceiling_viable = bool(ceiling["viable"])
    rationale = _v2_decision_rationale(
        best_label, best, ceiling, any_improved, ceiling_viable
    )
    decision, text = _v2_outcome(
        best,
        any_improved=any_improved,
        ceiling_viable=ceiling_viable,
    )
    result = _decision_payload(decision, text, rationale, best["compounding"])
    result["best_remedy_arm"] = best_label
    return result


def _remedy_header_lines(prov: dict, dec: int, decision_text: str) -> list[str]:
    return [
        "# Expert higher-precision remedies for multi-block residual fidelity",
        "",
        f"**Agent:** {prov.get('agent', REMEDY_AGENT_LINE)}",
        "**Predecessor:** PR [#72](https://github.com/rmems/grok-ozempic/pull/72) / "
        "#68 option 3 · **Baseline (#64):** PR #64 / #61",
        f"**Implementation commit:** `{_fmt_commit(prov.get('implementation'))}`",
        "",
        "## Decision",
        "",
        f"**Option {dec} — {decision_text}**",
        "",
        "Rationale:",
        "",
    ]


def _remedy_baseline_section(*, mismatch: str | None) -> list[str]:
    b = BASELINE_72
    if mismatch is None:
        title = "## #72 baseline (cited — comparable settings + packs)"
        footer = "Re-run only if packs or harness invalidate comparison; this report cites."
    elif mismatch == "pack_identity_not_comparable_to_72":
        title = "## #72 baseline (reference only — pack identity not comparable)"
        footer = (
            "Schedule (blocks/tokens/seed/top_k) matches #72, but pack SHA-256 "
            "differs; option 1–3 vs that baseline are disabled (decision option 4)."
        )
    else:
        title = "## #72 baseline (reference only — schedule not comparable)"
        footer = (
            "This run's blocks/tokens/seed/top_k differ from the cited #72 decision run; "
            "option 1–3 vs that baseline are disabled (decision option 4)."
        )
    return [
        title,
        "",
        f"Source: `{b['source']}`.",
        f"Tokens={b['tokens']}, seed={b['token_seed']}, blocks={b['blocks']}.",
        f"Chain-exit residual drift **{b['chain_exit_residual_drift']:.6f}**; "
        f"b3 top-1 **{b['router_top1'][-1]:.6f}**; "
        f"b3 block_out cos **{b['block_output_cosine'][-1]:.6f}**.",
        "Decision option **3** (single-scale ternary not multi-block viable).",
        footer,
        "",
    ]


def remedy_metrics_note(chain: dict) -> str:
    """Human metrics_note reflecting schedule vs pack mismatch precisely."""
    reason = settings_mismatch_reason(chain)
    if reason is None:
        return (
            "#72 single-scale ternary baseline cited from "
            "reports/grok-1-expert-only-multiblock/ "
            "(matching seed/tokens/blocks/top_k; pack SHA when recorded)."
        )
    if reason == "pack_identity_not_comparable_to_72":
        return (
            "Schedule settings match #72, but pack SHA-256 differs from the "
            "cited decision packs; not comparable for option 1–3."
        )
    return (
        "Chain settings differ from cited #72 baseline "
        "(blocks/tokens/seed/top_k); not comparable."
    )


def _remedy_method_lines(chain: dict, top_k: int) -> list[str]:
    lines = [
        "## Method",
        "",
        f"- Arm: `{chain.get('expert_mode', 'unknown')}` / label `{chain.get('arm_label', '')}`.",
    ]
    if chain.get("hp_schedule_prose"):
        lines.append(f"- {chain['hp_schedule_prose']}")
    lines += [
        "- Sequential chain with paired residual trajectories.",
        "- Attention / routers / norms never ternarized.",
        f"- Tokens: {chain['tokens']}, seed {chain['token_seed']}, top_k={top_k}.",
        "",
        "## Per-block metrics (remedy arm vs FP reference)",
        "",
    ]
    return lines


def write_remedy_results_md(path: Path, payload: dict) -> None:
    """#73 human report: Grok Build citation, #72 cite, arm C schedule prose."""
    d = payload["decision"]
    chain = payload["chain"]
    rows = chain["per_block"]
    top_k = int(chain.get("top_k") or 2)
    prov = payload.get("provenance") or {}
    mismatch = settings_mismatch_reason(chain)
    note = prov.get("metrics_note") or remedy_metrics_note(chain)
    lines = _remedy_header_lines(prov, int(d["decision"]), d["decision_text"])
    lines += [f"- `{r}`" for r in d.get("rationale", [])]
    lines += ["", f"**Metrics note:** {note}", ""]
    lines += ["", "### Why not the other options", "", _remedy_why_not(int(d["decision"])), ""]
    lines += _remedy_baseline_section(mismatch=mismatch)
    lines += _remedy_method_lines(chain, top_k)
    lines += _metrics_table(rows, top_k)
    lines += _fp16_table(rows, top_k)
    metrics_name = prov.get("metrics_filename") or "metrics.json"
    lines += [
        "",
        "## Provenance",
        "",
        f"See `{metrics_name}` for pack SHA-256, scales, τ, and applied `scale_sources`.",
        "Chain exit under `chain.end_of_chain.expert_only_chain_exit`.",
        "Arm A applied scales use `research_per_channel_side` (pack row may still be `pack_v2`).",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def _v2_baseline_lines(number: int, baseline: dict) -> list[str]:
    return [
        f"## #{number} baseline (cited — bit-comparable settings and packs)",
        "",
        f"Source: `{baseline['source']}` (not re-run).",
        (
            f"Tokens={baseline['tokens']}, seed={baseline['token_seed']}, "
            f"blocks={baseline['blocks']}, top_k={baseline['top_k']}."
        ),
        (
            f"Chain-exit residual drift **{baseline['chain_exit_residual_drift']:.6f}**; "
            f"b3 top-1 **{baseline['router_top1'][-1]:.6f}**; "
            f"b3 block_out cos **{baseline['block_output_cosine'][-1]:.6f}**."
        ),
        f"Prior decision: **{baseline['decision']}**.",
        "",
    ]


def _v2_table_value(value: float, best: float) -> str:
    rendered = f"{value:.6f}"
    return f"**{rendered}**" if abs(value - best) <= 1e-12 else rendered


def _v2_comparison_table(comparison: dict) -> list[str]:
    summaries = comparison["summaries"]
    columns = [
        ("#72 ternary", BASELINE_72),
        ("#74 N=2", BASELINE_74),
        ("C denser", summaries[V2_PRIMARY_ARM]),
        ("N=2+C+A", summaries[V2_STACKED_ARM]),
        ("HP ceiling", summaries[V2_CEILING_ARM]),
    ]
    rows = [
        ("b3 top-1", "router_top1", True),
        ("b3 top-2", "router_top2", True),
        ("b3 block_out cos", "block_output_cosine", True),
        ("chain-exit drift", "chain_exit_residual_drift", False),
    ]
    lines = [
        "| Signal | " + " | ".join(name for name, _ in columns) + " |",
        "|---|" + "|".join("---:" for _ in columns) + "|",
    ]
    for label, key, higher_is_better in rows:
        values = []
        for _, source in columns:
            value = source[key]
            if isinstance(value, list):
                value = value[-1]
            values.append(float(value))
        best = (max if higher_is_better else min)(values)
        lines.append(
            f"| {label} | "
            + " | ".join(_v2_table_value(value, best) for value in values)
            + " |"
        )
    return lines


def _v2_why_not(decision: int) -> str:
    reasons = {
        1: {
            1: "selected — a mostly-ternary remedy reaches every locked viability band.",
            2: "not chosen — the measured policy is already viable; further correction is not required.",
            3: "not chosen — a viable mostly-ternary arm contradicts the failure outcome.",
            4: "not chosen — the complete evidence resolves directly to viability.",
        },
        2: {
            1: "not chosen — neither mostly-ternary candidate met every locked viability band.",
            2: "selected — clear improvement and a viable HP ceiling both hold, but policy misses viability.",
            3: "not chosen — clear improvement plus a viable ceiling contradict total failure.",
            4: "not chosen — complete, comparable evidence satisfies the locked Option-2 pair.",
        },
        3: {
            1: "not chosen — no mostly-ternary candidate met every locked viability band.",
            2: "not chosen — improvement and a viable HP ceiling did not both hold.",
            3: "selected — remedies do not improve #74 and even the expert HP ceiling fails.",
            4: "not chosen — complete evidence resolves to the locked failure branch.",
        },
        4: {
            1: "not chosen — viability is not established by the available evidence.",
            2: "not chosen — the locked improvement-plus-ceiling pair is not established.",
            3: "not chosen — the locked joint-failure condition is not established.",
            4: "selected — evidence is incomplete/incomparable or outcomes miss every locked branch.",
        },
    }
    return "\n".join(
        f"- **Option {number}:** {reasons[decision][number]}"
        for number in (1, 2, 3, 4)
    )


def _v2_appendix_lines(label: str, payload: dict) -> list[str]:
    chain = payload["chain"]
    top_k = int(chain.get("top_k") or 2)
    lines = [
        f"### `{label}`",
        "",
        chain.get("hp_schedule_prose") or "No periodic schedule.",
        "",
    ]
    lines += _metrics_table(chain["per_block"], top_k)
    lines += _fp16_table(chain["per_block"], top_k, f"#### FP16 control — `{label}`")
    lines.append("")
    return lines


def _v2_report_header(payload: dict) -> list[str]:
    decision = payload["decision"]
    prov = payload.get("provenance") or {}
    dec = int(decision["decision"])
    return [
        "# Stacked and denser expert remedies for multi-block fidelity",
        "",
        f"**Agent:** {prov.get('agent', REMEDY_V2_AGENT_LINE)}",
        f"**Issue:** {prov.get('issue', 'GH #75 / Linear RM-462')}",
        "**Predecessor:** PR #74 / #73 / RM-362 (decision 2)",
        f"**Implementation commit:** `{_fmt_commit(prov.get('implementation'))}`",
        "",
        "## Decision",
        "",
        f"**Option {dec} — {decision['decision_text']}**",
        "",
        f"**Best mostly-ternary remedy:** `{decision.get('best_remedy_arm', 'unresolved')}`",
        "",
        "Rationale:",
        "",
        *[f"- `{item}`" for item in decision.get("rationale", [])],
        "",
        "### Why not the other options",
        "",
        _v2_why_not(dec),
        "",
    ]


def _v2_comparison_lines(comparison: dict) -> list[str]:
    lines = ["## Multi-arm comparison", ""]
    if comparison.get("validation_errors"):
        lines += [
            "Comparison validation failed:",
            "",
            *[f"- `{error}`" for error in comparison["validation_errors"]],
            "",
        ]
    else:
        lines += _v2_comparison_table(comparison)
        lines.append("")
    return lines


def _v2_primary_lines(chain: dict) -> list[str]:
    top_k = int(chain.get("top_k") or 2)
    lines = _remedy_method_lines(chain, top_k)
    lines += _metrics_table(chain["per_block"], top_k)
    lines += _fp16_table(chain["per_block"], top_k)
    return lines


def _v2_secondary_lines(comparison: dict) -> list[str]:
    lines = ["", "## Secondary-arm appendices", ""]
    secondary_arms = comparison.get("secondary_arms", {})
    for label in (V2_STACKED_ARM, V2_CEILING_ARM):
        if label in secondary_arms:
            lines += _v2_appendix_lines(label, secondary_arms[label])
    return lines


def _v2_provenance_lines() -> list[str]:
    return [
        "## Provenance",
        "",
        (
            "See `metrics.json` for the canonical decision, embedded secondary evidence, "
            "pack SHA-256, thresholds, schedules, and applied scale-source tags."
        ),
        "Secondary `metrics.json` files are evidence-only and intentionally contain no decision.",
        "The HP ceiling is an expert-tier bound, not a product recommendation.",
    ]


def write_remedy_v2_results_md(path: Path, payload: dict) -> None:
    """Write #75's sole decision plus all three measured-arm comparisons."""
    chain = payload["chain"]
    comparison = payload["comparison"]
    lines = _v2_report_header(payload)
    lines += _v2_baseline_lines(72, BASELINE_72)
    lines += _v2_baseline_lines(74, BASELINE_74)
    lines += _v2_comparison_lines(comparison)
    lines += _v2_primary_lines(chain)
    lines += _v2_secondary_lines(comparison)
    lines += _v2_provenance_lines()
    path.write_text("\n".join(lines).rstrip() + "\n")
