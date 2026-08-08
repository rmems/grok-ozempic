#!/usr/bin/env python3
"""Helpers for GH #68 multi-block residual fidelity (kept Lizard-clean)."""
from __future__ import annotations

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

# #73 / RM-362 — SpacexAI product line; model string holds Grok-4.5 (high).
REMEDY_AGENT_LINE = (
    "SpacexAI · Model: Grok-4.5 (high) · Issue: #73 / Linear RM-362"
)

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

BASELINE_64 = {
    "source": "reports/grok-1-full-block-forward/results.md (PR #64 / #61 / RM-249)",
    "agent_measurement": "Claude Code: Fable 5",
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


def parse_blocks(text: str) -> list[int]:
    """Parse ``0,1,2,3`` into a contiguous ascending chain."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ForwardError("--blocks must list at least one block index")
    blocks = [int(p) for p in parts]
    _assert_block_chain(blocks)
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
            return control, f"expert_periodic_hp_n{int(hp_period)}"
        return pack, "goz1_expert_ternary_only"
    if mode == "channel_alpha":
        return ChannelAlphaExperts(pack, reference), "research_per_channel_side"
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
    elif expert_mode == "periodic_hp" and block in hp_blocks:
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
        expert_mode, pack, reference, control, block=block, hp_blocks=hp, hp_period=hp_period
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
    end = chain.get("end_of_chain") or {}
    # Prefer new keys; accept legacy misnamed key for older metrics.json.
    for key in ("expert_only_chain_exit", "expert_only_end_residual_in"):
        val = end.get(key)
        if isinstance(val, dict):
            return val
    return None


def _exit_drift(chain: dict, out_drift: list[float]) -> float | None:
    """Drift after the final block (includes last hop). Prefer chain_exit keys."""
    exit_m = _chain_exit_metrics(chain)
    if exit_m is not None:
        if "residual_drift_relative_norm" in exit_m:
            return float(exit_m["residual_drift_relative_norm"])
        if "residual_in_drift_relative_norm" in exit_m:
            return float(exit_m["residual_in_drift_relative_norm"])
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


def _fp16_table(rows: list[dict], top_k: int) -> list[str]:
    if not any(r.get("fp16_control") for r in rows):
        return []
    topk_hdr = "top-2" if top_k == 2 else f"top-{top_k}"
    lines = [
        "",
        "### FP16 control",
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
        "**Design:** Grok Build super-research · **Baseline (#64):** Claude Fable 5",
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
    observed = (
        _chain_blocks(chain),
        int(chain.get("tokens") or 0),
        int(chain.get("token_seed") or -1),
        int(chain.get("top_k") or 2),
    )
    expected = (
        list(b72["blocks"]),
        int(b72["tokens"]),
        int(b72["token_seed"]),
        int(b72["top_k"]),
    )
    return observed == expected


def _pack_identity_match_72(chain: dict) -> bool:
    """When pack_provenance is present, require #72 pack SHA-256 identity."""
    packs = chain.get("pack_provenance")
    if not packs:
        # Unit tests / dry metrics without packs: schedule settings only.
        return True
    expected = BASELINE_72["pack_sha256"]
    observed = {int(r["block"]): r.get("pack_sha256") for r in packs}
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


def _remedy_header_lines(prov: dict, dec: int, decision_text: str) -> list[str]:
    return [
        "# Expert higher-precision remedies for multi-block residual fidelity",
        "",
        f"**Agent:** {prov.get('agent', REMEDY_AGENT_LINE)}",
        "**Predecessor:** PR [#72](https://github.com/rmems/grok-ozempic/pull/72) / "
        "#68 option 3 · **Baseline (#64):** Claude Fable 5",
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
    """#73 human report: SpacexAI citation, #72 cite, arm C schedule prose."""
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
