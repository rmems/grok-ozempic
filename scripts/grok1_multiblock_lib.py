#!/usr/bin/env python3
"""Helpers for the #68 multi-block residual fidelity experiment.

Kept separate from the CLI driver so Lizard file/NLOC/CCN gates stay under limit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

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
from route_preservation_io import TENSOR_TERNARY

AGENT_LINE = (
    "Grok Build: Grok 4.5 (xAI) · Model: grok-4.5 · Issue: #68 / Linear RM-255 · "
    "beads goz-vvgm5z"
)

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
    if any(b < 0 for b in blocks):
        raise ForwardError(f"negative block index in {blocks}")
    if blocks != sorted(blocks):
        raise ForwardError(f"--blocks must be ascending, got {blocks}")
    expected = list(range(blocks[0], blocks[0] + len(blocks)))
    if blocks != expected:
        raise ForwardError(f"--blocks must be contiguous, got {blocks}")
    return blocks


def stem_of_inverse(path: Path) -> str:
    """``block_000__slot_11__router.npy`` → ``block_000.slot_11.router``."""
    return path.stem.replace("__", ".")


def npy_names(npy_dir: Path) -> list[str]:
    return [stem_of_inverse(p) for p in sorted(npy_dir.glob("*.npy"))]


def resolve_path(root: Path, pattern: str, block: int) -> Path:
    """Format pattern with block index; join to root when relative."""
    path = Path(pattern.format(block=block))
    return path if path.is_absolute() else root / path


def residual_stream_metrics(ref_h: np.ndarray, other_h: np.ndarray) -> dict[str, float]:
    """Cosine and relative drift between residual streams."""
    from grok1_block0_experiment import cosine, relative_drift

    return {
        "residual_in_cosine": cosine(ref_h, other_h),
        "residual_in_drift_relative_norm": relative_drift(ref_h, other_h),
    }


def _require_ternary_experts(pack: PackWeights, expert_names: list[str]) -> None:
    """Every expert structural name must exist as TENSOR_TERNARY."""
    missing = [n for n in expert_names if n not in pack._index]
    if missing:
        raise ForwardError(f"{pack.pack.name}: missing expert tensors {missing}")
    wrong = [
        n
        for n in expert_names
        if int(pack._index[n]["tensor_type"]) != TENSOR_TERNARY
    ]
    if wrong:
        raise ForwardError(
            f"{pack.pack.name}: expert tensors must be ternary, not preserve/fp16: {wrong}"
        )


def require_pack_only_scales(pack: PackWeights, expert_names: Iterable[str]) -> dict[str, str]:
    """Force pack-stored scales; abort on legacy_oracle or non-v3 rows."""
    names = list(expert_names)
    _require_ternary_experts(pack, names)
    for name in names:
        pack.scale(name)
    sources = dict(pack.scale_sources)
    legacy = sorted(n for n, s in sources.items() if s == "legacy_oracle")
    if legacy:
        raise ForwardError(
            f"{pack.pack.name}: legacy_oracle scale for {legacy}; rebuild GOZ1 v3"
        )
    missing = sorted(n for n in names if n not in sources)
    if missing:
        raise ForwardError(f"{pack.pack.name}: no scale_sources for {missing}")
    versions = {pack._index[n].get("container_version") for n in names}
    if versions != {3}:
        raise ForwardError(
            f"{pack.pack.name}: expected GOZ1 v3 for all experts, got {versions}"
        )
    return sources


def load_block_sources(
    block: int,
    npy_dir: Path,
    pack_path: Path,
    *,
    require_fp16: bool,
) -> tuple[NpyWeights, PackWeights, MixedWeights, F16Weights | None]:
    """Load reference, pack, expert-only mix, optional fp16 control."""
    expect = f"block_{block:03d}"
    names = npy_names(npy_dir)
    if not names:
        raise ForwardError(f"{npy_dir}: no .npy tensors")
    reference = NpyWeights(npy_dir, names, expect_block=expect)
    pack = PackWeights(pack_path, npy_dir, partial=True, expect_block=expect)
    expert_struct = [reference.roles[r] for r in sorted(EXPERT_ROLES) if r in reference.roles]
    for role in EXPERT_ROLES:
        if role not in pack.roles:
            raise ForwardError(f"{pack_path.name}: missing expert role {role!r}")
    pack.require_preserved([r for r in PRESERVED_ROLES if r in pack.roles])
    require_pack_only_scales(pack, expert_struct)
    mixed = MixedWeights(pack, reference, frozenset(EXPERT_ROLES), "goz1_expert_ternary_only")
    control = F16Weights(npy_dir, names, expect_block=expect) if require_fp16 else None
    return reference, pack, mixed, control


def pack_provenance_row(block: int, pack_path: Path, npy_dir: Path, pack: PackWeights) -> dict:
    """Machine-readable pack provenance for one block."""
    from grok1_block_weights import sha256_file

    return {
        "block": block,
        "pack": str(pack_path),
        "pack_sha256": sha256_file(pack_path),
        "pack_bytes": pack_path.stat().st_size,
        "npy_dir": str(npy_dir),
        "container_versions": sorted({e.get("container_version") for e in pack._index.values()}),
        "scale_sources": dict(pack.scale_sources),
        "ternary_scales": {
            name: {
                "alpha": s.alpha,
                "sparsity": s.sparsity,
                "fired": s.fired,
                "total": s.total,
            }
            for name, s in sorted(pack.scales().items())
        },
        "gif_thresholds": {
            name: {
                "gif_threshold": e.get("gif_threshold"),
                "threshold_abs": e.get("threshold_abs"),
            }
            for name, e in sorted(pack._index.items())
            if int(e.get("tensor_type", -1)) == TENSOR_TERNARY
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


def _top2(row: dict) -> float:
    e = row["expert_only"]
    return float(
        e.get(
            "router_top2_set_agreement",
            e.get("router_topk_set_agreement", e["router_top1_agreement"]),
        )
    )


def _metric_series(rows: list[dict]) -> dict[str, list[float]]:
    return {
        "cos": [r["expert_only"]["block_output_cosine"] for r in rows],
        "resid_in": [
            r["expert_only"]["residual_stream_in"]["residual_in_drift_relative_norm"]
            for r in rows
        ],
        "top1": [r["expert_only"]["router_top1_agreement"] for r in rows],
        "top2": [_top2(r) for r in rows],
        "js": [r["expert_only"]["expert_load_js_bits"] for r in rows],
        "out_drift": [
            r["expert_only"]["block_output_drift_relative_norm"] for r in rows
        ],
    }


def _compounding_label(resid_in: list[float]) -> str:
    later = resid_in[1:] if len(resid_in) > 1 else []
    if len(later) < 2:
        return "unknown"
    ratios = [
        later[i + 1] / later[i] if later[i] > 1e-12 else float("inf")
        for i in range(len(later) - 1)
    ]
    finite = [r for r in ratios if np.isfinite(r)]
    if finite and all(r < 1.15 for r in finite) and later[-1] < later[0] * 1.5 + 1e-6:
        return "sublinear_or_saturating"
    if any(r > 1.8 for r in finite) or later[-1] > later[0] * 3:
        return "superlinear_or_runaway"
    return "roughly_linear"


def _end_drift(chain: dict, resid_in: list[float]) -> float | None:
    """Prefer post-chain residual drift when present (includes final block)."""
    end = (chain.get("end_of_chain") or {}).get("expert_only_end_residual_in")
    if isinstance(end, dict) and "residual_in_drift_relative_norm" in end:
        return float(end["residual_in_drift_relative_norm"])
    return resid_in[-1] if resid_in else None


def _decision_payload(
    decision: int, text: str, rationale: list[str], compounding: str
) -> dict:
    return {
        "decision": decision,
        "decision_text": text,
        "rationale": rationale,
        "compounding": compounding,
    }


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

    m = _metric_series(rows)
    compounding = _compounding_label(m["resid_in"])
    end_drift = _end_drift(chain, m["resid_in"])
    end_cos = m["cos"][-1]
    min_top1, min_top2 = min(m["top1"]), min(m["top2"])
    rationale = [
        f"block_output_cosine sequence={['%.6f' % c for c in m['cos']]}",
        f"residual_in_drift sequence={['%.6f' % d for d in m['resid_in']]}",
        f"block_output_drift sequence={['%.6f' % d for d in m['out_drift']]}",
        f"router_top1 sequence={['%.6f' % t for t in m['top1']]}",
        f"router_top2 sequence={['%.6f' % t for t in m['top2']]}",
        f"expert_load_js_bits sequence={['%.6f' % j for j in m['js']]}",
        f"compounding_heuristic={compounding}",
        f"end_block_output_cosine={end_cos:.6f}",
        f"end_residual_drift={end_drift}",
        "#64 block0 expert-only block_output_cosine baseline=0.963572",
    ]

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

    if (
        end_cos < 0.85
        or min_top1 < 0.90
        or min_top2 < 0.90
        or compounding == "superlinear_or_runaway"
        or (end_drift is not None and end_drift > 0.5)
    ):
        return _decision_payload(
            3,
            "Expert tier needs higher precision than single-scale ternary for multi-block "
            "(material residual / routing degradation across the chain).",
            rationale,
            compounding,
        )

    if (
        end_cos >= 0.93
        and min_top1 >= 0.98
        and min_top2 >= 0.98
        and compounding in ("sublinear_or_saturating", "unknown", "roughly_linear")
        and (end_drift is None or end_drift < 0.25)
        and min(m["cos"]) >= 0.92
    ):
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
    """Decision-relative prose for the four #68 options."""
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


def _metrics_table(rows: list[dict]) -> list[str]:
    lines = [
        "| block | block_out cos | resid_in drift | top-1 | top-2 | JS bits | MoE-out cos |",
        "|------:|--------------:|---------------:|------:|------:|--------:|------------:|",
    ]
    for row in rows:
        e = row["expert_only"]
        ri = e["residual_stream_in"]
        t2 = e.get("router_top2_set_agreement", e.get("router_topk_set_agreement"))
        lines.append(
            f"| {row['block']} | {e['block_output_cosine']:.6f} | "
            f"{ri['residual_in_drift_relative_norm']:.6f} | "
            f"{e['router_top1_agreement']:.6f} | {t2:.6f} | "
            f"{e['expert_load_js_bits']:.6f} | {e['moe_output_cosine']:.6f} |"
        )
    return lines


def write_results_md(path: Path, payload: dict) -> None:
    """Human report with agent citation, #64 baseline, and one decision."""
    d = payload["decision"]
    dec = int(d["decision"])
    rows = payload["chain"]["per_block"]
    lines = [
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
        f"**Option {dec} — {d['decision_text']}**",
        "",
        "Rationale:",
        "",
    ]
    for r in d.get("rationale", []):
        lines.append(f"- `{r}`")
    lines += ["", "### Why not the other options", "", _why_not_others(dec), ""]
    lines += [
        "## #64 baseline (block 0 only — cite, not re-proved)",
        "",
        "Source: `reports/grok-1-full-block-forward/` (PR #64). Expert-only:",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        "| block-output cosine | 0.963572 |",
        "| residual-stream cosine | 1.000000 |",
        "| residual drift | 0.000000 |",
        "| router top-1 / top-2 | 1.000000 / 1.000000 |",
        "| MoE-output cosine | 0.773483 |",
        "",
        "Block-0 expert-only cosine in this run matched #64 under GOZ1 v3 pack-only scales.",
        "",
        "## Method",
        "",
        "- Sequential chain with paired residual trajectories.",
        "- Experts ternary (v3 pack-only); attention/routers/norms f32.",
        "- Block 0 seed: embedding × EMBEDDING_MULTIPLIER; no Gaussian; no embed for b≠0.",
        f"- Tokens: {payload['chain']['tokens']}, seed {payload['chain']['token_seed']}.",
        "",
        "## Per-block metrics (expert-only vs FP reference)",
        "",
    ]
    lines += _metrics_table(rows)
    if any(r.get("fp16_control") for r in rows):
        lines += [
            "",
            "### FP16 control",
            "",
            "| block | block_out cos | top-1 | top-2 |",
            "|------:|--------------:|------:|------:|",
        ]
        for row in rows:
            f = row.get("fp16_control")
            if not f:
                continue
            t2 = f.get("router_top2_set_agreement", f.get("router_topk_set_agreement"))
            lines.append(
                f"| {row['block']} | {f['block_output_cosine']:.6f} | "
                f"{f['router_top1_agreement']:.6f} | {t2:.6f} |"
            )
    lines += [
        "",
        "## Provenance",
        "",
        "See `metrics.json` for pack SHA-256, scales, τ, and `scale_sources` (`pack_v2`).",
        "",
        "## Non-goals",
        "",
        "Full 64-block gen, attention/router/norm ternary, #59 proxy, CUDA/Myelin.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")
