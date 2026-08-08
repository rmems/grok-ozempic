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
from route_preservation_io import TENSOR_TERNARY


class LegacyOracleError(ForwardError):
    """Raised when a pack falls back to legacy oracle α (not v3 pack-only)."""

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
    _require_pack_experts(pack, pack_path)
    pack.require_preserved([r for r in PRESERVED_ROLES if r in pack.roles])
    require_pack_only_scales(pack, _expert_names(reference))
    mixed = MixedWeights(pack, reference, frozenset(EXPERT_ROLES), "goz1_expert_ternary_only")
    control = F16Weights(npy_dir, names, expect_block=expect) if require_fp16 else None
    return reference, pack, mixed, control


def _host_safe_name(path: Path) -> str:
    """Basename only — do not commit home-directory paths into reports."""
    return Path(path).name


def pack_provenance_row(block: int, pack_path: Path, npy_dir: Path, pack: PackWeights) -> dict:
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
        "scale_sources": dict(pack.scale_sources),
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
        "Chain exit residual metrics live under `end_of_chain.expert_only_chain_exit` "
        "(post-final-block residual stream, not residual-in to the last block).",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")
