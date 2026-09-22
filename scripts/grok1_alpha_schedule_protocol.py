"""Frozen GH125 four-cell analysis; historical GH85 decisions are not inputs."""

# Keep the existing public API while separating validation responsibilities.
from grok1_alpha_schedule_contract import (  # noqa: F401
    BLOCKS,
    CELLS,
    CONTRAST_KEYS,
    DIRECTIONS,
    INTERACTION,
    MARGIN_BANDS,
    PROTOCOL,
    RESOURCE_SCOPE,
    SEED,
    TOKENS,
    TOKEN_IDS_SHA256,
    TOP_K,
    Cell,
    number,
    require,
    sha,
)
from grok1_alpha_schedule_evidence import validate_cell
from grok1_alpha_schedule_measurements import validate_resources  # noqa: F401


def paired_contrasts(values):
    require(set(values) == set(CELLS), "contrasts require exactly A/B/C/D")
    require(all(number(x) for x in values.values()), "non-finite contrast input")
    a, b, c, d = (values[x] for x in "ABCD")
    result = {
        "B-A": b - a,
        "D-C": d - c,
        "C-A": c - a,
        "D-B": d - b,
        INTERACTION: (d - c) - (b - a),
    }
    require(all(number(x) for x in result.values()), "non-finite contrast result")
    return result


def _series(payload):
    result = {}
    for row in payload["chain"]["per_block"]:
        m, b = row["expert_only"], row["block"]
        for key in (
            "block_output_cosine",
            "moe_output_cosine",
            "router_top1_agreement",
            "router_top2_set_agreement",
        ):
            result[f"block_{b}/{key}"] = (m[key], DIRECTIONS[key])
        result[f"block_{b}/incoming_residual_drift"] = (
            m["residual_stream_in"]["residual_in_drift_relative_norm"],
            "lower",
        )
        for i, band in enumerate(m["flip_stratification_by_reference_margin"]):
            result[f"block_{b}/margin_{i}/top1_flip_rate"] = (band["top1_flip_rate"], "lower")
    end = payload["chain"]["end_of_chain"]["expert_only_chain_exit"]
    result["chain_exit_drift"] = (end["residual_drift_relative_norm"], "lower")
    result["chain_exit_cosine"] = (end["residual_cosine"], "higher")
    return result


def _shared_identity(payload):
    return (
        payload["provenance"],
        [
            (r["block"], r["pack_sha256"], r["npy_sha256"])
            for r in payload["chain"]["pack_provenance"]
        ],
    )


def _validate_matrix(cells):
    require(set(cells) == set(CELLS), "analysis requires exactly A/B/C/D")
    run_id = cells["A"].get("run_id")
    identities = []
    for cell in CELLS:
        payload = cells[cell]
        validate_cell(payload, cell, run_id)
        identities.append(_shared_identity(payload))
    require(all(x == identities[0] for x in identities), "shared provenance mismatch")
    return run_id


def _observed_effect(label, delta, direction):
    favorable = delta if direction == "higher" else -delta
    if label == INTERACTION:
        if favorable == 0:
            return "no observed interaction"
        change = "more" if favorable > 0 else "less"
        return f"scaling effect becomes {change} favorable with HP123"
    if favorable == 0:
        return "unchanged"
    return "improved" if favorable > 0 else "worsened"


def _metric_contrasts(values, direction):
    result = {"values": values, "favorable_direction": direction, "contrasts": None}
    if any(v is None for v in values.values()):
        result["unavailable_reason"] = "empty reference-margin band"
        return result
    result["contrasts"] = paired_contrasts(values)
    result["observed_effects"] = {
        label: _observed_effect(label, delta, direction)
        for label, delta in result["contrasts"].items()
    }
    return result


def analyze(cells):
    run_id = _validate_matrix(cells)
    series = {c: _series(cells[c]) for c in CELLS}
    contrasts = {}
    for metric, (_, direction) in series["A"].items():
        values = {c: series[c][metric][0] for c in CELLS}
        contrasts[metric] = _metric_contrasts(values, direction)
    return {
        "protocol": PROTOCOL,
        "run_id": run_id,
        "status": "complete",
        "cells": cells,
        "paired_contrasts": contrasts,
        "top1_diagnostic_band": 0.95,
        "interpretation": "Contrasts describe this sampled-token run. Higher cosine/agreement "
        "and lower drift/flips are favorable. Scaling, schedule and interaction effects are "
        "metric-specific; no language-quality or deployment certification. Repeated "
        "deterministic runs are reproducibility checks, not independent evidence.",
        "precision_cost_note": "A/B and C/D have different FP16 retention. Quality gains "
        "across schedules are not equal-byte compression wins; nibble packing is hypothetical.",
        "control_semantics": "fp16_roundtrip is weight-precision sensitivity, not native "
        "FP16 compute, an expert-only ceiling, or upstream parity.",
    }
