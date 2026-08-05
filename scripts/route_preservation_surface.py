r"""
The xai-dissect **route-preservation** gate surface, as data (GH #53 / RM-222).

run3's ``route-preservation-report.json`` declares fifteen metrics, their
scopes and their thresholds, but leaves every ``observed`` value ``null``.
This module holds that declaration and knows how to reduce measurements into
it; ``route_preservation_metrics.py`` does the measuring.
"""

from __future__ import annotations


def gate(observed: float, threshold: float, higher_is_better: bool = True) -> str:
    if observed is None:
        return "unknown"
    ok = observed >= threshold if higher_is_better else observed <= threshold
    return "pass" if ok else "fail"


# The run3 route-preservation surface, as data. Each row says where its observed
# value comes from: `agg` picks the reduction, `key` the field inside each
# routing/weights entry. Worst case is what a gate must see, so router/block
# metrics reduce with min and error metrics with max.
_SUMMARY_SPECS: tuple[tuple, ...] = (
    (
        "router_top1_agreement", "router_behavior", "routing_min",
        "router_top1_agreement", ">= 99.0%", 0.99,
        ("Worst case over evaluated d_model x d_model projections; router read from "
        "the pack (fp16 preserve tier)."),
    ),
    (
        "router_top2_set_agreement", "router_behavior", "routing_min",
        "router_top2_set_agreement", ">= 99.5%", 0.995,
        "Fraction of tokens whose unordered top-2 expert set is unchanged.",
    ),
    (
        "expert_load_distribution_delta", "router_behavior", "routing_max",
        "expert_load_distribution_delta", None, None,
        "Max per-expert share change in the top-1 load histogram.",
    ),
    (
        "expert_load_js_divergence", "router_behavior", "routing_max",
        "expert_load_js_divergence", None, None,
        ("Jensen-Shannon divergence (bits) between reference and pilot expert-load "
        "distributions."),
    ),
    (
        "router_logit_rank_correlation", "router_behavior", "routing_min",
        "router_logit_rank_correlation", None, None,
        "Mean per-token Spearman correlation over the router logits.",
    ),
    (
        "block_output_cosine", "block_behavior", "routing_min",
        "block_output_cosine", ">= 0.995", 0.995,
        ("Scoped to the quantized projection output, not a full block forward "
        "(attention roles are unassigned upstream; MoE not executed)."),
    ),
    (
        "block_output_rmse", "block_behavior", "routing_max",
        "block_output_rmse", None, None,
        "RMSE of the projection output against the f32 reference path.",
    ),
    (
        "residual_stream_drift", "block_behavior", "routing_max",
        "residual_stream_drift", None, None,
        "Projection-output RMSE normalized by reference output RMS.",
    ),
    (
        "weight_reconstruction_mse", "weight_reconstruction", "weights_max",
        "weight_reconstruction_mse", None, None,
        "Worst quantized tensor, at the least-squares optimal ternary scale.",
    ),
    (
        "weight_cosine_similarity", "weight_reconstruction", "weights_min",
        "weight_cosine_similarity", None, None,
        "Worst quantized tensor; independent of the chosen ternary scale.",
    ),
    (
        "weight_max_absolute_error", "weight_reconstruction", "weights_max",
        "weight_max_absolute_error", None, None,
        "Exact max |w - alpha*t| over all quantized tensors.",
    ),
    (
        "per_channel_scale_error_summary", "weight_reconstruction", "weights_max",
        "per_channel_scale_error.relative_error_max", None, None,
        ("Worst per-output-channel relative reconstruction error; full per-tensor "
        "spread under `weights`."),
    ),
    (
        "logit_kl", "model_behavior", "none", "", None, None,
        ("Requires whole-model inference; out of scope for a bounded single-block "
        "pilot (#53 non-goal)."),
    ),
    (
        "perplexity_delta", "model_behavior", "none", "", None, None,
        "Requires a calibration corpus and whole-model inference; not run.",
    ),
    (
        "generation_sanity_summary", "model_behavior", "none", "", None, None,
        "Requires whole-model inference; not run.",
    ),
)


def _dig(entry: dict, dotted: str):
    for part in dotted.split("."):
        entry = entry[part]
    return entry


def _observe(agg: str, key: str, weights: dict, routing: dict):
    """Reduce one metric across the measured tensors/projections."""
    if agg == "none":
        return None
    source = routing if agg.startswith("routing") else weights
    values = [_dig(v, key) for v in source.values()]
    if not values:
        return None
    return min(values) if agg.endswith("min") else max(values)


def build_summary(weights: dict[str, dict], routing: dict[str, dict]) -> list[dict]:
    """Assemble run3's route-preservation surface with observed values filled in."""
    summary = []
    for name, scope, agg, key, threshold, limit, detail in _SUMMARY_SPECS:
        observed = _observe(agg, key, weights, routing)
        if limit is not None:
            status = gate(observed, limit)
        elif agg == "none":
            status = "unknown"
        else:
            status = "measured" if observed is not None else "unknown"
        summary.append(
            {
                "name": name,
                "scope": scope,
                "status": status,
                "threshold": threshold,
                "observed": observed,
                "detail": detail,
            }
        )
    return summary
