"""Quality measurement and precision-cost validation for GH125."""

import math
from grok1_alpha_schedule_contract import (
    MARGIN_BANDS,
    RESOURCE_SCOPE,
    TOKENS,
    integer,
    number,
    require,
)


def _finite_evidence(value, path):
    """Reject non-finite retained JSON numbers without rewriting raw evidence.

    Null values (including intentional empty-margin rates) remain untouched;
    required field types and the permitted empty-band cases are checked below.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_evidence(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_evidence(item, f"{path}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"non-finite measurement evidence at {path}")


def validate_resources(r):
    require(isinstance(r, dict), "missing resource accounting")
    integers = (
        "quantized_parameters",
        "logical_code_bits",
        "actual_code_payload_bytes",
        "code_file_bytes",
        "scale_payload_bytes",
        "scale_file_bytes",
        "fp16_expert_payload_bytes",
        "measured_expert_payload_bytes",
        "high_precision_nonexpert_payload_bytes",
        "measured_block_payload_bytes",
        "hypothetical_nibble_code_bytes",
        "cache_disk_bytes",
        "scratch_disk_bytes",
        "process_tree_peak_rss_bytes",
    )
    for key in integers:
        require(integer(r.get(key)) and r[key] >= 0, f"invalid resource {key}")
    _validate_payload_costs(r)
    require(
        number(r.get("wall_seconds"), 0)
        and number(r.get("rss_sampling_interval_seconds"), 0.001, 1),
        "invalid timing evidence",
    )
    require(r["process_tree_peak_rss_bytes"] > 0, "missing sampled process-tree RSS")
    require(r.get("scope") == RESOURCE_SCOPE, "misleading resource scope")


def _validate_payload_costs(r):
    q = r["quantized_parameters"]
    require(q > 0 and r["code_dtype"] == "int8", "actual side-table codes must be int8")
    require(r["logical_code_bits"] == q * 4, "logical code bits are not storage bytes")
    require(r["actual_code_payload_bytes"] == q, "int8 payload must use actual byte cost")
    require(r["hypothetical_nibble_code_bytes"] == (q + 1) // 2, "invalid hypothetical nibble cost")
    require(r["code_file_bytes"] >= q, "code files smaller than code payload")
    require(r["scale_file_bytes"] >= r["scale_payload_bytes"] > 0, "missing scale metadata")
    require(
        r["measured_expert_payload_bytes"]
        == q + r["scale_payload_bytes"] + r["fp16_expert_payload_bytes"],
        "invalid measured expert payload total",
    )
    require(
        r["measured_block_payload_bytes"]
        == r["measured_expert_payload_bytes"] + r["high_precision_nonexpert_payload_bytes"],
        "total measured blocks omit nonexperts",
    )
    require(
        r["cache_disk_bytes"] >= r["code_file_bytes"] + r["scale_file_bytes"],
        "cache disk excludes active tables",
    )


def _metrics(m, control=False):
    require(isinstance(m, dict), "missing metrics or FP16 control")
    for key in ("block_output_cosine", "moe_output_cosine"):
        require(number(m.get(key), -1, 1 + 1e-9), f"invalid {key}")
    for key in ("router_top1_agreement", "router_top2_set_agreement"):
        require(number(m.get(key), 0, 1), f"invalid {key}")
    require(
        number(m.get("residual_stream_in", {}).get("residual_in_drift_relative_norm"), 0),
        "missing finite incoming residual drift",
    )
    if control:
        require(m["block_output_cosine"] >= 0.99, "failing FP16 weight-precision control")
    _validate_margin_bands(m)


def _margin_counts(row, limits):
    low, high = limits
    require(
        row.get("margin_low") == low and row.get("margin_high") == high,
        "changed router-margin bands",
    )
    n, f, rate = (row.get(k) for k in ("tokens", "top1_flips", "top1_flip_rate"))
    require(integer(n) and integer(f) and 0 <= f <= n, "invalid margin counts")
    if n == 0:
        require(rate is None, "invalid margin flip rate")
    else:
        require(
            number(rate, 0, 1) and math.isclose(rate, f / n, abs_tol=1e-12),
            "invalid margin flip rate",
        )
    return n, f


def _validate_margin_bands(m):
    bands = m.get("flip_stratification_by_reference_margin")
    require(isinstance(bands, list) and len(bands) == len(MARGIN_BANDS), "missing margin bands")
    count = flips = 0
    for row, limits in zip(bands, MARGIN_BANDS, strict=True):
        n, f = _margin_counts(row, limits)
        count += n
        flips += f
    require(count == TOKENS, "margin bands omit tokens")
    require(
        math.isclose(m["router_top1_agreement"], 1 - flips / TOKENS, abs_tol=1e-12),
        "margin flips disagree with top1",
    )
