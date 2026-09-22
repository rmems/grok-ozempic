"""Frozen GH125 four-cell contract. Historical GH85 decisions are not inputs."""
from dataclasses import dataclass
import hashlib
import math
import re
import struct
from types import MappingProxyType

PROTOCOL = "grok1-int4-alpha-schedule-v1"
BLOCKS = (0, 1, 2, 3)
TOKENS = 8192
SEED = 20260806
TOKEN_IDS_SHA256 = "57b0e364bb25ac4bd9047b592e28ae1470af94481be073a1f963c7cd0a5ee3de"
TOP_K = 2
MARGIN_BANDS = ((0., .01), (.01, .05), (.05, .15), (.15, .5), (.5, 1.01))
RESOURCE_SCOPE = "measured-block expert payload; not whole-model compression"
CONTRAST_KEYS = ("B-A", "D-C", "C-A", "D-B", "(D-C)-(B-A)")


@dataclass(frozen=True)
class Cell:
    mode: str
    hp: tuple[int, ...]

    @property
    def quantized(self):
        return tuple(b for b in BLOCKS if b not in self.hp)

    @property
    def channel_alpha(self):
        return self.quantized if self.mode == "int4_channel_alpha" else ()

    def scale_source(self, block):
        if block in self.hp:
            return "fp16_control"
        return ("research_int4_channel_alpha_side" if self.channel_alpha
                else "research_int4_side")


CELLS = MappingProxyType({"A": Cell("int4", ()), "B": Cell("int4_channel_alpha", ()),
                          "C": Cell("int4", (1, 2, 3)),
                          "D": Cell("int4_channel_alpha", (1, 2, 3))})
DIRECTIONS = MappingProxyType({"block_output_cosine": "higher", "moe_output_cosine": "higher",
                               "router_top1_agreement": "higher",
                               "router_top2_set_agreement": "higher",
                               "incoming_residual_drift": "lower",
                               "chain_exit_drift": "lower", "chain_exit_cosine": "higher",
                               "top1_flip_rate": "lower"})


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def number(value, lo=-math.inf, hi=math.inf):
    return (type(value) in (int, float) and math.isfinite(value) and lo <= value <= hi)


def sha(value, size=64):
    return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{size}}}", value) is not None


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


def paired_contrasts(values):
    require(set(values) == set(CELLS), "contrasts require exactly A/B/C/D")
    require(all(number(x) for x in values.values()), "non-finite contrast input")
    a, b, c, d = (values[x] for x in "ABCD")
    result = {"B-A": b-a, "D-C": d-c, "C-A": c-a, "D-B": d-b,
              "(D-C)-(B-A)": (d-c)-(b-a)}
    require(all(number(x) for x in result.values()), "non-finite contrast result")
    return result


def validate_resources(r):
    require(isinstance(r, dict), "missing resource accounting")
    integers = ("quantized_parameters", "logical_code_bits", "actual_code_payload_bytes",
                "code_file_bytes", "scale_payload_bytes", "scale_file_bytes",
                "fp16_expert_payload_bytes", "measured_expert_payload_bytes",
                "high_precision_nonexpert_payload_bytes", "measured_block_payload_bytes",
                "hypothetical_nibble_code_bytes", "cache_disk_bytes", "scratch_disk_bytes",
                "process_tree_peak_rss_bytes")
    for key in integers:
        require(type(r.get(key)) is int and r[key] >= 0, f"invalid resource {key}")
    q = r["quantized_parameters"]
    require(q > 0 and r["code_dtype"] == "int8", "actual side-table codes must be int8")
    require(r["logical_code_bits"] == q * 4, "logical code bits are not storage bytes")
    require(r["actual_code_payload_bytes"] == q, "int8 payload must use actual byte cost")
    require(r["hypothetical_nibble_code_bytes"] == (q+1)//2, "invalid hypothetical nibble cost")
    require(r["code_file_bytes"] >= q, "code files smaller than code payload")
    require(r["scale_file_bytes"] >= r["scale_payload_bytes"] > 0, "missing scale metadata")
    require(r["measured_expert_payload_bytes"] == q + r["scale_payload_bytes"]
            + r["fp16_expert_payload_bytes"], "invalid measured expert payload total")
    require(r["measured_block_payload_bytes"] == r["measured_expert_payload_bytes"]
            + r["high_precision_nonexpert_payload_bytes"], "total measured blocks omit nonexperts")
    require(r["cache_disk_bytes"] >= r["code_file_bytes"] + r["scale_file_bytes"],
            "cache disk excludes active tables")
    require(number(r.get("wall_seconds"), 0) and number(r.get("rss_sampling_interval_seconds"),
                                                       .001, 1), "invalid timing evidence")
    require(r["process_tree_peak_rss_bytes"] > 0, "missing sampled process-tree RSS")
    require(r.get("scope") == RESOURCE_SCOPE, "misleading resource scope")


def _metrics(m, control=False):
    require(isinstance(m, dict), "missing metrics or FP16 control")
    for key in ("block_output_cosine", "moe_output_cosine"):
        require(number(m.get(key), -1, 1+1e-9), f"invalid {key}")
    for key in ("router_top1_agreement", "router_top2_set_agreement"):
        require(number(m.get(key), 0, 1), f"invalid {key}")
    require(number(m.get("residual_stream_in", {}).get("residual_in_drift_relative_norm"), 0),
            "missing finite incoming residual drift")
    if control:
        require(m["block_output_cosine"] >= .99, "failing FP16 weight-precision control")
    bands = m.get("flip_stratification_by_reference_margin")
    require(isinstance(bands, list) and len(bands) == len(MARGIN_BANDS), "missing margin bands")
    count = flips = 0
    for row, (low, high) in zip(bands, MARGIN_BANDS, strict=True):
        require(row.get("margin_low") == low and row.get("margin_high") == high,
                "changed router-margin bands")
        n, f, rate = (row.get(k) for k in ("tokens", "top1_flips", "top1_flip_rate"))
        require(type(n) is int and type(f) is int and 0 <= f <= n, "invalid margin counts")
        require((n == 0 and rate is None) or (n > 0 and number(rate, 0, 1)
                and math.isclose(rate, f/n, abs_tol=1e-12)), "invalid margin flip rate")
        count += n
        flips += f
    require(count == TOKENS, "margin bands omit tokens")
    require(math.isclose(m["router_top1_agreement"], 1-flips/TOKENS, abs_tol=1e-12),
            "margin flips disagree with top1")


def validate_cell(payload, cell, run_id):
    """Reject malformed evidence before any same-run comparison."""
    try:
        spec = CELLS[cell]
        _finite_evidence(payload, cell)
        require(payload["protocol"] == PROTOCOL and payload["cell"] == cell,
                "wrong protocol or cell")
        require(isinstance(run_id, str) and bool(run_id) and payload["run_id"] == run_id,
                "stale run identity")
        c = payload["chain"]
        for key, value in {"tokens": TOKENS, "token_seed": SEED, "top_k": TOP_K,
                           "blocks": list(BLOCKS), "skip_fp16_control": False,
                           "hp_blocks": list(spec.hp), "int4_blocks": list(spec.quantized),
                           "channel_alpha_blocks": list(spec.channel_alpha),
                           "expert_mode": spec.mode, "ternary_blocks": [],
                           "hp_period": None, "hp_schedule_kind": "explicit"}.items():
            require(c.get(key) == value and type(c.get(key)) is type(value),
                    f"unexpected per-cell setting: {key}")
            if isinstance(value, list):
                require(all(type(v) is int for v in c[key]), f"noninteger block list: {key}")
        require([r["block"] for r in c["per_block"]] == list(BLOCKS), "invalid block order")
        for row in c["per_block"]:
            _metrics(row["expert_only"])
            _metrics(row["fp16_control"], control=True)
        for key in ("expert_only_chain_exit", "fp16_chain_exit"):
            end = c["end_of_chain"][key]
            require(number(end.get("residual_drift_relative_norm"), 0), "missing terminal drift")
            require(number(end.get("residual_cosine"), -1, 1+1e-9), "invalid terminal cosine")
        packs = c["pack_provenance"]
        require([r["block"] for r in packs] == list(BLOCKS), "invalid source coverage")
        for b, row in enumerate(packs):
            require(sha(row["pack_sha256"]) and sha(row["npy_sha256"]), "invalid input hash")
            require(row["container_versions"] == [3]
                    and all(type(v) is int for v in row["container_versions"]),
                    "pack_v2 scales require the canonical v3 container")
            require(set(row["scale_sources"]) == set(row["pack_scale_sources"]),
                    "applied and pack scale tensor identities differ")
            for field, source in (("scale_sources", spec.scale_source(b)),
                                  ("pack_scale_sources", "pack_v2")):
                experts = {k: v for k, v in row[field].items() if "moe_expert." in k}
                require(len(experts) == len(row[field]) == 3 and {k.rsplit(".", 1)[-1] for k in experts}
                        == {"gate", "down", "up"}
                        and all(k.startswith(f"block_{b:03d}.") for k in experts)
                        and set(experts.values()) == {source}, f"invalid {field} for {cell}/{b}")
        p = payload["provenance"]
        for field in ("embedding_sha256", "token_ids_sha256", "checkpoint_identity"):
            require(sha(p.get(field)), f"invalid {field}")
        ids = p["token_ids"]
        require(isinstance(ids, list) and len(ids) == TOKENS
                and all(type(i) is int and 0 <= i < 131072 for i in ids), "invalid token IDs")
        require(hashlib.sha256(struct.pack(f"<{TOKENS}q", *ids)).hexdigest()
                == p["token_ids_sha256"] == TOKEN_IDS_SHA256, "token IDs/order differ from frozen sample")
        impl = p["implementation"]
        require(sha(impl.get("commit"), 40) and impl.get("dirty") is False,
                "implementation must be a clean commit")
        require(isinstance(p["runtime"], dict) and bool(p["runtime"]), "missing runtime")
        validate_resources(payload["resources"])
        fp = payload["resources"]["fp16_expert_payload_bytes"]
        require((fp > 0) == bool(spec.hp), "resource payload contradicts HP schedule")
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f"malformed {cell} evidence: {exc}") from exc
    return payload


def _series(payload):
    result = {}
    for row in payload["chain"]["per_block"]:
        m, b = row["expert_only"], row["block"]
        for key in ("block_output_cosine", "moe_output_cosine", "router_top1_agreement",
                    "router_top2_set_agreement"):
            result[f"block_{b}/{key}"] = (m[key], DIRECTIONS[key])
        result[f"block_{b}/incoming_residual_drift"] = (
            m["residual_stream_in"]["residual_in_drift_relative_norm"], "lower")
        for i, band in enumerate(m["flip_stratification_by_reference_margin"]):
            result[f"block_{b}/margin_{i}/top1_flip_rate"] = (band["top1_flip_rate"], "lower")
    end = payload["chain"]["end_of_chain"]["expert_only_chain_exit"]
    result["chain_exit_drift"] = (end["residual_drift_relative_norm"], "lower")
    result["chain_exit_cosine"] = (end["residual_cosine"], "higher")
    return result


def analyze(cells):
    require(set(cells) == set(CELLS), "analysis requires exactly A/B/C/D")
    run_id = cells["A"].get("run_id")
    identities = []
    for cell in CELLS:
        payload = cells[cell]
        validate_cell(payload, cell, run_id)
        identities.append((payload["provenance"],
                           [(r["block"], r["pack_sha256"], r["npy_sha256"])
                            for r in payload["chain"]["pack_provenance"]]))
    require(all(x == identities[0] for x in identities), "shared provenance mismatch")
    series = {c: _series(cells[c]) for c in CELLS}
    contrasts = {}
    for metric, (_, direction) in series["A"].items():
        values = {c: series[c][metric][0] for c in CELLS}
        contrasts[metric] = {"values": values, "favorable_direction": direction,
                             "contrasts": paired_contrasts(values)
                             if all(v is not None for v in values.values()) else None}
        if contrasts[metric]["contrasts"] is None:
            contrasts[metric]["unavailable_reason"] = "empty reference-margin band"
        else:
            effects = {}
            for label, delta in contrasts[metric]["contrasts"].items():
                favorable = delta if direction == "higher" else -delta
                if label == "(D-C)-(B-A)":
                    effects[label] = ("no observed interaction" if favorable == 0 else
                                      "scaling effect becomes " + ("more" if favorable > 0 else "less")
                                      + " favorable with HP123")
                else:
                    effects[label] = "unchanged" if favorable == 0 else "improved" if favorable > 0 else "worsened"
            contrasts[metric]["observed_effects"] = effects
    return {"protocol": PROTOCOL, "run_id": run_id, "status": "complete",
            "cells": cells, "paired_contrasts": contrasts, "top1_diagnostic_band": .95,
            "interpretation": "Contrasts describe this sampled-token run. Higher cosine/agreement "
            "and lower drift/flips are favorable. Scaling, schedule and interaction effects are "
            "metric-specific; no language-quality or deployment certification. Repeated "
            "deterministic runs are reproducibility checks, not independent evidence.",
            "precision_cost_note": "A/B and C/D have different FP16 retention. Quality gains "
            "across schedules are not equal-byte compression wins; nibble packing is hypothetical.",
            "control_semantics": "fp16_roundtrip is weight-precision sensitivity, not native "
            "FP16 compute, an expert-only ceiling, or upstream parity."}
