"""Behavioral contracts for the matched four-cell experiment; no model or Git I/O."""
import copy
import importlib.util
from pathlib import Path
import re
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
FROZEN_IDS = np.sort(np.random.default_rng(20260806).choice(131072, size=8192, replace=False)).tolist()


def fixture(cell):
    hp = [1, 2, 3] if cell in "CD" else []
    quant = [0] if hp else [0, 1, 2, 3]
    alpha = cell in "BD"
    metric = {
        "block_output_cosine": .99, "moe_output_cosine": .98,
        "router_top1_agreement": 1., "router_top2_set_agreement": 1.,
        "residual_stream_in": {"residual_in_drift_relative_norm": .01},
        "flip_stratification_by_reference_margin": [
            {"margin_low": lo, "margin_high": hi, "tokens": n,
             "top1_flips": 0, "top1_flip_rate": 0. if n else None}
            for lo, hi, n in [(0., .01, 8192), (.01, .05, 0),
                              (.05, .15, 0), (.15, .5, 0), (.5, 1.01, 0)]],
    }
    packs = []
    for b in range(4):
        source = ("fp16_control" if b in hp else
                  "research_int4_channel_alpha_side" if alpha else "research_int4_side")
        keys = [f"block_{b:03d}.slot_{i:02d}.moe_expert.{role}"
                for i, role in enumerate(("gate", "down", "up"))]
        packs.append({"block": b, "pack_sha256": "a" * 64, "npy_sha256": "b" * 64,
                      "scale_sources": dict.fromkeys(keys, source),
                      "pack_scale_sources": dict.fromkeys(keys, "pack_v2"),
                      "container_versions": [3]})
    q = 12 if not hp else 3
    fp = 0 if not hp else 18
    return {
        "protocol": "grok1-int4-alpha-schedule-v1", "cell": cell, "run_id": "fresh-run",
        "provenance": {"implementation": {"commit": "a" * 40, "dirty": False},
                       "runtime": {"python": "test", "numpy": "test", "threads": "1"},
                       "embedding_sha256": "c" * 64,
                       "token_ids_sha256": "57b0e364bb25ac4bd9047b592e28ae1470af94481be073a1f963c7cd0a5ee3de",
                       "token_ids": list(FROZEN_IDS),
                       "checkpoint_identity": "e" * 64},
        "chain": {"tokens": 8192, "token_seed": 20260806, "blocks": [0, 1, 2, 3],
                  "top_k": 2, "skip_fp16_control": False, "hp_blocks": hp,
                  "int4_blocks": quant, "channel_alpha_blocks": quant if alpha else [],
                  "expert_mode": "int4_channel_alpha" if alpha else "int4",
                  "ternary_blocks": [], "hp_period": None, "hp_schedule_kind": "explicit",
                  "per_block": [{"block": b, "expert_only": copy.deepcopy(metric),
                                 "fp16_control": copy.deepcopy(metric)} for b in range(4)],
                  "end_of_chain": {key: {"residual_cosine": .99,
                                         "residual_drift_relative_norm": .03}
                                   for key in ("expert_only_chain_exit", "fp16_chain_exit")},
                  "pack_provenance": packs},
        "resources": {"quantized_parameters": q, "logical_code_bits": q * 4,
                      "code_dtype": "int8", "actual_code_payload_bytes": q,
                      "code_file_bytes": q + 128, "scale_payload_bytes": 4,
                      "scale_file_bytes": 132, "fp16_expert_payload_bytes": fp,
                      "measured_expert_payload_bytes": q + 4 + fp,
                      "high_precision_nonexpert_payload_bytes": 64,
                      "measured_block_payload_bytes": q + 4 + fp + 64,
                      "hypothetical_nibble_code_bytes": (q + 1) // 2,
                      "cache_disk_bytes": 1024, "scratch_disk_bytes": 100,
                      "wall_seconds": 1., "process_tree_peak_rss_bytes": 1024,
                      "rss_sampling_interval_seconds": .05,
                      "scope": "measured-block expert payload; not whole-model compression"},
    }


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("grok1_alpha_schedule_protocol"),
                             "four-cell protocol implementation is missing")
        import grok1_alpha_schedule_protocol
        self.p = grok1_alpha_schedule_protocol

    def test_literal_contrasts_and_missing_cell(self):
        self.assertEqual(self.p.paired_contrasts({"A": 1., "B": 3., "C": 4., "D": 9.}),
                         {"B-A": 2., "D-C": 5., "C-A": 3., "D-B": 6.,
                          "(D-C)-(B-A)": 3.})
        with self.assertRaises(ValueError):
            self.p.paired_contrasts({"A": 1., "B": 3., "C": 4.})

    def test_factors_differ_but_shared_identity_must_match(self):
        cells = {c: fixture(c) for c in "ABCD"}
        self.assertEqual(self.p.analyze(cells)["status"], "complete")
        for field in ("runtime", "embedding_sha256", "token_ids_sha256", "implementation"):
            bad = copy.deepcopy(cells)
            bad["C"]["provenance"][field] = "different"
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.p.analyze(bad)

    def test_invalid_cell_evidence_rejected(self):
        mutations = [
            lambda x: x["chain"].update(hp_blocks=[]),
            lambda x: x["chain"].update(channel_alpha_blocks=[0]),
            lambda x: x["chain"].pop("end_of_chain"),
            lambda x: x["chain"]["per_block"][0]["expert_only"].update(moe_output_cosine=float("nan")),
            lambda x: x["chain"]["per_block"][0]["fp16_control"].update(block_output_cosine=.9),
            lambda x: x["chain"]["pack_provenance"][0].update(scale_sources={}),
            lambda x: x["resources"].update(code_dtype="packed_int4"),
            lambda x: x["resources"].update(measured_expert_payload_bytes=0),
            lambda x: x["chain"].update(blocks=[False, 1, 2, 3]),
            lambda x: x["chain"]["pack_provenance"][0]["scale_sources"].update({"unexpected": "legacy_oracle"}),
        ]
        for mutation in mutations:
            payload = fixture("C")
            mutation(payload)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.p.validate_cell(payload, "C", "fresh-run")

    def test_fresh_run_and_complete_matrix_required(self):
        with self.assertRaises(ValueError):
            self.p.validate_cell(fixture("A"), "A", "new-run")
        with self.assertRaises(ValueError):
            self.p.analyze({c: fixture(c) for c in "ABC"})

    def test_token_order_and_content_are_frozen(self):
        for mutation in (lambda p: p["provenance"]["token_ids"].reverse(),
                         lambda p: p["provenance"].update(token_ids_sha256="a"*64),
                         lambda p: p["provenance"].pop("token_ids")):
            payload = fixture("A")
            mutation(payload)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.p.validate_cell(payload, "A", "fresh-run")

    def test_total_measured_blocks_include_high_precision_nonexperts(self):
        payload = fixture("A")
        payload["resources"]["measured_block_payload_bytes"] = payload["resources"]["measured_expert_payload_bytes"]
        with self.assertRaises(ValueError):
            self.p.validate_cell(payload, "A", "fresh-run")

    def test_effect_description_respects_lower_is_favorable(self):
        cells = {c: fixture(c) for c in "ABCD"}
        for c, drift in zip("ABCD", (.1, .3, .4, .9), strict=True):
            cells[c]["chain"]["end_of_chain"]["expert_only_chain_exit"]["residual_drift_relative_norm"] = drift
        result = self.p.analyze(cells)["paired_contrasts"]["chain_exit_drift"]
        self.assertEqual(result.get("observed_effects"), {
            "B-A": "worsened", "D-C": "worsened", "C-A": "worsened", "D-B": "worsened",
            "(D-C)-(B-A)": "scaling effect becomes less favorable with HP123"})

    def test_pack_v2_scale_metadata_requires_v3_container(self):
        payload = fixture("C")
        try:
            self.p.validate_cell(payload, "C", "fresh-run")
        except ValueError as exc:
            self.fail(f"v3 container with pack_v2 scale metadata was rejected: {exc}")
        payload["chain"]["pack_provenance"][0]["container_versions"] = [2]
        with self.assertRaises(ValueError):
            self.p.validate_cell(payload, "C", "fresh-run")

    def test_all_retained_numeric_evidence_must_be_finite(self):
        cases = (
            (("chain", "per_block", 0, "expert_only"), "router_margin_mean_observed",
             "A.chain.per_block[0].expert_only.router_margin_mean_observed"),
            (("chain", "per_block", 0, "fp16_control"), "router_logit_cosine",
             "A.chain.per_block[0].fp16_control.router_logit_cosine"),
            (("chain", "per_block", 0, "expert_only", "residual_stream_in"), "extra_samples",
             "A.chain.per_block[0].expert_only.residual_stream_in.extra_samples[1].value"),
            (("chain", "end_of_chain", "fp16_chain_exit"), "extra_samples",
             "A.chain.end_of_chain.fp16_chain_exit.extra_samples[1].value"),
        )
        for bad in (float("nan"), float("inf"), float("-inf")):
            for keys, field, expected_path in cases:
                cells = {c: fixture(c) for c in "ABCD"}
                target = cells["A"]
                for key in keys:
                    target = target[key]
                target[field] = [0., {"value": bad}] if field == "extra_samples" else bad
                with self.subTest(value=bad, path=expected_path):
                    with self.assertRaisesRegex(ValueError, "non-finite.*" + re.escape(expected_path)):
                        self.p.analyze(cells)

    def test_empty_margin_band_null_is_preserved(self):
        result = self.p.analyze({c: fixture(c) for c in "ABCD"})
        self.assertIsNone(result["cells"]["A"]["chain"]["per_block"][0]["expert_only"]
                          ["flip_stratification_by_reference_margin"][1]["top1_flip_rate"])
        self.assertIsNone(result["paired_contrasts"]["block_0/margin_1/top1_flip_rate"]["contrasts"])


if __name__ == "__main__":
    unittest.main()
