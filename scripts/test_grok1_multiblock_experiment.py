#!/usr/bin/env python3
"""Unit tests for the #68 multi-block residual fidelity harness."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grok1_block_forward import ForwardError  # noqa: E402
from grok1_multiblock_experiment import (  # noqa: E402
    decide,
    parse_blocks,
    require_pack_only_scales,
    residual_stream_metrics,
    write_results_md,
)


class ParseBlocksTests(unittest.TestCase):
    def test_parses_contiguous_chain(self) -> None:
        self.assertEqual(parse_blocks("0,1,2,3"), [0, 1, 2, 3])

    def test_rejects_gaps(self) -> None:
        with self.assertRaisesRegex(ForwardError, "contiguous"):
            parse_blocks("0,1,3")

    def test_rejects_non_ascending(self) -> None:
        with self.assertRaisesRegex(ForwardError, "ascending"):
            parse_blocks("2,1,0")

    def test_rejects_empty(self) -> None:
        with self.assertRaisesRegex(ForwardError, "at least one"):
            parse_blocks("")


class ResidualMetricsTests(unittest.TestCase):
    def test_identical_streams_have_zero_drift(self) -> None:
        x = np.random.default_rng(0).standard_normal((8, 4)).astype(np.float32)
        m = residual_stream_metrics(x, x.copy())
        self.assertAlmostEqual(m["residual_in_cosine"], 1.0, places=6)
        self.assertAlmostEqual(m["residual_in_drift_relative_norm"], 0.0, places=6)

    def test_orthogonal_streams_are_not_identity(self) -> None:
        a = np.zeros((4, 2), dtype=np.float32)
        a[:, 0] = 1.0
        b = np.zeros((4, 2), dtype=np.float32)
        b[:, 1] = 1.0
        m = residual_stream_metrics(a, b)
        self.assertAlmostEqual(m["residual_in_cosine"], 0.0, places=6)
        self.assertGreater(m["residual_in_drift_relative_norm"], 0.9)


class PackOnlyScaleTests(unittest.TestCase):
    def test_legacy_oracle_aborts(self) -> None:
        class _FakePack:
            def __init__(self) -> None:
                self.pack = Path("fake.goz1")
                self._index = {
                    "block_000.slot_00.moe_expert.gate": {
                        "tensor_type": 1,
                        "container_version": 1,
                        "numel": 8,
                    }
                }
                self.scale_sources: dict[str, str] = {}

            def scale(self, name: str) -> None:
                self.scale_sources[name] = "legacy_oracle"

        with self.assertRaisesRegex(ForwardError, "legacy_oracle"):
            require_pack_only_scales(_FakePack(), ["block_000.slot_00.moe_expert.gate"])

    def test_pack_v2_scales_accepted_only_on_v3(self) -> None:
        class _FakePack:
            def __init__(self) -> None:
                self.pack = Path("fake.goz1")
                self._index = {
                    "block_000.slot_00.moe_expert.gate": {
                        "tensor_type": 1,
                        "container_version": 3,
                        "numel": 8,
                    }
                }
                self.scale_sources: dict[str, str] = {}

            def scale(self, name: str) -> None:
                self.scale_sources[name] = "pack_v2"

        sources = require_pack_only_scales(
            _FakePack(), ["block_000.slot_00.moe_expert.gate"]
        )
        self.assertEqual(sources["block_000.slot_00.moe_expert.gate"], "pack_v2")

    def test_non_ternary_expert_aborts(self) -> None:
        class _FakePack:
            def __init__(self) -> None:
                self.pack = Path("fake.goz1")
                self._index = {
                    "block_000.slot_00.moe_expert.gate": {
                        "tensor_type": 0,
                        "container_version": 3,
                        "numel": 8,
                    }
                }
                self.scale_sources: dict[str, str] = {}

            def scale(self, name: str) -> None:
                self.scale_sources[name] = "pack_v2"

        with self.assertRaisesRegex(ForwardError, "must be ternary"):
            require_pack_only_scales(_FakePack(), ["block_000.slot_00.moe_expert.gate"])


def _expert_row(
    block: int,
    *,
    cos: float,
    resid_in_drift: float,
    top1: float = 1.0,
    top2: float = 1.0,
    js: float = 0.0,
    moe: float = 0.9,
    out_drift: float = 0.05,
    with_fp16: bool = True,
) -> dict:
    row = {
        "block": block,
        "expert_only": {
            "block_output_cosine": cos,
            "block_output_drift_relative_norm": out_drift,
            "moe_output_cosine": moe,
            "router_top1_agreement": top1,
            "router_top2_set_agreement": top2,
            "expert_load_js_bits": js,
            "residual_stream_in": {
                "residual_in_cosine": max(0.0, 1.0 - resid_in_drift),
                "residual_in_drift_relative_norm": resid_in_drift,
            },
        },
        "fp16_control": None,
    }
    if with_fp16:
        row["fp16_control"] = {
            "block_output_cosine": 0.9999,
            "router_top1_agreement": 0.999,
            "router_top2_set_agreement": 0.999,
        }
    return row


class DecideTests(unittest.TestCase):
    def test_bounded_chain_is_option_1(self) -> None:
        chain = {
            "per_block": [
                _expert_row(0, cos=0.963, resid_in_drift=0.0),
                _expert_row(1, cos=0.955, resid_in_drift=0.04),
                _expert_row(2, cos=0.950, resid_in_drift=0.07),
                _expert_row(3, cos=0.948, resid_in_drift=0.09),
            ],
            "end_of_chain": {
                "expert_only_end_residual_in": {
                    "residual_in_cosine": 0.99,
                    "residual_in_drift_relative_norm": 0.10,
                }
            },
        }
        self.assertEqual(decide(chain)["decision"], 1)

    def test_collapse_is_option_3(self) -> None:
        chain = {
            "per_block": [
                _expert_row(0, cos=0.96, resid_in_drift=0.0),
                _expert_row(1, cos=0.80, resid_in_drift=0.2, top1=0.85),
                _expert_row(2, cos=0.60, resid_in_drift=0.45, top1=0.70),
                _expert_row(3, cos=0.40, resid_in_drift=0.70, top1=0.50),
            ],
            "end_of_chain": {
                "expert_only_end_residual_in": {
                    "residual_in_cosine": 0.5,
                    "residual_in_drift_relative_norm": 0.75,
                }
            },
        }
        self.assertEqual(decide(chain)["decision"], 3)

    def test_fp16_failure_is_option_4(self) -> None:
        rows = [_expert_row(0, cos=0.96, resid_in_drift=0.0)]
        rows[0]["fp16_control"]["block_output_cosine"] = 0.5
        self.assertEqual(decide({"per_block": rows})["decision"], 4)

    def test_missing_fp16_is_option_4(self) -> None:
        chain = {
            "per_block": [
                _expert_row(0, cos=0.96, resid_in_drift=0.0, with_fp16=False),
            ]
        }
        self.assertEqual(decide(chain)["decision"], 4)

    def test_intermediate_is_option_2(self) -> None:
        chain = {
            "per_block": [
                _expert_row(0, cos=0.96, resid_in_drift=0.0),
                _expert_row(1, cos=0.91, resid_in_drift=0.10),
                _expert_row(2, cos=0.89, resid_in_drift=0.16),
                _expert_row(3, cos=0.87, resid_in_drift=0.22),
            ],
            "end_of_chain": {
                "expert_only_end_residual_in": {
                    "residual_in_cosine": 0.9,
                    "residual_in_drift_relative_norm": 0.28,
                }
            },
        }
        self.assertEqual(decide(chain)["decision"], 2)


class ReportTests(unittest.TestCase):
    def test_results_md_cites_agent_and_model(self) -> None:
        payload = {
            "provenance": {"implementation": {"commit": "abc", "dirty": False}},
            "decision": {
                "decision": 1,  # nosec B105 — option index, not a secret
                "decision_text": "viable",
                "rationale": ["ok"],
            },
            "chain": {
                "tokens": 8,
                "token_seed": 1,
                "per_block": [_expert_row(0, cos=0.96, resid_in_drift=0.0)],
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.md"
            write_results_md(path, payload)
            text = path.read_text()
        self.assertIn("Grok Build: Grok 4.5", text)
        self.assertIn("grok-4.5", text)
        self.assertIn("#64", text)
        self.assertIn("Option 1", text)


if __name__ == "__main__":
    unittest.main()
