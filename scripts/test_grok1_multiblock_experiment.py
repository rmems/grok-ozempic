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


class _FakePack:
    """Minimal PackWeights stand-in with the public accessors the harness uses."""

    def __init__(self, tensor_type: int = 1, container_version: int = 3) -> None:
        self.pack = Path("fake.goz1")
        self._rows = {
            "block_000.slot_00.moe_expert.gate": {
                "tensor_type": tensor_type,
                "container_version": container_version,
                "numel": 8,
            }
        }
        self.scale_sources: dict[str, str] = {}
        self._scale_tag = "pack_v2"

    def tensor_names(self) -> list[str]:
        return sorted(self._rows)

    def tensor_entry(self, name: str) -> dict:
        return self._rows[name]

    def container_version(self, name: str) -> int | None:
        return self._rows[name].get("container_version")

    def scale(self, name: str) -> None:
        self.scale_sources[name] = self._scale_tag


class PackOnlyScaleTests(unittest.TestCase):
    def test_legacy_oracle_aborts(self) -> None:
        pack = _FakePack(container_version=1)
        pack._scale_tag = "legacy_oracle"
        with self.assertRaisesRegex(ForwardError, "legacy_oracle"):
            require_pack_only_scales(pack, ["block_000.slot_00.moe_expert.gate"])

    def test_pack_v2_scales_accepted_only_on_v3(self) -> None:
        sources = require_pack_only_scales(
            _FakePack(container_version=3), ["block_000.slot_00.moe_expert.gate"]
        )
        self.assertEqual(sources["block_000.slot_00.moe_expert.gate"], "pack_v2")

    def test_non_ternary_expert_aborts(self) -> None:
        with self.assertRaisesRegex(ForwardError, "must be ternary"):
            require_pack_only_scales(
                _FakePack(tensor_type=0), ["block_000.slot_00.moe_expert.gate"]
            )

    def test_non_v3_container_aborts(self) -> None:
        with self.assertRaisesRegex(ForwardError, "expected GOZ1 v3"):
            require_pack_only_scales(
                _FakePack(container_version=2), ["block_000.slot_00.moe_expert.gate"]
            )


def _expert_row(block: int, metrics: dict, *, with_fp16: bool = True) -> dict:
    """Build one per-block row; ``metrics`` holds cos/top1/top2/drift fields."""
    cos = metrics["cos"]
    resid_in_drift = metrics["resid_in_drift"]
    row = {
        "block": block,
        "expert_only": {
            "block_output_cosine": cos,
            "block_output_drift_relative_norm": metrics.get("out_drift", 0.05),
            "moe_output_cosine": metrics.get("moe", 0.9),
            "router_top1_agreement": metrics.get("top1", 1.0),
            "router_top2_set_agreement": metrics.get("top2", 1.0),
            "expert_load_js_bits": metrics.get("js", 0.0),
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
                _expert_row(0, {"cos": 0.963, "resid_in_drift": 0.0}),
                _expert_row(1, {"cos": 0.955, "resid_in_drift": 0.04}),
                _expert_row(2, {"cos": 0.950, "resid_in_drift": 0.07}),
                _expert_row(3, {"cos": 0.948, "resid_in_drift": 0.09}),
            ],
            "top_k": 2,
            "end_of_chain": {
                "expert_only_chain_exit": {
                    "residual_cosine": 0.99,
                    "residual_drift_relative_norm": 0.10,
                }
            },
        }
        self.assertEqual(decide(chain)["decision"], 1)

    def test_collapse_is_option_3(self) -> None:
        chain = {
            "per_block": [
                _expert_row(0, {"cos": 0.96, "resid_in_drift": 0.0}),
                _expert_row(1, {"cos": 0.80, "resid_in_drift": 0.2, "top1": 0.85}),
                _expert_row(2, {"cos": 0.60, "resid_in_drift": 0.45, "top1": 0.70}),
                _expert_row(3, {"cos": 0.40, "resid_in_drift": 0.70, "top1": 0.50}),
            ],
            "top_k": 2,
            "end_of_chain": {
                "expert_only_chain_exit": {
                    "residual_cosine": 0.5,
                    "residual_drift_relative_norm": 0.75,
                }
            },
        }
        self.assertEqual(decide(chain)["decision"], 3)

    def test_fp16_failure_is_option_4(self) -> None:
        rows = [_expert_row(0, {"cos": 0.96, "resid_in_drift": 0.0})]
        rows[0]["fp16_control"]["block_output_cosine"] = 0.5
        self.assertEqual(decide({"per_block": rows})["decision"], 4)

    def test_missing_fp16_is_option_4(self) -> None:
        chain = {
            "per_block": [
                _expert_row(0, {"cos": 0.96, "resid_in_drift": 0.0}, with_fp16=False),
            ]
        }
        self.assertEqual(decide(chain)["decision"], 4)

    def test_intermediate_is_option_2(self) -> None:
        chain = {
            "per_block": [
                _expert_row(0, {"cos": 0.96, "resid_in_drift": 0.0}),
                _expert_row(1, {"cos": 0.91, "resid_in_drift": 0.10}),
                _expert_row(2, {"cos": 0.89, "resid_in_drift": 0.16}),
                _expert_row(3, {"cos": 0.87, "resid_in_drift": 0.22}),
            ],
            "top_k": 2,
            "end_of_chain": {
                "expert_only_chain_exit": {
                    "residual_cosine": 0.9,
                    "residual_drift_relative_norm": 0.28,
                }
            },
        }
        self.assertEqual(decide(chain)["decision"], 2)

    def test_final_hop_runaway_is_option_3(self) -> None:
        """Greptile/Cubic: chain-exit drift must enter compounding, not only resid_in."""
        chain = {
            "per_block": [
                _expert_row(0, {"cos": 0.96, "resid_in_drift": 0.0}),
                _expert_row(1, {"cos": 0.95, "resid_in_drift": 0.05}),
                _expert_row(2, {"cos": 0.94, "resid_in_drift": 0.08}),
                _expert_row(3, {"cos": 0.93, "resid_in_drift": 0.10}),
            ],
            "top_k": 2,
            "end_of_chain": {
                "expert_only_chain_exit": {
                    "residual_cosine": 0.7,
                    "residual_drift_relative_norm": 0.55,
                }
            },
        }
        d = decide(chain)
        self.assertEqual(d["decision"], 3)
        self.assertEqual(d["compounding"], "superlinear_or_runaway")

    def test_skip_fp16_flag_forces_option_4(self) -> None:
        chain = {
            "per_block": [_expert_row(0, {"cos": 0.96, "resid_in_drift": 0.0})],
            "top_k": 2,
            "skip_fp16_control": True,
        }
        # even with fp16_control present, explicit skip forces option 4
        self.assertEqual(decide(chain)["decision"], 4)


class ReportTests(unittest.TestCase):
    def test_results_md_cites_agent_and_model(self) -> None:
        option_viable = 1  # #68 decision option index (not a credential)
        payload = {
            "provenance": {"implementation": {"commit": "abc", "dirty": False}},
            "decision": {
                "decision": option_viable,
                "decision_text": "viable",
                "rationale": ["ok"],
            },
            "chain": {
                "tokens": 8,
                "token_seed": 20260806,
                "per_block": [_expert_row(0, {"cos": 0.96, "resid_in_drift": 0.0})],
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
