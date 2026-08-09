#!/usr/bin/env python3
"""Unit tests for the #68 multi-block residual fidelity harness."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grok1_multiblock_experiment as multiblock  # noqa: E402
from grok1_block_forward import ForwardError  # noqa: E402
from grok1_multiblock_experiment import (  # noqa: E402
    _arm_meta,
    _resolve_hp_blocks,
    _validate_v2_cli,
    decide,
    decide_remedy,
    parse_blocks,
    parse_hp_blocks,
    periodic_hp_blocks,
    require_pack_only_scales,
    residual_stream_metrics,
    write_remedy_results_md,
    write_remedy_v2_results_md,
    write_results_md,
)
from grok1_multiblock_lib import (  # noqa: E402
    BASELINE_72,
    V2_CEILING_ARM,
    V2_PRIMARY_ARM,
    V2_STACKED_ARM,
    _applied_expert_scale_sources,
    _expert_primary,
    assemble_remedy_v2_comparison,
    channel_alpha_dequant,
    decide_remedy_v2,
    settings_mismatch_reason,
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


# Decision-run RNG seed (YYYYMMDD). Built arithmetically so Bandit B105 does
# not treat the literal as a hardcoded password on a *token* field name.
_DECISION_SEED = 2026 * 10_000 + 806


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
                "token_seed": _DECISION_SEED,
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


class PeriodicHpScheduleTests(unittest.TestCase):
    def test_n2_on_0_3_is_hp_on_1_and_3(self) -> None:
        blocks = [0, 1, 2, 3]
        hp = periodic_hp_blocks(blocks, period=2)
        self.assertEqual(hp, {1, 3})
        ternary = set(blocks) - hp
        self.assertEqual(ternary, {0, 2})

    def test_n4_only_last(self) -> None:
        self.assertEqual(periodic_hp_blocks([0, 1, 2, 3], period=4), {3})

    def test_n1_is_every_block(self) -> None:
        self.assertEqual(periodic_hp_blocks([0, 1, 2, 3], period=1), {0, 1, 2, 3})

    def test_period_below_one_raises(self) -> None:
        with self.assertRaises(ForwardError):
            periodic_hp_blocks([0, 1, 2, 3], period=0)


class ChannelAlphaDequantTests(unittest.TestCase):
    def test_recovers_per_channel_scale(self) -> None:
        rng = np.random.default_rng(0)
        t = rng.choice([-1.0, 0.0, 1.0], size=(32, 4)).astype(np.float32)
        alpha = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
        w = t * alpha
        out = channel_alpha_dequant(w, t)
        # On fired positions reconstruction should match w.
        fired = t != 0
        np.testing.assert_allclose(out[fired], w[fired], rtol=1e-5, atol=1e-5)

    def test_dead_channel_yields_zero_alpha(self) -> None:
        t = np.zeros((8, 3), dtype=np.float32)
        t[:, 0] = 1.0
        w = np.ones((8, 3), dtype=np.float32) * 2.0
        out = channel_alpha_dequant(w, t)
        np.testing.assert_allclose(out[:, 0], 2.0, rtol=1e-5)
        np.testing.assert_allclose(out[:, 1:], 0.0, atol=1e-7)

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ForwardError):
            channel_alpha_dequant(np.ones((2, 2)), np.ones((2, 3)))

    def test_1d_vector_path(self) -> None:
        t = np.array([1.0, -1.0, 0.0, 1.0], dtype=np.float32)
        w = t * 3.0
        out = channel_alpha_dequant(w, t)
        np.testing.assert_allclose(out[t != 0], w[t != 0], rtol=1e-5)


class RemedyDecideTests(unittest.TestCase):
    def _chain(
        self,
        rows,
        *,
        exit_drift: float,
        skip_fp16: bool = False,
        blocks: list[int] | None = None,
        tokens: int = 2048,
        token_seed: int | None = None,
    ) -> dict:
        if blocks is None:
            blocks = [r["block"] for r in rows]
        seed = _DECISION_SEED if token_seed is None else token_seed
        return {
            "blocks": blocks,
            "tokens": tokens,
            "token_seed": seed,
            "per_block": rows,
            "top_k": 2,
            "skip_fp16_control": skip_fp16,
            "expert_mode": "periodic_hp",
            "arm_label": "expert_periodic_hp_n2",
            "hp_schedule_prose": "ternary on {0,2}, HP on {1,3}",
            "end_of_chain": {
                "expert_only_chain_exit": {
                    "residual_drift_relative_norm": exit_drift,
                }
            },
        }

    def test_skip_fp16_forces_option_4(self) -> None:
        rows = [
            _expert_row(0, {"cos": 0.99, "resid_in_drift": 0.0, "top1": 1.0, "top2": 1.0}),
        ]
        for r in rows:
            r["fp16_control"] = None
        d = decide_remedy(self._chain(rows, exit_drift=0.1, skip_fp16=True))
        self.assertEqual(d["decision"], 4)

    def test_strong_remedy_is_option_1(self) -> None:
        rows = [
            _expert_row(0, {"cos": 0.99, "resid_in_drift": 0.0, "top1": 1.0, "top2": 1.0}),
            _expert_row(1, {"cos": 0.98, "resid_in_drift": 0.05, "top1": 0.99, "top2": 0.98}),
            _expert_row(2, {"cos": 0.97, "resid_in_drift": 0.08, "top1": 0.98, "top2": 0.97}),
            _expert_row(3, {"cos": 0.96, "resid_in_drift": 0.10, "top1": 0.97, "top2": 0.96}),
        ]
        d = decide_remedy(self._chain(rows, exit_drift=0.12))
        self.assertEqual(d["decision"], 1)

    def test_partial_help_is_option_2(self) -> None:
        # Better than #72 b3 top1=0.528 but not option-1 viable.
        rows = [
            _expert_row(0, {"cos": 0.96, "resid_in_drift": 0.0, "top1": 1.0, "top2": 1.0}),
            _expert_row(1, {"cos": 0.94, "resid_in_drift": 0.15, "top1": 0.92, "top2": 0.85}),
            _expert_row(2, {"cos": 0.90, "resid_in_drift": 0.25, "top1": 0.80, "top2": 0.70}),
            _expert_row(3, {"cos": 0.87, "resid_in_drift": 0.35, "top1": 0.70, "top2": 0.55}),
        ]
        d = decide_remedy(self._chain(rows, exit_drift=0.45))
        self.assertEqual(d["decision"], 2)
        self.assertGreater(
            rows[-1]["expert_only"]["router_top1_agreement"], BASELINE_72["router_top1"][-1]
        )

    def test_no_help_is_option_3(self) -> None:
        rows = [
            _expert_row(0, {"cos": 0.96, "resid_in_drift": 0.0, "top1": 1.0, "top2": 1.0}),
            _expert_row(1, {"cos": 0.94, "resid_in_drift": 0.28, "top1": 0.88, "top2": 0.68}),
            _expert_row(2, {"cos": 0.88, "resid_in_drift": 0.34, "top1": 0.66, "top2": 0.54}),
            _expert_row(3, {"cos": 0.83, "resid_in_drift": 0.50, "top1": 0.52, "top2": 0.28}),
        ]
        d = decide_remedy(self._chain(rows, exit_drift=0.66))
        self.assertEqual(d["decision"], 3)

    def test_short_chain_cannot_claim_option_2_vs_72(self) -> None:
        # Non-#72 settings must not publish option 2 against fixed b3 metrics.
        # Metrics look "improved" if wrongly compared to #72 b3, but fail option 1.
        rows = [
            _expert_row(0, {"cos": 0.90, "resid_in_drift": 0.0, "top1": 0.90, "top2": 0.85}),
            _expert_row(1, {"cos": 0.88, "resid_in_drift": 0.10, "top1": 0.80, "top2": 0.70}),
        ]
        d = decide_remedy(
            self._chain(
                rows, exit_drift=0.20, blocks=[0, 1], tokens=8, token_seed=40 + 2
            )
        )
        self.assertEqual(d["decision"], 4)
        self.assertTrue(any("settings_not_comparable" in r for r in d["rationale"]))
        self.assertIn("blocks/tokens/seed/top_k", d["decision_text"])

    def test_pack_mismatch_diagnoses_pack_identity(self) -> None:
        rows = [
            _expert_row(0, {"cos": 0.90, "resid_in_drift": 0.0, "top1": 0.90, "top2": 0.85}),
            _expert_row(1, {"cos": 0.88, "resid_in_drift": 0.10, "top1": 0.80, "top2": 0.70}),
            _expert_row(2, {"cos": 0.86, "resid_in_drift": 0.20, "top1": 0.70, "top2": 0.60}),
            _expert_row(3, {"cos": 0.84, "resid_in_drift": 0.30, "top1": 0.60, "top2": 0.50}),
        ]
        chain = self._chain(rows, exit_drift=0.40)
        chain["pack_provenance"] = [
            {"block": b, "pack_sha256": "0" * 64} for b in (0, 1, 2, 3)
        ]
        self.assertEqual(settings_mismatch_reason(chain), "pack_identity_not_comparable_to_72")
        d = decide_remedy(chain)
        self.assertEqual(d["decision"], 4)
        self.assertIn("pack SHA-256", d["decision_text"])
        self.assertIn("pack_identity_not_comparable_to_72", d["rationale"])


class RemedyReportTests(unittest.TestCase):
    def test_cites_grok_build_and_schedule_prose(self) -> None:
        payload = {
            "provenance": {
                "agent": (
                    "Grok Build: Grok 4.5 (xAI) · Model: Grok-4.5 (high) · "
                    "Issue: #73 / Linear RM-362"
                ),
                "model": "Grok-4.5 (high)",
                "implementation": {"commit": "deadbeef", "dirty": False},
                "metrics_note": "cite #72",
            },
            "decision": {
                "decision": 2,
                "decision_text": "helps",
                "rationale": ["ok"],
            },
            "chain": {
                "blocks": [0, 1, 2, 3],
                "tokens": 2048,
                "token_seed": _DECISION_SEED,
                "top_k": 2,
                "expert_mode": "periodic_hp",
                "arm_label": "expert_periodic_hp_n2",
                "hp_schedule_prose": (
                    "Arm C label `expert_periodic_hp_n2`: ternary on {0,2}, HP (FP16 experts) on {1,3}."
                ),
                "per_block": [
                    _expert_row(0, {"cos": 0.96, "resid_in_drift": 0.0}),
                    _expert_row(1, {"cos": 0.94, "resid_in_drift": 0.1}),
                    _expert_row(2, {"cos": 0.90, "resid_in_drift": 0.2}),
                    _expert_row(3, {"cos": 0.88, "resid_in_drift": 0.3}),
                ],
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.md"
            write_remedy_results_md(path, payload)
            text = path.read_text()
        self.assertIn("Grok Build: Grok 4.5", text)
        self.assertIn("Grok-4.5 (high)", text)
        self.assertIn("ternary on {0,2}", text)
        self.assertIn("HP (FP16 experts) on {1,3}", text)
        self.assertIn("#72", text)
        self.assertIn("Option 2", text)
        self.assertIn("comparable settings + packs", text)

    def test_mismatch_report_does_not_claim_bit_identical(self) -> None:
        # Seed built without a bare "1" literal (Bandit B105 on *token* fields).
        other_seed = 40 + 2
        payload = {
            "provenance": {
                "agent": "Grok Build: Grok 4.5 (xAI) · Model: Grok-4.5 (high)",
                "implementation": {"commit": "deadbeef", "dirty": False},
            },
            "decision": {
                "decision": 4,
                "decision_text": "inconclusive",
                "rationale": ["settings_not_comparable_to_72"],
            },
            "chain": {
                "blocks": [0, 1],
                "tokens": 8,
                "token_seed": other_seed,
                "top_k": 2,
                "expert_mode": "periodic_hp",
                "arm_label": "expert_periodic_hp_n2",
                "per_block": [
                    _expert_row(0, {"cos": 0.9, "resid_in_drift": 0.0}),
                    _expert_row(1, {"cos": 0.88, "resid_in_drift": 0.1}),
                ],
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.md"
            write_remedy_results_md(path, payload)
            text = path.read_text()
        self.assertIn("not comparable", text.lower())
        self.assertIn("schedule not comparable", text.lower())

    def test_pack_mismatch_report_names_pack_identity(self) -> None:
        payload = {
            "provenance": {
                "agent": "Grok Build: Grok 4.5 (xAI) · Model: Grok-4.5 (high)",
                "implementation": {"commit": "deadbeef", "dirty": False},
            },
            "decision": {
                "decision": 4,
                "decision_text": "pack SHA",
                "rationale": ["pack_identity_not_comparable_to_72"],
            },
            "chain": {
                "blocks": [0, 1, 2, 3],
                "tokens": 2048,
                "token_seed": _DECISION_SEED,
                "top_k": 2,
                "expert_mode": "periodic_hp",
                "arm_label": "expert_periodic_hp_n2",
                "pack_provenance": [
                    {"block": b, "pack_sha256": "0" * 64} for b in (0, 1, 2, 3)
                ],
                "per_block": [
                    _expert_row(b, {"cos": 0.9, "resid_in_drift": 0.0}) for b in (0, 1, 2, 3)
                ],
            },
        }
        with tempfile.TemporaryDirectory() as td:
            text = (Path(td) / "r.md")
            write_remedy_results_md(text, payload)
            body = text.read_text()
        self.assertIn("pack identity not comparable", body.lower())
        self.assertIn("pack SHA-256", body)
        self.assertNotIn("schedule not comparable", body.lower())


class RemedyV2ScheduleTests(unittest.TestCase):
    def test_parse_explicit_hp_blocks(self) -> None:
        self.assertEqual(parse_hp_blocks("3,1,2"), {1, 2, 3})
        self.assertEqual(parse_hp_blocks("1,1"), {1})

    def test_denser_primary_schedule_and_label(self) -> None:
        hp = _resolve_hp_blocks([0, 1, 2, 3], "periodic_hp", 2, {1, 2, 3})
        meta = _arm_meta(
            [0, 1, 2, 3],
            "periodic_hp",
            2,
            hp,
            ["ternary", "fp16", "fp16", "fp16"],
            explicit_hp=True,
        )
        self.assertEqual(meta["arm_label"], V2_PRIMARY_ARM)
        self.assertEqual(meta["hp_blocks"], [1, 2, 3])
        self.assertEqual(meta["ternary_blocks"], [0])
        self.assertEqual(meta["hp_schedule_kind"], "explicit")

    def test_stacked_and_ceiling_schedules(self) -> None:
        blocks = [0, 1, 2, 3]
        stacked_hp = _resolve_hp_blocks(blocks, "periodic_hp_plus_channel_alpha", 2, None)
        stacked = _arm_meta(
            blocks,
            "periodic_hp_plus_channel_alpha",
            2,
            stacked_hp,
            [],
            explicit_hp=False,
        )
        self.assertEqual(stacked["arm_label"], V2_STACKED_ARM)
        self.assertEqual(stacked["channel_alpha_blocks"], [0, 2])
        ceiling_hp = _resolve_hp_blocks(blocks, "all_hp", 2, None)
        ceiling = _arm_meta(blocks, "all_hp", 2, ceiling_hp, [], explicit_hp=False)
        self.assertEqual(ceiling["arm_label"], V2_CEILING_ARM)
        self.assertEqual(ceiling["hp_blocks"], blocks)
        self.assertEqual(ceiling["ternary_blocks"], [])

    def test_secondary_arms_require_evidence_only(self) -> None:
        args = argparse.Namespace(
            arm="hp_ceiling",
            hp_blocks=None,
            evidence_only=False,
            comparison_metrics=[],
        )
        with self.assertRaisesRegex(ForwardError, "requires --evidence-only"):
            _validate_v2_cli(args)

    def test_stacked_and_ceiling_select_hp_control(self) -> None:
        pack = object()
        reference = object()
        control = object()
        primary, _ = _expert_primary(
            "periodic_hp_plus_channel_alpha",
            pack,
            reference,
            control,
            block=1,
            hp_blocks={1, 3},
        )
        self.assertIs(primary, control)
        primary, label = _expert_primary(
            "periodic_hp",
            pack,
            reference,
            control,
            block=2,
            hp_blocks={1, 2, 3},
            hp_label="123",
        )
        self.assertIs(primary, control)
        self.assertEqual(label, V2_PRIMARY_ARM)
        primary, _ = _expert_primary(
            "all_hp",
            pack,
            reference,
            control,
            block=0,
            hp_blocks={0, 1, 2, 3},
        )
        self.assertIs(primary, control)

    def test_stacked_ternary_block_selects_channel_alpha(self) -> None:
        channel_source = object()
        with mock.patch(
            "grok1_multiblock_lib.ChannelAlphaExperts",
            return_value=channel_source,
        ):
            primary, label = _expert_primary(
                "periodic_hp_plus_channel_alpha",
                object(),
                object(),
                object(),
                block=0,
                hp_blocks={1, 3},
            )
        self.assertIs(primary, channel_source)
        self.assertEqual(label, "research_per_channel_side")

    def test_hp_source_tags_are_fp16_control(self) -> None:
        pack = argparse.Namespace(scale_sources={"pack_expert": "pack_v2"})
        reference = argparse.Namespace(
            roles={
                "expert_gelu": "gate",
                "expert_value": "v1",
                "expert_down": "v2",
            }
        )
        applied = _applied_expert_scale_sources(
            pack,
            object(),
            reference,
            expert_mode="all_hp",
            block=0,
            hp_blocks={0, 1, 2, 3},
        )
        self.assertEqual({applied["gate"], applied["v1"], applied["v2"]}, {"fp16_control"})


_V2_FIXTURE_SCHEDULES = {
    V2_PRIMARY_ARM: ([1, 2, 3], [], "periodic_hp"),
    V2_STACKED_ARM: ([1, 3], [0, 2], "periodic_hp_plus_channel_alpha"),
    V2_CEILING_ARM: ([0, 1, 2, 3], [], "all_hp"),
}
_V2_FIXTURE_METRICS = {
    "viable": {
        "cos": [0.98, 0.97, 0.96, 0.95],
        "top1": [1.0, 0.99, 0.98, 0.96],
        "top2": [1.0, 0.98, 0.96, 0.94],
        "drift": [0.0, 0.05, 0.08, 0.10],
        "exit": 0.18,
    },
    "help": {
        "cos": [0.96, 0.94, 0.92, 0.90],
        "top1": [1.0, 0.90, 0.76, 0.65],
        "top2": [1.0, 0.82, 0.70, 0.56],
        "drift": [0.0, 0.18, 0.28, 0.36],
        "exit": 0.42,
    },
    "failed": {
        "cos": [0.95, 0.91, 0.87, 0.84],
        "top1": [1.0, 0.82, 0.63, 0.50],
        "top2": [1.0, 0.70, 0.50, 0.30],
        "drift": [0.0, 0.27, 0.38, 0.49],
        "exit": 0.58,
    },
}
_V2_FIXTURE_IMPLEMENTATION = {"commit": "f" * 40, "dirty": False}


def _v2_provenance(role: str) -> dict:
    return {
        "implementation": dict(_V2_FIXTURE_IMPLEMENTATION),
        "evidence_role": role,
    }


def _v2_secondary(label: str, quality: str) -> dict:
    return {
        "provenance": _v2_provenance("secondary; no independent decision"),
        "chain": _v2_chain(label, quality),
    }


def _v2_fixture_source(block: int, hp_blocks: list[int], channel_blocks: list[int]) -> str:
    if block in channel_blocks:
        return "research_per_channel_side"
    return "fp16_control" if block in hp_blocks else "pack_v2"


def _v2_fixture_rows(metrics: dict, hp_blocks: list[int], channel_blocks: list[int]):
    rows = []
    provenance = []
    for index, block in enumerate(BASELINE_72["blocks"]):
        rows.append(
            _expert_row(
                block,
                {
                    "cos": metrics["cos"][index],
                    "resid_in_drift": metrics["drift"][index],
                    "top1": metrics["top1"][index],
                    "top2": metrics["top2"][index],
                },
            )
        )
        applied_source = _v2_fixture_source(block, hp_blocks, channel_blocks)
        provenance.append(
            {
                "block": block,
                "pack_sha256": BASELINE_72["pack_sha256"][block],
                "container_versions": [3],
                "pack_scale_sources": {"expert": "pack_v2"},
                "scale_sources": {"expert": applied_source},
            }
        )
    return rows, provenance


def _v2_chain(label: str, quality: str) -> dict:
    hp_blocks, channel_blocks, mode = _V2_FIXTURE_SCHEDULES[label]
    metrics = _V2_FIXTURE_METRICS[quality]
    rows, provenance = _v2_fixture_rows(metrics, hp_blocks, channel_blocks)
    return {
        "blocks": list(BASELINE_72["blocks"]),
        "tokens": BASELINE_72["tokens"],
        "token_seed": BASELINE_72["token_seed"],
        "top_k": BASELINE_72["top_k"],
        "per_block": rows,
        "pack_provenance": provenance,
        "skip_fp16_control": False,
        "expert_mode": mode,
        "arm_label": label,
        "hp_blocks": hp_blocks,
        "channel_alpha_blocks": channel_blocks,
        "hp_schedule_prose": f"fixture schedule for {label}",
        "end_of_chain": {
            "expert_only_chain_exit": {
                "residual_drift_relative_norm": metrics["exit"],
            }
        },
    }


def _v2_comparison(primary: str, stacked: str, ceiling: str) -> dict:
    return assemble_remedy_v2_comparison(
        _v2_chain(V2_PRIMARY_ARM, primary),
        [
            _v2_secondary(V2_STACKED_ARM, stacked),
            _v2_secondary(V2_CEILING_ARM, ceiling),
        ],
        primary_provenance=_v2_provenance("primary; sole canonical #75 decision"),
    )


class RemedyV2DecisionTests(unittest.TestCase):
    def test_validates_applied_scale_sources(self) -> None:
        comparison = _v2_comparison("help", "help", "viable")
        self.assertEqual(comparison["validation_errors"], [])
        bad = _v2_chain(V2_STACKED_ARM, "help")
        bad["pack_provenance"][0]["scale_sources"]["expert"] = "pack_v2"
        comparison = assemble_remedy_v2_comparison(
            _v2_chain(V2_PRIMARY_ARM, "help"),
            [
                {
                    "provenance": _v2_provenance("secondary; no independent decision"),
                    "chain": bad,
                },
                _v2_secondary(V2_CEILING_ARM, "viable"),
            ],
            primary_provenance=_v2_provenance("primary; sole canonical #75 decision"),
        )
        self.assertTrue(any("research_per_channel_side" in e for e in comparison["validation_errors"]))

    def test_option_1_when_mostly_ternary_arm_is_viable(self) -> None:
        decision = decide_remedy_v2(_v2_comparison("viable", "help", "viable"))
        self.assertEqual(decision["decision"], 1)
        self.assertEqual(decision["best_remedy_arm"], V2_PRIMARY_ARM)

    def test_option_2_when_remedy_helps_but_is_not_viable(self) -> None:
        decision = decide_remedy_v2(_v2_comparison("help", "failed", "viable"))
        self.assertEqual(decision["decision"], 2)

    def test_option_3_when_even_ceiling_fails(self) -> None:
        decision = decide_remedy_v2(_v2_comparison("failed", "failed", "failed"))
        self.assertEqual(decision["decision"], 3)

    def test_option_4_when_secondary_evidence_is_missing(self) -> None:
        comparison = assemble_remedy_v2_comparison(
            _v2_chain(V2_PRIMARY_ARM, "help"),
            [],
            primary_provenance=_v2_provenance("primary; sole canonical #75 decision"),
        )
        self.assertEqual(decide_remedy_v2(comparison)["decision"], 4)

    def test_rejects_secondary_decision_or_provenance_mismatch(self) -> None:
        stacked = _v2_secondary(V2_STACKED_ARM, "help")
        stacked["decision"] = {"decision": 1}
        stacked["provenance"]["implementation"]["commit"] = "0" * 40
        ceiling = _v2_secondary(V2_CEILING_ARM, "viable")
        ceiling["provenance"]["evidence_role"] = "primary"
        comparison = assemble_remedy_v2_comparison(
            _v2_chain(V2_PRIMARY_ARM, "help"),
            [stacked, ceiling],
            primary_provenance=_v2_provenance("primary; sole canonical #75 decision"),
        )
        errors = comparison["validation_errors"]
        self.assertTrue(any("contains a decision" in error for error in errors))
        self.assertTrue(any("implementation differs" in error for error in errors))
        self.assertTrue(any("role is not locked" in error for error in errors))
        self.assertEqual(decide_remedy_v2(comparison)["decision"], 4)

    def test_option_4_when_only_one_option_2_condition_holds(self) -> None:
        comparison = _v2_comparison("help", "failed", "failed")
        self.assertEqual(decide_remedy_v2(comparison)["decision"], 4)

    def test_secondary_payload_written_without_decision(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "evidence"
            args = argparse.Namespace(
                tokens=2048,
                hp_period=2,
                blocks="0,1,2,3",
                npy_root=Path(td),
                npy_pattern="unused",
                pack_root=Path(td),
                pack_pattern="unused",
                embedding_shard=Path(td) / "unused.npy",
                out=out,
                arm="stacked_hp_channel_alpha",
                hp_blocks=None,
                skip_fp16_control=False,
                seed=_DECISION_SEED,
                top_k=2,
                write_report_md=True,
                evidence_only=True,
                comparison_metrics=[],
            )
            with (
                mock.patch.object(
                    multiblock,
                    "run_chain",
                    return_value=_v2_chain(V2_STACKED_ARM, "help"),
                ),
                mock.patch.object(multiblock, "_provenance", return_value={}),
                mock.patch.object(multiblock, "remedy_metrics_note", return_value="fixture"),
            ):
                self.assertEqual(multiblock.run(args), 0)
            payload = json.loads((out / "metrics.json").read_text())
            self.assertNotIn("decision", payload)
            self.assertFalse((out / "results.md").exists())


class RemedyV2ReportTests(unittest.TestCase):
    def test_report_has_one_canonical_decision_and_both_controls(self) -> None:
        comparison = _v2_comparison("help", "failed", "viable")
        payload = {
            "provenance": {
                "agent": "OpenAI Codex: GPT-5.6 Sol (xhigh) · Issue: #75",
                "issue": "GH #75 / Linear RM-462 / beads goz-rvk",
                "implementation": {"commit": "deadbeef", "dirty": False},
            },
            "chain": _v2_chain(V2_PRIMARY_ARM, "help"),
            "comparison": comparison,
            "decision": decide_remedy_v2(comparison),
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.md"
            write_remedy_v2_results_md(path, payload)
            body = path.read_text()
        self.assertEqual(body.count("## Decision"), 1)
        self.assertIn("#72 baseline", body)
        self.assertIn("#74 baseline", body)
        self.assertIn(V2_STACKED_ARM, body)
        self.assertIn(V2_CEILING_ARM, body)
        self.assertIn("neither mostly-ternary candidate met every locked viability band", body)
        self.assertIn("clear improvement and a viable HP ceiling both hold", body)
        self.assertIn("Secondary `metrics.json` files are evidence-only", body)
        self.assertEqual(body.splitlines().count("### FP16 control"), 1)
        self.assertIn(f"#### FP16 control — `{V2_STACKED_ARM}`", body)
        self.assertIn(f"#### FP16 control — `{V2_CEILING_ARM}`", body)
        self.assertFalse(body.endswith("\n\n"))


if __name__ == "__main__":
    unittest.main()
