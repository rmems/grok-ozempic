"""Unit tests for route-preservation gate surface + fail-closed exit (GH #53 / RM-222)."""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import route_preservation_metrics as rpm  # noqa: E402
import route_preservation_surface as rps  # noqa: E402
from route_preservation_measure import GATE_FAILURE_EXIT  # noqa: E402


class GateTests(unittest.TestCase):
    def test_pass_fail_higher_is_better(self) -> None:
        self.assertEqual(rps.gate(0.99, 0.99), "pass")
        self.assertEqual(rps.gate(0.991, 0.99), "pass")
        self.assertEqual(rps.gate(0.989, 0.99), "fail")

    def test_lower_is_better(self) -> None:
        self.assertEqual(rps.gate(0.1, 0.2, higher_is_better=False), "pass")
        self.assertEqual(rps.gate(0.2, 0.2, higher_is_better=False), "pass")
        self.assertEqual(rps.gate(0.3, 0.2, higher_is_better=False), "fail")

    def test_none_is_unknown(self) -> None:
        self.assertEqual(rps.gate(None, 0.99), "unknown")  # type: ignore[arg-type]


class SummarySpecsTests(unittest.TestCase):
    """Lock the run3 gated thresholds so a transcription error cannot flip pass/fail."""

    def test_gated_threshold_limits(self) -> None:
        by_name = {row[0]: row for row in rps._SUMMARY_SPECS}
        self.assertEqual(by_name["router_top1_agreement"][4:6], (">= 99.0%", 0.99))
        self.assertEqual(by_name["router_top2_set_agreement"][4:6], (">= 99.5%", 0.995))
        self.assertEqual(by_name["block_output_cosine"][4:6], (">= 0.995", 0.995))

    def test_exactly_three_gated_metrics(self) -> None:
        gated = [row for row in rps._SUMMARY_SPECS if row[5] is not None]
        self.assertEqual(
            [row[0] for row in gated],
            [
                "router_top1_agreement",
                "router_top2_set_agreement",
                "block_output_cosine",
            ],
        )

    def test_fifteen_surface_rows(self) -> None:
        self.assertEqual(len(rps._SUMMARY_SPECS), 15)


class BuildSummaryTests(unittest.TestCase):
    def _routing(self, **overrides: float) -> dict[str, dict]:
        base = {
            "router_top1_agreement": 1.0,
            "router_top2_set_agreement": 1.0,
            "expert_load_distribution_delta": 0.01,
            "expert_load_js_divergence": 0.02,
            "router_logit_rank_correlation": 0.9,
            "block_output_cosine": 1.0,
            "block_output_rmse": 0.001,
            "residual_stream_drift": 0.002,
        }
        base.update(overrides)
        return {"proj_a": dict(base)}

    def _weights(self) -> dict[str, dict]:
        return {
            "w0": {
                "weight_reconstruction_mse": 0.1,
                "weight_cosine_similarity": 0.9,
                "weight_max_absolute_error": 0.05,
                "per_channel_scale_error": {"relative_error_max": 0.03},
            }
        }

    def test_all_gates_pass(self) -> None:
        summary = rps.build_summary(self._weights(), self._routing())
        by_name = {m["name"]: m for m in summary}
        self.assertEqual(by_name["router_top1_agreement"]["status"], "pass")
        self.assertEqual(by_name["router_top2_set_agreement"]["status"], "pass")
        self.assertEqual(by_name["block_output_cosine"]["status"], "pass")
        self.assertEqual(by_name["block_output_rmse"]["status"], "measured")
        self.assertEqual(by_name["logit_kl"]["status"], "unknown")
        self.assertIsNone(by_name["logit_kl"]["observed"])

    def test_gate_fail_uses_worst_case_min(self) -> None:
        routing = {
            "good": {
                "router_top1_agreement": 1.0,
                "router_top2_set_agreement": 1.0,
                "expert_load_distribution_delta": 0.0,
                "expert_load_js_divergence": 0.0,
                "router_logit_rank_correlation": 1.0,
                "block_output_cosine": 1.0,
                "block_output_rmse": 0.0,
                "residual_stream_drift": 0.0,
            },
            "bad": {
                "router_top1_agreement": 0.98,
                "router_top2_set_agreement": 0.994,
                "expert_load_distribution_delta": 0.0,
                "expert_load_js_divergence": 0.0,
                "router_logit_rank_correlation": 1.0,
                "block_output_cosine": 0.994,
                "block_output_rmse": 0.0,
                "residual_stream_drift": 0.0,
            },
        }
        summary = rps.build_summary(self._weights(), routing)
        by_name = {m["name"]: m for m in summary}
        self.assertEqual(by_name["router_top1_agreement"]["observed"], 0.98)
        self.assertEqual(by_name["router_top1_agreement"]["status"], "fail")
        self.assertEqual(by_name["router_top2_set_agreement"]["observed"], 0.994)
        self.assertEqual(by_name["router_top2_set_agreement"]["status"], "fail")
        self.assertEqual(by_name["block_output_cosine"]["observed"], 0.994)
        self.assertEqual(by_name["block_output_cosine"]["status"], "fail")

    def test_missing_measurements_are_unknown_for_gates(self) -> None:
        summary = rps.build_summary({}, {})
        gated = [m for m in summary if m["threshold"]]
        self.assertTrue(gated)
        self.assertTrue(all(m["status"] == "unknown" for m in gated))
        self.assertTrue(all(m["observed"] is None for m in gated))


class ReportGatesTests(unittest.TestCase):
    def test_all_pass_exits_zero(self) -> None:
        summary = [
            {
                "name": "router_top1_agreement",
                "scope": "router_behavior",
                "status": "pass",
                "threshold": ">= 99.0%",
                "observed": 0.99,
                "detail": "",
            },
            {
                "name": "ungated",
                "scope": "x",
                "status": "measured",
                "threshold": None,
                "observed": 1.0,
                "detail": "",
            },
        ]
        with redirect_stdout(io.StringIO()):
            code = rpm.report_gates(summary)
        self.assertEqual(code, 0)

    def test_fail_exits_three(self) -> None:
        summary = [
            {
                "name": "router_top1_agreement",
                "scope": "router_behavior",
                "status": "fail",
                "threshold": ">= 99.0%",
                "observed": 0.5,
                "detail": "",
            }
        ]
        with redirect_stdout(io.StringIO()):
            code = rpm.report_gates(summary)
        self.assertEqual(code, GATE_FAILURE_EXIT)
        self.assertEqual(GATE_FAILURE_EXIT, 3)

    def test_unknown_gate_is_fail_closed(self) -> None:
        summary = [
            {
                "name": "block_output_cosine",
                "scope": "block_behavior",
                "status": "unknown",
                "threshold": ">= 0.995",
                "observed": None,
                "detail": "",
            }
        ]
        with redirect_stdout(io.StringIO()):
            code = rpm.report_gates(summary)
        self.assertEqual(code, 3)


if __name__ == "__main__":
    unittest.main()
