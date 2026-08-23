#!/usr/bin/env python3
"""Issue #85 completeness and canonical P0/P1 ranking tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grok1_multiblock_lib import (  # noqa: E402
    V4_CANDIDATE_ARMS,
    V4_INT4_BASELINE_ARM,
    V4_PRIMARY_ARM,
    V4_SECONDARY_ARM,
    assemble_remedy_v4_comparison,
    decide_remedy_v4,
)
from test_grok1_multiblock_experiment import (  # noqa: E402
    _V4_IMPL,
    _v4_chain,
    _v4_secondary_payload,
)


def _comparison(
    *,
    p0_top1: float = 0.97,
    p1_top1: float = 0.96,
    baseline_top1: float = 0.90,
    p0: dict | None = None,
    p1: dict | None = None,
    baseline: dict | None = None,
    implementation: dict | None = None,
) -> dict:
    impl = dict(_V4_IMPL if implementation is None else implementation)
    primary = _v4_chain(V4_PRIMARY_ARM, top1_last=p0_top1) if p0 is None else p0
    p1_chain = _v4_chain(V4_SECONDARY_ARM, top1_last=p1_top1) if p1 is None else p1
    base_chain = (
        _v4_chain(V4_INT4_BASELINE_ARM, top1_last=baseline_top1)
        if baseline is None
        else baseline
    )
    secondaries = []
    if base_chain:
        secondaries.append(_v4_secondary_payload(base_chain, impl))
    if p1_chain:
        secondaries.append(_v4_secondary_payload(p1_chain, impl))
    return assemble_remedy_v4_comparison(
        primary,
        secondaries,
        primary_provenance={"implementation": impl},
    )


class V4CompletenessTests(unittest.TestCase):
    def _assert_incomplete(self, comparison: dict, missing: str | None = None) -> None:
        self.assertFalse(comparison["protocol_complete"])
        self.assertNotIn("ranking", comparison)
        self.assertNotIn("best_remedy_arm", comparison)
        if missing is not None:
            self.assertIn(missing, comparison["missing_arms"])
        decision = decide_remedy_v4(comparison)
        self.assertEqual(decision["decision"], 4)
        self.assertFalse(decision["protocol_complete"])
        self.assertNotIn("best_remedy_arm", decision)

    def test_missing_baseline_forces_option_4(self) -> None:
        impl = dict(_V4_IMPL)
        comparison = assemble_remedy_v4_comparison(
            _v4_chain(V4_PRIMARY_ARM),
            [_v4_secondary_payload(_v4_chain(V4_SECONDARY_ARM), impl)],
            primary_provenance={"implementation": impl},
        )
        self._assert_incomplete(comparison, V4_INT4_BASELINE_ARM)

    def test_missing_p1_forces_option_4(self) -> None:
        impl = dict(_V4_IMPL)
        comparison = assemble_remedy_v4_comparison(
            _v4_chain(V4_PRIMARY_ARM),
            [_v4_secondary_payload(_v4_chain(V4_INT4_BASELINE_ARM), impl)],
            primary_provenance={"implementation": impl},
        )
        self._assert_incomplete(comparison, V4_SECONDARY_ARM)

    def test_missing_p0_forces_option_4(self) -> None:
        impl = dict(_V4_IMPL)
        comparison = assemble_remedy_v4_comparison(
            None,
            [
                _v4_secondary_payload(_v4_chain(V4_INT4_BASELINE_ARM), impl),
                _v4_secondary_payload(_v4_chain(V4_SECONDARY_ARM), impl),
            ],
            primary_provenance={"implementation": impl},
        )
        self._assert_incomplete(comparison, V4_PRIMARY_ARM)

    def test_incomplete_blocks_are_invalid_without_ranking(self) -> None:
        p1 = _v4_chain(V4_SECONDARY_ARM)
        p1["per_block"] = p1["per_block"][:-1]
        comparison = _comparison(p1=p1)
        self.assertIn(V4_SECONDARY_ARM, comparison["invalid_arms"])
        self._assert_incomplete(comparison)

    def test_provenance_mismatch_is_invalid_without_ranking(self) -> None:
        impl = dict(_V4_IMPL)
        p1_payload = _v4_secondary_payload(_v4_chain(V4_SECONDARY_ARM), {"commit": "other"})
        comparison = assemble_remedy_v4_comparison(
            _v4_chain(V4_PRIMARY_ARM),
            [
                _v4_secondary_payload(_v4_chain(V4_INT4_BASELINE_ARM), impl),
                p1_payload,
            ],
            primary_provenance={"implementation": impl},
        )
        self.assertIn(V4_SECONDARY_ARM, comparison["invalid_arms"])
        self._assert_incomplete(comparison)

    def test_global_validation_error_forces_option_4_without_ranking(self) -> None:
        impl = dict(_V4_IMPL)
        comparison = assemble_remedy_v4_comparison(
            _v4_chain(V4_PRIMARY_ARM),
            [
                _v4_secondary_payload(_v4_chain(V4_INT4_BASELINE_ARM), impl),
                _v4_secondary_payload(_v4_chain(V4_SECONDARY_ARM), impl),
            ],
            primary_provenance={"implementation": impl},
            load_errors=["synthetic unreadable artifact"],
        )
        self.assertIn(
            "synthetic unreadable artifact",
            comparison["validation"]["global_errors"],
        )
        self._assert_incomplete(comparison)


class V4RankingTests(unittest.TestCase):
    def _decision(self, **kwargs) -> tuple[dict, dict]:
        comparison = _comparison(**kwargs)
        self.assertTrue(comparison["protocol_complete"])
        self.assertEqual(comparison["missing_arms"], [])
        self.assertEqual(comparison["invalid_arms"], [])
        decision = decide_remedy_v4(comparison)
        self.assertIn(decision["decision"], (1, 2, 3))
        return comparison, decision

    def test_p0_wins_and_baseline_is_comparator_only(self) -> None:
        comparison, decision = self._decision(p0_top1=0.98, p1_top1=0.96)
        ranking = comparison["ranking"]
        self.assertEqual(ranking["winner"], V4_PRIMARY_ARM)
        self.assertEqual(decision["best_remedy_arm"], V4_PRIMARY_ARM)
        self.assertEqual(set(ranking["ordered_candidates"]), set(V4_CANDIDATE_ARMS))
        self.assertNotIn(V4_INT4_BASELINE_ARM, ranking["ordered_candidates"])

    def test_p1_can_win_canonical_ranking(self) -> None:
        comparison, decision = self._decision(p0_top1=0.96, p1_top1=0.98)
        self.assertEqual(comparison["ranking"]["winner"], V4_SECONDARY_ARM)
        self.assertEqual(decision["best_remedy_arm"], V4_SECONDARY_ARM)

    def test_exact_tie_prefers_lower_complexity_p0(self) -> None:
        comparison, decision = self._decision(p0_top1=0.97, p1_top1=0.97)
        ranking = comparison["ranking"]
        self.assertEqual(ranking["winner"], V4_PRIMARY_ARM)
        self.assertIn("exact metric-rank tie", ranking["tie_break_reason"])
        self.assertEqual(decision["best_remedy_arm"], V4_PRIMARY_ARM)

    def test_option_2_when_candidates_help_but_miss_viability(self) -> None:
        comparison, decision = self._decision(
            p0_top1=0.91, p1_top1=0.90, baseline_top1=0.85
        )
        self.assertEqual(decision["decision"], 2)
        self.assertGreater(
            comparison["ranking"]["baseline_deltas"][V4_PRIMARY_ARM]["b3_top1_gain"],
            0.0,
        )

    def test_option_3_when_neither_candidate_beats_baseline(self) -> None:
        _, decision = self._decision(p0_top1=0.89, p1_top1=0.88, baseline_top1=0.90)
        self.assertEqual(decision["decision"], 3)

    def test_option_3_when_only_losing_candidate_improves_other_metrics(self) -> None:
        baseline = _v4_chain(V4_INT4_BASELINE_ARM, top1_last=0.900)
        p0 = _v4_chain(V4_PRIMARY_ARM, top1_last=0.899)
        p1 = _v4_chain(V4_SECONDARY_ARM, top1_last=0.898)
        p0["per_block"][-1]["expert_only"]["block_output_cosine"] = 0.994
        p0["end_of_chain"]["expert_only_chain_exit"][
            "residual_drift_relative_norm"
        ] = 0.021
        p1["per_block"][-1]["expert_only"]["block_output_cosine"] = 0.997
        p1["end_of_chain"]["expert_only_chain_exit"][
            "residual_drift_relative_norm"
        ] = 0.018

        comparison, decision = self._decision(
            p0=p0,
            p1=p1,
            baseline=baseline,
        )

        self.assertEqual(comparison["ranking"]["winner"], V4_PRIMARY_ARM)
        self.assertEqual(decision["best_remedy_arm"], V4_PRIMARY_ARM)
        self.assertEqual(decision["decision"], 3)
        self.assertIn("Selected stacked / denser INT4 remedy fails", decision["decision_text"])
        self.assertNotIn("remedies fail", decision["decision_text"])
        self.assertIn(
            "selected_stack_improved_vs_remeasured_int4=False",
            decision["rationale"],
        )

    def test_historical_2048_evidence_remains_cite_only(self) -> None:
        comparison, decision = self._decision()
        note = comparison["baselines_cited"]["historical_80_int4_all"]["note"]
        self.assertIn("2048-token", note)
        self.assertIn("not same-budget", note)
        self.assertTrue(
            any("(cite only; not same-budget)" in line for line in decision["rationale"])
        )


if __name__ == "__main__":
    unittest.main()
