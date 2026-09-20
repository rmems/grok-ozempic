#!/usr/bin/env python3
"""Unit tests for scripts/grok1_tau_quality_sweep.py (goz-otdnlk external y)."""

from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grok1_tau_quality_sweep as sweep  # noqa: E402


def _gaussian(n: int = 4096, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


class QuantizeMirrorTests(unittest.TestCase):
    def test_ternary_quantize_matches_rust_semantics(self) -> None:
        w = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
        # rms = sqrt(mean(w^2)) = sqrt(8.5/5) ~ 1.3038; gif=0.5 -> tau ~ 0.6519
        q = sweep.ternary_quantize(w, 0.5)
        self.assertAlmostEqual(q["tau_abs"], 0.5 * math.sqrt(8.5 / 5))
        self.assertEqual(q["trits"].tolist(), [-1.0, 0.0, 0.0, 0.0, 1.0])
        self.assertAlmostEqual(q["sparsity"], 3.0 / 5.0)
        # alpha* = sum(w*t)/fired = (-2*-1 + 2*1)/2 = 2.0
        self.assertAlmostEqual(q["alpha_star"], 2.0)

    def test_gaussian_sparsity_tracks_erf_model(self) -> None:
        w = _gaussian(20000)
        # For near-Gaussian weights, zeros(tau) ~= erf(gif_threshold / sqrt(2))
        # (reports/grok-1-tau-sweep/results.md). Wide tolerance: this checks
        # the gate math, not the distribution.
        for g in (0.31, 0.65, 1.12):
            q = sweep.ternary_quantize(w, g)
            expected = math.erf(g / math.sqrt(2.0))
            self.assertAlmostEqual(q["sparsity"], expected, delta=0.03)


class QualityTargetTests(unittest.TestCase):
    def test_cosine_has_interior_optimum_not_monotone(self) -> None:
        # With optimal alpha*, ||w - alpha*t||^2 / ||w||^2 == 1 - cos^2(w,t).
        # Cosine is NOT monotone in tau: at tau->0 every weight fires and
        # cos -> E|w|/rms ~ 0.798 for Gaussian data; mid-range tau removes
        # below-gate noise and can score higher; extreme tau keeps so few
        # trits the full-vector cosine collapses. That interior optimum is
        # exactly what makes this column a usable external y for SAAQ 2.0.
        w = _gaussian()
        metrics = {}
        for g in (0.05, 0.65, 1.97):
            q = sweep.ternary_quantize(w, g)
            metrics[g] = sweep.reconstruction_metrics(w, q["trits"], q["alpha_star"])
        self.assertGreater(
            metrics[0.65]["reconstruction_cosine"],
            metrics[0.05]["reconstruction_cosine"],
        )
        self.assertGreater(
            metrics[0.65]["reconstruction_cosine"],
            metrics[1.97]["reconstruction_cosine"],
        )
        for g, m in metrics.items():
            self.assertAlmostEqual(
                m["relative_l2_error"],
                math.sqrt(max(0.0, 1.0 - m["reconstruction_cosine"] ** 2)),
                places=5,
                msg=f"g={g}: rel_l2 should equal sqrt(1-cos^2)",
            )

    def test_degenerate_tensor_gives_nan_cosine_not_zero(self) -> None:
        w = np.zeros(128, dtype=np.float32)
        q = sweep.ternary_quantize(w, 0.5)
        m = sweep.reconstruction_metrics(w, q["trits"], q["alpha_star"])
        self.assertTrue(math.isnan(m["reconstruction_cosine"]))
        self.assertEqual(q["alpha_star"], 0.0)


class SweepCsvTests(unittest.TestCase):
    def test_run_sweep_emits_expected_rows_and_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npy_dir = Path(tmp) / "npy"
            npy_dir.mkdir()
            np.save(npy_dir / "block_000__slot_00__moe_expert__gate.npy", _gaussian(1024))
            np.save(npy_dir / "block_000__slot_00__moe_expert__down.npy", _gaussian(512, 9))
            out = Path(tmp) / "tau_quality.csv"
            rows = sweep.run_sweep(
                npy_dir, [0.05, 0.65], out
            )
            self.assertEqual(len(rows), 4)
            with out.open() as f:
                read_rows = list(csv.DictReader(f))
            self.assertEqual(read_rows[0].keys(), {c: None for c in sweep.CSV_COLUMNS}.keys())
            names = {r["tensor_name"] for r in read_rows}
            self.assertEqual(
                names,
                {
                    "block_000.slot_00.moe_expert.gate",
                    "block_000.slot_00.moe_expert.down",
                },
            )
            self.assertEqual(
                {r["tensor_role"] for r in read_rows}, {"gate", "down"}
            )
            # Higher gif_threshold -> higher sparsity on the same tensor.
            gate_rows = [
                r for r in read_rows if r["tensor_name"].endswith("gate")
            ]
            self.assertLess(
                float(gate_rows[0]["sparsity"]), float(gate_rows[1]["sparsity"])
            )

    def test_parse_gif_thresholds_rejects_bad_values(self) -> None:
        with self.assertRaises(sweep.SweepError):
            sweep.parse_gif_thresholds("0.1,-0.5")
        with self.assertRaises(sweep.SweepError):
            sweep.parse_gif_thresholds("")
        self.assertEqual(sweep.parse_gif_thresholds("0.1, 0.2"), [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
