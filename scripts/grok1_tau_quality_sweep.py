#!/usr/bin/env python3
"""GIF-threshold sweep with a genuine reconstruction-quality target.

``tau_sweep_embedding.sh`` / ``grok1_block0_tau_sweep.py`` record *applied* tau
and sparsity, but neither emits a quality signal an external learner could fit:
the multiblock harness measures block-output cosine at whole-block granularity,
and the per-tensor alpha* stored in GOZ1 v3 was never joined back into a table
next to the tau that produced it. This script closes that gap for the
``goz-otdnlk`` "no circular rediscovery" requirement: it produces a CSV whose
target column is a real reconstruction metric, not the SAAQ formula's own
output.

For each ``.npy`` float tensor in ``--npy-dir`` and each ``gif_threshold`` in
the sweep it mirrors ``src/core/quantizer.rs::quantize_f32`` exactly:

    rms   = sqrt(mean(w^2))
    tau   = gif_threshold * rms
    t     = sign(w) where |w| >= tau else 0
    alpha = sum(w*t) / count(t != 0)   (least-squares scale; alpha* in GOZ1 v3)

and records per (tensor, gif_threshold) row:

    input signals:  tensor_name, tensor_role, num_elements, rms, kurtosis,
                    mean_abs, std
    control:        gif_threshold, tau_abs
    outcome:        sparsity, alpha_star
    quality target: reconstruction_cosine, reconstruction_mse,
                    relative_l2_error

``reconstruction_cosine`` is the ``external y`` Surrogate_Viz.jl's
``SAAQ_quality_discovery.jl`` fits; it is defined as
``cos(w, alpha*t)`` — cosine between the original weights and the optimal
ternary reconstruction.

Usage::

    python3 scripts/grok1_tau_quality_sweep.py \\
        --npy-dir ~/.models/xai-grok-1/export-npy \\
        --gif-thresholds 0.05,0.31,0.65,1.12,1.63,1.97 \\
        --out reports/grok-1-tau-quality/tau_quality.csv

Optional ``--verify-pack PACK.goz1`` cross-checks that the per-tensor
``gif_threshold`` / ``threshold_abs`` / ``scale`` stored in a GOZ1 v3 pack
match what this sweep would recompute from the same npy inputs — the applied-tau
audit the #58/#66 trap requires, run offline without re-quantizing.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import goz1_trit_histogram  # noqa: E402

CSV_COLUMNS = [
    "tensor_name",
    "tensor_role",
    "num_elements",
    "rms",
    "kurtosis",
    "mean_abs",
    "std",
    "gif_threshold",
    "tau_abs",
    "sparsity",
    "alpha_star",
    "reconstruction_cosine",
    "reconstruction_mse",
    "relative_l2_error",
]

DEFAULT_GIF_THRESHOLDS = "0.05,0.15,0.31,0.50,0.65,0.80,1.12,1.40,1.63,1.97"


class SweepError(RuntimeError):
    """Raised for bad inputs (missing dirs, unreadable npy, bad thresholds)."""


def stem_to_name(stem: str) -> str:
    """Mirror ``npy_stem_to_tensor_name``: ``a__b__c.npy`` -> ``a.b.c``."""
    return stem.replace("__", ".")


def tensor_role(name: str) -> str:
    """Last structural segment (``gate``/``up``/``down``/...) as a coarse tier."""
    return name.rsplit(".", 1)[-1] if name else ""


def tensor_stats(w: np.ndarray) -> dict:
    """Quantization-time-observable tensor statistics (the CSV's X columns)."""
    w = np.asarray(w, dtype=np.float64).ravel()
    n = w.size
    if n == 0:
        raise SweepError("empty tensor")
    mean = float(w.mean())
    std = float(w.std())
    rms = float(math.sqrt(float((w * w).mean())))
    mean_abs = float(np.abs(w).mean())
    kurtosis = float("nan") if std == 0.0 else float(((w - mean) ** 4).mean() / std**4 - 3.0)
    return {
        "num_elements": n,
        "rms": rms,
        "kurtosis": kurtosis,
        "mean_abs": mean_abs,
        "std": std,
    }


def ternary_quantize(w: np.ndarray, gif_threshold: float) -> dict:
    """Mirror of ``quantize_f32``: trits, tau_abs, sparsity, alpha*."""
    w = np.asarray(w, dtype=np.float64).ravel()
    rms = float(math.sqrt(float((w * w).mean()))) if w.size else 0.0
    tau_abs = gif_threshold * rms
    trits = np.where(np.abs(w) >= tau_abs, np.sign(w), 0.0)
    fired = int((trits != 0).sum())
    alpha_star = float((w * trits).sum() / fired) if fired else 0.0
    sparsity = 1.0 - (fired / w.size) if w.size else 0.0
    return {
        "trits": trits,
        "tau_abs": tau_abs,
        "sparsity": sparsity,
        "alpha_star": alpha_star,
    }


def reconstruction_metrics(w: np.ndarray, trits: np.ndarray, alpha_star: float) -> dict:
    """Quality target columns: how well ``alpha*t`` reconstructs ``w``.

    ``reconstruction_cosine`` is NaN when the reconstruction is identically
    zero (nothing fired, or alpha* == 0): the cosine is undefined there and an
    honest NaN is better than an invented score.
    """
    w = np.asarray(w, dtype=np.float64).ravel()
    recon = alpha_star * trits
    mse = float(((w - recon) ** 2).mean())
    w_norm = float(np.linalg.norm(w))
    r_norm = float(np.linalg.norm(recon))
    cosine = (
        float(np.dot(w, recon) / (w_norm * r_norm))
        if w_norm > 0.0 and r_norm > 0.0
        else float("nan")
    )
    rel_l2 = float(np.linalg.norm(w - recon) / w_norm) if w_norm > 0.0 else float("nan")
    return {
        "reconstruction_cosine": cosine,
        "reconstruction_mse": mse,
        "relative_l2_error": rel_l2,
    }


def sweep_tensor(w: np.ndarray, name: str, gif_thresholds: list[float]) -> list[dict]:
    """One row per gif_threshold for a single tensor."""
    stats = tensor_stats(w)
    w_flat = np.asarray(w, dtype=np.float64).ravel()
    rows = []
    for g in gif_thresholds:
        q = ternary_quantize(w_flat, g)
        qual = reconstruction_metrics(w_flat, q["trits"], q["alpha_star"])
        rows.append(
            {
                "tensor_name": name,
                "tensor_role": tensor_role(name),
                "num_elements": stats["num_elements"],
                "rms": f"{stats['rms']:.8g}",
                "kurtosis": f"{stats['kurtosis']:.8g}",
                "mean_abs": f"{stats['mean_abs']:.8g}",
                "std": f"{stats['std']:.8g}",
                "gif_threshold": f"{g:.8g}",
                "tau_abs": f"{q['tau_abs']:.8g}",
                "sparsity": f"{q['sparsity']:.8g}",
                "alpha_star": f"{q['alpha_star']:.8g}",
                "reconstruction_cosine": f"{qual['reconstruction_cosine']:.8g}",
                "reconstruction_mse": f"{qual['reconstruction_mse']:.8g}",
                "relative_l2_error": f"{qual['relative_l2_error']:.8g}",
            }
        )
    return rows


def load_npy_dir(npy_dir: Path) -> dict[str, np.ndarray]:
    if not npy_dir.is_dir():
        raise SweepError(f"--npy-dir is not a directory: {npy_dir}")
    tensors: dict[str, np.ndarray] = {}
    for path in sorted(npy_dir.glob("*.npy")):
        tensors[stem_to_name(path.stem)] = np.load(path)
    if not tensors:
        raise SweepError(f"no .npy files under {npy_dir}")
    return tensors


def run_sweep(npy_dir: Path, gif_thresholds: list[float], out_path: Path) -> list[dict]:
    tensors = load_npy_dir(npy_dir)
    rows: list[dict] = []
    for name, w in tensors.items():
        rows.extend(sweep_tensor(w, name, gif_thresholds))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def verify_pack(pack_path: Path, npy_dir: Path) -> list[str]:
    """Check stored per-tensor tau/alpha* against a fresh recompute.

    Returns a list of mismatch strings; empty means every ternary row that had
    a matching npy tensor agrees with the on-disk record (the #58/#66 audit).
    """
    tensors = load_npy_dir(npy_dir)
    with pack_path.open("rb") as f:
        _version, _metadata, rows, _data_start = goz1_trit_histogram.read_header(f)
    mismatches: list[str] = []
    checked = 0
    for row in rows:
        # fp16 rows carry gif_threshold=threshold_abs=0 by construction and
        # scale=1.0; only ternary rows exercise the GIF gate.
        if row["tensor_type"] != goz1_trit_histogram.TENSOR_TERNARY:
            continue
        name = row["name"]
        if name not in tensors:
            continue
        w = np.asarray(tensors[name], dtype=np.float64).ravel()
        rms = float(math.sqrt(float((w * w).mean())))
        expected_tau_abs = row["gif_threshold"] * rms
        if not math.isclose(
            expected_tau_abs, row["threshold_abs"], rel_tol=1e-4, abs_tol=1e-9
        ):
            mismatches.append(
                f"{name}: threshold_abs={row['threshold_abs']} != "
                f"gif_threshold({row['gif_threshold']})*rms({rms:.6g})={expected_tau_abs:.6g}"
            )
            continue
        q = ternary_quantize(w, row["gif_threshold"])
        if row["scale"] is not None and not math.isclose(
            q["alpha_star"], row["scale"], rel_tol=1e-3, abs_tol=1e-9
        ):
            mismatches.append(
                f"{name}: scale={row['scale']} != recomputed alpha*={q['alpha_star']:.6g}"
            )
            continue
        checked += 1
    if checked == 0:
        mismatches.append("no tensor names matched between pack and --npy-dir")
    return mismatches


def parse_gif_thresholds(raw: str) -> list[float]:
    try:
        values = [float(x) for x in raw.split(",") if x.strip()]
    except ValueError as exc:
        raise SweepError(f"bad --gif-thresholds {raw!r}: {exc}") from exc
    if not values:
        raise SweepError("--gif-thresholds produced an empty sweep")
    for v in values:
        if not math.isfinite(v) or v < 0.0:
            raise SweepError(f"gif_threshold must be finite and >= 0, got {v}")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--npy-dir", required=True, type=Path)
    parser.add_argument("--gif-thresholds", default=DEFAULT_GIF_THRESHOLDS)
    parser.add_argument("--out", type=Path, help="CSV output path (sweep mode)")
    parser.add_argument(
        "--verify-pack",
        type=Path,
        help="GOZ1 pack to audit against --npy-dir (no CSV written)",
    )
    args = parser.parse_args(argv)

    try:
        if args.verify_pack is not None:
            mismatches = verify_pack(args.verify_pack, args.npy_dir)
            if mismatches:
                for m in mismatches:
                    print(f"MISMATCH {m}")
                return 1
            print("verify-pack ok: every matched ternary row agrees with recompute")
            return 0

        if args.out is None:
            raise SweepError("--out is required in sweep mode")
        rows = run_sweep(args.npy_dir, parse_gif_thresholds(args.gif_thresholds), args.out)
        n_tensors = len({r["tensor_name"] for r in rows})
        print(
            f"wrote {len(rows)} rows ({n_tensors} tensors x "
            f"{len(rows) // max(n_tensors, 1)} thresholds) to {args.out}"
        )
        return 0
    except SweepError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
