#!/usr/bin/env python3
"""Measurement kernels for route-preservation (weights, preserve, routing)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from route_preservation_io import (  # noqa: E402
    MetricsError,
    read_f16,
    read_trits,
)

CHUNK_ELEMS = 1 << 24  # ~16 Mi values per streaming step (rounded down to whole rows)
DEFAULT_TOKENS = 4096
DEFAULT_SEED = 20260805
GATE_FAILURE_EXIT = 3


class _TernaryAccumulator:
    """Single-pass sufficient statistics for one ternary tensor.

    Everything needed is additive, so one streaming pass suffices. Channels are
    ``(leading index, last axis)`` pairs -- for a 3-D expert tensor
    ``(8, 6144, 32768)`` that is 8 x 32768 distinct channels, not 32768 pooled
    across experts.
    """

    def __init__(self, lead: int, rows: int, cols: int) -> None:
        self.lead, self.rows, self.cols = lead, rows, cols
        self.s2 = 0.0           # sum w^2
        self.s1_fired = 0.0     # sum |w| over fired trits
        self.n_fired = 0        # count of fired trits
        self.sign_mismatch = 0
        self.max_abs_unfired = 0.0
        self.max_abs_fired = 0.0
        self.min_abs_fired = math.inf
        self.col_s2 = np.zeros((lead, cols), dtype=np.float64)
        self.col_s1f = np.zeros((lead, cols), dtype=np.float64)
        self.col_nf = np.zeros((lead, cols), dtype=np.int64)

    def update(self, li: int, w: np.ndarray, t: np.ndarray) -> None:
        """Fold one chunk of reference weights and their trits into the totals."""
        fired = t != 0
        aw = np.abs(w)
        af = np.where(fired, aw, 0.0)

        self.s2 += float(np.einsum("ij,ij->", w, w))
        self.s1_fired += float(af.sum())
        self.n_fired += int(np.count_nonzero(fired))
        self.sign_mismatch += int(np.count_nonzero(fired & (np.sign(w).astype(np.int8) != t)))

        self.max_abs_fired = max(
            self.max_abs_fired, float(np.max(aw, where=fired, initial=0.0))
        )
        self.min_abs_fired = min(
            self.min_abs_fired, float(np.min(aw, where=fired, initial=math.inf))
        )
        self.max_abs_unfired = max(
            self.max_abs_unfired, float(np.max(aw, where=~fired, initial=0.0))
        )

        self.col_s2[li] += np.einsum("ij,ij->j", w, w)
        self.col_s1f[li] += af.sum(axis=0)
        self.col_nf[li] += fired.sum(axis=0)

    def _per_channel(self) -> dict:
        """Optimal per-channel alpha and relative error.

        Both arrays are shaped ``(lead, cols)`` -- one entry per channel, where
        a channel is a ``(leading index, last axis)`` pair. ``rows`` is the
        number of *elements averaged into* each channel, not the array length.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            nf = np.maximum(self.col_nf, 1)
            alpha_col = np.where(self.col_nf > 0, self.col_s1f / nf, 0.0)
            mse_col = (
                self.col_s2 - np.where(self.col_nf > 0, self.col_s1f**2 / nf, 0.0)
            ) / self.rows
            rms_col = np.sqrt(self.col_s2 / self.rows)
            rel_col = np.where(
                rms_col > 0, np.sqrt(np.maximum(mse_col, 0.0)) / rms_col, 0.0
            )
        return {
            "channels": int(alpha_col.size),
            "channel_definition": (
                f"(leading index, last axis) pairs: {self.lead} x {self.cols}, "
                f"{self.rows} elements each"
            ),
            "alpha_min": float(alpha_col.min()),
            "alpha_median": float(np.median(alpha_col)),
            "alpha_max": float(alpha_col.max()),
            "relative_error_min": float(rel_col.min()),
            "relative_error_median": float(np.median(rel_col)),
            "relative_error_max": float(rel_col.max()),
        }

    def finalize(self, n: int) -> dict:
        """Closed forms: ``cos = S1f / (||w||*sqrt(Nf))`` (scale-free),
        ``alpha* = S1f / Nf``, ``mse_min = (S2 - S1f^2/Nf) / n``.

        ``max |w - alpha*t|`` is exact from the extremes alone: error is ``|w|``
        where the trit is silent and ``||w| - alpha|`` where it fired, and the
        latter is monotone in ``|w|`` from both ends.
        """
        s2, s1f, nf = self.s2, self.s1_fired, self.n_fired
        alpha = s1f / nf if nf else 0.0
        cos = s1f / (math.sqrt(s2) * math.sqrt(nf)) if nf and s2 else 0.0
        mse = (s2 - (s1f**2 / nf if nf else 0.0)) / n
        rms = math.sqrt(s2 / n)
        max_abs_err = self.max_abs_unfired
        if nf:
            max_abs_err = max(
                max_abs_err,
                abs(self.max_abs_fired - alpha),
                abs(self.min_abs_fired - alpha),
            )
        return {
            "elements": n,
            "zeros": n - nf,
            "sparsity": (n - nf) / n,
            "rms": rms,
            "alpha_optimal": alpha,
            "weight_cosine_similarity": cos,
            "weight_reconstruction_mse": mse,
            "weight_nrmse": math.sqrt(mse) / rms if s2 else 0.0,
            "weight_max_absolute_error": max_abs_err,
            "per_channel_scale_error": self._per_channel(),
        }


def ternary_stats(npy_path: Path, pack: Path, entry: dict) -> dict:
    """Streaming exact reconstruction stats for one ternary tensor."""
    ref = np.load(npy_path, mmap_mode="r")
    shape = tuple(int(d) for d in entry["shape"])
    if tuple(ref.shape) != shape:
        raise MetricsError(
            f"{entry['name']}: npy shape {tuple(ref.shape)} != pack shape {shape}; "
            "a transposed or reshaped source would yield wrong per-channel statistics"
        )
    n = ref.size
    cols = shape[-1]
    rows = shape[-2] if len(shape) >= 2 else 1
    lead = n // (rows * cols)
    view = ref.reshape(lead, rows, cols)

    acc = _TernaryAccumulator(lead, rows, cols)
    rows_per_chunk = max(1, CHUNK_ELEMS // cols)
    for li in range(lead):
        for r0 in range(0, rows, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows)
            count = (r1 - r0) * cols
            t = read_trits(pack, entry, (li * rows + r0) * cols, count).reshape(-1, cols)
            acc.update(li, np.asarray(view[li, r0:r1], dtype=np.float64), t)

    if acc.sign_mismatch:
        raise MetricsError(
            f"{entry['name']}: {acc.sign_mismatch} fired trits disagree in sign with the "
            "source weight -- pack does not correspond to this npy"
        )
    return acc.finalize(n)


def reconstruct_full(pack: Path, entry: dict, alpha: float) -> np.ndarray:
    """Materialize ``alpha * trits`` in the tensor's shape, from the pack alone."""
    shape = tuple(int(d) for d in entry["shape"])
    t = read_trits(pack, entry, 0, entry["numel"])
    return (t.astype(np.float32) * np.float32(alpha)).reshape(shape)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits (0 = identical)."""
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-row Spearman rank correlation over small logit vectors."""
    ra = np.argsort(np.argsort(a, axis=1), axis=1).astype(np.float64)
    rb = np.argsort(np.argsort(b, axis=1), axis=1).astype(np.float64)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra**2).sum(axis=1) * (rb**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(den > 0, num / den, 1.0)
    return float(rho.mean())


def make_activations(norm_gain: np.ndarray, tokens: int, seed: int) -> np.ndarray:
    """Seeded standard-normal tokens shaped by the block's real RMSNorm gain."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((tokens, norm_gain.size), dtype=np.float32)
    x /= np.sqrt((x.astype(np.float64) ** 2).mean(axis=1, keepdims=True)).astype(np.float32)
    return x * norm_gain.astype(np.float32)


def routing_metrics(
    x: np.ndarray,
    w_ref: np.ndarray,
    w_pilot: np.ndarray,
    router_ref: np.ndarray,
    router_pilot: np.ndarray,
) -> dict:
    """Route preservation for one quantized projection feeding the router."""
    h_ref = x @ w_ref
    h_q = x @ w_pilot

    num = (h_ref.astype(np.float64) * h_q.astype(np.float64)).sum(axis=1)
    den = np.linalg.norm(h_ref, axis=1).astype(np.float64) * np.linalg.norm(h_q, axis=1).astype(
        np.float64
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_tok = np.where(den > 0, num / den, 1.0)
    rmse = float(np.sqrt(((h_ref.astype(np.float64) - h_q.astype(np.float64)) ** 2).mean()))
    ref_rms = float(np.sqrt((h_ref.astype(np.float64) ** 2).mean()))

    l_ref = h_ref @ router_ref
    l_q = h_q @ router_pilot
    experts = l_ref.shape[1]

    top1_ref, top1_q = l_ref.argmax(axis=1), l_q.argmax(axis=1)
    top1 = float((top1_ref == top1_q).mean())

    o_ref = np.argsort(-l_ref, axis=1)[:, :2]
    o_q = np.argsort(-l_q, axis=1)[:, :2]
    top2 = float(
        np.mean([len(set(a.tolist()) & set(b.tolist())) == 2 for a, b in zip(o_ref, o_q)])
    )

    load_ref = np.bincount(top1_ref, minlength=experts).astype(np.float64) / len(top1_ref)
    load_q = np.bincount(top1_q, minlength=experts).astype(np.float64) / len(top1_q)

    return {
        "tokens": int(x.shape[0]),
        "router_top1_agreement": top1,
        "router_top2_set_agreement": top2,
        "router_logit_rank_correlation": spearman(l_ref, l_q),
        "expert_load_distribution_delta": float(np.abs(load_ref - load_q).max()),
        "expert_load_js_divergence": js_divergence(load_ref, load_q),
        "expert_load_reference": load_ref.tolist(),
        "expert_load_pilot": load_q.tolist(),
        "block_output_cosine": float(cos_tok.mean()),
        "block_output_cosine_min": float(cos_tok.min()),
        "block_output_rmse": rmse,
        "residual_stream_drift": rmse / ref_rms if ref_rms else 0.0,
    }


def stem_of(name: str) -> str:
    return name.replace(".", "__")


def _require_npy(npy_dir: Path, name: str) -> np.ndarray:
    """Load a tensor required by the gates; a miss is fatal, not a traceback."""
    npy = npy_dir / f"{stem_of(name)}.npy"
    if not npy.exists():
        raise MetricsError(f"{name}: source npy missing at {npy}; required to evaluate gates")
    return np.load(npy).astype(np.float32)


def kind_of(name: str) -> str:
    """``block_000.slot_04.attn_proj_i8.model_width`` -> ``attn_proj_i8.model_width``."""
    return name.split(".", 2)[2] if name.count(".") >= 2 else name


def measure_weights(npy_dir: Path, pack: Path, ternary: dict[str, dict]) -> dict[str, dict]:
    """Streamed reconstruction stats for every quantized tensor in the pack."""
    weights: dict[str, dict] = {}
    for name in sorted(ternary):
        npy = npy_dir / f"{stem_of(name)}.npy"
        if not npy.exists():
            raise MetricsError(f"{name}: source npy missing at {npy}")
        # ternary_stats is the first reader; it streams via np.memmap and must not
        # be preceded by a full _require_npy load, which would materialize huge npys.
        st = ternary_stats(npy, pack, ternary[name])
        weights[name] = st
        print(
            f"  ternary {name:<46} zeros={st['sparsity'] * 100:6.2f}%  "
            f"cos={st['weight_cosine_similarity']:.6f}  nrmse={st['weight_nrmse']:.6f}"
        )
    return weights


def measure_preserve(npy_dir: Path, pack: Path, preserve: dict[str, dict]) -> dict[str, dict]:
    """fp16 round-trip error for the preserve tier (fp16-at-rest in GOZ1 v1)."""
    errors: dict[str, dict] = {}
    for name in sorted(preserve):
        ref = _require_npy(npy_dir, name)
        got = read_f16(pack, preserve[name])
        d = np.abs(ref - got).astype(np.float64)
        rms = float(np.sqrt((ref.astype(np.float64) ** 2).mean()))
        errors[name] = {
            "max_absolute_error": float(d.max()),
            "relative_rmse": float(np.sqrt((d**2).mean()) / rms) if rms else 0.0,
            "bit_exact": bool(np.array_equal(ref, got)),
        }
        print(
            f"  preserve {name:<45} fp16 max|err|={errors[name]['max_absolute_error']:.3e}  "
            f"rel_rmse={errors[name]['relative_rmse']:.3e}"
        )
    return errors


def _resolve_router(npy_dir: Path, preserve: dict[str, dict]) -> tuple[str, np.ndarray]:
    """Find the preserve-tier router by suffix and check its orientation.

    Located by suffix rather than a hardcoded slot, and a miss is fatal: a gate
    script that silently emitted an all-`unknown` report would look exactly like
    a passing pilot.
    """
    routers = [n for n in sorted(preserve) if n.endswith(".router")]
    if len(routers) != 1:
        raise MetricsError(
            f"expected exactly one preserve-tier router in the pack, found {routers or 'none'}"
        )
    ref = _require_npy(npy_dir, routers[0])
    if ref.ndim != 2 or ref.shape[0] <= ref.shape[1]:
        raise MetricsError(
            f"{routers[0]}: expected (d_model, experts) with d_model > experts, "
            f"got {ref.shape}; a transposed router would silently mis-size d_model"
        )
    return routers[0], ref


def _resolve_activations(
    npy_dir: Path, preserve: dict[str, dict], d_model: int, tokens: int, seed: int
) -> tuple[str, np.ndarray]:
    """Seeded tokens shaped by the block's real RMSNorm gain.

    A Grok-1 block carries four ``block_norm`` vectors and xai-dissect assigns
    no role to any of them, so there is no principled way to pick the one that
    actually feeds a given projection. The lowest-numbered slot is used for
    every projection and recorded in each result's ``activation_source``; the
    resulting numbers are therefore a consistent weight-perturbation
    comparison, not a claim about the block's true activation path.
    """
    norm_name = next((n for n in sorted(preserve) if n.endswith("block_norm")), None)
    if norm_name is None:
        raise MetricsError("no block_norm in pack; cannot build realistic activations")
    gain = _require_npy(npy_dir, norm_name)
    if gain.shape != (d_model,):
        raise MetricsError(
            f"{norm_name}: shape {gain.shape} does not match router d_model {d_model}"
        )
    return norm_name, make_activations(gain, tokens, seed)


def measure_routing(
    npy_dir: Path,
    pack: Path,
    ternary: dict[str, dict],
    preserve: dict[str, dict],
    weights: dict[str, dict],
    tokens: int,
    seed: int,
) -> dict[str, dict]:
    """Route preservation for each d_model x d_model projection (roles unassigned)."""
    router_name, router_ref = _resolve_router(npy_dir, preserve)
    router_pilot = read_f16(pack, preserve[router_name])
    d_model = router_ref.shape[0]
    norm_name, x = _resolve_activations(npy_dir, preserve, d_model, tokens, seed)

    square = [
        n
        for n, e in ternary.items()
        if tuple(int(d) for d in e["shape"]) == (d_model, d_model)
    ]
    if not square:
        # A weight-only mode (e.g. expert_only) legitimately has no d_model x
        # d_model projection. Aborting here would discard the weight and
        # preserve measurements already computed and force a multi-GiB
        # re-export, so return empty routing: the summary marks the routing
        # gates `unknown` and the run still exits non-zero via report_gates.
        print(
            f"  note: no {d_model}x{d_model} projection in this pack; routing gates "
            "stay unknown (weight and preserve metrics are still reported)"
        )
        return {}

    routing: dict[str, dict] = {}
    for name in sorted(square):
        w_ref = _require_npy(npy_dir, name)
        w_pilot = reconstruct_full(pack, ternary[name], weights[name]["alpha_optimal"])
        r = routing_metrics(x, w_ref, w_pilot, router_ref, router_pilot)
        r["activation_source"] = f"seeded N(0,1) tokens x RMSNorm gain {norm_name}"
        routing[name] = r
        print(
            f"  routing via {name:<42} top1={r['router_top1_agreement'] * 100:6.2f}%  "
            f"top2={r['router_top2_set_agreement'] * 100:6.2f}%  "
            f"cos={r['block_output_cosine']:.6f}"
        )
    return routing


