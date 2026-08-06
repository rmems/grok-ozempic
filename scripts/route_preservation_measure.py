#!/usr/bin/env python3
"""Measurement kernels for route-preservation (weights, preserve, routing)."""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_grok1_int8_scan import ExportError, scan_shard  # noqa: E402
from route_preservation_io import (  # noqa: E402
    MetricsError,
    read_f16,
    read_trits,
)

CHUNK_ELEMS = 1 << 24  # ~16 Mi values per streaming step (rounded down to whole rows)
DEFAULT_TOKENS = 4096
DEFAULT_SEED = 20260805
GATE_FAILURE_EXIT = 3


@dataclass(frozen=True)
class ActivationSpec:
    """How to build the activations the routing comparison runs on.

    ``embedding_shard`` set means real block-0 activations (embedding rows);
    ``None`` means the synthetic Gaussian fallback.
    """

    tokens: int = DEFAULT_TOKENS
    seed: int = DEFAULT_SEED
    embedding_shard: Path | None = None


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


def _rmsnorm(x: np.ndarray, norm_gain: np.ndarray) -> np.ndarray:
    """Apply RMSNorm with the block's real gain vector.

    Note this makes the result invariant to any global embedding scale: Grok-1
    multiplies the embedding lookup by a constant, but ``rmsnorm(c*x) ==
    rmsnorm(x)`` for ``c > 0``, so the measurement does not depend on knowing it.
    """
    rms = np.sqrt((x.astype(np.float64) ** 2).mean(axis=1, keepdims=True))
    return (x / rms.astype(np.float32)) * norm_gain.astype(np.float32)


def make_activations(norm_gain: np.ndarray, tokens: int, seed: int) -> np.ndarray:
    """Seeded standard-normal tokens shaped by the block's real RMSNorm gain.

    Fallback only. Gaussian coordinates are independent and isotropic, which real
    LLM activations are not, so prefer :func:`make_activations_from_embedding`
    whenever the checkpoint is reachable.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((tokens, norm_gain.size), dtype=np.float32)
    return _rmsnorm(x, norm_gain)


def _find_embedding_array(shard: Path):
    """Return the single 2-D f32 array in the shard, or fail closed."""
    try:
        specs = scan_shard(shard)
    except (ExportError, OSError, ValueError) as exc:
        raise MetricsError(f"{shard}: cannot read embedding shard: {exc}") from exc
    f32 = [s for s in specs if s.descr == "f4" and len(s.shape) == 2]
    if len(f32) != 1:
        raise MetricsError(
            f"{shard}: expected exactly one 2-D f32 array (the token embedding), "
            f"found {[(s.descr, s.shape) for s in specs]}"
        )
    return f32[0]


def _embedding_spec(shard: Path, norm_gain: np.ndarray, tokens: int):
    """Locate the token embedding and check it matches this block and sample size."""
    spec = _find_embedding_array(shard)
    vocab, d_model = spec.shape
    if tokens > vocab:
        raise MetricsError(
            f"--tokens {tokens} exceeds embedding vocabulary {vocab} in {shard}; "
            "choose a smaller sample or omit --embedding-shard"
        )
    if d_model != norm_gain.size:
        raise MetricsError(
            f"{shard}: embedding width {d_model} != block_norm width {norm_gain.size}; "
            "this shard is not the token embedding for this model"
        )
    return spec


def make_activations_from_embedding(
    shard: Path, norm_gain: np.ndarray, tokens: int, seed: int
) -> tuple[np.ndarray, dict]:
    """Real block-0 attention input: token-embedding rows through RMSNorm.

    A decoder block computes ``h = h + attn(rmsnorm(h))``, and for **block 0**
    ``h`` is the embedding lookup itself. So sampled rows of the token embedding,
    pushed through the block's own ``block_norm`` gain, are the actual
    distribution that block sees at inference time -- no calibration corpus
    required, and no synthetic assumption about correlation or isotropy.

    Rows are read from the pickle shard through ``numpy.memmap`` at the offset
    the opcode scanner reports, so only the sampled rows are touched rather than
    the full 3 GiB matrix.

    Returns ``(activations, provenance)``.
    """
    try:
        spec = _embedding_spec(shard, norm_gain, tokens)
        vocab = spec.shape[0]
        emb = np.memmap(shard, dtype="<f4", mode="r", offset=spec.offset, shape=spec.shape)
        rng = np.random.default_rng(seed)
        # Sorted, distinct row ids: distinct so no token is double-counted, sorted so
        # the memmap reads walk forward through the file.
        idx = np.sort(rng.choice(vocab, size=tokens, replace=False))
        x = _rmsnorm(np.asarray(emb[idx], dtype=np.float32), norm_gain)
    except (OSError, ValueError) as exc:
        raise MetricsError(f"{shard}: cannot read embedding rows: {exc}") from exc
    return x, {
        "source": "token_embedding_rows",
        "shard": str(shard),
        "vocab_size": int(vocab),
        "rows_sampled": int(tokens),
        "sampling": "uniform over vocab without replacement (no corpus token frequencies)",
        "detail": (
            "block 0's attention input is rmsnorm(embedding lookup), so these are "
            "real activations for this block rather than a synthetic distribution"
        ),
    }


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


def _shape_sources(
    ref: np.ndarray,
    got: np.ndarray,
    entry: dict,
    declared: tuple[int, ...] | None,
) -> dict[str, tuple[int, ...]]:
    """Every independent statement of this tensor's shape, labelled by origin."""
    shapes = {
        "source npy": tuple(int(d) for d in ref.shape),
        "decoded pack array": tuple(int(d) for d in got.shape),
        "pack header": tuple(int(d) for d in entry["shape"]),
    }
    if declared is not None:
        shapes["conversion manifest"] = declared
    return shapes


def _require_matching_shapes(
    name: str,
    ref: np.ndarray,
    got: np.ndarray,
    entry: dict,
    manifest_shapes: dict[str, tuple[int, ...]] | None,
) -> None:
    """Require source npy, decoded pack array, pack header and manifest to agree.

    All four must match *exactly* before any subtraction. NumPy would happily
    broadcast e.g. ``(6144,)`` against ``(6144, 1)`` and return a plausible
    error figure computed over the wrong pairing, so a mismatch has to raise
    rather than reshape.
    """
    shapes = _shape_sources(ref, got, entry, (manifest_shapes or {}).get(name))
    if len(set(shapes.values())) != 1:
        detail = ", ".join(f"{src}={shape}" for src, shape in shapes.items())
        raise MetricsError(
            f"{name}: preserve-tier shape disagreement ({detail}); refusing to "
            "compare — broadcasting would report an error over the wrong pairing"
        )


def measure_preserve(
    npy_dir: Path,
    pack: Path,
    preserve: dict[str, dict],
    manifest_shapes: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, dict]:
    """fp16 round-trip error for the preserve tier (fp16-at-rest in GOZ1 v1)."""
    errors: dict[str, dict] = {}
    for name in sorted(preserve):
        ref = _require_npy(npy_dir, name)
        got = read_f16(pack, preserve[name])
        _require_matching_shapes(name, ref, got, preserve[name], manifest_shapes)
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
    npy_dir: Path,
    preserve: dict[str, dict],
    d_model: int,
    spec: ActivationSpec,
) -> tuple[str, np.ndarray, dict]:
    """Activations for the routing measurement, real when the checkpoint is given.

    With ``embedding_shard``, rows of the real token embedding are used -- for
    block 0 that *is* the attention input, so no synthetic distribution is
    involved. Without it, falls back to seeded Gaussian tokens.

    A Grok-1 block carries four ``block_norm`` vectors and xai-dissect assigns
    no role to any of them, so there is no principled way to pick the one that
    actually feeds a given projection. The lowest-numbered slot is used for
    every projection and recorded in each result's ``rmsnorm_gain``.
    """
    norm_name = next((n for n in sorted(preserve) if n.endswith("block_norm")), None)
    if norm_name is None:
        raise MetricsError("no block_norm in pack; cannot build realistic activations")
    gain = _require_npy(npy_dir, norm_name)
    if gain.shape != (d_model,):
        raise MetricsError(
            f"{norm_name}: shape {gain.shape} does not match router d_model {d_model}"
        )
    if spec.embedding_shard is not None:
        x, provenance = make_activations_from_embedding(
            spec.embedding_shard, gain, spec.tokens, spec.seed
        )
    else:
        x = make_activations(gain, spec.tokens, spec.seed)
        provenance = {
            "source": "synthetic_gaussian",
            "sampling": "seeded standard-normal rows",
            "detail": (
                "FALLBACK: independent isotropic coordinates, which real activations "
                "are not; pass --embedding-shard for real block-0 activations"
            ),
        }
    provenance["rmsnorm_gain"] = norm_name
    return norm_name, x, provenance


def measure_routing(
    npy_dir: Path,
    pack: Path,
    ternary: dict[str, dict],
    preserve: dict[str, dict],
    weights: dict[str, dict],
    spec: ActivationSpec,
) -> dict[str, dict]:
    """Route preservation for each d_model x d_model projection (roles unassigned)."""
    router_name, router_ref = _resolve_router(npy_dir, preserve)
    router_pilot = read_f16(pack, preserve[router_name])
    d_model = router_ref.shape[0]
    _norm_name, x, activations = _resolve_activations(npy_dir, preserve, d_model, spec)
    print(f"  activations: {activations['source']} ({activations['sampling']})")

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
        r["activations"] = activations
        routing[name] = r
        print(
            f"  routing via {name:<42} top1={r['router_top1_agreement'] * 100:6.2f}%  "
            f"top2={r['router_top2_set_agreement'] * 100:6.2f}%  "
            f"cos={r['block_output_cosine']:.6f}"
        )
    return routing


