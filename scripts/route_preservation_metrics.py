#!/usr/bin/env python3
r"""
Fill the xai-dissect **route-preservation** surface for a bounded Grok-1 block
pilot (GH #53 / RM-222, beads ``goz-4ic2``).

``run3``'s ``route-preservation-report.json`` defines the gates
(``router_top1_agreement >= 99.0%``, ``router_top2_set_agreement >= 99.5%``,
``block_output_cosine >= 0.995``) but leaves every ``observed`` value ``null``:
xai-dissect defines the surface, it does not execute a runtime. This script
supplies the observed values from a real GOZ1 pack.

What is compared
----------------
``reference`` = the f32 npy the pack was built from (int8 x bf16 dequantized
official ckpt-0 weights). ``pilot`` = the tensors read back **out of the GOZ1
pack**: ternary trits for quantized tiers, fp16 for the preserve tier.

GOZ1 v1 stores trits {-1, 0, +1} with no per-tensor scale, so a consumer kernel
must supply one. This script uses the least-squares optimal scale
``alpha = sum(|w|) over fired trits / count(fired)``; reported reconstruction
metrics are therefore the **best case** achievable from this pack.

Because ``cos(w, alpha*t)`` is independent of ``alpha``, the cosine numbers hold
for any positive scale choice.

Router metrics
--------------
Routing flips when the router's *input* drifts, not because the router weights
moved -- they are preserve-tier. So the measurement is: push activations through
a quantized 6144x6144 projection, then read the result with the router taken
**from the pack** (fp16-at-rest, so the preserve tier's own round-trip error is
included), and compare expert selection against the all-f32 reference path.

xai-dissect labels the attention projections ``attn_proj_i8.narrow`` /
``.model_width`` with policy ``wrap_existing_int8_unknown`` -- it does **not**
assign q/k/v/o roles. This script therefore does not guess one: it evaluates
every ``model_width`` (6144x6144) projection independently and reports each,
since any of them may be the output projection feeding the residual stream.

Activations are synthetic: seeded standard-normal token vectors passed through
the block's **real** RMSNorm gain vector (``block_norm``, f32 from the
checkpoint), so per-channel scale is realistic. No calibration corpus is used,
so these are weight-perturbation route-preservation numbers, not corpus
perplexity. ``logit_kl`` / ``perplexity_delta`` / ``generation_sanity_summary``
stay ``unknown`` (they need model-level inference, out of scope for a pilot).

Usage::

    python3 scripts/route_preservation_metrics.py \
      --npy-dir  ~/.models/xai-grok-1/export-npy/block000 \
      --pack     ~/.models/xai-grok-1/artifacts/block-pilot/block_000-attention_only.goz1 \
      --block 0 --mode attention_only \
      --json-out /tmp/route-preservation.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from export_grok1_int8_npy import (
    MODES as EXPORT_MODES,
)
from export_grok1_int8_npy import (
    PRESERVE_KINDS as EXPORT_PRESERVE_KINDS,
)
from goz1_trit_histogram import (
    DATA_ALIGNMENT,
    TENSOR_F16,
    TENSOR_TERNARY,
    _align_up,
    _num_elements,
    _payload_nbytes,
    read_header,
)
from route_preservation_surface import build_summary

# Routing metrics need a d_model x d_model attention projection to drive the router.
ROUTING_MODES = {"attention_only", "attention_plus_expert"}

CHUNK_ELEMS = 1 << 24  # ~16 Mi values per streaming step (rounded down to whole rows)
DEFAULT_TOKENS = 4096
DEFAULT_SEED = 20260805
GATE_FAILURE_EXIT = 3

# Trit code -> value, matching quantizer.rs encode_trit (0b00=0, 0b01=+1, 0b10=-1).
# 0b11 is not a valid code; it is tracked separately so a corrupt pack raises
# instead of being silently counted as a zero (mirrors goz1_trit_histogram.py).
_TRIT_LUT = np.zeros((256, 4), dtype=np.int8)
_INVALID_LUT = np.zeros((256, 4), dtype=bool)
for _b in range(256):
    for _s in range(4):
        _code = (_b >> (2 * _s)) & 0b11
        _TRIT_LUT[_b, _s] = 1 if _code == 0b01 else (-1 if _code == 0b10 else 0)
        _INVALID_LUT[_b, _s] = _code == 0b11


class MetricsError(RuntimeError):
    """Pack/npy mismatch that makes a comparison meaningless."""


def load_pack_index(pack: Path) -> tuple[dict, dict[str, dict]]:
    """Return ``(metadata, {name: entry})`` with absolute payload offsets."""
    with pack.open("rb") as f:
        _version, metadata, tensors, data_start = read_header(f)
    index: dict[str, dict] = {}
    rel = 0
    for t in tensors:
        n = _num_elements(t["shape"])
        nbytes = _payload_nbytes(t["tensor_type"], n, t["name"])
        if t["data_offset"] != rel:
            raise MetricsError(f"{t['name']}: data_offset {t['data_offset']} != cumulative {rel}")
        index[t["name"]] = {
            **t,
            "numel": n,
            "nbytes": nbytes,
            "abs_offset": data_start + rel,
        }
        rel = _align_up(rel + nbytes, DATA_ALIGNMENT)
    return metadata, index


def read_trits(pack: Path, entry: dict, start: int, count: int) -> np.ndarray:
    """Decode ``count`` trits starting at flat index ``start``.

    Any flat start is accepted: the read is floored to the enclosing byte and the
    leading remainder sliced off, so callers are free to chunk by whole rows
    regardless of whether the row length is a multiple of 4.
    """
    byte0, skip = divmod(start, 4)
    nbytes = (skip + count + 3) // 4
    with pack.open("rb") as f:
        f.seek(entry["abs_offset"] + byte0)
        raw = f.read(nbytes)
    if len(raw) != nbytes:
        raise MetricsError(
            f"{entry['name']}: truncated pack -- wanted {nbytes} bytes at payload offset "
            f"{byte0} (flat trit {start}), got {len(raw)}"
        )
    buf = np.frombuffer(raw, dtype=np.uint8)
    if _INVALID_LUT[buf].any():
        raise MetricsError(
            f"{entry['name']}: invalid 0b11 trit code near flat index {start} -- corrupt pack"
        )
    return _TRIT_LUT[buf].reshape(-1)[skip : skip + count]


def read_f16(pack: Path, entry: dict) -> np.ndarray:
    """Read an fp16 preserve-tier tensor and fail closed on truncated payloads."""
    with pack.open("rb") as f:
        f.seek(entry["abs_offset"])
        raw = f.read(entry["nbytes"])
    if len(raw) != entry["nbytes"]:
        raise MetricsError(
            f"{entry['name']}: truncated pack -- wanted {entry['nbytes']} bytes for fp16 payload, "
            f"got {len(raw)}"
        )
    return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(entry["shape"])


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
        """Optimal per-channel alpha and relative error; each holds ``rows`` values."""
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
    """Seeded tokens shaped by the block's real RMSNorm gain."""
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

    square = [n for n, e in ternary.items() if list(e["shape"]) == [d_model, d_model]]
    if not square:
        raise MetricsError(
            f"no {d_model}x{d_model} projection among the pack's ternary tensors, so no "
            "routing gate can be evaluated; re-run with a mode that quantizes attention"
        )

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


def report_gates(summary: list[dict]) -> int:
    """Print the gate table; return the process exit code.

    Fail-closed: a threshold that is not ``pass`` must not exit 0, or a caller
    cannot tell a passing pilot from a failing one.
    """
    gated = [m for m in summary if m["threshold"]]
    print("\nroute-preservation gates:")
    for m in gated:
        obs = "null" if m["observed"] is None else f"{m['observed']:.6f}"
        print(f"  {m['name']:<28} {m['status']:>7}  observed={obs}  threshold={m['threshold']}")
    failing = sum(m["status"] != "pass" for m in gated)
    if failing:
        print(f"\n{failing} of {len(gated)} gates not passing")
        return GATE_FAILURE_EXIT
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Route-preservation metrics for a GOZ1 block pilot")
    p.add_argument("--npy-dir", type=Path, required=True, help="f32 npy the pack was built from")
    p.add_argument("--pack", type=Path, required=True, help="GOZ1 pack to read back")
    p.add_argument("--block", type=int, required=True)
    p.add_argument("--mode", required=True, choices=sorted(EXPORT_MODES.keys()))
    p.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--json-out", type=Path)
    p.add_argument("--conversion-manifest", type=Path, help="xai-dissect run3 conversion-manifest.json")
    return p.parse_args(argv)


def _json_out_conflicts_with_pack(json_out: Path, pack_path: Path) -> bool:
    """True if --json-out would overwrite the input pack (path or same inode)."""
    out_path = json_out.expanduser()
    out_resolved = out_path.resolve()
    if out_resolved == pack_path:
        return True
    if not out_path.exists():
        return False
    try:
        return out_resolved.samefile(pack_path)
    except OSError:
        return False


def _validate_tokens(args: argparse.Namespace) -> None:
    if args.tokens <= 0:
        raise MetricsError(f"--tokens must be positive, got {args.tokens}")


def _expected_ternary_kinds(mode: str) -> set[str]:
    return set(EXPORT_MODES[mode]) - set(EXPORT_PRESERVE_KINDS)


def _validate_block_prefix(args: argparse.Namespace, index: dict[str, dict]) -> None:
    prefix = f"block_{args.block:03d}."
    for name in index:
        if not name.startswith(prefix):
            raise MetricsError(
                f"pack tensor {name!r} does not belong to --block {args.block}"
            )


def _split_tiers(index: dict[str, dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    ternary = {n: e for n, e in index.items() if e["tensor_type"] == TENSOR_TERNARY}
    preserve = {n: e for n, e in index.items() if e["tensor_type"] == TENSOR_F16}
    return ternary, preserve


def _manifest_tensor_names(
    manifest_tensors: list[dict], block: int, kinds: set[str]
) -> list[str]:
    return sorted(
        t["structural_name"]
        for t in manifest_tensors
        if t.get("block") == block and t.get("kind") in kinds
    )


def _validate_ternary_inventory(
    args: argparse.Namespace,
    actual_ternary: dict[str, dict],
    manifest_tensors: list[dict],
) -> None:
    expected_names = _manifest_tensor_names(
        manifest_tensors, args.block, _expected_ternary_kinds(args.mode)
    )
    actual_names = sorted(actual_ternary)
    if expected_names != actual_names:
        expected_set = set(expected_names)
        missing = [n for n in expected_names if n not in actual_ternary]
        extra = [n for n in actual_names if n not in expected_set]
        raise MetricsError(
            f"pack ternary inventory for block {args.block} mode {args.mode} "
            f"does not match conversion-manifest: missing={missing}, extra={extra}"
        )


def _validate_preserve_inventory(
    args: argparse.Namespace,
    index: dict[str, dict],
    manifest_tensors: list[dict],
) -> None:
    expected_names = _manifest_tensor_names(
        manifest_tensors, args.block, set(EXPORT_PRESERVE_KINDS)
    )
    _, preserve = _split_tiers(index)
    actual_names = sorted(preserve)
    if expected_names != actual_names:
        expected_set = set(expected_names)
        missing = [n for n in expected_names if n not in preserve]
        extra = [n for n in actual_names if n not in expected_set]
        raise MetricsError(
            f"pack preserve inventory for block {args.block} mode {args.mode} "
            f"does not match conversion-manifest: missing={missing}, extra={extra}"
        )


def _validate_ternary_kinds_fallback(
    args: argparse.Namespace,
    actual_ternary: dict[str, dict],
) -> None:
    expected_kinds = _expected_ternary_kinds(args.mode)
    actual_kinds = {kind_of(n) for n in actual_ternary}
    if actual_kinds != expected_kinds:
        raise MetricsError(
            f"pack ternary kinds {sorted(actual_kinds)} do not match --mode "
            f"{args.mode} expected {sorted(expected_kinds)}"
        )


def _validate_pilot_args(
    args: argparse.Namespace,
    index: dict[str, dict],
    manifest_tensors: list[dict] | None = None,
) -> None:
    """Make sure --block and --mode match the pack contents."""
    _validate_block_prefix(args, index)
    actual_ternary, _ = _split_tiers(index)
    if manifest_tensors is not None:
        _validate_ternary_inventory(args, actual_ternary, manifest_tensors)
        _validate_preserve_inventory(args, index, manifest_tensors)
    else:
        _validate_ternary_kinds_fallback(args, actual_ternary)


def _load_manifest_tensors(path: Path | None) -> list[dict] | None:
    if path is None:
        return None
    with path.open("rb") as f:
        doc = json.load(f)
    tensors = doc.get("tensors")
    if not isinstance(tensors, list):
        raise MetricsError(f"{path}: no `tensors` array")
    return tensors


def _measure_routing(
    args: argparse.Namespace,
    ternary: dict[str, dict],
    preserve: dict[str, dict],
    weights: dict[str, dict],
) -> dict[str, dict]:
    if args.mode not in ROUTING_MODES:
        print(
            f"  mode {args.mode}: no d_model x d_model attention projection, skipping routing"
        )
        return {}
    return measure_routing(
        args.npy_dir, args.pack, ternary, preserve, weights, args.tokens, args.seed
    )


def _build_result(
    args: argparse.Namespace,
    metadata: dict,
    ternary: dict[str, dict],
    preserve: dict[str, dict],
    weights: dict[str, dict],
    preserve_err: dict[str, dict],
    routing: dict[str, dict],
) -> dict:
    summary = build_summary(weights, routing)
    return {
        "model_family": "grok-1",
        "produced_by": "grok-ozempic scripts/route_preservation_metrics.py (GH #53 / RM-222)",
        "pilot": _pilot_provenance(args, metadata, len(ternary), len(preserve)),
        "summary": summary,
        "weights": weights,
        "preserve_fp16_roundtrip": preserve_err,
        "routing": routing,
    }


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    _validate_tokens(args)
    pack_path = args.pack.expanduser().resolve()
    if args.json_out is not None and _json_out_conflicts_with_pack(args.json_out, pack_path):
        raise MetricsError(
            f"--json-out {args.json_out} resolves to the input pack; "
            "refusing to overwrite the GOZ1 artifact"
        )
    manifest_tensors = _load_manifest_tensors(args.conversion_manifest)
    metadata, index = load_pack_index(args.pack)
    _validate_pilot_args(args, index, manifest_tensors)
    ternary, preserve = _split_tiers(index)
    print(f"pack {args.pack.name}: {len(ternary)} ternary, {len(preserve)} preserve/fp16")

    weights = measure_weights(args.npy_dir, args.pack, ternary)
    preserve_err = measure_preserve(args.npy_dir, args.pack, preserve)
    routing = _measure_routing(args, ternary, preserve, weights)
    result = _build_result(args, metadata, ternary, preserve, weights, preserve_err, routing)
    _write_json(result, args.json_out)
    return report_gates(result["summary"])


def _pilot_provenance(
    args: argparse.Namespace, metadata: dict, n_ternary: int, n_preserve: int
) -> dict:
    """Everything a reader needs to know how these numbers were produced."""
    return {
        "block": args.block,
        "mode": args.mode,
        "pack": str(args.pack),
        "npy_dir": str(args.npy_dir),
        "tokens": args.tokens,
        "seed": args.seed,
        "ternary_tensors": n_ternary,
        "preserve_tensors": n_preserve,
        "pack_metadata": {k: v for k, v in metadata.items() if k.startswith("oz.")},
        "ternary_scale": (
            "least-squares optimal alpha = sum(|w| fired)/count(fired); "
            "GOZ1 v1 stores no scale"
        ),
        "activations": (
            "seeded standard-normal tokens shaped by the block's real RMSNorm gain; "
            "no calibration corpus"
        ),
    }


def _write_json(result: dict, json_out: Path | None) -> None:
    if json_out is None:
        return
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {json_out}")

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except MetricsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
