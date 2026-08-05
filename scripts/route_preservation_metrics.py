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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from goz1_trit_histogram import (  # noqa: E402  (path shim above is deliberate)
    DATA_ALIGNMENT,
    TENSOR_F16,
    TENSOR_TERNARY,
    _align_up,
    _num_elements,
    _payload_nbytes,
    read_header,
)

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
    with pack.open("rb") as f:
        f.seek(entry["abs_offset"])
        raw = f.read(entry["nbytes"])
    return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(entry["shape"])


def ternary_stats(npy_path: Path, pack: Path, entry: dict) -> dict:
    """Streaming exact reconstruction stats for one ternary tensor.

    Uses closed forms so a single pass suffices:
    ``cos = S1f / (||w|| * sqrt(Nf))`` (scale-free),
    ``alpha* = S1f / Nf``, ``mse_min = (S2 - S1f^2 / Nf) / n``.
    """
    ref = np.load(npy_path, mmap_mode="r")
    shape = tuple(int(d) for d in entry["shape"])
    if tuple(ref.shape) != shape:
        raise MetricsError(
            f"{entry['name']}: npy shape {tuple(ref.shape)} != pack shape {shape}; "
            "a transposed or reshaped source would yield wrong per-channel statistics"
        )
    n = ref.size

    # Channels are (leading index, last axis) pairs -- for a 3-D expert tensor
    # (8, 6144, 32768) that is 8 x 32768 distinct channels, not 32768 pooled
    # across experts. Chunks never cross a leading-axis boundary.
    cols = shape[-1]
    rows = shape[-2] if len(shape) >= 2 else 1
    lead = n // (rows * cols)
    view = ref.reshape(lead, rows, cols)

    s2 = 0.0            # sum w^2
    s1_fired = 0.0      # sum |w| over fired trits
    n_fired = 0         # count of fired trits
    sign_mismatch = 0
    max_abs_unfired = 0.0
    max_abs_fired = 0.0
    min_abs_fired = math.inf
    col_s2 = np.zeros((lead, cols), dtype=np.float64)
    col_s1f = np.zeros((lead, cols), dtype=np.float64)
    col_nf = np.zeros((lead, cols), dtype=np.int64)

    rows_per_chunk = max(1, CHUNK_ELEMS // cols)
    for li in range(lead):
        for r0 in range(0, rows, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows)
            count = (r1 - r0) * cols
            t = read_trits(pack, entry, (li * rows + r0) * cols, count).reshape(-1, cols)
            w = np.asarray(view[li, r0:r1], dtype=np.float64)
            fired = t != 0
            aw = np.abs(w)
            af = np.where(fired, aw, 0.0)

            s2 += float(np.einsum("ij,ij->", w, w))
            s1_fired += float(af.sum())
            n_fired += int(np.count_nonzero(fired))
            sign_mismatch += int(np.count_nonzero(fired & ((w > 0) != (t > 0))))

            max_abs_fired = max(max_abs_fired, float(np.max(aw, where=fired, initial=0.0)))
            min_abs_fired = min(min_abs_fired, float(np.min(aw, where=fired, initial=math.inf)))
            max_abs_unfired = max(
                max_abs_unfired, float(np.max(aw, where=~fired, initial=0.0))
            )

            col_s2[li] += np.einsum("ij,ij->j", w, w)
            col_s1f[li] += af.sum(axis=0)
            col_nf[li] += fired.sum(axis=0)

    if sign_mismatch:
        raise MetricsError(
            f"{entry['name']}: {sign_mismatch} fired trits disagree in sign with the source "
            "weight -- pack does not correspond to this npy"
        )

    norm_w = math.sqrt(s2)
    alpha = s1_fired / n_fired if n_fired else 0.0
    cos = s1_fired / (norm_w * math.sqrt(n_fired)) if n_fired and norm_w else 0.0
    mse = (s2 - (s1_fired**2 / n_fired if n_fired else 0.0)) / n
    max_abs_err = max_abs_unfired
    if n_fired:
        max_abs_err = max(
            max_abs_err, abs(max_abs_fired - alpha), abs(min_abs_fired - alpha)
        )

    # Per-channel: optimal alpha_j and relative reconstruction error. Each channel
    # holds `rows` elements (one column of one leading slice).
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha_col = np.where(col_nf > 0, col_s1f / np.maximum(col_nf, 1), 0.0)
        mse_col = (
            col_s2 - np.where(col_nf > 0, col_s1f**2 / np.maximum(col_nf, 1), 0.0)
        ) / rows
        rms_col = np.sqrt(col_s2 / rows)
        rel_col = np.where(rms_col > 0, np.sqrt(np.maximum(mse_col, 0.0)) / rms_col, 0.0)

    return {
        "elements": n,
        "zeros": n - n_fired,
        "sparsity": (n - n_fired) / n,
        "rms": math.sqrt(s2 / n),
        "alpha_optimal": alpha,
        "weight_cosine_similarity": cos,
        "weight_reconstruction_mse": mse,
        "weight_nrmse": math.sqrt(mse) / math.sqrt(s2 / n) if s2 else 0.0,
        "weight_max_absolute_error": max_abs_err,
        "per_channel_scale_error": {
            "channels": int(alpha_col.size),
            "channel_definition": (
                f"(leading index, last axis) pairs: {lead} x {cols}, {rows} elements each"
            ),
            "alpha_min": float(alpha_col.min()),
            "alpha_median": float(np.median(alpha_col)),
            "alpha_max": float(alpha_col.max()),
            "relative_error_min": float(rel_col.min()),
            "relative_error_median": float(np.median(rel_col)),
            "relative_error_max": float(rel_col.max()),
        },
    }


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


def gate(observed: float, threshold: float, higher_is_better: bool = True) -> str:
    if observed is None:
        return "unknown"
    ok = observed >= threshold if higher_is_better else observed <= threshold
    return "pass" if ok else "fail"


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Route-preservation metrics for a GOZ1 block pilot")
    p.add_argument("--npy-dir", type=Path, required=True, help="f32 npy the pack was built from")
    p.add_argument("--pack", type=Path, required=True, help="GOZ1 pack to read back")
    p.add_argument("--block", type=int, required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--json-out", type=Path)
    args = p.parse_args(argv)

    metadata, index = load_pack_index(args.pack)

    ternary = {n: e for n, e in index.items() if e["tensor_type"] == TENSOR_TERNARY}
    preserve = {n: e for n, e in index.items() if e["tensor_type"] == TENSOR_F16}
    print(f"pack {args.pack.name}: {len(ternary)} ternary, {len(preserve)} preserve/fp16")

    # --- weight reconstruction, streamed over every quantized tensor ---
    weights: dict[str, dict] = {}
    for name in sorted(ternary):
        npy = args.npy_dir / f"{stem_of(name)}.npy"
        if not npy.exists():
            raise MetricsError(f"{name}: source npy missing at {npy}")
        st = ternary_stats(npy, args.pack, ternary[name])
        weights[name] = st
        print(
            f"  ternary {name:<46} zeros={st['sparsity'] * 100:6.2f}%  "
            f"cos={st['weight_cosine_similarity']:.6f}  nrmse={st['weight_nrmse']:.6f}"
        )

    # --- preserve tier fp16 round-trip (routers/norms are fp16-at-rest in GOZ1 v1) ---
    preserve_err: dict[str, dict] = {}
    for name in sorted(preserve):
        npy = args.npy_dir / f"{stem_of(name)}.npy"
        if not npy.exists():
            print(f"  warning: {name} has no source npy at {npy}; fp16 round-trip skipped")
            continue
        ref = np.load(npy).astype(np.float32)
        got = read_f16(args.pack, preserve[name])
        d = np.abs(ref - got)
        rms = float(np.sqrt((ref.astype(np.float64) ** 2).mean()))
        preserve_err[name] = {
            "max_absolute_error": float(d.max()),
            "relative_rmse": float(np.sqrt((d.astype(np.float64) ** 2).mean()) / rms) if rms else 0.0,
            "bit_exact": bool(np.array_equal(ref, got)),
        }
        print(
            f"  preserve {name:<45} fp16 max|err|={preserve_err[name]['max_absolute_error']:.3e}  "
            f"rel_rmse={preserve_err[name]['relative_rmse']:.3e}"
        )

    # --- routing, per d_model x d_model projection (roles unassigned upstream) ---
    # Located by suffix rather than a hardcoded slot, and every failure below is
    # fatal: a gate script that silently emits an all-`unknown` report would look
    # exactly like a passing pilot.
    routers = [n for n in sorted(preserve) if n.endswith(".router")]
    routing: dict[str, dict] = {}
    if len(routers) != 1:
        raise MetricsError(
            f"expected exactly one preserve-tier router in the pack, found {routers or 'none'}"
        )
    router_name = routers[0]
    router_ref = np.load(args.npy_dir / f"{stem_of(router_name)}.npy").astype(np.float32)
    router_pilot = read_f16(args.pack, preserve[router_name])
    if router_ref.ndim != 2 or router_ref.shape[0] <= router_ref.shape[1]:
        raise MetricsError(
            f"{router_name}: expected (d_model, experts) with d_model > experts, "
            f"got {router_ref.shape}; a transposed router would silently mis-size d_model"
        )
    d_model = router_ref.shape[0]

    norm_name = next((n for n in sorted(preserve) if n.endswith("block_norm")), None)
    if norm_name is None:
        raise MetricsError("no block_norm in pack; cannot build realistic activations")
    gain = np.load(args.npy_dir / f"{stem_of(norm_name)}.npy").astype(np.float32)
    if gain.shape != (d_model,):
        raise MetricsError(
            f"{norm_name}: shape {gain.shape} does not match router d_model {d_model}"
        )
    x = make_activations(gain, args.tokens, args.seed)

    square = [n for n, e in ternary.items() if list(e["shape"]) == [d_model, d_model]]
    if not square:
        raise MetricsError(
            f"no {d_model}x{d_model} projection among the pack's ternary tensors, so no "
            "routing gate can be evaluated; re-run with a mode that quantizes attention"
        )
    for name in sorted(square):
        w_ref = np.load(args.npy_dir / f"{stem_of(name)}.npy").astype(np.float32)
        w_pilot = reconstruct_full(
            args.pack, ternary[name], weights[name]["alpha_optimal"]
        )
        r = routing_metrics(x, w_ref, w_pilot, router_ref, router_pilot)
        r["activation_source"] = f"seeded N(0,1) tokens x RMSNorm gain {norm_name}"
        routing[name] = r
        print(
            f"  routing via {name:<42} top1={r['router_top1_agreement'] * 100:6.2f}%  "
            f"top2={r['router_top2_set_agreement'] * 100:6.2f}%  "
            f"cos={r['block_output_cosine']:.6f}"
        )

    summary = build_summary(weights, routing)

    result = {
        "model_family": "grok-1",
        "produced_by": "grok-ozempic scripts/route_preservation_metrics.py (GH #53 / RM-222)",
        "pilot": {
            "block": args.block,
            "mode": args.mode,
            "pack": str(args.pack),
            "npy_dir": str(args.npy_dir),
            "tokens": args.tokens,
            "seed": args.seed,
            "ternary_tensors": len(ternary),
            "preserve_tensors": len(preserve),
            "pack_metadata": {k: v for k, v in metadata.items() if k.startswith("oz.")},
            "ternary_scale": "least-squares optimal alpha = sum(|w| fired)/count(fired); GOZ1 v1 stores no scale",
            "activations": "seeded standard-normal tokens shaped by the block's real RMSNorm gain; no calibration corpus",
        },
        "summary": summary,
        "weights": weights,
        "preserve_fp16_roundtrip": preserve_err,
        "routing": routing,
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.json_out}")

    gated = [m for m in summary if m["threshold"]]
    print("\nroute-preservation gates:")
    for m in gated:
        obs = "null" if m["observed"] is None else f"{m['observed']:.6f}"
        print(f"  {m['name']:<28} {m['status']:>7}  observed={obs}  threshold={m['threshold']}")

    # Fail-closed: a threshold that is not `pass` must not exit 0, or a caller
    # cannot tell a passing pilot from a failing one.
    if any(m["status"] != "pass" for m in gated):
        print(f"\n{sum(m['status'] != 'pass' for m in gated)} of {len(gated)} gates not passing")
        return GATE_FAILURE_EXIT
    return 0


def build_summary(weights: dict[str, dict], routing: dict[str, dict]) -> list[dict]:
    """Assemble run3's route-preservation surface with observed values filled in."""

    def worst(key: str, default=None):
        vals = [r[key] for r in routing.values()]
        return min(vals) if vals else default

    top1 = worst("router_top1_agreement")
    top2 = worst("router_top2_set_agreement")
    bcos = worst("block_output_cosine")
    return [
        {
            "name": "router_top1_agreement",
            "scope": "router_behavior",
            "status": gate(top1, 0.99),
            "threshold": ">= 99.0%",
            "observed": top1,
            "detail": "Worst case over evaluated d_model x d_model projections; router read from the pack (fp16 preserve tier).",
        },
        {
            "name": "router_top2_set_agreement",
            "scope": "router_behavior",
            "status": gate(top2, 0.995),
            "threshold": ">= 99.5%",
            "observed": top2,
            "detail": "Fraction of tokens whose unordered top-2 expert set is unchanged.",
        },
        {
            "name": "expert_load_distribution_delta",
            "scope": "router_behavior",
            "status": "measured" if routing else "unknown",
            "threshold": None,
            "observed": max([r["expert_load_distribution_delta"] for r in routing.values()], default=None),
            "detail": "Max per-expert share change in the top-1 load histogram.",
        },
        {
            "name": "expert_load_js_divergence",
            "scope": "router_behavior",
            "status": "measured" if routing else "unknown",
            "threshold": None,
            "observed": max([r["expert_load_js_divergence"] for r in routing.values()], default=None),
            "detail": "Jensen-Shannon divergence (bits) between reference and pilot expert-load distributions.",
        },
        {
            "name": "router_logit_rank_correlation",
            "scope": "router_behavior",
            "status": "measured" if routing else "unknown",
            "threshold": None,
            "observed": worst("router_logit_rank_correlation"),
            "detail": "Mean per-token Spearman correlation over the 8 router logits.",
        },
        {
            "name": "block_output_cosine",
            "scope": "block_behavior",
            "status": gate(bcos, 0.995),
            "threshold": ">= 0.995",
            "observed": bcos,
            "detail": "Scoped to the quantized projection output, not a full block forward (attention roles are unassigned upstream; MoE not executed).",
        },
        {
            "name": "block_output_rmse",
            "scope": "block_behavior",
            "status": "measured" if routing else "unknown",
            "threshold": None,
            "observed": max([r["block_output_rmse"] for r in routing.values()], default=None),
            "detail": "RMSE of the projection output against the f32 reference path.",
        },
        {
            "name": "residual_stream_drift",
            "scope": "block_behavior",
            "status": "measured" if routing else "unknown",
            "threshold": None,
            "observed": max([r["residual_stream_drift"] for r in routing.values()], default=None),
            "detail": "Projection-output RMSE normalized by reference output RMS.",
        },
        {
            "name": "weight_reconstruction_mse",
            "scope": "weight_reconstruction",
            "status": "measured" if weights else "unknown",
            "threshold": None,
            "observed": max([w["weight_reconstruction_mse"] for w in weights.values()], default=None),
            "detail": "Worst quantized tensor, at the least-squares optimal ternary scale.",
        },
        {
            "name": "weight_cosine_similarity",
            "scope": "weight_reconstruction",
            "status": "measured" if weights else "unknown",
            "threshold": None,
            "observed": min([w["weight_cosine_similarity"] for w in weights.values()], default=None),
            "detail": "Worst quantized tensor; independent of the chosen ternary scale.",
        },
        {
            "name": "weight_max_absolute_error",
            "scope": "weight_reconstruction",
            "status": "measured" if weights else "unknown",
            "threshold": None,
            "observed": max([w["weight_max_absolute_error"] for w in weights.values()], default=None),
            "detail": "Exact max |w - alpha*t| over all quantized tensors.",
        },
        {
            "name": "per_channel_scale_error_summary",
            "scope": "weight_reconstruction",
            "status": "measured" if weights else "unknown",
            "threshold": None,
            "observed": max(
                [w["per_channel_scale_error"]["relative_error_max"] for w in weights.values()],
                default=None,
            ),
            "detail": "Worst per-output-channel relative reconstruction error; full per-tensor spread under `weights`.",
        },
        {
            "name": "logit_kl",
            "scope": "model_behavior",
            "status": "unknown",
            "threshold": None,
            "observed": None,
            "detail": "Requires whole-model inference; out of scope for a bounded single-block pilot (#53 non-goal).",
        },
        {
            "name": "perplexity_delta",
            "scope": "model_behavior",
            "status": "unknown",
            "threshold": None,
            "observed": None,
            "detail": "Requires a calibration corpus and whole-model inference; not run.",
        },
        {
            "name": "generation_sanity_summary",
            "scope": "model_behavior",
            "status": "unknown",
            "threshold": None,
            "observed": None,
            "detail": "Requires whole-model inference; not run.",
        },
    ]


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except MetricsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
