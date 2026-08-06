#!/usr/bin/env python3
"""Genuine Grok-1 block-0 forward route-preservation experiment (GH #61 / RM-249).

Replaces the single-projection routing proxy of #57 with the real block body:
pre-attention RMSNorm, grouped-query attention with RoPE and the tanh soft-cap,
the output projection, the residual add, the pre-MoE RMSNorm that actually feeds
the router, the preserved router, and the top-2 GeGLU expert forward.

Every weight comes from the real checkpoint. Routers and norms are preserved in
the pack, so any routing damage is attributable to ternary attention/expert
weights rather than to a quantized router.

Usage::

    python3 scripts/grok1_block0_experiment.py \\
        --npy-dir  ~/.models/xai-grok-1/export-npy/goz53-block000-attn \\
        --pack     ~/.models/xai-grok-1/artifacts/block-pilot/block_000-attention_plus_expert.goz1 \\
        --embedding-shard ~/.models/xai-grok-1/export-npy/embedding__slot_00__token_embedding.npy \\
        --tokens 512 --out reports/grok-1-full-block-forward
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grok1_block_forward import (  # noqa: E402
    EMBEDDING_MULTIPLIER,
    NUM_EXPERTS,
    NUM_SELECTED_EXPERTS,
    ForwardError,
    UnresolvedArchitectureError,
    attention,
    expert_geglu,
    moe_combine,
    rmsnorm,
    router_logits,
    top_k_experts,
)
from grok1_block_weights import (  # noqa: E402
    ATTENTION_ROLES,
    EXPERT_ROLES,
    PRESERVED_ROLES,
    F16Weights,
    MixedWeights,
    NpyWeights,
    PackWeights,
    WeightSource,
    implementation_commit,
    sha256_file,
)
from route_preservation_io import MetricsError  # noqa: E402
from route_preservation_measure import js_divergence  # noqa: E402

EXIT_GATE_FAILURE = 3
# Distinct from 1 (operational failure) so a caller can tell an unresolved
# architecture -- RM-249 conclusion 4 -- from a missing file or bad path.
EXIT_UNRESOLVED = 4
# Block-0 activations are the token-embedding rows; for any other block those
# rows are not the residual stream, so the block identity is pinned.
EXPECT_BLOCK = "block_000"
# Router-margin bands for the flip stratification (fraction of top-1 probability).
_MARGIN_BANDS = ((0.0, 0.01), (0.01, 0.05), (0.05, 0.15), (0.15, 0.5), (0.5, 1.01))


@dataclass
class Trace:
    """Intermediates captured from one forward pass through block 0."""

    label: str
    attn_out: np.ndarray
    residual: np.ndarray
    moe_input: np.ndarray
    logits: np.ndarray
    expert_idx: np.ndarray
    gates: np.ndarray
    moe_out: np.ndarray
    block_out: np.ndarray
    # Post-norm sublayer deltas, i.e. exactly what is added to the residual.
    attn_delta: np.ndarray
    moe_delta: np.ndarray
    stream_in: np.ndarray
    seconds: float = 0.0
    experts_touched: list[int] = field(default_factory=list)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Global cosine similarity between two arrays, flattened."""
    x, y = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / denom) if denom else 0.0


def mean_row_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-token cosine -- a stricter view than the global cosine."""
    x, y = np.asarray(a, np.float64), np.asarray(b, np.float64)
    num = (x * y).sum(axis=1)
    den = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.where(den > 0, num / den, 1.0).mean())


def relative_drift(reference: np.ndarray, other: np.ndarray) -> float:
    """``||other - reference|| / ||reference||`` over the whole tensor."""
    ref = np.asarray(reference, np.float64)
    denom = float(np.linalg.norm(ref))
    return float(np.linalg.norm(np.asarray(other, np.float64) - ref) / denom) if denom else 0.0


def contribution_ratio(sublayer: np.ndarray, stream: np.ndarray) -> float:
    """``||sublayer|| / ||stream||`` -- how much a sublayer moves the residual.

    The key explanatory quantity for block 0: the residual entering the block is
    the raw embedding, so if the normalized attention output is small next to it,
    heavy attention damage is diluted before the router ever sees it.
    """
    denom = float(np.linalg.norm(np.asarray(stream, np.float64)))
    return float(np.linalg.norm(np.asarray(sublayer, np.float64)) / denom) if denom else 0.0


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, np.float64) - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def router_margins(logits: np.ndarray) -> np.ndarray:
    """Per-token top-1 minus top-2 routing probability."""
    probs = np.sort(_softmax(logits), axis=-1)
    return probs[:, -1] - probs[:, -2]


def expert_load(idx: np.ndarray) -> np.ndarray:
    """Fraction of (token, slot) assignments landing on each expert."""
    counts = np.bincount(np.asarray(idx).ravel(), minlength=NUM_EXPERTS).astype(np.float64)
    return counts / counts.sum() if counts.sum() else counts


# --- Forward pass -----------------------------------------------------------
def _moe_forward(
    hn: np.ndarray, source: WeightSource, top_k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Router + streamed top-k expert forward. Loads one expert at a time.

    Returns ``(logits, expert_idx, gates, moe_out, experts_touched)``.
    """
    logits = router_logits(hn, source.vector("router"))
    idx, gates = top_k_experts(logits, k=top_k)
    outputs: dict[int, np.ndarray] = {}
    touched: list[int] = []
    for expert in range(NUM_EXPERTS):
        rows = np.argwhere(idx == expert)[:, 0]
        if rows.size == 0:
            continue
        touched.append(expert)
        w_gelu = source.expert("expert_gelu", expert)
        w_value = source.expert("expert_value", expert)
        w_down = source.expert("expert_down", expert)
        outputs[expert] = expert_geglu(hn[rows], w_gelu, w_value, w_down)
        del w_gelu, w_value, w_down
    moe_out = moe_combine(outputs, idx, gates, hn.shape[0])
    return logits, idx, gates, moe_out, touched


def forward_block(
    h0: np.ndarray, source: WeightSource, *, top_k: int = NUM_SELECTED_EXPERTS
) -> Trace:
    """Run the real block-0 body and capture every intermediate #61 measures."""
    started = time.time()
    hn_attn = rmsnorm(h0, source.vector("norm_pre_attn"))
    attn_out = attention(
        hn_attn,
        source.matrix("query"),
        source.matrix("key"),
        source.matrix("value"),
        source.matrix("attn_out"),
    )
    # DecoderLayer normalizes the sublayer output *before* the residual add.
    attn_delta = rmsnorm(attn_out, source.vector("norm_post_attn"))
    residual = h0 + attn_delta
    moe_input = rmsnorm(residual, source.vector("norm_pre_moe"))
    logits, idx, gates, moe_out, touched = _moe_forward(moe_input, source, top_k)
    moe_delta = rmsnorm(moe_out, source.vector("norm_post_moe"))
    block_out = residual + moe_delta
    return Trace(
        label=source.label,
        attn_out=attn_out,
        residual=residual,
        moe_input=moe_input,
        logits=logits,
        expert_idx=idx,
        gates=gates,
        moe_out=moe_out,
        block_out=block_out,
        attn_delta=attn_delta,
        moe_delta=moe_delta,
        stream_in=h0,
        seconds=time.time() - started,
        experts_touched=touched,
    )


# --- Metrics ----------------------------------------------------------------
def _flip_stratification(ref: Trace, other: Trace) -> list[dict]:
    """Top-1 flip rate bucketed by the *reference* router margin.

    Separates near-tie flips, which barely change the computation, from
    high-confidence flips, which indicate real routing damage.
    """
    margins = router_margins(ref.logits)
    flipped = other.expert_idx[:, 0] != ref.expert_idx[:, 0]
    rows = []
    for low, high in _MARGIN_BANDS:
        in_band = (margins >= low) & (margins < high)
        n = int(in_band.sum())
        rows.append(
            {
                "margin_low": low,
                "margin_high": high,
                "tokens": n,
                "top1_flips": int(flipped[in_band].sum()),
                "top1_flip_rate": float(flipped[in_band].mean()) if n else None,
            }
        )
    return rows


def _routing_agreement(ref: Trace, other: Trace) -> dict:
    top1 = float((other.expert_idx[:, 0] == ref.expert_idx[:, 0]).mean())
    ref_sets = [frozenset(r) for r in ref.expert_idx]
    other_sets = [frozenset(r) for r in other.expert_idx]
    topk = float(np.mean([a == b for a, b in zip(ref_sets, other_sets)]))
    load_ref, load_other = expert_load(ref.expert_idx), expert_load(other.expert_idx)
    selected = int(ref.expert_idx.shape[1])
    agreement = {
        "selected_experts": selected,
        "router_topk_set_agreement": topk,
    }
    # Only publish the gate-named metric when k really is Grok-1's top-2, so a
    # --top-k override cannot produce a number labelled "top-2" that is not.
    if selected == NUM_SELECTED_EXPERTS:
        agreement["router_top2_set_agreement"] = topk
    return {
        "router_top1_agreement": top1,
        **agreement,
        "expert_load_reference": load_ref.tolist(),
        "expert_load_observed": load_other.tolist(),
        "expert_load_js_bits": js_divergence(load_ref, load_other),
        "expert_load_max_abs_delta": float(np.abs(load_other - load_ref).max()),
        "experts_unused_reference": int((load_ref == 0).sum()),
        "experts_unused_observed": int((load_other == 0).sum()),
    }


def _margin_summary(ref: Trace, other: Trace) -> dict:
    before, after = router_margins(ref.logits), router_margins(other.logits)
    return {
        "router_margin_mean_reference": float(before.mean()),
        "router_margin_mean_observed": float(after.mean()),
        "router_margin_mean_delta": float((after - before).mean()),
        "router_margin_median_reference": float(np.median(before)),
        "router_margin_median_observed": float(np.median(after)),
    }


def compare(ref: Trace, other: Trace) -> dict:
    """The eleven #61 measurements plus the margin-stratified flip table."""
    return {
        "label": other.label,
        "seconds": other.seconds,
        "attention_output_cosine": cosine(ref.attn_out, other.attn_out),
        "attention_output_cosine_per_token": mean_row_cosine(ref.attn_out, other.attn_out),
        "residual_stream_cosine": cosine(ref.residual, other.residual),
        "moe_input_normalized_cosine": cosine(ref.moe_input, other.moe_input),
        "router_logit_cosine": cosine(ref.logits, other.logits),
        "moe_output_cosine": cosine(ref.moe_out, other.moe_out),
        "block_output_cosine": cosine(ref.block_out, other.block_out),
        "block_output_cosine_per_token": mean_row_cosine(ref.block_out, other.block_out),
        "residual_drift_relative_norm": relative_drift(ref.residual, other.residual),
        "block_output_drift_relative_norm": relative_drift(ref.block_out, other.block_out),
        "attn_delta_over_stream": contribution_ratio(other.attn_delta, other.stream_in),
        "moe_delta_over_stream": contribution_ratio(other.moe_delta, other.residual),
        **_routing_agreement(ref, other),
        **_margin_summary(ref, other),
        "flip_stratification_by_reference_margin": _flip_stratification(ref, other),
        "experts_touched": other.experts_touched,
    }


# --- Activations ------------------------------------------------------------
def token_ids(count: int, seed: int, vocab: int) -> np.ndarray:
    """Deterministic token ids sampled uniformly without replacement.

    A uniform draw over the vocabulary covers the embedding distribution far more
    evenly than natural text, which concentrates on a few thousand frequent
    tokens. It is therefore a broader routing probe -- but it is *not* a natural
    sequence, so absolute attention statistics should be read with that in mind.
    """
    return np.sort(np.random.default_rng(seed).choice(vocab, size=count, replace=False))


def embedding_rows(shard: Path, ids: np.ndarray) -> np.ndarray:
    """Fetch the residual stream entering block 0 for ``ids``.

    For block 0 the residual stream *is* the token-embedding lookup times
    ``EMBEDDING_MULTIPLIER`` -- no calibration corpus and no upstream forward pass
    required. The multiplier is mandatory here: it puts the stream at unit RMS so
    the residual add is on the same scale as the post-norm sublayer deltas. The
    3.1 GiB table is memory-mapped and only the sampled rows are materialized.
    """
    table = np.load(shard, mmap_mode="r")
    if ids.max(initial=0) >= table.shape[0]:
        raise ForwardError(f"{shard}: token id {int(ids.max())} exceeds vocab {table.shape[0]}")
    rows = np.asarray(table[ids], dtype=np.float32)
    return rows * np.float32(EMBEDDING_MULTIPLIER)


# --- Reporting --------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--npy-dir", type=Path, required=True, help="dequantized f32 reference block")
    p.add_argument("--pack", type=Path, required=True, help="GOZ1 pack for the same block")
    p.add_argument("--embedding-shard", type=Path, required=True, help="token-embedding .npy")
    p.add_argument("--tokens", type=int, default=512, help="sequence length (default 512)")
    p.add_argument("--seed", type=int, default=20260806, help="token-sample seed")
    p.add_argument("--top-k", type=int, default=NUM_SELECTED_EXPERTS)
    p.add_argument("--out", type=Path, required=True, help="report directory")
    p.add_argument(
        "--skip-fp16-control",
        action="store_true",
        help="skip the fp16 round-trip baseline (not recommended: it validates the harness)",
    )
    return p


def _provenance(args: argparse.Namespace, ids: np.ndarray, pack: PackWeights) -> dict:
    return {
        "issue": "GH #61 / Linear RM-249",
        "agent": "Claude Code: Fable 5 (xhigh)",
        "implementation": implementation_commit(),
        "architecture_source": "github.com/xai-org/grok-1 model.py + run.py",
        "pack": pack.pack.name,
        "pack_sha256": sha256_file(pack.pack),
        "pack_bytes": pack.pack.stat().st_size,
        "pack_metadata": {k: v for k, v in sorted(pack.metadata.items())},
        # [15] pack_metadata is reproduced verbatim, and oz.gif_threshold in it is
        # known-wrong under per-tensor tau (#51 / #66): it records defaults||config,
        # not what was applied. Trust the measured sparsity in oracle_alpha instead.
        "pack_metadata_caveat": (
            "oz.gif_threshold is unreliable under per-tensor tau (records "
            "defaults||config only, see #66); derive the applied threshold from "
            "oracle_alpha[].sparsity"
        ),
        "npy_dir": args.npy_dir.name,
        "embedding_shard": args.embedding_shard.name,
        "tokens": int(ids.size),
        "token_seed": int(args.seed),
        "token_id_first": int(ids[0]),
        "token_id_last": int(ids[-1]),
        "top_k": int(args.top_k),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "oracle_alpha": {
            name: {
                "alpha": s.alpha,
                "sparsity": s.sparsity,
                "fired": s.fired,
                "total": s.total,
                # Nonzero would mean the pack's trits disagree in sign with the
                # reference weights, invalidating the "best case" claim.
                "sign_mismatches": s.sign_mismatches,
            }
            for name, s in sorted(pack.scales().items())
        },
        "ternary_scale_note": (
            "GOZ1 stores no per-tensor scale, so ternary weights are reconstructed with the "
            "least-squares optimal alpha derived from the original f32 weights. This is an "
            "oracle a real runtime does not have, so every number here is a best case."
        ),
    }


def _baselines(
    h0: np.ndarray,
    ref_trace: Trace,
    reference: NpyWeights,
    pack: PackWeights,
    args: argparse.Namespace,
    names: list[str],
) -> list[dict]:
    """Run every comparison baseline #61 requires against the reference trace."""
    results: list[dict] = []
    if not args.skip_fp16_control:
        print("fp16 round-trip control ...", flush=True)
        control = forward_block(h0, F16Weights(args.npy_dir, names), top_k=args.top_k)
        results.append(compare(ref_trace, control))
        print(f"  {control.seconds:.1f}s", flush=True)

    print("GOZ1 pack forward ...", flush=True)
    goz1 = forward_block(h0, pack, top_k=args.top_k)
    results.append(compare(ref_trace, goz1))
    print(f"  {goz1.seconds:.1f}s", flush=True)

    # Tier attribution: routing is computed upstream of the experts, so the
    # expert-only run is expected to reproduce reference routing exactly.
    for roles, label in (
        (ATTENTION_ROLES, "goz1_attention_ternary_only"),
        (EXPERT_ROLES, "goz1_expert_ternary_only"),
    ):
        print(f"{label} ...", flush=True)
        mixed = MixedWeights(pack, reference, roles, label)
        trace = forward_block(h0, mixed, top_k=args.top_k)
        results.append(compare(ref_trace, trace))
        print(f"  {trace.seconds:.1f}s", flush=True)
    return results


def _reference_summary(ref_trace: Trace) -> dict:
    margins = router_margins(ref_trace.logits)
    return {
        "expert_load": expert_load(ref_trace.expert_idx).tolist(),
        "router_margin_mean": float(margins.mean()),
        "router_margin_median": float(np.median(margins)),
        "experts_touched": ref_trace.experts_touched,
        "attn_delta_over_stream": contribution_ratio(ref_trace.attn_delta, ref_trace.stream_in),
        "moe_delta_over_stream": contribution_ratio(ref_trace.moe_delta, ref_trace.residual),
        "seconds": ref_trace.seconds,
    }


def run(args: argparse.Namespace) -> int:
    if args.tokens < 1:
        raise ForwardError(f"tokens must be >= 1, got {args.tokens}")
    ids = token_ids(args.tokens, args.seed, vocab=131072)
    names = [stem_of_inverse(p) for p in sorted(args.npy_dir.glob("*.npy"))]
    reference = NpyWeights(args.npy_dir, names, expect_block=EXPECT_BLOCK)
    h0 = embedding_rows(args.embedding_shard, ids)

    print(f"reference forward over {ids.size} tokens ...", flush=True)
    ref_trace = forward_block(h0, reference, top_k=args.top_k)
    print(f"  {ref_trace.seconds:.1f}s, experts touched {ref_trace.experts_touched}", flush=True)

    pack = PackWeights(args.pack, args.npy_dir, expect_block=EXPECT_BLOCK)
    # Premise of the whole experiment: the router and all four norms are
    # preserved, so any drift is attributable to the ternary tiers.
    pack.require_preserved(PRESERVED_ROLES)
    results = _baselines(h0, ref_trace, reference, pack, args, names)

    payload = {
        "provenance": _provenance(args, ids, pack),
        "slot_roles": dict(sorted(reference.roles.items())),
        "reference": _reference_summary(ref_trace),
        "comparisons": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_json = args.out / "block0-forward-metrics.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_json}")
    return 0


def stem_of_inverse(path: Path) -> str:
    """``block_000__slot_11__router.npy`` -> ``block_000.slot_11.router``."""
    return path.stem.replace("__", ".")


def write_unresolved_report(out: Path, reason: str) -> Path:
    """Emit RM-249's bounded conclusion 4 when the architecture cannot be resolved.

    Without this, a ForwardError from role resolution exits non-zero with no
    artifact, making "an architectural element could not be resolved" -- a real
    research outcome the issue asks for by name -- indistinguishable from an
    operational failure such as a missing file. Failing closed and recording the
    conclusion are not in tension: the exit code still signals failure, and the
    report says *which* kind.
    """
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "block0-forward-unresolved.json"
    dest.write_text(
        json.dumps(
            {
                "provenance": {
                    "issue": "GH #61 / Linear RM-249",
                    "agent": "Claude Code: Fable 5 (xhigh)",
                    "implementation": implementation_commit(),
                    "expect_block": EXPECT_BLOCK,
                },
                "conclusion": 4,
                "conclusion_text": (
                    "result remains inconclusive because an explicitly named "
                    "architectural element could not be resolved"
                ),
                "unresolved_reason": reason,
                "measurements": None,
            },
            indent=2,
        )
        + "\n"
    )
    return dest


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except UnresolvedArchitectureError as exc:
        # Only a genuine slot/shape/block contradiction is conclusion 4. A missing
        # export or bad path raises plain ForwardError below and stays exit 1, so
        # the two are not conflated -- which was the point of the finding.
        print(f"error: {exc}", file=sys.stderr)
        dest = write_unresolved_report(args.out, str(exc))
        print(f"wrote conclusion-4 report to {dest}", file=sys.stderr)
        return EXIT_UNRESOLVED
    except (ForwardError, MetricsError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
