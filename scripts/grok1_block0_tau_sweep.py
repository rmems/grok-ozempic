#!/usr/bin/env python3
"""Attention-tier tau sweep through the genuine block-0 forward (GH #61 / RM-249).

Within one decoder block the router reads ``rmsnorm(h0 + attn_out)``, which is
entirely upstream of the experts, so expert quantization cannot move routing at
all -- measured and confirmed in ``grok1_block0_experiment``. This sweep
therefore varies only the attention tier and draws experts from the f32
reference, which isolates the one knob that can affect route preservation while
avoiding a 19 GiB re-quantization per tau.

Usage::

    python3 scripts/grok1_block0_tau_sweep.py \\
        --npy-dir ~/.models/xai-grok-1/export-npy/goz53-block000-attn \\
        --attn-npy-dir ~/.models/xai-grok-1/export-npy/goz61-block000-attn-only \\
        --pack-dir ~/.models/xai-grok-1/artifacts/block0-forward-tau \\
        --embedding-shard ~/.models/xai-grok-1/export-npy/embedding__slot_00__token_embedding.npy \\
        --tokens 512 --out reports/grok-1-full-block-forward
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grok1_block0_experiment import (  # noqa: E402
    Trace,
    compare,
    embedding_rows,
    expert_load,
    forward_block,
    router_margins,
    stem_of_inverse,
    token_ids,
)
from grok1_block_forward import NUM_SELECTED_EXPERTS, ForwardError  # noqa: E402
from route_preservation_io import MetricsError  # noqa: E402
from grok1_block_weights import (  # noqa: E402
    ATTENTION_ROLES,
    PRESERVED_ROLES,
    MixedWeights,
    NpyWeights,
    PackWeights,
    WeightSource,
    implementation_commit,
)

EXPECT_BLOCK = "block_000"
# Gaussian sparsity is only an approximation for real weights, so allow a wide
# band: this catches a mislabelled pack, not a modest distributional deviation.
_SPARSITY_TOLERANCE = 0.12
_TAU_RE = re.compile(r"tau-([0-9]*\.?[0-9]+)\.goz1$")


def tau_of(pack: Path) -> float:
    """Recover the tau a sweep pack was built with from its filename."""
    match = _TAU_RE.search(pack.name)
    if not match:
        raise ForwardError(f"{pack.name}: cannot parse a tau from the filename")
    return float(match.group(1))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--npy-dir", type=Path, required=True, help="full f32 reference block")
    p.add_argument("--attn-npy-dir", type=Path, required=True, help="attention-tier npy dir")
    p.add_argument("--pack-dir", type=Path, required=True, help="dir of attn-tau-*.goz1 packs")
    p.add_argument("--embedding-shard", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260806)
    p.add_argument("--top-k", type=int, default=NUM_SELECTED_EXPERTS)
    p.add_argument("--out", type=Path, required=True)
    return p


def expected_sparsity(tau: float) -> float:
    """Fraction of a standard normal within +/-tau: the sparsity tau implies.

    GOZ1 sets tau = gif_threshold * rms, so for weights that are approximately
    Gaussian the zeroed fraction is erf(tau / sqrt(2)). Used only to sanity-check
    the tau parsed from a filename against what the pack actually contains.
    """
    return math.erf(tau / math.sqrt(2.0))


def _require_tau_matches_pack(pack_path: Path, tau: float, sparsity: dict[str, float]) -> None:
    """Reject a pack whose measured firing rate contradicts its filename tau.

    The tau is parsed from the filename, so a pack rebuilt at a different
    threshold under the same name would be reported under the wrong tau and
    silently corrupt the sweep. ``oz.gif_threshold`` cannot be used for this (the
    #51 trap: it records defaults||config, not the applied per-tensor value), so
    the check is against measured sparsity. The tolerance is deliberately loose
    because real weights are only approximately Gaussian -- it catches a
    mislabelled pack, not a small distributional deviation.
    """
    if not sparsity:
        return
    observed = sum(sparsity.values()) / len(sparsity)
    expected = expected_sparsity(tau)
    if abs(observed - expected) > _SPARSITY_TOLERANCE:
        raise ForwardError(
            f"{pack_path.name}: filename says tau={tau} (expected sparsity "
            f"~{expected:.3f}) but the pack fires at sparsity {observed:.3f}; "
            "the pack was probably rebuilt at a different threshold"
        )


def _sparsity(pack: PackWeights) -> dict[str, float]:
    """Observed ternary sparsity per attention tensor -- the applied tau's fingerprint.

    ``oz.gif_threshold`` in pack metadata records only ``defaults || config`` and is
    unreliable under per-tensor tau (the #51 trap), so the applied threshold is
    verified from measured firing rates instead.
    """
    return {
        name.split(".", 1)[1]: pack.scale(name).sparsity
        for name in sorted(pack.roles[r] for r in ATTENTION_ROLES)
    }


def _sweep_row(
    pack_path: Path,
    attn_npy_dir: Path,
    reference: WeightSource,
    ref: Trace,
    h0: np.ndarray,
    top_k: int,
) -> dict:
    """Measure one tau pack: attention from the pack, experts from the reference."""
    tau = tau_of(pack_path)
    pack = PackWeights(pack_path, attn_npy_dir, partial=True, expect_block=EXPECT_BLOCK)
    pack.require_preserved(PRESERVED_ROLES)
    mixed = MixedWeights(pack, reference, ATTENTION_ROLES, f"attn_tau_{tau}")
    trace = forward_block(h0, mixed, top_k=top_k)
    sparsity = _sparsity(pack)
    _require_tau_matches_pack(pack_path, tau, sparsity)
    row = {
        "tau": tau,
        "pack": pack_path.name,
        "pack_sha256": pack.pack_sha256(),
        "pack_bytes": pack_path.stat().st_size,
        "sparsity": sparsity,
        "sparsity_expected_gaussian": expected_sparsity(tau),
        **compare(ref, trace),
    }
    print(
        f"  tau={tau:<4} top1={row['router_top1_agreement']:.4f} "
        f"top2={row['router_top2_set_agreement']:.4f} "
        f"attn_cos={row['attention_output_cosine']:.4f} "
        f"block_cos={row['block_output_cosine']:.4f}",
        flush=True,
    )
    return row


def _payload(args: argparse.Namespace, reference, ref, ids: np.ndarray, rows: list[dict]) -> dict:
    """Assemble the machine-readable sweep artifact with full provenance."""
    return {
        "provenance": {
            "issue": "GH #61 / Linear RM-249",
            "agent": "Claude Code: Fable 5 (xhigh)",
            "implementation": implementation_commit(),
            "architecture_source": "github.com/xai-org/grok-1 model.py + run.py",
            "npy_dir": args.npy_dir.name,
            "attn_npy_dir": args.attn_npy_dir.name,
            "embedding_shard": args.embedding_shard.name,
            "expect_block": EXPECT_BLOCK,
            "python": platform.python_version(),
            "tau_source": (
                "parsed from each pack filename and cross-checked against measured "
                "sparsity vs erf(tau/sqrt2); oz.gif_threshold is unreliable under "
                "per-tensor tau (see #66)"
            ),
            "design": (
                "Attention tier only; experts held at the f32 reference because "
                "single-block routing is computed upstream of the experts."
            ),
            "tokens": int(ids.size),
            "token_seed": int(args.seed),
            "top_k": int(args.top_k),
            "numpy": np.__version__,
        },
        "slot_roles": dict(sorted(reference.roles.items())),
        "reference": {
            "expert_load": expert_load(ref.expert_idx).tolist(),
            "router_margin_median": float(np.median(router_margins(ref.logits))),
        },
        "sweep": rows,
    }


def run(args: argparse.Namespace) -> int:
    if args.tokens < 1:
        raise ForwardError(f"tokens must be >= 1, got {args.tokens}")
    packs = sorted(args.pack_dir.glob("attn-tau-*.goz1"), key=tau_of)
    if not packs:
        raise ForwardError(f"{args.pack_dir}: no attn-tau-*.goz1 packs found")
    names = [stem_of_inverse(p) for p in sorted(args.npy_dir.glob("*.npy"))]
    reference = NpyWeights(args.npy_dir, names, expect_block=EXPECT_BLOCK)
    ids = token_ids(args.tokens, args.seed, vocab=131072)
    h0 = embedding_rows(args.embedding_shard, ids)

    print(f"reference forward over {ids.size} tokens ...", flush=True)
    ref = forward_block(h0, reference, top_k=args.top_k)
    rows = [
        _sweep_row(p, args.attn_npy_dir, reference, ref, h0, args.top_k) for p in packs
    ]

    payload = _payload(args, reference, ref, ids, rows)
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "block0-forward-tau-sweep.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {dest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (ForwardError, MetricsError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
