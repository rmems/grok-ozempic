#!/usr/bin/env python3
"""Expert-only ternary multi-block residual fidelity (GH #68 / RM-255).

Sequential short chain with paired residual trajectories. Experts are GOZ1
ternary (v3 pack-only); attention, routers, and norms stay high-precision.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grok1_block0_experiment import (  # noqa: E402
    EXIT_UNRESOLVED,
    compare,
    embedding_rows,
    forward_block,
    token_ids,
)
from grok1_block_forward import (  # noqa: E402
    NUM_SELECTED_EXPERTS,
    ForwardError,
    UnresolvedArchitectureError,
)
from grok1_block_weights import implementation_commit  # noqa: E402
from grok1_multiblock_lib import (  # noqa: E402
    AGENT_LINE,
    BASELINE_64,
    decide,
    load_block_sources,
    pack_provenance_row,
    parse_blocks,
    require_pack_only_scales,
    residual_stream_metrics,
    resolve_path,
    write_results_md,
)
from route_preservation_io import MetricsError  # noqa: E402

# Re-export for unit tests that import from this module.
__all__ = [
    "decide",
    "parse_blocks",
    "require_pack_only_scales",
    "residual_stream_metrics",
    "write_results_md",
    "main",
]

EXIT_LEGACY_ORACLE = 5
EXIT_OK = 0
EXIT_OP = 1


@dataclass(frozen=True)
class ChainPaths:
    """Path patterns for npy dirs and packs along the block chain."""

    npy_root: Path
    npy_pattern: str
    pack_root: Path
    pack_pattern: str
    embedding_shard: Path


def _validate_embedding_shard(shard: Path, ids: np.ndarray) -> None:
    """Fail closed if the embedding table is not a 2-D float matrix."""
    table = np.load(shard, mmap_mode="r")
    if table.ndim != 2:
        raise ForwardError(f"{shard}: expected 2-D embedding table, got shape {table.shape}")
    if table.shape[1] != 6144:
        raise ForwardError(
            f"{shard}: expected model width 6144, got {table.shape[1]}"
        )
    if ids.max(initial=0) >= table.shape[0]:
        raise ForwardError(
            f"{shard}: token id {int(ids.max())} exceeds vocab {table.shape[0]}"
        )


def _run_block(
    b: int,
    paths: ChainPaths,
    h_ref: np.ndarray,
    h_pilot: np.ndarray,
    h_fp16: np.ndarray | None,
    top_k: int,
    skip_fp16: bool,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray | None, dict]:
    """Forward one block for ref / pilot / optional fp16; return metrics + new streams."""
    npy_dir = resolve_path(paths.npy_root, paths.npy_pattern, b)
    pack_path = resolve_path(paths.pack_root, paths.pack_pattern, b)
    if not npy_dir.is_dir():
        raise ForwardError(f"missing npy dir for block {b}: {npy_dir}")
    if not pack_path.is_file():
        raise ForwardError(f"missing pack for block {b}: {pack_path}")

    print(f"== block {b:03d}  residual_in shape={h_ref.shape}", flush=True)
    reference, pack, mixed, control = load_block_sources(
        b, npy_dir, pack_path, require_fp16=not skip_fp16
    )
    stream_pilot = residual_stream_metrics(h_ref, h_pilot)
    stream_fp16 = residual_stream_metrics(h_ref, h_fp16) if h_fp16 is not None else None

    print("  reference forward ...", flush=True)
    ref_trace = forward_block(h_ref, reference, top_k=top_k)
    print(f"    {ref_trace.seconds:.1f}s experts={ref_trace.experts_touched}", flush=True)

    print("  expert-only ternary ...", flush=True)
    pilot_trace = forward_block(h_pilot, mixed, top_k=top_k)
    pilot_cmp = compare(ref_trace, pilot_trace)
    pilot_cmp["residual_stream_in"] = stream_pilot
    print(
        f"    {pilot_trace.seconds:.1f}s  block_out_cos={pilot_cmp['block_output_cosine']:.6f} "
        f"top1={pilot_cmp['router_top1_agreement']:.6f} "
        f"resid_in_drift={stream_pilot['residual_in_drift_relative_norm']:.6f}",
        flush=True,
    )

    fp16_cmp = None
    next_fp16 = h_fp16
    if control is not None and h_fp16 is not None:
        print("  fp16 control ...", flush=True)
        fp16_trace = forward_block(h_fp16, control, top_k=top_k)
        fp16_cmp = compare(ref_trace, fp16_trace)
        fp16_cmp["residual_stream_in"] = stream_fp16
        print(
            f"    {fp16_trace.seconds:.1f}s  block_out_cos={fp16_cmp['block_output_cosine']:.6f}",
            flush=True,
        )
        next_fp16 = fp16_trace.block_out

    row = {
        "block": b,
        "reference_seconds": ref_trace.seconds,
        "expert_only": pilot_cmp,
        "fp16_control": fp16_cmp,
    }
    prov = pack_provenance_row(b, pack_path, npy_dir, pack)
    return row, ref_trace.block_out, pilot_trace.block_out, next_fp16, prov


def run_chain(
    blocks: list[int],
    paths: ChainPaths,
    *,
    tokens: int,
    seed: int,
    top_k: int,
    skip_fp16: bool,
) -> dict:
    """Run FP reference, expert-only, and (unless skipped) FP16 control chains."""
    if blocks[0] != 0:
        raise ForwardError(f"chain must start at block 0 (got blocks={blocks})")
    ids = token_ids(tokens, seed, vocab=131072)
    _validate_embedding_shard(paths.embedding_shard, ids)
    h0 = embedding_rows(paths.embedding_shard, ids)

    h_ref, h_pilot = h0, h0.copy()
    h_fp16 = h0.copy() if not skip_fp16 else None
    per_block: list[dict] = []
    pack_provenance: list[dict] = []

    for b in blocks:
        row, h_ref, h_pilot, h_fp16, prov = _run_block(
            b, paths, h_ref, h_pilot, h_fp16, top_k, skip_fp16
        )
        per_block.append(row)
        pack_provenance.append(prov)

    end = {
        "expert_only_end_residual_in": residual_stream_metrics(h_ref, h_pilot),
        "fp16_end_residual_in": residual_stream_metrics(h_ref, h_fp16)
        if h_fp16 is not None
        else None,
    }
    return {
        "blocks": blocks,
        "tokens": int(ids.size),
        "token_seed": int(seed),
        "token_id_first": int(ids[0]),
        "token_id_last": int(ids[-1]),
        "top_k": int(top_k),
        "per_block": per_block,
        "end_of_chain": end,
        "pack_provenance": pack_provenance,
        "skip_fp16_control": bool(skip_fp16),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--blocks", default="0,1,2,3")
    p.add_argument("--npy-root", type=Path, required=True)
    p.add_argument("--npy-pattern", default="goz68-block_{block:03d}-attn")
    p.add_argument("--pack-root", type=Path, required=True)
    p.add_argument("--pack-pattern", default="block_{block:03d}-attention_plus_expert.goz1")
    p.add_argument("--embedding-shard", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=20260806)
    p.add_argument("--top-k", type=int, default=NUM_SELECTED_EXPERTS)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--skip-fp16-control", action="store_true")
    p.add_argument("--write-report-md", action="store_true")
    return p


def run(args: argparse.Namespace) -> int:
    if args.tokens < 1:
        raise ForwardError(f"tokens must be >= 1, got {args.tokens}")
    blocks = parse_blocks(args.blocks)
    paths = ChainPaths(
        npy_root=args.npy_root.expanduser(),
        npy_pattern=args.npy_pattern,
        pack_root=args.pack_root.expanduser(),
        pack_pattern=args.pack_pattern,
        embedding_shard=args.embedding_shard.expanduser(),
    )
    args.out = args.out.expanduser()
    chain = run_chain(
        blocks,
        paths,
        tokens=args.tokens,
        seed=args.seed,
        top_k=args.top_k,
        skip_fp16=args.skip_fp16_control,
    )
    decision = decide(chain)
    payload = {
        "provenance": {
            "issue": "GH #68 / Linear RM-255",
            "agent": AGENT_LINE,
            "model": "grok-4.5",
            "design": "Grok Build super-research design (sequential chain, pack-only v3)",
            "baseline_64": BASELINE_64,
            "implementation": implementation_commit(),
            "architecture_source": "github.com/xai-org/grok-1 model.py + run.py",
            "numpy": np.__version__,
            "python": platform.python_version(),
            "embedding_shard": str(paths.embedding_shard),
            "skip_fp16_control": bool(args.skip_fp16_control),
            "activation_policy": (
                "paired residual trajectories; block0=embed*EMBEDDING_MULTIPLIER; "
                "no Gaussian; no embedding rows for b!=0"
            ),
            "ternary_policy": "experts only; attention/routers/norms high precision",
            "scale_policy": "GOZ1 v3 pack-only; abort on legacy_oracle",
        },
        "chain": chain,
        "decision": decision,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_json = args.out / "metrics.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_json}")
    print(f"DECISION option {decision['decision']}: {decision['decision_text']}")
    if args.write_report_md or not args.skip_fp16_control:
        md = args.out / "results.md"
        write_results_md(md, payload)
        print(f"wrote {md}")
    return EXIT_OK


def _write_unresolved(out: Path, reason: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "multiblock-unresolved.json"
    dest.write_text(
        json.dumps(
            {
                "provenance": {
                    "issue": "GH #68 / Linear RM-255",
                    "agent": AGENT_LINE,
                    "implementation": implementation_commit(),
                },
                "decision": 4,
                "decision_text": "Inconclusive — architectural element unresolved.",
                "unresolved_reason": reason,
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
        print(f"error: {exc}", file=sys.stderr)
        dest = _write_unresolved(args.out, str(exc))
        print(f"wrote conclusion-4 report to {dest}", file=sys.stderr)
        return EXIT_UNRESOLVED
    except ForwardError as exc:
        msg = str(exc)
        print(f"error: {exc}", file=sys.stderr)
        if "legacy_oracle" in msg and hasattr(args, "out"):
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "multiblock-legacy-oracle.json").write_text(
                json.dumps(
                    {
                        "decision": 4,  # nosec B105 — decision option index, not a password
                        "decision_text": (
                            "Inconclusive — GOZ1 pack used legacy_oracle scale; "
                            "rebuild v3 pack-only."
                        ),
                        "error": msg,
                        "agent": AGENT_LINE,
                    },
                    indent=2,
                )
                + "\n"
            )
            return EXIT_LEGACY_ORACLE
        return EXIT_OP
    except (MetricsError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_OP


if __name__ == "__main__":
    raise SystemExit(main())
