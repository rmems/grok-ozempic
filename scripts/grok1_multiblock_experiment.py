#!/usr/bin/env python3
"""Multi-block residual fidelity: #68 ternary baseline + #73 remedy arms."""
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
    MODEL_SIZE,
    NUM_SELECTED_EXPERTS,
    ForwardError,
    UnresolvedArchitectureError,
)
from grok1_block_weights import implementation_commit  # noqa: E402
from grok1_multiblock_lib import (  # noqa: E402
    AGENT_LINE,
    BASELINE_64,
    BASELINE_72,
    REMEDY_AGENT_LINE,
    LegacyOracleError,
    decide,
    decide_remedy,
    load_block_sources,
    pack_provenance_row,
    parse_blocks,
    periodic_hp_blocks,
    require_pack_only_scales,
    residual_stream_metrics,
    resolve_path,
    write_remedy_results_md,
    write_results_md,
)
from route_preservation_io import MetricsError  # noqa: E402

__all__ = [
    "decide",
    "decide_remedy",
    "parse_blocks",
    "periodic_hp_blocks",
    "require_pack_only_scales",
    "residual_stream_metrics",
    "write_results_md",
    "write_remedy_results_md",
    "main",
]

_ARM_TO_MODE = {
    "ternary_baseline": "ternary",
    "periodic_hp": "periodic_hp",
    "channel_alpha": "channel_alpha",
}
_REMEDY_ARMS = frozenset({"periodic_hp", "channel_alpha"}) & frozenset(_ARM_TO_MODE)

EXIT_LEGACY_ORACLE = 5
EXIT_OK = 0
EXIT_OP = 1


@dataclass(frozen=True)
class ChainPaths:
    npy_root: Path
    npy_pattern: str
    pack_root: Path
    pack_pattern: str
    embedding_shard: Path


def _validate_embedding_shard(shard: Path) -> None:
    """Shape-only checks; vocab bounds are enforced by ``embedding_rows``."""
    table = np.load(shard, mmap_mode="r")
    if table.ndim != 2:
        raise ForwardError(f"{shard}: expected 2-D embedding table, got shape {table.shape}")
    if table.shape[1] != MODEL_SIZE:
        raise ForwardError(f"{shard}: expected model width {MODEL_SIZE}, got {table.shape[1]}")


def _block_paths(paths: ChainPaths, b: int) -> tuple[Path, Path]:
    npy_dir = resolve_path(paths.npy_root, paths.npy_pattern, b)
    pack_path = resolve_path(paths.pack_root, paths.pack_pattern, b)
    if not npy_dir.is_dir():
        raise ForwardError(f"missing npy dir for block {b}: {npy_dir}")
    if not pack_path.is_file():
        raise ForwardError(f"missing pack for block {b}: {pack_path}")
    return npy_dir, pack_path


def _forward_fp16(control, h_fp16, ref_trace, stream_fp16, top_k):
    print("  fp16 control ...", flush=True)
    fp16_trace = forward_block(h_fp16, control, top_k=top_k)
    fp16_cmp = compare(ref_trace, fp16_trace)
    fp16_cmp["residual_stream_in"] = stream_fp16
    print(f"    {fp16_trace.seconds:.1f}s  block_out_cos={fp16_cmp['block_output_cosine']:.6f}", flush=True)
    return fp16_cmp, fp16_trace.block_out


@dataclass(frozen=True)
class _BlockRunCfg:
    top_k: int
    skip_fp16: bool
    expert_mode: str
    hp_blocks: frozenset[int]
    hp_period: int


def _run_block(b, paths, streams, cfg: _BlockRunCfg):
    """Forward one block; streams is (h_ref, h_pilot, h_fp16)."""
    h_ref, h_pilot, h_fp16 = streams
    npy_dir, pack_path = _block_paths(paths, b)
    print(f"== block {b:03d}  residual_in shape={h_ref.shape}", flush=True)
    reference, pack, mixed, control = load_block_sources(
        b,
        npy_dir,
        pack_path,
        require_fp16=not cfg.skip_fp16,
        expert_mode=cfg.expert_mode,
        hp_blocks=set(cfg.hp_blocks),
        hp_period=cfg.hp_period,
    )
    stream_pilot = residual_stream_metrics(h_ref, h_pilot)
    stream_fp16 = residual_stream_metrics(h_ref, h_fp16) if h_fp16 is not None else None

    print("  reference forward ...", flush=True)
    ref_trace = forward_block(h_ref, reference, top_k=cfg.top_k)
    print(f"    {ref_trace.seconds:.1f}s experts={ref_trace.experts_touched}", flush=True)

    print(f"  pilot ({mixed.label}) ...", flush=True)
    pilot_trace = forward_block(h_pilot, mixed, top_k=cfg.top_k)
    pilot_cmp = compare(ref_trace, pilot_trace)
    pilot_cmp["residual_stream_in"] = stream_pilot
    pilot_cmp["label"] = mixed.label
    print(
        f"    {pilot_trace.seconds:.1f}s  cos={pilot_cmp['block_output_cosine']:.6f} "
        f"top1={pilot_cmp['router_top1_agreement']:.6f} "
        f"drift={stream_pilot['residual_in_drift_relative_norm']:.6f}",
        flush=True,
    )
    fp16_cmp, next_fp16 = None, h_fp16
    if control is not None and h_fp16 is not None:
        fp16_cmp, next_fp16 = _forward_fp16(control, h_fp16, ref_trace, stream_fp16, cfg.top_k)
    row = {
        "block": b,
        "reference_seconds": ref_trace.seconds,
        "expert_only": pilot_cmp,
        "fp16_control": fp16_cmp,
        "pilot_label": mixed.label,
    }
    applied = getattr(mixed, "applied_scale_sources", None)
    prov = pack_provenance_row(b, pack_path, npy_dir, pack, applied_scale_sources=applied)
    return row, (ref_trace.block_out, pilot_trace.block_out, next_fp16), prov


def _chain_exit_block(h_ref, h_pilot, h_fp16) -> dict:
    expert_exit = residual_stream_metrics(h_ref, h_pilot)
    fp16_exit = None if h_fp16 is None else residual_stream_metrics(h_ref, h_fp16)
    return {
        "expert_only_chain_exit": {
            "residual_cosine": expert_exit["residual_in_cosine"],
            "residual_drift_relative_norm": expert_exit["residual_in_drift_relative_norm"],
            "note": "post-final-block residual stream (chain exit), not last-block residual-in",
        },
        "fp16_chain_exit": None
        if fp16_exit is None
        else {
            "residual_cosine": fp16_exit["residual_in_cosine"],
            "residual_drift_relative_norm": fp16_exit["residual_in_drift_relative_norm"],
            "note": "post-final-block residual stream under FP16 control",
        },
    }


def _arm_meta(blocks, expert_mode: str, hp_period: int, hp_blocks: set[int], labels: list) -> dict:
    ternary_set = sorted(set(blocks) - hp_blocks)
    hp_list = sorted(hp_blocks)
    prose = None
    arm_label = expert_mode
    if expert_mode == "periodic_hp":
        arm_label = f"expert_periodic_hp_n{hp_period}"
        prose = (
            f"Arm C label `{arm_label}`: "
            f"ternary on {{{','.join(map(str, ternary_set))}}}, "
            f"HP (FP16 experts) on {{{','.join(map(str, hp_list))}}}."
        )
    elif expert_mode == "channel_alpha":
        arm_label = "research_per_channel_side"
        ternary_set = []  # experts use channel-α side scales, not pack ternary path
    return {
        "expert_mode": expert_mode,
        "hp_period": int(hp_period) if expert_mode == "periodic_hp" else None,
        "hp_blocks": hp_list,
        "ternary_blocks": ternary_set if expert_mode in ("periodic_hp", "channel_alpha") else list(blocks),
        "hp_schedule_prose": prose,
        "arm_label": arm_label,
        "pilot_labels_per_block": labels,
    }


def run_chain(
    blocks, paths, *, tokens, seed, top_k, skip_fp16, expert_mode="ternary", hp_period=2
) -> dict:
    if blocks[0] != 0:
        raise ForwardError(f"chain must start at block 0 (got blocks={blocks})")
    hp_blocks = periodic_hp_blocks(blocks, hp_period) if expert_mode == "periodic_hp" else set()
    cfg = _BlockRunCfg(
        top_k=top_k,
        skip_fp16=skip_fp16,
        expert_mode=expert_mode,
        hp_blocks=frozenset(hp_blocks),
        hp_period=int(hp_period),
    )
    ids = token_ids(tokens, seed, vocab=131072)
    _validate_embedding_shard(paths.embedding_shard)
    h0 = embedding_rows(paths.embedding_shard, ids)
    streams = (h0, h0.copy(), h0.copy() if not skip_fp16 else None)
    per_block, pack_provenance = [], []
    for b in blocks:
        row, streams, prov = _run_block(b, paths, streams, cfg)
        per_block.append(row)
        pack_provenance.append(prov)
    h_ref, h_pilot, h_fp16 = streams
    meta = _arm_meta(
        blocks, expert_mode, hp_period, hp_blocks, [r.get("pilot_label") for r in per_block]
    )
    return {
        "blocks": blocks,
        "tokens": int(ids.size),
        "token_seed": int(seed),
        "token_id_first": int(ids[0]),
        "token_id_last": int(ids[-1]),
        "top_k": int(top_k),
        "per_block": per_block,
        "end_of_chain": _chain_exit_block(h_ref, h_pilot, h_fp16),
        "pack_provenance": pack_provenance,
        "skip_fp16_control": bool(skip_fp16),
        **meta,
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
    # YYYYMMDD decision-run seed; arithmetic form avoids Bandit B105 on *token*.
    p.add_argument("--seed", type=int, default=2026 * 10_000 + 806)
    p.add_argument("--top-k", type=int, default=NUM_SELECTED_EXPERTS)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--skip-fp16-control", action="store_true")
    p.add_argument("--write-report-md", action="store_true")
    p.add_argument(
        "--arm",
        choices=sorted(_ARM_TO_MODE),
        default="ternary_baseline",
        help="ternary_baseline=#68; periodic_hp=#73 arm C; channel_alpha=#73 arm A",
    )
    p.add_argument(
        "--hp-period",
        type=int,
        default=2,
        help="Arm C: period N (N=2 → ternary {0,2}, HP {1,3} on chain 0..3)",
    )
    return p


def _is_remedy_arm(arm: str) -> bool:
    return arm in _REMEDY_ARMS


def _provenance(paths: ChainPaths, skip_fp16: bool, arm: str) -> dict:
    if _is_remedy_arm(arm):
        return {
            "issue": "GH #73 / Linear RM-362",
            "agent": REMEDY_AGENT_LINE,
            "model": "Grok-4.5 (high)",
            "design": "SpacexAI design lock: arms C (periodic HP) and A (channel α side-table)",
            "baseline_64": BASELINE_64,
            "baseline_72": BASELINE_72,
            "implementation": implementation_commit(),
            "architecture_source": "github.com/xai-org/grok-1 model.py + run.py",
            "numpy": np.__version__,
            "python": platform.python_version(),
            "embedding_shard": Path(paths.embedding_shard).name,
            "skip_fp16_control": bool(skip_fp16),
            "activation_policy": "paired residuals; no Gaussian; no embed for b!=0",
            "ternary_policy": "experts only on ternary blocks; attention/routers/norms high precision",
            "scale_policy": "GOZ1 v3 pack-only on ternary path; abort on legacy_oracle",
            "arm": arm,
            "metrics_filename": "metrics.json",
            "metrics_note": (
                "#72 single-scale ternary baseline cited from "
                "reports/grok-1-expert-only-multiblock/ (bit-identical seed/tokens/packs)."
            ),
        }
    return {
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
        "skip_fp16_control": bool(skip_fp16),
        "activation_policy": "paired residuals; no Gaussian; no embed for b!=0",
        "ternary_policy": "experts only; attention/routers/norms high precision",
        "scale_policy": "GOZ1 v3 pack-only; abort on legacy_oracle",
    }


def run(args: argparse.Namespace) -> int:
    if args.tokens < 1:
        raise ForwardError(f"tokens must be >= 1, got {args.tokens}")
    if args.hp_period < 1:
        raise ForwardError(f"--hp-period must be >= 1, got {args.hp_period}")
    blocks = parse_blocks(args.blocks)
    paths = ChainPaths(
        npy_root=args.npy_root.expanduser(),
        npy_pattern=args.npy_pattern,
        pack_root=args.pack_root.expanduser(),
        pack_pattern=args.pack_pattern,
        embedding_shard=args.embedding_shard.expanduser(),
    )
    args.out = args.out.expanduser()
    expert_mode = _ARM_TO_MODE[args.arm]
    chain = run_chain(
        blocks,
        paths,
        tokens=args.tokens,
        seed=args.seed,
        top_k=args.top_k,
        skip_fp16=args.skip_fp16_control,
        expert_mode=expert_mode,
        hp_period=args.hp_period,
    )
    decision = decide_remedy(chain) if _is_remedy_arm(args.arm) else decide(chain)
    payload = {
        "provenance": _provenance(paths, args.skip_fp16_control, args.arm),
        "chain": chain,
        "decision": decision,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out / 'metrics.json'}")
    print(f"DECISION option {decision['decision']}: {decision['decision_text']}")
    if args.write_report_md or not args.skip_fp16_control:
        if _is_remedy_arm(args.arm):
            write_remedy_results_md(args.out / "results.md", payload)
        else:
            write_results_md(args.out / "results.md", payload)
        print(f"wrote {args.out / 'results.md'}")
    return EXIT_OK


def _agent_for_arm(arm: str | None) -> tuple[str, str]:
    remedy = _is_remedy_arm(arm or "ternary_baseline")
    if remedy:
        return "GH #73 / Linear RM-362", REMEDY_AGENT_LINE
    return "GH #68 / Linear RM-255", AGENT_LINE


def _write_unresolved(args: argparse.Namespace, exc: Exception) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "multiblock-unresolved.json"
    issue, agent = _agent_for_arm(getattr(args, "arm", None))
    dest.write_text(
        json.dumps(
            {
                "provenance": {
                    "issue": issue,
                    "agent": agent,
                    "arm": getattr(args, "arm", None),
                    "implementation": implementation_commit(),
                },
                "decision": 4,
                "decision_text": "Inconclusive — architectural element unresolved.",
                "unresolved_reason": str(exc),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote conclusion-4 report to {dest}", file=sys.stderr)
    return EXIT_UNRESOLVED


def _write_legacy(args: argparse.Namespace, exc: Exception) -> int:
    if hasattr(args, "out"):
        args.out.mkdir(parents=True, exist_ok=True)
        _, agent = _agent_for_arm(getattr(args, "arm", None))
        (args.out / "multiblock-legacy-oracle.json").write_text(
            json.dumps(
                {
                    "decision": 4,
                    "decision_text": "Inconclusive — legacy_oracle scale; rebuild v3 pack-only.",
                    "error": str(exc),
                    "agent": agent,
                    "arm": getattr(args, "arm", None),
                },
                indent=2,
            )
            + "\n"
        )
    return EXIT_LEGACY_ORACLE


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except UnresolvedArchitectureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _write_unresolved(args, exc)
    except LegacyOracleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _write_legacy(args, exc)
    except ForwardError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_OP
    except (MetricsError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_OP


if __name__ == "__main__":
    raise SystemExit(main())
