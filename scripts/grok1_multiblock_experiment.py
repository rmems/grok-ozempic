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
    BASELINE_74,
    REMEDY_AGENT_LINE,
    REMEDY_V2_AGENT_LINE,
    LegacyOracleError,
    assemble_remedy_v2_comparison,
    decide,
    decide_remedy,
    decide_remedy_v2,
    load_block_sources,
    pack_provenance_row,
    parse_blocks,
    parse_hp_blocks,
    periodic_hp_blocks,
    require_pack_only_scales,
    residual_stream_metrics,
    resolve_path,
    remedy_metrics_note,
    write_remedy_results_md,
    write_remedy_v2_results_md,
    write_results_md,
)
from route_preservation_io import MetricsError  # noqa: E402

__all__ = [
    "decide",
    "decide_remedy",
    "decide_remedy_v2",
    "parse_blocks",
    "periodic_hp_blocks",
    "require_pack_only_scales",
    "residual_stream_metrics",
    "write_results_md",
    "write_remedy_results_md",
    "write_remedy_v2_results_md",
    "main",
]

_ARM_TO_MODE = {
    "ternary_baseline": "ternary",
    "periodic_hp": "periodic_hp",
    "channel_alpha": "channel_alpha",
    "stacked_hp_channel_alpha": "periodic_hp_plus_channel_alpha",
    "hp_ceiling": "all_hp",
    }
_REMEDY_ARMS = frozenset(_ARM_TO_MODE) - {"ternary_baseline"}
_V2_REMEDY_ARMS = frozenset({"stacked_hp_channel_alpha", "hp_ceiling"})

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
    hp_label: str


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
        hp_label=cfg.hp_label,
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


def _explicit_schedule_label(hp_blocks: set[int]) -> str:
    return "".join(str(b) for b in sorted(hp_blocks))


_SCHEDULED_MODES = frozenset({"periodic_hp", "periodic_hp_plus_channel_alpha"})


def _schedule_label(hp_blocks: set[int], hp_period: int, explicit_hp: bool) -> str:
    return _explicit_schedule_label(hp_blocks) if explicit_hp else f"n{hp_period}"


def _block_set_text(blocks: list[int]) -> str:
    return ",".join(map(str, blocks))


def _arm_identity(
    expert_mode: str,
    schedule: str,
    ternary_blocks: list[int],
    hp_blocks: list[int],
) -> tuple[str, str | None, list[int]]:
    if expert_mode == "periodic_hp":
        label = f"expert_periodic_hp_{schedule}"
        prose = (
            f"Arm C label `{label}`: ternary on {{{_block_set_text(ternary_blocks)}}}, "
            f"HP (FP16 experts) on {{{_block_set_text(hp_blocks)}}}."
        )
        return label, prose, ternary_blocks
    if expert_mode == "periodic_hp_plus_channel_alpha":
        label = f"expert_periodic_hp_{schedule}_plus_channel_alpha"
        prose = (
            f"Stacked C+A label `{label}`: channel-α trits on "
            f"{{{_block_set_text(ternary_blocks)}}}, HP (FP16 experts) on "
            f"{{{_block_set_text(hp_blocks)}}}."
        )
        return label, prose, ternary_blocks
    if expert_mode == "all_hp":
        return "expert_hp_ceiling", "HP expert ceiling: FP16 experts on every measured block.", []
    if expert_mode == "channel_alpha":
        return "research_per_channel_side", None, ternary_blocks
    return expert_mode, None, ternary_blocks


def _schedule_metadata(expert_mode: str, hp_period: int, explicit_hp: bool) -> tuple[int | None, str]:
    if expert_mode == "all_hp":
        return None, "all"
    if explicit_hp:
        return None, "explicit"
    if expert_mode in _SCHEDULED_MODES:
        return int(hp_period), "periodic"
    return None, "none"


def _arm_meta(
    blocks,
    expert_mode: str,
    hp_period: int,
    hp_blocks: set[int],
    labels: list,
    *,
    explicit_hp: bool,
) -> dict:
    ternary_blocks = sorted(set(blocks) - hp_blocks)
    hp_list = sorted(hp_blocks)
    schedule = _schedule_label(hp_blocks, hp_period, explicit_hp)
    arm_label, prose, ternary_blocks = _arm_identity(
        expert_mode, schedule, ternary_blocks, hp_list
    )
    stored_period, schedule_kind = _schedule_metadata(expert_mode, hp_period, explicit_hp)
    channel_blocks = ternary_blocks if expert_mode in {"channel_alpha", "periodic_hp_plus_channel_alpha"} else []
    return {
        "expert_mode": expert_mode,
        "hp_period": stored_period,
        "hp_schedule_kind": schedule_kind,
        "hp_blocks": hp_list,
        "ternary_blocks": ternary_blocks,
        "channel_alpha_blocks": channel_blocks,
        "hp_schedule_prose": prose,
        "arm_label": arm_label,
        "pilot_labels_per_block": labels,
    }


def _validate_explicit_hp_blocks(
    blocks: list[int], explicit_hp_blocks: set[int] | None
) -> set[int] | None:
    if explicit_hp_blocks is None:
        return None
    outside = sorted(explicit_hp_blocks - set(blocks))
    if outside:
        raise ForwardError(
            f"--hp-blocks contains blocks outside --blocks: {outside}; chain={blocks}"
        )
    return set(explicit_hp_blocks)


def _resolve_ceiling_blocks(chain_blocks: set[int], explicit: set[int] | None) -> set[int]:
    if explicit is not None and explicit != chain_blocks:
        raise ForwardError(
            "hp_ceiling requires every chain block in --hp-blocks "
            f"(expected {sorted(chain_blocks)}, got {sorted(explicit)})"
        )
    return chain_blocks


def _resolve_hp_blocks(
    blocks: list[int],
    expert_mode: str,
    hp_period: int,
    explicit_hp_blocks: set[int] | None,
) -> set[int]:
    chain_blocks = set(blocks)
    explicit = _validate_explicit_hp_blocks(blocks, explicit_hp_blocks)
    if expert_mode == "all_hp":
        return _resolve_ceiling_blocks(chain_blocks, explicit)
    if expert_mode in _SCHEDULED_MODES:
        if explicit is not None:
            return explicit
        return periodic_hp_blocks(blocks, hp_period)
    if explicit is not None:
        raise ForwardError(
            f"--hp-blocks is only valid for scheduled HP arms, got mode {expert_mode!r}"
        )
    return set()


def run_chain(
    blocks,
    paths,
    *,
    tokens,
    seed,
    top_k,
    skip_fp16,
    expert_mode="ternary",
    hp_period=2,
    hp_blocks: set[int] | None = None,
) -> dict:
    if blocks[0] != 0:
        raise ForwardError(f"chain must start at block 0 (got blocks={blocks})")
    explicit_hp = hp_blocks is not None
    resolved_hp = _resolve_hp_blocks(blocks, expert_mode, hp_period, hp_blocks)
    cfg = _BlockRunCfg(
        top_k=top_k,
        skip_fp16=skip_fp16,
        expert_mode=expert_mode,
        hp_blocks=frozenset(resolved_hp),
        hp_period=int(hp_period),
        hp_label=_explicit_schedule_label(resolved_hp) if explicit_hp else f"n{hp_period}",
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
        blocks,
        expert_mode,
        hp_period,
        resolved_hp,
        [r.get("pilot_label") for r in per_block],
        explicit_hp=explicit_hp,
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
    p.add_argument(
        "--hp-blocks",
        type=parse_hp_blocks,
        help="Explicit comma-separated HP block set; overrides --hp-period",
    )
    p.add_argument(
        "--evidence-only",
        action="store_true",
        help="Write a #75 secondary-arm metrics payload without a decision",
    )
    p.add_argument(
        "--comparison-metrics",
        action="append",
        default=[],
        type=Path,
        help="Secondary evidence metrics.json; repeat for stacked and ceiling arms",
    )
    return p


def _is_remedy_arm(arm: str) -> bool:
    return arm in _REMEDY_ARMS


def _is_v2_primary(args: argparse.Namespace) -> bool:
    return args.arm == "periodic_hp" and getattr(args, "hp_blocks", None) == {1, 2, 3}


def _validate_v2_cli(args: argparse.Namespace) -> None:
    evidence_only = bool(getattr(args, "evidence_only", False))
    comparison_paths = list(getattr(args, "comparison_metrics", []))
    if args.arm in _V2_REMEDY_ARMS and not evidence_only:
        raise ForwardError(f"--arm {args.arm} requires --evidence-only")
    if evidence_only and args.arm not in _V2_REMEDY_ARMS:
        raise ForwardError("--evidence-only is reserved for #75 stacked and HP-ceiling arms")
    if comparison_paths and not _is_v2_primary(args):
        raise ForwardError(
            "--comparison-metrics is only valid for the #75 primary run "
            "(--arm periodic_hp --hp-blocks 1,2,3)"
        )


def _load_comparison_payloads(paths: list[Path]) -> tuple[list[dict], list[str]]:
    payloads: list[dict] = []
    errors: list[str] = []
    for raw_path in paths:
        path = raw_path.expanduser()
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"could not load secondary evidence {path}: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"secondary evidence {path} must contain a JSON object")
            continue
        payloads.append(payload)
    return payloads, errors


def _provenance(paths: ChainPaths, skip_fp16: bool, arm: str, *, v2: bool = False) -> dict:
    if _is_remedy_arm(arm):
        if v2:
            issue = "GH #75 / Linear RM-462 / beads goz-rvk"
            agent = REMEDY_V2_AGENT_LINE
            model = "GPT-5.6 Sol"
            design = "Codex design lock: C denser, N=2+C+A, and HP expert ceiling"
        else:
            issue = "GH #73 / Linear RM-362"
            agent = REMEDY_AGENT_LINE
            model = "Grok-4.5 (high)"
            design = "Grok Build design lock: arms C (periodic HP) and A (channel α side-table)"
        return {
            "issue": issue,
            "agent": agent,
            "model": model,
            "design": design,
            "baseline_64": BASELINE_64,
            "baseline_72": BASELINE_72,
            "baseline_74": BASELINE_74 if v2 else None,
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
            # metrics_note filled after chain when comparability is known
            "metrics_note": None,
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
    _validate_v2_cli(args)
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
        hp_blocks=args.hp_blocks,
    )
    is_v2 = args.arm in _V2_REMEDY_ARMS or args.hp_blocks is not None
    prov = _provenance(paths, args.skip_fp16_control, args.arm, v2=is_v2)
    if _is_remedy_arm(args.arm):
        prov["metrics_note"] = remedy_metrics_note(chain)
    evidence_only = bool(getattr(args, "evidence_only", False))
    if evidence_only:
        prov["evidence_role"] = "secondary; no independent decision"
        payload = {"provenance": prov, "chain": chain}
        decision = None
    elif _is_v2_primary(args):
        secondary, load_errors = _load_comparison_payloads(
            list(getattr(args, "comparison_metrics", []))
        )
        comparison = assemble_remedy_v2_comparison(
            chain,
            secondary,
            load_errors=load_errors,
        )
        decision = decide_remedy_v2(comparison)
        prov["evidence_role"] = "primary; sole canonical #75 decision"
        payload = {
            "provenance": prov,
            "chain": chain,
            "comparison": comparison,
            "decision": decision,
        }
    else:
        decision = decide_remedy(chain) if _is_remedy_arm(args.arm) else decide(chain)
        payload = {"provenance": prov, "chain": chain, "decision": decision}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out / 'metrics.json'}")
    if evidence_only:
        print("EVIDENCE ONLY: no decision emitted")
        return EXIT_OK
    assert decision is not None
    print(f"DECISION option {decision['decision']}: {decision['decision_text']}")
    if args.write_report_md or not args.skip_fp16_control:
        if _is_v2_primary(args):
            write_remedy_v2_results_md(args.out / "results.md", payload)
        elif _is_remedy_arm(args.arm):
            write_remedy_results_md(args.out / "results.md", payload)
        else:
            write_results_md(args.out / "results.md", payload)
        print(f"wrote {args.out / 'results.md'}")
    return EXIT_OK


def _agent_for_args(args: argparse.Namespace) -> tuple[str, str]:
    arm = getattr(args, "arm", None)
    if arm in _V2_REMEDY_ARMS or getattr(args, "hp_blocks", None) is not None:
        return "GH #75 / Linear RM-462 / beads goz-rvk", REMEDY_V2_AGENT_LINE
    remedy = _is_remedy_arm(arm or "ternary_baseline")
    if remedy:
        return "GH #73 / Linear RM-362", REMEDY_AGENT_LINE
    return "GH #68 / Linear RM-255", AGENT_LINE


def _write_unresolved(args: argparse.Namespace, exc: Exception) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "multiblock-unresolved.json"
    issue, agent = _agent_for_args(args)
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
        _, agent = _agent_for_args(args)
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
