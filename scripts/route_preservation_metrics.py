#!/usr/bin/env python3
r"""
Fill the xai-dissect **route-preservation** surface for a bounded Grok-1 block
pilot (GH #53 / RM-222, beads ``goz-4ic2``).

Implementation split:

* ``route_preservation_io`` — pack header / trit / fp16 reads
* ``route_preservation_measure`` — weight, preserve, routing measurements
* this module — CLI, gate reporting, JSON out

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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_grok1_int8_npy import (  # noqa: E402
    MODES as EXPORT_MODES,
    PRESERVE_KINDS as EXPORT_PRESERVE_KINDS,
)
from route_preservation_io import (  # noqa: E402
    MetricsError,
    TENSOR_F16,
    TENSOR_TERNARY,
    load_pack_index,
)
from route_preservation_measure import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_TOKENS,
    GATE_FAILURE_EXIT,
    kind_of,
    measure_preserve,
    measure_routing,
    measure_weights,
)
from route_preservation_surface import build_summary  # noqa: E402


def report_gates(summary: list[dict]) -> int:
    """Print the gate table; return the process exit code."""
    gated = [m for m in summary if m["threshold"]]
    print("\nroute-preservation gates:")
    for m in gated:
        obs = "null" if m["observed"] is None else f"{m['observed']:.6f}"
        print(
            f"  {m['name']:<28} {m['status']:>7}  observed={obs}  "
            f"threshold={m['threshold']}"
        )
    failing = sum(m["status"] != "pass" for m in gated)
    if failing:
        print(f"\n{failing} of {len(gated)} gates not passing")
        return GATE_FAILURE_EXIT
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Route-preservation metrics for a GOZ1 block pilot"
    )
    p.add_argument("--npy-dir", type=Path, required=True)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--block", type=int, required=True)
    p.add_argument("--mode", required=True, choices=sorted(EXPORT_MODES))
    p.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--json-out", type=Path)
    return p.parse_args(argv)


def _json_out_conflicts_with_pack(json_out: Path, pack_path: Path) -> bool:
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


def _validate_pilot_args(args: argparse.Namespace, index: dict[str, dict]) -> None:
    if args.tokens <= 0:
        raise MetricsError(f"--tokens must be positive, got {args.tokens}")
    block_prefix = f"block_{args.block:03d}."
    for name in index:
        if not name.startswith(block_prefix):
            raise MetricsError(
                f"pack tensor {name!r} does not belong to --block {args.block}"
            )
    expected_ternary_kinds = set(EXPORT_MODES[args.mode]) - set(EXPORT_PRESERVE_KINDS)
    actual_ternary_kinds = {
        kind_of(n) for n, e in index.items() if e["tensor_type"] == TENSOR_TERNARY
    }
    if actual_ternary_kinds != expected_ternary_kinds:
        raise MetricsError(
            f"pack ternary kinds {sorted(actual_ternary_kinds)} do not match --mode "
            f"{args.mode} expected {sorted(expected_ternary_kinds)}"
        )


def _pilot_provenance(
    args: argparse.Namespace, metadata: dict, n_ternary: int, n_preserve: int
) -> dict:
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


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.tokens <= 0:
        raise MetricsError(f"--tokens must be a positive integer, got {args.tokens}")
    pack_path = args.pack.expanduser().resolve()
    if args.json_out is not None and _json_out_conflicts_with_pack(args.json_out, pack_path):
        raise MetricsError(
            f"--json-out {args.json_out} resolves to the input pack; "
            "refusing to overwrite the GOZ1 artifact"
        )
    metadata, index = load_pack_index(args.pack)
    _validate_pilot_args(args, index)
    ternary = {n: e for n, e in index.items() if e["tensor_type"] == TENSOR_TERNARY}
    preserve = {n: e for n, e in index.items() if e["tensor_type"] == TENSOR_F16}
    print(f"pack {args.pack.name}: {len(ternary)} ternary, {len(preserve)} preserve/fp16")

    weights = measure_weights(args.npy_dir, args.pack, ternary)
    preserve_err = measure_preserve(args.npy_dir, args.pack, preserve)
    routing = measure_routing(
        args.npy_dir, args.pack, ternary, preserve, weights, args.tokens, args.seed
    )
    summary = build_summary(weights, routing)
    result = {
        "model_family": "grok-1",
        "produced_by": (
            "grok-ozempic scripts/route_preservation_metrics.py (GH #53 / RM-222)"
        ),
        "pilot": _pilot_provenance(args, metadata, len(ternary), len(preserve)),
        "summary": summary,
        "weights": weights,
        "preserve_fp16_roundtrip": preserve_err,
        "routing": routing,
    }
    _write_json(result, args.json_out)
    return report_gates(summary)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except MetricsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
