#!/usr/bin/env python3
r"""
Fill the xai-dissect route-preservation surface for a Grok-1 block pilot
(GH #53 / RM-222). Split: route_preservation_io / _measure / this CLI.
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
    ActivationSpec,
    kind_of,
    measure_preserve,
    measure_routing,
    measure_weights,
)
from route_preservation_surface import build_summary  # noqa: E402

ROUTING_MODES = {"attention_only", "attention_plus_expert"}

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
    p.add_argument(
        "--embedding-shard",
        type=Path,
        help=(
            "token-embedding pickle shard (e.g. $CKPT/tensor00000_000). For block 0 the "
            "attention input IS rmsnorm(embedding lookup), so this gives real "
            "activations instead of the synthetic Gaussian fallback"
        ),
    )
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
    """Read the conversion manifest, failing closed rather than by traceback."""
    if path is None:
        return None
    doc = _read_json(path)
    if not isinstance(doc, dict):
        raise MetricsError(
            f"{path}: conversion manifest root is {type(doc).__name__}, expected an object"
        )
    tensors = doc.get("tensors")
    if not isinstance(tensors, list):
        raise MetricsError(f"{path}: no `tensors` array")
    bad = [i for i, t in enumerate(tensors) if not isinstance(t, dict)]
    if bad:
        raise MetricsError(f"{path}: tensors[{bad[0]}] is not an object")
    return tensors


def _read_json(path: Path) -> object:
    try:
        with path.open("rb") as f:
            return json.load(f)
    except OSError as exc:
        raise MetricsError(f"{path}: cannot read conversion manifest: {exc}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MetricsError(f"{path}: conversion manifest is not valid JSON: {exc}") from exc


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
        args.npy_dir,
        args.pack,
        ternary,
        preserve,
        weights,
        ActivationSpec(
            tokens=args.tokens,
            seed=args.seed,
            embedding_shard=args.embedding_shard,
        ),
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
    # Expand every path argument up front. A quoted "~/..." is not expanded by
    # the shell, and writing the report to a literal directory named "~" would
    # look like success while the file lands somewhere nobody looks.
    args.pack = args.pack.expanduser()
    args.npy_dir = args.npy_dir.expanduser()
    if args.json_out is not None:
        args.json_out = args.json_out.expanduser()
    if getattr(args, "conversion_manifest", None) is not None:
        args.conversion_manifest = args.conversion_manifest.expanduser()
    if args.embedding_shard is not None:
        args.embedding_shard = args.embedding_shard.expanduser()
        if args.block != 0:
            # Only block 0's attention input is the embedding lookup; for a later
            # block the real input is the residual stream after every preceding
            # block, which needs a full forward pass this pilot does not run.
            raise MetricsError(
                f"--embedding-shard is only valid for block 0 (got block {args.block}); "
                "later blocks see the residual stream, not the embedding lookup"
            )
    pack_path = args.pack.resolve()
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
            "real token-embedding rows through the block's RMSNorm gain"
            if args.embedding_shard is not None
            else "SYNTHETIC seeded standard-normal rows (no --embedding-shard given)"
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
