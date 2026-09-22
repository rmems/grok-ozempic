#!/usr/bin/env python3
"""Run the frozen GH125 matched alpha/schedule experiment with fail-closed gates."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grok1_alpha_schedule_runner as runner  # noqa: E402
import numpy as np  # noqa: E402
from grok1_alpha_schedule_protocol import (  # noqa: E402
    BLOCKS,
    CELLS,
    PROTOCOL,
    SEED,
    TOKENS,
    TOKEN_IDS_SHA256,
    TOP_K,
    require,
)
from grok1_block0_experiment import token_ids  # noqa: E402
from grok1_block_weights import sha256_file  # noqa: E402
from grok1_multiblock_experiment import (  # noqa: E402
    ChainPaths,
    _validate_embedding_shard,
    run_chain,
)
from grok1_multiblock_lib import npy_dir_fingerprint, resolve_path  # noqa: E402
from grok1_multiblock_v4_supervisor import _portable_failure_value  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npy-root", type=Path, required=True)
    p.add_argument("--npy-pattern", default="goz68-block_{block:03d}-attn")
    p.add_argument("--pack-root", type=Path, required=True)
    p.add_argument("--pack-pattern", default="block_{block:03d}-attention_plus_expert.goz1")
    p.add_argument("--embedding-shard", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--int4-side-root", type=Path)
    p.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check inputs, resources and clean implementation; no forward",
    )
    p.add_argument("--cell", choices=tuple(CELLS), help=argparse.SUPPRESS)
    p.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    return p


def paths_for(args):
    return ChainPaths(
        args.npy_root.expanduser().resolve(),
        args.npy_pattern,
        args.pack_root.expanduser().resolve(),
        args.pack_pattern,
        args.embedding_shard.expanduser().resolve(),
    )


def _expert_arrays(folder, block):
    experts = sorted(folder.glob("*__moe_expert__*.npy"))
    require(
        len(experts) == 3
        and {p.stem.rsplit("__", 1)[-1] for p in experts} == {"gate", "up", "down"},
        f"expected three structural expert arrays in {folder}",
    )
    require(
        all(p.name.startswith(f"block_{block:03d}__") for p in experts), "wrong NPY block identity"
    )
    return experts


def _validate_destinations(destinations, sources):
    for destination in destinations:
        for source in sources:
            require(
                destination != source
                and destination not in source.parents
                and source not in destination.parents,
                "output/cache overlaps source inputs",
            )


def _validate_cache_scratch_layout(out, side):
    """Keep INT4 cache accounting disjoint from supervised cell scratch under out/runs."""
    resolved_out = out.resolve()
    resolved_side = side.resolve()
    runs_root = resolved_out / "runs"
    require(resolved_side != resolved_out, "int4-side-root must not equal --out")
    require(
        not runs_root.is_relative_to(resolved_side) and not resolved_side.is_relative_to(runs_root),
        "int4-side-root overlaps supervised cell scratch under --out/runs",
    )


def inspect_inputs(paths, out, side):
    validate_destinations(paths, out, side)
    inventory = []
    _validate_embedding_shard(paths.embedding_shard)
    for block in BLOCKS:
        folder = resolve_path(paths.npy_root, paths.npy_pattern, block).resolve()
        pack = resolve_path(paths.pack_root, paths.pack_pattern, block).resolve()
        require(folder.is_dir() and pack.is_file(), f"missing input for block {block}")
        experts = _expert_arrays(folder, block)
        inventory.extend(runner.tensor_inventory(p, block) for p in experts)
    return inventory


def validate_destinations(paths, out, side):
    """Resolve source paths without opening inputs or creating any filesystem entries."""
    sources = [paths.embedding_shard]
    for block in BLOCKS:
        sources.extend(
            resolve_path(root, pattern, block).resolve()
            for root, pattern in (
                (paths.npy_root, paths.npy_pattern),
                (paths.pack_root, paths.pack_pattern),
            )
        )
    _validate_cache_scratch_layout(out, side)
    _validate_destinations((out, side), sources)


def runtime_identity():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_config": json.loads(json.dumps(np.__config__.show(mode="dicts"))),
        "threads": {
            k: os.environ.get(k)
            for k in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        },
    }


def prepare_context(paths, inventory):
    implementation = runner.clean_implementation()
    ids = token_ids(TOKENS, SEED, 131072).astype("<i8")
    digest = hashlib.sha256(ids.tobytes()).hexdigest()
    require(digest == TOKEN_IDS_SHA256, "sampled token IDs/order changed from frozen protocol")
    inputs = [
        {
            "block": b,
            "npy_sha256": npy_dir_fingerprint(resolve_path(paths.npy_root, paths.npy_pattern, b)),
            "pack_sha256": sha256_file(resolve_path(paths.pack_root, paths.pack_pattern, b)),
        }
        for b in BLOCKS
    ]
    embedding = sha256_file(paths.embedding_shard)
    checkpoint = hashlib.sha256(
        json.dumps({"blocks": inputs, "embedding": embedding}, sort_keys=True).encode()
    ).hexdigest()
    return {
        "inventory": inventory,
        "inputs": inputs,
        "provenance": {
            "implementation": implementation,
            "runtime": runtime_identity(),
            "embedding_sha256": embedding,
            "token_ids_sha256": digest,
            "token_ids": ids.tolist(),
            "checkpoint_identity": checkpoint,
            "checkpoint_identity_kind": "measured NPY/pack/embedding content; not upstream checkpoint attestation",
        },
    }


def make_gate(paths, out, side):
    def gate():
        inventory = inspect_inputs(paths, out, side)
        requirements = runner.disk_requirements(out, side, inventory)
        snapshot = runner.resource_snapshot(requirements)
        runner.atomic_json(
            out / "resource-preflight.json",
            {
                "snapshot": snapshot,
                "requirements": {
                    k: {"additional": row["additional"]} for k, row in requirements.items()
                },
            },
        )
        runner.check_resources(snapshot, {k: r["additional"] for k, r in requirements.items()})
        return inventory

    return gate


def make_command(args, side):
    """Bind CLI paths once; children receive a controlled argv without a shell."""

    def command(cell, dest, run_id, context):
        manifest = dest / "launch.json"
        runner.atomic_json(
            manifest,
            {"run_id": run_id, "parent_pid": os.getpid(), "cell": cell, "context": context},
        )
        return [
            sys.executable,
            str(Path(__file__).resolve()),
            "--cell",
            cell,
            "--manifest",
            str(manifest),
            "--out",
            str(dest),
            "--npy-root",
            str(args.npy_root),
            "--npy-pattern",
            args.npy_pattern,
            "--pack-root",
            str(args.pack_root),
            "--pack-pattern",
            args.pack_pattern,
            "--embedding-shard",
            str(args.embedding_shard),
            "--int4-side-root",
            str(side),
        ]

    return command


def _launch_context(args, paths, out, side):
    require(
        args.manifest is not None and not args.preflight_only,
        "internal child requires launch manifest",
    )
    launch = json.loads(args.manifest.read_text())
    require(
        launch["parent_pid"] == os.getppid() and launch["cell"] == args.cell,
        "child must be launched by its live supervisor",
    )
    require(args.manifest.resolve() == out / "launch.json", "manifest must belong to cell output")
    context = launch["context"]
    gate = make_gate(paths, out, side)
    inventory = gate()
    require(inventory == context["inventory"], "input headers changed before child launch")
    require(
        runner.clean_implementation() == context["provenance"]["implementation"],
        "child implementation changed",
    )
    require(runtime_identity() == context["provenance"]["runtime"], "child runtime changed")
    require(
        sha256_file(paths.embedding_shard) == context["provenance"]["embedding_sha256"],
        "embedding changed",
    )
    return launch, inventory


def _validate_completed_chain(chain, paths, context):
    measured_inputs = [
        {k: row[k] for k in ("block", "npy_sha256", "pack_sha256")}
        for row in chain["pack_provenance"]
    ]
    require(measured_inputs == context["inputs"], "measured inputs differ from launch content")
    require(
        sha256_file(paths.embedding_shard) == context["provenance"]["embedding_sha256"],
        "embedding changed during forward",
    )
    require(
        runner.clean_implementation() == context["provenance"]["implementation"],
        "implementation changed during forward",
    )


def run_cell(args, paths, out, side):
    launch, inventory = _launch_context(args, paths, out, side)
    context = launch["context"]
    spec = CELLS[args.cell]
    chain = run_chain(
        list(BLOCKS),
        paths,
        tokens=TOKENS,
        seed=SEED,
        top_k=TOP_K,
        skip_fp16=False,
        expert_mode=spec.mode,
        hp_blocks=set(spec.hp),
        int4_side_root=side,
        progress_path=out / "progress.json",
        progress_base={"protocol": PROTOCOL, "run_id": launch["run_id"], "cell": args.cell},
    )
    _validate_completed_chain(chain, paths, context)
    resources = runner.actual_resources(inventory, args.cell, side, out)
    runner.atomic_json(
        out / "metrics.json",
        {
            "protocol": PROTOCOL,
            "run_id": launch["run_id"],
            "cell": args.cell,
            "chain": chain,
            "provenance": context["provenance"],
            "resources": resources,
        },
    )
    return 0


def portable_args(args, paths, out, side):
    """Resolved CLI paths for portable failure publication."""
    return argparse.Namespace(
        **{
            **vars(args),
            "npy_root": paths.npy_root,
            "pack_root": paths.pack_root,
            "embedding_shard": paths.embedding_shard,
            "out": out,
            "int4_side_root": side,
        }
    )


def portable_preflight(args, paths, out, side, result):
    """Reuse historical path redaction only at the public preflight boundary."""
    return _portable_failure_value(portable_args(args, paths, out, side), result)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        out = runner.validate_output_path(args.out)
        side = runner.validate_output_path(args.int4_side_root or out / "int4-side")
        paths = paths_for(args)
        validate_destinations(paths, out, side)
        if args.cell:
            return run_cell(args, paths, out, side)
        require(args.manifest is None, "manifest is internal to a supervised cell")
        gate = make_gate(paths, out, side)
        if args.preflight_only:
            with runner.output_lock(out, nonblocking=True):
                with runner.interrupt_handlers():
                    out.mkdir(parents=True, exist_ok=True)
                    result = portable_preflight(
                        args,
                        paths,
                        out,
                        side,
                        {
                            "protocol": PROTOCOL,
                            "status": "inconclusive",
                            "error": "preflight started; not yet validated",
                            "model_forward_executed": False,
                        },
                    )
                    runner.atomic_json(out / "preflight.json", result)

                    def publish_preflight(record):
                        runner.atomic_json(
                            out / "preflight.json",
                            portable_preflight(args, paths, out, side, record),
                        )

                    try:
                        inventory = gate()
                        implementation = runner.clean_implementation()
                        result = {
                            "protocol": PROTOCOL,
                            "status": "preflight_passed",
                            "inventory": inventory,
                            "implementation": implementation,
                            "model_forward_executed": False,
                            "content_hashes_verified": False,
                        }
                        code = 0
                    except (Exception, KeyboardInterrupt) as exc:
                        result = {
                            "protocol": PROTOCOL,
                            "status": "inconclusive",
                            "error": f"{type(exc).__name__}: {exc}",
                            "model_forward_executed": False,
                        }
                        code = 1
                    publish_preflight(result)
                    print(json.dumps(portable_preflight(args, paths, out, side, result), indent=2))
                    return code
        portable = portable_args(args, paths, out, side)
        return runner.supervise(
            out,
            lambda: prepare_context(paths, inspect_inputs(paths, out, side)),
            make_command(args, side),
            gate,
            portable,
        )
    except (Exception, KeyboardInterrupt) as exc:
        print(f"inconclusive: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
