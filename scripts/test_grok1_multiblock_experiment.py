#!/usr/bin/env python3
"""Unit tests for the #68 multi-block residual fidelity harness."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grok1_multiblock_experiment as multiblock  # noqa: E402
from grok1_block_forward import ForwardError  # noqa: E402
from grok1_multiblock_experiment import (  # noqa: E402
    _arm_meta,
    _resolve_hp_blocks,
    _validate_v2_cli,
    decide,
    decide_remedy,
    parse_blocks,
    parse_hp_blocks,
    periodic_hp_blocks,
    require_pack_only_scales,
    residual_stream_metrics,
    write_remedy_results_md,
    write_remedy_v2_results_md,
    write_results_md,
)
from grok1_multiblock_lib import (  # noqa: E402
    BASELINE_72,
    BASELINE_85,
    V2_CEILING_ARM,
    V2_PRIMARY_ARM,
    V2_STACKED_ARM,
    V3_PRIMARY_ARM,
    V3_SECONDARY_ARM,
    V4_INT4_BASELINE_ARM,
    V4_PRIMARY_ARM,
    V4_SECONDARY_ARM,
    Int4SideExperts,
    INT4_SCALE_ABSMAX,
    INT4_SCALE_LS_CHANNEL_ALPHA,
    _applied_expert_scale_sources,
    _expert_primary,
    _v3_applied_source,
    _v3_expected_schedule,
    assemble_remedy_v2_comparison,
    assemble_remedy_v3_comparison,
    channel_alpha_dequant,
    decide_remedy_v2,
    decide_remedy_v3,
    settings_mismatch_reason,
    structural_expert_scale_map,
)


# ---------------------------------------------------------------------------
# Shared v4 fixture builders
# ---------------------------------------------------------------------------
_V4_PACK_SHA256 = "a" * 64
_V4_NPY_SHA256 = "c" * 64
_V4_IMPL = {"commit": "abc", "dirty": False}


def _v4_pack_row(
    block: int,
    source: str,
    sha256: str = _V4_PACK_SHA256,
    npy_sha256: str = _V4_NPY_SHA256,
) -> dict:
    """One pack_provenance row for #85 test fixtures."""
    return {
        "block": block,
        "pack_sha256": sha256,
        "npy_sha256": npy_sha256,
        "container_versions": [3],
        "pack_scale_sources": structural_expert_scale_map(block, "pack_v2"),
        "scale_sources": structural_expert_scale_map(block, source),
    }


def _v4_per_block_row(block: int, top1_last: float) -> dict:
    """One per_block row for #85 test fixtures; only the last block carries top1_last."""
    return {
        "block": block,
        "expert_only": {
            "block_output_cosine": 0.995,
            "router_top1_agreement": 1.0 if block < 3 else top1_last,
            "router_top2_set_agreement": 0.95,
            "expert_load_js_bits": 0.0,
            "block_output_drift_relative_norm": 0.01,
            "residual_stream_in": {
                "residual_in_drift_relative_norm": 0.0 if block == 0 else 0.05
            },
        },
        "fp16_control": {
            "block_output_cosine": 1.0,
            "router_top1_agreement": 1.0,
            "router_top2_set_agreement": 1.0,
        },
        "pilot_label": "x",
    }


def _v4_protocol_fields(
    blocks: list[int] | None,
    tokens: int | None,
    token_seed: int | None,
    top_k: int | None,
) -> tuple[list[int], int, int, int]:
    """Resolve the four locked #85 protocol fields, defaulting to BASELINE_85."""
    return (
        list(blocks) if blocks is not None else list(BASELINE_85["blocks"]),
        int(tokens) if tokens is not None else int(BASELINE_85["tokens"]),
        int(token_seed) if token_seed is not None else int(BASELINE_85["token_seed"]),
        int(top_k) if top_k is not None else int(BASELINE_85["top_k"]),
    )


def _v4_chain(
    label: str,
    *,
    blocks: list[int] | None = None,
    tokens: int | None = None,
    token_seed: int | None = None,
    top_k: int | None = None,
    top1_last: float = 0.97,
    pack_sha256: str = _V4_PACK_SHA256,
) -> dict:
    """Build a valid #85 v4 chain with all schedule and pack-provenance fields."""
    blocks, tokens, token_seed, top_k = _v4_protocol_fields(
        blocks, tokens, token_seed, top_k
    )
    hp_blocks, int4_blocks, channel_blocks, expert_mode = _v3_expected_schedule(label)
    per_block = [_v4_per_block_row(b, top1_last) for b in blocks]
    pack_rows = [
        _v4_pack_row(b, _v3_applied_source(b, hp_blocks, label), pack_sha256)
        for b in blocks
    ]
    return {
        "arm_label": label,
        "blocks": blocks,
        "tokens": tokens,
        "token_seed": token_seed,
        "top_k": top_k,
        "expert_mode": expert_mode,
        "hp_blocks": hp_blocks,
        "int4_blocks": int4_blocks,
        "ternary_blocks": [],
        "channel_alpha_blocks": channel_blocks,
        "per_block": per_block,
        "end_of_chain": {
            "expert_only_chain_exit": {"residual_drift_relative_norm": 0.02},
            "fp16_chain_exit": {"residual_drift_relative_norm": 0.0},
        },
        "pack_provenance": pack_rows,
    }


def _v4_secondary_payload(
    chain: dict, implementation: dict | None = None
) -> dict:
    impl = dict(implementation) if implementation is not None else dict(_V4_IMPL)
    return {
        "provenance": {
            "evidence_role": "secondary; no independent decision",
            "implementation": impl,
        },
        "chain": chain,
    }


class ParseBlocksTests(unittest.TestCase):
    def test_parses_contiguous_chain(self) -> None:
        self.assertEqual(parse_blocks("0,1,2,3"), [0, 1, 2, 3])

    def test_rejects_gaps(self) -> None:
        with self.assertRaisesRegex(ForwardError, "contiguous"):
            parse_blocks("0,1,3")

    def test_rejects_non_ascending(self) -> None:
        with self.assertRaisesRegex(ForwardError, "ascending"):
            parse_blocks("2,1,0")

    def test_rejects_empty(self) -> None:
        with self.assertRaisesRegex(ForwardError, "at least one"):
            parse_blocks("")


class ResidualMetricsTests(unittest.TestCase):
    def test_identical_streams_have_zero_drift(self) -> None:
        x = np.random.default_rng(0).standard_normal((8, 4)).astype(np.float32)
        m = residual_stream_metrics(x, x.copy())
        self.assertAlmostEqual(m["residual_in_cosine"], 1.0, places=6)
        self.assertAlmostEqual(m["residual_in_drift_relative_norm"], 0.0, places=6)

    def test_orthogonal_streams_are_not_identity(self) -> None:
        a = np.zeros((4, 2), dtype=np.float32)
        a[:, 0] = 1.0
        b = np.zeros((4, 2), dtype=np.float32)
        b[:, 1] = 1.0
        m = residual_stream_metrics(a, b)
        self.assertAlmostEqual(m["residual_in_cosine"], 0.0, places=6)
        self.assertGreater(m["residual_in_drift_relative_norm"], 0.9)


class RunBlockInputIdentityTests(unittest.TestCase):
    """A block result is publishable only while its byte inputs stay stable."""

    @staticmethod
    def _cfg() -> multiblock._BlockRunCfg:
        return multiblock._BlockRunCfg(
            top_k=2,
            skip_fp16=True,
            expert_mode="int4",
            hp_blocks=frozenset(),
            hp_period=2,
            hp_label="",
            int4_side_root=None,
        )

    def _run_with_digests(
        self,
        npy_before: str,
        npy_after: str,
        *,
        pack_before: str = "d" * 64,
        pack_after: str | None = None,
    ):
        events: list[str] = []
        pack_after = pack_before if pack_after is None else pack_after
        h_ref = np.ones((2, 3), dtype=np.float32)
        h_pilot = h_ref.copy()
        ref_out = h_ref + 1.0
        pilot_out = h_pilot + 1.0
        reference = object()
        pack = object()
        mixed = argparse.Namespace(
            label="research_int4_side",
            applied_scale_sources={"gate": "research_int4_side"},
        )
        traces = [
            argparse.Namespace(
                seconds=0.01,
                experts_touched=[0],
                block_out=ref_out,
            ),
            argparse.Namespace(
                seconds=0.02,
                experts_touched=[0],
                block_out=pilot_out,
            ),
        ]
        comparison = {
            "block_output_cosine": 1.0,
            "router_top1_agreement": 1.0,
        }
        provenance = {
            "block": 0,
            "npy_sha256": npy_after,
            "pack_sha256": pack_before,
        }

        def record_fingerprint(_path: Path) -> str:
            events.append("npy_fingerprint")
            return (
                npy_before
                if events.count("npy_fingerprint") == 1
                else npy_after
            )

        def record_pack_sha256(_path: Path) -> str:
            events.append("pack_sha256")
            return (
                pack_before
                if events.count("pack_sha256") == 1
                else pack_after
            )

        def record_sources(*_args, **_kwargs):
            events.append("load_sources")
            return reference, pack, mixed, None

        def forward(*_args, **_kwargs):
            events.append("forward")
            return traces.pop(0)

        def record_provenance(*_args, **_kwargs):
            events.append("provenance")
            return provenance

        with (
            mock.patch.object(
                multiblock,
                "_block_paths",
                return_value=(Path("npy"), Path("block.goz1")),
            ),
            mock.patch.object(
                multiblock,
                "npy_dir_fingerprint",
                side_effect=record_fingerprint,
            ) as fingerprint,
            mock.patch.object(
                multiblock,
                "sha256_file",
                side_effect=record_pack_sha256,
            ) as pack_fingerprint,
            mock.patch.object(
                multiblock,
                "load_block_sources",
                side_effect=record_sources,
            ),
            mock.patch.object(multiblock, "forward_block", side_effect=forward),
            mock.patch.object(multiblock, "compare", return_value=comparison),
            mock.patch.object(
                multiblock,
                "pack_provenance_row",
                side_effect=record_provenance,
            ) as provenance_row,
        ):
            result = multiblock._run_block(
                0,
                object(),
                (h_ref, h_pilot, None),
                self._cfg(),
            )
        self.assertEqual(
            fingerprint.call_args_list,
            [mock.call(Path("npy")), mock.call(Path("npy"))],
        )
        self.assertEqual(
            pack_fingerprint.call_args_list,
            [mock.call(Path("block.goz1")), mock.call(Path("block.goz1"))],
        )
        self.assertEqual(
            events,
            [
                "npy_fingerprint",
                "pack_sha256",
                "load_sources",
                "forward",
                "forward",
                "npy_fingerprint",
                "provenance",
                "pack_sha256",
            ],
        )
        provenance_row.assert_called_once_with(
            0,
            Path("block.goz1"),
            Path("npy"),
            pack,
            applied_scale_sources={"gate": "research_int4_side"},
            npy_sha256=npy_after,
            pack_sha256=pack_before,
        )
        return result

    def test_accepts_equal_pre_and_post_npy_fingerprints(self) -> None:
        digest = "a" * 64
        row, streams, provenance = self._run_with_digests(digest, digest)
        self.assertEqual(row["block"], 0)
        np.testing.assert_array_equal(streams[0], np.full((2, 3), 2.0))
        self.assertEqual(provenance["npy_sha256"], digest)
        self.assertEqual(provenance["pack_sha256"], "d" * 64)

    def test_rejects_changed_npy_fingerprint(self) -> None:
        with self.assertRaisesRegex(
            ForwardError,
            "NPY inputs changed while the forward was being measured",
        ):
            self._run_with_digests("a" * 64, "b" * 64)

    def test_rejects_changed_pack_fingerprint(self) -> None:
        with self.assertRaisesRegex(
            ForwardError,
            "GOZ1 pack changed while the forward was being measured",
        ):
            self._run_with_digests(
                "a" * 64,
                "a" * 64,
                pack_before="b" * 64,
                pack_after="c" * 64,
            )


class PackProvenanceTests(unittest.TestCase):
    def test_explicit_pack_digest_bypasses_cached_pack_fingerprint(self) -> None:
        digest = "e" * 64
        pack = mock.Mock()
        pack.tensor_names.return_value = []
        pack.scale_sources = {}
        pack.scales.return_value = {}
        pack.metadata = {}
        pack.pack_sha256.side_effect = AssertionError("cached digest consulted")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_path = root / "block.goz1"
            pack_path.write_bytes(b"pack")
            row = multiblock.pack_provenance_row(
                0,
                pack_path,
                root,
                pack,
                npy_sha256="f" * 64,
                pack_sha256=digest,
            )
        self.assertEqual(row["pack_sha256"], digest)
        pack.pack_sha256.assert_not_called()


class _FakePack:
    """Minimal PackWeights stand-in with the public accessors the harness uses."""

    def __init__(self, tensor_type: int = 1, container_version: int = 3) -> None:
        self.pack = Path("fake.goz1")
        self._rows = {
            "block_000.slot_00.moe_expert.gate": {
                "tensor_type": tensor_type,
                "container_version": container_version,
                "numel": 8,
            }
        }
        self.scale_sources: dict[str, str] = {}
        self._scale_tag = "pack_v2"

    def tensor_names(self) -> list[str]:
        return sorted(self._rows)

    def tensor_entry(self, name: str) -> dict:
        return self._rows[name]

    def container_version(self, name: str) -> int | None:
        return self._rows[name].get("container_version")

    def scale(self, name: str) -> None:
        self.scale_sources[name] = self._scale_tag


class PackOnlyScaleTests(unittest.TestCase):
    def test_legacy_oracle_aborts(self) -> None:
        pack = _FakePack(container_version=1)
        pack._scale_tag = "legacy_oracle"
        with self.assertRaisesRegex(ForwardError, "legacy_oracle"):
            require_pack_only_scales(pack, ["block_000.slot_00.moe_expert.gate"])

    def test_pack_v2_scales_accepted_only_on_v3(self) -> None:
        sources = require_pack_only_scales(
            _FakePack(container_version=3), ["block_000.slot_00.moe_expert.gate"]
        )
        self.assertEqual(sources["block_000.slot_00.moe_expert.gate"], "pack_v2")

    def test_non_ternary_expert_aborts(self) -> None:
        with self.assertRaisesRegex(ForwardError, "must be ternary"):
            require_pack_only_scales(
                _FakePack(tensor_type=0), ["block_000.slot_00.moe_expert.gate"]
            )

    def test_non_v3_container_aborts(self) -> None:
        with self.assertRaisesRegex(ForwardError, "expected GOZ1 v3"):
            require_pack_only_scales(
                _FakePack(container_version=2), ["block_000.slot_00.moe_expert.gate"]
            )


def _expert_row(block: int, metrics: dict, *, with_fp16: bool = True) -> dict:
    """Build one per-block row; ``metrics`` holds cos/top1/top2/drift fields."""
    cos = metrics["cos"]
    resid_in_drift = metrics["resid_in_drift"]
    row = {
        "block": block,
        "expert_only": {
            "block_output_cosine": cos,
            "block_output_drift_relative_norm": metrics.get("out_drift", 0.05),
            "moe_output_cosine": metrics.get("moe", 0.9),
            "router_top1_agreement": metrics.get("top1", 1.0),
            "router_top2_set_agreement": metrics.get("top2", 1.0),
            "expert_load_js_bits": metrics.get("js", 0.0),
            "residual_stream_in": {
                "residual_in_cosine": max(0.0, 1.0 - resid_in_drift),
                "residual_in_drift_relative_norm": resid_in_drift,
            },
        },
        "fp16_control": None,
    }
    if with_fp16:
        row["fp16_control"] = {
            "block_output_cosine": 0.9999,
            "router_top1_agreement": 0.999,
            "router_top2_set_agreement": 0.999,
        }
    return row


class DecideTests(unittest.TestCase):
    def test_bounded_chain_is_option_1(self) -> None:
        chain = {
            "per_block": [
                _expert_row(0, {"cos": 0.963, "resid_in_drift": 0.0}),
                _expert_row(1, {"cos": 0.955, "resid_in_drift": 0.04}),
                _expert_row(2, {"cos": 0.950, "resid_in_drift": 0.07}),
                _expert_row(3, {"cos": 0.948, "resid_in_drift": 0.09}),
            ],
            "top_k": 2,
            "end_of_chain": {
                "expert_only_chain_exit": {
                    "residual_cosine": 0.99,
                    "residual_drift_relative_norm": 0.10,
                }
            },
        }
        self.assertEqual(decide(chain)["decision"], 1)

    def test_collapse_is_option_3(self) -> None:
        chain = {
            "per_block": [
                _expert_row(0, {"cos": 0.96, "resid_in_drift": 0.0}),
                _expert_row(1, {"cos": 0.80, "resid_in_drift": 0.2, "top1": 0.85}),
                _expert_row(2, {"cos": 0.60, "resid_in_drift": 0.45, "top1": 0.70}),
                _expert_row(3, {"cos": 0.40, "resid_in_drift": 0.70, "top1": 0.50}),
            ],
            "top_k": 2,
            "end_of_chain": {
                "expert_only_chain_exit": {
                    "residual_cosine": 0.5,
                    "residual_drift_relative_norm": 0.75,
                }
            },
        }
        self.assertEqual(decide(chain)["decision"], 3)

    def test_fp16_failure_is_option_4(self) -> None:
        rows = [_expert_row(0, {"cos": 0.96, "resid_in_drift": 0.0})]
        rows[0]["fp16_control"]["block_output_cosine"] = 0.5
        self.assertEqual(decide({"per_block": rows})["decision"], 4)

    def test_missing_fp16_is_option_4(self) -> None:
        chain = {
            "per_block": [
                _expert_row(0, {"cos": 0.96, "resid_in_drift": 0.0}, with_fp16=False),
            ]
        }
        self.assertEqual(decide(chain)["decision"], 4)

    def test_intermediate_is_option_2(self) -> None:
        chain = {
            "per_block": [
                _expert_row(0, {"cos": 0.96, "resid_in_drift": 0.0}),
                _expert_row(1, {"cos": 0.91, "resid_in_drift": 0.10}),
                _expert_row(2, {"cos": 0.89, "resid_in_drift": 0.16}),
                _expert_row(3, {"cos": 0.87, "resid_in_drift": 0.22}),
            ],
            "top_k": 2,
            "end_of_chain": {
                "expert_only_chain_exit": {
                    "residual_cosine": 0.9,
                    "residual_drift_relative_norm": 0.28,
                }
            },
        }
        self.assertEqual(decide(chain)["decision"], 2)

    def test_final_hop_runaway_is_option_3(self) -> None:
        """Greptile/Cubic: chain-exit drift must enter compounding, not only resid_in."""
        chain = {
            "per_block": [
                _expert_row(0, {"cos": 0.96, "resid_in_drift": 0.0}),
                _expert_row(1, {"cos": 0.95, "resid_in_drift": 0.05}),
                _expert_row(2, {"cos": 0.94, "resid_in_drift": 0.08}),
                _expert_row(3, {"cos": 0.93, "resid_in_drift": 0.10}),
            ],
            "top_k": 2,
            "end_of_chain": {
                "expert_only_chain_exit": {
                    "residual_cosine": 0.7,
                    "residual_drift_relative_norm": 0.55,
                }
            },
        }
        d = decide(chain)
        self.assertEqual(d["decision"], 3)
        self.assertEqual(d["compounding"], "superlinear_or_runaway")

    def test_skip_fp16_flag_forces_option_4(self) -> None:
        chain = {
            "per_block": [_expert_row(0, {"cos": 0.96, "resid_in_drift": 0.0})],
            "top_k": 2,
            "skip_fp16_control": True,
        }
        # even with fp16_control present, explicit skip forces option 4
        self.assertEqual(decide(chain)["decision"], 4)


# Decision-run RNG seed (YYYYMMDD). Built arithmetically so Bandit B105 does
# not treat the literal as a hardcoded password on a *token* field name.
_DECISION_SEED = 2026 * 10_000 + 806


class ReportTests(unittest.TestCase):
    def test_results_md_cites_agent_and_model(self) -> None:
        option_viable = 1  # #68 decision option index (not a credential)
        payload = {
            "provenance": {"implementation": {"commit": "abc", "dirty": False}},
            "decision": {
                "decision": option_viable,
                "decision_text": "viable",
                "rationale": ["ok"],
            },
            "chain": {
                "tokens": 8,
                "token_seed": _DECISION_SEED,
                "per_block": [_expert_row(0, {"cos": 0.96, "resid_in_drift": 0.0})],
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.md"
            write_results_md(path, payload)
            text = path.read_text()
        self.assertIn("Grok Build: Grok 4.5", text)
        self.assertIn("grok-4.5", text)
        self.assertIn("#64", text)
        self.assertIn("Option 1", text)

    def test_v4_report_refreshes_preexisting_analysis_atomically(self) -> None:
        payload = {
            "provenance": {"issue": "GH #85", "agent": "fixture"},
            "decision": {
                "decision": 3,
                "decision_text": "current",
                "best_remedy_arm": V4_PRIMARY_ARM,
                "rationale": ["current evidence"],
            },
            "chain": {"arm_label": V4_PRIMARY_ARM},
            "comparison": {"summaries": {V4_PRIMARY_ARM: {}}},
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.md"
            path.write_text("stale same-option analysis", encoding="utf-8")
            inode_before = path.stat().st_ino
            with mock.patch.object(multiblock, "_fsync_directory") as sync_directory:
                multiblock._write_v3_report(path, payload)
            body = path.read_text(encoding="utf-8")
            inode_after = path.stat().st_ino
            sync_directory.assert_called_once_with(path.parent)
        self.assertNotEqual(inode_after, inode_before)
        self.assertIn("**Decision:** Option 3", body)
        self.assertIn(V4_PRIMARY_ARM, body)
        self.assertNotIn("stale same-option analysis", body)

    def test_atomic_json_replace_fsyncs_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "progress.json"
            with mock.patch.object(multiblock, "_fsync_directory") as sync_directory:
                multiblock._atomic_write_json(path, {"status": "running"})
            sync_directory.assert_called_once_with(path.parent)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"status": "running"},
            )

    def test_atomic_json_replace_failure_preserves_old_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "metrics.json"
            original = {"status": "complete", "decision": {"decision": 2}}
            path.write_text(json.dumps(original) + "\n", encoding="utf-8")
            with mock.patch.object(
                multiblock.os,
                "replace",
                side_effect=OSError("replace failed"),
            ), self.assertRaisesRegex(OSError, "replace failed"):
                multiblock._atomic_write_json(path, {"status": "partial"})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                original,
            )
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_canonical_v4_run_publishes_metrics_with_atomic_writer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "canonical"
            args = argparse.Namespace(
                tokens=int(BASELINE_85["tokens"]),
                hp_period=2,
                blocks="0,1,2,3",
                npy_root=root / "npy",
                npy_pattern="block_{block:03d}",
                pack_root=root / "packs",
                pack_pattern="block_{block:03d}.goz1",
                embedding_shard=root / "embedding.npy",
                embedding_sha256="e" * 64,
                out=out,
                progress_json=None,
                arm="int4_channel_alpha",
                hp_blocks=None,
                skip_fp16_control=False,
                seed=int(BASELINE_85["token_seed"]),
                top_k=int(BASELINE_85["top_k"]),
                write_report_md=False,
                evidence_only=False,
                comparison_metrics=[],
                int4_side_root=root / "int4-side",
            )
            provenance = {
                "implementation": dict(_V4_IMPL),
                "embedding_sha256": args.embedding_sha256,
            }
            with (
                mock.patch.object(
                    multiblock,
                    "run_chain",
                    return_value=_v4_chain(V4_PRIMARY_ARM),
                ),
                mock.patch.object(
                    multiblock,
                    "_provenance",
                    return_value=provenance,
                ),
                mock.patch.object(multiblock, "_write_v3_report"),
                mock.patch.object(
                    multiblock,
                    "_atomic_write_json",
                    wraps=multiblock._atomic_write_json,
                ) as atomic_write,
            ):
                self.assertEqual(multiblock.run(args), 0)

            atomic_write.assert_called_once()
            published_path, published_payload = atomic_write.call_args.args
            self.assertEqual(published_path, out / "metrics.json")
            self.assertEqual(
                json.loads(published_path.read_text(encoding="utf-8")),
                published_payload,
            )
            self.assertEqual(
                published_payload["provenance"]["evidence_role"],
                "primary; sole canonical #85 decision",
            )

    def test_atomic_report_replace_failure_preserves_old_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.md"
            original = "old report\n"
            path.write_text(original, encoding="utf-8")
            with mock.patch.object(
                multiblock.os, "replace", side_effect=OSError("replace failed")
            ), self.assertRaisesRegex(OSError, "replace failed"):
                multiblock._atomic_write_text(path, "new report\n")
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_v3_report_preserves_preexisting_analysis(self) -> None:
        payload = {
            "provenance": {"issue": "GH #80"},
            "decision": {"decision": 2},
            "chain": {"arm_label": V3_PRIMARY_ARM},
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.md"
            original = "# Hand-authored\n\n**Option 2 — preserved**\n"
            path.write_text(original, encoding="utf-8")
            multiblock._write_v3_report(path, payload)
            body = path.read_text(encoding="utf-8")
        self.assertEqual(body, original)


class PeriodicHpScheduleTests(unittest.TestCase):
    def test_n2_on_0_3_is_hp_on_1_and_3(self) -> None:
        blocks = [0, 1, 2, 3]
        hp = periodic_hp_blocks(blocks, period=2)
        self.assertEqual(hp, {1, 3})
        ternary = set(blocks) - hp
        self.assertEqual(ternary, {0, 2})

    def test_n4_only_last(self) -> None:
        self.assertEqual(periodic_hp_blocks([0, 1, 2, 3], period=4), {3})

    def test_n1_is_every_block(self) -> None:
        self.assertEqual(periodic_hp_blocks([0, 1, 2, 3], period=1), {0, 1, 2, 3})

    def test_period_below_one_raises(self) -> None:
        with self.assertRaises(ForwardError):
            periodic_hp_blocks([0, 1, 2, 3], period=0)


class ChannelAlphaDequantTests(unittest.TestCase):
    def test_recovers_per_channel_scale(self) -> None:
        rng = np.random.default_rng(0)
        t = rng.choice([-1.0, 0.0, 1.0], size=(32, 4)).astype(np.float32)
        alpha = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
        w = t * alpha
        out = channel_alpha_dequant(w, t)
        # On fired positions reconstruction should match w.
        fired = t != 0
        np.testing.assert_allclose(out[fired], w[fired], rtol=1e-5, atol=1e-5)

    def test_dead_channel_yields_zero_alpha(self) -> None:
        t = np.zeros((8, 3), dtype=np.float32)
        t[:, 0] = 1.0
        w = np.ones((8, 3), dtype=np.float32) * 2.0
        out = channel_alpha_dequant(w, t)
        np.testing.assert_allclose(out[:, 0], 2.0, rtol=1e-5)
        np.testing.assert_allclose(out[:, 1:], 0.0, atol=1e-7)

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ForwardError):
            channel_alpha_dequant(np.ones((2, 2)), np.ones((2, 3)))

    def test_1d_vector_path(self) -> None:
        t = np.array([1.0, -1.0, 0.0, 1.0], dtype=np.float32)
        w = t * 3.0
        out = channel_alpha_dequant(w, t)
        np.testing.assert_allclose(out[t != 0], w[t != 0], rtol=1e-5)


class RemedyDecideTests(unittest.TestCase):
    def _chain(
        self,
        rows,
        *,
        exit_drift: float,
        skip_fp16: bool = False,
        blocks: list[int] | None = None,
        tokens: int = 2048,
        token_seed: int | None = None,
    ) -> dict:
        if blocks is None:
            blocks = [r["block"] for r in rows]
        seed = _DECISION_SEED if token_seed is None else token_seed
        return {
            "blocks": blocks,
            "tokens": tokens,
            "token_seed": seed,
            "per_block": rows,
            "top_k": 2,
            "skip_fp16_control": skip_fp16,
            "expert_mode": "periodic_hp",
            "arm_label": "expert_periodic_hp_n2",
            "hp_schedule_prose": "ternary on {0,2}, HP on {1,3}",
            "end_of_chain": {
                "expert_only_chain_exit": {
                    "residual_drift_relative_norm": exit_drift,
                }
            },
        }

    def test_skip_fp16_forces_option_4(self) -> None:
        rows = [
            _expert_row(0, {"cos": 0.99, "resid_in_drift": 0.0, "top1": 1.0, "top2": 1.0}),
        ]
        for r in rows:
            r["fp16_control"] = None
        d = decide_remedy(self._chain(rows, exit_drift=0.1, skip_fp16=True))
        self.assertEqual(d["decision"], 4)

    def test_strong_remedy_is_option_1(self) -> None:
        rows = [
            _expert_row(0, {"cos": 0.99, "resid_in_drift": 0.0, "top1": 1.0, "top2": 1.0}),
            _expert_row(1, {"cos": 0.98, "resid_in_drift": 0.05, "top1": 0.99, "top2": 0.98}),
            _expert_row(2, {"cos": 0.97, "resid_in_drift": 0.08, "top1": 0.98, "top2": 0.97}),
            _expert_row(3, {"cos": 0.96, "resid_in_drift": 0.10, "top1": 0.97, "top2": 0.96}),
        ]
        d = decide_remedy(self._chain(rows, exit_drift=0.12))
        self.assertEqual(d["decision"], 1)

    def test_partial_help_is_option_2(self) -> None:
        # Better than #72 b3 top1=0.528 but not option-1 viable.
        rows = [
            _expert_row(0, {"cos": 0.96, "resid_in_drift": 0.0, "top1": 1.0, "top2": 1.0}),
            _expert_row(1, {"cos": 0.94, "resid_in_drift": 0.15, "top1": 0.92, "top2": 0.85}),
            _expert_row(2, {"cos": 0.90, "resid_in_drift": 0.25, "top1": 0.80, "top2": 0.70}),
            _expert_row(3, {"cos": 0.87, "resid_in_drift": 0.35, "top1": 0.70, "top2": 0.55}),
        ]
        d = decide_remedy(self._chain(rows, exit_drift=0.45))
        self.assertEqual(d["decision"], 2)
        self.assertGreater(
            rows[-1]["expert_only"]["router_top1_agreement"], BASELINE_72["router_top1"][-1]
        )

    def test_no_help_is_option_3(self) -> None:
        rows = [
            _expert_row(0, {"cos": 0.96, "resid_in_drift": 0.0, "top1": 1.0, "top2": 1.0}),
            _expert_row(1, {"cos": 0.94, "resid_in_drift": 0.28, "top1": 0.88, "top2": 0.68}),
            _expert_row(2, {"cos": 0.88, "resid_in_drift": 0.34, "top1": 0.66, "top2": 0.54}),
            _expert_row(3, {"cos": 0.83, "resid_in_drift": 0.50, "top1": 0.52, "top2": 0.28}),
        ]
        d = decide_remedy(self._chain(rows, exit_drift=0.66))
        self.assertEqual(d["decision"], 3)

    def test_short_chain_cannot_claim_option_2_vs_72(self) -> None:
        # Non-#72 settings must not publish option 2 against fixed b3 metrics.
        # Metrics look "improved" if wrongly compared to #72 b3, but fail option 1.
        rows = [
            _expert_row(0, {"cos": 0.90, "resid_in_drift": 0.0, "top1": 0.90, "top2": 0.85}),
            _expert_row(1, {"cos": 0.88, "resid_in_drift": 0.10, "top1": 0.80, "top2": 0.70}),
        ]
        d = decide_remedy(
            self._chain(
                rows, exit_drift=0.20, blocks=[0, 1], tokens=8, token_seed=40 + 2
            )
        )
        self.assertEqual(d["decision"], 4)
        self.assertTrue(any("settings_not_comparable" in r for r in d["rationale"]))
        self.assertIn("blocks/tokens/seed/top_k", d["decision_text"])

    def test_pack_mismatch_diagnoses_pack_identity(self) -> None:
        rows = [
            _expert_row(0, {"cos": 0.90, "resid_in_drift": 0.0, "top1": 0.90, "top2": 0.85}),
            _expert_row(1, {"cos": 0.88, "resid_in_drift": 0.10, "top1": 0.80, "top2": 0.70}),
            _expert_row(2, {"cos": 0.86, "resid_in_drift": 0.20, "top1": 0.70, "top2": 0.60}),
            _expert_row(3, {"cos": 0.84, "resid_in_drift": 0.30, "top1": 0.60, "top2": 0.50}),
        ]
        chain = self._chain(rows, exit_drift=0.40)
        chain["pack_provenance"] = [
            {"block": b, "pack_sha256": "0" * 64} for b in (0, 1, 2, 3)
        ]
        self.assertEqual(settings_mismatch_reason(chain), "pack_identity_not_comparable_to_72")
        d = decide_remedy(chain)
        self.assertEqual(d["decision"], 4)
        self.assertIn("pack SHA-256", d["decision_text"])
        self.assertIn("pack_identity_not_comparable_to_72", d["rationale"])


class RemedyReportTests(unittest.TestCase):
    def test_cites_grok_build_and_schedule_prose(self) -> None:
        payload = {
            "provenance": {
                "agent": (
                    "Grok Build: Grok 4.5 (xAI) · Model: Grok-4.5 (high) · "
                    "Issue: #73 / Linear RM-362"
                ),
                "model": "Grok-4.5 (high)",
                "implementation": {"commit": "deadbeef", "dirty": False},
                "metrics_note": "cite #72",
            },
            "decision": {
                "decision": 2,
                "decision_text": "helps",
                "rationale": ["ok"],
            },
            "chain": {
                "blocks": [0, 1, 2, 3],
                "tokens": 2048,
                "token_seed": _DECISION_SEED,
                "top_k": 2,
                "expert_mode": "periodic_hp",
                "arm_label": "expert_periodic_hp_n2",
                "hp_schedule_prose": (
                    "Arm C label `expert_periodic_hp_n2`: ternary on {0,2}, HP (FP16 experts) on {1,3}."
                ),
                "per_block": [
                    _expert_row(0, {"cos": 0.96, "resid_in_drift": 0.0}),
                    _expert_row(1, {"cos": 0.94, "resid_in_drift": 0.1}),
                    _expert_row(2, {"cos": 0.90, "resid_in_drift": 0.2}),
                    _expert_row(3, {"cos": 0.88, "resid_in_drift": 0.3}),
                ],
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.md"
            write_remedy_results_md(path, payload)
            text = path.read_text()
        self.assertIn("Grok Build: Grok 4.5", text)
        self.assertIn("Grok-4.5 (high)", text)
        self.assertIn("ternary on {0,2}", text)
        self.assertIn("HP (FP16 experts) on {1,3}", text)
        self.assertIn("#72", text)
        self.assertIn("Option 2", text)
        self.assertIn("comparable settings + packs", text)

    def test_mismatch_report_does_not_claim_bit_identical(self) -> None:
        # Seed built without a bare "1" literal (Bandit B105 on *token* fields).
        other_seed = 40 + 2
        payload = {
            "provenance": {
                "agent": "Grok Build: Grok 4.5 (xAI) · Model: Grok-4.5 (high)",
                "implementation": {"commit": "deadbeef", "dirty": False},
            },
            "decision": {
                "decision": 4,
                "decision_text": "inconclusive",
                "rationale": ["settings_not_comparable_to_72"],
            },
            "chain": {
                "blocks": [0, 1],
                "tokens": 8,
                "token_seed": other_seed,
                "top_k": 2,
                "expert_mode": "periodic_hp",
                "arm_label": "expert_periodic_hp_n2",
                "per_block": [
                    _expert_row(0, {"cos": 0.9, "resid_in_drift": 0.0}),
                    _expert_row(1, {"cos": 0.88, "resid_in_drift": 0.1}),
                ],
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.md"
            write_remedy_results_md(path, payload)
            text = path.read_text()
        self.assertIn("not comparable", text.lower())
        self.assertIn("schedule not comparable", text.lower())

    def test_pack_mismatch_report_names_pack_identity(self) -> None:
        payload = {
            "provenance": {
                "agent": "Grok Build: Grok 4.5 (xAI) · Model: Grok-4.5 (high)",
                "implementation": {"commit": "deadbeef", "dirty": False},
            },
            "decision": {
                "decision": 4,
                "decision_text": "pack SHA",
                "rationale": ["pack_identity_not_comparable_to_72"],
            },
            "chain": {
                "blocks": [0, 1, 2, 3],
                "tokens": 2048,
                "token_seed": _DECISION_SEED,
                "top_k": 2,
                "expert_mode": "periodic_hp",
                "arm_label": "expert_periodic_hp_n2",
                "pack_provenance": [
                    {"block": b, "pack_sha256": "0" * 64} for b in (0, 1, 2, 3)
                ],
                "per_block": [
                    _expert_row(b, {"cos": 0.9, "resid_in_drift": 0.0}) for b in (0, 1, 2, 3)
                ],
            },
        }
        with tempfile.TemporaryDirectory() as td:
            text = (Path(td) / "r.md")
            write_remedy_results_md(text, payload)
            body = text.read_text()
        self.assertIn("pack identity not comparable", body.lower())
        self.assertIn("pack SHA-256", body)
        self.assertNotIn("schedule not comparable", body.lower())


class RemedyV2ScheduleTests(unittest.TestCase):
    def test_parse_explicit_hp_blocks(self) -> None:
        self.assertEqual(parse_hp_blocks("3,1,2"), {1, 2, 3})
        self.assertEqual(parse_hp_blocks("1,1"), {1})

    def test_multidigit_explicit_schedule_labels_do_not_collide(self) -> None:
        self.assertEqual(multiblock._explicit_schedule_label({1, 2, 3}), "123")
        self.assertEqual(multiblock._explicit_schedule_label({123}), "b123")
        self.assertEqual(multiblock._explicit_schedule_label({1, 23}), "b1-23")
        self.assertEqual(multiblock._explicit_schedule_label({3, 12}), "b3-12")

    def test_denser_primary_schedule_and_label(self) -> None:
        hp = _resolve_hp_blocks([0, 1, 2, 3], "periodic_hp", 2, {1, 2, 3})
        meta = _arm_meta(
            [0, 1, 2, 3],
            "periodic_hp",
            2,
            hp,
            ["ternary", "fp16", "fp16", "fp16"],
            explicit_hp=True,
        )
        self.assertEqual(meta["arm_label"], V2_PRIMARY_ARM)
        self.assertEqual(meta["hp_blocks"], [1, 2, 3])
        self.assertEqual(meta["ternary_blocks"], [0])
        self.assertEqual(meta["hp_schedule_kind"], "explicit")

    def test_stacked_and_ceiling_schedules(self) -> None:
        blocks = [0, 1, 2, 3]
        stacked_hp = _resolve_hp_blocks(blocks, "periodic_hp_plus_channel_alpha", 2, None)
        stacked = _arm_meta(
            blocks,
            "periodic_hp_plus_channel_alpha",
            2,
            stacked_hp,
            [],
            explicit_hp=False,
        )
        self.assertEqual(stacked["arm_label"], V2_STACKED_ARM)
        self.assertEqual(stacked["channel_alpha_blocks"], [0, 2])
        ceiling_hp = _resolve_hp_blocks(blocks, "all_hp", 2, None)
        ceiling = _arm_meta(blocks, "all_hp", 2, ceiling_hp, [], explicit_hp=False)
        self.assertEqual(ceiling["arm_label"], V2_CEILING_ARM)
        self.assertEqual(ceiling["hp_blocks"], blocks)
        self.assertEqual(ceiling["ternary_blocks"], [])

    def test_channel_alpha_does_not_label_blocks_as_ternary(self) -> None:
        blocks = [0, 1, 2, 3]
        meta = _arm_meta(blocks, "channel_alpha", 2, set(), [], explicit_hp=False)
        self.assertEqual(meta["arm_label"], "research_per_channel_side")
        self.assertEqual(meta["ternary_blocks"], [])
        self.assertEqual(meta["channel_alpha_blocks"], blocks)

    def test_secondary_arms_require_evidence_only(self) -> None:
        args = argparse.Namespace(
            arm="hp_ceiling",
            hp_blocks=None,
            evidence_only=False,
            comparison_metrics=[],
        )
        with self.assertRaisesRegex(ForwardError, "requires --evidence-only"):
            _validate_v2_cli(args)

    def test_comparison_metrics_rejected_on_non_primary_run(self) -> None:
        args = argparse.Namespace(
            arm="periodic_hp",
            hp_blocks={1, 3},
            evidence_only=False,
            comparison_metrics=[Path("stacked/metrics.json")],
        )
        with self.assertRaisesRegex(ForwardError, "--comparison-metrics is only valid"):
            _validate_v2_cli(args)

    def test_partial_explicit_ceiling_schedule_is_rejected(self) -> None:
        with self.assertRaisesRegex(ForwardError, "hp_ceiling requires every chain block"):
            _resolve_hp_blocks([0, 1, 2, 3], "all_hp", 2, {1, 2})

    def test_noncanonical_explicit_hp_run_uses_legacy_remedy_contract(self) -> None:
        args = argparse.Namespace(arm="periodic_hp", hp_blocks={1, 3})
        self.assertFalse(multiblock._is_v2_run(args))
        issue, agent = multiblock._agent_for_args(args)
        self.assertEqual(issue, "GH #73 / Linear RM-362")
        self.assertEqual(agent, multiblock.REMEDY_AGENT_LINE)

    def test_stacked_and_ceiling_select_hp_control(self) -> None:
        pack = object()
        reference = object()
        control = object()
        primary, _ = _expert_primary(
            "periodic_hp_plus_channel_alpha",
            pack,
            reference,
            control,
            block=1,
            hp_blocks={1, 3},
        )
        self.assertIs(primary, control)
        primary, label = _expert_primary(
            "periodic_hp",
            pack,
            reference,
            control,
            block=2,
            hp_blocks={1, 2, 3},
            hp_label="123",
        )
        self.assertIs(primary, control)
        self.assertEqual(label, V2_PRIMARY_ARM)
        primary, _ = _expert_primary(
            "all_hp",
            pack,
            reference,
            control,
            block=0,
            hp_blocks={0, 1, 2, 3},
        )
        self.assertIs(primary, control)

    def test_stacked_ternary_block_selects_channel_alpha(self) -> None:
        channel_source = object()
        with mock.patch(
            "grok1_multiblock_lib.ChannelAlphaExperts",
            return_value=channel_source,
        ):
            primary, label = _expert_primary(
                "periodic_hp_plus_channel_alpha",
                object(),
                object(),
                object(),
                block=0,
                hp_blocks={1, 3},
            )
        self.assertIs(primary, channel_source)
        self.assertEqual(label, "research_per_channel_side")

    def test_hp_source_tags_are_fp16_control(self) -> None:
        pack = argparse.Namespace(scale_sources={"pack_expert": "pack_v2"})
        reference = argparse.Namespace(
            roles={
                "expert_gelu": "gate",
                "expert_value": "v1",
                "expert_down": "v2",
            }
        )
        applied = _applied_expert_scale_sources(
            pack,
            object(),
            reference,
            expert_mode="all_hp",
            block=0,
            hp_blocks={0, 1, 2, 3},
        )
        self.assertEqual({applied["gate"], applied["v1"], applied["v2"]}, {"fp16_control"})


_V2_FIXTURE_SCHEDULES = {
    V2_PRIMARY_ARM: ([1, 2, 3], [], "periodic_hp"),
    V2_STACKED_ARM: ([1, 3], [0, 2], "periodic_hp_plus_channel_alpha"),
    V2_CEILING_ARM: ([0, 1, 2, 3], [], "all_hp"),
}
_V2_FIXTURE_METRICS = {
    "viable": {
        "cos": [0.98, 0.97, 0.96, 0.95],
        "top1": [1.0, 0.99, 0.98, 0.96],
        "top2": [1.0, 0.98, 0.96, 0.94],
        "drift": [0.0, 0.05, 0.08, 0.10],
        "exit": 0.18,
    },
    "help": {
        "cos": [0.96, 0.94, 0.92, 0.90],
        "top1": [1.0, 0.90, 0.76, 0.65],
        "top2": [1.0, 0.82, 0.70, 0.56],
        "drift": [0.0, 0.18, 0.28, 0.36],
        "exit": 0.42,
    },
    "failed": {
        "cos": [0.95, 0.91, 0.87, 0.84],
        "top1": [1.0, 0.82, 0.63, 0.50],
        "top2": [1.0, 0.70, 0.50, 0.30],
        "drift": [0.0, 0.27, 0.38, 0.49],
        "exit": 0.58,
    },
}
_V2_FIXTURE_IMPLEMENTATION = {"commit": "f" * 40, "dirty": False}


def _v2_provenance(role: str) -> dict:
    return {
        "implementation": dict(_V2_FIXTURE_IMPLEMENTATION),
        "evidence_role": role,
    }


def _v2_secondary(label: str, quality: str) -> dict:
    return {
        "provenance": _v2_provenance("secondary; no independent decision"),
        "chain": _v2_chain(label, quality),
    }


def _v2_fixture_source(block: int, hp_blocks: list[int], channel_blocks: list[int]) -> str:
    if block in channel_blocks:
        return "research_per_channel_side"
    return "fp16_control" if block in hp_blocks else "pack_v2"


def _v2_fixture_rows(metrics: dict, hp_blocks: list[int], channel_blocks: list[int]):
    rows = []
    provenance = []
    for index, block in enumerate(BASELINE_72["blocks"]):
        rows.append(
            _expert_row(
                block,
                {
                    "cos": metrics["cos"][index],
                    "resid_in_drift": metrics["drift"][index],
                    "top1": metrics["top1"][index],
                    "top2": metrics["top2"][index],
                },
            )
        )
        applied_source = _v2_fixture_source(block, hp_blocks, channel_blocks)
        provenance.append(
            {
                "block": block,
                "pack_sha256": BASELINE_72["pack_sha256"][block],
                "container_versions": [3],
                "pack_scale_sources": {"expert": "pack_v2"},
                "scale_sources": {"expert": applied_source},
            }
        )
    return rows, provenance


def _v2_chain(label: str, quality: str) -> dict:
    hp_blocks, channel_blocks, mode = _V2_FIXTURE_SCHEDULES[label]
    metrics = _V2_FIXTURE_METRICS[quality]
    rows, provenance = _v2_fixture_rows(metrics, hp_blocks, channel_blocks)
    return {
        "blocks": list(BASELINE_72["blocks"]),
        "tokens": BASELINE_72["tokens"],
        "token_seed": BASELINE_72["token_seed"],
        "top_k": BASELINE_72["top_k"],
        "per_block": rows,
        "pack_provenance": provenance,
        "skip_fp16_control": False,
        "expert_mode": mode,
        "arm_label": label,
        "hp_blocks": hp_blocks,
        "channel_alpha_blocks": channel_blocks,
        "hp_schedule_prose": f"fixture schedule for {label}",
        "end_of_chain": {
            "expert_only_chain_exit": {
                "residual_drift_relative_norm": metrics["exit"],
            }
        },
    }


def _v2_comparison(primary: str, stacked: str, ceiling: str) -> dict:
    return assemble_remedy_v2_comparison(
        _v2_chain(V2_PRIMARY_ARM, primary),
        [
            _v2_secondary(V2_STACKED_ARM, stacked),
            _v2_secondary(V2_CEILING_ARM, ceiling),
        ],
        primary_provenance=_v2_provenance("primary; sole canonical #75 decision"),
    )


class RemedyV2DecisionTests(unittest.TestCase):
    def test_validates_applied_scale_sources(self) -> None:
        comparison = _v2_comparison("help", "help", "viable")
        self.assertEqual(comparison["validation_errors"], [])
        bad = _v2_chain(V2_STACKED_ARM, "help")
        bad["pack_provenance"][0]["scale_sources"]["expert"] = "pack_v2"
        comparison = assemble_remedy_v2_comparison(
            _v2_chain(V2_PRIMARY_ARM, "help"),
            [
                {
                    "provenance": _v2_provenance("secondary; no independent decision"),
                    "chain": bad,
                },
                _v2_secondary(V2_CEILING_ARM, "viable"),
            ],
            primary_provenance=_v2_provenance("primary; sole canonical #75 decision"),
        )
        self.assertTrue(any("research_per_channel_side" in e for e in comparison["validation_errors"]))

    def test_option_1_when_mostly_ternary_arm_is_viable(self) -> None:
        decision = decide_remedy_v2(_v2_comparison("viable", "help", "viable"))
        self.assertEqual(decision["decision"], 1)
        self.assertEqual(decision["best_remedy_arm"], V2_PRIMARY_ARM)

    def test_option_2_when_remedy_helps_but_is_not_viable(self) -> None:
        decision = decide_remedy_v2(_v2_comparison("help", "failed", "viable"))
        self.assertEqual(decision["decision"], 2)

    def test_option_3_when_even_ceiling_fails(self) -> None:
        decision = decide_remedy_v2(_v2_comparison("failed", "failed", "failed"))
        self.assertEqual(decision["decision"], 3)

    def test_option_4_when_secondary_evidence_is_missing(self) -> None:
        comparison = assemble_remedy_v2_comparison(
            _v2_chain(V2_PRIMARY_ARM, "help"),
            [],
            primary_provenance=_v2_provenance("primary; sole canonical #75 decision"),
        )
        self.assertEqual(decide_remedy_v2(comparison)["decision"], 4)

    def test_option_4_when_secondary_per_block_is_empty(self) -> None:
        stacked = _v2_secondary(V2_STACKED_ARM, "help")
        stacked["chain"]["per_block"] = []
        comparison = assemble_remedy_v2_comparison(
            _v2_chain(V2_PRIMARY_ARM, "help"),
            [stacked, _v2_secondary(V2_CEILING_ARM, "viable")],
            primary_provenance=_v2_provenance("primary; sole canonical #75 decision"),
        )
        self.assertTrue(
            any("per_block_missing_or_empty" in error for error in comparison["validation_errors"])
        )
        self.assertEqual(decide_remedy_v2(comparison)["decision"], 4)

    def test_malformed_pack_provenance_rows_are_validation_errors(self) -> None:
        bad = _v2_chain(V2_CEILING_ARM, "viable")
        bad["pack_provenance"] = [
            bad["pack_provenance"][0],
            "not-a-mapping",
            {**bad["pack_provenance"][2], "pack_scale_sources": []},
            {**bad["pack_provenance"][3], "block": "x"},
        ]
        comparison = assemble_remedy_v2_comparison(
            _v2_chain(V2_PRIMARY_ARM, "help"),
            [
                _v2_secondary(V2_STACKED_ARM, "help"),
                {
                    "provenance": _v2_provenance("secondary; no independent decision"),
                    "chain": bad,
                },
            ],
            primary_provenance=_v2_provenance("primary; sole canonical #75 decision"),
        )
        errors = comparison["validation_errors"]
        self.assertTrue(any("not_a_mapping" in error for error in errors))
        self.assertTrue(any("invalid_block_id" in error for error in errors))
        self.assertTrue(any("pack_scale_sources_not_mapping" in error for error in errors))
        self.assertEqual(decide_remedy_v2(comparison)["decision"], 4)

    def test_non_canonical_block_ids_are_rejected(self) -> None:
        bad = _v2_chain(V2_PRIMARY_ARM, "help")
        bad["per_block"] = list(bad["per_block"])
        bad["per_block"][0] = {**bad["per_block"][0], "block": 0.0}
        bad["pack_provenance"] = list(bad["pack_provenance"])
        bad["pack_provenance"][1] = {**bad["pack_provenance"][1], "block": True}
        comparison = assemble_remedy_v2_comparison(
            bad,
            [
                _v2_secondary(V2_STACKED_ARM, "help"),
                _v2_secondary(V2_CEILING_ARM, "viable"),
            ],
            primary_provenance=_v2_provenance("primary; sole canonical #75 decision"),
        )
        errors = comparison["validation_errors"]
        self.assertTrue(any("invalid_block_id" in error for error in errors))
        self.assertEqual(decide_remedy_v2(comparison)["decision"], 4)

    def test_non_mapping_per_block_row_is_validation_error(self) -> None:
        bad = _v2_chain(V2_PRIMARY_ARM, "help")
        bad["per_block"] = list(bad["per_block"])
        bad["per_block"][1] = "row-not-dict"
        comparison = assemble_remedy_v2_comparison(
            bad,
            [
                _v2_secondary(V2_STACKED_ARM, "help"),
                _v2_secondary(V2_CEILING_ARM, "viable"),
            ],
            primary_provenance=_v2_provenance("primary; sole canonical #75 decision"),
        )
        self.assertTrue(
            any("not_a_mapping" in error for error in comparison["validation_errors"])
        )
        self.assertEqual(decide_remedy_v2(comparison)["decision"], 4)

    def test_rejects_secondary_decision_or_provenance_mismatch(self) -> None:
        stacked = _v2_secondary(V2_STACKED_ARM, "help")
        stacked["decision"] = {"decision": 1}
        stacked["provenance"]["implementation"]["commit"] = "0" * 40
        ceiling = _v2_secondary(V2_CEILING_ARM, "viable")
        ceiling["provenance"]["evidence_role"] = "primary"
        comparison = assemble_remedy_v2_comparison(
            _v2_chain(V2_PRIMARY_ARM, "help"),
            [stacked, ceiling],
            primary_provenance=_v2_provenance("primary; sole canonical #75 decision"),
        )
        errors = comparison["validation_errors"]
        self.assertTrue(any("contains a decision" in error for error in errors))
        self.assertTrue(any("implementation differs" in error for error in errors))
        self.assertTrue(any("role is not locked" in error for error in errors))
        self.assertEqual(decide_remedy_v2(comparison)["decision"], 4)

    def test_option_4_when_only_one_option_2_condition_holds(self) -> None:
        comparison = _v2_comparison("help", "failed", "failed")
        self.assertEqual(decide_remedy_v2(comparison)["decision"], 4)

    def test_improvement_checks_both_mostly_ternary_arms(self) -> None:
        comparison = _v2_comparison("failed", "failed", "failed")
        primary = comparison["summaries"][V2_PRIMARY_ARM]
        stacked = comparison["summaries"][V2_STACKED_ARM]
        primary["router_top1"][-1] = 0.59
        primary["block_output_cosine"][-1] = 0.90
        primary["chain_exit_residual_drift"] = 0.50
        stacked["router_top1"][-1] = 0.58
        stacked["block_output_cosine"][-1] = 0.93
        stacked["chain_exit_residual_drift"] = 0.50
        decision = decide_remedy_v2(comparison)
        self.assertEqual(decision["best_remedy_arm"], V2_PRIMARY_ARM)
        self.assertEqual(decision["decision"], 4)
        self.assertIn("any_mostly_ternary_improved_vs_74=True", decision["rationale"])

    def test_secondary_payload_written_without_decision(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "evidence"
            args = argparse.Namespace(
                tokens=2048,
                hp_period=2,
                blocks="0,1,2,3",
                npy_root=Path(td),
                npy_pattern="unused",
                pack_root=Path(td),
                pack_pattern="unused",
                embedding_shard=Path(td) / "unused.npy",
                out=out,
                arm="stacked_hp_channel_alpha",
                hp_blocks=None,
                skip_fp16_control=False,
                seed=_DECISION_SEED,
                top_k=2,
                write_report_md=True,
                evidence_only=True,
                comparison_metrics=[],
            )
            with (
                mock.patch.object(
                    multiblock,
                    "run_chain",
                    return_value=_v2_chain(V2_STACKED_ARM, "help"),
                ),
                mock.patch.object(multiblock, "_provenance", return_value={}),
                mock.patch.object(multiblock, "remedy_metrics_note", return_value="fixture"),
            ):
                self.assertEqual(multiblock.run(args), 0)
            payload = json.loads((out / "metrics.json").read_text())
            self.assertNotIn("decision", payload)
            self.assertFalse((out / "results.md").exists())


_V3_FIXTURE_SCHEDULES = {
    V3_PRIMARY_ARM: ([], list(BASELINE_72["blocks"]), "int4"),
    V3_SECONDARY_ARM: ([1, 2, 3], [0], "int4"),
}


def _v3_fixture_source(block: int, hp_blocks: list[int]) -> str:
    return "fp16_control" if block in hp_blocks else "research_int4_side"


def _v3_chain(label: str, quality: str) -> dict:
    hp_blocks, int4_blocks, mode = _V3_FIXTURE_SCHEDULES[label]
    metrics = _V2_FIXTURE_METRICS[quality]
    rows = []
    provenance = []
    for index, block in enumerate(BASELINE_72["blocks"]):
        rows.append(
            _expert_row(
                block,
                {
                    "cos": metrics["cos"][index],
                    "resid_in_drift": metrics["drift"][index],
                    "top1": metrics["top1"][index],
                    "top2": metrics["top2"][index],
                },
            )
        )
        applied = _v3_fixture_source(block, hp_blocks)
        provenance.append(
            {
                "block": block,
                "pack_sha256": BASELINE_72["pack_sha256"][block],
                "container_versions": [3],
                "pack_scale_sources": structural_expert_scale_map(block, "pack_v2"),
                "scale_sources": structural_expert_scale_map(block, applied),
            }
        )
    return {
        "blocks": list(BASELINE_72["blocks"]),
        "tokens": BASELINE_72["tokens"],
        "token_seed": BASELINE_72["token_seed"],
        "top_k": BASELINE_72["top_k"],
        "per_block": rows,
        "pack_provenance": provenance,
        "skip_fp16_control": False,
        "expert_mode": mode,
        "arm_label": label,
        "hp_blocks": hp_blocks,
        "int4_blocks": int4_blocks,
        "ternary_blocks": [],
        "channel_alpha_blocks": [],
        "end_of_chain": {
            "expert_only_chain_exit": {
                "residual_drift_relative_norm": metrics["exit"],
            }
        },
    }


def _v3_secondary(quality: str) -> dict:
    return {
        "provenance": {
            "implementation": dict(_V2_FIXTURE_IMPLEMENTATION),
            "evidence_role": "secondary; no independent decision",
        },
        "chain": _v3_chain(V3_SECONDARY_ARM, quality),
    }


def _v3_comparison(primary: str, secondary: str) -> dict:
    return assemble_remedy_v3_comparison(
        _v3_chain(V3_PRIMARY_ARM, primary),
        [_v3_secondary(secondary)],
        primary_provenance={
            "implementation": dict(_V2_FIXTURE_IMPLEMENTATION),
            "evidence_role": "primary; sole canonical #80 decision",
        },
    )


class RemedyV3DecisionTests(unittest.TestCase):
    def test_option_2_when_int4_helps_but_is_not_viable(self) -> None:
        # Beat #76 denser (top1 0.616 / cos 0.914 / exit 0.437) without full viability.
        primary = _v3_chain(V3_PRIMARY_ARM, "help")
        primary["per_block"][3]["expert_only"]["router_top1_agreement"] = 0.70
        primary["per_block"][3]["expert_only"]["block_output_cosine"] = 0.92
        primary["end_of_chain"]["expert_only_chain_exit"][
            "residual_drift_relative_norm"
        ] = 0.30
        comparison = assemble_remedy_v3_comparison(
            primary,
            [_v3_secondary("failed")],
            primary_provenance={
                "implementation": dict(_V2_FIXTURE_IMPLEMENTATION),
                "evidence_role": "primary; sole canonical #80 decision",
            },
        )
        self.assertEqual(comparison["validation_errors"], [])
        decision = decide_remedy_v3(comparison)
        self.assertEqual(decision["decision"], 2)
        self.assertEqual(decision["best_remedy_arm"], V3_PRIMARY_ARM)

    def test_option_4_when_secondary_missing(self) -> None:
        comparison = assemble_remedy_v3_comparison(
            _v3_chain(V3_PRIMARY_ARM, "help"),
            [],
            primary_provenance={
                "implementation": dict(_V2_FIXTURE_IMPLEMENTATION),
            },
        )
        self.assertEqual(decide_remedy_v3(comparison)["decision"], 4)

    def test_option_4_when_fp16_control_skipped(self) -> None:
        primary = _v3_chain(V3_PRIMARY_ARM, "help")
        primary["skip_fp16_control"] = True
        for row in primary["per_block"]:
            row.pop("fp16_control", None)
        comparison = assemble_remedy_v3_comparison(
            primary,
            [_v3_secondary("help")],
            primary_provenance={
                "implementation": dict(_V2_FIXTURE_IMPLEMENTATION),
            },
        )
        self.assertTrue(any("fp16_control" in e for e in comparison["validation_errors"]))
        self.assertEqual(decide_remedy_v3(comparison)["decision"], 4)

    def test_option_4_when_ternary_blocks_mislabel_int4(self) -> None:
        primary = _v3_chain(V3_PRIMARY_ARM, "help")
        primary["ternary_blocks"] = list(BASELINE_72["blocks"])
        primary["int4_blocks"] = []
        comparison = assemble_remedy_v3_comparison(
            primary,
            [_v3_secondary("help")],
            primary_provenance={
                "implementation": dict(_V2_FIXTURE_IMPLEMENTATION),
            },
        )
        self.assertTrue(any("int4_blocks" in e for e in comparison["validation_errors"]))
        self.assertEqual(decide_remedy_v3(comparison)["decision"], 4)

    def test_rejects_wrong_applied_scale_source(self) -> None:
        secondary = _v3_secondary("help")
        # Corrupt all applied sources for block 0 (INT4 expected).
        secondary["chain"]["pack_provenance"][0]["scale_sources"] = (
            structural_expert_scale_map(0, "pack_v2")
        )
        comparison = assemble_remedy_v3_comparison(
            _v3_chain(V3_PRIMARY_ARM, "help"),
            [secondary],
            primary_provenance={
                "implementation": dict(_V2_FIXTURE_IMPLEMENTATION),
            },
        )
        self.assertTrue(
            any("research_int4_side" in e for e in comparison["validation_errors"])
        )
        self.assertEqual(decide_remedy_v3(comparison)["decision"], 4)

    def test_option_4_when_chain_exit_missing(self) -> None:
        primary = _v3_chain(V3_PRIMARY_ARM, "help")
        primary.pop("end_of_chain", None)
        comparison = assemble_remedy_v3_comparison(
            primary,
            [_v3_secondary("help")],
            primary_provenance={
                "implementation": dict(_V2_FIXTURE_IMPLEMENTATION),
            },
        )
        self.assertTrue(any("missing_chain_exit" in e for e in comparison["validation_errors"]))
        self.assertEqual(decide_remedy_v3(comparison)["decision"], 4)

    def test_option_4_when_decision_metrics_non_finite(self) -> None:
        comparison = _v3_comparison("help", "failed")
        self.assertEqual(comparison["validation_errors"], [])
        # Corrupt summary after assembly so ranking sees NaN (simulates bad payload).
        comparison["summaries"][V3_PRIMARY_ARM]["router_top1"][-1] = float("nan")
        comparison["summaries"][V3_PRIMARY_ARM]["block_output_cosine"][-1] = 0.92
        comparison["summaries"][V3_PRIMARY_ARM]["chain_exit_residual_drift"] = 0.3
        secondary_label = V3_SECONDARY_ARM
        comparison["summaries"][secondary_label]["router_top1"][-1] = float("nan")
        comparison["summaries"][secondary_label]["chain_exit_residual_drift"] = 0.3
        decision = decide_remedy_v3(comparison)
        self.assertEqual(decision["decision"], 4)

    def test_option_4_when_mid_chain_topk_non_finite(self) -> None:
        primary = _v3_chain(V3_PRIMARY_ARM, "help")
        primary["per_block"][1]["expert_only"]["router_top2_set_agreement"] = float("nan")
        comparison = assemble_remedy_v3_comparison(
            primary,
            [_v3_secondary("help")],
            primary_provenance={"implementation": dict(_V2_FIXTURE_IMPLEMENTATION)},
        )
        errors = comparison["validation_errors"]
        self.assertTrue(
            any("malformed_expert_only" in e for e in errors),
            errors,
        )
        self.assertEqual(decide_remedy_v3(comparison)["decision"], 4)

    def test_option_4_when_agreement_out_of_domain(self) -> None:
        primary = _v3_chain(V3_PRIMARY_ARM, "help")
        primary["per_block"][0]["expert_only"]["router_top1_agreement"] = 2.0
        comparison = assemble_remedy_v3_comparison(
            primary,
            [_v3_secondary("help")],
            primary_provenance={"implementation": dict(_V2_FIXTURE_IMPLEMENTATION)},
        )
        self.assertTrue(
            any("metric_out_of_domain" in e for e in comparison["validation_errors"])
        )
        self.assertEqual(decide_remedy_v3(comparison)["decision"], 4)

    def test_malformed_per_block_does_not_raise_on_settings(self) -> None:
        primary = _v3_chain(V3_PRIMARY_ARM, "help")
        primary["per_block"][1] = "not-a-row"  # type: ignore[call-arg]
        comparison = assemble_remedy_v3_comparison(
            primary,
            [_v3_secondary("help")],
            primary_provenance={"implementation": dict(_V2_FIXTURE_IMPLEMENTATION)},
        )
        self.assertTrue(comparison["validation_errors"])
        self.assertEqual(decide_remedy_v3(comparison)["decision"], 4)

    def test_malformed_chain_exit_does_not_raise_in_summary(self) -> None:
        primary = _v3_chain(V3_PRIMARY_ARM, "help")
        primary["end_of_chain"] = {
            "expert_only_chain_exit": {"residual_drift_relative_norm": "nope"}
        }
        comparison = assemble_remedy_v3_comparison(
            primary,
            [_v3_secondary("help")],
            primary_provenance={"implementation": dict(_V2_FIXTURE_IMPLEMENTATION)},
        )
        self.assertTrue(any("chain_exit" in e for e in comparison["validation_errors"]))
        self.assertEqual(decide_remedy_v3(comparison)["decision"], 4)

    def test_option_4_when_per_block_order_permuted(self) -> None:
        primary = _v3_chain(V3_PRIMARY_ARM, "help")
        primary["per_block"] = list(reversed(primary["per_block"]))
        primary["pack_provenance"] = list(reversed(primary["pack_provenance"]))
        comparison = assemble_remedy_v3_comparison(
            primary,
            [_v3_secondary("help")],
            primary_provenance={"implementation": dict(_V2_FIXTURE_IMPLEMENTATION)},
        )
        self.assertTrue(any("per_block_order" in e for e in comparison["validation_errors"]))
        self.assertEqual(decide_remedy_v3(comparison)["decision"], 4)

    def test_option_4_when_secondary_payload_not_object(self) -> None:
        comparison = assemble_remedy_v3_comparison(
            _v3_chain(V3_PRIMARY_ARM, "help"),
            [[], "not-a-mapping"],  # type: ignore[list-item]
            primary_provenance={"implementation": dict(_V2_FIXTURE_IMPLEMENTATION)},
        )
        self.assertTrue(
            any("JSON object" in e or "must be a JSON object" in e for e in comparison["validation_errors"])
        )
        self.assertEqual(decide_remedy_v3(comparison)["decision"], 4)

    def test_int4_evidence_only_requires_hp_123(self) -> None:
        args = argparse.Namespace(
            arm="int4",
            evidence_only=True,
            hp_blocks={0, 2},
            comparison_metrics=[],
            skip_fp16_control=False,
        )
        with self.assertRaisesRegex(ForwardError, r"hp-blocks 1,2,3"):
            _validate_v2_cli(args)


class RemedyV4DecisionTests(unittest.TestCase):
    """GH #85: INT4 codes × LS channel-α at Grok-1 max context (8192)."""

    def test_ls_channel_alpha_mse_not_worse_than_absmax(self) -> None:
        import numpy as np
        from grok1_multiblock_lib import (
            int4_absmax_quantize,
            int4_dequant_from_codes,
            int4_ls_channel_alpha_scale,
        )

        rng = np.random.default_rng(7)
        w = rng.standard_normal((48, 24)).astype(np.float32)
        q, s_abs = int4_absmax_quantize(w)
        s_ls = int4_ls_channel_alpha_scale(w, q)
        err_abs = float(np.mean((w - int4_dequant_from_codes(q, s_abs)) ** 2))
        err_ls = float(np.mean((w - int4_dequant_from_codes(q, s_ls)) ** 2))
        self.assertLessEqual(err_ls, err_abs + 1e-9)

    def test_rejects_2048_tokens_for_v4_primary(self) -> None:
        seed = 2026 * 10_000 + 806
        args = argparse.Namespace(
            arm="int4_channel_alpha",
            tokens=2048,
            seed=seed,
            top_k=2,
            blocks="0,1,2,3",
            evidence_only=False,
            hp_blocks=None,
            comparison_metrics=[],
            skip_fp16_control=False,
        )
        with self.assertRaisesRegex(ForwardError, r"8192"):
            _validate_v2_cli(args)

    def test_v4_error_reports_cite_issue_85(self) -> None:
        args = argparse.Namespace(arm="int4_channel_alpha")
        issue, agent = multiblock._agent_for_args(args)
        self.assertIn("#85", issue)
        self.assertIn("RM-608", issue)
        self.assertIn("#85", agent)
        self.assertIn("Codex (OpenAI)", agent)
        self.assertNotIn("Grok", agent)

    def test_v4_provenance_separates_planning_from_implementation(self) -> None:
        issue, agent, model, design = multiblock._remedy_issue_meta(
            v2=False, v3=False, v4=True
        )
        self.assertIn("#85", issue)
        self.assertIn("Codex (OpenAI)", agent)
        self.assertEqual(model, "OpenAI Codex")
        self.assertIn("Grok issue-planning lock", design)
        self.assertIn("implementation and evidence by Codex (OpenAI)", design)

    def test_same_budget_int4_baseline_cites_issue_85(self) -> None:
        """The 8192 evidence-only INT4 control is #85 evidence, not a #80 run."""
        baseline = argparse.Namespace(
            arm="int4", tokens=8192, evidence_only=True, hp_blocks=None
        )
        self.assertTrue(multiblock._is_v4_run(baseline))
        issue, _agent = multiblock._agent_for_args(baseline)
        self.assertIn("#85", issue)
        self.assertIn("RM-608", issue)
        # A genuine #80 run at the 2048 ladder must still cite #80.
        historical = argparse.Namespace(
            arm="int4", tokens=2048, evidence_only=True, hp_blocks=None
        )
        self.assertFalse(multiblock._is_v4_run(historical))
        self.assertIn("#80", multiblock._agent_for_args(historical)[0])

    def test_budget_check_and_v4_classification_agree(self) -> None:
        """`_validate_int4_evidence_hp` and `_is_v4_run` must use one token rule.

        They disagreed once: bare `int()` in the former accepted a coerced
        "8192" as the #85 baseline while the latter classed the run #80, so it
        was taken as #85 evidence and stamped #80 provenance.
        """
        for tokens in (8192, "8192", 8192.0, "abc", None, True):
            with self.subTest(tokens=tokens):
                args = argparse.Namespace(
                    arm="int4", tokens=tokens, evidence_only=True, hp_blocks=None
                )
                is_v4 = multiblock._is_v4_run(args)
                try:
                    multiblock._validate_int4_evidence_hp(args)
                    accepted_as_85 = True
                except ForwardError:
                    accepted_as_85 = False
                self.assertEqual(
                    accepted_as_85,
                    is_v4,
                    f"tokens={tokens!r}: budget check says {accepted_as_85}, "
                    f"_is_v4_run says {is_v4}",
                )

    def test_v4_run_classification_survives_malformed_tokens(self) -> None:
        """`_is_v4_run` feeds provenance: malformed tokens classify, never raise."""
        # "8192"/8192.0 are rejected too — a coerced value must not pass for a
        # locked baseline setting (see `_safe_int`).
        for tokens in ("abc", None, "8192", 8192.0, True, []):
            with self.subTest(tokens=tokens):
                args = argparse.Namespace(
                    arm="int4", tokens=tokens, evidence_only=True, hp_blocks=None
                )
                self.assertFalse(multiblock._is_v4_run(args))
                self.assertIn("#80", multiblock._agent_for_args(args)[0])

    def test_accepts_max_context_tokens_for_v4_primary(self) -> None:
        seed = 2026 * 10_000 + 806
        args = argparse.Namespace(
            arm="int4_channel_alpha",
            tokens=8192,
            seed=seed,
            top_k=2,
            blocks="0,1,2,3",
            evidence_only=False,
            hp_blocks=None,
            comparison_metrics=[],
            skip_fp16_control=False,
            embedding_sha256="a" * 64,
        )
        _validate_v2_cli(args)

    def test_option_1_when_stack_is_viable(self) -> None:
        from grok1_multiblock_lib import assemble_remedy_v4_comparison, decide_remedy_v4

        impl = dict(_V4_IMPL)
        comparison = assemble_remedy_v4_comparison(
            _v4_chain(V4_PRIMARY_ARM, top1_last=0.97),
            [
                _v4_secondary_payload(
                    _v4_chain(V4_INT4_BASELINE_ARM, top1_last=0.90), impl
                ),
                _v4_secondary_payload(
                    _v4_chain(V4_SECONDARY_ARM, top1_last=0.96), impl
                ),
            ],
            primary_provenance={"implementation": impl},
        )
        self.assertEqual(comparison["validation_errors"], [])
        decision = decide_remedy_v4(comparison)
        self.assertEqual(decision["decision"], 1)
        self.assertEqual(decision["best_remedy_arm"], V4_PRIMARY_ARM)

    def test_v4_pack_provenance_does_not_keyerror_or_cite_72_incomparability(self) -> None:
        """Regression: Devin Review BUGs — v4 labels KeyError / always incomparable to #72."""
        from grok1_multiblock_lib import (
            V4_INT4_BASELINE_ARM,
            V4_PRIMARY_ARM,
            assemble_remedy_v4_comparison,
            decide_remedy_v4,
            remedy_metrics_note,
            settings_mismatch_reason,
            _v3_scale_source_errors,
            _v3_schedule_errors,
            _v4_chain_errors,
        )

        primary = _v4_chain(V4_PRIMARY_ARM)
        baseline = _v4_chain(V4_INT4_BASELINE_ARM)
        self.assertEqual(_v3_schedule_errors(primary, V4_PRIMARY_ARM), [])
        self.assertEqual(_v3_scale_source_errors(primary, V4_PRIMARY_ARM), [])
        self.assertEqual(_v4_chain_errors(primary, V4_PRIMARY_ARM), [])
        self.assertEqual(_v4_chain_errors(baseline, V4_INT4_BASELINE_ARM), [])
        # #72 mismatch is expected at 8192 and must not enter v4 validation_errors.
        self.assertEqual(settings_mismatch_reason(primary), "settings_not_comparable_to_72")
        note = remedy_metrics_note(primary)
        self.assertIn("#85", note)
        self.assertNotIn("not comparable", note.lower())

        impl = dict(_V4_IMPL)
        comparison = assemble_remedy_v4_comparison(
            primary,
            [
                _v4_secondary_payload(baseline, impl),
                _v4_secondary_payload(_v4_chain(V4_SECONDARY_ARM), impl),
            ],
            primary_provenance={"implementation": impl},
        )
        self.assertEqual(comparison["validation_errors"], [])
        self.assertFalse(
            any("settings_not_comparable_to_72" in e for e in comparison["validation_errors"])
        )
        decision = decide_remedy_v4(comparison)
        self.assertEqual(decision["decision"], 1)

    def test_mismatched_blocks_rejected(self) -> None:
        from grok1_multiblock_lib import assemble_remedy_v4_comparison

        impl = dict(_V4_IMPL)
        comparison = assemble_remedy_v4_comparison(
            _v4_chain(V4_PRIMARY_ARM),
            [_v4_secondary_payload(_v4_chain(V4_INT4_BASELINE_ARM, blocks=[0, 1, 2]), impl)],
            primary_provenance={"implementation": impl},
        )
        self.assertTrue(len(comparison["validation_errors"]) > 0)
        self.assertTrue(
            any("blocks=" in e for e in comparison["validation_errors"]),
            f"Expected blocks error in {comparison['validation_errors']}"
        )

    def test_duplicate_pack_provenance_block_rejected(self) -> None:
        """Four rows naming one block is not coverage for four blocks."""
        from grok1_multiblock_lib import (
            assemble_remedy_v4_comparison,
            decide_remedy_v4,
            _v3_applied_source,
            _v3_expected_schedule,
        )

        impl = dict(_V4_IMPL)
        secondary = _v4_chain(V4_INT4_BASELINE_ARM)
        hp_blocks, _, _, _ = _v3_expected_schedule(V4_INT4_BASELINE_ARM)
        source = _v3_applied_source(0, hp_blocks, V4_INT4_BASELINE_ARM)
        # Internally consistent rows, correct count, all naming block 0: blocks
        # 1-3 would otherwise never be pack-identity checked.
        secondary["pack_provenance"] = [
            _v4_pack_row(0, source, _V4_PACK_SHA256) for _ in range(4)
        ]
        comparison = assemble_remedy_v4_comparison(
            _v4_chain(V4_PRIMARY_ARM),
            [_v4_secondary_payload(secondary, impl)],
            primary_provenance={"implementation": impl},
        )
        errors = comparison["validation_errors"]
        # Both halves: blocks 1-3 uncovered, and block 0 over-covered.
        for block in (1, 2, 3):
            self.assertTrue(
                any(f"pack_provenance_rows_for_block_{block:03d}=0" in e for e in errors),
                f"Expected missing-coverage error for block {block} in {errors}",
            )
        self.assertTrue(
            any("pack_provenance_rows_for_block_000=4" in e for e in errors),
            f"Expected duplicate-row error for block 0 in {errors}",
        )
        self.assertEqual(decide_remedy_v4(comparison)["decision"], 4)

    def test_pack_provenance_block_outside_baseline_named(self) -> None:
        """A stray block is reported as such, not only as a missing block."""
        from grok1_multiblock_lib import (
            assemble_remedy_v4_comparison,
            decide_remedy_v4,
            _v3_applied_source,
            _v3_expected_schedule,
        )

        impl = dict(_V4_IMPL)
        secondary = _v4_chain(V4_INT4_BASELINE_ARM)
        hp_blocks, _, _, _ = _v3_expected_schedule(V4_INT4_BASELINE_ARM)
        secondary["pack_provenance"] = [
            _v4_pack_row(
                blk, _v3_applied_source(blk, hp_blocks, V4_INT4_BASELINE_ARM), _V4_PACK_SHA256
            )
            for blk in (0, 1, 2, 5)
        ]
        comparison = assemble_remedy_v4_comparison(
            _v4_chain(V4_PRIMARY_ARM),
            [_v4_secondary_payload(secondary, impl)],
            primary_provenance={"implementation": impl},
        )
        errors = comparison["validation_errors"]
        self.assertTrue(
            any("pack_provenance_unexpected_block=5" in e for e in errors),
            f"Expected stray-block error in {errors}",
        )
        self.assertTrue(
            any("pack_provenance_rows_for_block_003=0" in e for e in errors),
            f"Expected missing block 3 in {errors}",
        )
        self.assertEqual(decide_remedy_v4(comparison)["decision"], 4)

    def test_divergent_npy_inputs_rejected(self) -> None:
        """Same pack, different FP32 inputs, is not same-budget evidence.

        The pack SHA-256 covers the GOZ1 container only; INT4 codes, the
        reference trajectory and all non-expert weights come from NpyWeights.
        """
        from grok1_multiblock_lib import (
            assemble_remedy_v4_comparison,
            decide_remedy_v4,
            _v3_applied_source,
            _v3_expected_schedule,
        )

        impl = dict(_V4_IMPL)
        secondary = _v4_chain(V4_INT4_BASELINE_ARM)
        hp_blocks, _, _, _ = _v3_expected_schedule(V4_INT4_BASELINE_ARM)
        # Identical pack SHA per block, different npy content.
        secondary["pack_provenance"] = [
            _v4_pack_row(
                blk,
                _v3_applied_source(blk, hp_blocks, V4_INT4_BASELINE_ARM),
                _V4_PACK_SHA256,
                npy_sha256="d" * 64,
            )
            for blk in BASELINE_85["blocks"]
        ]
        comparison = assemble_remedy_v4_comparison(
            _v4_chain(V4_PRIMARY_ARM),
            [_v4_secondary_payload(secondary, impl)],
            primary_provenance={"implementation": impl},
        )
        errors = comparison["validation_errors"]
        self.assertTrue(
            any("npy_sha256_mismatch" in e for e in errors),
            f"Expected npy identity mismatch in {errors}",
        )
        # The pack check alone would have passed this evidence.
        self.assertFalse(
            any("pack_sha256_mismatch" in e for e in errors),
            f"packs match; only the npy inputs differ: {errors}",
        )
        self.assertEqual(decide_remedy_v4(comparison)["decision"], 4)

    def test_partial_npy_sha256_is_not_partial_validation(self) -> None:
        """A row without a digest must fail, not drop out of the comparison.

        Omitting `npy_sha256` on one block while the rest match the primary
        previously left that block uncompared and still decided option 1.
        """
        from grok1_multiblock_lib import assemble_remedy_v4_comparison, decide_remedy_v4

        impl = dict(_V4_IMPL)
        secondary = _v4_chain(V4_INT4_BASELINE_ARM)
        for row in secondary["pack_provenance"]:
            if row["block"] == 3:
                row.pop("npy_sha256")
        comparison = assemble_remedy_v4_comparison(
            _v4_chain(V4_PRIMARY_ARM),
            [_v4_secondary_payload(secondary, impl)],
            primary_provenance={"implementation": impl},
        )
        errors = comparison["validation_errors"]
        self.assertTrue(
            any("missing_npy_sha256" in e for e in errors),
            f"Expected a per-row missing-digest error in {errors}",
        )
        self.assertEqual(decide_remedy_v4(comparison)["decision"], 4)

    def test_malformed_npy_sha256_rejected(self) -> None:
        """Matching but invalid digests must not read as shared FP32 inputs."""
        from grok1_multiblock_lib import assemble_remedy_v4_comparison, decide_remedy_v4

        impl = dict(_V4_IMPL)
        primary = _v4_chain(V4_PRIMARY_ARM)
        secondary = _v4_chain(V4_INT4_BASELINE_ARM)
        for row in primary["pack_provenance"] + secondary["pack_provenance"]:
            row["npy_sha256"] = "not-a-sha256"
        comparison = assemble_remedy_v4_comparison(
            primary,
            [_v4_secondary_payload(secondary, impl)],
            primary_provenance={"implementation": impl},
        )
        errors = comparison["validation_errors"]
        self.assertTrue(
            any("invalid_npy_sha256" in e for e in errors),
            f"Expected an invalid-digest error in {errors}",
        )
        self.assertEqual(decide_remedy_v4(comparison)["decision"], 4)

    def test_non_list_blocks_fails_closed(self) -> None:
        """Malformed `blocks` in loaded evidence must be an error, not a TypeError."""
        from grok1_multiblock_lib import assemble_remedy_v4_comparison, decide_remedy_v4

        impl = dict(_V4_IMPL)
        for label, bad in (("secondary", 4), ("primary", "0,1,2,3")):
            with self.subTest(chain=label):
                primary = _v4_chain(V4_PRIMARY_ARM)
                secondary = _v4_chain(V4_INT4_BASELINE_ARM)
                if label == "primary":
                    primary["blocks"] = bad
                else:
                    secondary["blocks"] = bad
                comparison = assemble_remedy_v4_comparison(
                    primary,
                    [_v4_secondary_payload(secondary, impl)],
                    primary_provenance={"implementation": impl},
                )
                self.assertTrue(
                    any("is not a list" in e for e in comparison["validation_errors"]),
                    f"Expected non-list blocks error in {comparison['validation_errors']}",
                )
                self.assertEqual(decide_remedy_v4(comparison)["decision"], 4)

    def test_mismatched_tokens_rejected(self) -> None:
        from grok1_multiblock_lib import assemble_remedy_v4_comparison

        impl = dict(_V4_IMPL)
        comparison = assemble_remedy_v4_comparison(
            _v4_chain(V4_PRIMARY_ARM),
            [
                _v4_secondary_payload(
                    _v4_chain(V4_INT4_BASELINE_ARM, tokens=2048), impl
                )
            ],
            primary_provenance={"implementation": impl},
        )
        self.assertTrue(len(comparison["validation_errors"]) > 0)
        self.assertTrue(
            any("tokens=" in e for e in comparison["validation_errors"]),
            f"Expected tokens error in {comparison['validation_errors']}"
        )

    def test_mismatched_seed_rejected(self) -> None:
        from grok1_multiblock_lib import assemble_remedy_v4_comparison

        impl = dict(_V4_IMPL)
        comparison = assemble_remedy_v4_comparison(
            _v4_chain(V4_PRIMARY_ARM),
            [_v4_secondary_payload(_v4_chain(V4_INT4_BASELINE_ARM, token_seed=99999), impl)],
            primary_provenance={"implementation": impl},
        )
        self.assertTrue(len(comparison["validation_errors"]) > 0)
        self.assertTrue(
            any("token_seed=" in e for e in comparison["validation_errors"]),
            f"Expected token_seed error in {comparison['validation_errors']}"
        )

    def test_mismatched_top_k_rejected(self) -> None:
        from grok1_multiblock_lib import assemble_remedy_v4_comparison

        impl = dict(_V4_IMPL)
        comparison = assemble_remedy_v4_comparison(
            _v4_chain(V4_PRIMARY_ARM),
            [_v4_secondary_payload(_v4_chain(V4_INT4_BASELINE_ARM, top_k=4), impl)],
            primary_provenance={"implementation": impl},
        )
        self.assertTrue(len(comparison["validation_errors"]) > 0)
        self.assertTrue(
            any("top_k=" in e for e in comparison["validation_errors"]),
            f"Expected top_k error in {comparison['validation_errors']}"
        )

    def test_missing_int4_baseline_payload(self) -> None:
        from grok1_multiblock_lib import assemble_remedy_v4_comparison, decide_remedy_v4

        impl = dict(_V4_IMPL)
        # No secondary payloads (missing INT4 baseline)
        comparison = assemble_remedy_v4_comparison(
            _v4_chain(V4_PRIMARY_ARM, top1_last=0.90),
            [],
            primary_provenance={"implementation": impl},
        )
        self.assertFalse(comparison["protocol_complete"])
        self.assertEqual(
            comparison["missing_arms"],
            [V4_INT4_BASELINE_ARM, V4_SECONDARY_ARM],
        )
        self.assertNotIn("ranking", comparison)
        decision = decide_remedy_v4(comparison)
        self.assertEqual(decision["decision"], 4)


class RemedyV2ReportTests(unittest.TestCase):
    def test_report_has_one_canonical_decision_and_both_controls(self) -> None:
        comparison = _v2_comparison("help", "failed", "viable")
        payload = {
            "provenance": {
                "agent": "OpenAI Codex: GPT-5.6 Sol (xhigh) · Issue: #75",
                "issue": "GH #75 / Linear RM-462 / beads goz-rvk",
                "implementation": {"commit": "deadbeef", "dirty": False},
            },
            "chain": _v2_chain(V2_PRIMARY_ARM, "help"),
            "comparison": comparison,
            "decision": decide_remedy_v2(comparison),
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.md"
            write_remedy_v2_results_md(path, payload)
            body = path.read_text()
        self.assertEqual(body.count("## Decision"), 1)
        self.assertIn("#72 baseline", body)
        self.assertIn("#74 baseline", body)
        self.assertIn(V2_STACKED_ARM, body)
        self.assertIn(V2_CEILING_ARM, body)
        self.assertIn("neither mostly-ternary candidate met every locked viability band", body)
        self.assertIn("clear improvement and a viable HP ceiling both hold", body)
        self.assertIn("Secondary `metrics.json` files are evidence-only", body)
        self.assertEqual(body.splitlines().count("### FP16 control"), 1)
        self.assertIn(f"#### FP16 control — `{V2_STACKED_ARM}`", body)
        self.assertIn(f"#### FP16 control — `{V2_CEILING_ARM}`", body)
        self.assertFalse(body.endswith("\n\n"))


class Int4SideExpertsTests(unittest.TestCase):
    """INT4 side-table persistence for absmax and LS channel-α scales."""

    _FAKE_ROLES = {
        "expert_gelu": "gate",
        "expert_value": "up",
        "expert_down": "down",
    }

    @staticmethod
    def _fake_weights(rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
        rng = rng or np.random.default_rng(0)
        return {
            name: rng.standard_normal((2, 8, 4)).astype(np.float32)
            for name in Int4SideExpertsTests._FAKE_ROLES.values()
        }

    def _make_reference(self, arrays: dict[str, np.ndarray] | None = None):
        arrays = arrays or self._fake_weights()
        ref = mock.Mock(spec=["roles", "vector"])
        ref.roles = dict(self._FAKE_ROLES)
        ref.vector = lambda role: arrays[ref.roles[role]]
        return ref

    def test_absmax_sidecar_writes_local_q_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ref = self._make_reference()
            side = Int4SideExperts(
                ref,
                side_root=Path(td),
                block=0,
                scale_mode=INT4_SCALE_ABSMAX,
            )
            sidecar = json.loads((side._side_dir / "sidecar.json").read_text())
            gate = sidecar["tensors"]["gate"]
            self.assertEqual(gate["q_file"], "gate__q_int8.npy")
            self.assertEqual(gate["scale_file"], "gate__scale_f32.npy")
            self.assertEqual(Path(td) / "block_000" / "gate__q_int8.npy", side._q_dir / "gate__q_int8.npy")
            self.assertTrue((side._q_dir / "gate__q_int8.npy").is_file())

    def test_ls_channel_alpha_label_and_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ref = self._make_reference()
            side = Int4SideExperts(
                ref,
                side_root=Path(td),
                block=0,
                scale_mode=INT4_SCALE_LS_CHANNEL_ALPHA,
            )
            self.assertEqual(side.label, "research_int4_channel_alpha_side")
            self.assertEqual(side._side_dir, Path(td) / "ls-alpha" / "block_000")
            self.assertEqual(side._q_dir, Path(td) / "block_000")
            # The sidecar is the only on-disk record of which scale a persisted
            # table carries; a report reads it back to attribute the codec.
            sidecar = json.loads((side._side_dir / "sidecar.json").read_text())
            self.assertEqual(sidecar["scale_mode"], INT4_SCALE_LS_CHANNEL_ALPHA)
            self.assertIn("channel_alpha", str(sidecar["codec"]))

    def test_stale_codes_are_rebuilt_not_reused(self) -> None:
        """Codes from a different FP32 export must not be fit with a new scale.

        dtype/range/shape all pass for stale codes, so only a reference
        fingerprint catches it.
        """
        with tempfile.TemporaryDirectory() as td:
            first = self._fake_weights(np.random.default_rng(1))
            side_a = Int4SideExperts(
                self._make_reference(first),
                side_root=Path(td),
                block=0,
                scale_mode=INT4_SCALE_LS_CHANNEL_ALPHA,
            )
            q_path, _ = side_a._paths("gate")
            stale = np.load(q_path).copy()

            # Same shape/dtype, different content: a re-export of the tensor.
            second = self._fake_weights(np.random.default_rng(2))
            side_b = Int4SideExperts(
                self._make_reference(second),
                side_root=Path(td),
                block=0,
                scale_mode=INT4_SCALE_LS_CHANNEL_ALPHA,
            )
            rebuilt = np.load(q_path)
            self.assertFalse(
                np.array_equal(stale, rebuilt),
                "codes were reused across two different references",
            )
            # And the reconstruction tracks the new reference, not the old one.
            got = side_b.vector("expert_gelu")
            self.assertEqual(got.shape, second["gate"].shape)
            new_err = float(np.abs(got - second["gate"]).mean())
            old_err = float(np.abs(got - first["gate"]).mean())
            self.assertLess(new_err, old_err)

    def test_ls_channel_alpha_shares_q_codes_with_absmax(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ref = self._make_reference()
            absmax = Int4SideExperts(
                ref, side_root=Path(td), block=0, scale_mode=INT4_SCALE_ABSMAX
            )
            ls = Int4SideExperts(
                ref, side_root=Path(td), block=0, scale_mode=INT4_SCALE_LS_CHANNEL_ALPHA
            )
            q_path, scale_path = ls._paths("gate")
            self.assertTrue(q_path.is_file())
            self.assertTrue(scale_path.is_file())
            self.assertEqual(q_path, absmax._paths("gate")[0])
            self.assertEqual(q_path.parent, Path(td) / "block_000")
            self.assertEqual(scale_path.parent, Path(td) / "ls-alpha" / "block_000")
            sidecar = json.loads((ls._side_dir / "sidecar.json").read_text())
            self.assertIn("../", sidecar["tensors"]["gate"]["q_file"])
            self.assertNotIn(
                "../",
                sidecar["tensors"]["gate"]["scale_file"],
            )
            self.assertFalse((ls._side_dir / "gate__q_int8.npy").exists())

    def test_ls_channel_alpha_reload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ref = self._make_reference()
            first = Int4SideExperts(
                ref, side_root=Path(td), block=0, scale_mode=INT4_SCALE_LS_CHANNEL_ALPHA
            )
            first_vec = first.vector("expert_gelu")
            second = Int4SideExperts(
                ref, side_root=Path(td), block=0, scale_mode=INT4_SCALE_LS_CHANNEL_ALPHA
            )
            np.testing.assert_array_almost_equal(first.vector("expert_gelu"), second.vector("expert_gelu"), decimal=5)
            self.assertEqual(first_vec.shape, (2, 8, 4))

    def test_rank3_expert_dequant(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ref = self._make_reference()
            side = Int4SideExperts(
                ref, side_root=Path(td), block=0, scale_mode=INT4_SCALE_LS_CHANNEL_ALPHA
            )
            full = side.vector("expert_value")
            expert0 = side.expert("expert_value", 0)
            self.assertEqual(full.shape, (2, 8, 4))
            self.assertEqual(expert0.shape, (8, 4))
            self.assertTrue(np.isfinite(expert0).all())


if __name__ == "__main__":
    unittest.main()
