#!/usr/bin/env python3
"""Focused tests for #85 / RM-608 child progress snapshots."""
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


def _parser_args(*extra: str) -> argparse.Namespace:
    return multiblock.build_parser().parse_args(
        [
            "--npy-root",
            "/weights/npy",
            "--pack-root",
            "/weights/packs",
            "--embedding-shard",
            "/weights/embedding.npy",
            "--out",
            "/reports/v4",
            *extra,
        ]
    )


def _progress_base() -> dict:
    return {
        "arm": "int4_channel_alpha",
        "protocol": {
            "tokens": 2,
            "seed": 2026 * 10_000 + 806,
            "blocks": [0, 1],
            "top_k": 2,
        },
        "implementation": {"commit": "abc123", "dirty": False},
        "input_identity": {"embedding_shard": "/weights/embedding.npy"},
        "provenance_identity": {"issue": "GH #85 / Linear RM-608"},
    }


class ProgressParserTests(unittest.TestCase):
    def test_progress_json_is_optional_path(self) -> None:
        self.assertIsNone(_parser_args().progress_json)
        path = Path("run-progress.json")
        self.assertEqual(_parser_args("--progress-json", str(path)).progress_json, path)

    def test_parser_accepts_int4_channel_alpha_arm(self) -> None:
        self.assertEqual(
            _parser_args("--arm", "int4_channel_alpha").arm,
            "int4_channel_alpha",
        )

    def test_record_base_carries_protocol_and_identities(self) -> None:
        args = argparse.Namespace(
            arm="int4_channel_alpha",
            tokens=8192,
            seed=2026 * 10_000 + 806,
            top_k=2,
        )
        paths = multiblock.ChainPaths(
            npy_root=Path("/weights/npy"),
            npy_pattern="npy-{block}",
            pack_root=Path("/weights/packs"),
            pack_pattern="pack-{block}.goz1",
            embedding_shard=Path("/weights/embedding.npy"),
        )
        provenance = {
            "issue": "GH #85 / Linear RM-608",
            "agent": "agent",
            "model": "model",
            "design": "design",
            "architecture_source": "architecture",
            "activation_policy": "activation",
            "scale_policy": "scale",
            "implementation": {"commit": "abc123", "dirty": False},
        }

        record = multiblock._progress_record_base(args, [0, 1, 2, 3], paths, provenance)

        self.assertEqual(record["arm"], "int4_channel_alpha")
        self.assertEqual(
            record["protocol"],
            {
                "tokens": 8192,
                "seed": 2026 * 10_000 + 806,
                "blocks": [0, 1, 2, 3],
                "top_k": 2,
            },
        )
        self.assertEqual(record["implementation"], provenance["implementation"])
        self.assertEqual(record["input_identity"]["pack_pattern"], "pack-{block}.goz1")
        self.assertIsNone(record["input_identity"]["int4_side_root"])
        self.assertEqual(record["provenance_identity"]["issue"], provenance["issue"])


class ProgressRunChainTests(unittest.TestCase):
    @staticmethod
    def _run_chain(
        progress_path: Path | None,
        *,
        blocks: list[int] | None = None,
        expert_mode: str = "ternary",
        hp_blocks: set[int] | None = None,
    ):
        blocks = [0, 1] if blocks is None else blocks
        activations = np.ones((2, 4), dtype=np.float32)
        paths = multiblock.ChainPaths(
            npy_root=Path("/unused/npy"),
            npy_pattern="npy-{block}",
            pack_root=Path("/unused/packs"),
            pack_pattern="pack-{block}.goz1",
            embedding_shard=Path("/unused/embedding.npy"),
        )

        def fake_run_block(block, _paths, streams, _cfg):
            row = {"block": block, "pilot_label": f"fake-{block}"}
            return row, streams, {"block": block, "pack_sha256": str(block) * 64}

        real_writer = multiblock._atomic_write_json
        with (
            mock.patch.object(multiblock, "token_ids", return_value=np.array([4, 5])),
            mock.patch.object(multiblock, "_validate_embedding_shard"),
            mock.patch.object(multiblock, "embedding_rows", return_value=activations),
            mock.patch.object(multiblock, "_run_block", side_effect=fake_run_block),
            mock.patch.object(multiblock, "_atomic_write_json", wraps=real_writer) as writer,
        ):
            chain = multiblock.run_chain(
                blocks,
                paths,
                tokens=2,
                seed=2026 * 10_000 + 806,
                top_k=2,
                skip_fp16=False,
                expert_mode=expert_mode,
                hp_blocks=hp_blocks,
                progress_path=progress_path,
                progress_base=_progress_base(),
            )
        return chain, writer

    def test_writes_before_and_after_each_block(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            progress_path = Path(td) / "progress.json"
            chain, writer = self._run_chain(progress_path)
            final = json.loads(progress_path.read_text(encoding="utf-8"))

        snapshots = [call.args[1] for call in writer.call_args_list]
        self.assertEqual(
            [
                (item["current_block"], item["completed_blocks"])
                for item in snapshots
            ],
            [(0, []), (0, [0]), (1, [0]), (1, [0, 1])],
        )
        self.assertEqual(final["current_block"], 1)
        self.assertEqual(final["completed_blocks"], [0, 1])
        self.assertEqual(final["protocol"], _progress_base()["protocol"])
        self.assertEqual(final["implementation"], _progress_base()["implementation"])
        self.assertEqual(
            [len(item["pack_provenance"]) for item in snapshots],
            [0, 1, 1, 2],
        )
        self.assertEqual(
            [row["block"] for row in final["pack_provenance"]],
            [0, 1],
        )
        self.assertEqual([row["block"] for row in chain["per_block"]], [0, 1])

    def test_progress_distinguishes_p0_and_p1_schedules(self) -> None:
        cases = (
            (set(), "expert_int4_channel_alpha", []),
            ({1, 2, 3}, "expert_int4_channel_alpha_123", [1, 2, 3]),
        )
        for hp_blocks, arm_label, expected_hp in cases:
            with self.subTest(arm_label=arm_label), tempfile.TemporaryDirectory() as td:
                progress_path = Path(td) / "progress.json"
                self._run_chain(
                    progress_path,
                    blocks=[0, 1, 2, 3],
                    expert_mode="int4_channel_alpha",
                    hp_blocks=hp_blocks,
                )
                final = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(final["arm_label"], arm_label)
            self.assertEqual(final["hp_blocks"], expected_hp)

    def test_unset_progress_path_is_a_noop(self) -> None:
        _chain, writer = self._run_chain(None)
        writer.assert_not_called()


class AtomicProgressTests(unittest.TestCase):
    def test_interrupted_replace_keeps_previous_json_readable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "progress.json"
            multiblock._atomic_write_json(path, {"generation": 1})

            with mock.patch.object(
                multiblock.os,
                "replace",
                side_effect=OSError("simulated interruption"),
            ):
                with self.assertRaisesRegex(OSError, "simulated interruption"):
                    multiblock._atomic_write_json(path, {"generation": 2})

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"generation": 1},
            )
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
