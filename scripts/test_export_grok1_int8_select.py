"""Unit tests for manifest selection + output-path safety (GH #53 / RM-222).

Split out of ``test_export_grok1_int8_npy.py``: these exercise
``export_grok1_int8_select`` (mode selection, structural-stem safety, manifest
entry validation), which is a separate module from the pickle scanner.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_grok1_int8_npy as exp  # noqa: E402
from testdata.export_int8 import shard_factory as _sf  # noqa: E402

_write_shard = _sf.write_shard


class SelectionTests(unittest.TestCase):
    TENSORS: ClassVar[list[dict]] = [
        {"structural_name": "block_000.slot_00.moe_expert.gate", "block": 0, "kind": "moe_expert.gate"},
        {"structural_name": "block_000.slot_04.attn_proj_i8.model_width", "block": 0, "kind": "attn_proj_i8.model_width"},
        {"structural_name": "block_000.slot_07.block_norm", "block": 0, "kind": "block_norm"},
        {"structural_name": "block_000.slot_11.router", "block": 0, "kind": "router"},
        {"structural_name": "block_001.slot_11.router", "block": 1, "kind": "router"},
        {"structural_name": "embedding.slot_00.token_embedding", "block": None, "kind": "token_embedding"},
    ]

    def test_attention_only_excludes_experts(self) -> None:
        got = {
            t["kind"]
            for t in exp.select_tensors(self.TENSORS, block=0, mode="attention_only", names=[])
        }
        self.assertIn("attn_proj_i8.model_width", got)
        self.assertNotIn("moe_expert.gate", got)

    def test_every_mode_keeps_the_preserve_tier(self) -> None:
        for mode in ("attention_only", "expert_only", "attention_plus_expert"):
            got = {
                t["kind"]
                for t in exp.select_tensors(self.TENSORS, block=0, mode=mode, names=[])
            }
            self.assertIn("router", got, mode)
            self.assertIn("block_norm", got, mode)

    def test_token_embedding_is_in_no_mode(self) -> None:
        """The deferred tier must not appear in any mode's kind set.

        Asserted directly on ``MODES`` so it holds independently of how the
        selector filters — the block filter would otherwise mask a regression
        here, since the embedding carries ``block: None``.
        """
        for mode, kinds in exp.MODES.items():
            self.assertNotIn("token_embedding", kinds, mode)

    def test_selecting_the_embedding_block_is_rejected(self) -> None:
        """`block=None` makes the embedding the only block-matching candidate.

        So this exercises the *kind* filter rather than the block filter: if
        `token_embedding` were ever admitted to a mode, the call would succeed
        instead of raising, and the deferred tier would be silently packable.
        """
        block_none = [t for t in self.TENSORS if t["block"] is None]
        self.assertEqual(
            [t["structural_name"] for t in block_none],
            ["embedding.slot_00.token_embedding"],
            "fixture precondition: the embedding is the sole block=None tensor",
        )
        for mode in exp.MODES:
            with self.assertRaises(exp.ExportError, msg=mode) as ctx:
                exp.select_tensors(self.TENSORS, block=None, mode=mode, names=[])
            self.assertIn("no tensors for block None", str(ctx.exception))

    def test_block_filter_is_exact(self) -> None:
        got = exp.select_tensors(self.TENSORS, block=1, mode="preserve_only", names=[])
        self.assertEqual([t["structural_name"] for t in got], ["block_001.slot_11.router"])

    def test_unknown_structural_name_rejected(self) -> None:
        with self.assertRaises(exp.ExportError):
            exp.select_tensors(self.TENSORS, block=None, mode="attention_only", names=["nope"])

    def test_empty_selection_rejected(self) -> None:
        with self.assertRaises(exp.ExportError):
            exp.select_tensors(self.TENSORS, block=42, mode="attention_only", names=[])


class StemTests(unittest.TestCase):
    def test_structural_stem_round_trip(self) -> None:
        name = "block_000.slot_04.attn_proj_i8.model_width"
        stem = exp.structural_stem(name)
        self.assertEqual(stem, "block_000__slot_04__attn_proj_i8__model_width")
        # Mirrors npy_stem_to_tensor_name on the Rust side.
        self.assertEqual(stem.replace("__", "."), name)

    def test_path_separator_rejected(self) -> None:
        for bad in ("../foo.bar", "foo/bar", "foo\\bar"):
            with self.assertRaises(exp.ExportError) as ctx:
                exp.structural_stem(bad)
            self.assertIn("separator", str(ctx.exception).lower())

    def test_parent_reference_rejected(self) -> None:
        for bad in ("foo...bar", "foo..bar", ".foo.bar", "foo.bar."):
            with self.assertRaises(exp.ExportError) as ctx:
                exp.structural_stem(bad)
            self.assertIn("parent-reference", str(ctx.exception).lower())

    def test_drive_relative_rejected(self) -> None:
        for bad in ("C:foo.bar", "D:foo", "foo:Dbar"):
            with self.assertRaises(exp.ExportError) as ctx:
                exp.structural_stem(bad)
            self.assertIn("colon", str(ctx.exception).lower())


class ManifestValidationTests(unittest.TestCase):
    def _manifest(self, tensors):
        return json.dumps({"tensors": tensors}).encode("utf-8")

    def _entry(self, **overrides):
        """A well-formed entry, so each test isolates the one field it breaks."""
        base = {
            "structural_name": "block_000.slot_00.moe_expert.gate",
            "source_shard_path": "s",
            "shape": [1],
            "kind": "moe_expert.gate",
            "block": 0,
        }
        base.update(overrides)
        return base

    def test_well_formed_entry_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.json"
            p.write_bytes(self._manifest([self._entry()]))
            self.assertEqual(len(exp.load_manifest(p)), 1)

    def test_null_block_accepted_for_model_level_tensor(self) -> None:
        """The token embedding legitimately carries `block: null`."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.json"
            p.write_bytes(
                self._manifest(
                    [self._entry(structural_name="embedding.slot_00.token_embedding",
                                 kind="token_embedding", block=None)]
                )
            )
            self.assertEqual(len(exp.load_manifest(p)), 1)

    def test_missing_kind_rejected(self) -> None:
        entry = self._entry()
        del entry["kind"]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.json"
            p.write_bytes(self._manifest([entry]))
            with self.assertRaises(exp.ExportError) as ctx:
                exp.load_manifest(p)
            self.assertIn("kind", str(ctx.exception).lower())

    def test_non_string_kind_rejected(self) -> None:
        """A numeric kind matches no mode, so selection would silently under-pick."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.json"
            p.write_bytes(self._manifest([self._entry(kind=7)]))
            with self.assertRaises(exp.ExportError) as ctx:
                exp.load_manifest(p)
            self.assertIn("kind", str(ctx.exception).lower())

    def test_missing_block_rejected(self) -> None:
        entry = self._entry()
        del entry["block"]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.json"
            p.write_bytes(self._manifest([entry]))
            with self.assertRaises(exp.ExportError) as ctx:
                exp.load_manifest(p)
            self.assertIn("block", str(ctx.exception).lower())

    def test_non_int_block_rejected(self) -> None:
        """A stringy block never equals --block, so the selector would under-pick."""
        for bad in ("0", 1.5, True):
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "m.json"
                p.write_bytes(self._manifest([self._entry(block=bad)]))
                with self.assertRaises(exp.ExportError, msg=repr(bad)) as ctx:
                    exp.load_manifest(p)
                self.assertIn("block", str(ctx.exception).lower())

    def test_load_manifest_rejects_non_object_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.json"
            p.write_text("[]")
            with self.assertRaises(exp.ExportError) as ctx:
                exp.load_manifest(p)
            self.assertIn("root", str(ctx.exception).lower())

    def test_load_manifest_rejects_missing_required_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.json"
            p.write_bytes(self._manifest([{"structural_name": "a.b"}]))
            with self.assertRaises(exp.ExportError) as ctx:
                exp.load_manifest(p)
            self.assertIn("missing required key", str(ctx.exception).lower())

    def test_load_manifest_rejects_non_string_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.json"
            p.write_bytes(
                self._manifest([self._entry(structural_name=123)])
            )
            with self.assertRaises(exp.ExportError) as ctx:
                exp.load_manifest(p)
            self.assertIn("structural_name", str(ctx.exception).lower())

    def test_load_manifest_rejects_non_int_shape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.json"
            p.write_bytes(
                self._manifest([self._entry(shape=["x"])])
            )
            with self.assertRaises(exp.ExportError) as ctx:
                exp.load_manifest(p)
            self.assertIn("shape", str(ctx.exception).lower())

    def test_load_manifest_rejects_duplicate_structural_names(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.json"
            p.write_bytes(
                self._manifest(
                    [
                        self._entry(source_shard_path="s1"),
                        self._entry(source_shard_path="s2"),
                    ]
                )
            )
            with self.assertRaises(exp.ExportError) as ctx:
                exp.load_manifest(p)
            self.assertIn("duplicate", str(ctx.exception).lower())


class RunExportsTests(unittest.TestCase):
    def test_duplicate_selected_stem_is_rejected(self) -> None:
        """Two different names mapping to the same stem must not overwrite."""
        a = np.arange(4, dtype="<f4")
        b = np.arange(4, dtype="<f4")
        with tempfile.TemporaryDirectory() as td:
            shard_a, shard_b = Path(td) / "sa", Path(td) / "sb"
            _write_shard(shard_a, a)
            _write_shard(shard_b, b)
            # structural_stem rejects '__', but a literal duplicate name would.
            picked = [
                {
                    "structural_name": "block_000.slot_00.foo",
                    "source_shard_path": str(shard_a),
                    "shape": [4],
                },
                {
                    "structural_name": "block_000.slot_00.foo",
                    "source_shard_path": str(shard_b),
                    "shape": [4],
                },
            ]
            with self.assertRaises(exp.ExportError) as ctx:
                exp.run_exports(picked, Path(td), chunk_mib=1, dry_run=False)
            self.assertIn("duplicate output stem", str(ctx.exception).lower())


class BlockNameConsistencyTests(unittest.TestCase):
    """`block` must agree with the structural name.

    Selection matches on `block == --block`, so a mismatch is not a loud failure
    — the tensor is simply never picked and the export comes out short.
    """

    def _manifest(self, **overrides):
        base = {
            "structural_name": "block_000.slot_00.moe_expert.gate",
            "source_shard_path": "s",
            "shape": [1],
            "kind": "moe_expert.gate",
            "block": 0,
        }
        base.update(overrides)
        return json.dumps({"tensors": [base]}).encode("utf-8")

    def _load(self, **overrides):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.json"
            p.write_bytes(self._manifest(**overrides))
            return exp.load_manifest(p)

    def test_block_scoped_name_with_null_block_rejected(self) -> None:
        with self.assertRaises(exp.ExportError) as ctx:
            self._load(block=None)
        self.assertIn("silently skipped", str(ctx.exception))

    def test_block_number_must_match_the_name(self) -> None:
        with self.assertRaises(exp.ExportError) as ctx:
            self._load(block=7)
        self.assertIn("names block 0 but block is 7", str(ctx.exception))

    def test_negative_block_rejected(self) -> None:
        with self.assertRaises(exp.ExportError) as ctx:
            self._load(structural_name="block_001.slot_00.moe_expert.gate", block=-1)
        self.assertIn("expected >= 0", str(ctx.exception))

    def test_model_level_name_with_a_block_number_rejected(self) -> None:
        with self.assertRaises(exp.ExportError) as ctx:
            self._load(
                structural_name="embedding.slot_00.token_embedding",
                kind="token_embedding",
                block=0,
            )
        self.assertIn("not block-scoped", str(ctx.exception))

    def test_matching_pairs_accepted(self) -> None:
        self.assertEqual(len(self._load()), 1)
        self.assertEqual(
            len(
                self._load(
                    structural_name="embedding.slot_00.token_embedding",
                    kind="token_embedding",
                    block=None,
                )
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
