#!/usr/bin/env python3
"""Unit tests for export_grok1_int8_npy (golden fixtures, no pickle in tests).

Fixtures under ``scripts/testdata/export_int8/`` are genuine numpy/JAX pickle
frames generated offline by ``dev_generate_int8_export_fixtures.py``. Tests only
copy those bytes — they never import pickle or unpickle untrusted data.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_grok1_int8_npy as exp  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "testdata" / "export_int8"


def _fixture(name: str) -> bytes:
    path = FIXTURE_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"missing fixture {path}; run scripts/dev_generate_int8_export_fixtures.py"
        )
    return path.read_bytes()


def _write_fixture(path: Path, name: str) -> None:
    path.write_bytes(_fixture(name))


def _bf16(x: np.ndarray) -> np.ndarray:
    """Round f32 to bfloat16 precision (bit-pattern view), for reference dequant."""
    return (x.astype("<f4").view(np.uint32) >> 16).astype("<u2")


class ScanShardTests(unittest.TestCase):
    def test_plain_f32(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s"
            _write_fixture(p, "plain_f32_4x6.bin")
            specs = exp.scan_shard(p)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].shape, (4, 6))
        self.assertEqual(specs[0].descr, "f4")

    def test_quantized_pair(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s"
            _write_fixture(p, "quant_8x8.bin")
            specs = exp.scan_shard(p)
        self.assertEqual([x.descr for x in specs], ["i1", "bfloat16"])
        self.assertEqual(specs[0].shape, (8, 8))
        self.assertEqual(specs[1].shape, (1, 8))

    def test_empty_file_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s"
            p.write_bytes(b"")
            with self.assertRaises(exp.ExportError):
                exp.scan_shard(p)

    def test_truncated_payload_rejected(self) -> None:
        raw = _fixture("f32_4096.bin")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s"
            p.write_bytes(raw[: len(raw) // 2])
            with self.assertRaises(exp.ExportError) as ctx:
                exp.scan_shard(p)
            self.assertIn("truncated", str(ctx.exception))

    def test_fortran_order_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s"
            _write_fixture(p, "fortran_64x64.bin")
            with self.assertRaises(exp.ExportError) as ctx:
                exp.scan_shard(p)
            self.assertIn("Fortran", str(ctx.exception))

    def test_large_payload_is_not_materialized(self) -> None:
        """Scanner reports payload offsets without reading bulk payload bytes."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s"
            _write_fixture(p, "large_zeros_16k_f32.bin")
            materialized: list[int] = []
            real_read = exp._StopAtPayload.read

            def spy(self, size=-1):
                data = real_read(self, size)
                materialized.append(len(data))
                return data

            with unittest.mock.patch.object(exp._StopAtPayload, "read", spy):
                specs = exp.scan_shard(p)
        self.assertEqual(specs[0].nbytes, (1 << 14) * 4)
        self.assertTrue(
            all(r < exp._PAYLOAD_READ_LIMIT for r in materialized),
            f"scanner materialized a bulk read: max={max(materialized) if materialized else None}",
        )


class GroupingTests(unittest.TestCase):
    def _spec(self, shape, descr):
        item = {"i1": 1, "bfloat16": 2, "f4": 4}[descr]
        n = 1
        for d in shape:
            n *= d
        return exp.ArraySpec(shape, descr, 0, n * item)

    def test_ungrouped(self) -> None:
        lead, k, n, g = exp.grouping(
            self._spec((6144, 1024), "i1"), self._spec((1, 1024), "bfloat16")
        )
        self.assertEqual((lead, k, n, g), ((), 6144, 1024, 1))

    def test_grouped_with_lead(self) -> None:
        lead, k, n, g = exp.grouping(
            self._spec((8, 32768, 6144), "i1"), self._spec((8, 8, 6144), "bfloat16")
        )
        self.assertEqual((lead, k, n, g), ((8,), 32768, 6144, 8))

    def test_indivisible_groups_rejected(self) -> None:
        with self.assertRaises(exp.ExportError):
            exp.grouping(self._spec((10, 4), "i1"), self._spec((3, 4), "bfloat16"))

    def test_output_dim_mismatch_rejected(self) -> None:
        with self.assertRaises(exp.ExportError):
            exp.grouping(self._spec((8, 4), "i1"), self._spec((1, 5), "bfloat16"))

    def test_lead_mismatch_rejected(self) -> None:
        with self.assertRaises(exp.ExportError):
            exp.grouping(
                self._spec((2, 8, 4), "i1"), self._spec((3, 1, 4), "bfloat16")
            )


class DequantTests(unittest.TestCase):
    """Grouped-scale rule must match numpy's own result on golden fixtures."""

    def _ref_from_arrays(self, w: np.ndarray, s_f32: np.ndarray) -> np.ndarray:
        lead, k, n = w.shape[:-2], w.shape[-2], w.shape[-1]
        g = s_f32.shape[-2]
        s_trunc = (_bf16(s_f32).astype(np.uint32) << 16).view(np.float32)
        return (
            w.reshape(*lead, g, k // g, n).astype(np.float32)
            * s_trunc.astype(np.float32)[..., :, None, :]
        ).reshape(w.shape)

    def _roundtrip_fixture(self, fixture: str, w: np.ndarray, s_f32: np.ndarray):
        ref = self._ref_from_arrays(w, s_f32)
        with tempfile.TemporaryDirectory() as td:
            shard, out = Path(td) / "s", Path(td) / "o.npy"
            _write_fixture(shard, fixture)
            info = exp.export_tensor(shard, out, chunk_mib=1)
            got = np.load(out)
        return ref, got, info

    def test_ungrouped_scales(self) -> None:
        rng = np.random.default_rng(7)
        w = rng.integers(-128, 128, size=(64, 32), dtype=np.int8)
        s = rng.random((1, 32), dtype=np.float32) + 0.25
        ref, got, info = self._roundtrip_fixture("dequant_ungrouped.bin", w, s)
        np.testing.assert_array_equal(ref, got)
        self.assertEqual(info["scale_groups"], 1)

    def test_grouped_scales(self) -> None:
        rng = np.random.default_rng(11)
        w = rng.integers(-128, 128, size=(64, 16), dtype=np.int8)
        s = rng.random((8, 16), dtype=np.float32) + 0.25
        ref, got, info = self._roundtrip_fixture("dequant_grouped.bin", w, s)
        np.testing.assert_array_equal(ref, got)
        self.assertEqual((info["scale_groups"], info["group_rows"]), (8, 8))

    def test_leading_expert_axis(self) -> None:
        rng = np.random.default_rng(13)
        w = rng.integers(-128, 128, size=(3, 32, 8), dtype=np.int8)
        s = rng.random((3, 4, 8), dtype=np.float32) + 0.25
        ref, got, _ = self._roundtrip_fixture("dequant_lead.bin", w, s)
        np.testing.assert_array_equal(ref, got)

    def test_chunking_does_not_change_result(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            shard = Path(td) / "s"
            _write_fixture(shard, "dequant_chunk.bin")
            outs = []
            for mib in (1, 64):
                out = Path(td) / f"o{mib}.npy"
                exp.export_tensor(shard, out, chunk_mib=mib)
                outs.append(np.load(out))
        np.testing.assert_array_equal(outs[0], outs[1])

    def test_f32_passthrough_is_bit_exact(self) -> None:
        a = (np.arange(120, dtype="<f4") / 7.0).reshape(10, 12)
        with tempfile.TemporaryDirectory() as td:
            shard, out = Path(td) / "s", Path(td) / "o.npy"
            _write_fixture(shard, "f32_passthrough.bin")
            info = exp.export_tensor(shard, out)
            got = np.load(out)
        np.testing.assert_array_equal(a, got)
        self.assertIsNone(info["scale_groups"])

    def test_shape_mismatch_is_caught_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            shard, out = Path(td) / "s", Path(td) / "o.npy"
            _write_fixture(shard, "zeros_4x4.bin")
            with self.assertRaises(exp.ExportError) as ctx:
                exp.export_tensor(shard, out, expect_shape=(4, 5))
            self.assertIn("refusing to write", str(ctx.exception))
            # Assert before TemporaryDirectory cleanup (Grok review).
            self.assertFalse(out.exists())

    def test_matching_expect_shape_writes(self) -> None:
        a = np.arange(16, dtype="<f4").reshape(4, 4)
        with tempfile.TemporaryDirectory() as td:
            shard, out = Path(td) / "s", Path(td) / "o.npy"
            _write_fixture(shard, "arange_4x4.bin")
            exp.export_tensor(shard, out, expect_shape=(4, 4))
            np.testing.assert_array_equal(np.load(out), a)

    def test_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            shard, out = Path(td) / "s", Path(td) / "o.npy"
            _write_fixture(shard, "zeros_4x4.bin")
            info = exp.export_tensor(shard, out, dry_run=True)
            self.assertFalse(out.exists())
        self.assertEqual(info["out_bytes"], 64)

    def test_nonpositive_chunk_mib_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            shard, out = Path(td) / "s", Path(td) / "o.npy"
            _write_fixture(shard, "zeros_4x4.bin")
            with self.assertRaises(exp.ExportError) as ctx:
                exp.export_tensor(shard, out, chunk_mib=0)
            self.assertIn("chunk-mib", str(ctx.exception).lower())

    def test_negative_chunk_mib_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            shard, out = Path(td) / "s", Path(td) / "o.npy"
            _write_fixture(shard, "zeros_4x4.bin")
            with self.assertRaises(exp.ExportError) as ctx:
                exp.export_tensor(shard, out, chunk_mib=-1)
            self.assertIn("chunk-mib", str(ctx.exception).lower())


class NpyHeaderTests(unittest.TestCase):
    def test_payload_is_64_byte_aligned(self) -> None:
        for shape in [(1,), (6144,), (6144, 6144), (8, 6144, 32768)]:
            h = exp.npy_header(shape)
            self.assertEqual(len(h) % 64, 0, f"header for {shape} is {len(h)} bytes")
            self.assertTrue(h.startswith(b"\x93NUMPY"))

    def test_numpy_can_read_generated_header(self) -> None:
        a = np.arange(12, dtype="<f4").reshape(3, 4)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "o.npy"
            p.write_bytes(exp.npy_header((3, 4)) + a.tobytes())
            np.testing.assert_array_equal(np.load(p), a)


class SelectionTests(unittest.TestCase):
    TENSORS = [
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

    def test_no_mode_selects_the_deferred_embedding(self) -> None:
        for mode in exp.MODES:
            got = {
                t["structural_name"]
                for t in exp.select_tensors(self.TENSORS, block=0, mode=mode, names=[])
            }
            self.assertNotIn("embedding.slot_00.token_embedding", got)

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


if __name__ == "__main__":
    unittest.main()
