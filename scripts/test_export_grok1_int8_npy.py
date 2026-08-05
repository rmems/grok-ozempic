"""Unit tests for export_grok1_int8_npy.py (synthetic pickles, no checkpoint needed).

Fixtures are built with ``pickle.dumps`` over real numpy arrays so the opcode
scanner is exercised against genuine ``numpy.core.multiarray._reconstruct``
frames -- the same shape the official Grok-1 ckpt-0 shards have.
"""

from __future__ import annotations

# Fixtures are *written* with pickle.dumps; nothing here loads untrusted pickles.
import pickle  # nosec B403
import pickletools
import sys
import tempfile
import unittest
import unittest.mock
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_grok1_int8_npy as exp


@dataclass
class QuantizedWeight8bit:
    """Mirrors ``__main__.QuantizedWeight8bit`` in the official checkpoint."""

    weight: object
    scales: object


def _bf16(x: np.ndarray) -> np.ndarray:
    """Round f32 to bfloat16 precision, kept as a raw uint16 view carrier.

    ``ml_dtypes`` is not a test dependency, so fixtures store bf16 values as the
    truncated f32 bit pattern; the exporter reads scales as ``<u2`` regardless of
    which library wrote them.
    """
    return (x.astype("<f4").view(np.uint32) >> 16).astype("<u2")


def _write_shard(path: Path, obj) -> None:
    path.write_bytes(pickle.dumps(obj, protocol=4))  # nosemgrep: python.lang.security.deserialization.pickle.avoid-pickle


def _write_quantized(path: Path, weight: np.ndarray, scales_f32: np.ndarray) -> None:
    """Emit a shard whose scales pickle carries the bfloat16 descriptor.

    numpy cannot name a dtype ``bfloat16`` without ``ml_dtypes``, so the fixture
    is patched at the byte level: the scales array is pickled as ``<u2`` and the
    descriptor string is rewritten to ``bfloat16`` (same 2-byte itemsize, same
    payload), reproducing what JAX/ml_dtypes writes.
    """
    raw = pickle.dumps(  # nosemgrep: python.lang.security.deserialization.pickle.avoid-pickle
        QuantizedWeight8bit(weight=weight, scales=_bf16(scales_f32)), protocol=4
    )
    # Only the scales array is u2 in these fixtures, so a single rename is exact.
    old, new = b"\x8c\x02u2\x94", b"\x8c\x08bfloat16\x94"
    if raw.count(old) != 1:
        raise ValueError(
            f"fixture should contain exactly one u2 dtype, found {raw.count(old)}"
        )
    path.write_bytes(raw.replace(old, new))


class ScanShardTests(unittest.TestCase):
    def test_plain_f32(self) -> None:
        a = np.arange(24, dtype="<f4").reshape(4, 6)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s"
            _write_shard(p, a)
            specs = exp.scan_shard(p)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].shape, (4, 6))
        self.assertEqual(specs[0].descr, "f4")

    def test_quantized_pair(self) -> None:
        w = np.arange(-32, 32, dtype=np.int8).reshape(8, 8)
        s = np.linspace(0.5, 1.5, 8, dtype="<f4").reshape(1, 8)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s"
            _write_quantized(p, w, s)
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
        """A shard cut short mid-payload must fail loudly, not export garbage."""
        a = np.arange(4096, dtype="<f4")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s"
            raw = pickle.dumps(a, protocol=4)  # nosemgrep: python.lang.security.deserialization.pickle.avoid-pickle
            p.write_bytes(raw[: len(raw) // 2])
            with self.assertRaises(exp.ExportError) as ctx:
                exp.scan_shard(p)
            self.assertIn("truncated", str(ctx.exception))

    def test_fortran_order_rejected(self) -> None:
        """C-order is assumed by the writer, so Fortran order must not silently pass."""
        a = np.asfortranarray(np.arange(4096, dtype="<f4").reshape(64, 64))
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s"
            _write_shard(p, a)
            with self.assertRaises(exp.ExportError) as ctx:
                exp.scan_shard(p)
            self.assertIn("Fortran", str(ctx.exception))

    def test_memoized_dtype_reused_across_arrays(self) -> None:
        """A dtype object memoized by the pickler (BINGET) must not lose the descriptor."""
        a = np.arange(24, dtype="<f4").reshape(4, 6)
        b = np.arange(48, dtype="<f4").reshape(6, 8)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s"
            _write_shard(p, (a, b))
            specs = exp.scan_shard(p)
        self.assertEqual(len(specs), 2)
        self.assertEqual([s.descr for s in specs], ["f4", "f4"])
        self.assertEqual([s.shape for s in specs], [(4, 6), (6, 8)])

    def test_large_payload_is_not_materialized(self) -> None:
        """The scanner reports payload offsets without reading the payload."""
        a = np.zeros(1 << 20, dtype="<f4")  # 4 MiB, far above _PAYLOAD_READ_LIMIT
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s"
            _write_shard(p, a)
            materialized: list[int] = []
            real_read = exp._StopAtPayload.read

            def spy(self, size=-1):
                # Large reads raise _PayloadBoundary, so nothing is appended for them.
                data = real_read(self, size)
                materialized.append(len(data))
                return data

            with unittest.mock.patch.object(exp._StopAtPayload, "read", spy):
                specs = exp.scan_shard(p)
        self.assertEqual(specs[0].nbytes, a.nbytes)
        self.assertTrue(
            all(r < exp._PAYLOAD_READ_LIMIT for r in materialized),
            f"scanner materialized a bulk read: max={max(materialized)}",
        )

    def test_pop_between_arrays_does_not_corrupt_dtype(self) -> None:
        """POP removes a single stack value, not every value to the nearest MARK."""
        a = np.arange(6, dtype="<f4").reshape(2, 3)
        b = np.arange(12, dtype="<f4").reshape(3, 4)
        raw = bytearray(pickle.dumps((a, b), protocol=4))
        for op, _, pos in pickletools.genops(raw):
            if op.name == "BUILD":
                insert = pos + 1
                break
        else:
            raise AssertionError("no BUILD opcode in fixture")
        # Push and immediately pop a None so the stack is unchanged for TUPLE2.
        raw[insert:insert] = b"N0"  # NONE, POP
        raw[3:11] = (len(raw) - 11).to_bytes(8, "little")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s"
            p.write_bytes(raw)
            specs = exp.scan_shard(p)
        self.assertEqual(len(specs), 2)
        self.assertEqual([s.shape for s in specs], [(2, 3), (3, 4)])
        self.assertEqual([s.descr for s in specs], ["f4", "f4"])


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
    """The reshape-and-broadcast rule must match numpy's own result exactly."""

    def _roundtrip(self, w: np.ndarray, s_f32: np.ndarray) -> tuple:
        lead, k, n = w.shape[:-2], w.shape[-2], w.shape[-1]
        g = s_f32.shape[-2]
        # bfloat16 truncation is what the checkpoint stores, so the reference
        # must use the truncated values too.
        s_trunc = (_bf16(s_f32).astype(np.uint32) << 16).view(np.float32)
        ref = (
            w.reshape(*lead, g, k // g, n).astype(np.float32)
            * s_trunc.astype(np.float32)[..., :, None, :]
        ).reshape(w.shape)
        with tempfile.TemporaryDirectory() as td:
            shard, out = Path(td) / "s", Path(td) / "o.npy"
            _write_quantized(shard, w, s_f32)
            info = exp.export_tensor(shard, out, chunk_mib=1)
            got = np.load(out)
        return ref, got, info

    def test_ungrouped_scales(self) -> None:
        rng = np.random.default_rng(7)
        w = rng.integers(-128, 128, size=(64, 32), dtype=np.int8)
        s = rng.random((1, 32), dtype=np.float32) + 0.25
        ref, got, info = self._roundtrip(w, s)
        np.testing.assert_array_equal(ref, got)
        self.assertEqual(info["scale_groups"], 1)

    def test_grouped_scales(self) -> None:
        rng = np.random.default_rng(11)
        w = rng.integers(-128, 128, size=(64, 16), dtype=np.int8)
        s = rng.random((8, 16), dtype=np.float32) + 0.25
        ref, got, info = self._roundtrip(w, s)
        np.testing.assert_array_equal(ref, got)
        self.assertEqual((info["scale_groups"], info["group_rows"]), (8, 8))

    def test_leading_expert_axis(self) -> None:
        rng = np.random.default_rng(13)
        w = rng.integers(-128, 128, size=(3, 32, 8), dtype=np.int8)
        s = rng.random((3, 4, 8), dtype=np.float32) + 0.25
        ref, got, _ = self._roundtrip(w, s)
        np.testing.assert_array_equal(ref, got)

    def test_chunking_does_not_change_result(self) -> None:
        rng = np.random.default_rng(17)
        w = rng.integers(-128, 128, size=(128, 64), dtype=np.int8)
        s = rng.random((8, 64), dtype=np.float32) + 0.25
        with tempfile.TemporaryDirectory() as td:
            shard = Path(td) / "s"
            _write_quantized(shard, w, s)
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
            _write_shard(shard, a)
            info = exp.export_tensor(shard, out)
            got = np.load(out)
        np.testing.assert_array_equal(a, got)
        self.assertIsNone(info["scale_groups"])

    def test_shape_mismatch_is_caught_before_writing(self) -> None:
        """A manifest/shard disagreement must not leave a wrong file behind."""
        a = np.zeros((4, 4), dtype="<f4")
        with tempfile.TemporaryDirectory() as td:
            shard, out = Path(td) / "s", Path(td) / "o.npy"
            _write_shard(shard, a)
            with self.assertRaises(exp.ExportError) as ctx:
                exp.export_tensor(shard, out, expect_shape=(4, 5))
            self.assertIn("refusing to write", str(ctx.exception))
            self.assertFalse(out.exists())
            self.assertFalse(out.with_suffix(out.suffix + ".partial").exists())

    def test_matching_expect_shape_writes(self) -> None:
        a = np.arange(16, dtype="<f4").reshape(4, 4)
        with tempfile.TemporaryDirectory() as td:
            shard, out = Path(td) / "s", Path(td) / "o.npy"
            _write_shard(shard, a)
            exp.export_tensor(shard, out, expect_shape=(4, 4))
            np.testing.assert_array_equal(np.load(out), a)

    def test_dry_run_writes_nothing(self) -> None:
        a = np.zeros((4, 4), dtype="<f4")
        with tempfile.TemporaryDirectory() as td:
            shard, out = Path(td) / "s", Path(td) / "o.npy"
            _write_shard(shard, a)
            info = exp.export_tensor(shard, out, dry_run=True)
            self.assertFalse(out.exists())
            self.assertFalse(out.with_suffix(out.suffix + ".partial").exists())
        self.assertEqual(info["out_bytes"], 64)

    def test_nonpositive_chunk_mib_rejected(self) -> None:
        a = np.zeros((4, 4), dtype="<f4")
        with tempfile.TemporaryDirectory() as td:
            shard, out = Path(td) / "s", Path(td) / "o.npy"
            _write_shard(shard, a)
            with self.assertRaises(exp.ExportError) as ctx:
                exp.export_tensor(shard, out, chunk_mib=0)
            self.assertIn("chunk-mib", str(ctx.exception).lower())

    def test_negative_chunk_mib_rejected(self) -> None:
        a = np.zeros((4, 4), dtype="<f4")
        with tempfile.TemporaryDirectory() as td:
            shard, out = Path(td) / "s", Path(td) / "o.npy"
            _write_shard(shard, a)
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


class HeaderStateTests(unittest.TestCase):
    def test_pop_removes_only_top_value(self) -> None:
        state = exp._HeaderState()
        state.feed("BININT1", 1)
        state.feed("BININT1", 2)
        state.feed("POP", None)
        self.assertEqual(state.stack, [1])

    def test_reduce_dtype_tuple_with_flags(self) -> None:
        """A numpy dtype REDUCE may carry optional flag fields after the descriptor."""
        state = exp._HeaderState()
        for op, arg in (
            ("SHORT_BINUNICODE", "numpy"),
            ("SHORT_BINUNICODE", "dtype"),
            ("STACK_GLOBAL", None),
            ("SHORT_BINUNICODE", "f4"),
            ("NEWFALSE", None),
            ("NEWTRUE", None),
            ("TUPLE3", None),
            ("REDUCE", None),
        ):
            state.feed(op, arg)
        self.assertEqual(state.descr, "f4")

    def test_newobj_ex_pops_kwargs_args_cls(self) -> None:
        state = exp._HeaderState()
        for op, arg in (
            ("NONE", None),
            ("EMPTY_TUPLE", None),
            ("EMPTY_DICT", None),
            ("NEWOBJ_EX", None),
        ):
            state.feed(op, arg)
        self.assertEqual(len(state.stack), 1)
        self.assertIsInstance(state.stack[0], exp._Unknown)


if __name__ == "__main__":
    unittest.main()
