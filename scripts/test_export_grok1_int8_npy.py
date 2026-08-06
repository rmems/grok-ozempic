"""Unit tests for export_grok1_int8_npy.py (synthetic pickles, no checkpoint needed).

Fixtures are built via ``testdata.export_int8.shard_factory`` over real numpy arrays so the opcode
scanner is exercised against genuine ``numpy.core.multiarray._reconstruct``
frames -- the same shape the official Grok-1 ckpt-0 shards have.
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
from testdata.export_int8 import shard_factory as _sf  # noqa: E402

_bf16 = _sf.bf16
_write_shard = _sf.write_shard
_write_quantized = _sf.write_quantized


def _array_spec(shape: tuple[int, ...], descr: str) -> "exp.ArraySpec":
    """An ArraySpec with a self-consistent nbytes for `shape`/`descr`.

    Shared by the classes below. Argument order matches ArraySpec's own fields;
    the two class-local copies this replaces took (shape, descr) and
    (descr, shape) respectively, which was a standing footgun.
    """
    itemsize = {"i1": 1, "bfloat16": 2, "f4": 4}[descr]
    numel = 1
    for d in shape:
        numel *= d
    return exp.ArraySpec(shape, descr, 0, numel * itemsize)


class SplitQuantizedTests(unittest.TestCase):
    """`split_quantized` is the last gate before a pack input is accepted."""

    def test_int8_then_bf16_accepted(self) -> None:
        w, s = exp.split_quantized([_array_spec((4, 4), "i1"), _array_spec((1, 4), "bfloat16")])
        self.assertEqual((w.descr, s.descr), ("i1", "bfloat16"))

    def test_lone_f32_is_passthrough(self) -> None:
        w, s = exp.split_quantized([_array_spec((4, 4), "f4")])
        self.assertIsNone(s)

    def test_lone_int8_rejected(self) -> None:
        """A quantized weight with no scales cannot be dequantized."""
        with self.assertRaises(exp.ExportError):
            exp.split_quantized([_array_spec((4, 4), "i1")])

    def test_reversed_order_rejected(self) -> None:
        """bf16-then-int8 would treat the scales as the weight."""
        with self.assertRaises(exp.ExportError):
            exp.split_quantized([_array_spec((1, 4), "bfloat16"), _array_spec((4, 4), "i1")])

    def test_three_arrays_rejected(self) -> None:
        with self.assertRaises(exp.ExportError):
            exp.split_quantized(
                [_array_spec((4, 4), "i1"), _array_spec((1, 4), "bfloat16"), _array_spec((4, 4), "f4")]
            )

    def test_f32_pair_rejected(self) -> None:
        with self.assertRaises(exp.ExportError):
            exp.split_quantized([_array_spec((4, 4), "f4"), _array_spec((4, 4), "f4")])

    def test_empty_rejected(self) -> None:
        with self.assertRaises(exp.ExportError):
            exp.split_quantized([])


class StackGlobalDtypeTests(unittest.TestCase):
    """The real ckpt-0 names bfloat16 via ``STACK_GLOBAL ml_dtypes bfloat16``.

    ``write_quantized`` rewrites the descriptor into a plain string, so it does
    not exercise that framing. Without these, a scanner that stringifies the
    global into a junk descriptor passes the whole suite while every real-shard
    export fails.
    """

    def _shard(self, td: str):
        rng = np.random.default_rng(3)
        w = rng.integers(-128, 128, size=(64, 32), dtype=np.int8)
        s = rng.random((1, 32), dtype=np.float32) + 0.25
        p = Path(td) / "s"
        _sf.write_quantized_global_dtype(p, w, s)
        return p, w, s

    def test_descriptor_recovered_from_stack_global(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p, w, _s = self._shard(td)
            specs = exp.scan_shard(p)
        self.assertEqual([sp.descr for sp in specs], ["i1", "bfloat16"])
        self.assertEqual(specs[0].shape, w.shape)

    def test_untrusted_global_module_rejected(self) -> None:
        """The element size decides what gets exported, so only known globals count."""
        rng = np.random.default_rng(3)
        w = rng.integers(-128, 128, size=(64, 32), dtype=np.int8)
        s = rng.random((1, 32), dtype=np.float32) + 0.25
        for module, name in (("evil_pkg", "bfloat16"), ("ml_dtypes", "float8_e4m3")):
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "s"
                _sf.write_quantized_global_dtype(p, w, s, module=module, name=name)
                with self.assertRaises(exp.ExportError, msg=f"{module}.{name}"):
                    exp.scan_shard(p)

    def test_bare_bfloat16_string_descriptor_rejected(self) -> None:
        """`bfloat16` is an ml_dtypes global name, not a numpy dtype spelling.

        Accepting it as a loose string would let any occurrence of that word in
        the stream set a 2-byte element size.
        """
        rng = np.random.default_rng(3)
        w = rng.integers(-128, 128, size=(64, 32), dtype=np.int8)
        s = rng.random((1, 32), dtype=np.float32) + 0.25
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s"
            _sf.write_quantized_string_dtype(p, w, s)
            with self.assertRaises(exp.ExportError):
                exp.scan_shard(p)

    def test_native_string_descriptors_still_accepted(self) -> None:
        """numpy's own spellings must keep working as bare strings."""
        a = np.arange(24, dtype="<f4").reshape(4, 6)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s"
            _sf.write_shard(p, a)
            self.assertEqual([sp.descr for sp in exp.scan_shard(p)], ["f4"])


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
            raw = _sf.dumps(a, protocol=4)  # nosemgrep: python.lang.security.deserialization.pickle.avoid-pickle
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
        raw = bytearray(_sf.dumps((a, b), protocol=4))
        for op, _, pos in _sf.genops(raw):
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
    def test_ungrouped(self) -> None:
        lead, k, n, g = exp.grouping(
            _array_spec((6144, 1024), "i1"), _array_spec((1, 1024), "bfloat16")
        )
        self.assertEqual((lead, k, n, g), ((), 6144, 1024, 1))

    def test_grouped_with_lead(self) -> None:
        lead, k, n, g = exp.grouping(
            _array_spec((8, 32768, 6144), "i1"), _array_spec((8, 8, 6144), "bfloat16")
        )
        self.assertEqual((lead, k, n, g), ((8,), 32768, 6144, 8))

    def test_indivisible_groups_rejected(self) -> None:
        with self.assertRaises(exp.ExportError):
            exp.grouping(_array_spec((10, 4), "i1"), _array_spec((3, 4), "bfloat16"))

    def test_output_dim_mismatch_rejected(self) -> None:
        with self.assertRaises(exp.ExportError):
            exp.grouping(_array_spec((8, 4), "i1"), _array_spec((1, 5), "bfloat16"))

    def test_lead_mismatch_rejected(self) -> None:
        with self.assertRaises(exp.ExportError):
            exp.grouping(
                _array_spec((2, 8, 4), "i1"), _array_spec((3, 1, 4), "bfloat16")
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


class HeaderStateCloneTests(unittest.TestCase):
    """`snapshot`/`restore` must survive shared and self-referential containers.

    A pickle memo can reference one container from several places, and a hostile
    shard can encode a self-referential one. Naive recursion would either break
    the sharing or blow the stack with an uncaught RecursionError.
    """

    def _state(self):
        import export_grok1_int8_scan as scan

        return scan._HeaderState()

    def test_shared_container_stays_shared(self) -> None:
        st = self._state()
        shared: list = [1, 2]
        st.stack = [shared, shared]
        snap = st.snapshot()
        st.restore(snap)
        self.assertEqual(st.stack[0], [1, 2])
        self.assertIs(st.stack[0], st.stack[1], "sharing must be preserved")

    def test_self_referential_container_does_not_recurse_forever(self) -> None:
        st = self._state()
        cyclic: list = [1]
        cyclic.append(cyclic)
        st.memo = {0: cyclic}
        snap = st.snapshot()  # must not raise RecursionError
        st.restore(snap)
        restored = st.memo[0]
        self.assertEqual(restored[0], 1)
        self.assertIs(restored[1], restored, "cycle must be rebuilt as a cycle")

    def test_mark_sentinel_identity_preserved(self) -> None:
        """_pop_to_mark compares by identity, so the sentinel must not be copied."""
        st = self._state()
        st.stack = [st._MARK, 3]
        st.restore(st.snapshot())
        self.assertIs(st.stack[0], st._MARK)


class ShardFactoryGuardTests(unittest.TestCase):
    def test_overlong_descriptor_named_clearly(self) -> None:
        rng = np.random.default_rng(1)
        w = rng.integers(-128, 128, size=(8, 4), dtype=np.int8)
        s = rng.random((1, 4), dtype=np.float32) + 0.25
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError) as ctx:
                _sf.write_quantized_string_dtype(Path(td) / "s", w, s, descr="x" * 256)
            self.assertIn("too long", str(ctx.exception))


class CloneAliasingAcrossRootsTests(unittest.TestCase):
    """Stack and memo must be cloned with one shared identity map.

    A memoized container is routinely reachable from both. Cloning them
    independently restores it as two distinct objects, so a later BINGET would
    mutate a different object than the stack holds and the retry would model a
    pickle state the shard never had.
    """

    def _state(self):
        import export_grok1_int8_scan as scan

        return scan._HeaderState()

    def test_alias_between_stack_and_memo_survives_round_trip(self) -> None:
        st = self._state()
        shared: list = [1, 2]
        st.stack = [shared]
        st.memo = {0: shared}
        st.restore(st.snapshot())
        self.assertIs(st.stack[0], st.memo[0])

    def test_round_trip_does_not_alias_back_to_the_snapshot(self) -> None:
        """A second retry must not mutate the objects held by the first snapshot."""
        st = self._state()
        original: list = [1]
        st.stack = [original]
        snap = st.snapshot()
        st.restore(snap)
        st.stack[0].append(99)
        self.assertEqual(original, [1], "restore handed back the caller's object")


if __name__ == "__main__":
    unittest.main()
