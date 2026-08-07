"""Unit tests for GOZ1 pack readback in route-preservation metrics (GH #53 / RM-222).

Packs are synthesized here byte-for-byte against the container `weight_pack.rs`
writes, so the reader's fail-closed guarantees are locked by CI rather than by
one-off manual verification against a real pack.
"""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import route_preservation_io as rio  # noqa: E402

# Import the *production* layout helpers rather than reimplementing them: the
# reader validates cumulative offsets and truncation against these, so sharing
# them makes the synthesized pack track any future alignment change instead of
# silently diverging and letting these tests pass against a stale layout.
from goz1_trit_histogram import (  # noqa: E402
    DATA_ALIGNMENT,
    TENSOR_F16,
    TENSOR_TERNARY,
    _align_up,
    _payload_nbytes,
)

MAGIC = b"GOZ1"
META_STR = 1


def _u32(v: int) -> bytes:
    return struct.pack("<I", v)


def _u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def _s(text: str) -> bytes:
    raw = text.encode("utf-8")
    return _u64(len(raw)) + raw


def _align(n: int) -> int:
    return _align_up(n, DATA_ALIGNMENT)


def _assert_payload_size(name: str, shape: list[int], kind: int, payload: bytes) -> None:
    """Keep fixtures locked to the size the reader will compute."""
    numel = 1
    for d in shape:
        numel *= d
    want = _payload_nbytes(kind, numel, name)
    if len(payload) != want:
        raise AssertionError(
            f"{name}: fixture payload is {len(payload)} bytes, but the reader will "
            f"expect {want} for shape {shape} -- fixture and reader have diverged"
        )


def build_pack(
    path: Path,
    tensors: list[tuple[str, list[int], int, bytes]],
    *,
    truncate: int = 0,
    version: int = 1,
    scales: list[float] | None = None,
) -> Path:
    """Write a minimal GOZ1 pack.

    ``tensors`` items are ``(name, shape, tensor_type, payload_bytes)``; payloads
    are laid out at 32-byte-aligned cumulative offsets exactly as the Rust writer
    does. ``truncate`` drops that many bytes from the tail.

    ``version`` selects the tensor-row layout: 1 has no scale field, 2 appends a
    per-tensor ``f32`` (GH #65). The default stays 1 so the existing fixtures keep
    exercising the legacy path they were written for; v2 coverage is explicit.
    ``scales`` supplies those values (default 1.0 each) and is ignored for v1.
    """
    head = bytearray(MAGIC + _u32(version) + _u64(len(tensors)) + _u64(1))
    head += _s("oz.name") + _u32(META_STR) + _s("grok-ozempic")

    if scales is not None and len(scales) != len(tensors):
        raise AssertionError("scales must be one per tensor")

    rel = 0
    for i, (name, shape, kind, payload) in enumerate(tensors):
        if not truncate:
            _assert_payload_size(name, shape, kind, payload)
        head += _s(name) + _u32(len(shape))
        for d in shape:
            head += _u64(d)
        head += _u32(kind) + _u64(rel)
        if version >= 2:
            head += struct.pack("<f", 1.0 if scales is None else scales[i])
        rel = _align(rel + len(payload))

    data_start = _align(len(head))
    blob = bytearray(head + b"\x00" * (data_start - len(head)))
    rel = 0
    for _name, _shape, _kind, payload in tensors:
        blob[data_start + rel : data_start + rel + len(payload)] = payload
        rel = _align(rel + len(payload))
        blob.extend(b"\x00" * max(0, data_start + rel - len(blob)))

    raw = bytes(blob)
    path.write_bytes(raw[: len(raw) - truncate] if truncate else raw)
    return path


def _trits(codes: list[int]) -> bytes:
    """Pack 2-bit codes LSB-first, four per byte (matches quantizer.rs)."""
    out = bytearray((len(codes) + 3) // 4)
    for i, c in enumerate(codes):
        out[i // 4] |= (c & 0b11) << ((i % 4) * 2)
    return bytes(out)


class LoadPackIndexTests(unittest.TestCase):
    def test_indexes_offsets_and_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = build_pack(
                Path(td) / "p.goz1",
                [
                    ("a.ternary", [8], TENSOR_TERNARY, _trits([1] * 8)),
                    ("b.preserve", [4], TENSOR_F16, np.zeros(4, dtype="<f2").tobytes()),
                ],
            )
            _meta, index = rio.load_pack_index(p)
        self.assertEqual(sorted(index), ["a.ternary", "b.preserve"])
        self.assertEqual(index["a.ternary"]["numel"], 8)
        self.assertEqual(index["a.ternary"]["nbytes"], 2)
        self.assertEqual(index["b.preserve"]["nbytes"], 8)
        # Second payload starts on the next 32-byte boundary.
        self.assertEqual(
            index["b.preserve"]["abs_offset"] - index["a.ternary"]["abs_offset"],
            DATA_ALIGNMENT,
        )

    def test_duplicate_tensor_name_rejected(self) -> None:
        """Two payloads under one name would silently measure only the last."""
        with tempfile.TemporaryDirectory() as td:
            p = build_pack(
                Path(td) / "p.goz1",
                [
                    ("dup", [8], TENSOR_TERNARY, _trits([1] * 8)),
                    ("dup", [8], TENSOR_TERNARY, _trits([2] * 8)),
                ],
            )
            with self.assertRaises(rio.MetricsError) as ctx:
                rio.load_pack_index(p)
        self.assertIn("duplicate tensor name", str(ctx.exception))

    def test_truncated_pack_rejected_at_index_time(self) -> None:
        """Fail fast during indexing, not lazily inside the first read."""
        with tempfile.TemporaryDirectory() as td:
            p = build_pack(
                Path(td) / "p.goz1",
                [("a", [64], TENSOR_F16, np.zeros(64, dtype="<f2").tobytes())],
                truncate=64,
            )
            with self.assertRaises(rio.MetricsError) as ctx:
                rio.load_pack_index(p)
        self.assertIn("truncated pack", str(ctx.exception))

    def test_malformed_header_is_metrics_error(self) -> None:
        """Header parser errors must not escape as a traceback."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "p.goz1"
            p.write_bytes(b"NOPE" + b"\x00" * 64)
            with self.assertRaises(rio.MetricsError) as ctx:
                rio.load_pack_index(p)
        self.assertIn("malformed GOZ1 header", str(ctx.exception))

    def test_missing_pack_is_metrics_error(self) -> None:
        with self.assertRaises(rio.MetricsError):
            rio.load_pack_index(Path("/nonexistent/pack.goz1"))


class ReadTritsTests(unittest.TestCase):
    def _one(self, codes: list[int]):
        td = tempfile.TemporaryDirectory()
        p = build_pack(Path(td.name) / "p.goz1", [("t", [len(codes)], TENSOR_TERNARY, _trits(codes))])
        _meta, index = rio.load_pack_index(p)
        return td, p, index["t"]

    def _three(self, codes: list[int]):
        """Build a pack with tensors before and after the target."""
        td = tempfile.TemporaryDirectory()
        p = build_pack(
            Path(td.name) / "p.goz1",
            [
                ("before", [8], TENSOR_TERNARY, _trits([1] * 8)),
                ("target", [len(codes)], TENSOR_TERNARY, _trits(codes)),
                ("after", [8], TENSOR_TERNARY, _trits([2] * 8)),
            ],
        )
        _meta, index = rio.load_pack_index(p)
        return td, p, index["target"]

    def test_decodes_codes(self) -> None:
        td, p, entry = self._one([0, 1, 2, 0, 1, 2, 1, 0])
        try:
            got = rio.read_trits(p, entry, 0, 8)
            np.testing.assert_array_equal(got, [0, 1, -1, 0, 1, -1, 1, 0])
        finally:
            td.cleanup()

    def test_unaligned_start_slices_correctly(self) -> None:
        """Callers chunk by whole rows, which need not land on a byte boundary."""
        codes = [1, 2, 1, 2, 0, 1, 2, 0, 1, 1, 2, 2]
        td, p, entry = self._one(codes)
        try:
            expect = {0: 0, 1: 1, 2: -1}
            for start in range(len(codes)):
                got = rio.read_trits(p, entry, start, len(codes) - start)
                np.testing.assert_array_equal(
                    got, [expect[c] for c in codes[start:]], err_msg=f"start={start}"
                )
        finally:
            td.cleanup()

    def test_out_of_range_ranges_are_rejected(self) -> None:
        """Both bounds matter, and neither is caught by the truncation check.

        A negative start makes ``divmod`` yield a negative byte offset
        (``divmod(-8, 4) == (-2, 0)``), so the seek lands in the *preceding*
        tensor. An over-long end reads on into the *following* tensor. The
        truncation check only fires at end-of-file, so in a multi-tensor pack
        both return plausible trits from the wrong weights instead of erroring.

        The fixture contains tensors before and after the target, so invalid
        ranges would overlap neighboring payloads.
        """
        codes = [1, 2, 0, 1, 2, 0, 1, 2]
        td, p, entry = self._three(codes)
        try:
            for start, count in [(-8, 4), (-1, 2), (0, -4), (0, len(codes) + 1), (len(codes) - 2, 8)]:
                with self.assertRaises(rio.MetricsError, msg=f"{start},{count}") as ctx:
                    rio.read_trits(p, entry, start, count)
                self.assertIn("not within", str(ctx.exception))
        finally:
            td.cleanup()

    def test_exact_full_range_is_allowed(self) -> None:
        """The guard must not reject a legitimate whole-tensor read."""
        codes = [1, 2, 0, 1, 2, 0, 1, 2]
        td, p, entry = self._one(codes)
        try:
            self.assertEqual(rio.read_trits(p, entry, 0, len(codes)).size, len(codes))
            self.assertEqual(rio.read_trits(p, entry, len(codes), 0).size, 0)
        finally:
            td.cleanup()

    def test_invalid_code_raises(self) -> None:
        """0b11 is not a valid trit; counting it as zero would inflate sparsity."""
        td, p, entry = self._one([1, 2, 3, 0])
        try:
            with self.assertRaises(rio.MetricsError) as ctx:
                rio.read_trits(p, entry, 0, 4)
            self.assertIn("invalid 0b11", str(ctx.exception))
        finally:
            td.cleanup()


class ReadF16Tests(unittest.TestCase):
    def test_round_trips_values_and_shape(self) -> None:
        vals = np.array([[1.0, -2.5], [0.25, 3.0]], dtype="<f2")
        with tempfile.TemporaryDirectory() as td:
            p = build_pack(Path(td) / "p.goz1", [("n", [2, 2], TENSOR_F16, vals.tobytes())])
            _meta, index = rio.load_pack_index(p)
            got = rio.read_f16(p, index["n"])
        self.assertEqual(got.shape, (2, 2))
        np.testing.assert_array_equal(got, vals.astype(np.float32))


class ReadF16SliceTests(unittest.TestCase):
    """Same bounds contract as read_trits; both readers must agree."""

    def _one(self, values: list[float]):
        td = tempfile.TemporaryDirectory()
        raw = np.asarray(values, dtype="<f2").tobytes()
        p = build_pack(Path(td.name) / "p.goz1", [("t", [len(values)], TENSOR_F16, raw)])
        _meta, index = rio.load_pack_index(p)
        return td, p, index["t"]

    def _three(self, values: list[float]):
        """Build a pack with tensors before and after the target."""
        td = tempfile.TemporaryDirectory()
        raw = np.asarray(values, dtype="<f2").tobytes()
        p = build_pack(
            Path(td.name) / "p.goz1",
            [
                ("before", [4], TENSOR_F16, np.array([9.0, 8.0, 7.0, 6.0], dtype="<f2").tobytes()),
                ("target", [len(values)], TENSOR_F16, raw),
                ("after", [4], TENSOR_F16, np.array([5.0, 4.0, 3.0, 2.0], dtype="<f2").tobytes()),
            ],
        )
        _meta, index = rio.load_pack_index(p)
        return td, p, index["target"]

    def test_slices_the_requested_range(self) -> None:
        td, p, entry = self._one([1.0, 2.0, 3.0, 4.0, 5.0])
        try:
            np.testing.assert_allclose(rio.read_f16_slice(p, entry, 1, 3), [2.0, 3.0, 4.0])
            np.testing.assert_allclose(rio.read_f16_slice(p, entry, 0, 5), [1, 2, 3, 4, 5])
        finally:
            td.cleanup()

    def test_zero_length_and_end_offset_are_allowed(self) -> None:
        """Edge offsets are legitimate: a caller may chunk down to nothing."""
        td, p, entry = self._one([1.0, 2.0, 3.0])
        try:
            self.assertEqual(rio.read_f16_slice(p, entry, 3, 0).size, 0)
            self.assertEqual(rio.read_f16_slice(p, entry, 0, 0).size, 0)
        finally:
            td.cleanup()

    def test_out_of_range_is_rejected(self) -> None:
        """A negative start seeks into the preceding tensor; an over-long end
        reads into the following one. Neither is caught by a truncation check.

        The fixture contains tensors before and after the target, so invalid
        ranges would overlap neighboring payloads.
        """
        td, p, entry = self._three([1.0, 2.0, 3.0, 4.0])
        try:
            for start, count in [(-4, 2), (-1, 1), (0, -2), (0, 5), (3, 2)]:
                with self.subTest(start=start, count=count):
                    with self.assertRaises(rio.MetricsError) as ctx:
                        rio.read_f16_slice(p, entry, start, count)
                    self.assertIn("not within", str(ctx.exception))
        finally:
            td.cleanup()


class PreserveShapeAgreementTests(unittest.TestCase):
    """Every shape source must agree exactly before the fp16 subtraction.

    NumPy broadcasts (6144,) against (6144, 1) happily and returns a plausible
    error figure computed over the wrong pairing, so a disagreement has to raise.
    """

    def setUp(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import route_preservation_measure as rpm

        self.rpm = rpm

    def _entry(self, shape):
        return {"name": "block_000.slot_11.router", "shape": list(shape)}

    def test_all_sources_agreeing_is_accepted(self) -> None:
        ref = np.zeros((6144, 8), dtype=np.float32)
        got = np.zeros((6144, 8), dtype=np.float32)
        self.rpm._require_matching_shapes(
            "block_000.slot_11.router", ref, got, self._entry((6144, 8)),
            {"block_000.slot_11.router": (6144, 8)},
        )

    def test_npy_disagreeing_with_pack_raises(self) -> None:
        ref = np.zeros((8, 6144), dtype=np.float32)  # transposed
        got = np.zeros((6144, 8), dtype=np.float32)
        with self.assertRaises(rio.MetricsError) as ctx:
            self.rpm._require_matching_shapes(
                "block_000.slot_11.router", ref, got, self._entry((6144, 8)), None
            )
        self.assertIn("shape disagreement", str(ctx.exception))

    def test_broadcastable_mismatch_still_raises(self) -> None:
        """(6144,) vs (6144, 1) would broadcast silently — must not be allowed."""
        ref = np.zeros((6144,), dtype=np.float32)
        got = np.zeros((6144, 1), dtype=np.float32)
        with self.assertRaises(rio.MetricsError):
            self.rpm._require_matching_shapes(
                "block_000.slot_07.block_norm", ref, got, self._entry((6144,)), None
            )

    def test_manifest_shape_disagreement_raises(self) -> None:
        """npy, decoded array and pack header agree, but the manifest does not."""
        ref = np.zeros((6144, 8), dtype=np.float32)
        got = np.zeros((6144, 8), dtype=np.float32)
        with self.assertRaises(rio.MetricsError) as ctx:
            self.rpm._require_matching_shapes(
                "block_000.slot_11.router", ref, got, self._entry((6144, 8)),
                {"block_000.slot_11.router": (6144, 4)},
            )
        self.assertIn("conversion manifest", str(ctx.exception))

    def test_name_absent_from_manifest_map_is_tolerated(self) -> None:
        """A manifest without this tensor simply contributes no constraint."""
        ref = np.zeros((6144,), dtype=np.float32)
        got = np.zeros((6144,), dtype=np.float32)
        self.rpm._require_matching_shapes(
            "block_000.slot_07.block_norm", ref, got, self._entry((6144,)), {}
        )


class ContainerVersionTests(unittest.TestCase):
    """v1/v2 parsing (GH #65).

    The distinction is load-bearing rather than cosmetic: ``scale is None`` is
    what tells a consumer to fall back to the oracle path and tag its provenance,
    so reading a v2 scale as absent (or inventing one for v1) silently changes
    which number a report is built from.
    """

    def _one_ternary(self, tmp: Path, **kw) -> dict:
        pack = build_pack(
            tmp / "p.goz1",
            [("block_000.slot_05.query", [8], rio.TENSOR_TERNARY, _trits([1] * 8))],
            **kw,
        )
        _meta, index = rio.load_pack_index(pack)
        return index["block_000.slot_05.query"]

    def test_v2_scale_is_parsed_from_the_tensor_row(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            entry = self._one_ternary(Path(td), version=2, scales=[0.3125])
            self.assertAlmostEqual(entry["scale"], 0.3125, places=6)
            self.assertEqual(entry["container_version"], 2)

    def test_v1_reports_no_scale_rather_than_a_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            entry = self._one_ternary(Path(td), version=1)
            self.assertIsNone(
                entry["scale"],
                "a v1 pack carries no scale; None is the signal to use the oracle path",
            )
            self.assertEqual(entry["container_version"], 1)

    def test_v2_offsets_still_line_up_after_the_extra_field(self) -> None:
        """The scale widens every row, so the data section moves."""
        with tempfile.TemporaryDirectory() as td:
            entry = self._one_ternary(Path(td), version=2, scales=[2.0])
            with (Path(td) / "p.goz1").open("rb") as f:
                f.seek(entry["abs_offset"])
                self.assertEqual(f.read(2), _trits([1] * 8))

    def test_unknown_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(rio.MetricsError) as ctx:
                self._one_ternary(Path(td), version=3)
            self.assertIn("unsupported container version", str(ctx.exception))

    def test_version_zero_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(rio.MetricsError):
                self._one_ternary(Path(td), version=0)

    def test_a_legitimately_zero_scale_is_not_confused_with_absence(self) -> None:
        """A tensor where nothing fired stores 0.0, which must stay distinct from v1's None."""
        with tempfile.TemporaryDirectory() as td:
            entry = self._one_ternary(Path(td), version=2, scales=[0.0])
            self.assertIsNotNone(entry["scale"])
            self.assertEqual(entry["scale"], 0.0)


if __name__ == "__main__":
    unittest.main()
