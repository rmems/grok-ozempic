#!/usr/bin/env python3
"""Unit tests for export_grok1_embedding_npy.py (no 3 GiB fixture required)."""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

# Import sibling module without installing as package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_grok1_embedding_npy as exp  # noqa: E402


def _write_fake_shard(path: Path, prefix: bytes, floats: list[float]) -> None:
    path.write_bytes(prefix + struct.pack(f"<{len(floats)}f", *floats))


def argparse_ns(**kw):
    return SimpleNamespace(**kw)


class ParseShapeTests(unittest.TestCase):
    def test_valid_shape(self) -> None:
        self.assertEqual(exp._parse_shape_csv("131072,6144"), (131072, 6144))

    def test_empty_shape(self) -> None:
        with self.assertRaises(exp.LayoutError):
            exp._parse_shape_csv("")

    def test_non_int(self) -> None:
        with self.assertRaises(exp.LayoutError):
            exp._parse_shape_csv("131072,abc")

    def test_non_positive(self) -> None:
        with self.assertRaises(exp.LayoutError):
            exp._parse_shape_csv("0,6144")
        with self.assertRaises(exp.LayoutError):
            exp._parse_shape_csv("-1,2")

    def test_empty_dimension_rejected(self) -> None:
        with self.assertRaises(exp.LayoutError) as ctx:
            exp._parse_shape_csv("1,,3")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_zero_padded_decimal(self) -> None:
        self.assertEqual(exp._parse_shape_csv("0123,456"), (123, 456))


class StemTests(unittest.TestCase):
    def test_ok(self) -> None:
        stem = "embedding__slot_00__token_embedding"
        self.assertEqual(exp.validate_stem(stem), stem)

    def test_path_escape(self) -> None:
        with self.assertRaises(exp.LayoutError):
            exp.validate_stem("../escape")
        with self.assertRaises(exp.LayoutError):
            exp.validate_stem("a/b")
        with self.assertRaises(exp.LayoutError):
            exp.validate_stem("/abs")

    def test_suffix(self) -> None:
        with self.assertRaises(exp.LayoutError):
            exp.validate_stem("foo.npy")


class WriteNpyTests(unittest.TestCase):
    def test_roundtrip_numpy_if_available(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0]
        payload = memoryview(struct.pack("<4f", *data))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.npy"
            exp.write_npy_f32(path, (2, 2), payload)
            try:
                import numpy as np
            except ImportError:
                self.skipTest("numpy not installed")
            a = np.load(path)
            self.assertEqual(a.shape, (2, 2))
            self.assertEqual(a.dtype, np.float32)
            self.assertEqual(list(a.reshape(-1)), data)


class LayoutPolicyTests(unittest.TestCase):
    def _ns(self, **kw):
        defaults = dict(
            offset=None,
            shape=None,
            dtype=None,
            no_dissect=True,
            xai_dissect=None,
        )
        defaults.update(kw)
        return argparse_ns(**defaults)

    def test_default_shard_name_allows_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            shard = Path(td) / exp.DEFAULT_SHARD_NAME
            ns = self._ns(no_dissect=True)
            off, shape, dtype, nb = exp.resolve_layout(shard, ns)
            self.assertEqual(off, exp.DEFAULT_OFFSET)
            self.assertEqual(shape, exp.DEFAULT_SHAPE)
            self.assertEqual(dtype, exp.DEFAULT_DTYPE)
            self.assertIsNone(nb)

    def test_other_shard_requires_full_layout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            shard = Path(td) / "other_tensor_001"
            ns = self._ns(no_dissect=True, offset=10)
            with self.assertRaises(exp.LayoutError) as ctx:
                exp.resolve_layout(shard, ns)
            self.assertIn("missing", str(ctx.exception).lower())

    def test_other_shard_full_explicit_ok(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            shard = Path(td) / "other_tensor_001"
            ns = self._ns(
                no_dissect=True,
                offset=8,
                shape="2,2",
                dtype="f32",
            )
            off, shape, dtype, _ = exp.resolve_layout(shard, ns)
            self.assertEqual((off, shape, dtype), (8, (2, 2), "f32"))

    def test_invalid_dissect_path_rejected_even_with_complete_layout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            shard = Path(td) / "other_tensor_001"
            ns = self._ns(
                no_dissect=False,
                offset=8,
                shape="2,2",
                dtype="f32",
                xai_dissect="/nonexistent/fake-dissect",
            )
            with self.assertRaises(exp.LayoutError) as ctx:
                exp.resolve_layout(shard, ns)
            self.assertIn("xai-dissect", str(ctx.exception).lower())


class ExplicitDissectPathTests(unittest.TestCase):
    def test_bad_explicit_path_errors(self) -> None:
        with self.assertRaises(exp.LayoutError):
            exp._dissect_search_dirs("/nonexistent/not-xai-dissect")

    def test_wrong_basename_errors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "wrong-name"
            p.write_text("#!/bin/sh\n")
            p.chmod(0o755)
            with self.assertRaises(exp.LayoutError):
                exp._dissect_search_dirs(str(p))


class DissectExitCodeTests(unittest.TestCase):
    def test_nonzero_returncode_ignored_as_layout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            # Fake binary that exits 1 but prints a plausible plain layout line
            bin_path = td_path / exp.DISSECT_BIN
            bin_path.write_text(
                "#!/bin/sh\n"
                'echo "offset=999 nbytes=16 dtype=f32 shape=(2, 2)" >&2\n'
                "exit 1\n"
            )
            bin_path.chmod(0o755)
            shard = td_path / "shard_dir" / "tensorX"
            shard.parent.mkdir()
            shard.write_bytes(b"x" * 100)
            parsed = exp._run_dissect_in_dir(td_path, shard)
            self.assertIsNone(parsed)


class EndToEndCliTests(unittest.TestCase):
    def test_export_tiny_shard(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            shard = td_path / "other.bin"
            _write_fake_shard(shard, b"\x00" * 8, [0.0, 1.0, -1.0, 0.5])
            out = td_path / "out"
            ns = argparse_ns(
                shard=shard,
                output_dir=out,
                offset=8,
                shape="2,2",
                dtype="f32",
                stem="blk__0__ffn_up__weight",
                no_dissect=True,
                xai_dissect=None,
            )
            code = exp.export_embedding(ns)
            self.assertEqual(code, 0)
            npy = out / "blk__0__ffn_up__weight.npy"
            self.assertTrue(npy.is_file())

    def test_bad_shape_exit_1(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            shard = td_path / exp.DEFAULT_SHARD_NAME
            _write_fake_shard(shard, b"\x00" * 8, [1.0, 2.0, 3.0, 4.0])
            ns = argparse_ns(
                shard=shard,
                output_dir=td_path / "out",
                offset=None,
                shape="1,abc",
                dtype=None,
                stem=exp.DEFAULT_STEM,
                no_dissect=True,
                xai_dissect=None,
            )
            code = exp.export_embedding(ns)
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
