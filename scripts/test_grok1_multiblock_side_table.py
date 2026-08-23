#!/usr/bin/env python3
"""Bounded and crash-safe INT4 side-table tests for issue #85."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grok1_multiblock_lib as lib  # noqa: E402
from grok1_multiblock_lib import (  # noqa: E402
    INT4_SCALE_ABSMAX,
    INT4_SCALE_LS_CHANNEL_ALPHA,
    Int4SideExperts,
    _chunked_absmax_scale,
    _reference_fingerprint,
    _write_chunked_q,
    int4_absmax_quantize,
    int4_ls_channel_alpha_scale,
)


_ROLES = {
    "expert_gelu": "gate",
    "expert_value": "up",
    "expert_down": "down",
}


def _arrays(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        name: rng.standard_normal((3, 11, 7)).astype(np.float32) for name in _ROLES.values()
    }


def _reference(arrays: dict[str, np.ndarray]):
    ref = mock.Mock(spec=["roles", "vector"])
    ref.roles = dict(_ROLES)
    ref.vector = lambda role: arrays[ref.roles[role]]
    return ref


class ChunkedQuantizationTests(unittest.TestCase):
    def test_chunked_q_is_bit_equivalent_to_full_quantizer(self) -> None:
        weights = _arrays()["gate"]
        expected_q, expected_scale = int4_absmax_quantize(weights)
        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            lib, "INT4_CHUNK_BYTES", 64
        ):
            scale = _chunked_absmax_scale(weights)
            path = Path(td) / "q.npy"
            _write_chunked_q(path, weights, scale)
            got_q = np.load(path, allow_pickle=False)
            np.testing.assert_array_equal(got_q, expected_q)
            np.testing.assert_array_equal(scale, expected_scale)

    def test_incremental_fingerprint_matches_legacy_raw_bytes(self) -> None:
        source = np.asfortranarray(
            np.random.default_rng(4).standard_normal((5, 9, 7)).astype(np.float64)
        )
        legacy = np.ascontiguousarray(source, dtype=np.float32)
        expected = {
            "sha256": hashlib.sha256(legacy.tobytes()).hexdigest(),
            "dtype": "float32",
            "shape": [5, 9, 7],
        }
        with mock.patch.object(lib, "INT4_CHUNK_BYTES", 80):
            self.assertEqual(_reference_fingerprint(source), expected)

    def test_ls_statistics_are_float64_chunk_equivalent(self) -> None:
        weights = _arrays(8)["gate"]
        q, _ = int4_absmax_quantize(weights)
        num = np.multiply(weights, q, dtype=np.float64).sum(axis=1, dtype=np.float64)
        den = np.multiply(q, q, dtype=np.float64).sum(axis=1, dtype=np.float64)
        expected = np.divide(num, den, out=np.zeros_like(num), where=den > 0).astype(
            np.float32
        )
        with mock.patch.object(lib, "INT4_CHUNK_BYTES", 64):
            got = int4_ls_channel_alpha_scale(weights, q)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)

    def test_side_table_does_not_call_full_rank3_quantizer_or_contiguous_copy(self) -> None:
        numpy_proxy = types.SimpleNamespace(**vars(np))
        numpy_proxy.ascontiguousarray = mock.Mock(
            side_effect=AssertionError("full copy called")
        )
        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            lib, "INT4_CHUNK_BYTES", 64
        ), mock.patch.object(
            lib, "int4_absmax_quantize", side_effect=AssertionError("full quantizer called")
        ), mock.patch.object(
            lib, "np", numpy_proxy
        ):
            side = Int4SideExperts(
                _reference(_arrays()),
                side_root=Path(td),
                block=0,
                scale_mode=INT4_SCALE_ABSMAX,
            )
            self.assertEqual(side.vector("expert_gelu").shape, (3, 11, 7))
        numpy_proxy.ascontiguousarray.assert_not_called()


class AtomicSideTableTests(unittest.TestCase):
    def test_shared_q_invalidation_fsyncs_both_mode_directories(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            side = Int4SideExperts(
                _reference(_arrays()),
                side_root=Path(td),
                block=0,
                scale_mode=INT4_SCALE_LS_CHANNEL_ALPHA,
            )
            events: list[tuple[str, Path | str]] = []
            real_fsync = lib._fsync_directory_strict

            def record_fsync(path: Path) -> None:
                events.append(("fsync", path))
                real_fsync(path)

            def record_hook(_self, step: str, _path: Path) -> None:
                events.append(("hook", step))

            with mock.patch.object(
                lib, "_fsync_directory_strict", side_effect=record_fsync
            ), mock.patch.object(
                Int4SideExperts, "_publish_hook", autospec=True, side_effect=record_hook
            ):
                side._invalidate_shared_q_chain("gate")

            self.assertEqual(
                events,
                [
                    ("fsync", side._absmax_dir),
                    ("fsync", side._ls_dir),
                    ("hook", "sidecars_invalidated"),
                    ("fsync", side._absmax_dir),
                    ("fsync", side._ls_dir),
                    ("hook", "scale_modes_invalidated"),
                ],
            )

    def test_invalidation_fsync_failure_aborts_before_q_publication(self) -> None:
        for failure in ("open", "fsync"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as td:
                reference = _reference(_arrays(1))
                side = Int4SideExperts(
                    reference,
                    side_root=Path(td),
                    block=0,
                    scale_mode=INT4_SCALE_LS_CHANNEL_ALPHA,
                )
                source = _arrays(2)["gate"]
                fingerprint = _reference_fingerprint(source)
                events: list[str] = []

                def record(
                    _self,
                    step: str,
                    _path: Path,
                    _events: list[str] = events,
                ) -> None:
                    _events.append(step)

                failure_patch = (
                    mock.patch.object(lib.os, "open", side_effect=OSError("open failed"))
                    if failure == "open"
                    else mock.patch.object(
                        lib.os, "fsync", side_effect=OSError("fsync failed")
                    )
                )
                with failure_patch, mock.patch.object(
                    Int4SideExperts, "_publish_hook", autospec=True, side_effect=record
                ), self.assertRaisesRegex(lib.ForwardError, "INT4 cache directory"):
                    side._build_shared_codes(
                        "gate",
                        source,
                        tuple(int(value) for value in source.shape),
                        fingerprint,
                    )
                self.assertNotIn("q_published", events)
                self.assertNotIn("fingerprint_published", events)

    def test_publication_order_is_q_then_fingerprint_then_scale_then_sidecar(self) -> None:
        events: list[str] = []

        def record(_self, step: str, _path: Path) -> None:
            events.append(step)

        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            Int4SideExperts, "_publish_hook", autospec=True, side_effect=record
        ):
            Int4SideExperts(
                _reference(_arrays()),
                side_root=Path(td),
                block=0,
                scale_mode=INT4_SCALE_LS_CHANNEL_ALPHA,
            )
        first = {step: events.index(step) for step in set(events)}
        self.assertLess(first["sidecars_invalidated"], first["scale_modes_invalidated"])
        self.assertLess(first["scale_modes_invalidated"], first["q_temp_durable"])
        self.assertLess(first["q_temp_durable"], first["q_published"])
        self.assertLess(first["q_published"], first["fingerprint_published"])
        self.assertLess(first["fingerprint_published"], first["scale_published"])
        self.assertEqual(events[-1], "sidecar_published")

    def test_interruption_at_every_boundary_recovers_cleanly(self) -> None:
        steps = (
            "sidecars_invalidated",
            "scale_modes_invalidated",
            "q_temp_durable",
            "q_published",
            "fingerprint_published",
            "scale_published",
            "sidecar_published",
        )
        for interrupted_step in steps:
            with self.subTest(step=interrupted_step), tempfile.TemporaryDirectory() as td:
                fired = False

                def interrupt(
                    _self,
                    step: str,
                    _path: Path,
                    target_step: str = interrupted_step,
                ) -> None:
                    nonlocal fired
                    if step == target_step and not fired:
                        fired = True
                        raise RuntimeError(f"interrupt after {step}")

                with mock.patch.object(
                    Int4SideExperts, "_publish_hook", autospec=True, side_effect=interrupt
                ), self.assertRaisesRegex(RuntimeError, "interrupt after"):
                    Int4SideExperts(
                        _reference(_arrays()),
                        side_root=Path(td),
                        block=0,
                        scale_mode=INT4_SCALE_LS_CHANNEL_ALPHA,
                    )
                recovered = Int4SideExperts(
                    _reference(_arrays()),
                    side_root=Path(td),
                    block=0,
                    scale_mode=INT4_SCALE_LS_CHANNEL_ALPHA,
                )
                self.assertTrue(np.isfinite(recovered.vector("expert_gelu")).all())
                sidecar = json.loads((recovered._side_dir / "sidecar.json").read_text())
                self.assertEqual(set(sidecar["tensors"]), {"gate", "up", "down"})

    def test_q_replacement_invalidates_both_modes_before_publish(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = _reference(_arrays(1))
            abs_side = Int4SideExperts(
                first, side_root=root, block=0, scale_mode=INT4_SCALE_ABSMAX
            )
            ls_side = Int4SideExperts(
                first, side_root=root, block=0, scale_mode=INT4_SCALE_LS_CHANNEL_ALPHA
            )
            observed = False

            def inspect(_self, step: str, path: Path) -> None:
                nonlocal observed
                if step == "scale_modes_invalidated" and not observed:
                    observed = True
                    stem = path.name.removesuffix("__q_int8.npy")
                    name = stem.replace("__", ".")
                    abs_scale, ls_scale = ls_side._scale_paths(name)
                    self.assertFalse(abs_scale.exists())
                    self.assertFalse(ls_scale.exists())
                    self.assertTrue(all(not p.exists() for p in ls_side._sidecar_paths()))

            with mock.patch.object(
                Int4SideExperts, "_publish_hook", autospec=True, side_effect=inspect
            ):
                Int4SideExperts(
                    _reference(_arrays(2)),
                    side_root=root,
                    block=0,
                    scale_mode=INT4_SCALE_LS_CHANNEL_ALPHA,
                )
            self.assertTrue(observed)
            self.assertFalse(abs_side._side_dir.joinpath("sidecar.json").exists())

    def test_truncated_q_and_scale_trigger_rebuild(self) -> None:
        for target in ("q", "scale"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                ref = _reference(_arrays())
                first = Int4SideExperts(
                    ref, side_root=root, block=0, scale_mode=INT4_SCALE_ABSMAX
                )
                q_path, scale_path = first._paths("gate")
                (q_path if target == "q" else scale_path).write_bytes(b"truncated")
                rebuilt = Int4SideExperts(
                    ref, side_root=root, block=0, scale_mode=INT4_SCALE_ABSMAX
                )
                self.assertTrue(np.isfinite(rebuilt.vector("expert_gelu")).all())

    def test_cross_mode_stale_scale_is_rejected_after_q_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old = _reference(_arrays(5))
            Int4SideExperts(old, side_root=root, block=0, scale_mode=INT4_SCALE_ABSMAX)
            old_ls = Int4SideExperts(
                old, side_root=root, block=0, scale_mode=INT4_SCALE_LS_CHANNEL_ALPHA
            )
            gate_abs_scale, _ = old_ls._scale_paths("gate")
            old_abs = np.load(gate_abs_scale, allow_pickle=False).copy()
            new = _reference(_arrays(6))
            Int4SideExperts(
                new, side_root=root, block=0, scale_mode=INT4_SCALE_LS_CHANNEL_ALPHA
            )
            self.assertFalse(
                gate_abs_scale.exists(),
                "old absmax scale survived shared-q replacement",
            )
            rebuilt = Int4SideExperts(
                new, side_root=root, block=0, scale_mode=INT4_SCALE_ABSMAX
            )
            self.assertFalse(
                np.array_equal(old_abs, np.load(gate_abs_scale, allow_pickle=False))
            )
            self.assertTrue(np.isfinite(rebuilt.vector("expert_gelu")).all())

    def test_abandoned_temp_is_removed_and_never_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ref = _reference(_arrays())
            first = Int4SideExperts(
                ref, side_root=root, block=0, scale_mode=INT4_SCALE_ABSMAX
            )
            q_path, _ = first._paths("gate")
            abandoned = q_path.parent / f".{q_path.name}.abandoned.tmp"
            abandoned.write_bytes(b"not-an-npy")
            second = Int4SideExperts(
                ref, side_root=root, block=0, scale_mode=INT4_SCALE_ABSMAX
            )
            self.assertFalse(abandoned.exists())
            self.assertTrue(np.isfinite(second.vector("expert_gelu")).all())

    def test_complete_cache_reloads_q_read_only_through_mmap(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ref = _reference(_arrays())
            Int4SideExperts(ref, side_root=root, block=0, scale_mode=INT4_SCALE_ABSMAX)
            loaded = Int4SideExperts(
                ref, side_root=root, block=0, scale_mode=INT4_SCALE_ABSMAX
            )
            q, _ = loaded._cache["expert_gelu"]
            self.assertIsInstance(q, np.memmap)
            self.assertFalse(q.flags.writeable)


if __name__ == "__main__":
    unittest.main()
