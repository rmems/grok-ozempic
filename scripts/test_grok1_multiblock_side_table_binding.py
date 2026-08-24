#!/usr/bin/env python3
"""Content-binding regressions for the issue #85 INT4 side-table cache."""
from __future__ import annotations

import json
import sys
import tempfile
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
    _reference_fingerprint,
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
        name: rng.standard_normal((3, 11, 7)).astype(np.float32)
        for name in _ROLES.values()
    }


def _reference(arrays: dict[str, np.ndarray]):
    ref = mock.Mock(spec=["roles", "vector"])
    ref.roles = dict(_ROLES)
    ref.vector = lambda role: arrays[ref.roles[role]]
    return ref


class SideTableBindingTests(unittest.TestCase):
    def test_current_schema_reuses_bound_codes_and_scales(self) -> None:
        for mode in (INT4_SCALE_ABSMAX, INT4_SCALE_LS_CHANNEL_ALPHA):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                events: list[str] = []
                record = mock.Mock()

                root = Path(td)
                ref = _reference(_arrays())
                first = Int4SideExperts(
                    ref, side_root=root, block=0, scale_mode=mode
                )
                sidecar = json.loads((first._side_dir / "sidecar.json").read_text())
                self.assertEqual(
                    sidecar["schema_version"], lib.INT4_SIDECAR_SCHEMA_VERSION
                )
                self.assertEqual(
                    set(sidecar["tensors"]["gate"]["binding"]),
                    {"reference", "q_codes", "scale", "generation"},
                )

                with mock.patch.object(
                    Int4SideExperts,
                    "_publish_hook",
                    autospec=True,
                    side_effect=record,
                ):
                    second = Int4SideExperts(
                        ref, side_root=root, block=0, scale_mode=mode
                    )

                events.extend(call.args[1] for call in record.call_args_list)
                self.assertEqual(events, ["sidecar_published"])
                self.assertIsInstance(second._cache["expert_gelu"][0], np.memmap)

    def test_legacy_sidecar_rebuilds_interrupted_q_and_stale_scale(self) -> None:
        events: list[str] = []

        def record(_self, step: str, _path: Path) -> None:
            events.append(step)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_arrays = _arrays(5)
            first = Int4SideExperts(
                _reference(old_arrays),
                side_root=root,
                block=0,
                scale_mode=INT4_SCALE_ABSMAX,
            )
            old_gate_scale = np.load(
                first._paths("gate")[1], allow_pickle=False
            ).copy()
            sidecar_path = first._side_dir / "sidecar.json"
            legacy = json.loads(sidecar_path.read_text())
            legacy.pop("schema_version")
            for entry in legacy["tensors"].values():
                entry.pop("binding")
            lib._atomic_write_json(sidecar_path, legacy)

            # Reproduce the old writer's unsafe interruption: new q and matching
            # reference fingerprint were published while the prior scale and
            # unbound sidecar remained visible.
            new_arrays = _arrays(6)
            for name, weights in new_arrays.items():
                q_path, _ = first._paths(name)
                q, _ = int4_absmax_quantize(weights)
                np.save(q_path, q, allow_pickle=False)
                lib._atomic_write_json(
                    first._fingerprint_path(name),
                    _reference_fingerprint(weights),
                )
            _, expected_gate_scale = int4_absmax_quantize(new_arrays["gate"])
            self.assertFalse(np.array_equal(old_gate_scale, expected_gate_scale))

            with mock.patch.object(
                Int4SideExperts, "_publish_hook", autospec=True, side_effect=record
            ):
                recovered = Int4SideExperts(
                    _reference(new_arrays),
                    side_root=root,
                    block=0,
                    scale_mode=INT4_SCALE_ABSMAX,
                )

            self.assertEqual(events.count("q_published"), len(_ROLES))
            self.assertEqual(events.count("scale_published"), len(_ROLES))
            for role, name in _ROLES.items():
                expected_q, expected_scale = int4_absmax_quantize(
                    new_arrays[name]
                )
                got_q, got_scale = recovered._cache[role]
                np.testing.assert_array_equal(got_q, expected_q)
                np.testing.assert_array_equal(got_scale, expected_scale)

    def test_q_tamper_rebuilds_with_matching_reference_fingerprint(self) -> None:
        events: list[str] = []

        def record(_self, step: str, _path: Path) -> None:
            events.append(step)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            arrays = _arrays(9)
            first = Int4SideExperts(
                _reference(arrays),
                side_root=root,
                block=0,
                scale_mode=INT4_SCALE_ABSMAX,
            )
            q_path, _ = first._paths("gate")
            tampered = np.load(q_path, allow_pickle=False).copy()
            tampered.flat[0] = 6 if int(tampered.flat[0]) == 7 else 7
            np.save(q_path, tampered, allow_pickle=False)

            with mock.patch.object(
                Int4SideExperts, "_publish_hook", autospec=True, side_effect=record
            ):
                recovered = Int4SideExperts(
                    _reference(arrays),
                    side_root=root,
                    block=0,
                    scale_mode=INT4_SCALE_ABSMAX,
                )

            expected_q, _ = int4_absmax_quantize(arrays["gate"])
            np.testing.assert_array_equal(
                recovered._cache["expert_gelu"][0], expected_q
            )
            self.assertIn("q_published", events)

    def test_scale_tamper_recomputes_without_rebuilding_bound_q(self) -> None:
        events: list[str] = []

        def record(_self, step: str, _path: Path) -> None:
            events.append(step)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            arrays = _arrays(10)
            first = Int4SideExperts(
                _reference(arrays),
                side_root=root,
                block=0,
                scale_mode=INT4_SCALE_ABSMAX,
            )
            _, scale_path = first._paths("up")
            tampered = np.load(scale_path, allow_pickle=False).copy()
            tampered.flat[0] += np.float32(1.0)
            np.save(scale_path, tampered, allow_pickle=False)

            with mock.patch.object(
                Int4SideExperts, "_publish_hook", autospec=True, side_effect=record
            ):
                recovered = Int4SideExperts(
                    _reference(arrays),
                    side_root=root,
                    block=0,
                    scale_mode=INT4_SCALE_ABSMAX,
                )

            _, expected_scale = int4_absmax_quantize(arrays["up"])
            np.testing.assert_array_equal(
                recovered._cache["expert_value"][1], expected_scale
            )
            self.assertNotIn("q_published", events)
            self.assertEqual(events.count("scale_published"), 1)

    def test_non_float32_scale_rebuilds_even_when_values_match(self) -> None:
        events: list[str] = []

        def record(_self, step: str, _path: Path) -> None:
            events.append(step)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            arrays = _arrays(11)
            first = Int4SideExperts(
                _reference(arrays),
                side_root=root,
                block=0,
                scale_mode=INT4_SCALE_ABSMAX,
            )
            _, scale_path = first._paths("up")
            matching_values = np.load(scale_path, allow_pickle=False).astype(
                np.float64
            )
            np.save(scale_path, matching_values, allow_pickle=False)

            with mock.patch.object(
                Int4SideExperts, "_publish_hook", autospec=True, side_effect=record
            ):
                recovered = Int4SideExperts(
                    _reference(arrays),
                    side_root=root,
                    block=0,
                    scale_mode=INT4_SCALE_ABSMAX,
                )

            recovered_scale = recovered._cache["expert_value"][1]
            self.assertEqual(recovered_scale.dtype, np.dtype(np.float32))
            np.testing.assert_array_equal(
                recovered_scale,
                int4_absmax_quantize(arrays["up"])[1],
            )
            self.assertNotIn("q_published", events)
            self.assertEqual(events.count("scale_published"), 1)

    def test_ls_scale_tamper_recomputes_without_rebuilding_bound_q(self) -> None:
        events: list[str] = []

        def record(_self, step: str, _path: Path) -> None:
            events.append(step)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            arrays = _arrays(12)
            Int4SideExperts(
                _reference(arrays),
                side_root=root,
                block=0,
                scale_mode=INT4_SCALE_ABSMAX,
            )
            ls = Int4SideExperts(
                _reference(arrays),
                side_root=root,
                block=0,
                scale_mode=INT4_SCALE_LS_CHANNEL_ALPHA,
            )
            _, scale_path = ls._paths("up")
            tampered = np.load(scale_path, allow_pickle=False).copy()
            tampered.flat[0] += np.float32(1.0)
            np.save(scale_path, tampered, allow_pickle=False)

            with mock.patch.object(
                Int4SideExperts, "_publish_hook", autospec=True, side_effect=record
            ):
                recovered = Int4SideExperts(
                    _reference(arrays),
                    side_root=root,
                    block=0,
                    scale_mode=INT4_SCALE_LS_CHANNEL_ALPHA,
                )

            q = recovered._cache["expert_value"][0]
            expected_scale = int4_ls_channel_alpha_scale(arrays["up"], q)
            np.testing.assert_array_equal(
                recovered._cache["expert_value"][1], expected_scale
            )
            self.assertNotIn("q_published", events)
            self.assertEqual(events.count("scale_published"), 1)

    def test_missing_ls_sidecar_reuses_q_bound_by_absmax_sidecar(self) -> None:
        events: list[str] = []

        def record(_self, step: str, _path: Path) -> None:
            events.append(step)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            arrays = _arrays(13)
            absmax = Int4SideExperts(
                _reference(arrays),
                side_root=root,
                block=0,
                scale_mode=INT4_SCALE_ABSMAX,
            )
            ls = Int4SideExperts(
                _reference(arrays),
                side_root=root,
                block=0,
                scale_mode=INT4_SCALE_LS_CHANNEL_ALPHA,
            )
            (ls._side_dir / "sidecar.json").unlink()

            with mock.patch.object(
                Int4SideExperts, "_publish_hook", autospec=True, side_effect=record
            ):
                recovered = Int4SideExperts(
                    _reference(arrays),
                    side_root=root,
                    block=0,
                    scale_mode=INT4_SCALE_LS_CHANNEL_ALPHA,
                )

            np.testing.assert_array_equal(
                recovered._cache["expert_value"][0],
                absmax._cache["expert_value"][0],
            )
            self.assertNotIn("q_published", events)
            self.assertEqual(events.count("scale_published"), len(_ROLES))

    def test_shared_q_tamper_invalidates_both_mode_certificates(self) -> None:
        events: list[str] = []

        def record(_self, step: str, _path: Path) -> None:
            events.append(step)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            arrays = _arrays(14)
            absmax = Int4SideExperts(
                _reference(arrays),
                side_root=root,
                block=0,
                scale_mode=INT4_SCALE_ABSMAX,
            )
            ls = Int4SideExperts(
                _reference(arrays),
                side_root=root,
                block=0,
                scale_mode=INT4_SCALE_LS_CHANNEL_ALPHA,
            )
            q_path, _ = ls._paths("up")
            tampered = np.load(q_path, allow_pickle=False).copy()
            tampered.flat[0] = 6 if int(tampered.flat[0]) == 7 else 7
            np.save(q_path, tampered, allow_pickle=False)

            with mock.patch.object(
                Int4SideExperts, "_publish_hook", autospec=True, side_effect=record
            ):
                recovered = Int4SideExperts(
                    _reference(arrays),
                    side_root=root,
                    block=0,
                    scale_mode=INT4_SCALE_LS_CHANNEL_ALPHA,
                )

            expected_q, _ = int4_absmax_quantize(arrays["up"])
            np.testing.assert_array_equal(
                recovered._cache["expert_value"][0], expected_q
            )
            self.assertIn("q_published", events)
            self.assertFalse((absmax._side_dir / "sidecar.json").exists())
            self.assertTrue((ls._side_dir / "sidecar.json").is_file())
            self.assertFalse(absmax._paths("up")[1].exists())


if __name__ == "__main__":
    unittest.main()
