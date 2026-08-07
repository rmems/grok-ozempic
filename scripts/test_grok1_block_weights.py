#!/usr/bin/env python3
"""Tests for the Grok-1 block weight sources (GH #61 / RM-249).

These adapters decide *which tensor* every downstream metric is measured
against, so a routing bug here is invisible in the numbers: the experiment still
produces plausible cosines and agreements, just against the wrong weights. The
validation paths (`_validate_mix`, the oracle-alpha cache discard) therefore get
direct coverage rather than being exercised only incidentally by a full run.

No GOZ1 pack or checkpoint is required: the source protocol is small enough to
stub, which keeps these fast and runnable in CI.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grok1_block_forward import ForwardError  # noqa: E402
from grok1_block_weights import (  # noqa: E402
    ALPHA_CACHE_SUFFIX,
    ATTENTION_ROLES,
    EXPERT_ROLES,
    PRESERVED_ROLES,
    MixedWeights,
    PackWeights,
    TernaryScale,
    implementation_commit,
    sha256_file,
    stem_of,
)
from route_preservation_io import TENSOR_TERNARY  # noqa: E402
from test_route_preservation_io import _trits, build_pack  # noqa: E402


class _StubSource:
    """Stand-in weight source that reports *which* source answered.

    Every accessor returns the source's own ``tag``, so a routing assertion can
    check that MixedWeights consulted the intended side rather than merely that
    it constructed.
    """

    def __init__(self, roles: dict[str, str], label: str = "stub", tag: float = 1.0) -> None:
        self.roles = dict(roles)
        self.label = label
        self.tag = tag

    def vector(self, role: str) -> np.ndarray:
        return np.full(4, self.tag, dtype=np.float32)

    def matrix(self, role: str) -> np.ndarray:
        return np.full((2, 2), self.tag, dtype=np.float32)

    def expert(self, role: str, index: int) -> np.ndarray:
        return np.full((2, 2), self.tag + index, dtype=np.float32)


def _roles(names: list[str]) -> dict[str, str]:
    return {r: f"block_000.slot_xx.{r}" for r in names}


ALL_ROLES = sorted(ATTENTION_ROLES | EXPERT_ROLES | PRESERVED_ROLES)


class MixedWeightsValidationTests(unittest.TestCase):
    def test_routes_assigned_roles_to_primary_and_rest_to_fallback(self) -> None:
        """Assert the actual routing, not just that the object constructed.

        Each stub reports a distinct tag, so a swapped or ignored ``roles`` set
        shows up as the wrong source answering -- which a label/keys check alone
        would not catch.
        """
        primary = _StubSource(_roles(ALL_ROLES), "primary", tag=10.0)
        fallback = _StubSource(_roles(ALL_ROLES), "fallback", tag=20.0)
        mixed = MixedWeights(primary, fallback, frozenset(ATTENTION_ROLES), "mix")

        self.assertEqual(mixed.label, "mix")
        self.assertEqual(set(mixed.roles), set(ALL_ROLES))
        # Assigned roles come from the primary ...
        for role in sorted(ATTENTION_ROLES):
            self.assertEqual(mixed.matrix(role)[0][0], 10.0, msg=role)
        # ... everything else from the fallback.
        for role in sorted(PRESERVED_ROLES):
            self.assertEqual(mixed.vector(role)[0], 20.0, msg=role)
        for role in sorted(EXPERT_ROLES):
            self.assertEqual(mixed.expert(role, 0)[0][0], 20.0, msg=role)

    def test_primary_missing_an_assigned_role_is_rejected(self) -> None:
        primary = _StubSource(_roles([r for r in ALL_ROLES if r != "query"]))
        fallback = _StubSource(_roles(ALL_ROLES))
        with self.assertRaisesRegex(ForwardError, "lacks assigned roles"):
            MixedWeights(primary, fallback, frozenset(ATTENTION_ROLES), "mix")

    def test_role_served_by_neither_source_is_rejected(self) -> None:
        """The gap the old always-empty check missed.

        ``expert_gelu`` is claimed by the primary but sits *outside* ``roles``, so
        ``_pick`` routes it to the fallback -- which does not have it. Before the
        fix this surfaced as a bare KeyError inside forward_block, after the
        expensive reference pass had already run.
        """
        primary = _StubSource(_roles(ALL_ROLES))
        fallback = _StubSource(_roles([r for r in ALL_ROLES if r != "expert_gelu"]))
        with self.assertRaisesRegex(ForwardError, "no source serves roles"):
            MixedWeights(primary, fallback, frozenset(ATTENTION_ROLES), "mix")

    def test_disagreeing_role_mapping_is_rejected(self) -> None:
        """Two sources naming different tensors for one role cannot be mixed."""
        primary = _StubSource(_roles(ALL_ROLES))
        fallback = _StubSource(_roles(ALL_ROLES))
        fallback.roles["router"] = "block_000.slot_99.router_other"
        with self.assertRaisesRegex(ForwardError, "disagree on slot/role mapping"):
            MixedWeights(primary, fallback, frozenset(ATTENTION_ROLES), "mix")

    def test_expert_tier_mix_routes_experts_to_primary(self) -> None:
        """The `goz1_expert_ternary_only` baseline: experts from the pack, rest reference.

        Distinct tags are load-bearing. With both stubs on the default tag the
        expected value is identical whichever source answers, so the assertion
        holds even with routing inverted -- a vacuous test.
        """
        primary = _StubSource(_roles(ALL_ROLES), "pack", tag=10.0)
        fallback = _StubSource(_roles(ALL_ROLES), "reference", tag=20.0)
        mixed = MixedWeights(primary, fallback, frozenset(EXPERT_ROLES), "expert-only")

        # Experts come from the primary: tag 10.0 + index 3.
        self.assertEqual(mixed.expert("expert_down", 3)[0][0], 13.0)
        # Attention and the preserved tier come from the fallback.
        self.assertEqual(mixed.matrix("attn_out")[0][0], 20.0)
        self.assertEqual(mixed.vector("router")[0], 20.0)


class TernaryScaleTests(unittest.TestCase):
    def test_sparsity_is_the_unfired_fraction(self) -> None:
        self.assertAlmostEqual(TernaryScale(alpha=1.0, fired=25, total=100).sparsity, 0.75)

    def test_empty_tensor_reports_zero_sparsity_rather_than_dividing_by_zero(self) -> None:
        self.assertEqual(TernaryScale(alpha=0.0, fired=0, total=0).sparsity, 0.0)

    def test_sign_mismatches_defaults_to_zero_for_legacy_cache_entries(self) -> None:
        """A cache written before the field existed must still load."""
        self.assertEqual(TernaryScale(**{"alpha": 1.0, "fired": 2, "total": 4}).sign_mismatches, 0)


class HelperTests(unittest.TestCase):
    def test_stem_of_maps_dots_to_double_underscore(self) -> None:
        self.assertEqual(
            stem_of("block_000.slot_11.router"), "block_000__slot_11__router"
        )

    def test_sha256_matches_hashlib(self) -> None:
        import hashlib

        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "x.bin"
            f.write_bytes(b"grok" * 5000)
            self.assertEqual(sha256_file(f), hashlib.sha256(f.read_bytes()).hexdigest())

    def test_implementation_commit_reports_a_sha_and_dirty_flag(self) -> None:
        got = implementation_commit()
        self.assertEqual(set(got), {"commit", "dirty"})
        if got["commit"] is not None:
            self.assertRegex(got["commit"], r"^[0-9a-f]{40}$")
            self.assertIsInstance(got["dirty"], bool)

    def test_implementation_commit_outside_a_repo_is_not_fatal(self) -> None:
        """Provenance is best-effort: a run outside git must still write metrics."""
        with tempfile.TemporaryDirectory() as td:
            got = implementation_commit(Path(td))
        self.assertEqual(got, {"commit": None, "dirty": None})


class _FakeCacheHost:
    """Minimal host for :meth:`PackWeights._load_cache`.

    The cache loader only needs a cache path and a fingerprint, so binding the
    two real methods onto a small object exercises them directly -- no GOZ1 pack
    or 19 GiB of reference weights required. Bound at class scope rather than
    patched in ``setUp`` so they are ordinary methods to a reader and to pylint.
    """

    _load_cache = PackWeights._load_cache
    _discard_cache = PackWeights._discard_cache

    def __init__(self, path: Path, fingerprint: dict) -> None:
        self._cache_path = path
        self._fp = fingerprint

    def _fingerprint(self) -> dict:
        return self._fp


class AlphaCacheDiscardTests(unittest.TestCase):
    """A malformed cache must be recomputed, never fatal.

    The cache is derived data with deterministic recomputation, so aborting would
    let one corrupt file block the experiment until deleted by hand.
    """

    def _load(self, payload: str, fingerprint: dict | None = None):
        fp = {"pack_size": 1} if fingerprint is None else fingerprint
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "c.json"
            path.write_text(payload)
            stderr_capture = io.StringIO()
            with contextlib.redirect_stderr(stderr_capture):
                result = _FakeCacheHost(path, fp)._load_cache()
            return result, stderr_capture.getvalue()

    def test_unreadable_json_is_discarded(self) -> None:
        result, warning = self._load("{ not json")
        self.assertEqual(result, {})
        self.assertIn("unreadable", warning)

    def test_non_object_payload_is_discarded(self) -> None:
        result, warning = self._load("[1, 2, 3]")
        self.assertEqual(result, {})
        self.assertIn("not a JSON object", warning)

    def test_non_object_scales_is_discarded(self) -> None:
        payload = json.dumps({"fingerprint": {"pack_size": 1}, "scales": [1, 2]})
        result, warning = self._load(payload)
        self.assertEqual(result, {})
        self.assertIn("not a JSON object", warning)

    def test_malformed_scale_entry_is_discarded(self) -> None:
        payload = json.dumps({"fingerprint": {"pack_size": 1}, "scales": {"t": {"nope": 1}}})
        result, warning = self._load(payload)
        self.assertEqual(result, {})
        self.assertIn("malformed scale entry", warning)

    def test_fingerprint_mismatch_discards_quietly(self) -> None:
        payload = json.dumps({"fingerprint": {"pack_size": 999}, "scales": {}})
        result, warning = self._load(payload)
        self.assertEqual(result, {})
        self.assertEqual(warning, "")

    def test_missing_cache_file_is_an_empty_cache(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            host = _FakeCacheHost(Path(td) / "absent.json", {"pack_size": 1})
            self.assertEqual(host._load_cache(), {})

    def test_matching_fingerprint_loads_the_scales(self) -> None:
        fp = {"pack_size": 1}
        payload = json.dumps(
            {"fingerprint": fp, "scales": {"t": {"alpha": 0.5, "fired": 2, "total": 4}}}
        )
        got, warning = self._load(payload, fp)
        self.assertEqual(list(got), ["t"])
        self.assertAlmostEqual(got["t"].alpha, 0.5)
        self.assertAlmostEqual(got["t"].sparsity, 0.5)
        self.assertEqual(warning, "")


class StoredScaleTests(unittest.TestCase):
    """PackWeights must prefer the pack's own scale (GH #65).

    The point of the format change is that a pack dequantizes without the source
    weights. These build a real pack on disk and deliberately provide **no** npy
    directory, so any fall-through to the oracle path fails loudly instead of
    quietly producing a figure no runtime could reproduce.

    A norm-shaped tensor is used because it is the only role whose real shape
    (6144,) yields a fixture payload small enough to write inline; the scale
    logic is per-tensor and does not depend on which role it is.
    """

    NAME = "block_000.slot_07.block_norm"
    NUMEL = 6144

    def _pack(self, td: Path, **kw) -> Path:
        return build_pack(
            td / "p.goz1",
            [(self.NAME, [self.NUMEL], TENSOR_TERNARY, _trits([1] * self.NUMEL))],
            **kw,
        )

    def _weights(self, td: Path, **kw) -> PackWeights:
        # npy_dir points at a directory with nothing in it: reaching the oracle
        # path at all is the failure this asserts against.
        return PackWeights(self._pack(td, **kw), td / "absent-npy", partial=True)

    def test_v2_scale_comes_from_the_pack_with_no_reference_weights(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            w = self._weights(Path(td), version=2, scales=[0.4375])
            got = w.scale(self.NAME)
            self.assertAlmostEqual(got.alpha, 0.4375, places=6)
            self.assertEqual(w.scale_sources[self.NAME], "pack_v2")

    def test_v1_pack_without_reference_npy_fails_with_an_actionable_message(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            w = self._weights(Path(td), version=1)
            with self.assertRaises(ForwardError) as ctx:
                w.scale(self.NAME)
            msg = str(ctx.exception)
            self.assertIn("v1 pack", msg)
            self.assertIn("re-pack", msg.lower())

    def test_a_stale_oracle_cache_cannot_shadow_a_stored_scale(self) -> None:
        """The regression that would silently revert the whole change.

        A cache file left beside a pack from a previous v1 run has the same
        default path. If it were consulted for a v2 pack, `scale()` would return
        the oracle alpha and report it as pack-only.
        """
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pack = self._pack(tmp, version=2, scales=[0.25])
            npy_dir = tmp / "absent-npy"

            # The fingerprint must MATCH, or _load_cache discards the file before
            # it can shadow anything and the test proves nothing. (It did exactly
            # that with a dummy fingerprint: a mutant that consulted the cache for
            # v2 packs still passed.)
            probe = PackWeights(pack, npy_dir, partial=True)
            cache = pack.with_suffix(pack.suffix + ALPHA_CACHE_SUFFIX)
            cache.write_text(
                json.dumps(
                    {
                        "fingerprint": probe._fingerprint(),
                        "scales": {self.NAME: {"alpha": 999.0, "fired": 1, "total": 1}},
                    }
                )
            )
            # Sanity: this cache really would load if it were consulted.
            self.assertEqual(
                PackWeights(pack, npy_dir, partial=True)._load_cache()[self.NAME].alpha,
                999.0,
            )

            w = PackWeights(pack, npy_dir, partial=True)
            self.assertAlmostEqual(w.scale(self.NAME).alpha, 0.25, places=6)
            self.assertEqual(w.scale_sources[self.NAME], "pack_v2")

    def test_non_finite_stored_scale_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            w = self._weights(Path(td), version=2, scales=[float("nan")])
            with self.assertRaisesRegex(ForwardError, "non-finite scale"):
                w.scale(self.NAME)

    def test_v1_cache_preload_records_scale_source_as_legacy_oracle(self) -> None:
        """A scale pre-loaded from disk cache must appear in scale_sources.

        Without the fix, calling scale() for a name already in self._scales
        (because it was loaded from the on-disk v1 alpha cache during __init__)
        skipped the "if name not in self._scales:" branch entirely and never
        recorded the scale source, so scale_sources[name] was silently absent.

        This regression would break route-preservation certification: a metric
        that believes it is reporting pack-only figures could quietly include
        oracle-derived scales, invalidating the runtime-reproducibility claim.
        """
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pack = self._pack(tmp, version=1)
            npy_dir = tmp / "absent-npy"

            # Build a matching-fingerprint cache file. The fingerprint must match
            # or _load_cache discards the file before __init__ can pre-populate
            # self._scales from it, and the test proves nothing.
            probe = PackWeights(pack, npy_dir, partial=True)
            cache = pack.with_suffix(pack.suffix + ALPHA_CACHE_SUFFIX)
            cache.write_text(
                json.dumps(
                    {
                        "fingerprint": probe._fingerprint(),
                        "scales": {self.NAME: {"alpha": 0.1875, "fired": 100, "total": 6144}},
                    }
                )
            )

            # Construct a fresh PackWeights. Because this is a v1 pack,
            # __init__ will call _load_cache() and pre-populate self._scales.
            w = PackWeights(pack, npy_dir, partial=True)

            # Calling scale() should return the cached alpha...
            got = w.scale(self.NAME)
            self.assertAlmostEqual(got.alpha, 0.1875, places=6)

            # ...and must record the source as legacy_oracle, since v1 packs
            # never store scales in the container.
            self.assertEqual(w.scale_sources[self.NAME], "legacy_oracle")


class RoleSetTests(unittest.TestCase):
    def test_the_three_tiers_are_disjoint_and_cover_every_role(self) -> None:
        """A role in two tiers would make the attribution baselines overlap."""
        self.assertEqual(ATTENTION_ROLES & EXPERT_ROLES, frozenset())
        self.assertEqual(ATTENTION_ROLES & PRESERVED_ROLES, frozenset())
        self.assertEqual(EXPERT_ROLES & PRESERVED_ROLES, frozenset())
        self.assertEqual(len(ATTENTION_ROLES | EXPERT_ROLES | PRESERVED_ROLES), 12)

    def test_preserved_roles_are_the_router_and_all_four_norms(self) -> None:
        self.assertEqual(
            PRESERVED_ROLES,
            frozenset(
                {"router", "norm_pre_attn", "norm_post_attn", "norm_pre_moe", "norm_post_moe"}
            ),
        )


if __name__ == "__main__":
    unittest.main()
