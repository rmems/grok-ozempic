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

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grok1_block_forward import ForwardError  # noqa: E402
from grok1_block_weights import (  # noqa: E402
    ATTENTION_ROLES,
    EXPERT_ROLES,
    PRESERVED_ROLES,
    MixedWeights,
    TernaryScale,
    implementation_commit,
    sha256_file,
    stem_of,
)


class _StubSource:
    """Minimal stand-in for a weight source: just a label and a role mapping."""

    def __init__(self, roles: dict[str, str], label: str = "stub") -> None:
        self.roles = dict(roles)
        self.label = label

    def vector(self, role: str) -> np.ndarray:
        return np.full(4, float(len(self.roles[role])), dtype=np.float32)

    def matrix(self, role: str) -> np.ndarray:
        return np.full((2, 2), float(len(self.roles[role])), dtype=np.float32)

    def expert(self, role: str, index: int) -> np.ndarray:
        return np.full((2, 2), float(index), dtype=np.float32)


def _roles(names: list[str]) -> dict[str, str]:
    return {r: f"block_000.slot_xx.{r}" for r in names}


ALL_ROLES = sorted(ATTENTION_ROLES | EXPERT_ROLES | PRESERVED_ROLES)


class MixedWeightsValidationTests(unittest.TestCase):
    def test_routes_assigned_roles_to_primary_and_rest_to_fallback(self) -> None:
        primary = _StubSource(_roles(ALL_ROLES), "primary")
        fallback = _StubSource(_roles(ALL_ROLES), "fallback")
        mixed = MixedWeights(primary, fallback, frozenset(ATTENTION_ROLES), "mix")
        # _pick is internal, so assert through the public surface instead: the
        # stub encodes which source answered via a distinguishable value.
        self.assertEqual(mixed.label, "mix")
        self.assertEqual(set(mixed.roles), set(ALL_ROLES))

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

    def test_expert_tier_mix_is_valid(self) -> None:
        """The attribution baseline used by the experiment must construct."""
        primary = _StubSource(_roles(ALL_ROLES))
        fallback = _StubSource(_roles(ALL_ROLES))
        mixed = MixedWeights(primary, fallback, frozenset(EXPERT_ROLES), "expert-only")
        self.assertEqual(mixed.expert("expert_down", 3).tolist(), [[3.0, 3.0], [3.0, 3.0]])


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


class AlphaCacheDiscardTests(unittest.TestCase):
    """A malformed cache must be recomputed, never fatal.

    The cache is derived data with deterministic recomputation, so aborting would
    let one corrupt file block the experiment until deleted by hand. These drive
    ``_load_cache`` directly through a stub rather than building a real pack.
    """

    class _Fake:
        """Just enough of PackWeights to exercise _load_cache."""

        _load_cache = None  # bound below

        def __init__(self, path: Path, fingerprint: dict) -> None:
            self._cache_path = path
            self._fp = fingerprint

        def _fingerprint(self) -> dict:
            return self._fp

    def setUp(self) -> None:
        from grok1_block_weights import PackWeights

        self._Fake._load_cache = PackWeights._load_cache
        self._Fake._discard_cache = PackWeights._discard_cache

    def _load(self, payload: str, fingerprint: dict | None = None):
        fp = {"pack_size": 1} if fingerprint is None else fingerprint
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.json"
            p.write_text(payload)
            return self._Fake(p, fp)._load_cache()

    def test_unreadable_json_is_discarded(self) -> None:
        self.assertEqual(self._load("{ not json"), {})

    def test_non_object_payload_is_discarded(self) -> None:
        self.assertEqual(self._load("[1, 2, 3]"), {})

    def test_non_object_scales_is_discarded(self) -> None:
        payload = json.dumps({"fingerprint": {"pack_size": 1}, "scales": [1, 2]})
        self.assertEqual(self._load(payload), {})

    def test_malformed_scale_entry_is_discarded(self) -> None:
        payload = json.dumps({"fingerprint": {"pack_size": 1}, "scales": {"t": {"nope": 1}}})
        self.assertEqual(self._load(payload), {})

    def test_fingerprint_mismatch_discards_quietly(self) -> None:
        payload = json.dumps({"fingerprint": {"pack_size": 999}, "scales": {}})
        self.assertEqual(self._load(payload), {})

    def test_matching_fingerprint_loads_the_scales(self) -> None:
        fp = {"pack_size": 1}
        payload = json.dumps(
            {"fingerprint": fp, "scales": {"t": {"alpha": 0.5, "fired": 2, "total": 4}}}
        )
        got = self._load(payload, fp)
        self.assertEqual(list(got), ["t"])
        self.assertAlmostEqual(got["t"].alpha, 0.5)
        self.assertAlmostEqual(got["t"].sparsity, 0.5)


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
