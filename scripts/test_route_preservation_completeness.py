#!/usr/bin/env python3
"""Unit tests for the pack completeness gate (GH #107).

`.claude/rules/goz1-pipeline.md` is explicit that V2 fail-closed **cannot**
detect under-packing:

    ⚠ V2 fail-closed does not detect under-packing. It rejects an *input name
    that matches no rule* -- a misclassification guard. A tensor that is simply
    **absent** from the npy directory produces no name to match, so V2 is
    silent about it and the pack comes out short.

The compensating check is `_validate_ternary_inventory` /
`_validate_preserve_inventory` here: they compare pack contents against the
xai-dissect conversion manifest. Until GH #107 neither was called by any test,
so the guard standing between a short pack and a published "certified" result
was itself unguarded.

These tests need no pack and no mounted xai-dissect run: the validators take
plain dicts, so the contract is testable everywhere.
"""

from __future__ import annotations

import argparse
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import route_preservation_metrics as rpm  # noqa: E402
from route_preservation_measure import GATE_FAILURE_EXIT  # noqa: E402

BLOCK = 0
MODE = "attention_plus_expert"

# Kind -> tier, mirroring EXPORT_MODES/PRESERVE_KINDS. Ternary kinds for
# attention_plus_expert are the mode's kinds minus the preserve set.
TERNARY_KINDS = (
    "attn_proj_i8.model_width",
    "attn_proj_i8.narrow",
    "moe_expert.down",
    "moe_expert.gate",
    "moe_expert.up",
)
PRESERVE_KINDS_USED = ("block_norm", "router")


def _name(kind: str, slot: int, block: int = BLOCK) -> str:
    return f"block_{block:03d}.slot_{slot:02d}.{kind}"


def _manifest_tensor(kind: str, slot: int, block: int = BLOCK) -> dict:
    return {"structural_name": _name(kind, slot, block), "block": block, "kind": kind}


def _index_entry(tensor_type: int) -> dict:
    return {"tensor_type": tensor_type}


def _args(block: int = BLOCK, mode: str = MODE) -> argparse.Namespace:
    return argparse.Namespace(block=block, mode=mode)


def _full_fixture() -> tuple[list[dict], dict[str, dict]]:
    """A conversion manifest and a matching pack index (nothing missing)."""
    manifest: list[dict] = []
    index: dict[str, dict] = {}
    for slot, kind in enumerate(TERNARY_KINDS):
        manifest.append(_manifest_tensor(kind, slot))
        index[_name(kind, slot)] = _index_entry(rpm.TENSOR_TERNARY)
    for offset, kind in enumerate(PRESERVE_KINDS_USED):
        slot = 7 + offset
        manifest.append(_manifest_tensor(kind, slot))
        index[_name(kind, slot)] = _index_entry(rpm.TENSOR_F16)
    return manifest, index


class TernaryInventoryTests(unittest.TestCase):
    def test_matching_inventory_passes(self) -> None:
        manifest, index = _full_fixture()
        ternary, _ = rpm._split_tiers(index)
        rpm._validate_ternary_inventory(_args(), ternary, manifest)

    def test_missing_ternary_tensor_is_rejected(self) -> None:
        """The under-packing case fail-closed cannot see."""
        manifest, index = _full_fixture()
        dropped = _name(TERNARY_KINDS[0], 0)
        del index[dropped]

        ternary, _ = rpm._split_tiers(index)
        with self.assertRaises(rpm.MetricsError) as ctx:
            rpm._validate_ternary_inventory(_args(), ternary, manifest)
        msg = str(ctx.exception)
        self.assertIn("missing=", msg)
        self.assertIn(dropped, msg)

    def test_extra_ternary_tensor_is_rejected(self) -> None:
        manifest, index = _full_fixture()
        surprise = _name("attn_proj_i8.narrow", 42)
        index[surprise] = _index_entry(rpm.TENSOR_TERNARY)

        ternary, _ = rpm._split_tiers(index)
        with self.assertRaises(rpm.MetricsError) as ctx:
            rpm._validate_ternary_inventory(_args(), ternary, manifest)
        self.assertIn("extra=", str(ctx.exception))
        self.assertIn(surprise, str(ctx.exception))

    def test_tensor_demoted_to_preserve_counts_as_missing(self) -> None:
        """A ternary candidate that got preserved is under-packing too.

        The tensor is still in the pack, so a count of rows would not notice.
        Only the per-tier inventory catches it.
        """
        manifest, index = _full_fixture()
        demoted = _name(TERNARY_KINDS[1], 1)
        index[demoted] = _index_entry(rpm.TENSOR_F16)

        ternary, _ = rpm._split_tiers(index)
        with self.assertRaises(rpm.MetricsError) as ctx:
            rpm._validate_ternary_inventory(_args(), ternary, manifest)
        self.assertIn(demoted, str(ctx.exception))


class PreserveInventoryTests(unittest.TestCase):
    def test_matching_inventory_passes(self) -> None:
        manifest, index = _full_fixture()
        rpm._validate_preserve_inventory(_args(), index, manifest)

    def test_missing_preserve_tensor_is_rejected(self) -> None:
        """A dropped router or norm is the most dangerous omission."""
        manifest, index = _full_fixture()
        dropped = _name("router", 8)
        del index[dropped]

        with self.assertRaises(rpm.MetricsError) as ctx:
            rpm._validate_preserve_inventory(_args(), index, manifest)
        self.assertIn("missing=", str(ctx.exception))
        self.assertIn(dropped, str(ctx.exception))

    def test_preserve_tensor_ternarised_counts_as_missing(self) -> None:
        """A router that fell into the ternary tier is the #40 disaster.

        Fail-closed catches this when the *name* matches no rule; it cannot
        catch it when the name matched but the tier came out wrong.
        """
        manifest, index = _full_fixture()
        router = _name("router", 8)
        index[router] = _index_entry(rpm.TENSOR_TERNARY)

        with self.assertRaises(rpm.MetricsError) as ctx:
            rpm._validate_preserve_inventory(_args(), index, manifest)
        self.assertIn(router, str(ctx.exception))


class ValidatePilotArgsDispatchTests(unittest.TestCase):
    """Which check runs depends on whether a conversion manifest was supplied."""

    def test_with_manifest_uses_inventory_and_catches_absence(self) -> None:
        manifest, index = _full_fixture()
        del index[_name(TERNARY_KINDS[0], 0)]
        with self.assertRaises(rpm.MetricsError):
            rpm._validate_pilot_args(_args(), index, manifest)

    def test_without_manifest_a_short_pack_is_NOT_caught(self) -> None:
        """The gap the rules doc warns about, pinned.

        Kinds-only fallback compares the *set of kinds present*. Drop one of two
        tensors sharing a kind and the set is unchanged, so the check passes on
        a pack that is genuinely short. This is why a run without
        `--conversion-manifest` is diagnostic-only and must not be read as
        certification.
        """
        manifest, index = _full_fixture()
        # Two tensors of the same kind; removing one leaves the kind present.
        second = _name(TERNARY_KINDS[0], 90)
        index[second] = _index_entry(rpm.TENSOR_TERNARY)
        manifest.append(_manifest_tensor(TERNARY_KINDS[0], 90))
        del index[second]

        # Kinds-only: passes, because every expected kind still appears.
        rpm._validate_pilot_args(_args(), index, None)
        # Inventory-aware: rejects, because a named tensor is absent.
        with self.assertRaises(rpm.MetricsError):
            rpm._validate_pilot_args(_args(), index, manifest)


class CertificationExitTests(unittest.TestCase):
    """A diagnostic run must not look like a passing one."""

    @staticmethod
    def _all_passing_summary() -> list[dict]:
        return [
            {"name": "router_top1_agreement", "observed": 1.0, "threshold": ">= 99.0%", "status": "pass"},
            {"name": "block_output_cosine", "observed": 1.0, "threshold": ">= 0.995", "status": "pass"},
        ]

    def test_certified_run_with_all_gates_passing_exits_zero(self) -> None:
        with redirect_stdout(io.StringIO()):
            code = rpm.report_gates(self._all_passing_summary(), certified=True)
        self.assertEqual(code, 0)

    def test_uncertified_run_exits_nonzero_even_when_gates_pass(self) -> None:
        """Without a conversion manifest, passing gates are not a pass."""
        with redirect_stdout(io.StringIO()):
            code = rpm.report_gates(self._all_passing_summary(), certified=False)
        self.assertNotEqual(
            code,
            0,
            "an uncertified run must not exit 0 -- otherwise a diagnostic run "
            "is indistinguishable from a certified one",
        )
        self.assertEqual(code, GATE_FAILURE_EXIT)

    def test_uncertified_run_says_so_in_its_output(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rpm.report_gates(self._all_passing_summary(), certified=False)
        text = buf.getvalue().lower()
        self.assertTrue(
            "diagnostic" in text or "not certified" in text or "uncertified" in text,
            f"uncertified output should say so; got: {buf.getvalue()!r}",
        )


if __name__ == "__main__":
    unittest.main()
