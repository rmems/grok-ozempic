#!/usr/bin/env python3
"""Canonical JSON serialization and the committed-artifact guard (GH #108)."""

from __future__ import annotations

import json
import subprocess  # nosec B404 - fixed argv, shell disabled
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from json_canonical import canonical_json  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


class CanonicalFormTests(unittest.TestCase):
    def test_keys_are_sorted_regardless_of_insertion_order(self) -> None:
        a = canonical_json({"b": 1, "a": 2})
        b = canonical_json({"a": 2, "b": 1})
        self.assertEqual(a, b, "output must depend on content, not insertion order")
        self.assertLess(a.index('"a"'), a.index('"b"'))

    def test_is_idempotent(self) -> None:
        payload = {"z": [3, 1, 2], "a": {"y": 1, "x": 2}}
        once = canonical_json(payload)
        twice = canonical_json(json.loads(once))
        self.assertEqual(once, twice)

    def test_ends_with_exactly_one_newline(self) -> None:
        out = canonical_json({"a": 1})
        self.assertTrue(out.endswith("}\n"))
        self.assertFalse(out.endswith("\n\n"))

    def test_nested_keys_are_sorted_too(self) -> None:
        out = canonical_json({"outer": {"b": 1, "a": 2}})
        self.assertLess(out.index('"a"'), out.index('"b"'))


class CommittedArtifactsAreCanonicalTests(unittest.TestCase):
    """Every tracked JSON under reports/ must already be canonical.

    Before GH #108 three writers disagreed on `sort_keys`, so the same logical
    payload serialized to different bytes depending on which one produced it —
    and 20 of 24 committed artifacts were in the unsorted form. This test is the
    guard that stops that returning: a writer that bypasses `canonical_json`
    fails here the moment its output is committed.
    """

    @staticmethod
    def _tracked_report_json() -> list[Path]:
        out = subprocess.run(  # nosec B603 - fixed argv, no shell
            ["git", "ls-files", "reports/*.json", "reports/**/*.json"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return [REPO_ROOT / line for line in out.stdout.split() if line]

    def test_every_tracked_report_json_is_canonical(self) -> None:
        files = self._tracked_report_json()
        if not files:
            self.skipTest("no tracked reports/*.json (not a git checkout?)")

        offenders = []
        for path in files:
            raw = path.read_text(encoding="utf-8")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: unparseable ({exc})")
                continue
            if canonical_json(parsed) != raw:
                offenders.append(str(path.relative_to(REPO_ROOT)))

        self.assertEqual(
            offenders,
            [],
            "these committed artifacts are not in canonical form; regenerate them "
            "with canonical_json (GH #108):\n  " + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
