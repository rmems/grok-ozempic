#!/usr/bin/env python3
"""Adversarial fixtures for the ordered Grok-1 input contract (GH #127)."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grok1_ordered_inputs import (  # noqa: E402
    INPUT_MODE,
    SCHEMA_VERSION,
    SEED,
    InputManifestError,
    load_ordered_windows,
    sha256_file,
    token_content_sha256,
    token_order_sha256,
)
from grok1_block0_experiment import token_ids  # noqa: E402


class OrderedInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "FIXTURE-tokenizer.model").write_bytes(b"SYNTHETIC FIXTURE TOKENIZER; NOT GROK-1\n")
        (self.root / "FIXTURE-source.txt").write_text(
            "OWNER-AUTHORED SYNTHETIC FIXTURE; NOT A COMPATIBILITY CORPUS\n", encoding="utf-8"
        )
        arrays = {
            "cal": np.array([9, 2, 9, 1], dtype=np.int32),
            "eval-a": np.array([3, 4, 3, 5], dtype=np.int64),
            "eval-b": np.array([6, 7, 8, 6], dtype=np.uint16),
        }
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "input_mode": INPUT_MODE,
            "fixture": True,
            "selection_seed": SEED,
            "window_tokens": 4,
            "tokenizer": {
                "identity": "FIXTURE-only-tokenizer-NOT-GROK-1",
                "revision": "FIXTURE-v1",
                "vocab_size": 10,
                "special_tokens_policy": "FIXTURE: none",
                "model_file": {"path": "FIXTURE-tokenizer.model"},
            },
            "source": {
                "identity": "FIXTURE-owner-authored-synthetic-text",
                "revision": "FIXTURE-v1",
                "license": "CC0-1.0 fixture",
                "normalization_version": "FIXTURE-identity-v1",
                "artifact": {"path": "FIXTURE-source.txt"},
            },
            "windows": [],
        }
        self.manifest["tokenizer"]["model_file"]["sha256"] = sha256_file(
            self.root / "FIXTURE-tokenizer.model"
        )
        self.manifest["source"]["artifact"]["sha256"] = sha256_file(self.root / "FIXTURE-source.txt")
        for index, (name, values) in enumerate(arrays.items()):
            path = self.root / f"FIXTURE-{name}.npy"
            np.save(path, values, allow_pickle=False)
            self.manifest["windows"].append(
                {
                    "id": f"FIXTURE-{name}",
                    "partition": "calibration" if index == 0 else "held_out",
                    "document_id": f"FIXTURE-document-{name}",
                    "source_range": {"start": index * 10, "end": index * 10 + 4},
                    "sequence_boundary": "independent",
                    "token_path": path.name,
                    "token_count": 4,
                    "token_file_sha256": sha256_file(path),
                    "order_sha256": token_order_sha256(values),
                    "content_sha256": token_content_sha256(values),
                }
            )

    def tearDown(self):
        self.temp.cleanup()

    def write(self, manifest=None) -> Path:
        path = self.root / "FIXTURE-manifest.json"
        path.write_text(json.dumps(manifest or self.manifest), encoding="utf-8")
        return path

    def assert_rejected(self, pattern: str, manifest=None):
        with self.assertRaisesRegex(InputManifestError, pattern):
            load_ordered_windows(self.write(manifest), allow_fixture=True)

    def test_preserves_order_and_repeated_ids_exactly(self):
        _, windows = load_ordered_windows(self.write(), allow_fixture=True)
        np.testing.assert_array_equal(windows[0].token_ids, [9, 2, 9, 1])
        self.assertFalse(windows[0].token_ids.flags.writeable)

    def test_historical_sampled_id_path_remains_sorted_without_replacement(self):
        # Regression fence: the new loader must not silently "fix" old evidence.
        np.testing.assert_array_equal(token_ids(4, 20260806, 16), [2, 5, 6, 7])

    def test_fixture_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(InputManifestError, "not production"):
            load_ordered_windows(self.write())

    def test_rejects_wrong_tokenizer_digest(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["tokenizer"]["model_file"]["sha256"] = "0" * 64
        self.assert_rejected("tokenizer.model_file digest mismatch", manifest)

    def test_rejects_wrong_source_digest(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["source"]["artifact"]["sha256"] = "0" * 64
        self.assert_rejected("source.artifact digest mismatch", manifest)

    def test_rejects_float_dtype(self):
        manifest = copy.deepcopy(self.manifest)
        path = self.root / manifest["windows"][0]["token_path"]
        np.save(path, np.array([9, 2, 9, 1], dtype=np.float32), allow_pickle=False)
        manifest["windows"][0]["token_file_sha256"] = sha256_file(path)
        self.assert_rejected("dtype must be integer", manifest)

    def test_rejects_out_of_vocabulary_id(self):
        manifest = copy.deepcopy(self.manifest)
        path = self.root / manifest["windows"][0]["token_path"]
        np.save(path, np.array([9, 2, 10, 1], dtype=np.int64), allow_pickle=False)
        manifest["windows"][0]["token_file_sha256"] = sha256_file(path)
        self.assert_rejected(r"must be in \[0, 10\)", manifest)

    def test_rejects_order_hash_mismatch_even_when_content_matches(self):
        manifest = copy.deepcopy(self.manifest)
        path = self.root / manifest["windows"][0]["token_path"]
        reordered = np.array([1, 2, 9, 9], dtype=np.int32)
        np.save(path, reordered, allow_pickle=False)
        manifest["windows"][0]["token_file_sha256"] = sha256_file(path)
        # The old multiset hash still matches; the order hash must fail.
        self.assertEqual(manifest["windows"][0]["content_sha256"], token_content_sha256(reordered))
        self.assert_rejected("order_sha256 mismatch", manifest)

    def test_rejects_content_hash_mismatch(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["windows"][0]["content_sha256"] = "f" * 64
        self.assert_rejected("content_sha256 mismatch", manifest)

    def test_rejects_overlapping_ranges_and_cross_split_leakage(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["windows"][1]["document_id"] = manifest["windows"][0]["document_id"]
        manifest["windows"][1]["source_range"] = {"start": 2, "end": 6}
        self.assert_rejected("split leakage", manifest)

    def test_rejects_insufficient_unique_text(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["windows"] = manifest["windows"][:2]
        self.assert_rejected("exactly one calibration and two held-out", manifest)

    def test_rejects_duplicate_content_across_documents(self):
        manifest = copy.deepcopy(self.manifest)
        source = self.root / manifest["windows"][0]["token_path"]
        target = self.root / manifest["windows"][2]["token_path"]
        target.write_bytes(source.read_bytes())
        for field in ("token_file_sha256", "order_sha256", "content_sha256"):
            manifest["windows"][2][field] = manifest["windows"][0][field]
        self.assert_rejected("duplicate token content", manifest)

    def test_rejects_mixed_sampled_id_mode(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["input_mode"] = "sampled_ids+ordered_text"
        self.assert_rejected("mixed/sampled evidence", manifest)


if __name__ == "__main__":
    unittest.main()
