#!/usr/bin/env python3
"""One canonical JSON serialization for every artifact this repo writes (GH #108).

Three `_atomic_write_json` implementations existed --
`grok1_multiblock_lib.py`, `grok1_multiblock_experiment.py` and
`grok1_multiblock_v4_supervisor.py` -- with identical `indent`, `separators`
and `ensure_ascii`, differing in exactly one flag: the supervisor passed
`sort_keys=True` and the other two did not.

So the same logical payload serialized to *different bytes* depending on which
module happened to write it, in a pipeline whose provenance and certification
compare and hash JSON artifacts. The `progress-*.json` files were the clearest
symptom: written by both the supervisor (sorted) and the experiment (unsorted),
last writer wins, so the committed copies were unsorted even though the
supervisor had set `sort_keys=True` since its first commit.

Only the *serialization* is shared here. Each caller keeps its own durability
mechanics -- they differ in strictness but all are correct, and two of them are
monkeypatched by name in the side-table tests, so consolidating them is a
separate change with a different risk profile.
"""

from __future__ import annotations

import json

__all__ = ["canonical_json"]


def canonical_json(payload: object) -> str:
    """Serialize `payload` the one way this repo writes JSON.

    `sort_keys=True` is the load-bearing part: it makes output depend only on
    content, not on dict insertion order, so a hash or a diff of an artifact
    means something regardless of which writer produced it.

    The trailing newline is included so callers can `write()` the result
    directly and every artifact ends POSIX-clean.
    """
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
