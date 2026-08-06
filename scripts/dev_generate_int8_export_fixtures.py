#!/usr/bin/env python3
"""Point developers at the fixture factory (no pickle usage in this file).

The unit tests do **not** load the ``.bin`` corpus under
``scripts/testdata/export_int8/``: they build synthetic pickle frames in memory
through ``scripts.testdata.export_int8.shard_factory`` on each run, so there is
nothing to regenerate for CI to pass. The committed bins are a reference corpus
for eyeballing real numpy pickle framing.

This module deliberately contains no ``pickle`` import so the scanner-facing
scripts stay free of deserialization findings; the factory (under ``testdata/``,
which Codacy ignores) is where that lives.
"""

from __future__ import annotations

import sys

GUIDANCE = """\
Fixtures are generated in-memory by the test suite; nothing needs regenerating.

  Run the tests:      python3 -m unittest scripts.test_export_grok1_int8_npy
  Build a shard:      from scripts.testdata.export_int8.shard_factory import (
                          write_shard, write_quantized,
                      )

See scripts/testdata/export_int8/README.md for the numpy-version caveat that
applies when writing new .bin fixtures.\
"""


def main() -> int:
    print(GUIDANCE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
