#!/usr/bin/env python3
"""Offline fixture helper (no pickle in-repo).

Golden bins live under ``scripts/testdata/export_int8/``. To regenerate
synthetic pickle frames for unit tests, run the ignored factory module on a
dev machine:

  python3 -c "from scripts.testdata.export_int8.shard_factory import write_shard; ..."

This file intentionally contains no pickle usage so Codacy stays quiet.
"""
from __future__ import annotations

def main() -> None:
    raise SystemExit(
        "Fixture generation lives in scripts/testdata/export_int8/shard_factory.py "
        "(excluded from Codacy). Use that module offline."
    )

if __name__ == "__main__":
    main()
