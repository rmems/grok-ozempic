# Manifests (V1 vs V2)

## Files in-tree

| File | Convention | Runtime `quantize-goz1` / `stream::resolve_manifest` |
|------|------------|------------------------------------------------------|
| `dissect/grok-1/baseline.json` | V1 | **Accepted** today |
| `dissect/grok-1/structural-manifest.json` | V2 `block_{NNN}.slot_{SS}.{kind}` | **Rejected** until #40 |

V2 is valid for **alignment / dry-run** (`src/core/alignment.rs`, embedded structural fixture). `stream::resolve_manifest` hard-errors on `MANIFEST_NAME_CONVENTION_V2` until checkpoint↔structural name translation (or structural-named inputs) is wired — GitHub **#40** / Linear **RM-191**.

## Classification order

Inside a loaded manifest: **preserve > fp16 > ternary_candidates > defaults**.

Routers, norms, and other preserve rules must not fall into default ternary because of a name mismatch. That is the core risk #40 closes.

Example mismatch (not the same string):

| Layer | Example name |
|-------|----------------|
| NPY stem → logical (export) | `embedding.slot_00.token_embedding` |
| V2 structural manifest | often `block_*` / slot patterns; embedding may appear under structural rules, not V1 `blk.*` strings |
| V1 baseline | `blk.{L}.{role}.weight`-style / empty `ternary_candidates` + default ternary |

If the stream classifies with V1 names against a V2 rule list (or the reverse), preserve entries never match → default `ternary_snn` wins incorrectly.

## Authority

- **`xai-dissect` is authoritative** for manifests. In-tree copies are reference fallbacks.
- This crate **never writes** manifests and must not invent schema fields.
- Delivery precedence in `stream::resolve_manifest`:
  1. Explicit `manifest_path` / CLI `--manifest`
  2. Nonempty `GROK_OZEMPIC_MANIFEST` env
  3. Embedded Grok-1 baseline **only if** `use_embedded_baseline` / `--use-embedded-baseline` is set (opt-in; default off)
  4. Else `None` → legacy `router_patterns` substring heuristic in selection

Do not assume the in-tree `baseline.json` is active unless a path/env was supplied or the opt-in flag is on. The `quantize-goz1` recipes pass `--manifest` explicitly.

See `docs/dissect-manifest.md` and `src/core/stream.rs` / `manifest.rs`.
