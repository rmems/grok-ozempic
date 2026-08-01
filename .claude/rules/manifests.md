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

Example of a **real** V1 vs V2 string mismatch (router preserve):

| Layer | Example name |
|-------|----------------|
| V2 structural preserve | `block_*.slot_11.router` |
| V1 baseline preserve (different convention) | `blk.*.moe_gate.weight` / `blk.*.expert_router.weight` |
| NPY file → logical name | stem `embedding__slot_00__token_embedding` → logical `embedding.slot_00.token_embedding` (`__` → `.`); that logical name **is** a V2 ternary candidate, not a preserve mismatch |

If the stream classifies checkpoint/logical names against the wrong convention’s rule list, preserve entries never match → default `ternary_snn` wins incorrectly. That is what #40’s name bridge must prevent.

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
