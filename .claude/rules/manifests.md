# Manifests (V1 vs V2)

## Files in-tree

| File | Convention | Runtime `quantize-goz1` / `stream::resolve_manifest` |
|------|------------|------------------------------------------------------|
| `dissect/grok-1/baseline.json` | V1 | **Accepted** today |
| `dissect/grok-1/structural-manifest.json` | V2 `block_{NNN}.slot_{SS}.{kind}` | **Rejected** until #40 |

V2 is valid for **alignment / dry-run** (`src/core/alignment.rs`, embedded structural fixture). `stream::resolve_manifest` hard-errors on `MANIFEST_NAME_CONVENTION_V2` until checkpoint↔structural name translation (or structural-named inputs) is wired — GitHub **#40** / Linear **RM-191**.

## Classification order

Inside a loaded manifest: **preserve > fp16 > ternary_candidates > defaults**.

Routers, norms, and other preserve rules must **never** fall into default ternary because of a name mismatch. That is the core risk #40 closes for structural names like `embedding.slot_00.token_embedding` vs V2 `embedding.slot_00.token_embedding` / block patterns.

## Authority

- **`xai-dissect` is authoritative** for manifests. In-tree copies are reference fallbacks.
- This crate **never writes** manifests and must not invent schema fields.
- Delivery precedence: explicit path → `GROK_OZEMPIC_MANIFEST` → embedded baseline → legacy router substring heuristic.

See `docs/dissect-manifest.md` and `src/core/manifest.rs`.
