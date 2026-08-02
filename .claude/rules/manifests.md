# Manifests (V1 vs V2)

## Files in-tree

| File | Convention | Runtime `quantize-goz1` / `stream::resolve_manifest` |
|------|------------|------------------------------------------------------|
| `dissect/grok-1/baseline.json` | V1 | **Accepted** (defaults fallthrough allowed) |
| `dissect/grok-1/structural-manifest.json` | V2 `block_{NNN}.slot_{SS}.{kind}` | **Accepted** (#40 / RM-191) — fail-closed |

V2 requires **structural-named inputs** (export-script npy stems, `__` → `.`). Under a V2 manifest, a tensor matching no explicit rule is a **hard error** (`ManifestV2UnmatchedTensor`) — never a `defaults` fallthrough. That is the #40 guarantee: routers/norms cannot be silently ternary-quantized by a name-convention mismatch. V2 also remains valid for **alignment / dry-run** (`src/core/alignment.rs`, embedded structural fixture).

## Classification order

Inside a loaded manifest: **preserve > fp16 > ternary_candidates > defaults**.

Routers, norms, and other preserve rules must not fall into default ternary because of a name mismatch. That is the core risk #40 closes.

Example of a **real** V1 vs V2 string mismatch (router preserve):

| Layer | Example name |
|-------|----------------|
| V2 structural preserve | `block_*.slot_11.router` |
| V1 baseline preserve (different convention) | `blk.*.moe_gate.weight` / `blk.*.expert_router.weight` |
| NPY file → logical name | stem `embedding__slot_00__token_embedding` → logical `embedding.slot_00.token_embedding` (`__` → `.`); that logical name **is** a V2 ternary candidate, not a preserve mismatch |

If the stream classifies checkpoint/logical names against the wrong convention’s rule list, preserve entries never match. Under V1 that silently defaults to `ternary_snn`; under V2 the run **aborts** on the first unmatched tensor (`ManifestV2UnmatchedTensor`) — the #40 name-bridge guarantee.

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
