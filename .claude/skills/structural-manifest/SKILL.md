---
name: structural-manifest
description: V2 structural xai-dissect manifests, alignment dry-run, and the #40 stream resolve_manifest bridge. Use when working on structural-manifest.json, MANIFEST_NAME_CONVENTION_V2, or preserve-rule name matching.
---

# Structural manifest (V2) skill

## Status

| Layer | V2 structural names |
|-------|---------------------|
| Parse (`manifest.rs`) | Allowed |
| Alignment / dry-run (`alignment.rs`) | Used heavily |
| Runtime stream `resolve_manifest` | **Accepted** (GH **#40** / Linear **RM-191**) — fail-closed on unmatched names |

Constant: `MANIFEST_NAME_CONVENTION_V2 = "block_{NNN}.slot_{SS}.{kind}"` in `src/core/manifest.rs`. Under a V2 manifest, a tensor matching no explicit rule aborts with `ManifestV2UnmatchedTensor` (no `defaults` fallthrough) — inputs must be structural-named (export-script npy stems, `__` → `.`). Authoritative structural names: `~/rmems/grok-result/xai-dissect/LATEST_CORRECT_GROK1_RUN/manifests/xai-grok-1-ckpt-0/`.

## Fixture

`dissect/grok-1/structural-manifest.json` — preserve includes routers (`block_*.slot_11.router`), norms, etc.; defaults `ternary_snn` + `gif_threshold` 0.05.

## Bridge design (landed in #40)

1. Accepts structural-named inputs (export-script npy stems); no checkpoint↔structural translation table in this crate.
2. `stream::resolve_manifest` loads V2 like V1; classification is fail-closed under V2 (`ManifestV2UnmatchedTensor` on any unmatched name).
3. V1 baseline path unchanged (defaults fallthrough still allowed).
4. Tests: `v2_structural_manifest_end_to_end_npy`, `v2_manifest_fails_closed_on_unmatched_name`, `v2_manifest_accepted_via_env_var`, plus the path-gated `run3_conversion_manifest_names_fully_classified` oracle against the latest xai-dissect run.

## Do not

- Invent schema fields (xai-dissect is authoritative)
- Quietly reclassify preserve tensors as ternary
- Feed legacy `blk.*`-named inputs to a V2 manifest (expect `ManifestV2UnmatchedTensor` — use V1 `baseline.json` for those)

## Related

- Rule: `.claude/rules/manifests.md`
- Command: `/v2-bridge`
- Docs: `docs/dissect-manifest.md`
