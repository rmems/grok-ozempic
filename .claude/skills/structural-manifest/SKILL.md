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
| Runtime stream `resolve_manifest` | **Rejected** until GH **#40** / Linear **RM-191** |

Constant: `MANIFEST_NAME_CONVENTION_V2 = "block_{NNN}.slot_{SS}.{kind}"` in `src/core/manifest.rs`.

## Fixture

`dissect/grok-1/structural-manifest.json` — preserve includes routers (`block_*.slot_11.router`), norms, etc.; defaults `ternary_snn` + `gif_threshold` 0.05.

## Implementation checklist (#40)

1. Map checkpoint / npy logical names ↔ structural names **or** accept structural-named inputs.
2. Stop rejecting V2 in `stream::resolve_manifest` once safe.
3. Keep V1 baseline path working.
4. Tests must prove routers/norms **preserve**, embedding can ternary under structural rules.
5. Update README/docs to prefer structural-manifest for real quant after merge.

## Do not

- Invent schema fields (xai-dissect is authoritative)
- Quietly reclassify preserve tensors as ternary
- Force structural-manifest through `quantize-goz1` before the bridge exists (expect hard error)

## Related

- Rule: `.claude/rules/manifests.md`
- Command: `/v2-bridge`
- Docs: `docs/dissect-manifest.md`
