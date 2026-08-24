# Post-publication INT4 cache-binding audit (#85)

**Agent:** Codex (OpenAI)

**Issue:** GH #85 / Linear RM-608 / beads `goz-3h3`

**Attribution:** Grok supplied only the issue-planning lock. Codex (OpenAI) performed the implementation, review remediation, cache audit, migration, validation, and publication described here.

**Audit time:** 2026-08-24T22:04:34Z

**Canonical producer commit:** `a159d89df1dbe2374b56a1c43ec2b4ac80dbe82e`

**Canonical evidence commit:** `972145ee2aabfa88cbaf3a000316b8111a62fc89`

**Final reviewed cache implementation SHA-256:** `30d08e512c1bff6ee7f53d74e4e6417dbbe7c0cf1efe6c9e961019d37ba558b2` (`scripts/grok1_multiblock_lib.py`)

## Why this audit exists

A current-head review found that the pre-hardening sidecars in external cache `gh85-v4-8192-int4-side-ce90d12` described filenames, shapes, and codec metadata but did not bind the actual q-code and scale contents into one generation. An interrupted legacy writer could therefore have exposed new q codes and a matching reference fingerprint beside an older, shape-compatible scale. The canonical run reused entries from an interrupted earlier attempt, so structural validation alone was not sufficient to dismiss that case.

The implementation now uses schema-2 sidecars. Each tensor entry binds the streamed FP32 reference fingerprint, actual int8 q-code fingerprint, actual float32 scale fingerprint, and a deterministic generation digest. Legacy or incomplete sidecars rebuild fail-closed. Loaded q and scale files must also have their exact promised on-disk dtypes.

## Procedure and immutable observations

1. Before changing the cache, Codex hashed all 56 non-lock entries. The complete scratch manifest had SHA-256 `0f79749aabddaa1071c8328f901ce52047877f082a5b1e1d0e402dee5714f024`; its 36-file q/scale subset had SHA-256 `9ce92bb77c2ff62c4dd68898572c253550554229a3bf74ed6493588aea825d28` when recorded with absolute local paths.
2. Still before mutation, Codex streamed the FP32 reference tensors and independently recomputed each absmax scale, q tensor, and LS channel-alpha scale with the production approximately 64 MiB whole-row intra-expert traversal. All 12 q tensors, 12 absmax scales, and 12 LS scales matched the cache bit-for-bit.
3. Codex then opened each block in absmax and LS mode through the schema-2 implementation under the persistent per-block cache lock. This rebuilt the legacy/unbound entries and published content-bound sidecars using the crash-safe transaction.
4. The original pre-migration q/scale manifest verified **36/36** after migration. Every scientific data payload retained exactly the same SHA-256 digest.
5. After the strict on-disk float32 review fix, Codex reopened all eight block/mode combinations through the final reviewed source. Every entry took the bound-reuse path with q dtype `int8` and scale dtype `float32`. The resulting eight schema-2 sidecars contain three complete generation bindings each.
6. A second data-manifest verification after that final-source pass also verified **36/36**. The committed relative-path manifest is `cache-binding-audit.sha256`, whose SHA-256 is `a9e35c53f23cec1a6ab781b269fc0e359eedf21f3dc0ca36ea9392ed270d0e79`.

## Result

The legacy sidecar format was insufficient, but the concrete external cache used by the canonical run did not contain the feared mixed-generation q/scale state. Its 36 scientific payload files independently matched deterministic reference recomputation before migration and remained bit-identical after schema-2 migration and final-source validation.

Therefore the canonical 8192-token metrics and locked ranking remain supported without rerunning the three-hour forward experiment: baseline remains comparator-only, P1 `_123` remains canonical rank 1, and P0 remains rank 2. This conclusion is specific to the audited cache content and does not excuse legacy/unbound caches generally; the implementation now rejects and rebuilds those caches.

The canonical JSON metrics were not edited for this audit. This Markdown file, the SHA-256 manifest, and the short addendum in `results.md` are explicitly post-publication audit records.

## Local verification

With `GH85_SIDE_ROOT` set to the external cache root:

```bash
cd "$GH85_SIDE_ROOT"
sha256sum -c /path/to/cache-binding-audit.sha256
```

Expected result: all 36 entries report `OK`.
