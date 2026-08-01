# xai-dissect Manifest — schema v1

Machine-readable JSON contract that lets `grok-ozempic` consume structural
analysis produced by the upstream [`xai-dissect`](https://github.com/rmems/xai-dissect)
repository, without depending on it as a runtime crate.

This document freezes **schema v1**. The loader and runtime resolution path in
`stream::resolve_manifest` are active; V2 structural naming is still rejected
for runtime GOZ1 packs until #40 / RM-191.

## Authority and source of truth

- **`xai-dissect` is authoritative.** Manifests are produced there.
- The `dissect/grok-1/baseline.json` file committed in this repo is a
  **non-authoritative fallback / reference copy** regenerated from
  `xai-dissect` releases. Do not treat it as ground truth.
- `grok-ozempic` **never writes manifests** and never depends on
  `xai-dissect` as a Cargo crate.

## Delivery

Runtime resolution is implemented in `stream::resolve_manifest` (first hit wins):

1. Explicit `QuantizationConfig.manifest_path` / CLI `--manifest` (caller-provided).
2. Nonempty UTF-8 environment variable `GROK_OZEMPIC_MANIFEST`.
3. Embedded Grok-1 baseline **only if** `use_embedded_baseline` /
   `--use-embedded-baseline` is set (**opt-in**; default off). The in-tree
   `dissect/grok-1/baseline.json` is a reference copy and is **not** loaded
   automatically just because it exists on disk.
4. Otherwise `None` → legacy `router_patterns` substring heuristic in selection.

**V2 structural naming** (`block_{NNN}.slot_{SS}.{kind}`) parses for alignment /
dry-run but is **rejected** by `resolve_manifest` for runtime GOZ1 packs until
GitHub #40 / RM-191 (checkpoint↔structural name bridge).

## Manifest precedence over legacy `router_patterns`

When a manifest is resolved (path, env, or opt-in embedded baseline), it
**wins** over the legacy `QuantizationConfig.router_patterns` substring list.
If both are present, a deprecation warning may be logged. The legacy field
remains supported only when no manifest is resolved.

## Schema v1

```json
{
  "schema": "xai-dissect.manifest",
  "schema_version": 1,
  "model": {
    "family": "grok-1",
    "source": "xai-org/grok-1",
    "tensor_name_convention": "blk.{L}.{role}.weight"
  },
  "produced_by": {
    "tool": "xai-dissect",
    "version": "0.x.y",
    "commit": "optional-sha"
  },
  "defaults": {
    "precision": "ternary_snn",
    "gif_threshold": 0.05
  },
  "preserve": [
    { "name": "blk.*.attn_router.weight", "reason": "routing-critical" }
  ],
  "fp16": [
    { "name": "token_embd.weight", "reason": "embedding table" }
  ],
  "ternary_candidates": [
    { "name": "blk.0.ffn_up.weight",   "rank": 0.98, "gif_threshold": 0.04 },
    { "name": "blk.0.ffn_down.weight", "rank": 0.95 }
  ],
  "blocks": [
    { "index": 0, "experts": 8, "role": "moe" }
  ]
}
```

### Resolution order inside a manifest

`preserve` > `fp16` > `ternary_candidates` > `defaults`.

### Name matching

- Exact tensor names or simple globs where `*` matches **exactly one**
  dotted segment (e.g. `blk.*.attn_router.weight` matches
  `blk.0.attn_router.weight` but **not**
  `blk.0.sub.attn_router.weight`).
- Matching is anchored at dotted segments. The pattern and the tensor
  name must have the same segment count; each segment must equal `*` or
  match the literal.
- **No regular expressions in v1.**
- `gate` substring matches are intentionally not performed; this is why
  globs must be segment-anchored. Historical false positives like
  `ffn_gate` being swept up by a `gate` substring are impossible in v1.

### Precision tiers (v1)

| Tier           | Meaning                                                 |
| -------------- | ------------------------------------------------------- |
| `preserve`     | Routing-critical / no-touch tier. **GOZ1 v1: FP16-at-rest** — emits the same on-disk bytes as `fp16` (both use `TENSOR_F16`). Kept as a distinct tier to carry manifest intent (which list claimed the tensor) and to leave room for a future GOZ1 format version that may introduce true source-dtype passthrough. This is the **final, documented behavior** for GOZ1 v1, not a transitional shortcut. |
| `fp16`         | Force FP16 passthrough (current router behavior).       |
| `ternary_snn` | Ternary {-1, 0, +1} with GIF saliency threshold.         |

### Per-tensor fields

- `rank` — optional hint in `[0, 1]`. Higher = stronger ternary candidate.
- `gif_threshold` — optional per-tensor override of the global threshold.
- `reason` — free-form human string, ignored by the loader.

## Hard-fail validation

### Loader (`load_manifest` / parse)

Reject with typed errors rather than best-effort parse:

- `schema_version` other than `1`.
- `model.tensor_name_convention` other than V1 `"blk.{L}.{role}.weight"` **or**
  V2 `"block_{NNN}.slot_{SS}.{kind}"` (both parse; unknown conventions fail).
- Non-existent / unreadable manifest file.
- Malformed JSON / invalid precision strings.

Unknown top-level fields are **tolerated** for forward compatibility.

### Runtime GOZ1 stream (`resolve_manifest`)

Even when a V2 manifest **parses**, `stream::resolve_manifest` **rejects** V2
for live `quantize-goz1` / `run_quantization` until #40. Alignment and dry-run
may still load embedded V2 fixtures via other entry points (`alignment.rs`).

## Versioning

Future schema versions bump `schema_version`. Loaders must refuse any
version they do not explicitly understand. This prevents silent drift
between `xai-dissect` and `grok-ozempic`.
