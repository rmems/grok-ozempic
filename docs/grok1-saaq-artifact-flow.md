# Grok-1 SAAQ artifact flow

This document is the sprint handoff contract for GitHub issues #13, #14,
#15, #17, #18, and #19.

## Ingest validation

Before conversion, validate the `xai-dissect` manifest and optional checkpoint
checksum map:

```bash
cargo run --features cli -- \
  validate-ingest \
  --manifest dissect/grok-1/baseline.json \
  --checkpoint /path/to/ckpt-0
```

Validation enforces:

- readable JSON manifest
- supported manifest schema version
- supported tensor name convention
- `model.family = grok-1`
- optional checkpoint directory existence
- optional `checksums.json` SHA-256 integrity when present in the checkpoint
  directory

The checksum file is a JSON object whose keys are checkpoint-relative file paths
and whose values are either raw lowercase SHA-256 hex strings or
`sha256:<hex>` strings.

## First quantization target

The first bounded target remains:

```text
embedding.slot_00.token_embedding
```

In this sprint artifact writer, that target is represented in the deterministic
artifact index with policy:

```text
candidate_saaq_embedding
```

Routers and norms are protected as pass-through f32 metadata, while existing
int8 expert and dense payloads are wrapped rather than requantized. This keeps
the first PR concrete and structurally safe without claiming full inference or
quality benchmarking.

## One-block smoke gate

Run the smoke slice before claiming a full artifact conversion:

```bash
cargo run --features cli -- \
  smoke-grok1 \
  --manifest dissect/grok-1/baseline.json \
  --block 0 \
  --include-embedding true \
  --include-final-norm true \
  --output-root /tmp/grok1-smoke \
  --dry-run
```

Expected outputs:

```text
smoke.summary.md
smoke.index.json
smoke.checksums.json
smoke.warnings.json
```

The smoke gate validates `block_000` as 12 tensors with one protected f32 router,
four protected f32 block norms, three expert tensor families, and four wrapped
unknown dense slots. Unresolved expert projection labels are warnings, not hard
failures.

## Full conversion metadata writer

Run the full deterministic metadata writer:

```bash
cargo run --features cli -- \
  convert-grok1 \
  --manifest dissect/grok-1/baseline.json \
  --output-root /tmp/grok1-artifact \
  --format saaq-g1-v0 \
  --protect-routers true \
  --protect-norms true \
  --dry-run
```

Expected outputs:

```text
manifest.used.json
conversion.summary.md
artifact.index.json
checksums.json
warnings.json
```

The writer refuses non-`grok-1` manifests, validates 770 planned tensor entries,
64 routers, protected f32 routers with shape `(6144, 8)`, protected norms,
`64 * 8 * 3` logical expert associations, deterministic offsets, and stable JSON
ordering.

## Final SAAQ-based validation gate

The final sprint gate is scriptable and grounded in the SAAQ/custom-format
artifact contract:

```bash
cargo run --features cli -- \
  validate-grok1-artifact \
  --manifest dissect/grok-1/baseline.json \
  --artifact-index /tmp/grok1-artifact/artifact.index.json \
  --checksums /tmp/grok1-artifact/checksums.json \
  --output-root /tmp/grok1-validation \
  --strict-router-protection true
```

Expected outputs:

```text
validation.summary.md
validation.report.json
validation.failures.json
validation.warnings.json
```

Pass criteria:

- source manifest is Grok-1 and schema-valid
- full artifact has exactly 770 tensor entries
- every source tensor is represented once in the artifact index
- exactly 64 routers exist
- every router is protected/pass-through/f32 with shape `(6144, 8)`
- block norms and final norm are protected/pass-through/f32
- every block has 8 experts and 3 expert tensor families
- unresolved expert projections remain warnings
- unknown dense slots remain warnings unless count/shape/dtype changes
- source byte accounting equals artifact byte accounting
- full raw source byte accounting equals `318114914304`

Stable failure categories include:

```text
missing_tensor
duplicate_tensor
shape_mismatch
dtype_mismatch
byte_mismatch
router_count_mismatch
router_policy_violation
norm_policy_violation
expert_family_missing
manifest_artifact_mismatch
```

## Non-goals

This sprint path does not run full inference, benchmark model quality, quantize
routers, perform cloud provisioning, or resolve every projection-name ambiguity.
