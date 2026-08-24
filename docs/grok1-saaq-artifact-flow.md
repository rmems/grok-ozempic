# Grok-1 metadata and first real-weight GOZ1 runbook

SAAQ means **Spiking Adaptive Activity Quantization**, terminology coined by
this project's creator. This runbook connects its metadata validation path to
the first verified real Grok-1 embedding pack. The paths are complementary, but
they produce different artifacts: the SAAQ commands write structural metadata,
while `quantize-goz1` reads exported tensor values and writes a GOZ1 container.

## 1. Build and choose portable locations

From the repository root:

```bash
cargo build --release --features cli

GROK1_CKPT="${GROK1_CKPT:-$HOME/.models/xai-grok-1/ckpt-0}"
GROK1_NPY="${GROK1_NPY:-$HOME/.models/xai-grok-1/export-npy}"
GROK1_ARTIFACTS="${GROK1_ARTIFACTS:-$HOME/.models/xai-grok-1/artifacts}"
mkdir -p "$GROK1_NPY" "$GROK1_ARTIFACTS"
```

`GROK1_CKPT` must point to a local official `ckpt-0` directory. The measured
checkpoint used for the first experiment contained 770 shards and occupied
about 297 GiB, but `validate-ingest` does not establish those counts. GitHub
#35 owns exact local checkpoint inventory validation.

## 2. Validate ingest metadata

```bash
./target/release/grok-ozempic validate-ingest \
  --manifest dissect/grok-1/baseline.json \
  --checkpoint "$GROK1_CKPT"
```

This checks manifest identity and schema, checkpoint-directory presence, and
the entries in `checksums.json` when that optional file exists. If present,
those entries cause real shard reads for hashing. It does not independently
count all checkpoint shards or quantize weights.

## 3. Run the metadata-only smoke and conversion gates

```bash
./target/release/grok-ozempic smoke-grok1 \
  --manifest dissect/grok-1/baseline.json \
  --block 0 \
  --include-embedding true \
  --include-final-norm true \
  --output-root /tmp/grok1-smoke \
  --dry-run

./target/release/grok-ozempic convert-grok1 \
  --manifest dissect/grok-1/baseline.json \
  --output-root /tmp/grok1-artifact \
  --format saaq-g1-v0 \
  --protect-routers true \
  --protect-norms true \
  --dry-run

./target/release/grok-ozempic validate-grok1-artifact \
  --manifest dissect/grok-1/baseline.json \
  --artifact-index /tmp/grok1-artifact/artifact.index.json \
  --checksums /tmp/grok1-artifact/checksums.json \
  --output-root /tmp/grok1-validation \
  --strict-router-protection true
```

The smoke output is a block-0 structural slice. The conversion output includes
`artifact.index.json`, `conversion.summary.md`, `checksums.json`,
`manifest.used.json`, and `warnings.json`. The final command validates that
metadata contract, including protected routers and norms. It still does not
write packed tensor payloads. GitHub #36 owns the complete 770-entry,
64-router metadata execution gate.

## 4. Export the real embedding from pickle to NPY

The official shard is a JAX pickle frame, which `quantize-goz1` cannot consume
directly. The known embedding payload is `tensor00000_000`, offset 151, f32,
shape `(131072, 6144)`:

```bash
python3 scripts/export_grok1_embedding_npy.py \
  --shard "$GROK1_CKPT/tensor00000_000" \
  --output-dir "$GROK1_NPY"
```

The command reads the real approximately 3 GiB payload and writes
`$GROK1_NPY/embedding__slot_00__token_embedding.npy`. The NPY loader converts
the stem's `__` separators to the logical name
`embedding.slot_00.token_embedding`. Safetensors checkpoints can instead be
passed directly to the real-weight packer with `--input-format safetensors`;
the official pickle shard cannot.

## 5. Pack and verify a real GOZ1 artifact

```bash
./target/release/grok-ozempic quantize-goz1 \
  --input-dir "$GROK1_NPY" \
  --output "$GROK1_ARTIFACTS/grok1-first-embed.goz1" \
  --manifest dissect/grok-1/structural-manifest.json \
  --input-format npy \
  --verify
```

This is the first command in the runbook that quantizes and packs weight
values. Current writes use GOZ1 layout version 3, with per-tensor
reconstruction scale, GIF threshold, applied absolute threshold, and the row
sentinel documented in [`goz1-format.md`](./goz1-format.md). The V2 structural
manifest path fails closed when an input name has no explicit rule, protecting
against silent router/norm misclassification.

The original measured experiment used the then-current baseline manifest and
GOZ1 version 1; it produced a verified 192.00 MiB embedding artifact. See the
canonical [`results.md`](../reports/grok-1-first-embed-goz1/results.md) for its
immutable command, hashes, compression, and trit histogram. A new v3 pack is a
reproduction of the workflow, not a promise of byte identity with that v1 file.

## 6. Inspect the result and understand the boundary

`--verify` reopens the container and validates its header, tensor table,
payload bounds, scale/threshold fields, and row sentinel. For a histogram using
the repository's independent parser:

```bash
python3 scripts/goz1_trit_histogram.py \
  "$GROK1_ARTIFACTS/grok1-first-embed.goz1"
```

The first embedding experiment proves only export → ternary pack → container
verification for one tensor. It does not run full inference or establish
routing or model-quality preservation.

## Advanced multi-block evidence

GitHub #85 and PR #89 extend the research to blocks 0–3 at the locked 8192-token
Grok-1 context. The canonical entry point is the surviving out-of-process
[`grok1_multiblock_v4_supervisor.py`](../scripts/grok1_multiblock_v4_supervisor.py),
not three manually launched experiment commands. It runs one high-memory child
at a time and fails closed if an arm cannot publish valid evidence. Its
approximately 64 MiB intra-expert chunks bound side-table construction without
splitting an expert row.

Read the canonical
[`results.md`](../reports/grok-1-expert-precision-remedy-v4/results.md) and
[`metrics.json`](../reports/grok-1-expert-precision-remedy-v4/metrics.json)
rather than copying their decision values into this runbook. The experiment's
persisted INT4 code/scale side tables are research caches; they are not the GOZ1
container format and must not be presented as deployable GOZ1 artifacts.

## Non-goals

This runbook does not claim completion of #35 or #36, run full-model inference,
quantize routers or norms, provision cloud resources, or rerun the multi-hour
#85 experiment.
