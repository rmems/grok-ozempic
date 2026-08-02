# GOZ1 pipeline invariants

## Two CLI paths (do not confuse them)

| Path | Commands | Reads weights? |
|------|----------|----------------|
| **SAAQ metadata** | `validate-ingest`, `smoke-grok1`, `convert-grok1`, `validate-grok1-artifact` | Index/metadata only (may hash shards if checksums present) |
| **GOZ1 real quant** | `quantize-goz1` | Yes — **safetensors or `.npy` only** |

Official Grok-1 `ckpt-0` shards are JAX **pickle**. `quantize-goz1` / `run_quantization` **reject pickle**.

### Export script scope

`scripts/export_grok1_embedding_npy.py` is **stdlib-only**. Defaults target the token embedding on one pickle shard (not a full 770-tensor pack). With `--stem` and layout flags it can export **another single f32 tensor** from a shard — still one file per invocation, not bulk export.

```bash
python3 scripts/export_grok1_embedding_npy.py \
  --shard "$CKPT/tensor00000_000" \
  --output-dir "$OUT"
```

Default stem mapping: `embedding__slot_00__token_embedding.npy` → logical `embedding.slot_00.token_embedding` (`__` → `.`).

## Real pack recipes

**NPY directory** (JAX export path; stems are structural, so prefer the V2 structural manifest):

```bash
cargo run --release --features cli --locked -- quantize-goz1 \
  --input-dir /path/to/npy-dir \
  --output /path/to/out.goz1 \
  --manifest dissect/grok-1/structural-manifest.json \
  --input-format npy \
  --verify
```

**Safetensors directory** (do not use `--input-format npy`; use the manifest whose convention matches the tensor names):

```bash
cargo run --release --features cli --locked -- quantize-goz1 \
  --input-dir /path/to/safetensors-dir \
  --output /path/to/out.goz1 \
  --manifest dissect/grok-1/structural-manifest.json \
  --input-format safetensors \
  --verify
```

Since #40 / RM-191, **runtime** packs prefer the V2 `structural-manifest.json` whenever tensor names are structural (`block_{NNN}.slot_{SS}.{kind}`, export-script stems). V2 is fail-closed: an unmatched name hard-errors instead of defaulting to ternary. For legacy `blk.*`-named inputs use V1 `baseline.json` (defaults fallthrough allowed). Prefer `--verify`. Authoritative structural names / planning surface: `~/rmems/grok-result/xai-dissect/LATEST_CORRECT_GROK1_RUN/manifests/xai-grok-1-ckpt-0/` (tests honor `GROK_OZEMPIC_DISSECT_RUN`).

**Evidence sources:**

| Metric | Where |
|--------|--------|
| Ternary / fp16-preserve counts | Pack CLI summary line (`… X ternary, Y fp16/preserve`) |
| Total GOZ1 **file_size** (bytes) | `--verify` line: `GOZ1 verify ok: … file_size=…` |
| Wall / max RSS | External: `/usr/bin/time -v`, `gtime -v`, or BSD `time -l` |

Claim ternary only when the CLI ternary counter matches expectation.

## Ownership

- GOZ1 format, streaming, manifests, Grok-1 glue → this crate.
- CUDA / ternary GEMV kernels → `myelin-accelerator` (see `kernel-boundary` rule).
