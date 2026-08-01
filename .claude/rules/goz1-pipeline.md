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

**NPY directory** (JAX export path):

```bash
cargo run --release --features cli --locked -- quantize-goz1 \
  --input-dir /path/to/npy-dir \
  --output /path/to/out.goz1 \
  --manifest dissect/grok-1/baseline.json \
  --input-format npy \
  --verify
```

**Safetensors directory** (do not use `--input-format npy`):

```bash
cargo run --release --features cli --locked -- quantize-goz1 \
  --input-dir /path/to/safetensors-dir \
  --output /path/to/out.goz1 \
  --manifest dissect/grok-1/baseline.json \
  --input-format safetensors \
  --verify
```

Until #40 / RM-191 lands, **runtime** packs use V1 `baseline.json` (default ternary). Prefer `--verify`.

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
