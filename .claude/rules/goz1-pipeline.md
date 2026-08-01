# GOZ1 pipeline invariants

## Two CLI paths (do not confuse them)

| Path | Commands | Reads weights? |
|------|----------|----------------|
| **SAAQ metadata** | `validate-ingest`, `smoke-grok1`, `convert-grok1`, `validate-grok1-artifact` | Index/metadata only (may hash shards if checksums present) |
| **GOZ1 real quant** | `quantize-goz1` | Yes — **safetensors or `.npy` only** |

Official Grok-1 `ckpt-0` shards are JAX **pickle**. `quantize-goz1` / `run_quantization` **reject pickle**. Export first:

```bash
python3 scripts/export_grok1_embedding_npy.py \
  --shard "$CKPT/tensor00000_000" \
  --output-dir "$OUT"
```

Stem mapping: `embedding__slot_00__token_embedding.npy` → logical name `embedding.slot_00.token_embedding` (`__` → `.`).

## Real pack recipe

```bash
cargo run --release --features cli -- quantize-goz1 \
  --input-dir /path/to/npy-or-safetensors \
  --output /path/to/out.goz1 \
  --manifest dissect/grok-1/baseline.json \
  --input-format npy \
  --verify
```

Until #40 / RM-191 lands, **runtime** packs use V1 `baseline.json` (default ternary). Prefer `--verify`. Never claim a ternary pack without CLI summary metrics (ternary count, size, wall time).

## Ownership

- GOZ1 format, streaming, manifests, Grok-1 glue → this crate.
- CUDA / ternary GEMV kernels → `myelin-accelerator` (see `kernel-boundary` rule).
