---
name: goz1-quantize
description: Pack Grok weights into GOZ1 via quantize-goz1 / run_quantization. Use for ternary SNN packing, npy export, gif_threshold, --verify, and first-embed experiments.
---

# GOZ1 quantize skill

## When to use

- User asks to quantize, pack GOZ1, run `quantize-goz1`, or measure embedding packs
- Debugging trit packing, gif threshold, or stream pipeline
- Distinguishing metadata-only SAAQ commands from real weight streaming

## Hard constraints

1. **No pickle inputs** to `quantize-goz1` — export with `scripts/export_grok1_embedding_npy.py` first.
2. Prefer `--input-format npy` for JAX-export paths; safetensors also supported.
3. Runtime manifest: **`dissect/grok-1/baseline.json`** until #40; structural V2 is alignment-only today.
4. Use `--verify` on real packs.
5. Cloud VMs lack multi-GiB home checkpoints — code/tests only unless weights are mounted.

## Code map

| Concern | Location |
|---------|----------|
| CLI | `src/bin/grok-ozempic/quantize.rs` |
| Stream / resolve_manifest | `src/core/stream.rs` |
| Ternary quant | `src/core/quantizer.rs` |
| NPY load | `src/core/npy.rs` |
| GOZ1 format | `src/core/weight_pack.rs`, `src/core/weight_pack_read.rs` |
| Docs | `README.md`, `docs/grok1-saaq-artifact-flow.md` |

## Commands

```bash
cargo run --features cli -- quantize-goz1 --help
cargo run --release --features cli -- quantize-goz1 \
  --input-dir "$OUT" \
  --output "$ART/model.goz1" \
  --manifest dissect/grok-1/baseline.json \
  --input-format npy \
  --verify
```

## Metrics to report

Wall clock, max RSS, output bytes, exact command.

**Compute path today:** `run_quantization` → `quantizer::quantize_f32` / FP16 helpers on CPU. It does **not** go through `BackendKernel` / `LocalBackend` / `MyelinBackend`. Myelin is a future stub — do not report myelin as the backend for `quantize-goz1` runs.

**CLI summary line** looks like:
`GOZ1 written to … (N source file(s), T tensors: X ternary, Y fp16/preserve; …)`

Four numbers: source files `N`, total tensors `T`, precision **`ternary`**, precision **`fp16/preserve`** (preserve is not separate from fp16). Size and wall time are **not** in the CLI summary.
