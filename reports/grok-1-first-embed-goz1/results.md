# First real Grok-1 embedding → GOZ1 experiment

- **GitHub:** #39
- **Linear:** RM-190
- **Host:** ShipOfTheseus
- **Date:** 2026-07-31
- **Agent:** Grok Build: Grok 4.5 (high)

## Target

| Field | Value |
|-------|--------|
| Logical | `embedding.slot_00.token_embedding` |
| Physical pickle | `/home/raulmc/.models/xai-grok-1/ckpt-0/tensor00000_000` |
| Export npy | `/home/raulmc/.models/xai-grok-1/export-npy/embedding__slot_00__token_embedding.npy` |
| Shape / dtype | `(131072, 6144)` f32 |
| Payload size (exact) | `131072 × 6144 × 4 = 3_221_225_472` bytes = **3.0 GiB** f32 (+ small NPY header on disk) |
| Manifest | `dissect/grok-1/baseline.json` (V1; default `ternary_snn`) |
| Compute path | CPU `quantizer::quantize_f32` inside `run_quantization` / `quantize-goz1` (no myelin / CUDA). Same math as `LocalBackend`; stream does **not** dispatch through the `BackendKernel` trait. |

## Commands

Wall / max RSS below come from `/usr/bin/time -v` on both steps:

```bash
/usr/bin/time -v python3 scripts/export_grok1_embedding_npy.py \
  --shard /home/raulmc/.models/xai-grok-1/ckpt-0/tensor00000_000 \
  --output-dir /home/raulmc/.models/xai-grok-1/export-npy

cargo build --release --features cli --locked
/usr/bin/time -v ./target/release/grok-ozempic quantize-goz1 \
  --input-dir /home/raulmc/.models/xai-grok-1/export-npy \
  --output /home/raulmc/.models/xai-grok-1/artifacts/grok1-first-embed.goz1 \
  --manifest dissect/grok-1/baseline.json \
  --input-format npy \
  --verify
```

## Metrics

| Metric | Export | Quantize |
|--------|--------|----------|
| Wall clock | 2.07 s | 4.64 s |
| Max RSS | ~3.02 GiB (3161876 KiB) | ~5.29 GiB (5541932 KiB) |
| Output size | ~3.0 GiB npy (f32 payload 3.0 GiB + header) | **~192 MiB** GOZ1 (`201327136` bytes ≈ 192.001 MiB) |
| CLI summary | — | 1 source file, **1 ternary**, 0 fp16/preserve |
| Verify | — | GOZ1 version=1, 1 tensor header |

## Acceptance

- [x] One GOZ1 from real embedding weights
- [x] Ternary packing path (CLI: 1 ternary)
- [x] Embedding-only input directory
- [x] Metrics recorded
- [x] No CUDA/myelin

## Notes

- Compression ≈ 3.0 GiB f32 payload → ~192 MiB ternary GOZ1 (**~16×** size reduction for this tensor). Earlier “3.1 GiB / 193 MiB” labels were `ls -lh`-style rounding, not exact binary sizes.
- V2 `structural-manifest.json` not used (stream rejects V2 until #40 / RM-191).
