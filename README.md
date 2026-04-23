# grok-ozempic

SNN-inspired **ternary quantization** for Grok-scale MoE weights: keep routing sharp in **FP16**, pack the rest into **2-bit {-1, 0, +1}** using a saliency (GIF-style) threshold. Think “Ozempic for Grok” — less bulk, same routing story.

**[Grok-1](https://huggingface.co/xai-org/grok-1)** ships as **JAX / Flax** weights; this crate does **not** load checkpoints directly. Export tensors to **`.npy`** (recommended for JAX) or **safetensors**, then run the batch pipeline. **Pickle** is not supported.

The batch pass is **out-of-core** (one tensor at a time in RAM). Output is our own **GOZ1** packed file (magic `GOZ1`, see [`src/core/weight_pack.rs`](src/core/weight_pack.rs)) — **not GGUF**.

## Features

- **GOZ1 writer** — streaming packed checkpoint: metadata + tensor table + payloads (`PackStreamWriter` in `weight_pack`).
- **Verifier** — `verify_pack_file` (`weight_pack_read`) checks layout and offsets.
- **Inputs:** mmap **`*.safetensors`** or flat **`.npy`** (`QuantizationInputFormat::NpyDir`). Filename mapping: `blk__0__weight.npy` → `blk.0.weight`. **Fortran-order `.npy` is rejected.**
- **Manifest-driven mixed precision** — three tiers (`preserve`, `fp16`, `ternary_snn`) selected per tensor by an [xai-dissect](https://github.com/rmems/xai-dissect) manifest (schema v1). Segment-anchored globs, per-tensor GIF-threshold overrides, hard-fail validation. See [`docs/dissect-manifest.md`](docs/dissect-manifest.md). Legacy `router_patterns` substring heuristic remains as the manifest-less fallback.
- **Hybrid runtime:** `HybridModel`, `Projector`, `OzempicMoE`; default hidden **6144** (`GROK1_HIDDEN_DIM`). Load gates via `OzempicMoE::load_gates_from_fp16_stacked_experts`.

## xai-dissect manifest

Upstream [`xai-dissect`](https://github.com/rmems/xai-dissect) is the authoritative source of truth for tensor classification; `grok-ozempic` consumes the JSON manifest but never depends on `xai-dissect` as a Cargo crate.

Resolution order inside `run_quantization` (first hit wins; errors bubble):

1. `QuantizationConfig.manifest_path` — explicit caller override.
2. `GROK_OZEMPIC_MANIFEST` environment variable.
3. Embedded Grok-1 baseline (`dissect/grok-1/baseline.json`), **only when** `QuantizationConfig.use_embedded_baseline = true` (opt-in, preserves legacy behavior by default).
4. Legacy `router_patterns` substring heuristic.

Precision tiers in GOZ1 v1:

| Tier          | On-disk                                                        |
| ------------- | -------------------------------------------------------------- |
| `preserve`    | Routing-critical / no-touch. **FP16-at-rest** in GOZ1 v1 (same bytes as `fp16`; kept distinct for manifest-intent traceability and forward compatibility with a future format version). |
| `fp16`        | FP16 passthrough for router tensors.                           |
| `ternary_snn` | 2-bit ternary {-1, 0, +1} with GIF saliency threshold.         |

## Build

```bash
cargo build --release
cargo test
```

Rust **2024** edition. Trailing **`[workspace]`** in `Cargo.toml` helps when this crate lives under another workspace.

## Usage sketch

- **Batch:** [`QuantizationConfig`](src/types.rs) + [`run_quantization`](src/core/stream.rs) → GOZ1 file at `output_path`.
- **Check:** [`verify_pack_file`](src/core/weight_pack_read.rs).
- **Runtime:** [`HybridConfig`](src/types.rs) + [`HybridModel`](src/core/mod.rs).
- **Manifest:** point `QuantizationConfig.manifest_path` at an [xai-dissect](https://github.com/rmems/xai-dissect) schema-v1 JSON, set `GROK_OZEMPIC_MANIFEST`, or flip `use_embedded_baseline = true` for the in-tree Grok-1 reference.

## Repository

**https://github.com/Spikenaut/grok-ozempic**

## License

GPL-3.0-or-later (see `Cargo.toml`).
