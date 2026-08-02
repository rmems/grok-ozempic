# Grok-1 embedding τ (gif_threshold) sweep — spiking-sparse GOZ1

- **GitHub:** #51
- **Linear:** RM-201
- **Host:** ShipOfTheseus (32 cores, 60 GiB RAM)
- **Date:** 2026-08-01
- **Agent:** Claude Fable 5
- **Branch:** `claude/fable5-51-tau-sweep`

## Question

GH #39 / RM-190 packed `embedding.slot_00.token_embedding` at the default
`gif_threshold = 0.05` and got ~4.17% zeros — a **dense, sign-like** pack, not
spiking-sparse activity. Is that the weight distribution or the knob? Answer:
**the knob.** This report sweeps τ on the real embedding, measures exact trit
histograms by reading the GOZ1 packs back, and maps τ → sparsity so the next
multi-tensor run can *choose* its sparsity regime.

## Target

| Field | Value |
|-------|--------|
| Logical tensor | `embedding.slot_00.token_embedding` |
| Shape / dtype | `(131072, 6144)` f32 — 805,306,368 elements |
| Source | `~/.models/xai-grok-1/ckpt-0/tensor00000_000` (official pickle → `.npy` via `scripts/export_grok1_embedding_npy.py`) |
| Quantizer | weight-space `τ = gif_threshold × rms`; `abs(w) < τ → 0`, `w ≥ τ → +1`, `w ≤ −τ → −1` (`src/core/quantizer.rs`) |
| Compute path | CPU `quantizer::quantize_f32` via `run_quantization` / `quantize-goz1` — no myelin / CUDA |
| Measurement | `scripts/goz1_trit_histogram.py` (new; stdlib-only GOZ1 readback, exact trit counts) |
| Weight rms | 0.012758015 (f64 accumulation over all elements) |
| Tail shape | kurtosis 3.60 (Gaussian = 3.0) → near-Gaussian, slightly heavy-tailed |

## Sweep results (measured from packs)

**Notation:** columns labeled `gif_threshold` are the dimensionless CLI /
manifest multiplier (`--gif-threshold`). The actual weight-space firing
threshold is `τ = gif_threshold × rms` (for this embedding,
`rms ≈ 0.012758`, so `gif_threshold = 0.65` ⇒ `τ ≈ 0.00829`). Tables below
report the CLI multiplier unless stated otherwise.

Each row is one `quantize-goz1` run against a manifest **without**
`defaults.gif_threshold` (see trap below), `--input-format npy --verify`,
followed by exact trit counts from GOZ1 readback. GNU time 1.9 (`/usr/bin/time -v`).

| gif_threshold | zeros | zeros % | +1 % | −1 % | Gaussian pred. % | pack bytes | wall | max RSS |
|-----|------------|---------|---------|---------|---------|-------------|---------|---------|
| 0.0 | 0 | 0.0000 | 49.7377 | 50.2623 | 0.0000 | 201,327,136 | 5.73 s | 6.17 GiB |
| 0.05 | 33,567,192 | 4.1683 | 47.6529 | 48.1788 | 3.9878 | 201,327,136 | 5.96 s | 6.19 GiB |
| 0.1 | 67,043,353 | 8.3252 | 45.5760 | 46.0988 | 7.9656 | 201,327,136 | 6.17 s | 6.19 GiB |
| 0.2 | 133,281,944 | 16.5505 | 41.4683 | 41.9812 | 15.8519 | 201,327,136 | 6.62 s | 6.18 GiB |
| 0.3 | 197,917,581 | 24.5767 | 37.4624 | 37.9610 | 23.5823 | 201,327,136 | 6.65 s | 6.19 GiB |
| 0.5 | 319,850,852 | 39.7179 | 29.9150 | 30.3671 | 38.2925 | 201,327,136 | 7.14 s | 6.19 GiB |
| 0.7 | 428,572,307 | 53.2185 | 23.1931 | 23.5884 | 51.6073 | 201,327,136 | 7.01 s | 6.18 GiB |
| 1.0 | 561,416,253 | 69.7146 | 14.9950 | 15.2903 | 68.2689 | 201,327,136 | 4.84 s | 6.19 GiB |

Validation:

- **gif_threshold = 0.0 control**: exactly 0 zeros — pure sign pack
  (`w ≥ 0 → +1`), as the gate predicts. The +1/−1 split (49.74/50.26) shows a
  slight negative skew.
- **gif_threshold = 0.05 control**: 4.1683% zeros and `file_size=201327136` —
  reproduces GH #39 exactly, validating the new histogram script against known
  ground truth.
- Trit totals equal 805,306,368 and invalid `0b11` codes are 0 in every pack.
- An independent npy-side prediction (numpy, chunked
  `P(abs(w) < gif_threshold·rms)`) matches the pack-side measurement to 4
  decimal places at **all eight** gif_threshold values.
- Pack size is gif_threshold-independent by design (2 bits/trit regardless of
  value); sparsity is a *kernel-side* win (event-driven skip), not a *storage*
  win in GOZ1 v1.
- `oz.gif_threshold` pack metadata matched the CLI multiplier in every sweep run.

### gif_threshold → sparsity map (from abs(w)/rms quantiles, 1e-4 bin resolution)

| target zeros | required gif_threshold |
|--------------|------------|
| 25% | ≈ 0.31 |
| 50% | ≈ 0.65 |
| 75% | ≈ 1.12 |
| 90% | ≈ 1.63 |
| 95% | ≈ 1.97 |

The embedding is near-Gaussian, so
`zeros(gif_threshold) ≈ erf(gif_threshold/√2)` is a good first-order model
(measured runs ~0.2–1.5 pp above Gaussian; heavier-than-Gaussian center).

## ⚠ The manifest τ trap (documented + demonstrated)

Threshold precedence in `src/core/precision.rs::resolve_threshold`:

1. per-tensor `gif_threshold` on a `ternary_candidates` entry
2. **`manifest.defaults.gif_threshold`** ← `dissect/grok-1/baseline.json` bakes `0.05` here
3. `config.gif_threshold` (CLI `--gif-threshold`)

So with the stock baseline manifest, **CLI `--gif-threshold` is silently
ignored**. Demonstrated live:

```text
quantize-goz1 --manifest dissect/grok-1/baseline.json --gif-threshold 0.5 ...
→ zeros = 33,567,192 (4.1683%)   # identical, trit-for-trit, to gif_threshold=0.05
→ oz.gif_threshold = 0.05        # pack metadata records what actually applied
```

**Rule for sweeps:** strip **both** `defaults.gif_threshold` **and** any
`ternary_candidates[].gif_threshold` (per-tensor wins over defaults and CLI;
`oz.gif_threshold` metadata only records defaults||config, so it cannot detect
per-tensor overrides). The driver derives such a manifest from `baseline.json`
at runtime; nothing under `dissect/` is modified (xai-dissect stays authoritative).
**Always check `oz.gif_threshold` in the pack metadata** — it records the
effective baseline multiplier for the defaults/CLI path.
The sweep driver (`scripts/tau_sweep_embedding.sh`) **fails the run** if pack
metadata does not match the requested TAU (compared at CLI `f32` precision) or
if any invalid trit codes appear.

## Second tensor family: blocked-by-export (evidence)

`xai-dissect dissect` over official ckpt-0 shards shows **no f32/bf16 expert or
attention weights exist in the checkpoint** — the official Grok-1 release ships
them 8-bit quantized:

| Shard | Role | Dtype | Shape | Interpretation |
|-------|------|-------|-------|----------------|
| `tensor00000_000` | tensor | f32 | (131072, 6144) | token embedding (swept above) |
| `tensor00001_000` | tensor | f32 | (6144,) | norm/scale vector (preserve-tier) |
| `tensor00002_000` | quant.weight | **int8** | (8, 6144, 32768) | MoE expert up-proj, 8 experts |
| `tensor00003_000` | quant.weight | **int8** | (8, 32768, 6144) | MoE expert down-proj |
| `tensor00005_000` | quant.weight | **int8** | (6144, 1024) | attention proj (kv-width) |
| `tensor00006_000` | quant.weight | **int8** | (6144, 6144) | attention proj (full-width) |
| `tensor00013_000` | tensor | f32 | (6144, 8) | MoE router — **preserve, never ternary** |

Blockers, precisely:

1. `scripts/export_grok1_embedding_npy.py` is f32-only by design (stdlib, one
   tensor per invocation) — it cannot export int8 payloads.
2. Even with an exporter, the float-only stream path rejects int8
   (`stream.rs`: unsupported dtype; int8 arrives via artifact wrapping, not
   quantization).
3. Meaningful GIF-ternary on experts requires **dequantized** values
   (int8 × scale → f32) — a new export capability, out of scope here.

The only other exportable f32 matrix is the router, and router ternary is an
explicit non-goal (#51) — routing readout must stay preserve.

## Per-tier τ policy draft (input to #48 / RM-196)

| Tier | Recommendation | Rationale |
|------|----------------|-----------|
| Embedding | Dense sign-like is *acceptable* (gif_threshold ∈ [0, 0.05]) **unless** event-driven embedding lookup is the goal; then gif_threshold ≈ 0.65 for ~50% zeros. | Embedding rows are read once per token (lookup, not GEMV); sparsity buys little compute unless the runtime skips zero trits inside fused kernels. Sign information is the dominant signal. |
| MoE experts | Target **event-driven**: gif_threshold for 50–75% zeros (≈ 0.65–1.12 *if* expert weights are near-Gaussian — must be re-measured after dequant export; official weights are int8). | Experts dominate FLOPs and memory traffic; zero-skipping GEMV is where spiking-sparse pays. Per-expert rms (slice of the (8, …) tensor) rather than whole-tensor rms should be evaluated. |
| Attention proj | Intermediate: start gif_threshold ≈ 0.3–0.5 (25–40% zeros) and validate quality before pushing higher. | Attention is more sensitive to weight perturbation than MoE experts in most PTQ literature; be conservative until measured. |
| Router / gates | **Preserve (fp16). Never ternary. No gif_threshold.** | Routing decisions collapse under sign-only weights; explicit non-goal. |
| Norm/scale vectors | Preserve. | Tiny (6144 floats), quality-critical. |

**When is dense sign-like OK?** When the consumer is a dense ternary kernel
(sign-GEMV) or a lookup — compression is already 16× and zeros add nothing.
**When is sparsity required?** When the runtime is event-driven (skip zero
trits): then zeros% is the compute-reduction knob, and gif_threshold should be
*chosen per tier from a target sparsity*, not left at the 0.05 legacy default.
The `defaults.gif_threshold: 0.05` in `baseline.json` should be treated as a
sign-pack default, not an SNN default.

Caveats: these gif_threshold values are calibrated on the **embedding**
distribution (near-Gaussian, kurtosis 3.6). Expert/attention distributions may
differ — re-run this sweep per tier once a dequant export exists. No quality
(perplexity / downstream) claims are made here; this is distribution + packing
science only.

## Reproduce

```bash
cargo build --release --features cli --locked

# Full sweep (export → 8 packs → histograms); packs+logs under
# ~/.models/xai-grok-1/artifacts/tau-sweep/, stage cleaned up on exit
scripts/tau_sweep_embedding.sh
# or e.g.: TAUS="0.05 0.65" scripts/tau_sweep_embedding.sh

# Exact trit histogram of any GOZ1 pack
python3 scripts/goz1_trit_histogram.py PACK.goz1 [--json]
# human + JSON artifact from one analysis:
# python3 scripts/goz1_trit_histogram.py PACK.goz1 --json-out hist.json

# Trap demo (CLI gif_threshold ignored under stock baseline manifest)
target/release/grok-ozempic quantize-goz1 \
  --input-dir ~/.models/xai-grok-1/export-npy \
  --output /tmp/trap-demo.goz1 \
  --manifest dissect/grok-1/baseline.json \
  --input-format npy --gif-threshold 0.5 --verify
python3 scripts/goz1_trit_histogram.py /tmp/trap-demo.goz1   # → 4.1683%, oz.gif_threshold=0.05
```

The rms/kurtosis/gif_threshold-quantile numbers came from a one-off chunked
numpy pass over the exported npy (`np.load(..., mmap_mode="r")`; f64
accumulation for `Σw²`, `Σw⁴`; 160,000-bin histogram of `abs(w)/rms` on
`[0, 16)` for quantiles) — numpy is a host-side analysis convenience only, not
a pipeline dependency.

## Relation to open work

- **#48 / RM-196 (multi-tensor epic):** per-tier gif_threshold table above is
  the policy input; expert/attention sweeps are blocked on an int8-dequant
  export step.
- **#40 / RM-191 (V2 name bridge):** untouched here. Note the trap compounds
  with #40's risk: a preserve-name mismatch would send routers into default
  ternary *at whatever gif_threshold defaults carry* — both bugs hide in
  `defaults`.
- Packs (8 × 192 MiB) and logs remain under
  `~/.models/xai-grok-1/artifacts/tau-sweep/` — host-only, never committed.
