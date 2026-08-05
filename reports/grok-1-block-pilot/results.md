# Grok-1 bounded block pilot — run3 planning surface → GOZ1 → route preservation

- **GitHub:** #53
- **Linear:** RM-222
- **Beads:** `goz-4ic2` (pilot), `goz-dus0` (int8 dequant export)
- **Host:** ShipOfTheseus (Linux 7.1.4-200.fc44, 32 cores, 60 GiB RAM, 201 GiB free on `/home`)
- **Date:** 2026-08-05
- **Agent:** Claude Code: Fable 5 (xhigh)
- **Crate version:** 0.2.0

## Question

xai-dissect **run3** ships a planning surface (`quant-plan`, `pilot-selection-plan`,
`route-preservation-report`) whose runtime gates are all `status: unknown` —
xai-dissect defines the gates but does not execute a quantization runtime. Can
`grok-ozempic` pack a real Grok-1 block under that plan, and what are the
observed route-preservation numbers?

**Answer: yes to the pack, no to the gates.** Both pilots pack cleanly with the
policy honored exactly, and every route-preservation threshold **fails** by a
wide margin. The failure is not a tuning miss — it is the information-theoretic
ceiling of single-scale ternary (§ "Why the gate is unreachable").

Routing is measured against **real block-0 activations** — rows of the actual
token embedding through this block's own RMSNorm gain, which is exactly what
block 0 sees at inference (§ "Measurement scope"). An earlier revision of this
report used synthetic Gaussian activations; that understated routing damage and
inverted one conclusion about expert load. Every routing number below is from the
real checkpoint.

## Blocker cleared first: int8 → f32 dequant export (`goz-dus0`)

Official `ckpt-0` ships every attention projection and MoE expert as
`__main__.QuantizedWeight8bit`, and the float stream rejects int8
(`SourceDtype::Other`). New: **`scripts/export_grok1_int8_npy.py`**.

Observed on-disk layout (recovered with `pickletools.genops`; the shard is
**never unpickled**, so no `STACK_GLOBAL` target is ever imported or called):

```text
__main__.QuantizedWeight8bit
  .weight  ndarray int8       ( *lead, K, N )
  .scales  ndarray bfloat16   ( *lead, G, N )    K % G == 0
```

Scales are **grouped along the contracting axis** `K`: group `g` covers rows
`[g·K/G, (g+1)·K/G)`. `G` is the tensor-parallel shard count of that axis in the
original 8-way checkpoint — `G = 8` where `K` was sharded, `G = 1` where the
output axis `N` was sharded instead.

```text
w_f32 = weight.reshape(*lead, G, K//G, N) * scales[..., :, None, :]
```

Block 0 layout, as parsed:

| structural name | weight | scales | G |
|---|---|---|---|
| `block_000.slot_00.moe_expert.gate` | int8 `(8, 6144, 32768)` | bf16 `(8, 1, 32768)` | 1 |
| `block_000.slot_01.moe_expert.down` | int8 `(8, 32768, 6144)` | bf16 `(8, 8, 6144)` | 8 |
| `block_000.slot_02.moe_expert.up` | int8 `(8, 6144, 32768)` | bf16 `(8, 1, 32768)` | 1 |
| `block_000.slot_03.attn_proj_i8.narrow` | int8 `(6144, 1024)` | bf16 `(1, 1024)` | 1 |
| `block_000.slot_04.attn_proj_i8.model_width` | int8 `(6144, 6144)` | bf16 `(8, 6144)` | 8 |
| `block_000.slot_05.attn_proj_i8.model_width` | int8 `(6144, 6144)` | bf16 `(1, 6144)` | 1 |
| `block_000.slot_06.attn_proj_i8.narrow` | int8 `(6144, 1024)` | bf16 `(1, 1024)` | 1 |
| `block_000.slot_07..10.block_norm` | f32 `(6144,)` | — | passthrough |
| `block_000.slot_11.router` | f32 `(6144, 8)` | — | passthrough |

**Correctness evidence.** The dequant was validated **bit-exact** against
ground truth obtained by genuinely unpickling three shards with a stub
`QuantizedWeight8bit` and `ml_dtypes` — covering the grouped (`G=8`),
ungrouped (`G=1`) and f32-passthrough cases:

```text
tensor00005_000: int8(6144, 1024) x bfloat16(1, 1024)   -> bit-exact=True
tensor00006_000: int8(6144, 6144) x bfloat16(8, 6144)   -> bit-exact=True
tensor00013_000: plain float32(6144, 8)                 -> bit-exact=True
tensor00009_000: plain float32(6144,)                   -> bit-exact=True
```

55 unit tests (`scripts/test_export_grok1_int8_npy.py`, wired into
`.github/workflows/python-scripts.yml`) cover the opcode scanner, the grouping
rule and its rejections, chunk-invariance, header alignment, mode selection, and
the fail-loud paths below. Fixtures frame bfloat16 scales exactly as the real
checkpoint does (`STACK_GLOBAL ml_dtypes bfloat16`), so they exercise the same
code path a real shard takes.

The exporter refuses rather than guesses:

| Rejected | Why it matters |
|---|---|
| scale groups that do not divide the contracting axis; disagreeing leading or output dims | a wrong broadcast produces a plausible-looking but wrong tensor |
| Fortran-ordered arrays | the writer emits C-order and must not silently transpose |
| a payload running past end-of-file | truncated shard |
| a manifest shape that differs from the shard | checked **before** any bytes are written, so a mismatch cannot leave a wrong multi-GiB file for a later pack to consume |
| `bfloat16` named as a bare dtype string, or any `STACK_GLOBAL` outside `ml_dtypes.bfloat16` | the descriptor sets the element size, so an untrusted spelling could change how many bytes each value consumes |
| unknown dtypes, unreadable/malformed manifests, non-int shape entries | surfaced as `ExportError` (clean `error:`, exit 2) rather than a traceback |

> **Dependency note.** Unlike `export_grok1_embedding_npy.py` (stdlib-only,
> byte-copy), dequantization is real arithmetic over up to 1.6e9 elements, so
> this script **requires numpy** — already a CI dependency of the Python-scripts
> workflow. No tensor is ever fully materialized, but `--chunk-mib` bounds the
> compute buffer, **not** peak RSS: peak is dominated by source-shard mmap
> residency (measured 2.03 GiB exporting all of block 0 — see the results table).
> Size hosts from that number, not from the chunk setting.

## Pack policy (run3 quant-plan, enforced not assumed)

`scripts/block_pilot_goz1.sh` derives the pilot manifest **at runtime** from the
in-tree V2 structural manifest; nothing under `dissect/` is modified and no
xai-dissect schema field is invented. The deriver **hard-fails** if a preserve
rule names a family not in run3's `keep_fp32`, or a ternary candidate names a
family not in `pilot_quantize`.

| Tier | Families | Treatment |
|---|---|---|
| preserve | `router`, `block_norm`, `final_norm` | never ternary, no τ |
| ternary | `attn_proj_i8.*` | τ = **0.4** |
| ternary | `moe_expert.*` | τ = **0.9** |
| deferred | `token_embedding` | **dropped from candidates** — a stray embedding npy hard-errors instead of packing |

τ values come from the #51 / RM-201 per-tier table (attention 0.3–0.5,
experts 0.65–1.12 for the 50–75 % event-driven band). `defaults.gif_threshold`
is stripped and per-tensor τ set instead, because defaults silently outrank CLI
`--gif-threshold` (the #51 τ trap, `src/core/precision.rs::resolve_threshold`).

## Results

Both pilots on **block 0** (pilot-selection-plan rationale: *early baseline*).

| Mode | Source npy | Pack `file_size` | ternary | fp16/preserve | export wall / RSS | pack wall / RSS |
|---|---|---|---|---|---|---|
| `attention_only` | 0.328 GiB | 22,168,640 | 4 | 5 | 0.28 s / 355 MiB | 0.61 s / 300 MiB |
| `attention_plus_expert` | 18.328 GiB | 1,230,128,448 | 7 | 5 | 16.11 s / 2.03 GiB | 31.69 s / 12.37 GiB |

Both verified: `GOZ1 verify ok: version=1, N tensor header(s), file_size=…`.

The exporter never materializes a payload: it records offsets from a bounded
opcode window and jumps past the data, so the 1.6 GB int8 expert arrays are read
only through `numpy.memmap` in `--chunk-mib` slices. Peak RSS is dominated by
mmap page residency over the source shard rather than by chunk size.

**Preserve/ternary counters match intent exactly.** In both modes the 5
preserve-tier tensors are the 4 `block_norm` vectors plus the router; **no
router or norm was ternaryized**, and no expert/attention tensor escaped
quantization.

### Measured sparsity (exact trit counts read back from the packs)

| Tensor | τ | zeros | zeros % |
|---|---|---|---|
| `slot_00.moe_expert.gate` | 0.9 | 1,051,486,556 | 65.28 |
| `slot_01.moe_expert.down` | 0.9 | 1,026,408,312 | 63.73 |
| `slot_02.moe_expert.up` | 0.9 | 1,037,831,274 | 64.44 |
| `slot_03.attn_proj_i8.narrow` | 0.4 | 2,117,334 | 33.65 |
| `slot_04.attn_proj_i8.model_width` | 0.4 | 12,596,418 | 33.37 |
| `slot_05.attn_proj_i8.model_width` | 0.4 | 12,161,834 | 32.22 |
| `slot_06.attn_proj_i8.narrow` | 0.4 | 2,047,031 | 32.54 |

Experts land in the 63–65 % band — inside #51's 50–75 % event-driven target.
Invalid `0b11` trit codes: **0** in every tensor.

> **Per-tensor τ is recorded in the histogram and route-preservation JSONs; the
> pack header's `oz.gif_threshold` only stores the pipeline-level default.** The
> pack writer records `defaults || config` in `oz.gif_threshold` (here the CLI
> default 0.05), which cannot represent per-tensor overrides. The trustworthy
> evidence is the per-tensor `effective_tau` field (0.4 for `attn_proj_i8.*`, 0.9
> for `moe_expert.*`) and the measured sparsity — 4 % zeros would indicate τ=0.05,
> while 33 %/65 % confirms 0.4/0.9.

### Route-preservation surface — `unknown` → observed

Filling run3's `route-preservation-report.json` gates for block 0. Reference =
the dequantized f32 npy; pilot = tensors **read back out of the pack** (trits for
ternary, fp16 for preserve, so the preserve tier's own round-trip error is
included). Routing metrics are worst case over the two evaluated `model_width`
projections; weight-reconstruction metrics are worst case over all quantized
tensors in the `attention_plus_expert` pack.

| Metric | Scope | Threshold | Observed | Status |
|---|---|---|---|---|
| `router_top1_agreement` | router_behavior | ≥ 99.0 % | **63.96 %** | ❌ fail |
| `router_top2_set_agreement` | router_behavior | ≥ 99.5 % | **42.11 %** | ❌ fail |
| `block_output_cosine` | block_behavior | ≥ 0.995 | **0.8492** | ❌ fail |
| `expert_load_distribution_delta` | router_behavior | — | 0.2485 | measured |
| `expert_load_js_divergence` | router_behavior | — | 0.1102 bits | measured |
| `router_logit_rank_correlation` | router_behavior | — | 0.7528 | measured |
| `block_output_rmse` | block_behavior | — | 0.7659 | measured |
| `residual_stream_drift` | block_behavior | — | 0.5341 | measured |
| `weight_reconstruction_mse` | weight_reconstruction | — | 1.426e-4 | measured |
| `weight_cosine_similarity` | weight_reconstruction | — | 0.8597 | measured |
| `weight_max_absolute_error` | weight_reconstruction | — | 0.7098 | measured |
| `per_channel_scale_error_summary` | weight_reconstruction | — | 0.7655 | measured |
| `logit_kl` | model_behavior | — | null | **unknown** |
| `perplexity_delta` | model_behavior | — | null | **unknown** |
| `generation_sanity_summary` | model_behavior | — | null | **unknown** |

12 of 15 metrics now carry observed values. The three `model_behavior` metrics
stay `unknown`: they need whole-model inference, an explicit #53 non-goal.

Per-projection routing detail (4096 real token-embedding rows, seed 20260805):

| Projection | top-1 | top-2 set | rank corr | out cosine (mean / min) | load JS |
|---|---|---|---|---|---|
| `slot_04.attn_proj_i8.model_width` | 68.63 % | 44.24 % | 0.7740 | 0.8492 / 0.7120 | 0.0689 |
| `slot_05.attn_proj_i8.model_width` | 63.96 % | 42.11 % | 0.7528 | **0.9630** / 0.9383 | 0.1102 |

Top-1 expert load (reference → pilot):

```text
slot_04  ref   [0.0308, 0.1567, 0.0308, 0.4509, 0.0125, 0.2197, 0.0894, 0.0093]
         pilot [0.0613, 0.0852, 0.0518, 0.4419, 0.0410, 0.0789, 0.1885, 0.0515]
slot_05  ref   [0.4634, 0.0000, 0.0000, 0.4487, 0.0474, 0.0383, 0.0022, 0.0000]
         pilot [0.4272, 0.0002, 0.0002, 0.2002, 0.1277, 0.2437, 0.0007, 0.0000]
```

**Cosine and routing agreement decouple — this is the load-bearing observation.**
`slot_05` reaches 0.9630 projection-output cosine, far the best of any tensor
measured, and still routes **36 % of tokens to a different expert**. Its expert-3
share collapses 44.9 % → 20.0 % while expert-5 rises 3.8 % → 24.4 %. A pipeline
that gated on output cosine alone would wave this through.

> ⚠ **Correction from the synthetic-activation draft.** An earlier revision of
> this report measured routing against seeded Gaussian activations and concluded
> that aggregate expert load "barely moves (JS ≈ 0.008 bits)" while per-token
> routing flipped — i.e. that a load-balance check would miss the damage. Real
> activations **reverse that**: load JS is 0.1102 bits (13× higher) and the worst
> per-expert share moves 0.2485 (8.5× higher), so a load-balance check *would*
> flag this pack. The Gaussian result was an artifact of isotropic activations
> spreading routing decisions evenly across all 8 experts; real embeddings
> concentrate on 2 dominant experts and leave 3 essentially unused, which makes
> the distribution far easier to disturb. Every routing number in this report is
> now from real activations.

Preserve-tier fp16 round-trip (GOZ1 v1 stores the preserve tier as fp16-at-rest,
so run3's `keep_fp32` is honored *as a tier*, not as f32 bits):

| Tensor | max abs err | relative RMSE |
|---|---|---|
| `slot_11.router` | 6.01e-05 | 2.09e-04 |
| `slot_07..10.block_norm` | 8.30e-04 … 1.93e-03 | ~2.0e-04 |

Router perturbation is ~2e-4 relative — four orders of magnitude smaller than the
routing drift, confirming the flips come from the **quantized projection input**,
not from the preserved router itself.

#### Measurement scope (read before quoting these numbers)

- **Activations are real, for the block that was piloted.** A decoder block
  computes `h = h + attn(rmsnorm(h))`, and for **block 0** `h` *is* the token
  embedding lookup. So 4096 rows sampled from the real `(131072, 6144)` f32
  embedding, pushed through this block's own `block_norm` gain, are the actual
  distribution block 0 sees at inference. No calibration corpus is needed for
  that, and the measurement is invariant to Grok-1's embedding scale multiplier
  because `rmsnorm(c·x) = rmsnorm(x)`. Rows are read via `numpy.memmap` at the
  offset the opcode scanner reports, so only sampled rows are touched.
  Two residual caveats: rows are sampled **uniformly over the vocabulary**, not
  by corpus token frequency; and this equivalence holds *only* for block 0 —
  blocks 8/28/60/63 see the residual stream after every preceding block, which
  needs a forward pass this bounded pilot does not run, so `--embedding-shard`
  hard-errors for `--block != 0` rather than quietly mislabeling synthetic rows
  as real (#59).
- **`block_output_cosine` is scoped to a single projection's output**, not a full
  block forward. xai-dissect labels the attention projections
  `attn_proj_i8.narrow` / `.model_width` with policy `wrap_existing_int8_unknown`
  — it assigns **no q/k/v/o roles**. Rather than invent a role mapping, both
  `model_width` projections are evaluated independently and both reported. MoE
  expert routing is not executed.
- **Ternary reconstruction uses the least-squares optimal scale**
  `α = Σ|w| over fired / count(fired)`. GOZ1 v1 stores no per-tensor scale, so
  these are **best-case** numbers for this container. `cos(w, α·t)` is
  independent of `α`, so the cosine figures hold for any positive scale.
- **A "channel" is a (leading index, last axis) pair**, so a 3-D expert tensor
  `(8, 6144, 32768)` has 8 × 32768 channels rather than 32768 pooled across
  experts. Pooling would understate the worst case — it read 0.6915 before this
  was corrected, against 0.7655 measured per-expert.
- **The RMSNorm gain is one arbitrary choice of four.** A block carries four
  `block_norm` vectors and upstream assigns none of them a role, so the
  lowest-numbered slot shapes the activations for *every* projection (recorded
  per result as `activation_source`). That keeps the comparison internally
  consistent; it is not a claim about the block's true activation path.
- **Summary rows are per-metric worst cases and may come from different
  tensors.** `block_output_cosine` (0.8492, `slot_04`) and `block_output_rmse`
  (0.7659, `slot_05`) are each the worst over the evaluated projections, so
  quoting two summary numbers together describes no single tensor. Per-projection
  values are preserved under `routing` in the JSON — use those when a coherent
  picture of one projection is needed.

## Why the gate is unreachable (not a tuning miss)

Sweeping τ over the attention projections — weight cosine, and router agreement
against **real** block-0 activations (numpy pass over the exported npy, 4096
embedding rows):

| τ | zeros % | w_cos `slot_04` | top-1 `slot_04` | w_cos `slot_05` | top-1 `slot_05` |
|---|---|---|---|---|---|
| 0.0 | 0.00 | 0.7674 | 61.23 % | 0.7865 | 52.22 % |
| 0.2 | ~16.5 | 0.8241 | 65.36 % | 0.8428 | 52.69 % |
| 0.4 | ~32.2 | 0.8597 | 68.63 % | 0.8785 | 63.96 % |
| **0.6** | ~46.6 | **0.8722** | 71.46 % | **0.8917** | 68.53 % |
| 0.8 | ~59.1 | 0.8621 | 72.09 % | 0.8826 | 76.07 % |
| 1.0 | ~69.6 | 0.8320 | 74.76 % | 0.8530 | 75.56 % |
| 1.2 | ~77.9 | — | 71.48 % | — | **79.81 %** ← best |
| 2.0 | ~95.2 | — | 62.79 % | — | 57.54 % |
| 3.0 | ~99.5 | — | 32.86 % | — | 50.51 % |

**Routing-optimal τ is not weight-optimal τ.** Weight cosine peaks at τ ≈ 0.6,
but router agreement keeps climbing to τ ≈ 0.8–1.2 — a *sparser* pack routes
better on real activations than the one with the best weight reconstruction.
Plausibly because a higher τ raises the surviving ternary scale α to match the
large-magnitude structure that anisotropic real activations actually excite,
while the small weights it zeroes contribute mostly noise in those directions.
Either way, picking τ by weight reconstruction optimizes the wrong objective if
routing is what you care about — and this is invisible under Gaussian
activations, where top-1 was flat at ~79 % across τ ∈ [0.4, 1.0].

The best top-1 anywhere in the extended sweep is **79.81 % at τ = 1.2**, after
which it declines to 32–50 % by τ = 3.0. The gate needs **99.0 %**.

Weight cosine peaks at **τ ≈ 0.6** and tops out near **0.89**. For reference, the
analytic ceiling for single-scale ternary on a Gaussian matrix is

```text
max_τ  E[|z|·1{|z|>τ}] / sqrt(P(|z|>τ))  =  0.899903   at τ = 0.612·σ
```

Measured 0.8722 (`slot_04`, kurtosis 8.21 — heavy-tailed, so below the Gaussian
ceiling) and 0.8917 (`slot_05`, kurtosis 3.42 — near-Gaussian, essentially at the
ceiling). The measurement matches theory, which is itself a check on the harness.

Giving every output channel its own scale — the obvious next lever, and one GOZ1
v1 cannot express — moves cosine only to **0.8877 / 0.8962** at τ = 0.6. Still
nowhere near 0.995.

**Conclusion.** For the two `model_width` attention projections evaluated here,
a ≥ 0.995 **projection-output** cosine is **not reachable with 2-bit ternary at
any τ, with or without per-channel scales**; the ceiling is ~0.90. The gate is
named `block_output_cosine` upstream, but what was measured is one projection's
output, not a full block forward — see "Measurement scope" above. That scope is
empirical — it is not a universal claim about all 2-bit ternary weight matrices.
Router agreement is bounded even harder: the best top-1 anywhere in the extended
τ sweep is **79.81 %** against a 99.0 % gate, and it *falls* on either side of
τ ≈ 1.2.

Meeting run3's route-preservation gates requires a different mechanism — a
higher-precision tier for attention, residual/error-feedback quantization, or
gate thresholds renegotiated for a spiking-sparse target. Ternary remains
appropriate where the consumer is an event-driven kernel and the gate is
compute reduction rather than bit-fidelity: the expert tier hits 63–65 % zeros
at 16× compression, which is exactly what #48 wants from the MoE FLOPs.

Two things real activations changed that the synthetic draft got wrong, both
worth carrying into #48's acceptance design:

1. **Expert-load drift is large, not negligible** (JS 0.1102 bits vs 0.008), so
   load balance *is* a usable signal here — the opposite of what the Gaussian
   run implied.
2. **Output cosine does not track routing.** `slot_05` posts 0.9630 cosine and
   still misroutes 36 % of tokens, so a cosine-only gate is not a safe proxy for
   the router-behavior gates.

## Reproduce

```bash
cargo build --release --features cli --locked

# run3 handoff (preferred over May run2); tests honor GROK_OZEMPIC_DISSECT_RUN
export GROK_OZEMPIC_DISSECT_RUN=~/rmems/grok-result/xai-dissect/LATEST_CORRECT_GROK1_RUN
RUN3="$GROK_OZEMPIC_DISSECT_RUN/manifests/xai-grok-1-ckpt-0"

# whole pilot: dequant export -> derived manifest -> pack --verify -> histogram -> metrics
BLOCK=0 MODE=attention_only          scripts/block_pilot_goz1.sh
BLOCK=0 MODE=attention_plus_expert   scripts/block_pilot_goz1.sh

# just the dequant export (any pilot block / mode)
python3 scripts/export_grok1_int8_npy.py \
  --conversion-manifest "$RUN3/conversion-manifest.json" \
  --block 0 --mode attention_plus_expert \
  --output-dir ~/.models/xai-grok-1/export-npy/block000
python3 scripts/export_grok1_int8_npy.py --inspect ~/.models/xai-grok-1/ckpt-0/tensor00006_000

# metrics against an existing pack, with REAL block-0 activations
# (--embedding-shard is block-0 only; omit it and routing falls back to
#  synthetic Gaussian rows, labelled as such in the JSON)
python3 scripts/route_preservation_metrics.py \
  --npy-dir ~/.models/xai-grok-1/export-npy/block000 \
  --pack ~/.models/xai-grok-1/artifacts/block-pilot/block_000-attention_only.goz1 \
  --embedding-shard ~/.models/xai-grok-1/ckpt-0/tensor00000_000 \
  --block 0 --mode attention_only --json-out /tmp/route-preservation.json

python3 -m unittest scripts.test_export_grok1_int8_npy -v
```

Host paths (nothing under `~/.models` is committed):

| Artifact | Path |
|---|---|
| Checkpoint | `~/.models/xai-grok-1/ckpt-0/` |
| run3 handoff | `~/rmems/grok-result/xai-dissect/LATEST_CORRECT_GROK1_RUN/manifests/xai-grok-1-ckpt-0/` (→ `grok1_run3_20260802T023050Z`) |
| Packs, derived manifests, metrics, logs | `~/.models/xai-grok-1/artifacts/block-pilot/` |
| f32 npy stage | `mktemp` dir under `~/.models/xai-grok-1/`, removed on exit (`KEEP_NPY=1` to keep) |

The τ-sweep and per-channel-ceiling tables came from one-off chunked numpy passes
over the exported npy (mirroring `quantizer.rs`: `τ = gif_threshold × rms`, then
trits, then optimal α). numpy is a host-side analysis convenience, not a pipeline
dependency — consistent with #51.

## Remaining work

Blocks 8, 28, 60 and 63, and the `expert_only` mode, were not run. The block-0
result is mode- and τ-independent in its conclusion (the ceiling is a property of
2-bit ternary, not of a block), so the remaining pilots are confirmation rather
than discovery — but they are still open against #53's full checklist.

## Relation to open work

- **#48 / RM-196 (multi-tensor epic):** the ternary ceiling above is the decisive
  input — the route-preservation gates need a mechanism change, not a τ sweep.
- **#51 / RM-201:** per-tier τ table consumed as specified; the τ trap avoided;
  `oz.gif_threshold` blind spot now demonstrated on a per-tensor manifest.
- **#40 / RM-191:** the V2 fail-closed bridge is what makes the preserve counters
  trustworthy — an unmatched name aborts instead of defaulting to ternary.
- **#50:** event-driven kernels that would cash in the 63–65 % expert sparsity
  belong to `myelin-accelerator`, not here.
