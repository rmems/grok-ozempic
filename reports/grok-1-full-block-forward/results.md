# Genuine Grok-1 block-0 forward: route-preservation results

**Issue:** GH [#61](https://github.com/rmems/grok-ozempic/issues/61) / Linear RM-249
**Agent:** Claude Code: Fable 5 (xhigh)
**Date:** 2026-08-06

Replaces PR #57's single-projection routing proxy with the real block-0 body:
pre-attention RMSNorm → grouped-query attention (RoPE, tanh soft-cap) → output
projection → residual add → pre-MoE RMSNorm → preserved router → top-2 GeGLU
expert forward. All weights come from the real `ckpt-0` checkpoint; routers and
all four norms are preserved in every pack.

**Conclusion: option 3 — attention should move to a higher-precision tier.**
Scoped precisely: *only* attention. See [Decision](#decision).

---

## 1. Architectural tensor roles — resolved

This was #61's blocking unknown (also filed upstream as `rmems/xai-dissect#52`).
It is now resolved from authoritative source — `github.com/xai-org/grok-1`
`model.py` and `run.py` — not inferred from shapes alone.

**Mechanism:** `ckpt-0` flattens parameters in **alphabetical order of the Haiku
parameter path**, not module creation order. Within one `DecoderLayer` the paths
sort as `moe_layer/… < multi_head_attention/… < rms_norm{,_1,_2,_3} < router`,
which reproduces observed slots 00–11 exactly.

| Slot | Shape | Haiku param | Architectural role |
|------|-------|-------------|--------------------|
| 00 | (8, 6144, 32768) | `linear` | expert **GELU** branch |
| 01 | (8, 32768, 6144) | `linear_1` | expert down projection |
| 02 | (8, 6144, 32768) | `linear_v` | expert un-activated branch |
| 03 | (6144, 1024) | `key` | **K** |
| 04 | (6144, 6144) | `linear` | **attention output projection** |
| 05 | (6144, 6144) | `query` | **Q** |
| 06 | (6144, 1024) | `value` | **V** |
| 07 | (6144,) | `rms_norm` | pre-attention norm |
| 08 | (6144,) | `rms_norm_1` | **post-attention** norm (applied to attn output *before* the residual add) |
| 09 | (6144,) | `rms_norm_2` | **pre-MoE norm — the router input** |
| 10 | (6144,) | `rms_norm_3` | post-MoE norm |
| 11 | (6144, 8) | `router/w` | router |

### Five independent confirmations

1. **Named in source.** `MoELayer._inference_call` literally indexes
   `params["linear"]`, `params["linear_1"]`, `params["linear_v"]`. Sorted, those
   are gate/down/up, matching the observed expert shapes.
2. **Unnamed output projection.** `MultiHeadAttention` names `"query"`, `"key"`,
   `"value"` explicitly and leaves the final projection **unnamed**, so Haiku
   defaults it to `linear`. Sorted: `key < linear < query < value` →
   `narrow, wide, wide, narrow` = slots 03–06. **Creation order would give
   `wide, narrow, narrow, wide`, which does not match** — this rules out the
   competing hypothesis outright.
3. **Norm call order.** The four `layer_norm(...)` call sites in
   `DecoderLayer.__call__` create `rms_norm`, `rms_norm_1`, `rms_norm_2`,
   `rms_norm_3`; alphabetical order coincides with call order.
4. **Tensor-parallel grouped scales — physical checkpoint evidence.** Only
   slot_04 and slot_01 carry `groups=8`, i.e. a sharded *contracting* axis. That
   is the signature of an output/down projection; Q/K/V and gate/up shard their
   *output* axis (`groups=1`). This is independent of naming entirely.
5. **Shape self-consistency.** `48*128 == 6144` (query, output),
   `8*128 == 1024` (key, value), `ffn_size(6144, 8) == 32768`.

### This settles PR #57's 20-point norm ambiguity — unfavourably

PR #57 could not decide between `(post=08, pre=09)` and `(post=09, pre=08)`,
a ~20-point swing in block-0 top-1. **The correct assignment is
`(post=08, pre=09)`** — the *less* favourable branch. PR #57's more optimistic
residual-inclusive figures (90.65% / 96.97%) rested on the wrong ordering and
should not be carried forward.

Separately, **slot_05 is `query`**, whose output never reaches the router. PR
#57's measurement of slot_05 through the router was never an architecturally
meaningful path.

### Exact kernel semantics recovered

| Element | Value |
|---------|-------|
| RMSNorm | `scale * x * rsqrt(mean(x²) + 1e-5)`, fp32, **plain multiply** (scale inits to 0 but is *not* used as `1+scale`) |
| Attention scale | `1/sqrt(128)` = `0.08838834764831845` |
| Soft cap | `logits = 30 * tanh(logits / 30)`, applied **before** masking |
| Softmax | always fp32 |
| RoPE | half-split (NeoX), base 10000, applied to **Q and K only** |
| GQA grouping | `reshape(b, t, kv_h, h//kv_h, d)` — **contiguous** (q heads 0–5 share kv head 0) |
| Expert FFN | GeGLU: `down(gelu(slot_00(h)) * slot_02(h))` |
| Router | `h @ w` fp32, softmax over **all 8** experts |
| Top-k | `k=2`; gates are the **raw** softmax probs, *not* renormalized over the selected pair |
| GELU | **tanh** approximation — `jax.nn.gelu` defaults to `approximate=True`, unlike PyTorch |

⚠️ **Trap:** xai-dissect labels slot_00 `gate` and slot_02 `up`, but the **GELU
sits on slot_00**. This is the opposite of the usual SwiGLU naming intuition, and
swapping the branches is shape-legal — a silent correctness bug. Asserted in
tests.

---

## 2. Methodological correction: the embedding multiplier

An early run produced `‖attn_delta‖ / ‖stream‖ = 72.3`, which is architecturally
impossible — a sublayer cannot be 72× the residual stream it feeds. Cause: I had
omitted `Transformer.__call__`'s `input_embeddings *= embedding_multiplier_scale`
(`model.py:1237`, value `78.38367176906169`).

RMSNorm makes the *pre-attention* path scale-invariant, which is why PR #57's
norm-only proxy was largely unaffected. **The residual add is not
scale-invariant** — `h = h0 + rmsnorm(attn_out)` compares the two magnitudes
directly. Raw embedding rows have RMS `0.0127436`, and
`0.0127436 × 78.38367 = 0.998890`: the multiplier exists precisely to place the
residual stream at unit RMS, commensurate with post-norm sublayer deltas.

With it applied, `‖attn_delta‖/‖stream‖ = 0.876` (reference) and all 8 experts
are exercised instead of 5. Regression-tested in
`scripts/test_grok1_block_forward.py`.

A secondary finding: RMSNorm's `eps=1e-5` is **not** negligible at raw embedding
magnitudes (`mean(x²) ≈ 1.6e-4`), so scale-invariance breaks by ~3% there. Prior
notes claiming exact invariance were approximate.

---

## 3. Headline results (2048 tokens)

Reference: all 8 experts used; expert load
`[0.132, 0.102, 0.166, 0.126, 0.114, 0.127, 0.128, 0.105]`;
`‖attn_delta‖/‖stream‖ = 0.876`; `‖moe_delta‖/‖stream‖ = 0.483`;
**router margin median = 0.00617**.

| Metric | fp16 control | expert-ternary only | attention-ternary only | full pack (τ=0.4/0.9) |
|--------|-------------:|--------------------:|-----------------------:|----------------------:|
| attention-output cosine | 1.000000 | 1.000000 | 0.694002 | 0.694002 |
| residual-stream cosine | 1.000000 | 1.000000 | 0.852064 | 0.852037 |
| normalized MoE-input cosine | 1.000000 | 1.000000 | 0.901552 | 0.901539 |
| router-logit cosine | 1.000000 | 1.000000 | 0.991957 | 0.991956 |
| MoE-output cosine | 0.999981 | 0.773483 | 0.744791 | 0.531836 |
| block-output cosine | 0.999987 | 0.963572 | 0.806453 | 0.772300 |
| **router top-1 agreement** | **0.997070** | **1.000000** | **0.551270** | **0.550781** |
| **router top-2 set agreement** | **0.999023** | **1.000000** | **0.304199** | **0.303711** |
| expert-load JS (bits) | 0.000001 | 0.000000 | 0.108955 | 0.108996 |
| max per-expert load delta | 0.000488 | 0.000000 | 0.111816 | 0.111816 |
| residual drift (rel. norm) | 0.000215 | 0.000000 | 0.528990 | 0.529035 |

### The FP16 control validates the harness

top-1 **99.71%**, top-2 **99.90%**, all cosines ≥ 0.99998. The control **passes**
the run3 gates (99% / 99.5%) while GOZ1 fails them by ~44 points. The
implementation itself introduces no meaningful route drift, so the observed
damage is attributable to ternary weights, not to the measurement.

### Expert ternarization is *exactly* free for routing

`expert-ternary only` gives top-1 and top-2 agreement of **precisely 1.000000**
and residual drift of **precisely 0.000000**. This is structural, not luck:
within one block the router reads `rmsnorm(h0 + attn_out)`, which is entirely
upstream of the experts. Confirming it by measurement rather than argument also
guards the wiring — a harness that leaked expert error into the router would show
it here.

Parameter accounting for block 0's ternary-eligible tensors:

| Tier | Parameters | Share | Routing damage |
|------|-----------:|------:|----------------|
| Experts (slots 00–02) | 4,831,838,208 | **98.21%** | **none — exactly zero** |
| Attention (slots 03–06) | 88,080,384 | 1.79% | **all of it** |

**98.2% of the block's parameters can be ternarized with zero routing damage.**
The entire route-preservation problem lives in the 1.8% attention tier.

Expert ternarization is not free for *output* fidelity: MoE-output cosine 0.773
and block-output cosine 0.964 (vs a 0.995 gate). That is a separate,
downstream-accumulation question, not a routing question.

---

## 4. τ sweep — the gates are not reachable by tuning

Attention tier only, experts held at f32 reference (justified above), 2048 tokens.

| τ | mean sparsity | attn-out cosine | MoE-input cosine | **top-1** | **top-2** | block-out cosine |
|---|--------------:|----------------:|-----------------:|----------:|----------:|-----------------:|
| 0.1 | 0.085 | 0.6592 | 0.8701 | 0.5059 | 0.2778 | 0.7908 |
| 0.2 | 0.169 | 0.6684 | 0.8806 | 0.5068 | 0.2695 | 0.7889 |
| 0.4 | 0.329 | 0.6940 | 0.9016 | 0.5513 | 0.3042 | 0.8065 |
| **0.6** | 0.475 | 0.7304 | **0.9168** | **0.5664** | **0.3081** | 0.8172 |
| 0.8 | 0.601 | 0.7858 | 0.9040 | 0.5244 | 0.2710 | 0.8497 |
| 1.0 | 0.705 | 0.8404 | 0.8903 | 0.4932 | 0.2480 | 0.8818 |
| 1.2 | 0.787 | 0.8899 | 0.8935 | 0.4663 | 0.1792 | 0.8981 |
| 1.6 | 0.896 | 0.8994 | 0.8246 | 0.4893 | 0.1484 | 0.9173 |

The MoE-input cosine peaks at **τ=0.6 (0.9168)** — precisely where top-1 peaks —
then falls, while attention-output cosine keeps climbing. The router's own input
is the predictive quantity; the projection output is not.

**Best achievable top-1 ≈ 56.6% at τ ≈ 0.6; best top-2 ≈ 30.8%.** Required:
99% and 99.5%. The gap is ~42 and ~69 points, and top-1 varies only over
46–57% across the whole sweep. This is not a tuning problem.

Note the τ=0.8 peak seen at 512 tokens moved to τ=0.6 at 2048 tokens — the peak
is shallow and partly sample noise, which reinforces that no τ is materially
better.

### Cosine is actively anti-correlated with routing at high τ

Attention-output cosine rises monotonically with τ (0.659 → 0.899) while top-2
agreement **falls** (0.278 → 0.148). The best-reconstructing setting is the
worst-routing one.

The metric that *does* track routing is the **normalized MoE-input cosine** —
the router's actual input — which peaks and then falls in step with agreement.
Practical consequence: **a weight- or projection-output-cosine gate is unsafe for
routing decisions.** This sharpens PR #57's "cosine does not track routing" into
"cosine can point the opposite way."

---

## 5. Flip stratification by reference router margin

| Reference margin | tokens | top-1 flips | flip rate |
|------------------|-------:|------------:|----------:|
| [0.00, 0.01) | 1173 | 706 | 0.6019 |
| [0.01, 0.05) | 443 | 153 | 0.3454 |
| [0.05, 0.15) | 346 | 58 | 0.1676 |
| [0.15, 0.50) | 86 | 3 | 0.0349 |
| [0.50, 1.01) | 0 | 0 | — |

Damage concentrates on near-ties — but is **not confined** to them: a 16.8% flip
rate persists at margins of 0.05–0.15, and 3.5% even at 0.15–0.50. So the failure
cannot be dismissed as "only unavoidable coin-flips."

Important context: block-0 routing is **inherently near-degenerate** — median
reference margin 0.00617, and 57% of tokens sit below 0.01. A router that is
this close to tied is intrinsically fragile, which is *why* modest residual
perturbation produces large flip counts. This is a property of Grok-1 block 0,
not of GOZ1, and it may well differ at blocks 8/28/60/63 (out of scope here).

---

## 6. Format gap found: GOZ1 stores no ternary scale

`quantize_f32` computes `rms` and `threshold` per tensor
(`src/core/quantizer.rs`), but the GOZ1 container persists **neither** — a
ternary tensor header carries only name, shape, type, and offset. **A GOZ1 pack
cannot be dequantized from its own contents.**

This experiment therefore reconstructs with the **least-squares optimal
`alpha = Σ|w|(fired) / count(fired)` derived from the original f32 weights** — an
oracle no runtime has. Every number above is consequently a **best case**; a real
runtime picking a scale without the original weights would do no better.

Also confirmed: `oz.gif_threshold` in pack metadata reads `0.05` for a pack built
with per-tensor τ of 0.4/0.9 — the #51 trap. Applied τ was verified from measured
sparsity instead (0.329 ≈ τ0.4, 0.601 ≈ τ0.8 for a Gaussian).

Follow-ups worth filing: persist a per-tensor scale in the GOZ1 header, and fix
or drop `oz.gif_threshold`.

---

## 7. Reproduction

```bash
# 1. Export block 0 to f32 npy (18.3 GiB; int8 x bf16 -> f32 dequant)
python3 scripts/export_grok1_int8_npy.py \
  --conversion-manifest "$RUN3/conversion-manifest.json" \
  --block 0 --mode attention_plus_expert \
  --output-dir ~/.models/xai-grok-1/export-npy/goz53-block000-attn

# 2. Main experiment: reference, fp16 control, full pack, tier attribution
python3 scripts/grok1_block0_experiment.py \
  --npy-dir ~/.models/xai-grok-1/export-npy/goz53-block000-attn \
  --pack ~/.models/xai-grok-1/artifacts/block-pilot/block_000-attention_plus_expert.goz1 \
  --embedding-shard ~/.models/xai-grok-1/export-npy/embedding__slot_00__token_embedding.npy \
  --tokens 2048 --out reports/grok-1-full-block-forward

# 3. Attention-tier tau sweep (attention-only packs; experts from reference)
python3 scripts/grok1_block0_tau_sweep.py \
  --npy-dir ~/.models/xai-grok-1/export-npy/goz53-block000-attn \
  --attn-npy-dir ~/.models/xai-grok-1/export-npy/goz61-block000-attn-only \
  --pack-dir ~/.models/xai-grok-1/artifacts/block0-forward-tau \
  --embedding-shard ~/.models/xai-grok-1/export-npy/embedding__slot_00__token_embedding.npy \
  --tokens 2048 --out reports/grok-1-full-block-forward

# 4. Kernel tests (36)
python3 -m unittest scripts.test_grok1_block_forward
```

Cost: reference 9 s, fp16 control 28 s, pack 33 s, tier attribution 50 s at 2048
tokens; peak RSS **4.98 GiB** (expert tensors are streamed one at a time, never
materialized whole). Oracle-alpha computation is a one-time ~50 s per pack,
cached to `<pack>.oracle-alpha.json`.

Machine-readable output: `block0-forward-metrics.json`,
`block0-forward-tau-sweep.json`.

### Token selection

2048 deterministic token ids, seed `20260806`, sampled uniformly without
replacement from the 131072-token vocabulary, sorted ascending. A uniform draw
covers the embedding distribution far more evenly than natural text (which
concentrates on a few thousand frequent tokens), so it is a broader routing
probe — but it is **not** a natural sequence, so absolute attention statistics
should be read with that caveat. No tokenizer is required, and the choice is
fully reproducible from the seed.

---

## 8. Decision

**Option 3 — attention should move to a higher-precision tier.**

Evidence:

- Best achievable top-1 is **56.6%** against a 99% gate; top-2 **30.8%** against
  99.5%. Across τ ∈ [0.1, 1.6] top-1 never exceeds 57%.
- The reconstruction already uses the **least-squares optimal oracle scale**, so
  the shortfall is not a scale-selection artifact — it is the information limit
  of single-scale ternary. PR #57's per-output-channel scales reached only
  0.896 cosine, consistent with the ~0.90 analytic ceiling.
- The FP16 control passes the same gates comfortably (99.71% / 99.90%), so the
  target is achievable at higher precision.

Why not option 2 (correction mechanism): a correction would have to close ~42
points of top-1. The two cheapest candidates are already excluded — optimal
per-tensor scale is in use, and per-channel scaling was measured in #57 at ≈0.896
cosine. Option 2 is not *disproven* — a learned or residual-compensating
correction was not tested here and is a legitimate follow-up — but nothing in
this evidence suggests it can cover that gap.

Why not option 1: a 44-point top-1 deficit and 69-point top-2 deficit are not
viable for block-matrix testing.

Why not option 4: every named architectural element was resolved from
authoritative source with five independent confirmations (§1).

### Scope of the recommendation — this is the actionable part

The recommendation applies to the **attention tier only (1.79% of block
parameters)**. The expert tier — **98.21% of parameters** — preserves routing
*exactly* and should remain ternary. The right policy change is narrow:

> promote slots 03–06 (Q, K, V, output projection) to a higher-precision tier;
> keep slots 00–02 (experts) ternary.

The open question this leaves is not routing but **output fidelity**:
expert-only ternary still gives block-output cosine 0.964 against a 0.995 gate,
and how that accumulates across 64 blocks is unmeasured here (explicitly out of
scope). That is the natural next experiment.

---

## 9. Limitations

- **Block 0 only** (per scope). Block 0 is special: its residual stream is the
  raw embedding, and its router is unusually close to tied (median margin
  0.00617). Later blocks may behave differently.
- Token ids are a uniform vocabulary sample, not natural text.
- Ternary reconstruction uses an oracle scale, so results are a **best case**.
- Expert-tier effects on routing are exactly zero *within* one block by
  construction; across blocks they would enter via the residual stream, which is
  not measured here.
- No full-model inference, no other blocks, no CUDA/Myelin — all out of scope.
