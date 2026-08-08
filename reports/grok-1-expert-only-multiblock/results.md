# Expert-only ternary multi-block residual fidelity

**Agent:** Grok Build: Grok 4.5 (xAI) · Model: grok-4.5 · Issue: #68 / Linear RM-255 · beads goz-vvgm5z
**Design:** Grok Build super-research design · **Baseline measurement (#64):** Claude Code: Fable 5
**Issue:** GH [#68](https://github.com/rmems/grok-ozempic/issues/68) / Linear RM-255 · beads `goz-vvgm5z`
**Predecessor:** PR [#64](https://github.com/rmems/grok-ozempic/pull/64) / #61 / RM-249
**Implementation commit:** `abb232ea0f4b3436072e679c690976704dfd7bb9`

## Decision

**Option 3 — Expert tier needs higher precision than single-scale ternary for multi-block (material residual / routing degradation across the chain).**

Rationale:

- `block_output_cosine sequence=['0.963572', '0.945765', '0.884956', '0.839144']`
- `residual_in_drift sequence=['0.000000', '0.277351', '0.341935', '0.498308']`
- `router_top1 sequence=['1.000000', '0.887695', '0.666504', '0.528320']`
- `router_top2 sequence=['1.000000', '0.680664', '0.548828', '0.289551']`
- `expert_load_js_bits sequence=['0.000000', '0.005559', '0.009347', '0.024023']`
- `compounding_heuristic=roughly_linear`
- `end_block_output_cosine=0.839144`
- `end_residual_in_drift=0.4983081493750823`
- `#64 block0 expert-only block_output_cosine baseline=0.963572`

### Why not the other options

- **Option 1 (viable multi-block):** rejected — top-1 falls to 0.53 and residual_in drift reaches ~0.50 by block 3; not bounded/non-compounding.
- **Option 2 (correction mechanism):** not selected as the primary decision — residual-driven routing collapse is large; a residual feedback / scale-refresh / occasional HP expert block may still help but does not by itself reclassify single-scale expert ternary as multi-block safe.
- **Option 3 (higher expert precision):** **selected** — see headline.
- **Option 4 (inconclusive):** rejected — roles resolved, v3 pack-only scales used, FP16 control passes all four blocks.

## #64 baseline (block 0 only — cite, not re-proved)

Source: `reports/grok-1-full-block-forward/` (PR #64; Claude Fable 5). Expert-only ternary:

| Metric | Value |
|--------|------:|
| block-output cosine | 0.963572 |
| residual-stream cosine | 1.000000 |
| residual drift | 0.000000 |
| router top-1 / top-2 | 1.000000 / 1.000000 |
| MoE-output cosine | 0.773483 |

Routing is free within one block under expert-only ternary; this report measures **cross-block residual accumulation** on chain 0→1→2→3. Block-0 expert-only block-output cosine in this run matched #64 to six digits (**0.963572**) under GOZ1 v3 pack-only scales — the multi-block result is not a single-block re-measurement artefact.

## Method

- Sequential chain with **paired residual trajectories** (pilot residual carries prior expert error).
- Experts ternary from GOZ1 **v3 pack-only** scales/τ; attention + routers + norms from f32 reference (`MixedWeights`).
- Block 0 seed: embedding rows × `EMBEDDING_MULTIPLIER` (78.383…).
- No Gaussian; no embedding rows for b≠0; abort if any ternary scale is `legacy_oracle`.
- Tokens: 2048, seed 20260806.

## Per-block metrics (expert-only vs FP reference)

| block | block_out cos | resid_in drift | top-1 | top-2 | JS bits | MoE-out cos |
|------:|--------------:|---------------:|------:|------:|--------:|------------:|
| 0 | 0.963572 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.773483 |
| 1 | 0.945765 | 0.277351 | 0.887695 | 0.680664 | 0.005559 | 0.676099 |
| 2 | 0.884956 | 0.341935 | 0.666504 | 0.548828 | 0.009347 | 0.555468 |
| 3 | 0.839144 | 0.498308 | 0.528320 | 0.289551 | 0.024023 | 0.468450 |

### FP16 control (harness check)

| block | block_out cos | top-1 | top-2 |
|------:|--------------:|------:|------:|
| 0 | 0.999987 | 0.997070 | 0.999023 |
| 1 | 0.999986 | 0.999512 | 0.999023 |
| 2 | 0.999968 | 0.999512 | 0.999512 |
| 3 | 0.999920 | 0.999512 | 0.998047 |

## Provenance

See `metrics.json` for pack SHA-256, per-tensor scales, gif_threshold / threshold_abs, and `scale_sources` (must be `pack_v2` for every ternary expert).

## Non-goals

Full 64-block generation, attention/router/norm ternaryization, #59 proxy matrix, CUDA/Myelin, new SAAQ formula, re-proving #64 single-block routing.

