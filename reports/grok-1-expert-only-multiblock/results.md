# Expert-only ternary multi-block residual fidelity

**Agent:** Grok Build: Grok 4.5 (xAI) · Model: grok-4.5 · Issue: #68 / Linear RM-255 · beads goz-vvgm5z
**Design:** Grok Build super-research · **Baseline (#64):** Claude Fable 5
**Issue:** GH [#68](https://github.com/rmems/grok-ozempic/issues/68) / RM-255
**Predecessor:** PR [#64](https://github.com/rmems/grok-ozempic/pull/64) / #61
**Implementation commit:** `e48f996fbb609b1f65c27fb00bb58bc78b45dd30`

## Decision

**Option 3 — Expert tier needs higher precision than single-scale ternary for multi-block (material residual / routing degradation across the chain).**

Rationale:

- `block_output_cosine sequence=['0.963572', '0.945765', '0.884956', '0.839144']`
- `residual_in_drift sequence=['0.000000', '0.277351', '0.341935', '0.498308']`
- `block_output_drift sequence=['0.277351', '0.341935', '0.498308', '0.653886']`
- `router_top1 sequence=['1.000000', '0.887695', '0.666504', '0.528320']`
- `router_top2_set_agreement sequence=['1.000000', '0.680664', '0.548828', '0.289551']`
- `expert_load_js_bits sequence=['0.000000', '0.005559', '0.009347', '0.024023']`
- `compounding_heuristic=roughly_linear`
- `end_block_output_cosine=0.839144`
- `last_block_residual_in_drift=0.4983081493750823`
- `chain_exit_residual_drift=0.6538863846584987`
- `#64 block0 expert-only block_output_cosine baseline=0.963572`

### Why not the other options

- **Option 1 (not chosen):** rejected — residual and/or routing degrade beyond bounded thresholds.
- **Option 2 (not chosen):** rejected as primary — evidence favors higher expert precision first.
- **Option 3 (selected):** selected — residual-driven multi-block collapse; raise expert precision.
- **Option 4 (not chosen):** rejected — architecture, pack v3 scales, and FP16 control resolved.

## #64 baseline (block 0 only — cite, not re-proved)

Source: `reports/grok-1-full-block-forward/` (PR #64). Block-0 expert-only cosine **0.963572** matches baseline **0.963572**.

## Method

- Sequential chain with paired residual trajectories.
- Experts ternary (v3 pack-only); attention/routers/norms f32.
- Tokens: 2048, seed 20260806, top_k=2.

## Per-block metrics (expert-only vs FP reference)

| block | block_out cos | resid_in drift | top-1 | top-2 | JS bits | MoE-out cos |
|------:|--------------:|---------------:|------:|------:|--------:|------------:|
| 0 | 0.963572 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.773483 |
| 1 | 0.945765 | 0.277351 | 0.887695 | 0.680664 | 0.005559 | 0.676099 |
| 2 | 0.884956 | 0.341935 | 0.666504 | 0.548828 | 0.009347 | 0.555468 |
| 3 | 0.839144 | 0.498308 | 0.528320 | 0.289551 | 0.024023 | 0.468450 |

### FP16 control

| block | block_out cos | top-1 | top-2 |
|------:|--------------:|------:|------:|
| 0 | 0.999987 | 0.997070 | 0.999023 |
| 1 | 0.999986 | 0.999512 | 0.999023 |
| 2 | 0.999968 | 0.999512 | 0.999512 |
| 3 | 0.999920 | 0.999512 | 0.998047 |

## Provenance

See `metrics.json` for pack SHA-256, scales, τ, and `scale_sources` (`pack_v2`).
Chain exit residual metrics live under `end_of_chain.expert_only_chain_exit` (post-final-block residual stream, not residual-in to the last block).

