# Expert higher-precision remedies for multi-block residual fidelity

**Agent:** Grok Build: Grok 4.5 (xAI) · Model: Grok-4.5 (high) · Issue: #73 / Linear RM-362
**Predecessor:** PR [#72](https://github.com/rmems/grok-ozempic/pull/72) / #68 option 3 · **Baseline (#64):** Claude Fable 5
**Implementation commit:** `352b2c791f8c2021a1ca5f4bcddb94ad3af458fb`

## Decision

**Option 2 — Remedy helps but needs further correction (clear improvement vs #72; residual still compounds).**

Rationale:

- `block_output_cosine sequence=['0.972652', '0.960093', '0.907956', '0.866169']`
- `residual_in_drift sequence=['0.000000', '0.236060', '0.282125', '0.428825']`
- `block_output_drift sequence=['0.236060', '0.282125', '0.428825', '0.564145']`
- `router_top1 sequence=['1.000000', '0.905273', '0.710449', '0.563477']`
- `router_top2_set_agreement sequence=['1.000000', '0.704102', '0.576172', '0.325684']`
- `expert_load_js_bits sequence=['0.000000', '0.004023', '0.009103', '0.017105']`
- `compounding_heuristic=roughly_linear`
- `end_block_output_cosine=0.866169`
- `last_block_residual_in_drift=0.42882545404308425`
- `chain_exit_residual_drift=0.5641451735503821`
- `#64 block0 expert-only block_output_cosine baseline=0.963572`
- `#72 baseline chain_exit_drift=0.653886 b3_top1=0.528320 b3_cos=0.839144 (cited, not re-run)`
- `arm=research_per_channel_side`

**Metrics note:** #72 single-scale ternary baseline cited from reports/grok-1-expert-only-multiblock/ (bit-identical seed/tokens/packs).


### Why not the other options

- **Option 1 (not chosen):** rejected — routing/residual still outside viability bands.
- **Option 2 (selected):** selected — clear help vs #72 but residual/routing still need more.
- **Option 3 (not chosen):** rejected — remedy improved enough to prefer option 1 or 2.
- **Option 4 (not chosen):** rejected — FP16 control and pack-honest path resolved.

## #72 baseline (cited — bit-identical settings)

Source: `reports/grok-1-expert-only-multiblock/ (PR #72 / #68 / RM-255)`.
Tokens=2048, seed=20260806, blocks=[0, 1, 2, 3].
Chain-exit residual drift **0.653886**; b3 top-1 **0.528320**; b3 block_out cos **0.839144**.
Decision option **3** (single-scale ternary not multi-block viable).
Re-run only if packs or harness invalidate comparison; this report cites.

## Method

- Arm: `channel_alpha` / label `research_per_channel_side`.
- Sequential chain with paired residual trajectories.
- Attention / routers / norms never ternarized.
- Tokens: 2048, seed 20260806, top_k=2.

## Per-block metrics (remedy arm vs FP reference)

| block | block_out cos | resid_in drift | top-1 | top-2 | JS bits | MoE-out cos |
|------:|--------------:|---------------:|------:|------:|--------:|------------:|
| 0 | 0.972652 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.857502 |
| 1 | 0.960093 | 0.236060 | 0.905273 | 0.704102 | 0.004023 | 0.834762 |
| 2 | 0.907956 | 0.282125 | 0.710449 | 0.576172 | 0.009103 | 0.714700 |
| 3 | 0.866169 | 0.428825 | 0.563477 | 0.325684 | 0.017105 | 0.647582 |

### FP16 control

| block | block_out cos | top-1 | top-2 |
|------:|--------------:|------:|------:|
| 0 | 0.999987 | 0.997070 | 0.999023 |
| 1 | 0.999986 | 0.999512 | 0.999023 |
| 2 | 0.999968 | 0.999512 | 0.999512 |
| 3 | 0.999920 | 0.999512 | 0.998047 |

## Provenance

See `metrics-channel_alpha.json` for pack SHA-256, scales, τ, and `scale_sources`.
Chain exit under `chain.end_of_chain.expert_only_chain_exit`.
Arm A uses `research_per_channel_side` (not pack_v2) when present.

