# Stacked and denser expert remedies for multi-block fidelity

**Agent:** OpenAI Codex: GPT-5.6 Sol (xhigh) · Issue: #75 / Linear RM-462 · beads goz-rvk
**Issue:** GH #75 / Linear RM-462 / beads goz-rvk
**Predecessor:** PR #74 / #73 / RM-362 (decision 2)
**Implementation commit:** `07f0a697dc04002f78b222362149ff9a2a7c7892`

## Decision

**Option 2 — Stronger remedies help, but full-HP experts or another correction remain required.**

**Best mostly-ternary remedy:** `expert_periodic_hp_123`

Rationale:

- `best_mostly_ternary_arm=expert_periodic_hp_123`
- `best_b3_top1=0.615723`
- `best_b3_cos=0.913853`
- `best_chain_exit_drift=0.4370545723375058`
- `#74_b3_top1=0.547852 (cited, not re-run)`
- `#74_b3_cos=0.882308 (cited, not re-run)`
- `#74_chain_exit_drift=0.529212 (cited, not re-run)`
- `hp_ceiling_viable=True`
- `best_improved_vs_74=True`
- `locked_option_2_requires_improvement_and_ceiling=True`

### Why not the other options

- **Option 1:** not chosen — neither mostly-ternary candidate met every locked viability band.
- **Option 2:** selected — clear improvement and a viable HP ceiling both hold, but policy misses viability.
- **Option 3:** not chosen — clear improvement plus a viable ceiling contradict total failure.
- **Option 4:** not chosen — complete, comparable evidence satisfies the locked Option-2 pair.

## #72 baseline (cited — bit-comparable settings and packs)

Source: `reports/grok-1-expert-only-multiblock/ (PR #72 / #68 / RM-255)` (not re-run).
Tokens=2048, seed=20260806, blocks=[0, 1, 2, 3], top_k=2.
Chain-exit residual drift **0.653886**; b3 top-1 **0.528320**; b3 block_out cos **0.839144**.
Prior decision: **3**.

## #74 baseline (cited — bit-comparable settings and packs)

Source: `reports/grok-1-expert-precision-remedy/ (PR #74 / #73 / RM-362)` (not re-run).
Tokens=2048, seed=20260806, blocks=[0, 1, 2, 3], top_k=2.
Chain-exit residual drift **0.529212**; b3 top-1 **0.547852**; b3 block_out cos **0.882308**.
Prior decision: **2**.

## Multi-arm comparison

| Signal | #72 ternary | #74 N=2 | C denser | N=2+C+A | HP ceiling |
|---|---:|---:|---:|---:|---:|
| b3 top-1 | 0.528320 | 0.547852 | 0.615723 | 0.592285 | **1.000000** |
| b3 top-2 | 0.289551 | 0.317383 | 0.408691 | 0.376465 | **0.999512** |
| b3 block_out cos | 0.839144 | 0.882308 | 0.913853 | 0.905379 | **0.999984** |
| chain-exit drift | 0.653886 | 0.529212 | 0.437055 | 0.454413 | **0.005690** |

## Method

- Arm: `periodic_hp` / label `expert_periodic_hp_123`.
- Arm C label `expert_periodic_hp_123`: ternary on {0}, HP (FP16 experts) on {1,2,3}.
- Sequential chain with paired residual trajectories.
- Attention / routers / norms never ternarized.
- Tokens: 2048, seed 20260806, top_k=2.

## Per-block metrics (remedy arm vs FP reference)

| block | block_out cos | resid_in drift | top-1 | top-2 | JS bits | MoE-out cos |
|------:|--------------:|---------------:|------:|------:|--------:|------------:|
| 0 | 0.963572 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.773483 |
| 1 | 0.963211 | 0.277351 | 0.887695 | 0.680664 | 0.005559 | 0.901761 |
| 2 | 0.939684 | 0.279321 | 0.729004 | 0.592773 | 0.008548 | 0.851689 |
| 3 | 0.913853 | 0.348475 | 0.615723 | 0.408691 | 0.024466 | 0.785866 |

### FP16 control

| block | block_out cos | top-1 | top-2 |
|------:|--------------:|------:|------:|
| 0 | 0.999987 | 0.997070 | 0.999023 |
| 1 | 0.999986 | 0.999512 | 0.999023 |
| 2 | 0.999968 | 0.999512 | 0.999512 |
| 3 | 0.999920 | 0.999512 | 0.998047 |

## Secondary-arm appendices

### `expert_periodic_hp_n2_plus_channel_alpha`

Stacked C+A label `expert_periodic_hp_n2_plus_channel_alpha`: channel-α trits on {0,2}, HP (FP16 experts) on {1,3}.

| block | block_out cos | resid_in drift | top-1 | top-2 | JS bits | MoE-out cos |
|------:|--------------:|---------------:|------:|------:|--------:|------------:|
| 0 | 0.972652 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.857502 |
| 1 | 0.971191 | 0.236060 | 0.905273 | 0.704102 | 0.004023 | 0.925383 |
| 2 | 0.927502 | 0.242170 | 0.769531 | 0.654297 | 0.004506 | 0.751475 |
| 3 | 0.905379 | 0.384552 | 0.592285 | 0.376465 | 0.012466 | 0.791595 |

#### FP16 control — `expert_periodic_hp_n2_plus_channel_alpha`

| block | block_out cos | top-1 | top-2 |
|------:|--------------:|------:|------:|
| 0 | 0.999987 | 0.997070 | 0.999023 |
| 1 | 0.999986 | 0.999512 | 0.999023 |
| 2 | 0.999968 | 0.999512 | 0.999512 |
| 3 | 0.999920 | 0.999512 | 0.998047 |

### `expert_hp_ceiling`

HP expert ceiling: FP16 experts on every measured block.

| block | block_out cos | resid_in drift | top-1 | top-2 | JS bits | MoE-out cos |
|------:|--------------:|---------------:|------:|------:|--------:|------------:|
| 0 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 1 | 0.999997 | 0.000051 | 1.000000 | 0.999512 | 0.000000 | 0.999991 |
| 2 | 0.999998 | 0.002299 | 1.000000 | 1.000000 | 0.000000 | 0.999994 |
| 3 | 0.999984 | 0.001945 | 1.000000 | 0.999512 | 0.000000 | 0.999939 |

#### FP16 control — `expert_hp_ceiling`

| block | block_out cos | top-1 | top-2 |
|------:|--------------:|------:|------:|
| 0 | 0.999987 | 0.997070 | 0.999023 |
| 1 | 0.999986 | 0.999512 | 0.999023 |
| 2 | 0.999968 | 0.999512 | 0.999512 |
| 3 | 0.999920 | 0.999512 | 0.998047 |

## Provenance

See `metrics.json` for the canonical decision, embedded secondary evidence, pack SHA-256, thresholds, schedules, and applied scale-source tags.
Secondary `metrics.json` files are evidence-only and intentionally contain no decision.
The HP ceiling is an expert-tier bound, not a product recommendation.
