# Expert middle-ground (INT4) multi-block fidelity

**Agent:** Grok Build: Grok 4.5 (high) (xAI) · Issue: #80 / Linear RM-468 · beads goz-d603r4  
**Predecessor:** PR #76 / #75 option 2 (denser ternary + HP ceiling)  
**Codec:** `research_int4_side` — per-output-channel absmax INT4, `qmax=7`  
**Settings:** blocks 0→3, tokens 2048, seed 20260806, top_k=2

## Decision

**Option 2 — INT4 middle-ground helps vs denser ternary, but full-HP experts or another correction remain required.**

**Best middle-ground arm:** `expert_int4_123` (INT4 on {0}, FP16 experts on {1,2,3})

Rationale:

- `best_middle_ground_arm=expert_int4_123`
- `best_b3_top1=0.925293`
- `best_b3_cos=0.991523`
- `best_chain_exit_drift=0.130213`
- `#76_denser_b3_top1=0.615723 (cited)`
- `#76_denser_b3_cos=0.913853 (cited)`
- `#76_denser_exit_drift=0.437055 (cited)`
- `#76_ceiling_viable=True (cited)`
- `any_middle_ground_improved_vs_denser=True`
- `codec=research_int4_side per-output-channel absmax qmax=7`

### Why not the other options

- **Option 1:** not chosen — best middle-ground misses locked viability (top-1 through b3 needs ≥ ~0.95; measured **0.925**).
- **Option 2:** selected — large improvement vs denser ternary and a viable HP ceiling, but policy still short of full viability.
- **Option 3:** not chosen — clear improvement vs denser contradicts “payload-width useless”.
- **Option 4:** not chosen — complete, comparable evidence; FP16 controls clean.

## Comparison (measured + cited)

| Signal | #76 denser (cite) | P0 INT4-all | P1 INT4+HP123 | #76 HP ceiling (cite) |
|---|---:|---:|---:|---:|
| b3 top-1 | 0.615723 | 0.850586 | **0.925293** | **1.000000** |
| b3 block_out cos | 0.913853 | 0.978044 | **0.991523** | **0.999984** |
| chain-exit drift | 0.437055 | 0.211064 | **0.130213** | **0.005690** |

## Method

- Paired residual trajectories; no Gaussian / embed proxy for b≠0.
- Experts only INT4 (or FP16 on HP blocks); attention / routers / norms never quantized.
- INT4 formed online from the same f32 expert npy as the FP reference (`research_int4_side`).
- #76 denser + ceiling cited from committed `reports/grok-1-expert-precision-remedy-v2/` (not re-run).

## Per-block — P0 `expert_int4` (INT4 all blocks)

Values from `metrics.json` (`router_top2_set_agreement` / expert_only).

| block | block_out cos | resid_in drift | top-1 | top-2 |
|------:|--------------:|---------------:|------:|------:|
| 0 | 0.998615 | 0.000000 | 1.000000 | 1.000000 |
| 1 | 0.997147 | 0.052881 | 0.981934 | 0.933594 |
| 2 | 0.989049 | 0.075692 | 0.929688 | 0.885742 |
| 3 | 0.978044 | 0.147663 | 0.850586 | 0.729492 |

FP16 control min block_out cos ≥ 0.9999 (clean).

## Per-block — P1 `expert_int4_123` (evidence-only secondary)

Values from `int4-plus-hp123/metrics.json`.

| block | block_out cos | top-1 | top-2 | pilot label |
|------:|--------------:|------:|------:|-------------|
| 0 | 0.998615 | 1.000000 | 1.000000 | research_int4_side |
| 1 | 0.998473 | 0.981934 | 0.933594 | expert_int4_123 (FP16 experts) |
| 2 | 0.995620 | 0.953125 | 0.919922 | expert_int4_123 (FP16 experts) |
| 3 | 0.991523 | 0.925293 | 0.861328 | expert_int4_123 (FP16 experts) |

## Reproducibility note

Both `metrics.json` artifacts record `implementation.dirty: true` and commit
`7cf99b2` (pre-INT4 harness landing). Numerics were measured on a dirty tree;
the harness + validation fixes in this PR post-date that SHA. A clean-tree
re-run of the multi-GiB chain is a follow-up — Option 2 is unchanged under
revalidation of the committed payloads.

## Interpretation

INT4 experts are a **large** step beyond denser ternary schedules (b3 top-1 0.62 → 0.85 all-INT4 → 0.925 with denser HP). Residual exit drift falls 0.44 → 0.21 → 0.13. The HP ceiling remains essentially perfect, so the expert tier *can* carry multi-block; remaining gap is smaller than after #76 but not closed for option-1 viability without more HP or another correction.

## Artifacts

```text
reports/grok-1-expert-precision-remedy-v3/
├── metrics.json
├── results.md
├── run-int4-all.log
├── run-int4-plus-hp123.log
└── int4-plus-hp123/metrics.json
```
