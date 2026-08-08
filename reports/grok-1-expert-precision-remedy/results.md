# Expert higher-precision remedies for multi-block residual fidelity

**Agent:** Grok Build: Grok 4.5 (xAI) · **Model:** Grok-4.5 (high) · **Issue:** #73 / Linear RM-362
**Design:** Grok Build design lock · **Predecessor:** PR #72 / #68 option 3 · **Baseline (#64):** PR #64 / #61
**Issue:** GH [#73](https://github.com/rmems/grok-ozempic/issues/73) / [RM-362](https://linear.app/rpd-34/issue/RM-362)
**Implementation commit:** `352b2c791f8c2021a1ca5f4bcddb94ad3af458fb`

## Decision (primary arm: C, N=2)

**Option 2 — Remedy helps but needs further correction (clear improvement vs #72; residual still compounds).**

Primary evidence is **Arm C** `expert_periodic_hp_n2`: **ternary on {0,2}, HP (FP16 experts) on {1,3}**.

| Signal | #72 ternary | Arm C N=2 (primary) | Arm A channel-α |
|--------|------------:|--------------:|----------------:|
| b3 block_out cos | 0.839 | **0.882** | 0.866 |
| b3 top-1 | 0.528 | 0.548 | **0.563** |
| b3 top-2 set | 0.290 | 0.317 | **0.326** |
| chain_exit residual drift | 0.654 | **0.529** | 0.564 |

*Bold marks the better remedy value per row (not “primary arm”). Primary decision is still Arm C N=2.*

**Option-2 sensitivity (Arm A):** top-1 gain ≈0.035 and cosine gain ≈0.027 are both below their disjunct thresholds (0.05 / 0.03); only exit-drift gain ≈0.090 clears the 0.08 bar (margin ≈0.010).

- Arm C cuts exit drift **+0.125** vs #72 and lifts b3 top-1 by **+0.020**.
- Still below viability bands (option 1 needs top-1 ≥ ~0.95 through b3 and exit drift < 0.25).
- Arm A (side-table per-channel α, `research_per_channel_side`) also lands **option 2** — slightly better b0 cos / b3 top-1 than C, similar residual story. **Not** pack-only (`scale_sources` tag is research side, not `pack_v2`).
- FP16 control ≥ 0.9999 on all blocks for both arms.

### Why not the other options

- **Option 1 (not chosen):** rejected — routing/residual still outside viability bands after C (and A).
- **Option 2 (selected):** selected — clear help vs #72; residual still compounds; further mechanism or higher expert precision still required.
- **Option 3 (not chosen):** rejected — remedies improved enough that “no help” is false.
- **Option 4 (not chosen):** rejected — FP16 control clean; ternary path pack-only v3.

## #72 baseline (cited — bit-identical settings)

Source: `reports/grok-1-expert-only-multiblock/` (PR #72). Tokens=2048, seed=20260806, blocks=0..3, same packs. Decision option **3**. Not re-run.

## Method

- Sequential chain 0→1→2→3, paired residual trajectories.
- Attention / routers / norms never ternarized.
- Tokens: 2048, seed 20260806, top_k=2.
- **Arm C:** `expert_periodic_hp_n2` — ternary on {0,2}, HP (FP16 experts) on {1,3}.
- **Arm A (appendix):** same ternary trits × per-output-channel LS α from npy (harness-local; no GOZ1 layout bump).
- N=4 sensitivity **not** run (decision is on N=2 only).

## Per-block metrics — Arm C (primary)

| block | pilot label | block_out cos | resid_in drift | top-1 | top-2 | JS bits |
|------:|-------------|--------------:|---------------:|------:|------:|--------:|
| 0 | `goz1_expert_ternary_only` | 0.963572 | 0.000000 | 1.000000 | 1.000000 | 0.000000 |
| 1 | `expert_periodic_hp_n2` | 0.963211 | 0.277351 | 0.887695 | 0.680664 | 0.005559 |
| 2 | `goz1_expert_ternary_only` | 0.905360 | 0.279321 | 0.729004 | 0.592773 | 0.008548 |
| 3 | `expert_periodic_hp_n2` | 0.882308 | 0.449088 | 0.547852 | 0.317383 | 0.025243 |

### FP16 control (Arm C)

| block | block_out cos | top-1 | top-2 |
|------:|--------------:|------:|------:|
| 0 | 0.999987 | 0.997070 | 0.999023 |
| 1 | 0.999986 | 0.999512 | 0.999023 |
| 2 | 0.999968 | 0.999512 | 0.999512 |
| 3 | 0.999920 | 0.999512 | 0.998047 |

### End of chain (Arm C)

- residual cos = 0.882308
- residual drift = **0.529212** (key `chain.end_of_chain.expert_only_chain_exit`)

## Appendix — Arm A (channel-α side-table)

Full dump: `metrics-channel_alpha.json`, `results-channel_alpha.md`. Decision also **option 2**.

| block | block_out cos | resid_in drift | top-1 | top-2 | JS bits |
|------:|--------------:|---------------:|------:|------:|--------:|
| 0 | 0.972652 | 0.000000 | 1.000000 | 1.000000 | 0.000000 |
| 1 | 0.960093 | 0.236060 | 0.905273 | 0.704102 | 0.004023 |
| 2 | 0.907956 | 0.282125 | 0.710449 | 0.576172 | 0.009103 |
| 3 | 0.866169 | 0.428825 | 0.563477 | 0.325684 | 0.017105 |

- chain_exit residual drift = **0.564145**
- scale provenance: `research_per_channel_side` (not a GOZ1 pack field)

## Provenance

- Primary metrics: `metrics.json` (Arm C).
- Pack SHA-256 / v3 τ / scale_sources in pack_provenance.
- Logs: `run-periodic_hp.log`, `run-channel_alpha.log`.
- Host paths stored as basenames where the harness sanitizes them.

