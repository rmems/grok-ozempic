# GH #80 Expert Middle-Ground (INT4) Design

**Issue:** [GH #80](https://github.com/rmems/grok-ozempic/issues/80) / Linear RM-468 / beads `goz-d603r4`

**Agent:** Grok Build: Grok 4.5 (xAI)

**Status:** Design lock 2026-08-12 (brainstorming)

**Parallel (orthogonal):** [GH #70](https://github.com/rmems/grok-ozempic/issues/70) — remove GOZ1 v1 fallback; **separate PR**, does not block #80.

**Predecessor:** PR [#76](https://github.com/rmems/grok-ozempic/pull/76) / [#75](https://github.com/rmems/grok-ozempic/issues/75) option 2 — denser / stacked ternary miss viability; HP expert ceiling is viable. Report: `reports/grok-1-expert-precision-remedy-v2/`.

---

## Purpose

Measure whether a **pack-honest higher-bit expert payload** (prefer INT4) closes the multi-block residual/routing gap between the best mostly-ternary schedule (#76 denser) and full-HP experts (#76 ceiling), on the same real-weight 0→3 chain.

### Core question

> After denser periodic HP and stacked channel-α still miss multi-block viability, while a full-HP expert ceiling is viable on the same 0→3 chain, does a middle-ground expert payload (INT4 / higher-bit, pack-honest) close enough of the gap for a mostly-compressed expert policy — or is the policy choice effectively ternary+schedule vs full-HP experts?

### Why this is next

| Closed finding | Implication |
|----------------|-------------|
| #76 denser / stacked | Further N-tweaks on ternary are low value |
| #76 HP ceiling viable | Expert tier *can* carry multi-block at full precision |
| Gap | **Payload width / reconstruction quality** between 2-bit ternary and FP16 (deferred Arm B from #73/#75) |

Do **not** re-run denser ternary schedules as primary science. Cite #76 denser + ceiling as bounds.

---

## Goals

- Reuse `scripts/grok1_multiblock_experiment.py` and `scripts/grok1_multiblock_lib.py`; no activation/routing proxy.
- Paired residual trajectories (pilot carries its own error block-to-block).
- Ship **both**:
  - **P0:** INT4 experts on **all** blocks 0–3.
  - **P1:** INT4 on non-HP blocks + FP16 experts on `{1,2,3}` (denser HP schedule).
- Same chain/settings as #76: blocks `0,1,2,3`, tokens `2048`, seed `20260806`, top-k `2`.
- Pack-honest middle-ground artifacts via a **research side-table** (not silent in-memory quant; not GOZ1 v4).
- Cite #76 denser + HP ceiling (and ladder #72/#74 as context) without re-running denser N as primary.
- Emit full provenance (codec, scales, shapes, commit, seed, schedule, scale source tags).
- Exactly **one** canonical decision (options 1–4), assembled like #75/#76.

## Non-goals

- Full 64-block generation / text quality.
- Router, norm, or attention quantization in primary arms.
- GH #59 proxy pilot matrix.
- CUDA / Myelin (#50).
- New SAAQ ΔQ formula.
- Product GOZ1 v4 layout for 4-bit experts (prefer side-table first).
- Merging #70 container hygiene into the research PR.
- Re-litigating denser ternary schedules as the primary arm set.

---

## Selected approach

### Storage: research side-table (locked)

| Approach | Verdict |
|----------|---------|
| **Research side-table INT4 + scales** | **Chosen** — pack-honest, no container bump |
| GOZ1 4-bit extension | Deferred — productizing early |
| In-memory-only quant from f32 | Rejected — weak provenance |

### Decision assembly: clone #75/#76 (locked)

1. Run **P1** as `--evidence-only` → secondary `metrics.json` (no `decision`).
2. Run **P0** as **primary** with `--comparison-metrics` pointing at P1 (and optional cite payloads if needed).
3. `assemble_remedy_v3_comparison` + `decide_remedy_v3` → exactly one of {1,2,3,4}.
4. Write `reports/grok-1-expert-precision-remedy-v3/{metrics.json,results.md}`.

Primary = pure middle-ground (INT4-all-blocks). Secondary = INT4 + denser HP. Canonical decision ranks the **best middle-ground** against #76 denser / ceiling bounds.

Three independent decision-bearing reports are rejected (multiple decision numbers). A single multi-pilot residual process is rejected for this PR (larger rewrite; not required).

### Parallel: #70

GH #70 removes GOZ1 v1 write/read fallback and silent `legacy_oracle` on the certified path. It is **orthogonal** (#80 dependency table). Run on a **separate branch/PR**. Coordination only: do not break `require_pack_only_scales`; re-pack leftover v1 pilot artifacts if #70 lands first.

---

## INT4 codec lock

| Parameter | Lock |
|-----------|------|
| Codec | Signed **INT4**, values in a documented integer range (default probe: symmetric quant into `int8`-hosted nibble storage or packed int4 array — document exact encoding in report) |
| Roles | **Experts only** (`EXPERT_ROLES`); attention / routers / norms never INT4’d in primary arms |
| Scale layout | Prefer **per-output-channel** (or group along contracting axis, Grok-1-compatible). Exact G and shape **must** appear in report + sidecar JSON. If implementer falls back to per-tensor α, that is an **explicit** report amendment, not silent |
| Artifact layout | Per block, e.g. research root / `int4/block_{BBB}/` with: int4 (or int8-hosted) weight `.npy`, float scale `.npy`, sidecar JSON (shapes, G, codec id, source npy identity, git commit) |
| Scale source tag | `research_int4_side` — **never** claim `pack_v2` |
| Fail-closed | Missing files, shape mismatch, non-finite scales, wrong role set → validation error → decision **4** |

Ternary GOZ1 packs remain for **cite-only** arms and any residual ternary path; they stay v3 + `pack_v2` as today.

---

## Harness changes

### Public arms / modes

| Public `--arm` | Expert source | HP schedule |
|----------------|---------------|-------------|
| `int4` (P0 primary) | `Int4SideExperts` on all chain blocks | none |
| `int4` + `--hp-blocks 1,2,3` (P1) | INT4 on non-HP; FP16 control experts on HP blocks | explicit `{1,2,3}` |
| Existing denser / ceiling | **Cite only** from committed #76 JSON | n/a |

P1 reuses existing `--hp-blocks` (not a second arm enum). Self-describing `arm_label` must include the schedule suffix (e.g. `expert_int4_123` when HP is `{1,2,3}`; `expert_int4` when no HP).

### Weight selection

Extend `_expert_primary` / arm identity so:

- INT4 blocks return `Int4SideExperts` (dequant to f32 for the existing forward).
- HP blocks keep existing FP16 control expert path.
- Non-experts remain on the reference / preserved path (unchanged).

`MixedWeights` / role splits stay authoritative: only expert roles may come from INT4 or ternary primary.

### CLI

- New arm choice(s) for INT4.
- Keep `--evidence-only` for P1.
- Keep `--comparison-metrics` for primary assembly (accept P1 path; optionally accept #76 denser/ceiling paths as **cite** inputs if the assembler needs structured bounds — otherwise hard-code #76 numbers as `BASELINE_76_*` like `BASELINE_74`).
- Prefer **cited constants** from committed #76 `metrics.json` for denser + ceiling (bit-identical settings already locked), not a third live re-run.

### Comparison / decision (v3)

Add (names indicative):

- `BASELINE_76_DENSER` / `BASELINE_76_CEILING` (or embed cite structs from #76 report).
- `assemble_remedy_v3_comparison(primary_chain, secondary_payloads, …)`.
- `decide_remedy_v3(comparison)` → exactly one option.

Validation must include: arm labels, blocks/tokens/seed/top_k, FP16 control clean, INT4 scale tags `research_int4_side`, no `legacy_oracle` on any ternary cite path, schedule match for P1.

Ranking: best **middle-ground** among {P0, P1} by the same viability / improvement spirit as v2 (document exact rank key in code comments + report). Ceiling is a **bound**, not a product recommendation.

### Decision rubric

| Option | When |
|--------|------|
| **1** | Best middle-ground restores multi-block viability (e.g. top-1 ≥ ~0.95 through b3, bounded residual growth, clean FP16) — compressed experts + documented codec is policy-shaped |
| **2** | Clear help vs #76 denser; still compounds / misses bands; ceiling shows full HP can close the rest |
| **3** | Middle-ground fails to beat denser ternary meaningfully → gap is not payload-width |
| **4** | Incomplete/incomparable evidence, failed control, not pack-honest, activation/harness failure |

Report emits exactly one `Option N` heading and rejects non-selected options in prose.

---

## Report layout

```text
reports/grok-1-expert-precision-remedy-v3/
├── metrics.json                 # canonical: provenance, chain, comparison, decision
├── results.md                   # sole decision heading + comparison table
├── run-int4-all.log
├── run-int4-plus-hp123.log
├── int4-plus-hp123/metrics.json # evidence-only secondary
└── (optional) int4-export/      # or external path documented in provenance
```

Agent citation line on results.md (Grok Build / model / #80 / RM-468).

---

## Error handling

- Operational failures (missing npy, bad INT4 sidecar, shape errors) fail the command; do not publish a green decision.
- Evidence-only still fails hard on ops/provenance errors; only suppresses decision publication.
- Malformed comparison inputs → decision **4** with listed `validation_errors`.
- Preserve fail-closed pack scale policy for any GOZ1 path (`legacy_oracle` → hard error / decision 4).

---

## Testing

Python unit tests:

- INT4 dequant shape / scale broadcasting / fail-closed missing scale.
- Arm identity labels and HP schedule for P0 vs P1.
- Scale tags `research_int4_side` vs `fp16_control` vs `pack_v2` (cites).
- Evidence-only omits decision.
- Comparison validation and decision outcomes 1–4 (fixtures).
- Report has exactly one decision heading.
- Experts-only: attention/router/norm never selected as INT4 primary.

Verification ladder:

1. Focused Python tests.
2. `just check` while iterating.
3. INT4 side-table build for blocks 0–3 experts.
4. Real-weight P1 evidence + P0 primary runs.
5. Artifact inspection + `just ci` before publish.
6. Draft PR linked to #80 / RM-468.

---

## Acceptance mapping

| Acceptance (issue) | Design coverage |
|--------------------|-----------------|
| ≥1 primary middle-ground arm | P0 required; P1 also required as secondary |
| #76 denser + ceiling cited | Constants / cite paths; not re-run as primary |
| FP16 clean or decision 4 | Same control gate as v2 |
| Residual + routing + provenance | Reuse metric helpers; sidecar + tags |
| reports/…-v3/ MD + JSON | Layout above |
| Exactly one decision | `decide_remedy_v3` |
| No router/norm/attention quant | Role split unchanged |
| Self-describing scales | `research_int4_side`, no silent oracle |

---

## Implementation sketch (for writing-plans)

| Phase | Work | Branch |
|-------|------|--------|
| 0 | This design doc | docs PR or first commit on research branch |
| 1a | INT4 export + `Int4SideExperts` + unit tests | `research/gh-80-expert-middle-ground` |
| 1b | #70 v1 fallback removal | `fix/gh-70-require-goz1-v2` (**parallel**) |
| 2 | CLI arms, v3 comparison/decision, fixtures | #80 branch |
| 3 | Real-weight runs + reports | #80 branch |
| 4 | Review decision prose; open PRs; do not squash-mix #70+#80 | two PRs |

---

## Risks

| Risk | Mitigation |
|------|------------|
| INT4 group layout wrong for Grok expert shapes | Probe one expert npy; document G; unit-test reshape |
| P1 schedule confuses labels | Use explicit `--hp-blocks 1,2,3` + self-describing arm_label |
| #70 races pack readers | Separate PR; #80 ternary cites already require pack_v2 |
| Decision 4 from weak FP16 | Same as #76; do not weaken control |

---

## References

- Issue #80 / RM-468
- Design #75: `docs/superpowers/specs/2026-08-08-gh-75-expert-remedy-v2-design.md`
- Evidence #76: `reports/grok-1-expert-precision-remedy-v2/`
- Parallel #70: remove GOZ1 v1 fallback after #65

---

*Design lock by Grok Build brainstorming 2026-08-12. Cite: Grok 4.5 (xAI).*
