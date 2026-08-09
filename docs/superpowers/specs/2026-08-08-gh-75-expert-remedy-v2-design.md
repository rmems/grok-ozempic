# GH #75 Expert Precision Remedy V2 Design

**Issue:** [GH #75](https://github.com/rmems/grok-ozempic/issues/75) / Linear RM-462

**Agent:** OpenAI Codex (GPT-5.6 Sol xhigh)

**Status:** Approved on 2026-08-08

## Purpose

PR #74 showed that periodic high-precision (HP) expert refresh at N=2 and
per-output-channel alpha each improve a four-block Grok-1 chain, but neither
restores multi-block residual or routing fidelity. This change measures the
next bounded remedies on the same real-weight 0→3 chain:

1. **C denser:** block 0 uses GOZ1 v3 ternary experts; blocks 1, 2, and 3 use
   FP16 experts.
2. **C+A stacked:** the #74 N=2 schedule keeps FP16 experts on blocks 1 and 3,
   while blocks 0 and 2 use GOZ1 trits with the existing
   `research_per_channel_side` scale source.
3. **HP expert ceiling:** every expert is FP16, while attention, routers, and
   norms remain on the existing high-precision reference path.

The result must be one comparison-backed decision, not three independent
decision claims.

## Goals

- Reuse `scripts/grok1_multiblock_experiment.py` and
  `scripts/grok1_multiblock_lib.py`; do not introduce an activation or routing
  proxy.
- Preserve paired residual trajectories, so every pilot carries its own error
  from block to block.
- Run all three new arms with blocks `0,1,2,3`, 2,048 tokens, seed `20260806`,
  and top-k 2.
- Cite the bit-comparable #72 ternary and #74 N=2 baselines without rerunning
  them.
- Emit full pack, scale, threshold, schedule, commit, and runtime provenance.
- Add Codex-scoped commit attribution without stamping commits made by the
  repository owner or other agents.

## Non-goals

- No GOZ1 v4 or other container layout change.
- No INT4 payload, hybrid expert split, full 64-block evaluation, text-quality
  evaluation, CUDA/Myelin work, or non-expert ternarization.
- No manual Linear twin. GH #75's GitHub→Linear synchronization owns RM-462;
  RM-463 is a canceled duplicate.
- The HP ceiling is a bound, not a product recommendation.

## Selected Approach

Each expensive arm runs independently. The stacked and ceiling runs emit
evidence-only JSON with no decision. The C-denser run is the primary run and
loads those two evidence payloads to assemble one canonical comparison.

This approach preserves the current one-pilot residual-stream implementation
and bounded peak RSS. A shared multi-arm forward would avoid repeated reference
work but would require multiple live pilot residuals and a broader rewrite of
the block loop. Three independent ordinary reports were rejected because they
would publish multiple decision numbers.

## Harness Changes

### Explicit HP schedules

Add optional `--hp-blocks` parsing to the existing experiment CLI. The value is
a comma-separated set of block indices. Every listed HP block must occur in
`--blocks`; duplicates are normalized. An explicit schedule takes precedence
over periodic derivation. `hp_ceiling` always selects every chain block and
rejects a contradictory partial explicit schedule.

The existing `periodic_hp` arm remains backward compatible. With
`--hp-blocks 1,2,3`, its self-describing label is
`expert_periodic_hp_123`, not the misleading `expert_periodic_hp_n2`.

### New arm modes

Add two public arms and internal modes:

| Public arm | Internal mode | Expert source |
|---|---|---|
| `stacked_hp_channel_alpha` | `periodic_hp_plus_channel_alpha` | FP16 control on HP blocks; `ChannelAlphaExperts` elsewhere |
| `hp_ceiling` | `all_hp` | FP16 control on every block |

The stacked decision run uses the unchanged N=2 derivation, giving HP blocks
`{1,3}` and channel-alpha blocks `{0,2}`. Its label is
`expert_periodic_hp_n2_plus_channel_alpha`. Ternary-block scale provenance must
say `research_per_channel_side`; HP blocks must say `fp16_control`.

The ceiling label is `expert_hp_ceiling`. It uses FP16 only for expert roles;
the `MixedWeights` fallback keeps non-expert roles on the f32 reference path.

Existing `EXPERT_ROLES`, `PRESERVED_ROLES`, `require_pack_only_scales`, and
`legacy_oracle` guards remain authoritative. No arm may ternarize attention,
routers, or norms.

## Evidence and Canonical Assembly

Add an `--evidence-only` switch for secondary runs. Evidence-only payloads
contain `provenance` and `chain`, but omit `decision`, do not print a decision,
and do not write a decision report.

Add repeatable `--comparison-metrics PATH` inputs for the primary remedy run.
Canonical assembly validates that every payload has:

- a unique expected arm label;
- blocks `0,1,2,3`, 2,048 tokens, seed `20260806`, and top-k 2;
- the same block-wise GOZ1 pack SHA-256 identities;
- a clean FP16 control;
- pack-v3, non-legacy scale provenance; and
- the expected HP/channel-alpha schedule.

Missing, duplicate, mismatched, or malformed secondary evidence forces decision
4. It must never be silently excluded from the comparison.

The committed report layout is:

```text
reports/grok-1-expert-precision-remedy-v2/
├── metrics.json
├── results.md
├── run-denser-hp.log
├── run-stacked-channel-alpha.log
├── run-hp-ceiling.log
├── stacked-channel-alpha/metrics.json
└── hp-ceiling/metrics.json
```

`metrics.json` is the canonical payload and embeds or references the two
validated secondary chains. `results.md` contains the only decision heading,
the full comparison table, and secondary-arm appendices. Secondary JSON files
are raw evidence and contain no decision field.

## Baselines and Decision Policy

Add `BASELINE_74` beside `BASELINE_72`, using the committed #74 Arm C N=2
values and pack identities. Baseline sections explicitly say “cited, not
rerun.” The main comparison includes #72 ternary, #74 N=2, C denser, N=2+C+A,
and HP ceiling.

The canonical decision evaluates the best **mostly-ternary** candidate; the HP
ceiling only bounds the remaining gap:

1. **Option 1:** C denser or stacked C+A reaches the existing viability shape,
   including top-1 of at least 0.95 on every measured block through block 3,
   bounded residual growth, and a clean FP16 control.
2. **Option 2:** at least one mostly-ternary arm clearly improves #74 but misses
   viability, while a clean HP ceiling shows that full-HP experts can bound the
   gap but are still required.
3. **Option 3:** denser and stacked remedies fail to improve the prior arm and
   even the expert HP ceiling cannot carry the chain under the current
   non-expert policy.
4. **Option 4:** evidence is incomplete or incomparable, pack/scale honesty
   fails, or any required FP16 control is not clean.

The report states why each non-selected option was rejected and emits exactly
one `Option N` decision heading. It never presents the ceiling as a product
policy.

## Error Handling

- Reject `--hp-period < 1`, HP blocks outside the chain, a partial ceiling
  schedule, and comparison inputs on non-remedy runs with `ForwardError`.
- Preserve exit code 5 for `legacy_oracle` detection.
- Write unresolved/error artifacts with GH #75 / RM-462 and the selected arm.
- Treat malformed JSON, missing arm metadata, schedule mismatch, pack mismatch,
  and failed controls as explicit decision-4 reasons.
- Evidence-only mode still fails the command on operational or provenance
  errors; it only suppresses decision publication.

## Codex Commit Attribution

Create project-local `.codex/hooks.json` with Codex `PreToolUse` and
`PostToolUse` handlers for Bash. The scripts mirror the safety properties of
the existing Claude hook while remaining Codex-specific:

- record the pre-command HEAD per session and repository;
- only act on a successful `git commit` created by that tool call;
- skip dry runs, help, amend, merges, rebases, cherry-picks, reverts, published
  commits, and an already-present trailer;
- amend message-only with `--only`, preserving the index after partial commits;
  and
- use `Co-Authored-By: Codex <noreply@openai.com>`, overrideable with
  `CODEX_COAUTHOR` and disableable with `CODEX_COAUTHOR=0`.

Repository-local Codex hooks require project trust and explicit review through
`/hooks`, as documented by OpenAI at <https://learn.chatgpt.com/docs/hooks>.
Because this session started before the new hook existed, implementation
commits also pass the trailer explicitly.

Update `AGENTS.md` to document the Codex hook, trust step, manual trailer
fallback, and GitHub-only issue creation with automatic Linear synchronization.
Do not replace the existing Claude attribution path or install a global Git
hook.

## Testing

Python unit tests cover:

- explicit HP parsing, normalization, precedence, and subset validation;
- denser, stacked, and ceiling labels and schedule metadata;
- `_expert_primary` source selection for all new modes;
- honest `research_per_channel_side` and `fp16_control` scale tags;
- evidence-only payloads omitting decisions;
- comparison validation for settings, packs, controls, and arm uniqueness;
- decision outcomes 1 through 4, including the ceiling-not-policy rule; and
- a report assertion that exactly one canonical decision heading exists.

Hook tests use temporary Git repositories to verify positive attribution,
idempotence, partial-index preservation, published/merge-operation guards, and
the `CODEX_COAUTHOR=0` escape hatch.

Verification proceeds in increasing cost:

1. focused Python and hook tests;
2. `just check` while iterating;
3. three real-weight decision runs with the locked settings;
4. artifact/provenance inspection and comparison regeneration;
5. `just ci` before publication; and
6. branch push plus a draft PR linked to GH #75 / RM-462.

## Acceptance Mapping

- All three requested arms run on real Grok-1 weights through the existing
  multi-block harness.
- #72 and #74 baselines appear on identical settings and pack identities.
- Every arm includes a clean FP16 control or the canonical decision is 4.
- Markdown and JSON live under
  `reports/grok-1-expert-precision-remedy-v2/` with complete provenance.
- The canonical report emits one of the four issue decisions exactly once.
- Non-expert roles remain high precision, and side-table/schedule metadata is
  self-describing.
- Codex-authored commits carry the Codex co-author trailer through an
  agent-scoped repository hook plus an explicit current-session fallback.
