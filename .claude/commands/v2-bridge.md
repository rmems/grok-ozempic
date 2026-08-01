# /v2-bridge — GH #40 / Linear RM-191

Tracked work: GitHub **#40** / Linear **RM-191** as linked IDs. **Use `bd` for task status** when installed (`bd update` / `bd close` + `bd dolt push`). Markdown below is acceptance/scope only.

## Problem

- `dissect/grok-1/structural-manifest.json` has correct preserve/ternary rules (V2 names).
- `stream::resolve_manifest` **rejects** `MANIFEST_NAME_CONVENTION_V2`.
- V1 `baseline.json` has empty `ternary_candidates` and relies on default ternary — wrong for structural-named preserve tensors if names do not match.

## Scope (acceptance)

1. Design checkpoint name ↔ structural name map **or** accept V2 when inputs already use structural names.
2. Wire into `resolve_manifest` / classification **without breaking V1**.
3. Tests: embedding ternary + router/norm **preserve** under V2 (no silent ternary).
4. Prefer `structural-manifest` in docs for first real quant once wired.

## Touch points

- `src/core/stream.rs` — `resolve_manifest`
- `src/core/manifest.rs` — `MANIFEST_NAME_CONVENTION_V1` / `V2`
- `src/core/alignment.rs` — already exercises V2 for alignment
- `src/core/selection.rs` / precision classification
- `src/bin/grok-ozempic/quantize.rs` — CLI manifest path / env

## Acceptance

- GOZ1 path can classify using structural-manifest rules
- Routers/norms cannot fall into default ternary by name mismatch
- V1 baseline path still works
- `cargo test --all-targets --all-features --locked` green

## Non-goals

- Full multi-model ModelInventory epic (#32)
- CUDA / myelin work

## Done protocol

1. Quality gates (`/pr-ready` section 1 — full CI matrix)
2. `bd close` / `bd update` when bd is available, then `bd dolt push`; else GH/Linear status only (cloud exception)
3. Commit: **imperative subject**, body explains **why**, include `(#40 / RM-191)` when applicable
4. `git pull --rebase && git push` until `git status` is up to date with origin
5. PR title includes `(#40 / RM-191)`
6. File follow-up issues for remaining work
7. Clean temp artifacts / worktree noise
8. Short handoff note (what shipped, what remains)
