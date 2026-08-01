# /v2-bridge — GH #40 / Linear RM-191

Tracked work: GitHub **#40** / Linear **RM-191** as linked IDs only. **bd is the task tracker** when installed. Markdown below is acceptance/scope, not status.

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

1. Quality gates (`/pr-ready` section 1 — full CI matrix including build + doc; path-scoped extras if needed)
2. File follow-up **bd** issues for remaining work (before beads push)
3. `bd close` for finished work / `bd update` for in-progress notes, then **`bd dolt push`** (required when bd is available)
4. **Exception — no bd/Dolt (e.g. Claude Code cloud):** GH/Linear status only for handoff links; do not invent MEMORY.md
5. Commit: **imperative subject**, body explains **why**, include `(#40 / RM-191)` when applicable
6. `git pull --rebase && git push` (resolve errors, then retry) until branch is up to date and working tree is clean
7. PR title includes `(#40 / RM-191)`; use `gh pr create` / `gh pr edit` as needed
8. Clean temp artifacts, clear stashes, and prune remote branches; if cleanup touches tracked files, commit + push again and re-check `git status`
9. Short handoff note (what shipped, what remains)
