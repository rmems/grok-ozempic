# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

**Claude Code project pack:** `.claude/` (rules, commands, skills, agents, cloud-safe hooks).

## Issue tracking (source of truth)

**Canonical SoT = GitHub issues + Linear twins (`RM-*`).** Status, milestones,
close state, and handoff are decided there.

`.beads/issues.jsonl` **is tracked on purpose**: agents read it to recover full
project context with no network access and no API tokens. Treat it as a committed
*cache* of GitHub + Linear rather than the place a decision is made -- when the
two disagree, GitHub/Linear wins and the export is refreshed.

- `.beads/interactions.jsonl` is **not** tracked: a short log with little context
  value, so committing it is per-commit churn for no recovery benefit.
- Let the beads hooks re-export; don't hand-edit the JSONL. On a rebase/merge
  conflict in it, **re-export instead of merging by hand** (`bd export`, or just
  commit and let the hook do it). Hand-edit only if the export is corrupt and
  `bd` cannot rewrite it.
- Also tracked, as beads scaffolding rather than data: `.beads/config.yaml`,
  `metadata.json`, `README.md`, `.gitignore`, and `hooks/`. Editing those changes
  behaviour for every clone, so treat them as project config, not local state.
- Machine-local and gitignored: `*.db`, `export-state.json`, `last_pull`,
  `last-touched`, `.linear-sync.lock`, `backup/`, `embeddeddolt/`,
  `events.jsonl`. Beads maintains its own `.beads/.gitignore` covering most of
  these, so the root `.gitignore` only adds what that file misses.

**Board scope — keep both sides aligned.** The same work is tracked as GitHub
Project [Grok Quantization](https://github.com/users/rmems/projects/6) and the
Linear project **Grok Quantization** (team `rmems`). Beads' Linear sync is pinned
to that project so it does not drag in other repos' issues:

```bash
bd config set linear.project_id 36d0c86f-6348-44d8-a5b3-27930c488bce
bd config set github.org rmems
bd config set github.repo grok-ozempic
```

An unscoped sync previously pulled the whole `rmems` team into the local DB (261
issues for a ~67-issue repo). Do not widen it. Conversely, an issue left off the
project falls out of sync scope, so put new issues on the project when filing.

The section below is **generated and owned by `bd`** -- edit it via `bd`, not by
hand, or the next injection will overwrite your changes.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->

## After a squash merge: delete the branch, never rebase it

PRs here land as **squash merges**, which create one new commit on `main` with
**no ancestry link** to the branch's individual commits. The local branch still
holds all of them.

So the "PUSH TO REMOTE" step above (`git pull --rebase`) is **wrong on a branch
whose PR has already been merged**: git replays every original commit onto a
`main` that already contains the same content under a different SHA, cannot tell
they are equivalent, and stops on `both added (AA)` conflicts in essentially
every file the branch touched.

```bash
# WRONG after your PR was squash-merged -- 27 commits, ~8 AA conflicts
git checkout experiment/my-branch && git pull --rebase

# RIGHT: the work is already on main; start clean
git checkout main && git pull --rebase
git branch -D experiment/my-branch      # verify first, below
git checkout -b feat/next-thing
```

Confirm the branch really is fully merged before deleting — an empty diff means
it contributes nothing beyond `main`:

```bash
git diff --stat origin/main <branch>    # empty => safe to delete
```

Do **not** try to resolve the conflicts, and do **not** hand-merge
`.beads/issues.jsonl` (re-export instead — see above). If a rebase is already
mid-flight and every file is `AA`, that is this situation: `git rebase --abort`.

This has bitten the repo repeatedly (`ef451fe "refresh artifacts after rebasing
onto merged #67"` is one such cleanup commit).

## Beads cross-machine sync

The `dolt` CLI is installed (v2.2.3), but **no beads Dolt remote is configured**
(`bd dolt remote list` → "No remotes configured"), so `bd dolt push` / `bd dolt
pull` are **not** part of the session-completion checklist. Beads state travels
via the committed `.beads/issues.jsonl` export plus GitHub/Linear, which remain
the source of truth.

Having the CLI available is not the same as having a remote wired up. If one is
ever configured, add `bd dolt push` to the checklist and update this note and
`.claude/rules/agent-workflow.md` together.

## Build & Test

Rust edition 2024 crate. CLI binary is feature-gated.

Prefer the root `justfile` (#62 / RM-250):

```bash
just --list
just check              # fmt + clippy (cli) while iterating
just test               # cargo test --features cli + Python unittests
just ci                 # pre-PR parity with GitHub Actions
just doctor             # env/path diagnosis (ok/warn/missing; non-zero only if recipe broken)
just experiment-smoke   # release CLI --help + local data probe
```

If `just` is unavailable, mirror the recipes manually (include Python — do not cargo-only):

```bash
# just check
cargo fmt --all -- --check
cargo clippy --all-targets --features cli --locked -- -D warnings

# just test (numpy required for several scripts/test_* modules)
python3 -c 'import numpy; print(numpy.__version__)'
cargo test --features cli --locked
python3 -m unittest scripts.test_export_grok1_embedding_npy -v
python3 -m unittest scripts.test_export_grok1_int8_npy -v
python3 -m unittest scripts.test_export_grok1_int8_select -v
python3 -m unittest scripts.test_route_preservation_surface -v
python3 -m unittest scripts.test_route_preservation_io -v

# just ci (pre-PR parity; --locked is intentional and stricter than GHA)
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-targets --all-features --locked
cargo build --all-targets --all-features --locked
cargo doc --no-deps --all-features --locked
# + the five python3 -m unittest lines above
for f in scripts/*.sh; do bash -n "$f"; done

# CLI smoke
cargo run --features cli -- --help
cargo run --features cli -- quantize-goz1 --help
```

Python 3 is only required for Grok-1 pickle → `.npy` export and host-side analysis:

| Script | Deps | Scope |
|--------|------|-------|
| `scripts/export_grok1_embedding_npy.py` | **stdlib only** | one f32 tensor per invocation (byte copy) |
| `scripts/export_grok1_int8_npy.py` | **numpy** | int8 `QuantizedWeight8bit` → f32 dequant export; manifest-driven, whole pilot block per invocation (`--block`/`--mode`). `--structural-name` is a debug/repair hatch for partial re-export — not the pilot contract |
| `scripts/goz1_trit_histogram.py` | stdlib only | exact GOZ1 trit counts |
| `scripts/route_preservation_metrics.py` | numpy | fills the run3 route-preservation surface from a pack |

Slash shortcuts (Claude Code): `/smoke`, `/quantize-embed`, `/v2-bridge`, `/pr-ready`.

## Architecture Overview

**grok-ozempic** = Grok-1-specific SNN-style quantization and GOZ1 packing orchestration.

| Layer | Role |
|-------|------|
| Manifests | xai-dissect JSON → preserve / fp16 / ternary_snn |
| Stream | Out-of-core three-pass quant (`src/core/stream.rs`) |
| GOZ1 | Binary weight pack (`weight_pack*`) |
| Backend | `LocalBackend` CPU today; `MyelinBackend` FFI later |
| Kernels | **Not here** — `myelin-accelerator` owns CUDA |

Key docs: `docs/ARCHITECTURE.md`, `docs/dissect-manifest.md`, `docs/grok1-saaq-artifact-flow.md`, `README.md`.

### Critical pipeline facts

1. Official Grok-1 pickle shards are **not** accepted by `quantize-goz1` — export npy first.
2. Runtime V2 structural manifests are **accepted** in `resolve_manifest` (**#40 / RM-191**); V2 requires structural-named inputs and hard-errors on any unmatched tensor (no defaults fallthrough).
3. Prefer `dissect/grok-1/structural-manifest.json` for real packs when inputs are structural-named (export-script npy stems); `baseline.json` (V1) remains for legacy `blk.*` names. Authoritative structural names: `~/rmems/grok-result/xai-dissect/LATEST_CORRECT_GROK1_RUN/manifests/xai-grok-1-ckpt-0/`.
4. Preserve > fp16 > ternary_candidates > defaults — a name mismatch that ternary-quantizes routers/norms is a classification bug (fix the matcher; do not paper over with defaults).

### Open critical path

| Issue | Topic |
|-------|--------|
| GH [#40](https://github.com/rmems/grok-ozempic/issues/40) / Linear RM-191 | V2 structural name bridge for stream `resolve_manifest` |
| #39 / RM-190 | First embedding → GOZ1 experiment (results when present under `reports/`) |

## Conventions & Patterns

- **Commits:** imperative subject, why in the body, GH + Linear IDs when known. Claude Code cloud adds `Claude-Session:` trailers automatically.
- **PR titles:** include GH + Linear IDs, e.g. `(#40 / RM-191)`.
- **Tracking:** bd when available; GH/Linear on Claude cloud if bd missing.
- **No secrets** in repo or cloud environment variables that are world-readable.
- **Feature flag:** production CLI uses `--features cli`.

### Claude Code on the web

- Anthropic-managed VM (Ubuntu, Rust preinstalled). Repo clone only — **no** home checkpoints.
- Customize packages via **claude.ai cloud environment** setup script (not a repo Dockerfile).
- SessionStart runs `scripts/claude-session-start.sh` (soft-fails without `bd`).
- Multi-GiB quant experiments need local machine or Remote Control.

### Multi-agent environment handoff

| Path | Audience |
|------|----------|
| `.claude/` + this file | Claude Code local + cloud |
| `.devcontainer/` | Codespaces / VS Code Dev Containers |
| `.cursor/Dockerfile` | Cursor cloud agents |
| `.devin/blueprint.yaml` | Devin |

## Agent pack map

| Path | Purpose |
|------|---------|
| `.claude/rules/` | Standing project constraints (GOZ1, manifests, kernels, workflow) |
| `.claude/commands/` | `/smoke`, `/quantize-embed`, `/v2-bridge`, `/pr-ready` |
| `.claude/skills/` | Deep playbooks for quantize + structural V2 |
| `.claude/agents/goz1-reviewer.md` | Review specialist |
| `scripts/claude-session-start.sh` | Cloud-safe SessionStart / PreCompact |
