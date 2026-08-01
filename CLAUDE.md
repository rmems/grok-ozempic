# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

**Claude Code project pack:** `.claude/` (rules, commands, skills, agents, cloud-safe hooks).

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

## Build & Test

Rust edition 2024 crate. CLI binary is feature-gated.

```bash
# Unit + integration tests (CLI surface)
cargo test --features cli --locked

# Lint / format (match CI spirit)
cargo fmt --all -- --check
cargo clippy --all-targets --features cli --locked -- -D warnings

# CLI binary
cargo run --features cli -- --help
cargo run --features cli -- quantize-goz1 --help

# Full matrix (slower)
cargo test --all-targets --all-features --locked
```

Python 3 is only required for Grok-1 pickle → `.npy` export (`scripts/export_grok1_embedding_npy.py`; **stdlib only**, no NumPy).

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
2. Runtime V2 structural manifests are **rejected** in `resolve_manifest` until **#40 / RM-191**.
3. Use `dissect/grok-1/baseline.json` for real packs today; structural-manifest is alignment/dry-run.
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
