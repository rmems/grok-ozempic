# Agent workflow

## Tracking

- **bd is required** for task tracking whenever `bd` is installed (`bd ready`, `bd show`, `bd update --claim`, `bd close`, `bd remember` for durable notes). Markdown under `.claude/` is acceptance/scope, not the tracker.
- Every local `bd close` / `bd update` that should be shared **must** be followed by `bd dolt push` (git push does not sync Dolt beads).
- **Exception — Claude Code cloud / hosts without bd or Dolt:** GH issues + Linear (`RM-*`) for status and handoff only. SessionStart does not install bd. Do not invent MEMORY.md.

## Quality gates before done

Match CI when shipping (see `.github/workflows/rust.yml`; `--locked` is stricter than CI and preferred):

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-targets --all-features --locked
cargo build --all-targets --all-features --locked
cargo doc --no-deps --all-features --locked
```

CLI-focused fast path while iterating:

```bash
cargo clippy --all-targets --features cli --locked -- -D warnings
cargo test --features cli --locked
```

Use `/smoke` or `/pr-ready` slash commands.

## Commits and PRs

- Clear imperative subjects; body explains **why**. Link GH + Linear IDs when known.
- Claude Code on the web already attaches a `Claude-Session:` trailer / session URL — do not invent a third-party tool citation.
- PR titles include GH issue + Linear when known, e.g. `(#40 / RM-191)`.
- Do not commit secrets, multi-GiB weights, or accidental `.beads` noise unless intentional.
- Session is not complete until **`git push`** succeeds (mandatory session-completion protocol in `CLAUDE.md` / `AGENTS.md`). Do not leave finished work only on the local machine.

## GitHub issues / PRs (MCP first + xAI citation)

Prefer the **GitHub MCP** (`github__issue_write`, `github__add_issue_comment`, `github__triage_issue`) over bare `gh` for create/update/triage when the server is connected — set labels, milestone, assignees, and body in one shot.

**Always cite the agent** on create and on material triage comments (Grok Build / xAI). Do not ship bare issues.

Required first line of every new issue/PR body (and Linear twin description):

```markdown
**Agent:** Grok Build: Grok 4.5 (xAI) · **Issue:** #N / Linear RM-XXX · beads `goz-…`
```

If Linear/bd IDs are unknown at create time, still open with:

```markdown
**Agent:** Grok Build: Grok 4.5 (xAI)
```

…then patch the body once twins exist.

Also set on create (do not leave for a follow-up turn):

| Field | Default for this repo |
|-------|------------------------|
| `assignees` | `rmems` (owner) unless user names someone else |
| `milestone` | Active path → `2` (`Spiking-sparse multi-tensor GOZ1`); GPU consumer → `3` (`MyelinBackend RTX 5080`); long-horizon → `4` (`Backlog / later`) |
| `labels` | Match siblings (`repo:grok-ozempic`, `GOZ1`/`SAAQ`/`quantization`/`experiment`/`fable-sprint` as applicable) |

Non-Grok agents use their own product line (e.g. `Claude Code: Fable 5`) — never omit agent attribution.

## Claude Code on the web

- Runs on Anthropic-managed Ubuntu VMs (~4 vCPU / 16 GB / 30 GB). Repo clone only — **no** `~/.models/xai-grok-1` unless the user mounts data or uses Remote Control.
- No committed `.claude/Dockerfile`. Customize VM via **claude.ai cloud environment** setup script if needed (`gh`, extra apt).
- Project pack: `.claude/rules`, `commands`, `skills`, `agents` + root `CLAUDE.md`.

## Multi-agent handoff (other surfaces)

| Path | Consumer |
|------|----------|
| `.devcontainer/` | Codespaces / VS Code Dev Containers |
| `.cursor/Dockerfile` | Cursor cloud agents |
| `.devin/blueprint.yaml` | Devin |
| `.claude/` + `CLAUDE.md` | Claude Code (local + web) |
