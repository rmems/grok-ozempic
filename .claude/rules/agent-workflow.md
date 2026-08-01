# Agent workflow

## Tracking

- **Local agents:** use **bd** when installed (`bd ready`, `bd show`, `bd update --claim`, `bd close`) — same duty as `AGENTS.md` / `CLAUDE.md` beads block.
- **Claude Code cloud:** `bd`/Dolt are often missing — use GitHub issues + Linear (rmems team, `RM-*`). Hooks fail soft; do not invent MEMORY.md.
- Markdown under `.claude/commands` is **acceptance/scope**, not a second tracker. Real work is GH/Linear/bd.

## Quality gates before done

```bash
cargo fmt --all -- --check
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
