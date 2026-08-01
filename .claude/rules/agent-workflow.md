# Agent workflow

## Tracking

- Prefer **bd** (`bd ready`, `bd show`, `bd update --claim`, `bd close`) when `bd` is installed.
- On **Claude Code cloud**, `bd`/Dolt often are **not** available — use GitHub issues + Linear (rmems team, `RM-*`) instead. Hooks fail soft; do not invent MEMORY.md.

## Quality gates before done

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --features cli -- -D warnings
cargo test --features cli
```

Use `/smoke` or `/pr-ready` slash commands.

## Commits and PRs

- Every agent commit message body **must** end with attribution appropriate to the agent. For Grok Build sessions:

  ```text
  Grok Build: Grok 4.5 (high)
  ```

- PR titles include GH issue + Linear when known, e.g. `(#40 / RM-191)`.
- Do not commit secrets, multi-GiB weights, or accidental `.beads` noise unless intentional.
- Session is not complete until **`git push`** succeeds when the project protocol requires it (see CLAUDE.md beads block for local agent duty).

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
