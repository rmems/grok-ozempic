# Agent workflow

## Tracking

- **Canonical SoT = GitHub issues + Linear (`RM-*`).** Create/update/close there for anything that must survive machines or agents. Markdown under `.claude/` is acceptance/scope, not the tracker.
- **`.beads/issues.jsonl` is a committed offline mirror** so agents can recover project context with no network access or API tokens. It is a *cache* of GitHub + Linear, never where a decision is made -- if they disagree, GitHub/Linear wins and the export is refreshed. `.beads/interactions.jsonl` and the other `.beads/` runtime files stay gitignored.
- The export regenerates on commit, so churn is expected. **Never hand-merge a `.beads/issues.jsonl` conflict** -- take either side and re-export (`bd export`, or just commit and let the hook do it).
- **Board scope:** GitHub Project [Grok Quantization](https://github.com/users/rmems/projects/6) and Linear project **Grok Quantization** (team `rmems`) track the same work. Beads' Linear sync is pinned to that project via `linear.project_id`, with `github.org=rmems` / `github.repo=grok-ozempic`. Do not widen it to the whole team; an unscoped sync pulled 261 issues into a ~67-issue repo. File new issues onto the project or they fall out of sync scope.
- The `dolt` CLI is installed (v2.2.3) but **no Dolt beads remote is configured**
  (`bd dolt remote list` → "No remotes configured"), so `bd dolt push` is not
  required. Having the CLI is not the same as having a remote; see the beads
  sync note in `CLAUDE.md` before adding it to any checklist.
- **After a squash merge, branch deletion is manual — never `git pull --rebase` it.**
  Verify with `git diff --stat origin/main <branch>` first (should be empty if fully merged), then delete manually only if intended. Squash merges leave no ancestry link, so replaying commits onto `main` conflicts (`AA`) on every file it touched. See `CLAUDE.md`.

### Issue relationships (what next after close)

- **Beads owns the readiness graph** (`bd dep` / `bd link`). GitHub sub-issues under epics (e.g. #48) mirror the same parent for humans.
- Types: `blocks` (hard sequence — `bd dep add NEW --blocked-by DONE`), `parent-child` (epic ownership), `relates-to` / `relate` (soft), `supersedes` (replacement).
- **On close:** run `bd dep list <id>` and `bd ready`. Start the highest-priority **unblocked** successor. If the close creates follow-on work, file it and wire `bd dep add follow-on --blocked-by closed-id` (and GH parent/sub-issue when under an epic) in the **same session**.
- `bd ready` ignores closed blockers; historical `blocks` edges remain for `bd dep tree` context.
- Do **not** invent markdown dependency boards; keep edges in beads + GH/Linear.

## Quality gates before done

Prefer the root `justfile` tiers (#62 / RM-250):

```bash
just check   # iterate (fmt + clippy --features cli)
just ci      # before done / pre-PR (matches rust.yml + python-scripts.yml spirit)
just doctor  # env/data diagnosis (always exit 0)
```

Slash commands `/smoke` (`just check` + `just test`) and `/pr-ready` (`just ci`) call the same recipes.

Equivalent cargo matrix if `just` is unavailable (see `.github/workflows/rust.yml`; `--locked` is stricter than CI and preferred):

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-targets --all-features --locked
cargo build --all-targets --all-features --locked
cargo doc --no-deps --all-features --locked
```

CLI-focused fast path while iterating (fallback without `just`):

```bash
cargo clippy --all-targets --features cli --locked -- -D warnings
cargo test --features cli --locked
```

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
