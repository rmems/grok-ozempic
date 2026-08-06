# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd prime` for full workflow context.

> **Architecture in one line:** Issues live in a local Dolt database
> (`.beads/dolt/`); cross-machine sync uses `bd dolt push/pull` (a
> git-compatible protocol), stored under `refs/dolt/data` on your git
> remote — separate from `refs/heads/*` where your code lives.
> `.beads/issues.jsonl` is a passive export, not the wire protocol.
>
> See [SYNC_CONCEPTS.md](https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md)
> for the one-screen overview and anti-patterns (don't treat JSONL as the
> source of truth; don't `bd import` during normal operation; don't
> reach for third-party Dolt hosting before trying the default).

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work atomically
bd close <id>         # Complete work
bd dolt push          # Push beads data to remote
```

## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var

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
- Everything else under `.beads/` (`*.db`, `export-state.json`, `last_pull`,
  `.linear-sync.lock`, `backup/`, `embeddeddolt/`, `events.jsonl`) is
  machine-local and gitignored.

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
2. **Run quality gates** (if code changed) - Prefer `just ci` before ship / `just check` while iterating (`just --list` for tiers; cargo matrix fallback in `CLAUDE.md`)
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
