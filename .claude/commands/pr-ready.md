# /pr-ready — ship checklist

## 1. Quality gates (match CI)

Canonical matrix (same steps as `.github/workflows/rust.yml`, with `--locked`):

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-targets --all-features --locked
cargo build --all-targets --all-features --locked
cargo doc --no-deps --all-features --locked
```

Fast smoke while iterating on CLI-only edits (not sufficient alone before merge):

```bash
cargo clippy --all-targets --features cli --locked -- -D warnings
cargo test --features cli --locked
```

Path-scoped extras (run when those paths change in the PR):

```bash
# scripts/export_*.py or related tests → .github/workflows/python-scripts.yml
python3 -m unittest scripts.test_export_grok1_embedding_npy -v

# Dockerfile / docker-compose / .devcontainer / .cursor Dockerfiles → docker.yml spirit
docker build --target tester -t grok-ozempic:test .
```

## 2. Diff hygiene

- No secrets, tokens, or credentials
- No multi-GiB weight artifacts
- Avoid accidental `.beads/*` noise unless intentional sync
- Scope matches the issue (do not mix #40 bridge with unrelated refactors unless requested)

## 3. Commit message

- Imperative subject + body that explains **why**
- GH + Linear IDs when applicable
- Claude Code cloud sessions already add a `Claude-Session:` trailer — leave that alone

## 4. Push until the remote is current

```bash
git pull --rebase
git push
git status   # branch up to date with origin AND clean working tree
# expect: nothing to commit, working tree clean; tracking branch up to date
```

If push fails, **fix the cause** (auth, non-fast-forward, protected branch, hooks), then retry. Do not spin on the same error. Do not hand off with only a local commit or with dirty uncommitted work.

## 5. PR

- Title: `type: summary (#N / RM-xxx)`
- Body: problem, approach, test plan, link issues
- Create: `gh pr create` when needed
- Update existing PR metadata: `gh pr edit` (not `gh pr view` — view is read-only)
- Verify: `gh pr view`

## 6. Tracker + handoff

**bd is the project task tracker when installed** (see `AGENTS.md` / `CLAUDE.md` beads block):

```bash
# when bd is available
bd close <id>   # or bd update … for in-progress notes
bd dolt push    # share beads state; local close alone is not enough
```

**Exception — Claude Code cloud (and any host without bd/Dolt):** update the GitHub issue / Linear twin for handoff links only; do not invent MEMORY.md. Prefer installing/using bd on long-lived local agents.

Then comment on the GitHub issue (two-way Linear sync when configured) with PR URL and residual risks. File follow-ups for unfinished work, clean temp artifacts, and leave a short handoff note.
