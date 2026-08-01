# /pr-ready — ship checklist

## 1. Quality gates

Default (matches most PRs / `cli` surface):

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --features cli --locked -- -D warnings
cargo test --features cli --locked
```

If the change touches optional features outside `cli` (e.g. `async`), also run:

```bash
cargo test --all-targets --all-features --locked
cargo clippy --all-targets --all-features --locked -- -D warnings
```

## 2. Diff hygiene

- No secrets, tokens, or credentials
- No multi-GiB weight artifacts
- Avoid accidental `.beads/*` noise unless intentional sync
- Scope matches the issue (do not mix #40 bridge with unrelated refactors unless requested)

## 3. Commit message

- Imperative subject + body that explains **why**
- GH + Linear IDs when applicable
- Claude Code cloud sessions already add a `Claude-Session:` trailer — leave that alone; no extra product brand lines needed

## 4. Push until the remote is current

```bash
git pull --rebase
git push
git status   # must show up to date with origin on this branch
```

Retry push until it succeeds. Do not hand off with only a local commit.

## 5. PR

- Title: `type: summary (#N / RM-xxx)`
- Body: problem, approach, test plan, link issues
- Open/update PR when authenticated (`gh pr create` / `gh pr view`)

## 6. Handoff

- Update issue status when applicable (`bd close` / GH close / Linear) after the work is actually done
- Comment on the GitHub issue (two-way Linear sync for rmems/grok-ozempic when configured) with PR URL and residual risks
