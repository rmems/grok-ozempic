# /pr-ready — ship checklist

## 1. Quality gates

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --features cli --locked -- -D warnings
cargo test --features cli --locked
```

## 2. Diff hygiene

- No secrets, tokens, or credentials
- No multi-GiB weight artifacts
- Avoid accidental `.beads/*` noise unless intentional sync
- Scope matches the issue (do not mix #40 bridge with unrelated refactors unless requested)

## 3. Commit message

- Imperative subject + why in body
- GH + Linear IDs when applicable
- Claude Code cloud sessions already add a `Claude-Session:` trailer — leave that alone; no extra product brand lines needed

## 4. PR

- Title: `type: summary (#N / RM-xxx)`
- Body: problem, approach, test plan, link issues
- Push branch; open PR with `gh pr create` when authenticated

## 5. Handoff

Comment on the GitHub issue (two-way Linear sync for rmems/grok-ozempic when configured) with PR URL and residual risks.
