# /pr-ready — ship checklist

## 1. Quality gates (match CI)

Prefer the root `justfile` (#62 / RM-250):

```bash
just ci
```

Canonical matrix fallback (same steps as `.github/workflows/rust.yml`, with `--locked`) if `just` is unavailable:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-targets --all-features --locked
cargo build --all-targets --all-features --locked
cargo doc --no-deps --all-features --locked
```

Fast smoke while iterating on CLI-only edits (not sufficient alone before merge):

```bash
just check
just test
# or without just:
cargo clippy --all-targets --features cli --locked -- -D warnings
cargo test --features cli --locked
```

Path-scoped extras (only when those paths change; not part of `just ci`):

```bash
# scripts/export_grok1_embedding_npy.py or scripts/test_export_grok1_embedding_npy.py
python3 -c 'import numpy; print(numpy.__version__)'
python3 -m unittest scripts.test_export_grok1_embedding_npy -v

# Cargo.toml or Cargo.lock → cargo-audit.yml
cargo audit

# Root Docker build inputs (Dockerfile, Cargo.toml, Cargo.lock, src/**, dissect/**,
# or docker-compose that builds it) → both docker.yml targets
docker build --target tester -t grok-ozempic:test .
docker build --target runtime -t grok-ozempic:latest .

# .devcontainer/* or .cursor/Dockerfile: image build only if you changed those Dockerfiles
# docker build -f .devcontainer/Dockerfile .
# docker build -f .cursor/Dockerfile .
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

## 4. Tracker update

Before the final code push, file follow-up issues and close finished work in `bd`.
Run `bd dolt push` after every status update that must be shared. On the explicit
cloud/no-Dolt exception, update only the linked GitHub/Linear handoff state.

## 5. Push until the remote is current

```bash
git pull --rebase
git push
git status   # branch up to date with origin AND clean working tree
# expect: nothing to commit, working tree clean; tracking branch up to date
```

If push fails, **fix the cause** (auth, non-fast-forward, protected branch, hooks), then retry. Do not spin on the same error. Do not hand off with only a local commit or with dirty uncommitted work.

## 6. PR

- Title: `type: summary (#N / RM-xxx)`
- Body: problem, approach, test plan, link issues
- Create: `gh pr create` when needed
- Update existing PR metadata: `gh pr edit` (not `gh pr view` — view is read-only)
- Verify: `gh pr view`

## 7. Handoff

Comment on the GitHub issue (two-way Linear sync when configured) with the PR URL
and residual risks. Clean temp artifacts and leave a short handoff note.
