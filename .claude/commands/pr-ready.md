# /pr-ready — ship checklist

## 1. Quality gates (match CI)

Prefer the root `justfile` (#62 / RM-250):

```bash
just ci
```

Fallback if `just` is unavailable (`--locked` is intentional and stricter than
GHA). **This is not a hand-maintained copy of the module list** -- that is what
went stale before: this file listed five Python modules while `just ci` ran
thirteen, and named no linters at all, while claiming parity.

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-targets --all-features --locked
cargo build --all-targets --all-features --locked
cargo doc --no-deps --all-features --locked

# Python modules: read the list from the justfile rather than retyping it.
python3 -c 'import numpy; print(numpy.__version__)'
sed -n '/^    mods=(/,/^    )/p' justfile | grep -oE 'scripts\.[a-z0-9_]+' \
  | while read -r m; do python3 -m unittest "$m" -v || exit 1; done

for f in scripts/*.sh; do bash -n "$f"; done

# Linters -- blocking in CI via .github/workflows/lint.yml (GH #98 / #101).
# Omitting these is the most common way a "green" local run disagrees with CI.
ruff check scripts/                      # pip install 'ruff==0.15.14'
shellcheck scripts/*.sh .githooks/pre-commit .githooks/pre-push \
           .codex/hooks/*.sh .beads/hooks/pre-commit .beads/hooks/pre-push
actionlint
```

Fast smoke while iterating on CLI-only edits (not sufficient alone before merge):

```bash
just check
just test
# or without just (mirrors just check + just test):
cargo fmt --all -- --check
cargo clippy --all-targets --features cli --locked -- -D warnings
cargo test --features cli --locked

# `just test` also runs every Python module. A comment pointing at the loop
# above does not run it, so the loop is repeated here -- this block is only a
# mirror of `just check` + `just test` if it actually executes both halves.
python3 -c 'import numpy; print(numpy.__version__)'
sed -n '/^    mods=(/,/^    )/p' justfile | grep -oE 'scripts\.[a-z0-9_]+' \
  | while read -r m; do python3 -m unittest "$m" -v || exit 1; done
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

**Do not run `bd dolt push`.** No beads Dolt remote is configured in this repo:

```console
$ bd dolt remote list
No remotes configured.
```

Beads state travels via the committed `.beads/issues.jsonl` export plus
GitHub/Linear, which stay canonical. This matches `CLAUDE.md` and
`.claude/rules/agent-workflow.md`; until GH #96 this file contradicted both.
If a Dolt remote is ever configured, add the push here **and** update those two
files in the same change.

On the cloud/no-`bd` path, update only the linked GitHub/Linear handoff state.

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
