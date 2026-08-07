# REVIEW — pre-push quality gate

Run this before `git push`. Prefer **`just`** recipes; this file documents hooks,
the JetBrains Qodana CLI, and every Rust/Python command that is **not** already
covered by `just check` / `just test` / `just ci`.

| Who | What to run |
|-----|-------------|
| Human, fast iterate | `just check` then `just test` |
| Human / agent, before push | **`just review`** |
| Full static analysis | `just review-full` (includes local Qodana) |
| Env diagnosis only | `just doctor` (always exit 0) |

Discovery: `just --list`.

---

## 1. Recommended gate (`just review`)

```bash
just review
```

This is the default **pre-push** recipe. It runs, in order:

1. **`just ci`** — GHA parity for Rust + the five CI Python unittests + `bash -n`
   on `scripts/*.sh` + optional `actionlint` / `shellcheck` when installed
2. **`cargo audit`** — if `cargo-audit` is on `PATH` (matches
   `.github/workflows/cargo-audit.yml`); otherwise prints `skip:` and continues
3. **Extra Python unittests** not yet in `python-scripts.yml` / `just ci`:
   - `scripts.test_grok1_block_forward`
   - `scripts.test_grok1_block_weights`

No multi-GiB weights. No Docker image builds. No Qodana (use `just review-full`
or `just qodana` for that).

### Faster tiers (not enough alone before push)

```bash
just check    # cargo fmt --check + clippy --features cli --locked -D warnings
just test     # cargo test --features cli --locked + CI Python unittests
just build    # cargo build --all-targets --all-features --locked
just ci       # full pre-PR matrix (no cargo-audit, no extra block unittests)
```

### Data-dependent smoke (local machine only)

```bash
just experiment-smoke   # release CLI --help + CKPT / run3 path presence
# Full pilot (not a just recipe): BLOCK=0 MODE=attention_plus_expert scripts/block_pilot_goz1.sh
```

---

## 2. Git hooks (call `just`)

Hooks live in **`.githooks/`** (tracked). Enable once per clone:

```bash
git config core.hooksPath .githooks
```

| Hook | Runs | Skip |
|------|------|------|
| `.githooks/pre-push` | `just review` | `git push --no-verify` (escape hatch only) |
| `.githooks/pre-commit` | `just check` | `git commit --no-verify` |

Requirements on `PATH`: `just`, `cargo`, `python3`, and for Python tests `numpy`
(`python3 -m pip install --user 'numpy>=1.26,<3'`).

Verify:

```bash
git config --get core.hooksPath   # expect: .githooks
just review                       # same gate the hook will run
```

Agents: if `core.hooksPath` is unset, still run `just review` before any push.
Do not disable hooks permanently; use `--no-verify` only for documented emergencies.

---

## 3. Local Qodana CLI

Config: root [`qodana.yaml`](qodana.yaml) (`linter: qodana-rust`). CI workflow:
[`.github/workflows/qodana.yml`](.github/workflows/qodana.yml).

### Install

```bash
# Official installer (or JetBrains Toolbox / package manager)
# https://www.jetbrains.com/help/qodana/getting-started.html
command -v qodana && qodana --version
```

Docker is used when the CLI runs the linter in a container (`--within-docker=true`
or default depending on environment). Native mode: `--within-docker=false`.

Optional Cloud upload needs `QODANA_TOKEN` (same secret as GHA). Local reports
do not require a token.

### Recipes

```bash
just qodana          # scan with repo qodana.yaml; print problems; local results dir
just review-full     # just review && just qodana
```

Equivalent manual CLI (kept for environments without the new recipes):

```bash
# Full project (matches CI pr-mode: false intent — whole tree, not PR-diff-only)
qodana scan \
  --linter qodana-rust \
  --project-dir . \
  --results-dir .qodana/results \
  --report-dir .qodana/report \
  --print-problems \
  --save-report

# Open HTML report (after a successful scan)
qodana show --report-dir .qodana/report
# or: qodana view --sarif .qodana/results/qodana.sarif.json

# Diff-only against main (optional, faster feedback while iterating)
qodana scan --linter qodana-rust --diff-start origin/main --print-problems
```

`.qodana/` is gitignored (local cache/results). Do not commit SARIF or HTML
reports.

---

## 4. Commands **not** covered by existing `just` tiers

These are outside `just check` / `just test` / `just ci` (unless noted as now
wrapped by `just review` / `just qodana`).

### Rust

| Command | When | Notes |
|---------|------|--------|
| `cargo audit` | Any change to `Cargo.toml` / `Cargo.lock`; always in `just review` when installed | Install: `cargo install cargo-audit --locked` |
| `cargo clippy --all-targets --all-features --locked -- -D warnings` | Full matrix | Already in `just ci` (not in `just check`, which uses `--features cli` only) |
| `cargo test --all-targets --all-features --locked` | Full matrix | Already in `just ci` |
| `cargo doc --no-deps --all-features --locked` | Docs / public API | Already in `just ci` |
| `cargo build --release --features cli --locked` | Release binary / experiment path | In `just experiment-smoke`, not `just ci` |
| `cargo bench --locked` | N/A | `just bench` exits 1 until a harness exists; kernels → `myelin-accelerator` |
| `cargo fmt --all` | Fix formatting | Check-only is in `just check` / `just ci`; omit `-- --check` to rewrite |

GHA `rust.yml` omits `--locked`; local `just` is **stricter** (intentional).

### Python

CI / `just ci` only run five modules. Also run (and **`just review` does**):

```bash
python3 -c 'import numpy; print(numpy.__version__)'   # required for all modules below
python3 -m unittest scripts.test_grok1_block_forward -v
python3 -m unittest scripts.test_grok1_block_weights -v
```

Path-scoped re-runs (same modules as CI — already inside `just test` / `just ci`):

```bash
python3 -m unittest scripts.test_export_grok1_embedding_npy -v
python3 -m unittest scripts.test_export_grok1_int8_npy -v
python3 -m unittest scripts.test_export_grok1_int8_select -v
python3 -m unittest scripts.test_route_preservation_surface -v
python3 -m unittest scripts.test_route_preservation_io -v
```

Manual script syntax checks (stdlib-only scripts; not unittest):

```bash
python3 -m py_compile scripts/export_grok1_embedding_npy.py
python3 -m py_compile scripts/export_grok1_int8_npy.py
python3 -m py_compile scripts/goz1_trit_histogram.py
python3 -m py_compile scripts/route_preservation_metrics.py
# add any other scripts/*.py you touched
```

Shell (already in `just ci` via `_bash-n-scripts`):

```bash
for f in scripts/*.sh; do bash -n "$f"; done
```

### Optional host linters (soft in `just ci`)

```bash
actionlint                          # .github/workflows/*
shellcheck scripts/*.sh             # when shellcheck is installed
```

### Docker (path-scoped; not in `just review`)

When you touch `Dockerfile`, `Cargo.toml`, `Cargo.lock`, `src/**`, or `dissect/**`:

```bash
docker build --target tester -t grok-ozempic:test .
docker build --target runtime -t grok-ozempic:latest .
```

### CLI smoke without `just`

```bash
cargo run --features cli --locked -- --help
cargo run --features cli --locked -- quantize-goz1 --help
```

---

## 5. Fallback when `just` is missing

Mirror **`just review`** without the just binary:

```bash
# --- just ci ---
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-targets --all-features --locked
cargo build --all-targets --all-features --locked
cargo doc --no-deps --all-features --locked
python3 -c 'import numpy; print(numpy.__version__)'
python3 -m unittest scripts.test_export_grok1_embedding_npy -v
python3 -m unittest scripts.test_export_grok1_int8_npy -v
python3 -m unittest scripts.test_export_grok1_int8_select -v
python3 -m unittest scripts.test_route_preservation_surface -v
python3 -m unittest scripts.test_route_preservation_io -v
for f in scripts/*.sh; do bash -n "$f"; done
# optional: actionlint; shellcheck scripts/*.sh

# --- review extras ---
command -v cargo-audit >/dev/null && cargo audit || echo "skip: cargo-audit"
python3 -m unittest scripts.test_grok1_block_forward -v
python3 -m unittest scripts.test_grok1_block_weights -v

# --- optional full ---
qodana scan --linter qodana-rust --project-dir . \
  --results-dir .qodana/results --report-dir .qodana/report \
  --print-problems --save-report
```

---

## 6. Agent checklist (before remote push)

1. `just doctor` if the environment is unfamiliar
2. **`just review`** (or `just review-full` when static analysis matters)
3. Diff hygiene: no secrets, no multi-GiB weights, no accidental `.beads` noise
4. Commit with imperative subject + GH / Linear IDs when known
5. `git pull --rebase` then `git push` (project session-completion protocol)
6. If the pre-push hook fails, fix failures — do not habitually `--no-verify`

Related: `/pr-ready` (`.claude/commands/pr-ready.md`), root `justfile` (#62 / RM-250),
`docs/ARCHITECTURE.md`.
