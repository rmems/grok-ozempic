# /smoke — quality gates

Run from repo root. Stop on first failure and fix before continuing.

Prefer the root `justfile` (#62 / RM-250):

```bash
just check
just test
```

Fallback if `just` is unavailable. Note `just test` also runs **thirteen**
Python unittest modules, which the Rust commands alone do not -- so this is not
equivalent unless you include the Python loop:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --features cli --locked -- -D warnings
cargo test --features cli --locked

# The Python half of `just test`. Read the list from the justfile rather than
# retyping it -- a hand-copied list is what drifted to five modules before.
python3 -c 'import numpy; print(numpy.__version__)'
sed -n '/^    mods=(/,/^    )/p' justfile | grep -oE 'scripts\.[a-z0-9_]+' \
  | while read -r m; do python3 -m unittest "$m" -v || exit 1; done
```

Optional broader check (slower) — or use `just ci`:

```bash
cargo test --all-targets --all-features --locked
cargo clippy --all-targets --all-features --locked -- -D warnings
```

CLI help smoke (no weights required):

```bash
cargo run --features cli -- quantize-goz1 --help
cargo run --features cli -- validate-ingest --help
```

Report: pass/fail per command, first error snippet, rustc version.
