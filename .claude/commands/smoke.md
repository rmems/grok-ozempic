# /smoke — quality gates

Run from repo root. Stop on first failure and fix before continuing.

Prefer the root `justfile` (#62 / RM-250):

```bash
just check
just test
```

Equivalent fallback if `just` is unavailable:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --features cli --locked -- -D warnings
cargo test --features cli --locked
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
