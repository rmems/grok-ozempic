# AGENTS.md

## Cursor Cloud specific instructions

This is a pure Rust crate (no external services, databases, or Docker required).

### Toolchain

Requires Rust edition 2024 (Rust >= 1.85). The update script ensures the latest stable toolchain is installed.

### Commands

All CI-matching commands (see `.github/workflows/rust.yml`):

| Task   | Command |
|--------|---------|
| Format | `cargo fmt --all -- --check` |
| Lint   | `cargo clippy --all-targets --all-features -- -D warnings` |
| Test   | `cargo test --all-targets --all-features` |
| Build  | `cargo build --all-targets --all-features` |
| Docs   | `cargo doc --no-deps --all-features` |

### Running the CLI

The binary requires the `cli` feature:

```
cargo run --features cli -- artifacts generate --manifest dissect/grok-1/baseline.json --output-dir /tmp/reports
cargo run --features cli -- artifacts validate --report-dir /tmp/reports --manifest dissect/grok-1/baseline.json
```

### Notes

- The crate uses edition 2024. If you see `edition is not supported` errors, run `rustup update stable`.
- All tests are self-contained (no model weights needed). The embedded `dissect/grok-1/baseline.json` manifest is used as a fallback for testing.
- The `cli` feature is not in the default feature set; use `--features cli` or `--all-features` to build/test the binary.
