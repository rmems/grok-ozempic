FROM rust:1.97.1-slim AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY dissect/ dissect/

# Documenting feature split (Issue #30):
# The builder stage builds with `--features cli` to produce a minimal production binary
# containing only the CLI dependencies (clap + anyhow). This keeps future optional
# dependencies (e.g. the planned myelin-accelerator FFI backend) out of the production
# image. The `async`/tokio feature this comment used to cite was removed in #99 -- it
# gated no code and pulled 14 packages into every --all-features build.
# We also use BuildKit cache mounts (Issue #31) for cargo registry and build target dir
# to speed up repeated compilation runs.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build --release --features cli --locked && \
    cp /app/target/release/grok-ozempic /usr/local/bin/

FROM rust:1.97.1-slim AS tester
WORKDIR /app
RUN rustup component add clippy rustfmt
COPY . .

# Documenting feature split (Issue #30):
# The tester stage builds with `--all-features` to run clippy/tests against every feature
# (today just `cli`; the planned myelin backend will join it) for a comprehensive check.
# We also use BuildKit cache mounts (Issue #31) here to preserve build artifacts.
# Note: /app/target is NOT cached in tester stage to ensure deterministic test results
# (avoids stale artifacts masking failures when files are removed/renamed).
RUN --mount=type=cache,target=/usr/local/cargo/registry \
        --mount=type=cache,target=/usr/local/cargo/git \
        cargo test --all-targets --all-features --locked && \
        cargo clippy --all-targets --all-features --locked -- -D warnings && \
        cargo fmt --all -- --check

FROM debian:bookworm-slim AS runtime
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --shell /bin/bash appuser
COPY --from=builder /usr/local/bin/grok-ozempic /usr/local/bin/
USER appuser
ENTRYPOINT ["grok-ozempic"]
