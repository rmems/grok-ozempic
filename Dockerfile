FROM rust:1.96-slim AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY dissect/ dissect/

# Documenting feature split (Issue #30):
# The builder stage builds with `--features cli` to produce a minimal production binary
# containing only the CLI dependencies (clap + anyhow). This avoids compilation of heavy
# async or testing dependencies (e.g. tokio, myelin) in the production image.
# We also use BuildKit cache mounts (Issue #31) for cargo registry and build target dir
# to speed up repeated compilation runs.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build --release --features cli --locked && \
    cp /app/target/release/grok-ozempic /usr/local/bin/

FROM rust:1.96-slim AS tester
WORKDIR /app
RUN rustup component add clippy rustfmt
COPY . .

# Documenting feature split (Issue #30):
# The tester stage builds with `--all-features` to run clippy/tests against all features
# (including the async feature for tokio / myelin) for comprehensive code quality check.
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
