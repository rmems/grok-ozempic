FROM rust:1.96-slim AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY dissect/ dissect/
RUN cargo build --release --all-features --locked

FROM rust:1.96-slim AS tester
WORKDIR /app
RUN rustup component add clippy rustfmt
COPY . .
RUN cargo test --all-targets --all-features --locked && \
    cargo clippy --all-targets --all-features --locked -- -D warnings && \
    cargo fmt --all -- --check

FROM debian:bookworm-slim AS runtime
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --shell /bin/bash appuser
COPY --from=builder /app/target/release/grok-ozempic /usr/local/bin/
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["grok-ozempic", "--help"]
USER appuser
ENTRYPOINT ["grok-ozempic"]
