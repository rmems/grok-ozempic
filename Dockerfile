FROM rust:1.85-slim AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY dissect/ dissect/
RUN cargo build --release --all-features

FROM rust:1.85-slim AS tester
WORKDIR /app
COPY . .
RUN cargo test --all-targets --all-features && \
    cargo clippy --all-targets --all-features -- -D warnings && \
    cargo fmt --all -- --check

FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/grok-ozempic /usr/local/bin/
ENTRYPOINT ["grok-ozempic"]
