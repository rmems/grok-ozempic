# Cursor cloud-agent / local image for grok-ozempic.
# Do NOT COPY the project — Cursor checks out the correct commit itself.
# Rust: latest stable (1.97.1) — matches host rustup + CI dtolnay/rust-toolchain@stable.
FROM rust:1.97.1-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_TERM_COLOR=always

# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        pkg-config \
        libssl-dev \
        python3 \
        python3-numpy \
        python3-pip \
        sudo \
    && rm -rf /var/lib/apt/lists/*

RUN rustup component add clippy rustfmt

# Passwordless sudo is required by some Cursor cloud-agent tooling (agent install steps).
# This is a developer sandbox image, not a production runtime — intentional tradeoff.
RUN useradd -m -s /bin/bash ubuntu \
    && echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu \
    && mkdir -p /workspace \
    && chown ubuntu:ubuntu /workspace

USER ubuntu
WORKDIR /workspace
