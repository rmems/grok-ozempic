# Cursor cloud-agent / local image for grok-ozempic.
# Do NOT COPY the project — Cursor checks out the correct commit itself.
# Rust: latest stable (1.99.0) — matches CI dtolnay/rust-toolchain@stable (MSRV 1.88).
FROM rust:1.99.0-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ENV CARGO_TERM_COLOR=always

# shellcheck: lint.yml. curl/ca-certificates: the uv installer below.
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
        shellcheck \
        sudo \
    && rm -rf /var/lib/apt/lists/*

RUN rustup component add clippy rustfmt

# Passwordless sudo is required by some Cursor cloud-agent tooling (agent install steps).
# This is a developer sandbox image, not a production runtime — intentional tradeoff.
# Cursor's install/terminal shells do not inherit this image's ENV PATH, so the
# rustup proxies are linked into /usr/local/bin, and the toolchain dirs are handed
# to ubuntu so `cargo fetch` can write the registry cache.
RUN useradd -m -s /bin/bash ubuntu \
    && echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu \
    && mkdir -p /workspace \
    && chown ubuntu:ubuntu /workspace \
    && chown -R ubuntu:ubuntu "$CARGO_HOME" "$RUSTUP_HOME" \
    && ln -sf "$CARGO_HOME"/bin/* /usr/local/bin/

# uv for the Python 3.12 venv that .cursor/install.sh builds (lint.yml, python-scripts.yml),
# installed straight into /usr/local/bin so every shell finds it.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh

USER ubuntu
WORKDIR /workspace
