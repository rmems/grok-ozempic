# Dev container for grok-ozempic (Codespaces / VS Code).
# Rust: latest stable 1.97.1 (host + CI). Python for npy export scripts.
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

RUN rustup component add clippy rustfmt llvm-tools-preview

# Passwordless sudo for VS Code / Codespaces tooling (dev sandbox, not prod).
RUN useradd -m -s /bin/bash -u 1000 vscode \
    && echo 'vscode ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode \
    && mkdir -p /workspace \
    && chown -R vscode:vscode /workspace /usr/local/cargo /usr/local/rustup

USER vscode
ENV PATH=/usr/local/cargo/bin:$PATH
WORKDIR /workspace
