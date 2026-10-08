#!/usr/bin/env bash
# Cursor Cloud Agent install script for grok-ozempic (`install` in .cursor/environment.json).
#
# Cursor runs this from the repository root during every Build, on its default
# Ubuntu base image (CPU only: cloud agents have no GPU), then snapshots the disk.
# It must be idempotent. Shell exports don't survive into agent runs, so the tools
# it installs are exposed through /etc/profile.d and /usr/local/bin.
# See https://cursor.com/docs/cloud-agent/setup
#
# Installs only what this repo's CI and manifests need:
#   - apt: build-essential, pkg-config, shellcheck, curl, ca-certificates
#   - Rust stable (+rustfmt, clippy) [default]
#   - uv + Python 3.12 venv with ruff==0.15.14 and numpy>=1.26,<3 (lint.yml, python-scripts.yml)
#   - cargo fetch --locked
#   - tool directories exposed to later shells (/etc/profile.d + /usr/local/bin links)
#
# It ends with a dependency fetch/prebuild, not a test run.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

# Install apt packages that are not already present.
apt_install() {
  local missing=() pkg
  for pkg in "$@"; do
    if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
      missing+=("$pkg")
    fi
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    $SUDO apt-get -o Acquire::Retries=5 update -qq
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=5 install -y --no-install-recommends "${missing[@]}"
  fi
}

# --- System packages (C toolchain for rustc; shellcheck as in lint.yml; curl for the installers) ---
apt_install build-essential pkg-config shellcheck curl ca-certificates

# --- Rust (rustup) ---
export PATH="$HOME/.cargo/bin:$PATH"
if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
    sh -s -- -y --default-toolchain none --profile minimal
fi
# CI (rust.yml) uses dtolnay/rust-toolchain stable; MSRV is 1.88.
rustup toolchain install stable --profile minimal --component rustfmt --component clippy
rustup default stable

# --- Python (uv) ---
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# lint.yml / python-scripts.yml use the runner's python3 (3.12 on ubuntu-latest).
uv python install 3.12
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python 'ruff==0.15.14' 'numpy>=1.26,<3'

# --- Prefetch crates (no build, no tests) ---
cargo fetch --locked

# --- Expose the tools to later shells ---
# The PATH exports above last only for this script; Cursor starts the agent's shells
# separately. Login shells get these directories from /etc/profile.d, and every other
# shell finds the entry points through symlinks in /usr/local/bin (on the default PATH).
# In login shells the venv's bin comes first, so python3 is the venv's Python.
# The links skip python and pip so the system python3 stays the default elsewhere.
repo_root="$(pwd)"
tool_dirs=("$HOME/.cargo/bin" "$repo_root/.venv/bin" "$HOME/.local/bin")
# shellcheck disable=SC2016 # $PATH must expand when the profile is sourced, not now.
printf 'export PATH=%q:$PATH\n' "$(IFS=:; echo "${tool_dirs[*]}")" |
  $SUDO tee /etc/profile.d/cursor-env-grok-ozempic.sh >/dev/null
for dir in "${tool_dirs[@]}"; do
  [ -d "$dir" ] || continue
  for tool in "$dir"/*; do
    name="${tool##*/}"
    case "$name" in
      # uv's installer also drops env/env.fish (sourced, not run) in ~/.local/bin.
      python* | pip* | activate* | deactivate | Activate.ps1 | env | env.fish) continue ;;
    esac
    if [ -f "$tool" ] && [ -x "$tool" ]; then
      $SUDO ln -sfn "$tool" "/usr/local/bin/$name"
    fi
  done
done

# .venv/ is not in .gitignore here, so keep it out of `git status`.
exclude_file="$(git rev-parse --git-path info/exclude)"
mkdir -p "$(dirname "$exclude_file")"
grep -qxF '/.venv/' "$exclude_file" 2>/dev/null || echo '/.venv/' >>"$exclude_file"

echo "Cursor install for grok-ozempic finished."
