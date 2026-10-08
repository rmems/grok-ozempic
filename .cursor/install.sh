#!/usr/bin/env bash
# Cursor Cloud Agent install step for grok-ozempic (`install` in .cursor/environment.json).
#
# Runs from the repository root during every Build, on top of .cursor/Dockerfile
# (Rust stable + clippy/rustfmt, shellcheck, uv). CPU only. It must be idempotent.
# See https://cursor.com/docs/cloud-agent/setup
#
#   - uv + Python 3.12 venv with ruff==0.15.14 and numpy>=1.26,<3 (lint.yml, python-scripts.yml)
#   - cargo fetch --locked
#   - venv tools exposed to later shells (/usr/local/bin links)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# lint.yml / python-scripts.yml use the runner's python3 (3.12 on ubuntu-latest).
uv python install 3.12
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python 'ruff==0.15.14' 'numpy>=1.26,<3'

# --- Prefetch crates (no build, no tests) ---
cargo fetch --locked

# Agent shells don't see this script's environment; link the venv's ruff onto PATH.
sudo ln -sfn "$(pwd)/.venv/bin/ruff" /usr/local/bin/ruff

# .venv/ is not in .gitignore here, so keep it out of `git status`.
exclude_file="$(git rev-parse --git-path info/exclude)"
mkdir -p "$(dirname "$exclude_file")"
grep -qxF '/.venv/' "$exclude_file" 2>/dev/null || echo '/.venv/' >>"$exclude_file"

echo "Cursor install for grok-ozempic finished."
