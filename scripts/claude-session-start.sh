#!/usr/bin/env bash
# Session bootstrap for Claude Code (local CLI + CLAUDE_CODE_REMOTE cloud).
# Must always exit 0 so missing bd/cargo never bricks a cloud session.
set -u

ROOT="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT" || exit 0

if command -v bd >/dev/null 2>&1; then
  bd prime 2>/dev/null || true
fi

# Warm the crate graph when cargo + network are available (cached envs stay fast).
if command -v cargo >/dev/null 2>&1; then
  cargo fetch --quiet 2>/dev/null || true
fi

exit 0
