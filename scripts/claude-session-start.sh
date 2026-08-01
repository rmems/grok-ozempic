#!/usr/bin/env bash
# Session bootstrap for Claude Code (local CLI + CLAUDE_CODE_REMOTE cloud).
# Must always exit 0 so missing bd/cargo never bricks a cloud session.
#
# Usage:
#   claude-session-start.sh           # full: bd prime + optional cargo fetch
#   claude-session-start.sh prime     # lightweight: bd prime only (PreCompact)
set -u

MODE="${1:-full}"

if [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
  ROOT="$CLAUDE_PROJECT_DIR"
else
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
cd "$ROOT" || exit 0

if command -v bd >/dev/null 2>&1; then
  bd prime 2>/dev/null || true
fi

# Network fetch only on full SessionStart — not PreCompact.
# --locked: do not rewrite Cargo.lock. Bound with timeout/gtimeout only;
# if neither exists, skip fetch (cloud-safe: never hang SessionStart offline).
if [ "$MODE" = "full" ] && command -v cargo >/dev/null 2>&1; then
  if command -v timeout >/dev/null 2>&1; then
    timeout 60s cargo fetch --quiet --locked 2>/dev/null || true
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout 60s cargo fetch --quiet --locked 2>/dev/null || true
  fi
fi

exit 0
