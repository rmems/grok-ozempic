#!/usr/bin/env bash
# PreToolUse companion to coauthor-commit.sh: records HEAD *before* a git commit
# runs, so the PostToolUse side can tell whether the command actually created a
# commit rather than inferring it from timestamps and command text.
#
# PreToolUse and PostToolUse are separate processes and cannot share state in
# memory, so it goes through a file keyed by the session id.
set -u
state_dir="${TMPDIR:-/tmp}/claude-coauthor"
mkdir -p "$state_dir" 2>/dev/null || exit 0
payload=$(timeout 2 cat 2>/dev/null || true)
session=$(printf '%s' "$payload" | jq -r '.session_id // "nosession"' 2>/dev/null || echo nosession)
head=$(git rev-parse --verify HEAD 2>/dev/null || echo none)
printf '%s\n' "$head" > "$state_dir/$session.head" 2>/dev/null || true
exit 0
