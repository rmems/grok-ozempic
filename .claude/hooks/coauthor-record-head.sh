#!/usr/bin/env bash
# PreToolUse companion to coauthor-commit.sh: records HEAD *before* a git commit
# runs, so the PostToolUse side can tell whether the command actually created a
# commit rather than inferring it from timestamps and command text.
#
# PreToolUse and PostToolUse are separate processes and cannot share state in
# memory, so it goes through a file keyed by the session id.
set -u
# Key the state by session *and* repository. Keying on session alone lets one
# session that touches several repos compare repo A's recorded HEAD against repo
# B's current HEAD -- they always differ, so a no-op commit in B would look like a
# new commit and B's unrelated HEAD would be amended.
# The session id reaches us from a JSON payload, so it is sanitized rather than
# interpolated into a path directly.
state_key() {
  _sess=$(printf '%s' "${1:-}" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-64)
  [ -n "$_sess" ] || _sess=nosession
  _repo=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || echo norepo)
  _repo=$(printf '%s' "$_repo" | cksum | cut -d' ' -f1)
  printf '%s-%s' "$_sess" "$_repo"
}

state_dir="${TMPDIR:-/tmp}/claude-coauthor"
mkdir -p "$state_dir" 2>/dev/null || exit 0
if command -v timeout >/dev/null 2>&1; then
  payload=$(timeout 2 cat 2>/dev/null || true)
else
  # `-r` as in coauthor-commit.sh: without it a backslash in the JSON is eaten
  # and jq yields an empty session_id, so this side records HEAD under the
  # `nosession` key while the PostToolUse side may key it correctly (or not) --
  # a mismatch that either skips the amend or, worse, matches another session.
  payload=$(IFS= read -r -t 2 -d '' _pl 2>/dev/null; printf '%s' "$_pl")
fi
# Self-gate, mirroring coauthor-commit.sh: settings.json carries no `if`
# condition (the `Bash(git commit*)` form anchors at the start of the command and
# never matched `cd <dir> && git commit ...`), so this runs on every Bash call.
# Recording HEAD for unrelated commands would overwrite the value the pending
# commit needs, which is one way the comparison ends up reading equal.
cmd=$(printf '%s' "$payload" | jq -r '.tool_input.command // ""' 2>/dev/null || true)
case "$cmd" in
  *"git commit"*) ;;
  *) exit 0 ;;
esac

session=$(printf '%s' "$payload" | jq -r '.session_id // ""' 2>/dev/null || true)
head=$(git rev-parse --verify HEAD 2>/dev/null || echo none)
printf '%s\n' "$head" > "$state_dir/$(state_key "$session").head" 2>/dev/null || true
exit 0
