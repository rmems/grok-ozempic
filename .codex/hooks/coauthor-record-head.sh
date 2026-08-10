#!/usr/bin/env bash
# PreToolUse companion for coauthor-commit.sh. It records HEAD before a Codex
# Bash tool runs a git commit, allowing the PostToolUse hook to prove that the
# tool call created a commit even when `git commit -q` emits no commit summary.
set -u

state_key() {
  _sess=$(printf '%s' "${1:-}" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-64)
  [ -n "$_sess" ] || _sess=nosession
  _repo=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || echo norepo)
  _repo=$(printf '%s' "$_repo" | cksum | cut -d' ' -f1)
  printf '%s-%s' "$_sess" "$_repo"
}

read_payload() {
  if command -v timeout >/dev/null 2>&1; then
    timeout 2 cat 2>/dev/null || true
  else
    IFS= read -r -t 2 -d '' _pl 2>/dev/null
    printf '%s' "${_pl:-}"
  fi
}

has_git_commit_command() {
  # Match an executable git commit command, ignoring text inside quoted or
  # escaped strings and allowing wrappers like env/command and git globals
  # such as -C, -c, --git-dir, --work-tree, --no-pager, etc.
  # Pass the whole command as a single AWK variable so heredoc body lines do
  # not become separate records and get parsed as executable commands.
  awk -v cmd="${1:-}" -f "$(dirname "$0")/has_git_commit_command.awk" 2>/dev/null
}

payload=$(read_payload)
cmd=$(printf '%s' "$payload" | jq -r '.tool_input.command // ""' 2>/dev/null || true)
has_git_commit_command "$cmd" || exit 0
case "$cmd" in
  *--dry-run* | *--help* | *' -h'* | *'git commit --amend'*) exit 0 ;;
esac

state_dir="${TMPDIR:-/tmp}/codex-coauthor"
mkdir -p "$state_dir" 2>/dev/null || exit 0
session=$(printf '%s' "$payload" | jq -r '.session_id // ""' 2>/dev/null || true)
head=$(git rev-parse --verify HEAD 2>/dev/null || echo none)
printf '%s\n' "$head" > "$state_dir/$(state_key "$session").head" 2>/dev/null || true
exit 0
