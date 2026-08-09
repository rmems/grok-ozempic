#!/usr/bin/env bash
# Codex PostToolUse backstop: add a Codex co-author trailer only to a commit
# proven to have been created by the completed Codex Bash tool call.
#
# Repository-local lifecycle scope is intentional. A Git post-commit hook would
# also stamp commits made by the owner, Claude, Grok, or another agent.
#
# Env: CODEX_COAUTHOR overrides the identity; CODEX_COAUTHOR=0 disables.
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

TRAILER_NAME='Co-Authored-By'
COAUTHOR="${CODEX_COAUTHOR-Codex <noreply@openai.com>}"
case "$COAUTHOR" in 0 | "") exit 0 ;; esac

payload=$(read_payload)
cmd=$(printf '%s' "$payload" | jq -r '.tool_input.command // ""' 2>/dev/null || true)
case "$cmd" in
  *"git commit"*) ;;
  *) exit 0 ;;
esac
case "$cmd" in
  *--dry-run* | *--help* | *' -h'* | *'git commit --amend'*) exit 0 ;;
esac

head=$(git rev-parse --verify HEAD 2>/dev/null) || exit 0
reported=$(printf '%s' "$payload" |
  jq -r '(.tool_response.stdout // "") + "\n" + (.tool_response.stderr // "")' 2>/dev/null |
  sed -n 's/^\[[^]]* \([0-9a-f]\{7,40\}\)\].*/\1/p' | head -n 1)

if [ -n "$reported" ] && [ "$(git rev-parse --verify "$reported^{commit}" 2>/dev/null)" = "$head" ]; then
  :
else
  session=$(printf '%s' "$payload" | jq -r '.session_id // ""' 2>/dev/null || true)
  state="${TMPDIR:-/tmp}/codex-coauthor/$(state_key "$session").head"
  [ -r "$state" ] || exit 0
  [ "$(cat "$state" 2>/dev/null)" = "$head" ] && exit 0
fi

# Never rewrite merges, in-progress history operations, or published history.
[ "$(git log -1 --format=%P | wc -w)" -gt 1 ] && exit 0
for marker in rebase-merge rebase-apply CHERRY_PICK_HEAD REVERT_HEAD; do
  [ -e "$(git rev-parse --git-path "$marker")" ] && exit 0
done
[ -n "$(git branch -r --contains HEAD 2>/dev/null)" ] && exit 0

git log -1 --format=%B | grep -qF "$TRAILER_NAME: $COAUTHOR" && exit 0

# --only changes the message without folding unrelated staged paths into a
# partial commit or clearing them from the index.
git commit --amend --only --no-edit --trailer "$TRAILER_NAME=$COAUTHOR" >/dev/null 2>&1 || exit 0
