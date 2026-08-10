#!/usr/bin/env bash
# Muse PostToolUse backstop: add a Muse co-author trailer only to a commit
# proven to have been created by the completed Muse Bash tool call.
#
# Repository-local lifecycle scope is intentional. A Git post-commit hook would
# also stamp commits made by the owner, Codex, Grok, or another agent.
#
# Env: MUSE_COAUTHOR overrides the identity; MUSE_COAUTHOR=0 disables.
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
COAUTHOR="${MUSE_COAUTHOR-Muse-Spark-1.2-contributor <noreply@muse-spark.rmems.com>}"
case "$COAUTHOR" in 0 | "") exit 0 ;; esac

payload=$(read_payload)
cmd=$(printf '%s' "$payload" | jq -r '.tool_input.command // ""' 2>/dev/null || true)
case "$cmd" in
  *"git commit"*) ;;
  *) exit 0 ;;
esac
case "$cmd" in
  *--dry-run* | *--help* | *' -h'* | *'git commit --amend'* | *'git push'*) exit 0 ;;
esac

head=$(git rev-parse --verify HEAD 2>/dev/null) || exit 0
reported=$(printf '%s' "$payload" |
  jq -r '(.tool_response.stdout // "") + "\n" + (.tool_response.stderr // "")' 2>/dev/null |
  sed -n 's/^\[[^]]* \([0-9a-f]\{7,40\}\)\].*/\1/p' |
  sed -n '1p')

if [ -n "$reported" ] && [ "$(git rev-parse --verify "$reported^{commit}" 2>/dev/null)" = "$head" ]; then
  :
else
  # Fallback proof must show HEAD was just created by this tool call; a
  # bare mention of "git commit" plus a branch checkout would otherwise
  # change HEAD without a new commit and trigger a false amend.
  session=$(printf '%s' "$payload" | jq -r '.session_id // ""' 2>/dev/null || true)
  state="${TMPDIR:-/tmp}/muse-coauthor/$(state_key "$session").head"
  [ -r "$state" ] || exit 0
  prev=$(cat "$state" 2>/dev/null)
  [ -n "$prev" ] || exit 0
  [ "$prev" = "$head" ] && exit 0
  if ! git merge-base --is-ancestor "$prev" "$head" 2>/dev/null; then
    exit 0
  fi
  if [ "$(git rev-list --count "$prev..$head" 2>/dev/null)" != "1" ]; then
    exit 0
  fi
fi

# Never amend a commit that was just pushed (including untracked remote refs).
case "${payload:-}" in
  *git\ push* ) exit 0 ;;
esac
case "${cmd:-}" in
  *git\ push* ) exit 0 ;;
esac

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
