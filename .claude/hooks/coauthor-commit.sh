#!/usr/bin/env bash
# PostToolUse backstop: append the Claude co-author trailer to a commit that was
# just made, when the trailer is absent. Belt-and-braces behind
# attribution.commit in .claude/settings.json, for messages written by hand.
#
# Scoped to .claude/ deliberately: a git hook under core.hooksPath would apply to
# every committer in the repo, stamping Claude onto other agents' and the owner's
# commits.
#
# Env: CLAUDE_COAUTHOR overrides the identity; CLAUDE_COAUTHOR=0 disables.
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

TRAILER_NAME='Co-Authored-By'
COAUTHOR="${CLAUDE_COAUTHOR-Claude <noreply@anthropic.com>}"
case "$COAUTHOR" in 0 | "") exit 0 ;; esac

# The hook receives the tool call as JSON on stdin. Read it before anything else;
# `git commit --dry-run` and `git commit --help` both succeed without creating a
# commit, so acting on them would retroactively stamp whatever unpushed HEAD
# already existed -- a commit this call did not author.
# Bounded read: a plain `cat` blocks forever if stdin is never closed, which
# would hang the hook and silently skip the amend. 2s is ample for a JSON blob.
if command -v timeout >/dev/null 2>&1; then
  payload=$(timeout 2 cat 2>/dev/null || true)
else
  payload=$(read -t 2 -d '' _pl 2>/dev/null; printf '%s' "$_pl")
fi
cmd=$(printf '%s' "$payload" | jq -r '.tool_input.command // ""' 2>/dev/null || true)
case "$cmd" in
  *--dry-run* | *--help* | *' -h'* | *'git commit --amend'*) exit 0 ;;
esac

head=$(git rev-parse --verify HEAD 2>/dev/null) || exit 0

# Authoritative check that this call actually created a commit: compare HEAD with
# what the PreToolUse companion recorded before the command ran. A `git commit`
# variant that succeeds without committing leaves HEAD unchanged, and is skipped
# here rather than having its pre-existing HEAD retroactively stamped.
session=$(printf '%s' "$payload" | jq -r '.session_id // ""' 2>/dev/null || true)
state="${TMPDIR:-/tmp}/claude-coauthor/$(state_key "$session").head"
# No recorded state means we cannot prove this call created the commit, so do
# nothing. A timestamp fallback was tried and is unsafe: it amends any commit
# made within the window, including an unrelated one in a different repository
# the same session happened to touch. This is a convenience backstop --
# attribution.commit in settings.json is the primary mechanism -- so skipping is
# strictly better than guessing.
[ -r "$state" ] || exit 0
[ "$(cat "$state" 2>/dev/null)" = "$head" ] && exit 0

# A merge commit is detected by parent count, not MERGE_HEAD: git removes
# MERGE_HEAD as part of a successful commit, so post-commit it is always gone.
[ "$(git log -1 --format=%P | wc -w)" -gt 1 ] && exit 0

# Mid-rebase / cherry-pick / revert: HEAD is not a commit we authored.
for marker in rebase-merge rebase-apply CHERRY_PICK_HEAD REVERT_HEAD; do
  [ -e "$(git rev-parse --git-path "$marker")" ] && exit 0
done

# Never rewrite history that is already published.
[ -n "$(git branch -r --contains HEAD 2>/dev/null)" ] && exit 0

# Idempotent.
git log -1 --format=%B | grep -qF "$TRAILER_NAME: $COAUTHOR" && exit 0

# --only is essential: a plain --amend re-reads the index, so after a partial
# commit (`git commit -- path`, `git commit -o`) it would silently fold the still
# staged files into the commit and clear the staging area. --only amends the
# message alone and leaves the index untouched.
git commit --amend --only --no-edit --trailer "$TRAILER_NAME=$COAUTHOR" >/dev/null 2>&1 || exit 0
