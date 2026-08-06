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

TRAILER_NAME='Co-Authored-By'
COAUTHOR="${CLAUDE_COAUTHOR-Claude <noreply@anthropic.com>}"
case "$COAUTHOR" in 0 | "") exit 0 ;; esac

# The hook receives the tool call as JSON on stdin. Read it before anything else;
# `git commit --dry-run` and `git commit --help` both succeed without creating a
# commit, so acting on them would retroactively stamp whatever unpushed HEAD
# already existed -- a commit this call did not author.
# Bounded read: a plain `cat` blocks forever if stdin is never closed, which
# would hang the hook and silently skip the amend. 2s is ample for a JSON blob.
payload=$(timeout 2 cat 2>/dev/null || true)
cmd=$(printf '%s' "$payload" | jq -r '.tool_input.command // ""' 2>/dev/null || true)
case "$cmd" in
  *--dry-run* | *--help* | *' -h'* | *'git commit --amend'*) exit 0 ;;
esac

head=$(git rev-parse --verify HEAD 2>/dev/null) || exit 0

# Authoritative check that this call actually created a commit: compare HEAD with
# what the PreToolUse companion recorded before the command ran. A `git commit`
# variant that succeeds without committing leaves HEAD unchanged, and is skipped
# here rather than having its pre-existing HEAD retroactively stamped.
session=$(printf '%s' "$payload" | jq -r '.session_id // "nosession"' 2>/dev/null || echo nosession)
state="${TMPDIR:-/tmp}/claude-coauthor/$session.head"
if [ -r "$state" ]; then
  [ "$(cat "$state" 2>/dev/null)" = "$head" ] && exit 0
else
  # No recorded state (first run after a config change, or the PreToolUse hook
  # did not fire). Fall back to a recency bound so a commit made earlier in the
  # session is still never rewritten.
  committed=$(git log -1 --format=%ct 2>/dev/null || echo 0)
  [ $(($(date +%s) - committed)) -gt 120 ] && exit 0
fi

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
