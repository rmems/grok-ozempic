#!/usr/bin/env bash
# PostToolUse backstop: append the Claude co-author trailer to the commit just
# made, if absent. Only ever touches an UNPUSHED HEAD.
set -u
T='Co-Authored-By: Claude <noreply@anthropic.com>'
git rev-parse --verify HEAD >/dev/null 2>&1 || exit 0
# Skip a merge commit. Detect it by parent count, NOT by MERGE_HEAD: git deletes
# MERGE_HEAD once the commit succeeds, so post-commit that file is always gone.
[ "$(git log -1 --format=%P | wc -w)" -gt 1 ] && exit 0
# Skip mid-rebase / mid-cherry-pick / mid-revert: HEAD is not a commit we authored.
for p in rebase-merge rebase-apply CHERRY_PICK_HEAD REVERT_HEAD; do
  [ -e "$(git rev-parse --git-path "$p")" ] && exit 0
done
# Never rewrite a commit that is already on a remote.
[ -n "$(git branch -r --contains HEAD 2>/dev/null)" ] && exit 0
# Idempotent.
git log -1 --format=%B | grep -qF "$T" && exit 0
git commit --amend --no-edit --trailer "Co-Authored-By=Claude <noreply@anthropic.com>" >/dev/null 2>&1 || exit 0
