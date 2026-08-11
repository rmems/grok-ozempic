#!/usr/bin/env bash
set -euo pipefail

for required in git jq; do
  command -v "$required" >/dev/null 2>&1 || {
    echo "missing required command: $required" >&2
    exit 1
  }
done

repo_root=$(git rev-parse --show-toplevel)
pre_hook="$repo_root/.codex/hooks/coauthor-record-head.sh"
post_hook="$repo_root/.codex/hooks/coauthor-commit.sh"
muse_pre_hook="$repo_root/.muse/hooks/coauthor-record-head.sh"
muse_post_hook="$repo_root/.muse/hooks/coauthor-commit.sh"
test_root=$(mktemp -d)
trap 'rm -rf "$test_root"' EXIT

# Keep per-test coauthor state isolated so stale .head tokens are detectable.
export TMPDIR="$test_root"

# Project hook commands must be harmless when Codex invokes Bash elsewhere.
nonrepo="$test_root/nonrepo"
mkdir -p "$nonrepo"
for phase in PreToolUse PostToolUse; do
  command=$(jq -r ".hooks.${phase}[0].hooks[0].command" "$repo_root/.codex/hooks.json")
  (cd "$nonrepo" && bash -c "$command" </dev/null)
done

repo="$test_root/repo"
git init -q -b main "$repo"
git -C "$repo" config user.name Tester
git -C "$repo" config user.email tester@example.com
printf 'initial\n' > "$repo/tracked.txt"
git -C "$repo" add tracked.txt
git -C "$repo" commit -q -m initial

payload() {
  jq -cn \
    --arg session_id codex-hook-test \
    --arg command "$1" \
    --arg stdout "${2:-}" \
    '{session_id:$session_id,tool_input:{command:$command},tool_response:{stdout:$stdout,stderr:""}}'
}

run_hook() {
  hook=$1
  command=$2
  stdout=${3:-}
  payload "$command" "$stdout" > "$test_root/payload.json"
  (cd "$repo" && bash "$hook" < "$test_root/payload.json")
}

assert_trailer_count() {
  expected=$1
  actual=$(git -C "$repo" log -1 --format=%B | grep -cF 'Co-Authored-By: Codex <noreply@openai.com>' || true)
  [ "$actual" -eq "$expected" ] || {
    echo "expected $expected Codex trailers, got $actual" >&2
    exit 1
  }
}

assert_state_clean() {
  state_dir=${1:-codex-coauthor}
  if [ -d "$TMPDIR/$state_dir" ]; then
    remaining=$(find "$TMPDIR/$state_dir" -type f -name '*.head' 2>/dev/null || true)
    [ -z "$remaining" ] || {
      echo "stale state files remain after attribution: $remaining" >&2
      exit 1
    }
  fi
}

# Git's reported commit id is the strongest proof that this call made HEAD.
printf 'reported\n' >> "$repo/tracked.txt"
git -C "$repo" add tracked.txt
summary=$(git -C "$repo" commit -m reported)
run_hook "$post_hook" 'git commit -m reported' "$summary"
assert_trailer_count 1
once=$(git -C "$repo" rev-parse HEAD)
run_hook "$post_hook" 'git commit -m reported' "$summary"
[ "$(git -C "$repo" rev-parse HEAD)" = "$once" ]
assert_trailer_count 1
assert_state_clean

# Quoted text is not proof of a commit command. Even if HEAD subsequently moves
# by one commit, the fallback must not attribute the checked-out descendant.
base=$(git -C "$repo" rev-parse HEAD)
git -C "$repo" switch -q -c quoted-descendant
printf 'descendant\n' > "$repo/descendant.txt"
git -C "$repo" add descendant.txt
git -C "$repo" commit -q -m descendant
descendant=$(git -C "$repo" rev-parse HEAD)
git -C "$repo" switch -q --detach "$base"
run_hook "$pre_hook" 'echo "git commit" && git checkout quoted-descendant'
git -C "$repo" checkout -q "$descendant"
run_hook "$post_hook" 'echo "git commit" && git checkout quoted-descendant'
[ "$(git -C "$repo" rev-parse HEAD)" = "$descendant" ]
assert_trailer_count 0
assert_state_clean
# A separator inside quoted output is text too, not a command boundary.
run_hook "$pre_hook" 'printf "; git commit -m fake"'
run_hook "$post_hook" 'printf "; git commit -m fake"'
assert_trailer_count 0
assert_state_clean
git -C "$repo" switch -q main

# A compound command may echo "git commit" and still perform a real commit.
# The record-head guard must not confuse the two.
printf 'compound\n' > "$repo/compound.txt"
git -C "$repo" add compound.txt
run_hook "$pre_hook" 'echo "git commit" && git commit -q -m compound'
summary=$(git -C "$repo" commit -q -m compound)
run_hook "$post_hook" 'echo "git commit" && git commit -q -m compound' "$summary"
assert_trailer_count 1
assert_state_clean

# A skipped commit-and-push payload still consumes both agents' state tokens.
run_hook "$pre_hook" 'git commit -m skipped && git push origin HEAD'
run_hook "$post_hook" 'git commit -m skipped && git push origin HEAD'
assert_state_clean
run_hook "$muse_pre_hook" 'git commit -m skipped && git push origin HEAD'
run_hook "$muse_post_hook" 'git commit -m skipped && git push origin HEAD'
assert_state_clean muse-coauthor

# A later command that only echoes "git commit" must not reattribute HEAD.
run_hook "$pre_hook" 'echo "git commit"'
run_hook "$post_hook" 'echo "git commit"'
assert_trailer_count 1
assert_state_clean

# A quiet partial commit exercises the recorded-HEAD fallback and proves that
# message-only amend leaves the unrelated staged path intact.
printf 'partial\n' > "$repo/partial.txt"
printf 'still staged\n' > "$repo/staged.txt"
git -C "$repo" add partial.txt staged.txt
run_hook "$pre_hook" 'git commit -q --only -m partial -- partial.txt'
git -C "$repo" commit -q --only -m partial -- partial.txt
run_hook "$post_hook" 'git commit -q --only -m partial -- partial.txt'
assert_trailer_count 1
[ "$(git -C "$repo" diff --cached --name-only)" = staged.txt ]
git -C "$repo" commit -q -m staged

# Explicit disable leaves the new commit unattributed.
printf 'disabled\n' >> "$repo/tracked.txt"
git -C "$repo" add tracked.txt
summary=$(git -C "$repo" commit -m disabled)
payload 'git commit -m disabled' "$summary" > "$test_root/payload.json"
(cd "$repo" && CODEX_COAUTHOR=0 bash "$post_hook" < "$test_root/payload.json")
assert_trailer_count 0

# A published commit is immutable to the hook even if the payload names it.
remote="$test_root/remote.git"
git init -q --bare "$remote"
git -C "$repo" remote add origin "$remote"
printf 'published\n' >> "$repo/tracked.txt"
git -C "$repo" add tracked.txt
summary=$(git -C "$repo" commit -m published)
published=$(git -C "$repo" rev-parse HEAD)
git -C "$repo" push -q -u origin main
run_hook "$post_hook" 'git commit -m published' "$summary"
[ "$(git -C "$repo" rev-parse HEAD)" = "$published" ]
assert_trailer_count 0

# An explicit push refspec may publish HEAD without a local remote-tracking ref.
git -C "$repo" switch -q -c untracked-push
printf 'untracked push\n' >> "$repo/tracked.txt"
git -C "$repo" add tracked.txt
summary=$(git -C "$repo" commit -m untracked-push)
published=$(git -C "$repo" rev-parse HEAD)
git -C "$repo" push -q origin HEAD:refs/heads/untracked-push
run_hook "$post_hook" \
  'git commit -m untracked-push && git push origin HEAD:refs/heads/untracked-push' \
  "$summary"
[ "$(git -C "$repo" rev-parse HEAD)" = "$published" ]
assert_trailer_count 0
git -C "$repo" switch -q main

# A merge commit is also immutable to the hook.
git -C "$repo" switch -q -c side
printf 'side\n' > "$repo/side.txt"
git -C "$repo" add side.txt
git -C "$repo" commit -q -m side
git -C "$repo" switch -q main
printf 'main\n' > "$repo/main.txt"
git -C "$repo" add main.txt
git -C "$repo" commit -q -m main
git -C "$repo" merge -q --no-commit side
summary=$(git -C "$repo" commit -m merge)
merged=$(git -C "$repo" rev-parse HEAD)
run_hook "$post_hook" 'git commit -m merge' "$summary"
[ "$(git -C "$repo" rev-parse HEAD)" = "$merged" ]
assert_trailer_count 0

# Disabled attribution still consumes the PreToolUse token so a later quiet
# one-commit advance cannot reuse it as a fallback proof.
printf 'consume\n' >> "$repo/tracked.txt"
git -C "$repo" add tracked.txt
run_hook "$pre_hook" 'git commit -q -m consume'
git -C "$repo" commit -q -m consume
(cd "$repo" && CODEX_COAUTHOR=0 bash "$post_hook" < "$test_root/payload.json")
assert_trailer_count 0
assert_state_clean
printf 'after-disabled\n' > "$repo/after-disabled.txt"
git -C "$repo" add after-disabled.txt
git -C "$repo" commit -q -m after-disabled
run_hook "$post_hook" 'git commit -q -m after-disabled'
assert_trailer_count 0
assert_state_clean

# An escaped quote inside a double-quoted string does not end the string, so
# the following "; git commit" text is not treated as a real commit.
base=$(git -C "$repo" rev-parse HEAD)
git -C "$repo" switch -q -c escaped-descendant
printf 'escaped\n' > "$repo/escaped.txt"
git -C "$repo" add escaped.txt
git -C "$repo" commit -q -m escaped
escaped=$(git -C "$repo" rev-parse HEAD)
git -C "$repo" switch -q --detach "$base"
run_hook "$pre_hook" 'printf "foo\\\" ; git commit -m fake" && git checkout escaped-descendant'
git -C "$repo" checkout -q "$escaped"
run_hook "$post_hook" 'printf "foo\\\" ; git commit -m fake" && git checkout escaped-descendant'
[ "$(git -C "$repo" rev-parse HEAD)" = "$escaped" ]
assert_trailer_count 0
assert_state_clean
git -C "$repo" switch -q main

# git global options, env assignments, and command wrappers still trigger
# attribution because the parser looks past them to the commit subcommand.
for wrapped in \
  'git -C . commit -m optioned' \
  'git -c core.editor=true commit -m configed' \
  'env GIT_AUTHOR_NAME=Tester git commit -m enved' \
  'command git commit -m command'; do
  file=$(printf '%s' "$wrapped" | sed 's/[^a-zA-Z0-9]/-/g')
  printf '%s\n' "$file" > "$repo/$file.txt"
  git -C "$repo" add "$file.txt"
  run_hook "$pre_hook" "$wrapped"
  summary=$(git -C "$repo" commit -m "$file")
  run_hook "$post_hook" "$wrapped" "$summary"
  assert_trailer_count 1
  assert_state_clean
done

# Heredoc body containing "; git commit" must not be parsed as a real commit,
# even when the delimiter contains punctuation like EOF.1 or EOF!.
base=$(git -C "$repo" rev-parse HEAD)
for delim in EOF EOF.1 'EOF!' ; do
  git -C "$repo" switch -q -c "heredoc-$delim"
  printf '%s\n' "$delim" > "$repo/heredoc-$delim.txt"
  git -C "$repo" add "heredoc-$delim.txt"
  git -C "$repo" commit -q -m "heredoc-$delim"
  heredoc=$(git -C "$repo" rev-parse HEAD)
  git -C "$repo" switch -q --detach "$base"
  cmd="cat <<$delim
; git commit -m fake
$delim"
  run_hook "$pre_hook" "$cmd"
  git -C "$repo" checkout -q "$heredoc"
  run_hook "$post_hook" "$cmd"
  [ "$(git -C "$repo" rev-parse HEAD)" = "$heredoc" ]
  assert_trailer_count 0
  assert_state_clean
done
git -C "$repo" switch -q main

# A real git commit that uses a heredoc for its message still attributes.
msg_file="$test_root/heredoc-commit-msg"
cat > "$msg_file" <<'EOF'
message from heredoc
EOF
printf 'heredoc-commit\n' > "$repo/heredoc-commit.txt"
git -C "$repo" add heredoc-commit.txt
run_hook "$pre_hook" 'git commit -F - <<EOF
message from heredoc
EOF'
summary=$(git -C "$repo" commit -F "$msg_file")
run_hook "$post_hook" 'git commit -F - <<EOF
message from heredoc
EOF' "$summary"
assert_trailer_count 1
assert_state_clean

# Shell comments and trailing "#; git commit" must not be parsed as commands.
base=$(git -C "$repo" rev-parse HEAD)
git -C "$repo" switch -q -c commented-descendant
printf 'commented\n' > "$repo/commented.txt"
git -C "$repo" add commented.txt
git -C "$repo" commit -q -m commented
commented=$(git -C "$repo" rev-parse HEAD)
git -C "$repo" switch -q --detach "$base"
run_hook "$pre_hook" 'git checkout commented-descendant #; git commit'
git -C "$repo" checkout -q "$commented"
run_hook "$post_hook" 'git checkout commented-descendant #; git commit'
[ "$(git -C "$repo" rev-parse HEAD)" = "$commented" ]
assert_trailer_count 0
assert_state_clean
git -C "$repo" switch -q main

# A commit executed through backtick command substitution still attributes.
printf 'backtick\n' > "$repo/backtick.txt"
git -C "$repo" add backtick.txt
run_hook "$pre_hook" '`git commit -q -m backtick`'
summary=$(git -C "$repo" commit -q -m backtick)
run_hook "$post_hook" '`git commit -q -m backtick`' "$summary"
assert_trailer_count 1
assert_state_clean

echo "Codex co-author hook tests passed"
