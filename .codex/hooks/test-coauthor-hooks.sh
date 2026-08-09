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
test_root=$(mktemp -d)
trap 'rm -rf "$test_root"' EXIT

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

echo "Codex co-author hook tests passed"
