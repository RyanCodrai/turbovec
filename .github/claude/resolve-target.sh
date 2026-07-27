#!/usr/bin/env bash
# Deterministically resolve and check out the TRUE target branch — outside the
# LLM. On issue_comment the runner starts on `main`; leaving branch placement to
# the model's memory is exactly how work lands nowhere. This fixes it in bash.
#
# Reads (env): GOAL_EVENT_NAME, GOAL_PAYLOAD_PATH, GITHUB_REPOSITORY, GH_TOKEN.
# Writes: branch=, base=, pr=  to $GITHUB_OUTPUT.
set -euo pipefail

event="$GOAL_EVENT_NAME"
payload="$GOAL_PAYLOAD_PATH"
pr=""

if [ "$event" = "issue_comment" ]; then
  # Issue comments fire for PRs too; .issue.pull_request is present only for PRs.
  if jq -e '.issue.pull_request' "$payload" >/dev/null 2>&1; then
    pr="$(jq -r '.issue.number' "$payload")"
  fi
elif [ "$event" = "pull_request_review_comment" ] || [ "$event" = "pull_request_review" ]; then
  pr="$(jq -r '.pull_request.number' "$payload")"
fi

if [ -n "$pr" ]; then
  # Fork PRs: the head ref isn't on origin and we could never push to it — bail
  # loudly rather than dying under `set -e` with no explanation.
  if [ "$(gh pr view "$pr" -R "$GITHUB_REPOSITORY" --json isCrossRepository --jq '.isCrossRepository')" = "true" ]; then
    gh pr comment "$pr" -R "$GITHUB_REPOSITORY" --body \
      "Harness can't auto-deliver to a fork PR (no push access to the fork branch). Please merge manually."
    exit 1
  fi
  # Work on the PR's own head branch.
  read -r branch base < <(
    gh pr view "$pr" -R "$GITHUB_REPOSITORY" --json headRefName,baseRefName \
      --jq '"\(.headRefName) \(.baseRefName)"'
  )
  git fetch origin "$branch" "$base"
  git checkout "$branch"
  git pull --ff-only origin "$branch" || true
else
  # Issue-origin: create a fresh working branch off main; a PR is opened later.
  base="main"
  issue="$(jq -r '.issue.number // "adhoc"' "$payload")"
  branch="claude/harness-${issue}"
  git fetch origin main
  git checkout -B "$branch" origin/main
fi

{
  echo "branch=$branch"
  echo "base=$base"
  echo "pr=$pr"
} >> "$GITHUB_OUTPUT"

echo "Resolved: branch=$branch base=$base pr=${pr:-<none>}"
