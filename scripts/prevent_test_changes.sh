#!/usr/bin/env bash
set -euo pipefail

# Block commits that modify tests unless an override label is present in the commit message
# Allow override by including string: [allow-test-changes]

if git rev-parse --verify HEAD >/dev/null 2>&1; then
  against=HEAD
else
  # Initial commit
  against=4b825dc642cb6eb9a060e54bf8d69288fbee4904
fi

changed_files=$(git diff --name-only --cached "$against")

if echo "$changed_files" | grep -E "^(tests/|.*/tests/|test_/|.*/test_).+\.py$" >/dev/null 2>&1; then
  # Fetch commit message from prepared msg file if running in commit-msg stage
  commit_msg_file=${1:-}
  if [ -n "$commit_msg_file" ] && [ -f "$commit_msg_file" ]; then
    if grep -q "\[allow-test-changes\]" "$commit_msg_file"; then
      echo "[prevent-test-changes] Override detected in commit message. Allowing test modifications."
      exit 0
    fi
  fi

  # Also allow override via environment variable for CI use-cases
  if [ "${ALLOW_TEST_CHANGES:-}" = "1" ]; then
    echo "[prevent-test-changes] Override via ALLOW_TEST_CHANGES=1. Allowing test modifications."
    exit 0
  fi

  echo "[prevent-test-changes] Commit modifies test files but no override provided."
  echo "Add [allow-test-changes] to your commit message or set ALLOW_TEST_CHANGES=1 to proceed."
  echo "Changed test files:"
  echo "$changed_files" | grep -E "^(tests/|.*/tests/|test_/|.*/test_).+\.py$" || true
  exit 1
fi

exit 0
