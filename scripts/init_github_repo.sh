#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/init_github_repo.sh [github_repo_url]
#
# Examples:
#   ./scripts/init_github_repo.sh
#   ./scripts/init_github_repo.sh git@github.com:<user>/<repo>.git

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git init -b main
fi

# Ensure main branch name is stable
current_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
if [[ "${current_branch}" != "main" ]]; then
  git branch -M main
fi

git add -A

if ! git diff --cached --quiet; then
  git commit -m "chore: initialize repo with backup-safe ignore rules"
else
  echo "[init_github_repo] Nothing to commit"
fi

if [[ $# -ge 1 && -n "${1:-}" ]]; then
  repo_url="$1"
  if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "${repo_url}"
  else
    git remote add origin "${repo_url}"
  fi
  git push -u origin main
  echo "[init_github_repo] Pushed to ${repo_url}"
else
  echo "[init_github_repo] Local repo initialized."
  echo "[init_github_repo] Next: ./scripts/init_github_repo.sh <github_repo_url>"
fi
