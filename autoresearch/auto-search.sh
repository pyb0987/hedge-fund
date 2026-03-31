#!/bin/bash
# auto-search.sh — Hedge Fund autoresearch launcher
# Creates a dedicated branch and optionally runs headless.
#
# Usage:
#   ./autoresearch/auto-search.sh                  # interactive mode
#   ./autoresearch/auto-search.sh headless 5.00    # headless, $5 budget

set -euo pipefail

MODE="${1:-interactive}"
MAX_BUDGET="${2:-5.00}"

# Ensure clean working tree
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: Working tree is dirty. Commit or stash changes first."
    exit 1
fi

# Create session branch
BRANCH="auto-search/session-$(date +%Y%m%d-%H%M%S)"
git checkout -b "$BRANCH"

echo "=== Hedge Fund Auto-Search ==="
echo "Branch: $BRANCH"
echo "Mode: $MODE"
echo "Budget: \$$MAX_BUDGET"
echo ""

# Delete stale data cache (force fresh data per session)
rm -f autoresearch/data_cache.pkl

if [ "$MODE" = "headless" ]; then
    echo "Starting headless autoresearch..."
    claude -p --dangerously-skip-permissions --max-budget-usd "$MAX_BUDGET" \
        "autoresearch/program.md를 읽고 auto-search loop를 실행하라. experiments.jsonl에서 마지막 실험 번호를 확인하고 이어서 진행."
else
    echo "Ready for interactive autoresearch."
    echo "Run: claude 'autoresearch/program.md를 읽고 auto-search loop를 실행하라'"
fi
