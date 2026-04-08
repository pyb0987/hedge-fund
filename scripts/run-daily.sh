#!/bin/bash
# Daily paper trading runner with hard timeout.
# Invoked by ~/Library/LaunchAgents/com.hedgefund.paper.plist (under caffeinate -s).
# Prevents the launchd job from hanging indefinitely on stuck external API calls
# (Upbit / yfinance / Alpaca), which previously blocked subsequent scheduled runs.

set -u

cd /Users/fainders/personal/hedge-fund || exit 1
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

TIMEOUT=1800   # 30 minutes hard cap — far above the ~1 min normal cycle

./.venv/bin/python3 -m hedgefund daily-run &
PID=$!

(
    sleep "$TIMEOUT"
    if kill -0 "$PID" 2>/dev/null; then
        echo "[$(date)] daily-run exceeded ${TIMEOUT}s — sending SIGTERM" >&2
        kill -TERM "$PID" 2>/dev/null
        sleep 30
        if kill -0 "$PID" 2>/dev/null; then
            echo "[$(date)] still alive after 30s — SIGKILL" >&2
            kill -KILL "$PID" 2>/dev/null
        fi
    fi
) &
WATCHDOG=$!

wait "$PID"
EXIT=$?

kill "$WATCHDOG" 2>/dev/null
wait "$WATCHDOG" 2>/dev/null

exit "$EXIT"
