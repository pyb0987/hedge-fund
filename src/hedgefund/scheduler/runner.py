"""Daily paper trading scheduler.

cwaa 프로젝트 패턴 기반:
- launchd plist (macOS) / cron (Linux) 로 외부 트리거
- UTC 기반 일일 멱등성 가드 (DarkWake 방지)
- 하루 1회 실행: paper-run → report 생성 → Telegram digest 전송

사용:
    uv run python -m hedgefund daily-run          # 1회 실행 (가드 적용)
    uv run python -m hedgefund install-scheduler   # launchd plist 설치
"""

import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_CYCLE_TIMEOUT = 600  # 10 minutes — safety net for entire paper-run
PLIST_NAME = "com.hedgefund.paper"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{PLIST_NAME}.plist"
RUN_LOG_PATH = Path("data/paper_state/run_log.json")
LOG_DIR = Path.home() / "Library" / "Logs" / "hedgefund"


# --- Daily idempotency guard (UTC-based, cwaa pattern) ---


def has_run_today(log_path: Path = RUN_LOG_PATH) -> bool:
    """Check if daily cycle already ran today (UTC date)."""
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    if not log_path.exists():
        return False
    try:
        data = json.loads(log_path.read_text())
        return data.get("last_run_date") == today
    except (json.JSONDecodeError, OSError):
        return False


def mark_run_today(log_path: Path = RUN_LOG_PATH) -> None:
    """Mark today's run as completed (UTC date)."""
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            {
                "last_run_date": today,
                "last_run_utc": datetime.now(UTC).isoformat(),
            }
        )
    )


# --- Daily cycle: paper-run → report → Telegram digest ---


def run_daily_cycle(
    config_dir: Path = Path("config"),
    force: bool = False,
) -> bool:
    """Execute the full daily paper trading cycle.

    1. Check idempotency guard (skip if already ran today)
    2. Run paper trading cycle
    3. Generate validation report
    4. Send Telegram daily digest

    Args:
        config_dir: path to config directory
        force: skip idempotency guard

    Returns:
        True if cycle executed, False if skipped
    """
    if not force and has_run_today():
        logger.info("Already ran today (UTC) — skipping")
        return False

    logger.info("=== Daily Paper Trading Cycle Start ===")

    from hedgefund.app import run_once
    from hedgefund.config.loader import load_global_settings
    from hedgefund.data.store import DataStore
    from hedgefund.monitoring.paper_report import (
        ReportThresholds,
        format_report,
        generate_report,
    )

    # Step 1: Run paper trading cycle (with timeout safety net)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(run_once, config_dir=config_dir, dry_run=False)
            result = future.result(timeout=_CYCLE_TIMEOUT)
        logger.info(
            "Paper run complete: signals=%d, trades=%d, value=%.0f",
            len(result.cycle_result.signals),
            result.cycle_result.num_trades,
            result.portfolio_value,
        )
    except FuturesTimeout:
        logger.error("Paper run timed out after %ds", _CYCLE_TIMEOUT)
        _send_error_notification(
            config_dir,
            f"Paper run timed out after {_CYCLE_TIMEOUT}s\n"
            "API 응답 없음 — 네트워크 또는 거래소 점검 확인 필요",
        )
        mark_run_today()
        return False
    except Exception:
        tb = traceback.format_exc()
        logger.exception("Paper run failed")
        _send_error_notification(config_dir, f"Paper run failed\n\n<pre>{tb[-500:]}</pre>")
        return False

    # Step 2: Generate validation report
    try:
        settings = load_global_settings(config_dir)
        store = DataStore(settings.data.sqlite_path)
        report = generate_report(store, ReportThresholds())
        report_text = format_report(report)
        logger.info("Report generated:\n%s", report_text)
    except Exception:
        tb = traceback.format_exc()
        logger.exception("Report generation failed")
        _send_error_notification(config_dir, f"Report generation failed\n\n<pre>{tb[-500:]}</pre>")
        mark_run_today()
        return True

    # Step 3: Send Telegram daily digest (includes validation data)
    try:
        tg_config = settings.monitoring.telegram
        if tg_config.enabled and tg_config.bot_token:
            from hedgefund.monitoring.telegram import TelegramConfig, TelegramNotifier

            notifier = TelegramNotifier(
                TelegramConfig(
                    bot_token=tg_config.bot_token,
                    chat_id=tg_config.chat_id,
                    enabled=True,
                )
            )
            notifier.notify_daily_digest(report)
            notifier.close()
            logger.info("Telegram daily digest sent")
    except Exception:
        logger.exception("Telegram digest failed — continuing")

    mark_run_today()
    logger.info("=== Daily Paper Trading Cycle Complete ===")
    return True


def _send_error_notification(config_dir: Path, message: str) -> None:
    """Send error notification via Telegram."""
    try:
        from hedgefund.config.loader import load_global_settings
        from hedgefund.monitoring.telegram import TelegramConfig, TelegramNotifier

        settings = load_global_settings(config_dir)
        tg = settings.monitoring.telegram
        if not tg.enabled or not tg.bot_token:
            return

        notifier = TelegramNotifier(
            TelegramConfig(
                bot_token=tg.bot_token,
                chat_id=tg.chat_id,
                enabled=True,
            )
        )
        notifier.send_message(f"<b>ERROR</b>\n{message}")
        notifier.close()
    except Exception:
        logger.exception("Error notification also failed")


# --- launchd plist installer (macOS, cwaa pattern) ---


def generate_plist(
    project_dir: Path,
    python_path: Path,
    hour: int = 21,
    minute: int = 0,
) -> str:
    """Generate launchd plist XML for daily paper trading.

    Uses StartCalendarInterval for exact 21:00 KST trigger.
    caffeinate -s prevents sleep during execution.

    Args:
        project_dir: absolute path to project root
        python_path: absolute path to Python interpreter
        hour: local hour to run (default: 21 = 9PM KST)
        minute: local minute to run (default: 0)
    """
    log_dir = LOG_DIR
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{PLIST_NAME}</string>

    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/caffeinate</string>
        <string>-s</string>
        <string>{python_path}</string>
        <string>-m</string>
        <string>hedgefund</string>
        <string>daily-run</string>
    </array>

    <key>WorkingDirectory</key>
    <string>{project_dir}</string>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{hour}</integer>
        <key>Minute</key>
        <integer>{minute}</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>{log_dir}/hedgefund.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/hedgefund-error.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:{project_dir}/.venv/bin</string>
    </dict>
</dict>
</plist>"""


def install_plist(
    project_dir: Path | None = None,
    python_path: Path | None = None,
    hour: int = 21,
    minute: int = 0,
) -> Path:
    """Install launchd plist for daily paper trading.

    Returns the installed plist path.
    """
    if project_dir is None:
        project_dir = Path.cwd()
    if python_path is None:
        python_path = Path(sys.executable)

    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Generate and write plist
    plist_content = generate_plist(project_dir, python_path, hour, minute)
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.write_text(plist_content)
    logger.info("Plist written to %s", PLIST_PATH)

    # Unload existing (ignore errors if not loaded)
    subprocess.run(
        ["launchctl", "unload", str(PLIST_PATH)],
        capture_output=True,
    )

    # Load new plist
    result = subprocess.run(
        ["launchctl", "load", str(PLIST_PATH)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("launchctl load failed: %s", result.stderr)
    else:
        logger.info("launchd job loaded: %s", PLIST_NAME)

    return PLIST_PATH


def uninstall_plist() -> bool:
    """Uninstall launchd plist."""
    if not PLIST_PATH.exists():
        return False

    subprocess.run(
        ["launchctl", "unload", str(PLIST_PATH)],
        capture_output=True,
    )
    PLIST_PATH.unlink()
    logger.info("Plist removed: %s", PLIST_PATH)
    return True
