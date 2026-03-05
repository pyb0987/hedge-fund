"""APScheduler-based trading scheduler.

전략별 리밸런싱 주기에 따라 run_cycle을 자동 실행합니다.
- Crypto Momentum: 주간 (일요일 21:00 KST)
- ETF Mean Reversion: 주간 (금요일 22:00 KST / US market close)
- Dual Momentum: 월간 (매월 1일)
"""

import logging
from datetime import datetime
from typing import Callable

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)


def create_scheduler(
    run_cycle_fn: Callable[[], None],
    crypto_cron: str = "0 21 * * 0",  # Sunday 21:00
    etf_cron: str = "0 22 * * 5",  # Friday 22:00
    dual_cron: str = "0 9 1 * *",  # 1st of month 09:00
    timezone: str = "Asia/Seoul",
) -> BlockingScheduler:
    """Create and configure the trading scheduler.

    Args:
        run_cycle_fn: function to call on each schedule trigger
        crypto_cron: cron expression for crypto momentum
        etf_cron: cron expression for ETF mean reversion
        dual_cron: cron expression for dual momentum
        timezone: timezone for cron triggers

    Returns:
        Configured BlockingScheduler (call .start() to run)
    """
    scheduler = BlockingScheduler(timezone=timezone)

    scheduler.add_job(
        run_cycle_fn,
        trigger=CronTrigger.from_crontab(crypto_cron, timezone=timezone),
        id="crypto_momentum_cycle",
        name="Crypto Momentum Weekly Rebalance",
        misfire_grace_time=3600,
    )

    scheduler.add_job(
        run_cycle_fn,
        trigger=CronTrigger.from_crontab(etf_cron, timezone=timezone),
        id="etf_mean_reversion_cycle",
        name="ETF Mean Reversion Weekly Rebalance",
        misfire_grace_time=3600,
    )

    scheduler.add_job(
        run_cycle_fn,
        trigger=CronTrigger.from_crontab(dual_cron, timezone=timezone),
        id="dual_momentum_cycle",
        name="Dual Momentum Monthly Rebalance",
        misfire_grace_time=3600,
    )

    logger.info("Scheduler configured with 3 jobs (timezone: %s)", timezone)
    return scheduler


def run_scheduler(scheduler: BlockingScheduler) -> None:
    """Start the scheduler (blocking)."""
    logger.info("Starting trading scheduler...")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user")
    finally:
        scheduler.shutdown(wait=False)
