"""Telegram Bot API client for paper trading notifications.

Uses httpx (sync) to send messages via the Telegram Bot API.
All send operations are fire-and-forget — failures are logged, never raised.

Transient network failures (DNS, connect, read timeout, server-side disconnect)
are retried with backoff. Authentication / 4xx errors are NOT retried.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime

import httpx

from hedgefund.monitoring.formatters import (
    format_cycle_html,
    format_daily_digest_html,
    format_risk_alert_html,
)
from hedgefund.monitoring.paper_report import PaperReport

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"

_TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)


@dataclass(frozen=True)
class TelegramConfig:
    """Telegram notification configuration."""

    bot_token: str
    chat_id: str
    enabled: bool = True


class TelegramNotifier:
    """Sync Telegram Bot API client."""

    _MAX_ATTEMPTS = 3
    _RETRY_BACKOFF: tuple[float, ...] = (1.0, 3.0)

    def __init__(self, config: TelegramConfig) -> None:
        self._config = config
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0),
            transport=httpx.HTTPTransport(retries=2),
        )
        self._base_url = f"{TELEGRAM_API_BASE}/bot{config.bot_token}"

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message. Returns True on success, False on failure.

        Transient errors are retried up to ``_MAX_ATTEMPTS`` times with
        ``_RETRY_BACKOFF`` between attempts. Non-transient errors (HTTP 4xx,
        decoding) fail immediately.
        """
        if not self._config.enabled:
            return False

        last_error: BaseException | None = None
        for attempt in range(self._MAX_ATTEMPTS):
            try:
                response = self._client.post(
                    f"{self._base_url}/sendMessage",
                    json={
                        "chat_id": self._config.chat_id,
                        "text": text,
                        "parse_mode": parse_mode,
                    },
                )
                response.raise_for_status()
                return True
            except _TRANSIENT_EXCEPTIONS as e:
                last_error = e
                if attempt + 1 >= self._MAX_ATTEMPTS:
                    break
                backoff = self._RETRY_BACKOFF[min(attempt, len(self._RETRY_BACKOFF) - 1)]
                logger.info(
                    "Telegram transient error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1,
                    self._MAX_ATTEMPTS,
                    e,
                    backoff,
                )
                time.sleep(backoff)
            except httpx.HTTPError as e:
                logger.warning("Telegram send failed: %s", e)
                return False

        logger.warning(
            "Telegram send failed after %d attempts: %s",
            self._MAX_ATTEMPTS,
            last_error,
        )
        return False

    def notify_cycle(
        self,
        timestamp: datetime,
        signals: int,
        trades: int,
        commission: float,
        portfolio_value: float,
        risk_blocked: bool,
        risk_reason: str | None = None,
        errors: tuple[str, ...] = (),
    ) -> bool:
        """Send cycle summary notification."""
        message = format_cycle_html(
            timestamp=timestamp,
            signals=signals,
            trades=trades,
            commission=commission,
            portfolio_value=portfolio_value,
            risk_blocked=risk_blocked,
            risk_reason=risk_reason,
            errors=errors,
        )
        return self.send_message(message)

    def notify_risk_alert(
        self,
        reason: str,
        drawdown: float,
        timestamp: datetime,
    ) -> bool:
        """Send high-priority risk alert."""
        message = format_risk_alert_html(reason, drawdown, timestamp)
        return self.send_message(message)

    def notify_daily_digest(self, report: PaperReport) -> bool:
        """Send daily performance digest."""
        message = format_daily_digest_html(report)
        return self.send_message(message)

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
