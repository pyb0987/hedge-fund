"""Tests for Telegram notifier."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from hedgefund.monitoring.telegram import TelegramConfig, TelegramNotifier


@pytest.fixture
def config() -> TelegramConfig:
    return TelegramConfig(bot_token="test-token", chat_id="12345", enabled=True)


@pytest.fixture
def disabled_config() -> TelegramConfig:
    return TelegramConfig(bot_token="test-token", chat_id="12345", enabled=False)


class TestTelegramNotifier:
    def test_disabled_returns_false(self, disabled_config: TelegramConfig) -> None:
        notifier = TelegramNotifier(disabled_config)
        result = notifier.send_message("test")
        assert result is False
        notifier.close()

    @patch("hedgefund.monitoring.telegram.httpx.Client")
    def test_send_message_success(self, mock_client_cls: MagicMock, config: TelegramConfig) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        notifier = TelegramNotifier(config)
        result = notifier.send_message("Hello")

        assert result is True
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "sendMessage" in call_args[0][0]
        assert call_args[1]["json"]["text"] == "Hello"
        assert call_args[1]["json"]["chat_id"] == "12345"

        notifier.close()

    @patch("hedgefund.monitoring.telegram.httpx.Client")
    def test_send_message_http_error(
        self, mock_client_cls: MagicMock, config: TelegramConfig
    ) -> None:
        import httpx

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.HTTPError("Connection failed")
        mock_client_cls.return_value = mock_client

        notifier = TelegramNotifier(config)
        result = notifier.send_message("Hello")

        assert result is False
        notifier.close()

    @patch("hedgefund.monitoring.telegram.httpx.Client")
    def test_notify_cycle(self, mock_client_cls: MagicMock, config: TelegramConfig) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        notifier = TelegramNotifier(config)
        result = notifier.notify_cycle(
            timestamp=datetime(2024, 6, 1, 12, 0),
            signals=5,
            trades=3,
            commission=1500,
            portfolio_value=1_000_000,
            risk_blocked=False,
        )
        assert result is True

        sent_text = mock_client.post.call_args[1]["json"]["text"]
        assert "Signals: 5" in sent_text
        assert "Trades: 3" in sent_text

        notifier.close()

    @patch("hedgefund.monitoring.telegram.httpx.Client")
    def test_notify_risk_alert(self, mock_client_cls: MagicMock, config: TelegramConfig) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        notifier = TelegramNotifier(config)
        result = notifier.notify_risk_alert(
            reason="Max DD breached",
            drawdown=0.20,
            timestamp=datetime(2024, 6, 1),
        )
        assert result is True

        sent_text = mock_client.post.call_args[1]["json"]["text"]
        assert "RISK ALERT" in sent_text

        notifier.close()

    def test_close(self, config: TelegramConfig) -> None:
        notifier = TelegramNotifier(config)
        notifier.close()
        # No exception = success

    @patch("hedgefund.monitoring.telegram.time.sleep")
    @patch("hedgefund.monitoring.telegram.httpx.Client")
    def test_send_retries_on_transient_error_then_succeeds(
        self,
        mock_client_cls: MagicMock,
        mock_sleep: MagicMock,
        config: TelegramConfig,
    ) -> None:
        """ConnectError on first try, success on second — message goes through."""
        import httpx

        mock_client = MagicMock()
        ok_response = MagicMock()
        ok_response.raise_for_status = MagicMock()
        mock_client.post.side_effect = [
            httpx.ConnectError("dns lookup failed"),
            ok_response,
        ]
        mock_client_cls.return_value = mock_client

        notifier = TelegramNotifier(config)
        result = notifier.send_message("Hello")

        assert result is True
        assert mock_client.post.call_count == 2
        assert mock_sleep.call_count == 1
        notifier.close()

    @patch("hedgefund.monitoring.telegram.time.sleep")
    @patch("hedgefund.monitoring.telegram.httpx.Client")
    def test_send_gives_up_after_max_attempts(
        self,
        mock_client_cls: MagicMock,
        mock_sleep: MagicMock,
        config: TelegramConfig,
    ) -> None:
        """All attempts hit transient errors → False, no extra calls."""
        import httpx

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ReadTimeout("read timed out")
        mock_client_cls.return_value = mock_client

        notifier = TelegramNotifier(config)
        result = notifier.send_message("Hello")

        assert result is False
        assert mock_client.post.call_count == TelegramNotifier._MAX_ATTEMPTS
        assert mock_sleep.call_count == TelegramNotifier._MAX_ATTEMPTS - 1
        notifier.close()

    @patch("hedgefund.monitoring.telegram.time.sleep")
    @patch("hedgefund.monitoring.telegram.httpx.Client")
    def test_send_does_not_retry_4xx(
        self,
        mock_client_cls: MagicMock,
        mock_sleep: MagicMock,
        config: TelegramConfig,
    ) -> None:
        """HTTP 4xx (auth/bad request) is not transient — fail fast, no retry."""
        import httpx

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized", request=MagicMock(), response=MagicMock(status_code=401)
        )
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        notifier = TelegramNotifier(config)
        result = notifier.send_message("Hello")

        assert result is False
        mock_client.post.assert_called_once()
        mock_sleep.assert_not_called()
        notifier.close()
