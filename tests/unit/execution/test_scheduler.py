"""Tests for trading scheduler configuration."""

import pytest

from hedgefund.scheduler.runner import create_scheduler


class TestCreateScheduler:
    def test_creates_three_jobs(self) -> None:
        scheduler = create_scheduler(lambda: None)
        jobs = scheduler.get_jobs()
        assert len(jobs) == 3

    def test_job_ids(self) -> None:
        scheduler = create_scheduler(lambda: None)
        job_ids = {job.id for job in scheduler.get_jobs()}
        assert job_ids == {
            "crypto_momentum_cycle",
            "etf_mean_reversion_cycle",
            "dual_momentum_cycle",
        }

    def test_job_names(self) -> None:
        scheduler = create_scheduler(lambda: None)
        job_names = {job.name for job in scheduler.get_jobs()}
        assert "Crypto Momentum Weekly Rebalance" in job_names
        assert "ETF Mean Reversion Weekly Rebalance" in job_names
        assert "Dual Momentum Monthly Rebalance" in job_names

    def test_custom_cron_expressions(self) -> None:
        scheduler = create_scheduler(
            lambda: None,
            crypto_cron="0 10 * * 1",
            etf_cron="0 15 * * 3",
            dual_cron="0 12 15 * *",
        )
        jobs = scheduler.get_jobs()
        assert len(jobs) == 3

    def test_misfire_grace_time(self) -> None:
        scheduler = create_scheduler(lambda: None)
        for job in scheduler.get_jobs():
            assert job.misfire_grace_time == 3600

    def test_custom_timezone(self) -> None:
        scheduler = create_scheduler(lambda: None, timezone="UTC")
        jobs = scheduler.get_jobs()
        assert len(jobs) == 3
