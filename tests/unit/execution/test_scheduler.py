"""Tests for daily paper trading scheduler."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hedgefund.scheduler.runner import (
    generate_plist,
    has_run_today,
    mark_run_today,
)


class TestIdempotencyGuard:
    def test_no_log_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "run_log.json"
        assert has_run_today(log_path) is False

    def test_mark_and_check(self, tmp_path: Path) -> None:
        log_path = tmp_path / "run_log.json"
        mark_run_today(log_path)
        assert has_run_today(log_path) is True

    def test_different_day(self, tmp_path: Path) -> None:
        log_path = tmp_path / "run_log.json"
        log_path.write_text(json.dumps({"last_run_date": "2020-01-01"}))
        assert has_run_today(log_path) is False

    def test_same_day(self, tmp_path: Path) -> None:
        log_path = tmp_path / "run_log.json"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_path.write_text(json.dumps({"last_run_date": today}))
        assert has_run_today(log_path) is True

    def test_corrupted_log(self, tmp_path: Path) -> None:
        log_path = tmp_path / "run_log.json"
        log_path.write_text("not valid json{{{")
        assert has_run_today(log_path) is False

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        log_path = tmp_path / "nested" / "dir" / "run_log.json"
        mark_run_today(log_path)
        assert log_path.exists()
        assert has_run_today(log_path) is True


class TestGeneratePlist:
    def test_contains_label(self) -> None:
        plist = generate_plist(Path("/opt/hedgefund"), Path("/usr/bin/python3"))
        assert "com.hedgefund.paper" in plist

    def test_contains_project_dir(self) -> None:
        plist = generate_plist(Path("/opt/hedgefund"), Path("/usr/bin/python3"))
        assert "/opt/hedgefund" in plist

    def test_contains_python_path(self) -> None:
        plist = generate_plist(Path("/opt/hedgefund"), Path("/usr/bin/python3"))
        assert "/usr/bin/python3" in plist

    def test_default_hour(self) -> None:
        plist = generate_plist(Path("/opt/hedgefund"), Path("/usr/bin/python3"))
        assert "<integer>21</integer>" in plist

    def test_custom_hour(self) -> None:
        plist = generate_plist(
            Path("/opt/hedgefund"), Path("/usr/bin/python3"), hour=9, minute=30
        )
        assert "<integer>9</integer>" in plist
        assert "<integer>30</integer>" in plist

    def test_caffeinate_present(self) -> None:
        plist = generate_plist(Path("/opt/hedgefund"), Path("/usr/bin/python3"))
        assert "caffeinate" in plist

    def test_daily_run_command(self) -> None:
        plist = generate_plist(Path("/opt/hedgefund"), Path("/usr/bin/python3"))
        assert "daily-run" in plist

    def test_valid_xml_structure(self) -> None:
        plist = generate_plist(Path("/opt/hedgefund"), Path("/usr/bin/python3"))
        assert plist.startswith("<?xml")
        assert "</plist>" in plist
