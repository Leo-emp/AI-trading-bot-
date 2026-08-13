# tests/test_macro_calendar.py
# Tests for the Macro Event Calendar — the bot's shield against volatility.

import json
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.intelligence.macro_calendar import MacroCalendar, MacroEvent


class TestMacroEvent:
    """Test individual macro event logic."""

    def test_event_active_during_pause_window(self):
        """Event should be active within its pause window."""
        event = MacroEvent(
            name="FOMC", description="Fed rate decision",
            event_time=datetime(2026, 8, 15, 14, 0, tzinfo=timezone.utc),
            pause_before_hours=2, pause_after_hours=1,
        )
        # 1 hour before event → should be active (within 2hr before window)
        check_time = datetime(2026, 8, 15, 13, 0, tzinfo=timezone.utc)
        assert event.is_active(check_time) is True

    def test_event_inactive_outside_window(self):
        """Event should not be active outside its pause window."""
        event = MacroEvent(
            name="FOMC", description="Fed rate decision",
            event_time=datetime(2026, 8, 15, 14, 0, tzinfo=timezone.utc),
            pause_before_hours=2, pause_after_hours=1,
        )
        # 5 hours before event → outside 2hr window
        check_time = datetime(2026, 8, 15, 9, 0, tzinfo=timezone.utc)
        assert event.is_active(check_time) is False

    def test_event_active_after_event(self):
        """Event should be active during the after-pause window."""
        event = MacroEvent(
            name="CPI", description="CPI release",
            event_time=datetime(2026, 8, 12, 14, 0, tzinfo=timezone.utc),
            pause_before_hours=1, pause_after_hours=1,
        )
        # 30 minutes after event → within 1hr after window
        check_time = datetime(2026, 8, 12, 14, 30, tzinfo=timezone.utc)
        assert event.is_active(check_time) is True

    def test_time_until_clear(self):
        """Should return correct remaining pause duration."""
        event = MacroEvent(
            name="NFP", description="Jobs report",
            event_time=datetime(2026, 8, 7, 14, 0, tzinfo=timezone.utc),
            pause_before_hours=1, pause_after_hours=1,
        )
        # At event time, 1 hour until clear
        now = datetime(2026, 8, 7, 14, 0, tzinfo=timezone.utc)
        delta = event.time_until_clear(now)
        assert abs(delta.total_seconds() - 3600) < 1  # 1 hour


class TestMacroCalendar:
    """Test the full calendar system."""

    @pytest.fixture
    def temp_calendar(self, tmp_path):
        """Create a calendar with test events."""
        config = {
            "recurring_events": [
                {
                    "name": "Test FOMC",
                    "description": "Test Fed decision",
                    "pause_hours_before": 2,
                    "pause_hours_after": 1,
                    "dates_2026": ["2026-08-15"],
                }
            ],
            "one_time_events": [],
        }
        config_file = tmp_path / "test_events.json"
        config_file.write_text(json.dumps(config))
        return MacroCalendar(config_path=str(config_file))

    def test_check_returns_pause_during_event(self, temp_calendar):
        """Should pause during an active event."""
        # 1 hour before FOMC at 14:00 UTC
        now = datetime(2026, 8, 15, 13, 0, tzinfo=timezone.utc)
        result = temp_calendar.check(now=now)
        assert result["should_pause"] is True
        assert "FOMC" in result["reason"]
        assert result["minutes_until_clear"] > 0

    def test_check_returns_clear_no_event(self, temp_calendar):
        """Should return clear when no events active."""
        # Way before any event
        now = datetime(2026, 8, 10, 10, 0, tzinfo=timezone.utc)
        result = temp_calendar.check(now=now)
        assert result["should_pause"] is False
        assert result["event_name"] is None

    def test_missing_config_file_does_not_crash(self):
        """Calendar should handle missing config gracefully."""
        calendar = MacroCalendar(config_path="/nonexistent/path.json")
        result = calendar.check()
        assert result["should_pause"] is False

    def test_get_upcoming_events(self, temp_calendar):
        """Should list events in the next N days."""
        now = datetime(2026, 8, 14, 10, 0, tzinfo=timezone.utc)
        upcoming = temp_calendar.get_upcoming(days=7, now=now)
        assert len(upcoming) == 1
        assert "FOMC" in upcoming[0]["name"]

    def test_reload_clears_and_reloads(self, temp_calendar):
        """Reload should refresh events from file."""
        temp_calendar.reload()
        now = datetime(2026, 8, 15, 13, 0, tzinfo=timezone.utc)
        result = temp_calendar.check(now=now)
        assert result["should_pause"] is True
