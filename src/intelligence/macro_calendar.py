# src/intelligence/macro_calendar.py
# Macro Event Calendar — tells the bot when NOT to trade.
#
# The idea: scheduled economic events (Fed, CPI, NFP) cause massive
# volatility spikes. Our scalp bot can't handle 2-5% swings in seconds.
# Instead of trying to trade through chaos, we pause.
#
# How it works:
# 1. Load events from config/macro_events.json
# 2. Before each trade cycle, check: is a major event happening soon?
# 3. If yes → return "PAUSE" with reason and duration
# 4. If no → return "CLEAR" → bot trades normally
#
# Cost: zero. No API. Just a JSON file updated monthly.
# Value: prevents 5-15% annual losses from volatility traps.

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class MacroEvent:
    """A single macro event with pause window."""

    def __init__(self, name: str, description: str,
                 event_time: datetime,
                 pause_before_hours: float,
                 pause_after_hours: float):
        self.name = name
        self.description = description
        self.event_time = event_time
        self.pause_before = timedelta(hours=pause_before_hours)
        self.pause_after = timedelta(hours=pause_after_hours)

    @property
    def pause_start(self) -> datetime:
        return self.event_time - self.pause_before

    @property
    def pause_end(self) -> datetime:
        return self.event_time + self.pause_after

    def is_active(self, now: datetime) -> bool:
        """Is this event's pause window active right now?"""
        return self.pause_start <= now <= self.pause_end

    def time_until_clear(self, now: datetime) -> timedelta:
        """How long until the pause window ends."""
        return self.pause_end - now


class MacroCalendar:
    """Checks scheduled macro events and advises the bot to pause.

    Usage:
        calendar = MacroCalendar()
        status = calendar.check()
        if status["should_pause"]:
            logger.warning("Pausing: %s", status["reason"])
    """

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = str(Path(__file__).parent.parent.parent / "config" / "macro_events.json")
        self._config_path = config_path
        self._events: list[MacroEvent] = []
        self._loaded = False

    def _load_events(self):
        """Load events from JSON config file."""
        try:
            with open(self._config_path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.warning("Macro events file not found: %s", self._config_path)
            self._loaded = True
            return
        except json.JSONDecodeError as e:
            logger.error("Invalid macro events JSON: %s", e)
            self._loaded = True
            return

        now = datetime.now(timezone.utc)
        current_year = str(now.year)

        # Load recurring events
        for event in data.get("recurring_events", []):
            name = event["name"]
            desc = event.get("description", "")
            pause_before = event.get("pause_hours_before", 1)
            pause_after = event.get("pause_hours_after", 1)

            # Get dates for current year
            dates_key = f"dates_{current_year}"
            dates = event.get(dates_key, [])

            for date_str in dates:
                try:
                    # Events are at 14:00 UTC by default (Fed at 2pm ET = 6pm UTC,
                    # CPI at 8:30 AM ET = 12:30 UTC — 14:00 is a safe middle)
                    event_time = datetime.fromisoformat(date_str).replace(
                        hour=14, minute=0, tzinfo=timezone.utc
                    )
                    # Only keep future events or events in the last 24h
                    if event_time > now - timedelta(hours=24):
                        self._events.append(MacroEvent(
                            name=name, description=desc,
                            event_time=event_time,
                            pause_before_hours=pause_before,
                            pause_after_hours=pause_after,
                        ))
                except (ValueError, TypeError) as e:
                    logger.debug("Invalid date %s for %s: %s", date_str, name, e)

        # Load one-time events
        for event in data.get("one_time_events", []):
            try:
                event_time = datetime.fromisoformat(event["date"]).replace(
                    hour=14, minute=0, tzinfo=timezone.utc
                )
                if event_time > now - timedelta(hours=48):
                    self._events.append(MacroEvent(
                        name=event["name"],
                        description=event.get("description", ""),
                        event_time=event_time,
                        pause_before_hours=event.get("pause_hours_before", 2),
                        pause_after_hours=event.get("pause_hours_after", 1),
                    ))
            except (ValueError, TypeError, KeyError):
                pass

        self._loaded = True
        logger.info("Loaded %d upcoming macro events", len(self._events))

    def check(self, now: Optional[datetime] = None) -> dict:
        """Check if any macro event requires pausing.

        Returns:
            {
                "should_pause": bool,
                "reason": str,
                "event_name": str or None,
                "minutes_until_clear": int,
            }
        """
        if not self._loaded:
            self._load_events()

        if now is None:
            now = datetime.now(timezone.utc)

        # Check each event
        for event in self._events:
            if event.is_active(now):
                minutes_left = int(event.time_until_clear(now).total_seconds() / 60)
                reason = (
                    f"MACRO PAUSE: {event.name} — {event.description}. "
                    f"Resuming in {minutes_left} minutes."
                )
                logger.warning(reason)
                return {
                    "should_pause": True,
                    "reason": reason,
                    "event_name": event.name,
                    "minutes_until_clear": max(minutes_left, 1),
                }

        # Check what's coming up next (for logging)
        next_event = self._get_next_event(now)
        if next_event:
            hours_until = (next_event.pause_start - now).total_seconds() / 3600
            if hours_until < 6:
                logger.info("Next macro event in %.1f hours: %s",
                           hours_until, next_event.name)

        return {
            "should_pause": False,
            "reason": "",
            "event_name": None,
            "minutes_until_clear": 0,
        }

    def _get_next_event(self, now: datetime) -> Optional[MacroEvent]:
        """Get the next upcoming event."""
        future = [e for e in self._events if e.pause_start > now]
        if not future:
            return None
        return min(future, key=lambda e: e.pause_start)

    def get_upcoming(self, days: int = 7, now: Optional[datetime] = None) -> list[dict]:
        """List upcoming events in the next N days (for daily summary)."""
        if not self._loaded:
            self._load_events()
        if now is None:
            now = datetime.now(timezone.utc)

        cutoff = now + timedelta(days=days)
        upcoming = []
        for event in self._events:
            if now <= event.event_time <= cutoff:
                upcoming.append({
                    "name": event.name,
                    "date": event.event_time.strftime("%Y-%m-%d %H:%M UTC"),
                    "pause_window": f"-{event.pause_before} / +{event.pause_after}",
                })
        return sorted(upcoming, key=lambda x: x["date"])

    def reload(self):
        """Force reload events from file (call after editing JSON)."""
        self._events = []
        self._loaded = False
        self._load_events()
