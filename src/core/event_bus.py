# src/core/event_bus.py
# Simple publish-subscribe event bus for decoupled component communication.
# Components emit events (like "trade_executed", "protection_triggered")
# and other components subscribe to react without tight coupling.
#
# This is how the engine communicates with notifications, logging, etc.
# without those systems being imported directly into the trading loop.

import logging
from typing import Callable

logger = logging.getLogger(__name__)


class EventBus:
    """Lightweight pub-sub event system.

    Usage:
        bus = EventBus()
        bus.on("trade_executed", my_handler)
        await bus.emit("trade_executed", trade_data)
    """

    def __init__(self):
        # event_name -> list of async callback functions
        self._handlers: dict[str, list[Callable]] = {}

    def on(self, event: str, handler: Callable):
        """Subscribe a handler to an event.

        handler: async function that takes **kwargs
        """
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)
        logger.debug("Handler registered for event '%s'", event)

    async def emit(self, event: str, **kwargs):
        """Emit an event, calling all registered handlers.

        Handlers are called in registration order.
        Errors in one handler don't block others.
        """
        handlers = self._handlers.get(event, [])
        for handler in handlers:
            try:
                await handler(**kwargs)
            except Exception as e:
                logger.error("Handler error for event '%s': %s", event, e)
