"""What the start-up is doing, for the splash screen: one listener, plain English steps.

The composition reports its steps (``report("Building Analyze…")``); the launcher listens while the splash is
shown and translates them there. Without a listener (tests, a window built by a script) a report does nothing.
Kept free of Qt and of the program's packages, so the launcher can import it before the heavy imports.
"""

from __future__ import annotations

from typing import Callable, Optional

_listener: Optional[Callable[[str], None]] = None


def listen(listener: Optional[Callable[[str], None]]) -> None:
    """``listener(text)`` for each step from now on; ``None`` stops listening."""
    global _listener
    _listener = listener


def report(text: str) -> None:
    """Tell the listener (the splash) what starts now; an error in the listener never stops the start-up."""
    listener = _listener
    if listener is None:
        return
    try:
        listener(text)
    except Exception:  # noqa: BLE001 - the splash is only feedback
        pass


__all__ = ["listen", "report"]
