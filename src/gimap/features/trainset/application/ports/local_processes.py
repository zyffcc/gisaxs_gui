"""Trainset local process lifecycle port."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from ..models import TrainsetLocalProcessRequest


class TrainsetLocalProcessPort(Protocol):
    def is_running(self) -> bool: ...

    def start(
        self,
        request: TrainsetLocalProcessRequest,
        *,
        on_started: Callable[[], None],
        on_progress: Callable[[int, str], None],
        on_log: Callable[[str], None],
        on_finished: Callable[[int], None],
        on_error: Callable[[str], None],
    ) -> None: ...

    def set_paused(self, paused: bool) -> bool: ...

    def cancel(self) -> bool: ...
