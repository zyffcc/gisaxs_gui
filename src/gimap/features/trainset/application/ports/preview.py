"""Trainset preview application port."""

from __future__ import annotations

from typing import Protocol

from ..models import TrainsetPreviewRequest, TrainsetWhatIfRequest


class TrainsetPreviewPort(Protocol):
    def generate_preview(self, request: TrainsetPreviewRequest, *, on_progress=None) -> dict: ...

    def simulate_what_if(self, request: TrainsetWhatIfRequest) -> dict: ...
