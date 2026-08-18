"""Port for the legacy detector-to-q implementation."""

from __future__ import annotations

from typing import Protocol


class QSpacePort(Protocol):
    def create_detector(self, **geometry): ...

    def axis_labels_and_extent(self, detector): ...
