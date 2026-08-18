"""Trainset detector-design data port."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np


class TrainsetDesignPort(Protocol):
    def load_reference(self, path: Path) -> np.ndarray: ...

    def crop(self, image: np.ndarray, roi: dict[str, int]) -> np.ndarray: ...

    def threshold_summary(
        self,
        image: np.ndarray,
        roi: dict[str, int],
        threshold: dict[str, Any],
        *,
        automatic_upper: bool,
        lower: float,
        upper: float,
    ) -> dict[str, Any]: ...

    def overlay(
        self,
        image: np.ndarray,
        roi: dict[str, int],
        config: dict[str, Any],
        random_mask: np.ndarray | None,
    ) -> dict[str, Any]: ...

    def random_mask(
        self, shape: tuple[int, int], config: dict[str, Any]
    ) -> np.ndarray: ...
