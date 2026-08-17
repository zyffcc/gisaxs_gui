"""Trainset simulation application port。"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class SimulationPort(Protocol):
    def is_available(self) -> bool: ...

    def simulate(
        self,
        config: dict[str, Any],
        sampled: dict[str, Any],
    ) -> np.ndarray: ...
