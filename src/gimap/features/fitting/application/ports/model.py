"""Manual fitting scientific model port。"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np


class FittingModelPort(Protocol):
    def parameter_names(self, shapes: tuple[str, ...]) -> tuple[str, ...]: ...

    def evaluate(
        self,
        shapes: tuple[str, ...],
        q_model: np.ndarray,
        parameters: tuple[float, ...],
    ) -> np.ndarray: ...

    def components(
        self,
        shapes: tuple[str, ...],
        q_model: np.ndarray,
        parameters: tuple[float, ...],
    ) -> dict: ...

    def build_function(self, shapes: tuple[str, ...]) -> Callable: ...
