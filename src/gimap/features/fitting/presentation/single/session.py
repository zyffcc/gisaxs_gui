"""What the single-curve page holds (no Qt): the curve, which of its points are fitted, the model,
the last fit, the solutions of the other methods, and the undo history of the model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.gimap.app.presentation.i18n import tr

from ...application.single_fit import FAMILIES, Curve, FitData, FitModel, FitResult, new_component, prepare_curve

HISTORY = 50


@dataclass(frozen=True)
class Solution:
    """One solution of ``Find the particle shape``, 1D Predict or a wide search, as a model."""

    label: str
    model: FitModel
    chi2: float
    source: str
    deviation: float = 0.0
    """How much the model's curve differs from the solution's own (quick fit, AI)."""


def model_label(model: FitModel) -> str:
    """“Sphere”, “Random cylinder + Sphere”: the families of a model, in the interface language."""
    return " + ".join(tr(FAMILIES[component.family][0]) for component in model.components) or tr("no particle")


def starting_model() -> FitModel:
    return FitModel((new_component("sphere"),))


@dataclass
class FitSession:
    curve: Optional[Curve] = None
    side: str = "mean"
    q_range: Optional[tuple] = None
    """|q| from–to in nm⁻¹; ``None``: every point."""
    model: FitModel = field(default_factory=starting_model)
    result: Optional[FitResult] = None
    solutions: list = field(default_factory=list)
    edited: bool = False
    """Whether the person or a fit has changed the starting model."""
    _undo: list = field(default_factory=list)
    _redo: list = field(default_factory=list)

    # -- the points ---------------------------------------------------------------------

    def data(self) -> Optional[FitData]:
        """The points fitted (the halves chosen, inside the range)."""
        if self.curve is None:
            return None
        return prepare_curve(self.curve, self.side, self.q_range)

    def all_points(self) -> Optional[FitData]:
        if self.curve is None:
            return None
        return prepare_curve(self.curve, self.side, None)

    def set_curve(self, curve: Curve) -> None:
        self.curve = curve
        self.q_range = None
        self.result = None
        if not curve.signed:
            self.side = "mean"

    # -- the model ----------------------------------------------------------------------

    def set_model(self, model: FitModel, *, record: bool = True) -> bool:
        if model == self.model:
            return False
        if record:
            self._undo.append(self.model)
            del self._undo[:-HISTORY]
            self._redo.clear()
        self.model = model
        self.edited = True
        return True

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self.model)
        self.model = self._undo.pop()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self.model)
        self.model = self._redo.pop()
        return True

    def result_is_current(self) -> bool:
        return self.result is not None and self.result.model == self.model


__all__ = ["FitSession", "Solution", "model_label", "starting_model"]
