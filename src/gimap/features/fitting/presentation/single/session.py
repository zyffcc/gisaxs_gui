"""What the single-curve page holds (no Qt): the curve, which of its points are fitted, the model,
the last fit, the solutions of the other methods, and the undo history — of the model, the fitting
range and the left-out points together (``Snapshot``)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from src.gimap.app.presentation.i18n import tr

from ...application.single_fit import FAMILIES, INFO, Curve, FitData, FitModel, FitResult, new_component, prepare_curve

HISTORY = 50
COALESCE_SECONDS = 1.0
"""Range changes this close together (a drag of the band, typing, the arrows of a spin box) are one step of Undo."""


@dataclass(frozen=True)
class Snapshot:
    """One step of Undo: the model, the fitting range and the left-out points — the last two of ``curve``
    (going back past another curve brings back only the model)."""

    model: FitModel
    q_range: Optional[tuple]
    excluded: frozenset
    curve: object = field(default=None, compare=False)


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


def path_name(model: FitModel, path: tuple, *, unit: bool = False, translate: bool = True, full: bool = False) -> str:
    """“R” for a model of one particle, “1·Sphere R” for one of several (``full``: always so), or
    “background” for a global value, in the interface language (and its unit); ``translate=False`` for
    the axes of a figure. Shown only: the saved tables and records name every value in full."""
    owner, key = path
    say = tr if translate else str
    label = say(INFO[key].label) + (f" ({INFO[key].unit})" if unit and INFO[key].unit else "")
    if owner == "globals" or owner >= len(model.components) or (len(model.components) == 1 and not full):
        return label
    return f"{owner + 1}·{say(FAMILIES[model.components[owner].family][0])} {label}"


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
    result_selection: Optional[tuple] = None
    """The points the last fit was made on: ``selection()`` when it started."""
    solutions: list = field(default_factory=list)
    solutions_selection: Optional[tuple] = None
    """The points the solutions' χ² is of: ``selection()`` when their search started; ``None`` for a
    solution of Analyze (no search on this page; its χ² is Analyze's)."""
    edited: bool = False
    """Whether the person or a fit has changed the starting model."""
    excluded: set = field(default_factory=set)
    """Points left out of the fit (``point_key`` of their |q|)."""
    _undo: list = field(default_factory=list)
    _redo: list = field(default_factory=list)
    _last_kind: str = ""
    """What the last step recorded was (``record``): range changes in a burst are one step."""
    _last_time: float = float("-inf")

    # -- the points ---------------------------------------------------------------------

    def data(self) -> Optional[FitData]:
        """The points fitted (the halves chosen, inside the range)."""
        if self.curve is None:
            return None
        return prepare_curve(self.curve, self.side, self.q_range, self.excluded)

    def all_points(self) -> Optional[FitData]:
        """Every point of the halves chosen (also those outside the range or left out)."""
        if self.curve is None:
            return None
        return prepare_curve(self.curve, self.side, None)

    def selection(self) -> tuple:
        """Which points are fitted: the halves, the range and the left-out points."""
        return self.side, self.q_range, frozenset(self.excluded)

    def set_curve(self, curve: Curve) -> None:
        self.curve = curve
        self.q_range = None
        self.result = None
        self.result_selection = None
        self.solutions = []  # the solutions of the curve before
        self.solutions_selection = None
        self.excluded = set()
        self._last_kind = ""  # a range change on this curve is a step of its own
        if not curve.signed:
            self.side = "mean"

    # -- the model ----------------------------------------------------------------------

    def set_model(self, model: FitModel, *, record: bool = True) -> bool:
        if model == self.model:
            return False
        if record:
            self.record()
        self.model = model
        self.edited = True
        return True

    def set_range(self, q_range, *, record: bool = True, coalesce: bool = False) -> bool:
        """The fitting range (``None``: every point); ``coalesce``: a drag or typing, one step with the
        range changes just before it. ``False`` when it is the range already."""
        q_range = None if q_range is None else tuple(sorted(float(value) for value in q_range))
        if q_range == self.q_range:
            return False
        if record:
            self.record("range" if coalesce else "")
        self.q_range = q_range
        return True

    def set_excluded(self, excluded) -> bool:
        """The left-out points (one step of Undo); ``False`` when they are these already."""
        excluded = set(excluded)
        if excluded == self.excluded:
            return False
        self.record()
        self.excluded = excluded
        return True

    # -- undo -----------------------------------------------------------------------------

    def snapshot(self) -> Snapshot:
        return Snapshot(self.model, self.q_range, frozenset(self.excluded), self.curve)

    def record(self, kind: str = "", *, now: Optional[float] = None) -> None:
        """Remember the model, range and left-out points before a change. A change of the same ``kind``
        (``"range"``) within ``COALESCE_SECONDS`` of the last one continues that step."""
        now = time.monotonic() if now is None else now
        continuing = bool(kind) and kind == self._last_kind and now - self._last_time < COALESCE_SECONDS and self._undo
        self._last_kind, self._last_time = kind, now
        if continuing:
            self._redo.clear()
            return
        self._undo.append(self.snapshot())
        del self._undo[:-HISTORY]
        self._redo.clear()

    def _is_current(self, snapshot: Snapshot) -> bool:
        """Bringing ``snapshot`` back would change nothing."""
        if snapshot.model != self.model:
            return False
        return snapshot.curve is not self.curve or (
            snapshot.q_range == self.q_range and snapshot.excluded == frozenset(self.excluded))

    def _bring_back(self, snapshot: Snapshot) -> None:
        self.model = snapshot.model
        if snapshot.curve is self.curve:  # the range and points of another curve mean nothing for this one
            self.q_range = snapshot.q_range
            self.excluded = set(snapshot.excluded)
        self._last_kind = ""  # the next change is a step of its own

    def _step(self, source: list, target: list) -> bool:
        while source and self._is_current(source[-1]):
            source.pop()  # a burst that came back where it started: nothing to undo there
        if not source:
            return False
        target.append(self.snapshot())
        self._bring_back(source.pop())
        return True

    def clear_history(self) -> None:
        """No step to undo or redo (a project opened: a new start, not a change of the work before)."""
        self._undo.clear()
        self._redo.clear()
        self._last_kind = ""

    def can_undo(self) -> bool:
        return any(not self._is_current(snapshot) for snapshot in self._undo)

    def can_redo(self) -> bool:
        return any(not self._is_current(snapshot) for snapshot in self._redo)

    def undo(self) -> bool:
        """The model, range and left-out points before the last change (a fit, an edit, points left out …)."""
        return self._step(self._undo, self._redo)

    def redo(self) -> bool:
        return self._step(self._redo, self._undo)

    def result_model_is_current(self) -> bool:
        return self.result is not None and self.result.model == self.model

    def result_selection_is_current(self) -> bool:
        return self.result is not None and self.result_selection == self.selection()

    def result_is_current(self) -> bool:
        """The last fit is of this model on these points (its quality and errors hold)."""
        return self.result_model_is_current() and self.result_selection_is_current()

    def solutions_searched(self) -> bool:
        """The solutions come from a search on this page (not from Analyze)."""
        return bool(self.solutions) and self.solutions_selection is not None

    def solutions_are_current(self) -> bool:
        """The solutions were searched on these points (their χ² holds)."""
        return self.solutions_searched() and self.solutions_selection == self.selection()


__all__ = ["COALESCE_SECONDS", "FitSession", "Snapshot", "Solution", "model_label", "path_name", "starting_model"]
