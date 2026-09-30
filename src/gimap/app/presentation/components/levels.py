"""Colour limits of an image view: quick to drag, exact to type, automatic per frame or fixed.

As in ImageJ (Brightness & Contrast), silx / pyFAI (colormap dialog), napari (auto-contrast *once* or
*continuous*) and Mantid (the *Autoscale* check box that stops the scale reverting when new data arrive):

* **The bar** next to the image is a histogram of the displayed values with two handles at absolute
  positions: drag them (or the band between them) for a quick change; the mouse wheel over the bar zooms
  it for finer moves. No snapping back, no rounding.
* **Levels ▾**: *Auto for every frame* (each new frame gets its own limits by the chosen rule) or not
  (the limits stay: the same scale for every frame); the rule (1–99.7 %, 0.1–99.9 %, 5–95 %, min–max,
  mean ± 3σ); **Min** / **Max** typed in intensity units; **Auto Once**.
* Dragging or typing fixes the limits: they stay through redraws, other frames and re-analysis until
  Auto. Limits are kept in intensity units, so they mean the same in log and linear display, and per
  *context* (detector, q map, cake …) of the view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

AUTO_RULES = (
    ("p1", "1–99.7 %"),
    ("p0.1", "0.1–99.9 %"),
    ("p5", "5–95 %"),
    ("minmax", "Min–max"),
    ("std3", "Mean ± 3σ"),
)
PERCENTILES = {"p1": (1.0, 99.7), "p0.1": (0.1, 99.9), "p5": (5.0, 95.0)}
DEFAULT_RULE = "p1"
HISTOGRAM_BINS = 64
SAMPLE = 400_000
"""Values sampled from a large frame to find its limits."""


def auto_levels(values, rule: str = DEFAULT_RULE) -> Optional[tuple[float, float]]:
    """Colour limits of the displayed values by ``rule`` (``None`` when nothing is finite)."""
    array = np.asarray(values)
    stride = max(1, int(np.ceil(np.sqrt(array.size / SAMPLE)))) if array.ndim == 2 else 1
    sample = array[::stride, ::stride] if array.ndim == 2 else array.ravel()
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return None
    if rule == "minmax":
        low, high = float(finite.min()), float(finite.max())
    elif rule == "std3":
        mean, spread = float(finite.mean()), float(finite.std())
        low, high = max(float(finite.min()), mean - 3 * spread), min(float(finite.max()), mean + 3 * spread)
    else:
        low, high = (float(value) for value in np.percentile(finite, PERCENTILES.get(rule, PERCENTILES[DEFAULT_RULE])))
    if not high > low:
        high = low + (abs(low) * 1e-6 or 1.0)
    return low, high


def to_display(levels, log: bool) -> Optional[tuple[float, float]]:
    """Intensity limits → the displayed values (log₁₀ I when ``log``)."""
    if levels is None:
        return None
    low, high = (float(value) for value in levels)
    if not log:
        return low, high
    if high <= 0:
        return None
    high_log = math.log10(high)
    low_log = math.log10(low) if low > 0 else high_log - 6.0
    return low_log, max(high_log, low_log + 1e-9)


def to_intensity(levels, log: bool) -> tuple[float, float]:
    low, high = (float(value) for value in levels)
    return (10.0 ** low, 10.0 ** high) if log else (low, high)


@dataclass
class LevelState:
    auto: bool = True
    rule: str = DEFAULT_RULE
    fixed: Optional[tuple[float, float]] = None
    """Intensity limits kept for every frame (``auto`` off)."""
    shown: Optional[tuple[float, float]] = None
    """The intensity limits last shown."""


_BAR_CLASS = None


def level_bar(image_item):
    """A slim vertical histogram with absolute min/max handles for ``image_item`` (pyqtgraph)."""
    global _BAR_CLASS
    if _BAR_CLASS is None:
        import pyqtgraph as pg

        class LevelBar(pg.HistogramLUTItem):
            def __init__(self, image):
                self._programmatic = True
                super().__init__(image=image, fillHistogram=True)
                self._programmatic = False
                self.edited = _Edits()
                self.gradient.showTicks(False)
                self.gradient.backgroundRect.hide()  # the hatch meant for transparent colour maps
                self.gradient.mouseClickEvent = lambda event: event.ignore()  # the colour map comes from the combo
                self.vb.setMaximumWidth(56)
                self.vb.setMinimumWidth(32)

            def regionChanged(self):  # noqa: N802 - pyqtgraph API
                super().regionChanged()
                if not self._programmatic:
                    low, high = self.getLevels()
                    self.edited.levels.emit(float(low), float(high))

            def imageChanged(self, autoLevel=False, autoRange=False):  # noqa: N802 - pyqtgraph API
                """A readable histogram (64 bins; counts on a log axis are spiky with fine bins); the handles
                where the image's levels are."""
                image = self.imageItem()
                if image is None:
                    return
                self._programmatic = True
                try:
                    histogram = image.getHistogram(bins=HISTOGRAM_BINS, targetImageSize=400)
                    if histogram[0] is not None:
                        self.plot.setData(*histogram)
                    levels = image.getLevels()
                    if levels is not None and np.size(levels) == 2:
                        self.region.setRegion([float(levels[0]), float(levels[1])])
                finally:
                    self._programmatic = False

            def set_levels(self, levels) -> None:
                self._programmatic = True
                try:
                    self.setLevels(*levels)
                finally:
                    self._programmatic = False

            def levels(self):
                return tuple(float(value) for value in self.getLevels())

            def setColorMap(self, colormap):  # noqa: N802 - Qt-style API
                self.gradient.setColorMap(colormap)
                self.gradient.showTicks(False)  # a new map brings a tick per colour stop: they would hatch the bar

            def set_label(self, text: str) -> None:
                self.axis.setLabel(text)

        _BAR_CLASS = LevelBar
    return _BAR_CLASS(image_item)


class _Edits(QObject):
    levels = pyqtSignal(float, float)
    """The person dragged the handles to these displayed limits."""


class LevelControl(QObject):
    """The limits of one view, per context, and its **Levels ▾** button."""

    changed = pyqtSignal()
    """The limits or the mode changed: apply them to the image."""

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._states: dict[str, LevelState] = {}
        self.context = "image"
        self.unit = "I"
        self.button: Optional[QToolButton] = None

    @property
    def state(self) -> LevelState:
        return self._states.setdefault(self.context, LevelState())

    def state_for(self, context: str) -> LevelState:
        return self._states.setdefault(context, LevelState())

    def set_context(self, context: str) -> None:
        self.context = str(context or "image")
        self._refresh_button()

    # -- the limits ----------------------------------------------------------------------

    def levels_for(self, shown, log: bool) -> Optional[tuple[float, float]]:
        """The displayed limits for ``shown`` (auto by the rule, or the fixed ones), recorded as shown."""
        state = self.state
        levels = None if state.auto or state.fixed is None else to_display(state.fixed, log)
        if levels is None:
            levels = auto_levels(shown, state.rule)
        if levels is not None:
            state.shown = to_intensity(levels, log)
        return levels

    def edited(self, low: float, high: float, log: bool) -> None:
        """The person dragged the handles: these limits stay."""
        state = self.state
        state.fixed = state.shown = to_intensity((low, high), log)
        state.auto = False
        self._refresh_button()

    def fix(self, low: float, high: float) -> None:
        """Typed intensity limits: they stay for every frame."""
        low, high = sorted((float(low), float(high)))
        if not high > low:
            high = low + (abs(low) * 1e-6 or 1.0)
        state = self.state
        state.fixed, state.auto = (low, high), False
        self._refresh_button()
        self.changed.emit()

    def auto_once(self, shown, log: bool) -> None:
        """The rule's limits for the image shown now, kept for the next frames."""
        levels = auto_levels(shown, self.state.rule) if shown is not None else None
        if levels is None:
            return
        self.state.fixed, self.state.auto = to_intensity(levels, log), False
        self._refresh_button()
        self.changed.emit()

    def set_auto(self, auto: bool) -> None:
        """On: each frame its own limits. Off: the limits shown now stay."""
        if not auto:
            self.state.fixed = self.state.shown or self.state.fixed
        self.state.auto = bool(auto)
        self._refresh_button()
        self.changed.emit()

    def set_rule(self, rule: str) -> None:
        if rule in dict(AUTO_RULES):
            self.state.rule = rule
            self.changed.emit()

    def describe(self) -> str:
        from ..i18n import tr

        state = self.state
        rule = tr(dict(AUTO_RULES)[state.rule])
        if state.auto:
            return tr("Auto for every frame ({rule})").format(rule=rule)
        low, high = state.fixed or state.shown or (math.nan, math.nan)
        return tr("Fixed for every frame: {low} … {high}").format(low=f"{low:.4g}", high=f"{high:.4g}")

    # -- the button and its panel ----------------------------------------------------------

    def make_button(self, parent: QWidget, auto_once) -> QToolButton:
        """``auto_once()`` is what a click on the button itself does."""
        from ..i18n import tr

        button = QToolButton(parent)
        button.setObjectName("levelsButton")
        button.setPopupMode(QToolButton.MenuButtonPopup)
        button.clicked.connect(auto_once)
        menu = QMenu(button)
        panel = QWidget(menu)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)
        self.auto_check = QCheckBox(tr("Auto for every frame"), panel)
        self.auto_check.setToolTip(tr(
            "Each new frame gets its own limits by the rule. Off: the limits stay the same for every frame "
            "(to compare frames); dragging the bar or typing limits turns it off"))
        rule_row = QHBoxLayout()
        rule_row.addWidget(QLabel(tr("Rule"), panel))
        self.rule_combo = QComboBox(panel)
        for key, text in AUTO_RULES:
            self.rule_combo.addItem(tr(text), key)
        rule_row.addWidget(self.rule_combo, 1)
        grid = QGridLayout()
        self.min_edit, self.max_edit = QLineEdit(panel), QLineEdit(panel)
        validator = QDoubleValidator(panel)
        validator.setNotation(QDoubleValidator.ScientificNotation)
        for row, (label, edit) in enumerate(((tr("Min"), self.min_edit), (tr("Max"), self.max_edit))):
            edit.setValidator(validator)
            edit.setMinimumWidth(110)
            edit.returnPressed.connect(self._apply_typed)
            grid.addWidget(QLabel(label, panel), row, 0)
            grid.addWidget(edit, row, 1)
        self.unit_label = QLabel("", panel)
        self.unit_label.setProperty("gimapRole", "muted")
        grid.addWidget(self.unit_label, 0, 2, 2, 1, Qt.AlignVCenter)
        buttons = QHBoxLayout()
        once = QPushButton(tr("Auto Once"), panel)
        once.setToolTip(tr("The rule's limits for this frame, kept for the next frames"))
        apply = QPushButton(tr("Apply"), panel)
        apply.setProperty("gimapRole", "primary")
        buttons.addWidget(once)
        buttons.addStretch(1)
        buttons.addWidget(apply)
        hint = QLabel(tr("Drag the handles of the bar for a quick change (the wheel zooms the bar); typed limits "
                         "are exact. Changed limits stay for every frame until Auto."), panel)
        hint.setWordWrap(True)
        hint.setProperty("gimapRole", "muted")
        hint.setMaximumWidth(280)
        for item in (self.auto_check, rule_row, grid, buttons, hint):
            (layout.addWidget if isinstance(item, QWidget) else layout.addLayout)(item)
        action = QWidgetAction(menu)
        action.setDefaultWidget(panel)
        menu.addAction(action)
        menu.aboutToShow.connect(self._fill_panel)
        self.auto_check.toggled.connect(lambda on: self.set_auto(on) if on != self.state.auto else None)
        self.rule_combo.activated.connect(lambda _index: self.set_rule(self.rule_combo.currentData()))
        apply.clicked.connect(lambda: (self._apply_typed(), menu.close()))
        once.clicked.connect(lambda: (auto_once(), menu.close()))
        button.setMenu(menu)
        self.button = button
        self._refresh_button()
        return button

    def _fill_panel(self) -> None:
        state = self.state
        self.auto_check.blockSignals(True)
        self.auto_check.setChecked(state.auto)
        self.auto_check.blockSignals(False)
        self.rule_combo.setCurrentIndex(max(0, self.rule_combo.findData(state.rule)))
        low, high = state.fixed if not state.auto and state.fixed else (state.shown or (None, None))
        self.min_edit.setText("" if low is None else f"{low:.6g}")
        self.max_edit.setText("" if high is None else f"{high:.6g}")
        self.unit_label.setText(self.unit)

    def _apply_typed(self) -> None:
        try:
            low, high = float(self.min_edit.text().replace(",", ".")), float(self.max_edit.text().replace(",", "."))
        except ValueError:
            return
        self.fix(low, high)

    def _refresh_button(self) -> None:
        if self.button is None:
            return
        from ..i18n import tr

        self.button.setText(tr("Levels") if self.state.auto else tr("Levels (fixed)"))
        self.button.setToolTip(self.describe() + "\n" + tr("Click: auto once; ▾ for the options and exact limits"))


__all__ = ["AUTO_RULES", "DEFAULT_RULE", "LevelControl", "LevelState", "auto_levels", "level_bar", "to_display",
           "to_intensity"]
