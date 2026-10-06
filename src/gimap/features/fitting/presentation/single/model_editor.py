"""The model of one curve as cards: one per particle component, one for background and resolution.

Each parameter is a row — name (unit), value, ± error of the last fit, “fit” (free or fixed), and,
when ranges are shown, its min and max. A card's header chooses the family, switches the
interparticle distance (paracrystal S(q)) on or off, and removes the component. Every edit emits
``edited`` with the new model; values typed outside their range widen the range.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Mapping, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.i18n import tr

from ...application.single_fit import FAMILIES, GLOBALS, INFO, LINEAR, Component, FitModel, new_component, parameter_text
from ..layout_primitives import ScientificDoubleSpinBox

VALUE_LIMIT = 1e15


def _spin(parent: QWidget, value: float, name: str) -> ScientificDoubleSpinBox:
    spin = ScientificDoubleSpinBox(parent)
    spin.setObjectName(name)
    spin.setRange(-VALUE_LIMIT, VALUE_LIMIT)
    spin.setDecimals(12)
    spin.setKeyboardTracking(False)
    _set(spin, value)
    return spin


def _set(spin: ScientificDoubleSpinBox, value: float) -> None:
    spin.blockSignals(True)
    spin.setValue(float(value) if math.isfinite(value) else VALUE_LIMIT)
    spin.setSingleStep(max(abs(float(value)) * 0.05, 1e-6) if math.isfinite(value) else 1.0)
    spin.blockSignals(False)


def _label(key: str) -> str:
    """“Peak w (nm⁻¹)”: the name in the interface language, the unit as it is."""
    info = INFO[key]
    return tr(info.label) + (f" ({info.unit})" if info.unit else "")


class _Row:
    """The widgets of one parameter."""

    def __init__(self, editor: "FitModelEditor", grid: QGridLayout, line: int, path: tuple, parent: QWidget):
        """Two grid lines: name, value, ± error, fit; then (when ranges are shown) min – max."""
        self.editor, self.path = editor, path
        parameter = editor.model.get(path)
        key = path[1]
        self.name = QLabel(_label(key), parent)
        self.name.setToolTip(tr(INFO[key].tip))
        self.value = _spin(parent, parameter.value, f"fitValue_{path[0]}_{key}")
        self.error = QLabel("", parent)
        self.error.setObjectName(f"fitError_{path[0]}_{key}")
        self.error.setProperty("gimapRole", "muted")
        self.free = QCheckBox(tr("fit"), parent)
        self.free.setObjectName(f"fitFree_{path[0]}_{key}")
        self.free.setToolTip(tr("Ticked: the fit may change this value; unticked: it stays fixed"))
        self.free.setChecked(parameter.free)
        self.value.setMinimumWidth(96)
        self.error.setMinimumWidth(56)
        self.range = QWidget(parent)
        span = QHBoxLayout(self.range)
        span.setContentsMargins(0, 0, 0, 4)
        span.setSpacing(4)
        self.lower = _spin(self.range, parameter.lower, f"fitLower_{path[0]}_{key}")
        self.upper = _spin(self.range, parameter.upper, f"fitUpper_{path[0]}_{key}")
        between = QLabel("–", self.range)
        between.setProperty("gimapRole", "muted")
        for spin in (self.lower, self.upper):
            spin.setToolTip(tr("Range of this parameter: the fit keeps it inside, “Search the ranges” searches across it"))
        span.addWidget(self.lower, 1)
        span.addWidget(between)
        span.addWidget(self.upper, 1)
        for column, widget in enumerate((self.name, self.value, self.error, self.free)):
            grid.addWidget(widget, 2 * line, column)
        grid.addWidget(self.range, 2 * line + 1, 1, 1, 3)
        self.value.valueChanged.connect(self._value)
        self.free.toggled.connect(lambda on: editor.change(path, free=bool(on)))
        self.lower.valueChanged.connect(lambda value: editor.change(path, lower=float(value)))
        self.upper.valueChanged.connect(lambda value: editor.change(path, upper=float(value)))
        self.show_range(editor.show_ranges)

    def _value(self, value: float) -> None:
        parameter = self.editor.model.get(self.path)
        changes = {"value": float(value)}
        if value < parameter.lower:
            changes["lower"] = float(value) if value <= 0 else float(value) / 2
        if value > parameter.upper:
            changes["upper"] = float(value) * 2 if value > 0 else float(value)
        self.editor.change(self.path, **changes)

    def refresh(self, errors: Mapping, at_bounds) -> None:
        parameter = self.editor.model.get(self.path)
        _set(self.value, parameter.value)
        _set(self.lower, parameter.lower)
        _set(self.upper, parameter.upper)
        self.free.blockSignals(True)
        self.free.setChecked(parameter.free)
        self.free.blockSignals(False)
        error = errors.get(self.path)
        if self.path in at_bounds:
            self.error.setText(tr("at a bound"))
            self.error.setToolTip(tr("The fit stopped at a bound of the range: widen the range or fix this value"))
        elif error is not None:
            shown = parameter_text(self.path[1], parameter.value, error)
            self.error.setText("± —" if not math.isfinite(error) else "± " + shown.split("± ", 1)[-1])
            self.error.setToolTip(tr("1σ error of the last fit") + f": {shown}")
        else:
            self.error.setText("")
            self.error.setToolTip("")

    def show_range(self, on: bool) -> None:
        # Scales are solved exactly (≥ 0) while fitting: they have no range to set.
        self.range.setVisible(on and self.path[1] not in LINEAR)


class FitModelEditor(QWidget):
    edited = pyqtSignal(object)
    """The model after an edit by the person (``FitModel``)."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("fitModelEditor")
        self.model = FitModel()
        self.show_ranges = False
        self._errors: Mapping = {}
        self._at_bounds: tuple = ()
        self._rows: dict[tuple, _Row] = {}
        self._shape: tuple = ()
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self._rebuild()

    # -- from the page ------------------------------------------------------------------

    def set_model(self, model: FitModel, errors: Optional[Mapping] = None, at_bounds=()) -> None:
        self.model = model
        self._errors, self._at_bounds = dict(errors or {}), tuple(at_bounds)
        if self._structure(model) != self._shape:
            self._rebuild()
        else:
            for row in self._rows.values():
                row.refresh(self._errors, self._at_bounds)

    def set_show_ranges(self, on: bool) -> None:
        self.show_ranges = bool(on)
        for row in self._rows.values():
            row.show_range(self.show_ranges)

    def refresh_language(self) -> None:
        """After a switch of the interface language: the cards again (names with their units, families, tips)."""
        self._rebuild()
        for row in self._rows.values():
            row.refresh(self._errors, self._at_bounds)

    def add_component(self, family: str) -> None:
        radius = self.model.components[-1].value("R") if self.model.components else None
        component = new_component(family, radius=radius * 2 if radius else None)
        self._emit(replace(self.model, components=(*self.model.components, component)))

    # -- edits --------------------------------------------------------------------------

    def change(self, path: tuple, **changes) -> None:
        self._errors = {key: value for key, value in self._errors.items() if key != path}
        self._emit(self.model.with_parameter(path, **changes), rebuild=False)

    def _replace_component(self, index: int, component: Optional[Component]) -> None:
        components = list(self.model.components)
        if component is None:
            del components[index]
        else:
            components[index] = component
        self._errors = {}
        self._emit(replace(self.model, components=tuple(components)))

    def _family(self, index: int, family: str) -> None:
        old = self.model.components[index]
        if family == old.family:
            return
        fresh = new_component(family, structure=old.structure)
        params = {key: old.params.get(key, value) for key, value in fresh.params.items()}
        self._replace_component(index, replace(fresh, params=params))

    def _emit(self, model: FitModel, *, rebuild: bool = True) -> None:
        self.model = model
        if rebuild and self._structure(model) != self._shape:
            self._rebuild()
        else:
            for row in self._rows.values():
                row.refresh(self._errors, self._at_bounds)
        self.edited.emit(model)

    # -- cards --------------------------------------------------------------------------

    @staticmethod
    def _structure(model: FitModel) -> tuple:
        return tuple((component.family, component.structure) for component in model.components)

    def _rebuild(self) -> None:
        while self._layout.count():
            widget = self._layout.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._rows = {}
        self._shape = self._structure(self.model)
        for index, component in enumerate(self.model.components):
            self._layout.addWidget(self._component_card(index, component))
        if not self.model.components:
            empty = QLabel(tr("No particle yet: add one above, or let Fit ▸ Find the particle shape choose."), self)
            empty.setWordWrap(True)
            empty.setProperty("gimapRole", "muted")
            self._layout.addWidget(empty)
        self._layout.addWidget(self._globals_card())
        self.set_show_ranges(self.show_ranges)

    def _card(self, name: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame(self)
        card.setObjectName(name)
        card.setProperty("gimapInfoCard", True)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        return card, layout

    @staticmethod
    def _grid(card: QFrame) -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)
        grid.setColumnStretch(1, 1)
        return grid

    def _component_card(self, index: int, component: Component) -> QFrame:
        card, layout = self._card(f"fitComponent_{index}")
        header = QHBoxLayout()
        title = QLabel(f"{index + 1}", card)
        title.setProperty("gimapRole", "strong")
        family = QComboBox(card)
        family.setObjectName(f"fitFamily_{index}")
        for key, (name, _keys) in FAMILIES.items():
            family.addItem(tr(name), key)
        family.setCurrentIndex(family.findData(component.family))
        family.currentIndexChanged.connect(lambda _i, box=family, at=index: self._family(at, box.currentData()))
        structure = QCheckBox(tr("Distance D"), card)
        structure.setObjectName(f"fitStructure_{index}")
        structure.setToolTip(tr("Interference between neighbours: a paracrystal of distance D and disorder σD/D"))
        structure.setChecked(component.structure)
        structure.toggled.connect(lambda on, at=index: self._replace_component(
            at, replace(self.model.components[at], structure=bool(on))))
        remove = QToolButton(card)
        remove.setObjectName(f"fitRemove_{index}")
        remove.setText("×")
        remove.setAutoRaise(True)
        remove.setToolTip(tr("Remove this component"))
        remove.clicked.connect(lambda _checked=False, at=index: self._replace_component(at, None))
        header.addWidget(title)
        header.addWidget(family, 1)
        header.addWidget(structure)
        header.addWidget(remove)
        layout.addLayout(header)
        grid = self._grid(card)
        for line, key in enumerate(component.keys()):
            self._rows[(index, key)] = _Row(self, grid, line, (index, key), card)
        layout.addLayout(grid)
        return card

    def _globals_card(self) -> QFrame:
        card, layout = self._card("fitGlobals")
        title = QLabel(tr("Background and resolution peak"), card)
        title.setProperty("gimapRole", "strong")
        title.setToolTip(tr("I(q) = BG + k·[Σ particles + A/(1 + (|q|/w)^ν)]: the peak is the direct and "
                            "reflected beam's tail and the resolution near q = 0"))
        layout.addWidget(title)
        grid = self._grid(card)
        for line, key in enumerate(GLOBALS):
            self._rows[("globals", key)] = _Row(self, grid, line, ("globals", key), card)
        layout.addLayout(grid)
        return card

    def row(self, path: tuple) -> Optional[_Row]:
        return self._rows.get(path)


__all__ = ["FitModelEditor"]
