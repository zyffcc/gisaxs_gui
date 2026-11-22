"""One parameter of the Model step on one line: name (unit), value, ± error of the last fit, min, max (when ranges are
shown), fit (free or fixed) and the link button (a link to another parameter or an expression of others).

A linked parameter shows its value greyed (it follows the others), its expression in the tooltip and a highlighted
link button; its “fit” box is off. Values typed outside their range widen the range.
"""

from __future__ import annotations

import math
from typing import Mapping

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QGridLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QToolButton,
    QWidget,
)

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.theme import set_role
from src.gimap.shared.numbers import format_number

from ...application.fit_uncertainty import parameter_name
from ...application.single_fit import LINEAR, parameter_info, parameter_text
from ..layout_primitives import ScientificDoubleSpinBox

VALUE_LIMIT = 1e15
COLUMNS = ("", "value", "±", "min", "max", "", "fit", "")
"""The header of a card's grid (``header``): name, value, error, min, max, ∞, fit, link."""
RANGE_WIDTH = 52
VALUE_WIDTH = 64
BUTTON_WIDTH = 22
LINK_TIP = "Link this value to another parameter or to an expression of others (e.g. 2*R_1); it then follows them"
EXPRESSION_HELP = ("{name} = an expression of other parameters, e.g. 2*R_1, sigma_R_1, R_1 + t_shell_1.\n"
                   "Names: {names}. Functions: sqrt, exp, log, sin, cos, abs, min, max; pi.")
LINKED_TIP = "Linked: = {expression} (follows the other values; not fitted)"


class Elided(QLabel):
    """A label that may be narrower than its text: drawn with “…” (``text()`` and the tooltip keep it whole)."""

    def __init__(self, text: str, parent: QWidget, minimum: int):
        super().__init__(text, parent)
        self._minimum = minimum

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        hint = super().minimumSizeHint()
        return QSize(min(hint.width(), self._minimum), hint.height())

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        shown = self.fontMetrics().elidedText(self.text(), Qt.ElideRight, self.contentsRect().width())
        self.style().drawItemText(painter, self.contentsRect(), int(self.alignment()), self.palette(), self.isEnabled(),
                                  shown, self.foregroundRole())


def spin(parent: QWidget, value: float, name: str, width: int, *, fixed: bool = False) -> ScientificDoubleSpinBox:
    """A value box of one line: no arrows (the wheel and the keys step it), as narrow as ``width`` (its own size hint
    would be that of ±10¹⁵ with 12 decimals)."""
    box = ScientificDoubleSpinBox(parent)
    box.setObjectName(name)
    box.setRange(-VALUE_LIMIT, VALUE_LIMIT)
    box.setDecimals(12)
    box.setKeyboardTracking(False)
    box.setButtonSymbols(QAbstractSpinBox.NoButtons)
    if fixed:
        box.setFixedWidth(width)
    else:
        box.setMinimumWidth(width)
    set_value(box, value)
    return box


def set_value(box: ScientificDoubleSpinBox, value: float) -> None:
    box.blockSignals(True)
    box.setValue(float(value) if math.isfinite(value) else VALUE_LIMIT)
    box.setSingleStep(max(abs(float(value)) * 0.05, 1e-6) if math.isfinite(value) else 1.0)
    box.blockSignals(False)


def label_text(key: str, component=None) -> str:
    """“Tail w (nm⁻¹)”: the name in the interface language, the unit as it is."""
    info = parameter_info(component, key)
    return tr(info.label) + (f" ({info.unit})" if info.unit else "")


def header(grid: QGridLayout, parent: QWidget) -> None:
    """The column titles of a card's grid (row 0)."""
    for column, text in enumerate(COLUMNS):
        if text:
            title = QLabel(tr(text), parent)
            title.setProperty("gimapRole", "muted")
            grid.addWidget(title, 0, column, Qt.AlignLeft | Qt.AlignBottom)


class ParameterRow:
    """The widgets of one parameter (grid row ``line`` + 1, below the header)."""

    def __init__(self, editor, grid: QGridLayout, line: int, path: tuple, parent: QWidget):
        self.editor, self.path = editor, path
        parameter = editor.model.get(path)
        key = path[1]
        component = None if path[0] == "globals" else editor.model.components[path[0]]
        info = parameter_info(component, key)
        self.name = Elided(label_text(key, component), parent, 44)
        self.name.setToolTip(tr(info.tip) + "\n" + trf("Name in links: {name}", name=parameter_name(path)))
        self.value = spin(parent, parameter.value, f"fitValue_{path[0]}_{key}", VALUE_WIDTH, fixed=True)
        self.error = Elided("", parent, 40)
        self.error.setObjectName(f"fitError_{path[0]}_{key}")
        self.error.setProperty("gimapRole", "muted")
        self.free = QCheckBox(parent)
        self.free.setObjectName(f"fitFree_{path[0]}_{key}")
        self.free.setToolTip(tr("Ticked: the fit may change this value; unticked: it stays fixed"))
        self.free.setChecked(parameter.free)
        self.lower = spin(parent, parameter.lower, f"fitLower_{path[0]}_{key}", RANGE_WIDTH, fixed=True)
        self.upper = spin(parent, parameter.upper, f"fitUpper_{path[0]}_{key}", RANGE_WIDTH, fixed=True)
        self.unbounded = QToolButton(parent)  # a checkable “∞”: no upper limit (scales only)
        self.unbounded.setObjectName(f"fitUnbounded_{path[0]}_{key}")
        self.unbounded.setText("∞")
        self.unbounded.setFixedWidth(BUTTON_WIDTH)
        self.unbounded.setCheckable(True)
        self.unbounded.setAutoRaise(True)
        self.unbounded.setToolTip(tr("No upper limit"))
        self.unbounded.setChecked(math.isinf(parameter.upper))
        self.upper.setEnabled(not self.unbounded.isChecked())
        for box in (self.lower, self.upper):
            box.setToolTip(tr("Range of this parameter: the fit keeps it inside, “Search the ranges” searches across it"))
        self.link = QToolButton(parent)
        self.link.setObjectName(f"fitLink_{path[0]}_{key}")
        self.link.setText("=")
        self.link.setAutoRaise(True)
        self.link.setFixedWidth(BUTTON_WIDTH)
        self.link.setToolTip(tr(LINK_TIP))
        self.link.clicked.connect(self._show_menu)
        row = line + 1
        for column, widget in enumerate((self.name, self.value, self.error, self.lower, self.upper, self.unbounded,
                                         self.free, self.link)):
            grid.addWidget(widget, row, column)
        self.value.valueChanged.connect(self._value)
        self.free.toggled.connect(lambda on: editor.change(path, free=bool(on)))
        self.lower.valueChanged.connect(lambda value: editor.change(path, lower=float(value)))
        self.upper.valueChanged.connect(lambda value: editor.change(path, upper=float(value)))
        self.unbounded.toggled.connect(lambda on: editor.change(path, upper=math.inf if on else self.upper.value()))
        self.show_range(editor.show_ranges)
        self._linked_state(parameter)

    def _value(self, value: float) -> None:
        parameter = self.editor.model.get(self.path)
        changes = {"value": float(value)}
        if value < parameter.lower:
            changes["lower"] = float(value) if value <= 0 else float(value) / 2
        if value > parameter.upper:
            changes["upper"] = float(value) * 2 if value > 0 else float(value)
        self.editor.change(self.path, **changes)

    # -- the link ---------------------------------------------------------------------------------

    def _show_menu(self) -> None:
        menu = QMenu(self.link)
        self._fill_menu(menu)
        menu.exec_(self.link.mapToGlobal(self.link.rect().bottomLeft()))

    def _fill_menu(self, menu: QMenu) -> None:
        model, key = self.editor.model, self.path[1]
        same = [path for path, _p in model.parameters() if path[1] == key and path != self.path]
        if same:
            share = menu.addMenu(tr("Same as"))
            for path in same:
                share.addAction(f"{parameter_name(path)} · {self.editor.path_title(path)}",
                                lambda other=path: self.editor.link(self.path, parameter_name(other)))
        menu.addAction(tr("Expression…"), self.ask_expression)
        if model.get(self.path).tie:
            menu.addSeparator()
            menu.addAction(tr("Unlink"), lambda: self.editor.link(self.path, None))

    def ask_expression(self) -> None:
        model = self.editor.model
        names = ", ".join(parameter_name(path) for path, _p in model.parameters() if path != self.path)
        current = model.get(self.path).tie or ""
        text, ok = QInputDialog.getText(self.link, tr("Link"), trf(EXPRESSION_HELP, name=parameter_name(self.path),
                                                                   names=names), text=current)
        if ok and text.strip():
            self.editor.link(self.path, text.strip())

    def _linked_state(self, parameter) -> None:
        linked = bool(parameter.tie)
        self.value.setEnabled(not linked)
        self.free.setEnabled(not linked)
        self.value.setToolTip(trf(LINKED_TIP, expression=parameter.tie) if linked else "")
        self.link.setText("=")
        set_role(self.link, "primary" if linked else None)

    # -- from the editor --------------------------------------------------------------------------

    def refresh(self, errors: Mapping, at_bounds) -> None:
        parameter = self.editor.model.get(self.path)
        set_value(self.value, parameter.value)
        set_value(self.lower, parameter.lower)
        set_value(self.upper, parameter.upper)
        self.unbounded.blockSignals(True)
        self.unbounded.setChecked(math.isinf(parameter.upper))
        self.unbounded.blockSignals(False)
        self.upper.setEnabled(not self.unbounded.isChecked())
        self.free.blockSignals(True)
        self.free.setChecked(parameter.free)
        self.free.blockSignals(False)
        self._linked_state(parameter)
        error = errors.get(self.path)
        # The ± of the value without its unit (the name has it; the tooltip says it whole). A value at a bound has no
        # error and may be wrong: a warning, not a muted note beside the errors.
        set_role(self.error, "warning" if self.path in at_bounds else "muted")
        if self.path in at_bounds:
            self.error.setText("⚠ " + tr("at a bound"))
            self.error.setToolTip(tr("The fit stopped at a bound of the range: widen the range or fix this value"))
        elif error is not None:
            shown = parameter_text(self.path[1], parameter.value, error)
            plus = format_number(error, error, value_only=True) if error >= 0 else "—"
            self.error.setText("± " + plus)
            self.error.setToolTip(tr("1σ error of the last fit") + f": {shown}")
        else:
            self.error.setText("")
            self.error.setToolTip("")

    def show_range(self, on: bool) -> None:
        self.lower.setVisible(on)
        self.upper.setVisible(on)
        self.unbounded.setVisible(on and self.path[1] in LINEAR)


__all__ = ["COLUMNS", "Elided", "ParameterRow", "header", "label_text"]
