"""The steps of an Analyze run as a compact checklist tree, like a task list (a widget of the live view and of
Details ▸ Steps).

One row per step: a mark — ○ pending, ● running, ✓ done, ✗ failed, – skipped —, its title and, at the right, the one
line of what it found (cut with “…”, whole in the tooltip). The step at work has the line of what happens now under
it (wrapped). A parent shows its children while it runs or failed and folds them away once it is done (▸ / ▾ opens
and closes them by hand; that choice holds for the run). A step whose details can be shown ends in “›”; a click on it
emits ``stepChosen(id)`` — a click on any row does, the page decides what it shows. The row shown is marked
(``set_current``).

The rows are built again only when steps come or go (a network starts); otherwise their words and marks change in
place. Nothing here knows the package: ``show_steps`` takes the nodes (``application.live_steps.StepNode``) and the words in
the interface language.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from src.gimap.app.presentation.theme import set_role

from .views.sputter_main_view import ElidedLabel

MARKS = {"pending": "○", "running": "●", "done": "✓", "failed": "✗", "skipped": "–"}
ROLES = {"pending": "muted", "running": "accent", "done": "success", "failed": "danger", "skipped": "muted"}
OPEN_STATES = ("running", "failed")
"""A parent in one of these states shows its children; otherwise they are folded away."""
DETAIL_MARK = "  ›"
FOLDED, OPENED = "▸", "▾"
INDENT = 18


class _Wrapped(QLabel):
    """A wrapped line as tall as its text at the width it has (a wrapped label in a row of fixed height would be cut)."""

    def __init__(self, parent: QWidget):
        super().__init__("", parent)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        super().setText(text)
        self._fit()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        margins = self.contentsMargins()
        room = max(40, self.width() - margins.left() - margins.right())
        box = self.fontMetrics().boundingRect(0, 0, room, 100_000, int(Qt.AlignLeft | Qt.TextWordWrap), self.text())
        needed = box.height() + margins.top() + margins.bottom()
        if self.height() != needed or self.minimumHeight() != needed:
            self.setFixedHeight(needed)


class _Title(QLabel):
    """A step's title: as wide as it needs while there is room, cut with “…” when there is not (it never widens the
    tree)."""

    def __init__(self, parent: QWidget):
        super().__init__("", parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    def set_full_text(self, text: str) -> None:
        self._full = str(text)
        self._fit()
        self.updateGeometry()

    def full_text(self) -> str:
        return self._full

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(self.fontMetrics().horizontalAdvance(self._full) + 4, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(40, super().minimumSizeHint().height())

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        shown = self.fontMetrics().elidedText(self._full, Qt.ElideRight, max(20, self.width()))
        if shown != self.text():
            self.setText(shown)


class _Toggle(QLabel):
    """▸ / ▾ of a parent: a click shows or folds its children."""

    clicked = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(FOLDED, parent)
        self.setFixedWidth(14)
        self.setAlignment(Qt.AlignCenter)
        self.setProperty("gimapRole", "muted")
        policy = self.sizePolicy()
        policy.setRetainSizeWhenHidden(True)  # the rows without children keep their marks in line
        self.setSizePolicy(policy)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class _Row(QWidget):
    clicked = pyqtSignal()

    def __init__(self, step_id: str, depth: int, parent: QWidget):
        super().__init__(parent)
        self.step_id = step_id
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setProperty("gimapStep", True)
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("sputterStep_" + step_id.replace(".", "_"))
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6 + depth * INDENT, 3, 6, 3)
        outer.setSpacing(1)
        line = QHBoxLayout()
        line.setSpacing(6)
        self.toggle = _Toggle(self)  # a parent's: shows or folds its children
        self.toggle.setVisible(False)
        self.mark = QLabel("", self)
        self.mark.setFixedWidth(14)
        self.mark.setAlignment(Qt.AlignCenter)
        self.title = _Title(self)
        self.title.setProperty("gimapStepTitle", depth == 0)
        self.summary = ElidedLabel(self)
        self.summary.setProperty("gimapMeta", True)
        self.summary.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        line.addWidget(self.toggle)
        line.addWidget(self.mark)
        line.addWidget(self.title)
        line.addWidget(self.summary, 1)
        outer.addLayout(line)
        self.hint = _Wrapped(self)  # what happens now, under the step at work
        self.hint.setProperty("gimapRole", "info")
        self.hint.setContentsMargins(14 + 6 + 14 + 6, 0, 0, 0)  # under the title
        self.hint.setVisible(False)
        outer.addWidget(self.hint)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class StepChecklist(QWidget):
    stepChosen = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("sputterStepTree")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(1)
        self._rows: dict = {}
        self._ids: tuple = ()
        self._parents: dict = {}
        self._opened: dict = {}
        """Parent id -> shown or not, as the user chose with its arrow (else the state decides)."""
        self._nodes: list = []
        self._current = ""

    # -- what is shown --------------------------------------------------------------------------------------------
    def show_steps(self, nodes, *, title: Callable[[object], str], summary: Callable[[object], str], hint: str = "",
                   hint_at: str = "") -> None:
        """The steps ``nodes`` (in their order; ``title(node)`` and ``summary(node)`` in the interface language) and
        ``hint`` under step ``hint_at``."""
        nodes = list(nodes)
        ids = tuple(node.id for node in nodes)
        if ids != self._ids:
            self._build(nodes)
        self._nodes = nodes
        for node in nodes:
            row = self._rows[node.id]
            row.mark.setText(MARKS.get(node.state, MARKS["pending"]))
            set_role(row.mark, ROLES.get(node.state, "muted"))
            row.title.set_full_text(title(node) + (DETAIL_MARK if node.detail else ""))
            words = summary(node)
            row.summary.set_full_text(words)
            row.setToolTip("\n".join(part for part in (title(node), words) if part))
            shown = node.id == hint_at and bool(hint)
            if shown:
                row.hint.setText(hint)
            row.hint.setVisible(shown)
        self._fold()

    def clear(self) -> None:
        """No steps (the rows go)."""
        self._build([])
        self._nodes, self._opened, self._current = [], {}, ""

    def set_current(self, step_id: str) -> None:
        """Mark the row shown ("" none)."""
        self._current = step_id
        for key, row in self._rows.items():
            if bool(row.property("current")) != (key == step_id):
                row.setProperty("current", key == step_id)
                row.style().unpolish(row)
                row.style().polish(row)

    def current(self) -> str:
        return self._current

    def open_children(self, parent: str) -> None:
        """Show the children of ``parent`` (as its arrow does)."""
        if parent in self._parents:
            self._opened[parent] = True
            self._fold()

    def row(self, step_id: str) -> Optional[_Row]:
        return self._rows.get(step_id)

    def shown_ids(self) -> list:
        """The steps whose rows are shown (not folded away), in their order."""
        return [key for key, row in self._rows.items() if not row.isHidden()]

    # -- building -------------------------------------------------------------------------------------------------
    def _build(self, nodes: list) -> None:
        for row in self._rows.values():
            self._layout.removeWidget(row)
            row.hide()
            row.deleteLater()
        self._rows, self._parents = {}, {}
        for node in nodes:
            depth = 1 if node.parent else 0
            row = _Row(node.id, depth, self)
            row.clicked.connect(lambda key=node.id: self.stepChosen.emit(key))
            self._layout.addWidget(row)
            self._rows[node.id] = row
            if node.parent:
                self._parents.setdefault(node.parent, []).append(node.id)
        for parent in self._parents:
            if parent in self._rows:
                toggle = self._rows[parent].toggle
                toggle.setVisible(True)
                toggle.clicked.connect(lambda _checked=False, key=parent: self._flip(key))
        self._ids = tuple(node.id for node in nodes)
        self.set_current(self._current)

    def _flip(self, parent: str) -> None:
        self._opened[parent] = not self._open(parent)
        self._fold()

    def _open(self, parent: str) -> bool:
        if parent in self._opened:
            return self._opened[parent]
        node = next((node for node in self._nodes if node.id == parent), None)
        children = [node for node in self._nodes if node.parent == parent]
        return node is not None and (node.state in OPEN_STATES or any(child.state in OPEN_STATES for child in children))

    def _fold(self) -> None:
        for parent, children in self._parents.items():
            opened = self._open(parent)
            if parent in self._rows:
                self._rows[parent].toggle.setText(OPENED if opened else FOLDED)
            for child in children:
                self._rows[child].setVisible(opened)


__all__ = ["MARKS", "StepChecklist"]
