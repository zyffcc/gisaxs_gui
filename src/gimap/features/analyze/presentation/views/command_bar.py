"""The Analyze command bar: never clips a control, drops the least important ones when narrow.

With the AI panel docked, or on a small screen, the bar can be ~800 px wide. Below
``COMPACT_WIDTH``, or as soon as the full row does not fit (several files listed add the
stepping arrows, “3 / 40” and Batch Export…), it hides the file stepping arrows and the file
details and shortens the long button texts; the full texts stay in the tooltips. A bar too
narrow even for that puts the actions on a second row. The file name is elided in the middle
rather than cut.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtWidgets import QFrame, QLabel, QSizePolicy, QWidget

from src.gimap.app.presentation.i18n import tr

COMPACT_WIDTH = 1180
"""Below this bar width the compact texts and the hidden extras apply (also when the full row would fit)."""
FULL_AGAIN = 12
"""A compact bar shows the full row again once it has this much room (px) more than the row needs."""


class ElidedLabel(QLabel):
    """A label that shows as much of its text as fits, with “…” in the middle; the tooltip has it all."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumWidth(60)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        self._full = str(text)
        self._refresh()

    def full_text(self) -> str:
        return self._full

    def text(self) -> str:  # noqa: D401 - Qt API
        """The whole text, not the shortened one drawn: what a caller set, and what the language walker looks up
        (“No data yet — o…detector frames” is in no table, so a narrow bar kept the sentence in English)."""
        return self._full

    def sizeHint(self):  # noqa: N802 - Qt API
        hint = super().sizeHint()
        hint.setWidth(min(self.fontMetrics().horizontalAdvance(self._full) + 24, 320))
        return hint

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        room = max(20, self.width() - 16)
        shown = self.fontMetrics().elidedText(self._full, Qt.ElideMiddle, room)
        super().setText(shown)
        if shown != self._full and not self.toolTip():
            self.setToolTip(self._full)


def least_width(widget: QWidget) -> int:
    """The least width a box layout gives ``widget`` (Qt's rule): its minimum width when one is set; else nothing
    for an ``Ignored`` policy, the minimum size hint when it may shrink, the size hint when it may not."""
    if widget.minimumWidth() > 0:
        return widget.minimumWidth()
    policy = widget.sizePolicy().horizontalPolicy()
    if policy == QSizePolicy.Ignored:
        return 0
    width = widget.minimumSizeHint().width()
    if not policy & QSizePolicy.ShrinkFlag:
        width = max(width, widget.sizeHint().width())
    return min(width, widget.maximumWidth())


class CommandBar(QFrame):
    """The bar's row, in four states by the room it has — what each needs is computed from the widgets
    (``row_need``), so more files (the arrows, “3 / 40”, Batch Export…) or a longer language change state earlier
    instead of clipping the mode control and the buttons:

    * full — every part, the full button texts;
    * ``short`` — when the full row does not fit: the texts of ``set_compact_parts(texts=...)`` are the short ones;
    * ``compact`` — below ``COMPACT_WIDTH``, or when that does not fit either: the parts named in
      ``set_compact_parts(hidden=...)`` are left out too;
    * ``wrapped`` — when the compact row does not fit: the actions (``set_rows``) go to a second row.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._hidden: tuple = ()
        self._texts: tuple = ()
        self._row = self._second = self._separator = self._filler = None
        self._actions: tuple = ()
        self.compact: Optional[bool] = None
        self.short = False
        self.wrapped = False

    def set_rows(self, row, second, *, actions=(), separator: Optional[QWidget] = None,
                 filler: Optional[QWidget] = None) -> None:
        """``row``: the box layout holding every part in order, the ``actions`` last (after ``separator``);
        ``second``: the box layout under it, where the actions go when the bar is too narrow for one row;
        ``filler``: the part that takes the room left over while the bar is compact (the file name: the part
        with the stretch is left out then, and the buttons would be stretched instead)."""
        self._row, self._second = row, second
        self._actions, self._separator, self._filler = tuple(actions), separator, filler

    def set_compact_parts(self, *, hidden=(), texts=()) -> None:
        """``texts``: ``(button, full, short)``; buttons keep at least their text width."""
        self._hidden = tuple(hidden)
        self._texts = tuple(texts)
        self._fit()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._fit()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        self._fit()

    def event(self, event) -> bool:  # noqa: N802 - Qt API
        # A part appeared or changed its text without a resize (the file arrows, Batch Export…, the language).
        if event.type() == QEvent.LayoutRequest and self.isVisible():
            self._fit()
        return super().event(event)

    def _parts(self) -> list:
        """Every widget of the bar, in the order of the one row."""
        layouts = [layout for layout in (self._row, self._second) if layout is not None] or (
            [self.layout()] if self.layout() is not None else [])
        found = [layout.itemAt(index).widget() for layout in layouts for index in range(layout.count())]
        return [widget for widget in found if widget is not None]

    def row_need(self, *, short: bool = False, compact: bool = False) -> int:
        """The least width (px) of the bar as one row: with the full texts and every wanted part; with the
        ``short`` texts; or ``compact``, the short texts and without the parts left out. Computed from the
        widgets in the same way whatever the bar's state is now, so a state never changes what the next needs."""
        need = parts = 0
        for widget in self._parts():
            if widget in self._hidden:
                shown = not compact and widget.property("gimapWanted") is not False
            else:  # (the separator of the actions is hidden only while they are on the second row)
                shown = widget is self._separator or not widget.isHidden()
            if shown:
                need += least_width(widget)
                parts += 1
        for button, full, brief in self._texts:  # the texts the buttons would have, against those they have
            if not button.isHidden():
                metrics = button.fontMetrics()
                need += (metrics.horizontalAdvance(tr(brief if short or compact else full))
                         - metrics.horizontalAdvance(tr(brief if self.short else full)))
        layout = self.layout()
        margins = layout.contentsMargins() if layout is not None else self.contentsMargins()
        spacing = max(0, self._row.spacing() if self._row is not None else layout.spacing() if layout is not None else 0)
        return need + margins.left() + margins.right() + spacing * max(0, parts - 1)

    def full_need(self) -> int:
        """``row_need()``: what the full row needs."""
        return self.row_need()

    def _fit(self) -> None:
        """Choose the state for the room there is (a few px more are needed to go back to a wider state, so a
        width at the edge never flips the bar to and fro)."""
        room = self.contentsRect().width()

        def lacks(need: int, now: bool) -> bool:
            return room < need + (FULL_AGAIN if now else 0)

        compact = self.width() < COMPACT_WIDTH or lacks(self.row_need(short=True), bool(self.compact))
        short = compact or lacks(self.row_need(), self.short)
        self._apply(compact)
        self._shorten(short)
        self._wrap(compact and lacks(self.row_need(compact=True), self.wrapped))

    def _apply(self, compact: bool) -> None:
        """The compact row or not: its parts left out, and its texts (short with it, full without)."""
        if compact == self.compact:
            return
        self.compact = compact
        for widget in self._hidden:
            widget.setVisible(not compact and widget.property("gimapWanted") is not False)
        self.short = not compact  # so that the texts are set, also the first time
        self._shorten(compact)
        if self._row is not None and self._filler is not None:
            self._row.setStretchFactor(self._filler, 1 if compact else 0)

    def _shorten(self, short: bool) -> None:
        if short == self.short:
            return
        self.short = short
        for button, full, brief in self._texts:
            button.setText(tr(brief if short else full))
            button.setToolTip(button.toolTip() or tr(full))

    def retranslate(self) -> None:
        """After a switch of the interface language: the texts of the state the bar is in, said again. A text
        the bar set itself in the other language is not one the language walker can turn back exactly (the
        short “拟合” of “Fitting” is also the Chinese of “fit”)."""
        for button, full, brief in self._texts:
            button.setText(tr(brief if self.short else full))

    def _wrap(self, wrapped: bool) -> None:
        """The actions on the second row (their separator is not needed there), or back at the end of the first."""
        if wrapped == self.wrapped or self._row is None or self._second is None:
            return
        self.wrapped = wrapped
        source, target = (self._row, self._second) if wrapped else (self._second, self._row)
        if self._separator is not None:
            self._separator.setVisible(not wrapped)
        for widget in self._actions:
            source.removeWidget(widget)
            target.addWidget(widget)


__all__ = ["COMPACT_WIDTH", "FULL_AGAIN", "CommandBar", "ElidedLabel", "least_width"]
