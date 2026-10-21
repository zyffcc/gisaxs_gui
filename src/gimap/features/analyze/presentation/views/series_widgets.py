"""Widgets of the Series tab that size themselves (``views/series_view.py`` lays them out).

``SeriesScroll`` scrolls the tab only when its window is too short; ``SideBySide`` puts the two plots side by
side or one above the other; ``Snug`` keeps a picture's least height from growing with the height it last
had; ``HintLabel`` is a sentence on one line while every line counts; ``ShrinkingCombo`` a combo that gives
way before the plots must stack.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import QEvent, QSize, Qt
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QBoxLayout,
    QComboBox,
    QFrame,
    QLabel,
    QLayout,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyleOptionComboBox,
    QStylePainter,
    QWidget,
)


class HintLabel(QLabel):
    """The sentence under the Series controls: whole, over as many lines as it takes — or, ``set_one_line(True)``
    while a map is shown and every line counts, on one line cut with “…”, the whole sentence in its tooltip.
    ``text()`` is always the whole sentence (what was set, and what the language walker looks up)."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._whole, self._one_line = "", False
        self.setWordWrap(True)

    def set_one_line(self, one_line: bool) -> None:
        if bool(one_line) == self._one_line:
            return
        self._one_line = bool(one_line)
        self.setWordWrap(not self._one_line)
        # One line: as wide as the tab, never asking for the room of the whole sentence.
        self.setSizePolicy(QSizePolicy.Ignored if self._one_line else QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._draw()

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        self._whole = str(text)
        self._draw()

    def text(self) -> str:
        return self._whole

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if self._one_line:
            self._draw()

    def _draw(self) -> None:
        shown = self._whole
        if self._one_line:
            shown = self.fontMetrics().elidedText(self._whole, Qt.ElideRight, max(0, self.width()))
        super().setText(shown)
        self.setToolTip(self._whole if shown != self._whole else "")


class ShrinkingCombo(QComboBox):
    """A combo as wide as its longest item while there is room, down to ``least_characters`` when there is not
    (``minimumSizeHint``), its text then shortened with “…”; its list shows the items whole."""

    def __init__(self, parent: QWidget, *, least_characters: int):
        super().__init__(parent)
        self._least_characters = int(least_characters)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        full = super().minimumSizeHint()
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        text = QSize(self._least_characters * self.fontMetrics().horizontalAdvance("x"), full.height())
        least = self.style().sizeFromContents(QStyle.CT_ComboBox, option, text, self)
        return QSize(min(full.width(), least.width()), full.height())

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        field = self.style().subControlRect(QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, self)
        option.currentText = self.fontMetrics().elidedText(option.currentText, Qt.ElideRight, max(0, field.width()))
        painter.drawComplexControl(QStyle.CC_ComboBox, option)
        painter.drawControl(QStyle.CE_ComboBoxLabel, option)


class SideBySide(QWidget):
    """Two widgets side by side while both get their least width, else one above the other, so a narrow panel
    keeps both whole (the frame and trace plots of the Series tab). Its least width is the wider one's."""

    def __init__(self, first: QWidget, second: QWidget, parent: QWidget, *, spacing: int = 6,
                 least_height: tuple[int, int] = (0, 0)):
        super().__init__(parent)
        self._pair = (first, second)
        self._least_height = tuple(int(value) for value in least_height)
        """``(side by side, each when one is above the other)``: the least heights (px)."""
        self._box = QBoxLayout(QBoxLayout.LeftToRight, self)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._box.setSpacing(spacing)
        self._box.setSizeConstraint(QLayout.SetNoConstraint)  # the least width is ``minimumSizeHint``'s
        for widget in self._pair:
            self._box.addWidget(widget, 1)
        self._keep_least_height()

    def _keep_least_height(self) -> None:
        """The least height as the widget's own minimum: a layout that sizes by height-for-width (the Series tab,
        with its wrapping sentence) asks the box inside for its least height and would give the pair less."""
        self.setMinimumHeight(self.minimumSizeHint().height())

    def stacked(self) -> bool:
        return self._box.direction() == QBoxLayout.TopToBottom

    def side_by_side_width(self) -> int:
        return sum(widget.minimumSizeHint().width() for widget in self._pair) + self._box.spacing()

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        hints = [widget.minimumSizeHint() for widget in self._pair]
        side_by_side, stacked = self._least_height
        if self.stacked():
            height = sum(max(hint.height(), stacked) for hint in hints) + self._box.spacing()
        else:
            height = max(side_by_side, *(hint.height() for hint in hints))
        return QSize(max(hint.width() for hint in hints), height)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        """As tall as side by side, also when one is above the other: the map above keeps its share of the
        height, and the two plots share theirs."""
        hints = [widget.sizeHint() for widget in self._pair]
        width = sum(hint.width() for hint in hints) + self._box.spacing()
        return QSize(width, max(self.minimumSizeHint().height(), *(hint.height() for hint in hints)))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        stacked = event.size().width() < self.side_by_side_width()
        if stacked != self.stacked():
            self._box.setDirection(QBoxLayout.TopToBottom if stacked else QBoxLayout.LeftToRight)
            self._keep_least_height()
            self.updateGeometry()


class StackedPlots(SideBySide):
    """Series curves always share the full panel width, including Wide mode and a large window."""

    def __init__(self, first, second, parent, **options):
        super().__init__(first, second, parent, **options)
        self._box.setDirection(QBoxLayout.TopToBottom)
        self._keep_least_height()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        QWidget.resizeEvent(self, event)


class SeriesScroll(QScrollArea):
    """The Series tab in a scroll area that changes nothing while the tab fits: its content is then exactly as
    tall as the view, laid out as without a scroll area. Only when the view is shorter than the content's least
    height (a laptop window: the map and the plots keep ``SERIES_MAP_HEIGHT`` / ``SERIES_PLOTS_HEIGHT``) does
    the content keep that height and scroll. The content is always as wide as the view, so there is never a
    horizontal bar, and the tab's least width is the content's (plus the vertical bar's).

    Not ``setWidgetResizable(True)``: Qt then gives the content its preferred height, which with the plots
    (their size hints follow their last size) grows after the window has been large."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidgetResizable(False)

    def setWidget(self, widget: QWidget) -> None:  # noqa: N802 - Qt API
        super().setWidget(widget)
        widget.installEventFilter(self)
        for control in (*widget.findChildren(QComboBox), *widget.findChildren(QAbstractSpinBox)):
            control.installEventFilter(self)  # the wheel over them scrolls the tab (``eventFilter``)
        self._fit()

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        hint, content = super().minimumSizeHint(), self.widget()
        if content is None:
            return hint
        return QSize(content.minimumSizeHint().width() + self.verticalScrollBar().sizeHint().width(), hint.height())

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API (also when the vertical bar comes or goes)
        super().resizeEvent(event)
        self._fit()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if watched is self.widget():
            if event.type() == QEvent.LayoutRequest:
                self.updateGeometry()  # the least width may have changed
                self._fit()  # a part was shown or hidden, a sentence changed: another least height
        # Scrolling the tab with the pointer over a combo or a spin box must not change it (the number of stages,
        # every n-th frame): the wheel changes a control only once it was clicked, else it scrolls the tab.
        # Only those controls: Qt also filters the scroll bars' own events through here, and passing their
        # wheel on to ``wheelEvent`` (which sends it to the bar) would never end.
        elif (event.type() == QEvent.Wheel and isinstance(watched, (QComboBox, QAbstractSpinBox))
              and not watched.hasFocus() and self.verticalScrollBar().maximum() > 0):
            self.wheelEvent(event)
            return True
        return super().eventFilter(watched, event)  # (the scroll area follows its content's size there)

    def least_height(self, width: int) -> int:
        """The least height of the content laid out ``width`` px wide."""
        layout = self.widget().layout()
        if layout is None:
            return self.widget().minimumSizeHint().height()
        return layout.minimumHeightForWidth(width) if layout.hasHeightForWidth() else layout.minimumSize().height()

    def _fit(self) -> None:
        content = self.widget()
        if content is None:
            return
        view = self.viewport().size()
        size = QSize(view.width(), max(view.height(), self.least_height(view.width())))
        if content.size() != size:
            content.resize(size)


class Snug(QWidget):
    """Holds one widget and asks its layout for that widget's *least* height at the width it has — not for
    the height the widget last had.

    A pyqtgraph view's size hint is the size it has, and a layout takes the hint of a widget that has a
    layout of its own for that widget's least height. So the least height of the Series tab grew every time
    the map or the plots had been tall (a large window, a section opened and folded again), until the tab
    scrolled in a window it had fitted before. ``least(width)``: the held widget's least height (by default
    that of its ``minimumSizeHint``). Shown and hidden with the widget it holds."""

    def __init__(self, inner: QWidget, parent: QWidget, least: Optional[Callable[[int], int]] = None):
        super().__init__(parent)
        self._inner = inner
        self._least = least or (lambda _width: inner.minimumSizeHint().height())
        self._asked = -1
        inner.setParent(self)
        inner.installEventFilter(self)
        self.setVisible(not inner.isHidden())

    def least_height(self) -> int:
        return max(int(self._least(max(1, self.width()))), self._inner.minimumHeight())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(self._inner.minimumSizeHint().width(), self.least_height())

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return self.minimumSizeHint()

    def _settle(self) -> None:
        """The held widget fills this one; a least height that changed (a toolbar that wraps, plots that
        stack) is passed on to the layout around."""
        if self._inner.geometry() != self.rect():
            self._inner.setGeometry(self.rect())
        least = self.least_height()
        if least != self._asked:
            self._asked = least
            self.updateGeometry()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._settle()

    def event(self, event) -> bool:  # noqa: A003 - Qt API
        if event.type() == QEvent.LayoutRequest:  # the held widget's needs changed
            self._settle()
        return super().event(event)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if watched is self._inner and event.type() in (QEvent.ShowToParent, QEvent.HideToParent):
            self.setVisible(event.type() == QEvent.ShowToParent)
        return False


__all__ = ["HintLabel", "SeriesScroll", "ShrinkingCombo", "SideBySide", "Snug", "StackedPlots"]
