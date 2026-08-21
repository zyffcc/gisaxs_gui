"""Stable Single analysis / In-situ series context container for Fitting."""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation import apply_design_system

from .fitting_theme import fitting_stylesheet
from .insitu_series_page import InSituSeriesPage


class _CurrentContextStack(QStackedWidget):
    """Do not let a hidden work context determine the visible page geometry."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.currentChanged.connect(lambda _index: self.updateGeometry())

    def sizeHint(self):
        current = self.currentWidget()
        return current.sizeHint() if current is not None else super().sizeHint()

    def minimumSizeHint(self):
        current = self.currentWidget()
        if current is not None:
            return current.minimumSizeHint()
        return super().minimumSizeHint()


class FittingContextContainer(QWidget):
    """Switch work contexts while preserving both child pages in memory."""

    def __init__(self, single_page: QWidget, insitu_view_model, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("fittingContextContainer")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.context_bar = QFrame(self)
        self.context_bar.setObjectName("fittingContextBar")
        bar_layout = QHBoxLayout(self.context_bar)
        bar_layout.setContentsMargins(16, 8, 16, 8)
        bar_layout.setSpacing(6)
        context_label = QLabel("Fitting", self.context_bar)
        context_label.setObjectName("fittingContextLabel")
        context_label.setProperty("gimapSectionTitle", True)
        bar_layout.addWidget(context_label)
        bar_layout.addSpacing(12)

        self.single_button = QPushButton("Single analysis", self.context_bar)
        self.single_button.setObjectName("fittingSingleContextButton")
        self.insitu_button = QPushButton("In-situ series", self.context_bar)
        self.insitu_button.setObjectName("fittingInsituContextButton")
        self.button_group = QButtonGroup(self.context_bar)
        self.button_group.setExclusive(True)
        for index, button in enumerate((self.single_button, self.insitu_button)):
            button.setCheckable(True)
            button.setProperty("fittingContextButton", True)
            self.button_group.addButton(button, index)
            bar_layout.addWidget(button)
        bar_layout.addStretch(1)

        self.stack = _CurrentContextStack(self)
        self.stack.setObjectName("fittingContextStack")
        self.insitu_page = InSituSeriesPage(insitu_view_model, self.stack)
        self.stack.addWidget(single_page)
        self.stack.addWidget(self.insitu_page)
        root.addWidget(self.context_bar)
        root.addWidget(self.stack, 1)

        self.button_group.buttonClicked[int].connect(
            lambda index: self.show_context("insitu" if index == 1 else "single")
        )
        self.insitu_page.return_to_single_requested.connect(
            lambda: self.show_context("single")
        )
        for widget in (self.context_bar, self.stack):
            apply_design_system(widget)
            widget.setStyleSheet(widget.styleSheet() + "\n" + fitting_stylesheet())
        self.show_context("single")

    def show_context(self, context: str) -> None:
        insitu = str(context).strip().casefold() in {"insitu", "in-situ", "series"}
        self.stack.setCurrentIndex(1 if insitu else 0)
        self.single_button.setChecked(not insitu)
        self.insitu_button.setChecked(insitu)
        if insitu:
            self.insitu_page.render_recipe(self.insitu_page.view_model.recipe)
            self.insitu_page.render_workflow(self.insitu_page.view_model.state)


__all__ = ["FittingContextContainer"]
