"""Responsive Style section for the Trainset page."""

from __future__ import annotations


from pathlib import Path

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QScrollArea

from src.gimap.app.presentation.theme import style_widget

TRAINSET_QSS = Path(__file__).resolve().parents[1] / "trainset_theme.qss"


class ResponsiveStyleMixin:
    """Own the responsive style section."""

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._apply_responsive_layout)
        QTimer.singleShot(80, self._apply_responsive_layout)

    def _apply_responsive_layout(self) -> None:
        if not hasattr(self, "stack"):
            return
        content_width = self._workflow_content_width()
        if hasattr(self, "impact_responsive_stack"):
            self.impact_responsive_stack.setCurrentIndex(0 if content_width >= 1040 else 1)
        if hasattr(self, "dataset_splitter"):
            # Side by side only when both panes fit whole: the form (its content plus the
            # vertical scroll bar), the preview and the splitter handle. Otherwise the form
            # would get a horizontal scroll bar and the preview would cover it.
            desired_orientation = (
                Qt.Horizontal if content_width >= self._dataset_side_by_side_width() else Qt.Vertical
            )
            if desired_orientation != getattr(self, "_dataset_splitter_orientation", None):
                self.dataset_splitter.setOrientation(desired_orientation)
                self._dataset_splitter_orientation = desired_orientation
            self.dataset_splitter.setStretchFactor(0, 1)
            self.dataset_splitter.setStretchFactor(1, 1)
            QTimer.singleShot(0, self._balance_dataset_splitter)
        if hasattr(self, "monitor_splitter"):
            self.monitor_splitter.setOrientation(
                Qt.Horizontal if content_width >= 900 else Qt.Vertical
            )
        if hasattr(self, "step_panel"):
            self.step_panel.setMaximumWidth(218 if self.width() >= 1180 else 190)

    def _workflow_content_width(self) -> int:
        """Width of the workflow pages: the stack as laid out, once it is shown.

        Before the first layout pass the stack has no real width yet; then the page width minus
        the step list is the estimate. It overestimates the stack by the content splitter's
        margins and handle, so it is never used while the stack is laid out: a form that does
        not fit beside the preview must not be put beside it.
        """
        measured = self.stack.width()
        if self.stack.isVisible() and measured > 0:
            return measured
        steps = self.step_panel.width() if hasattr(self, "step_panel") else 0
        return max(measured, self.width() - steps - 42)

    def _dataset_side_by_side_width(self) -> int:
        splitter = self.dataset_splitter
        return (
            _pane_min_width(splitter.widget(0))
            + _pane_min_width(splitter.widget(1))
            + splitter.handleWidth()
        )

    def _balance_dataset_splitter(self) -> None:
        """Give both design panes usable space after Qt finishes the resize pass."""
        if not hasattr(self, "dataset_splitter"):
            return
        first = self.dataset_splitter.widget(0)
        second = self.dataset_splitter.widget(1)
        stacked = self.dataset_splitter.orientation() == Qt.Vertical
        ui = getattr(self, "_dataset_page_ui", None)
        for name in ("trainsetDesignPreviewDescription", "designPreviewHint"):
            # Stacked (narrow window): the preview keeps its image; the canvas strip says what to do.
            label = getattr(ui, name, None)
            if label is not None:
                label.setVisible(not stacked)
        if stacked:
            first.setMinimumSize(0, 120)
            second.setMinimumSize(0, 220)
            available = max(340, self.dataset_splitter.height() - self.dataset_splitter.handleWidth())
            preview = min(max(int(available * 0.5), second.minimumSizeHint().height()), available - 120)
            self.dataset_splitter.setSizes([available - preview, preview])
        else:
            first_min, second_min = _pane_min_width(first), _pane_min_width(second)
            first.setMinimumSize(first_min, 0)
            second.setMinimumSize(second_min, 0)
            available = self.dataset_splitter.width() - self.dataset_splitter.handleWidth()
            first_width = max(first_min, int(available * 0.60))
            second_width = max(second_min, available - first_width)
            first_width = max(first_min, available - second_width)
            self.dataset_splitter.setSizes([first_width, second_width])

    def _apply_style(self) -> None:
        style_widget(self, TRAINSET_QSS)


def _pane_min_width(pane) -> int:
    """Narrowest width that shows a pane whole; a scroll area counts its content and scroll bar."""
    if isinstance(pane, QScrollArea) and pane.widget() is not None:
        return (
            pane.widget().minimumSizeHint().width()
            + pane.verticalScrollBar().sizeHint().width()
            + 2 * pane.frameWidth()
        )
    return pane.minimumSizeHint().width()
