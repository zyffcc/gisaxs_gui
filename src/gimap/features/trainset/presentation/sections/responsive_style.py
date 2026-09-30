"""Responsive Style section for the Trainset page."""

from __future__ import annotations


from pathlib import Path

from PyQt5.QtCore import QTimer, Qt

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
        measured = self.stack.width()
        fallback = self.width() - (self.step_list.width() if hasattr(self, "step_list") else 0) - 42
        content_width = max(measured, fallback)
        if hasattr(self, "impact_responsive_stack"):
            self.impact_responsive_stack.setCurrentIndex(0 if content_width >= 1040 else 1)
        if hasattr(self, "dataset_splitter"):
            # The design form and preview have tested minimums of 480 + 340 px.
            # Keep them side-by-side on a 1280×720 screen; stack only on truly
            # narrow windows where those minimums cannot fit.
            desired_orientation = Qt.Horizontal if content_width >= 820 else Qt.Vertical
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
        if hasattr(self, "step_list"):
            self.step_list.setMaximumWidth(218 if self.width() >= 1180 else 190)

    def _balance_dataset_splitter(self) -> None:
        """Give both design panes usable space after Qt finishes the resize pass."""
        if not hasattr(self, "dataset_splitter"):
            return
        first = self.dataset_splitter.widget(0)
        second = self.dataset_splitter.widget(1)
        if self.dataset_splitter.orientation() == Qt.Vertical:
            first.setMinimumSize(0, 220)
            second.setMinimumSize(0, 220)
            available = max(440, self.dataset_splitter.height())
            self.dataset_splitter.setSizes(
                [max(220, int(available * 0.50)), max(220, int(available * 0.50))]
            )
        else:
            first.setMinimumSize(480, 0)
            second.setMinimumSize(340, 0)
            available = max(820, self.dataset_splitter.width())
            self.dataset_splitter.setSizes(
                [max(480, int(available * 0.60)), max(340, int(available * 0.40))]
            )

    def _apply_style(self) -> None:
        style_widget(self, TRAINSET_QSS)
