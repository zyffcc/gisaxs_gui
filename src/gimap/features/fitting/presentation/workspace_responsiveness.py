"""Responsive sizing and persistence for the fitting workspace shell."""

from __future__ import annotations

from typing import Sequence

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QSizePolicy, QWidget

from src.gimap.app.presentation.layout_primitives import CARD_SPACING
from src.gimap.app.presentation.responsive_layout import scale_value

from .run_card import FittingControlsCard


class FittingWorkspaceResponsivenessMixin:
    def _page_min_width(self) -> int:
        return self._control_min_width() + self._preview_min_width() + self.page_splitter.handleWidth()

    def _control_min_width(self) -> int:
        return {"compact": 440, "normal": 480, "wide": 520}.get(self.profile.key, 480)

    def _control_target_width(self) -> int:
        return {"compact": 520, "normal": 580, "wide": 620}.get(self.profile.key, 580)

    def _preview_min_width(self) -> int:
        return max(self.profile.preview_min, scale_value(420, self.profile, 340))

    def _apply_page_overflow_policy(self) -> None:
        self.page_splitter.setMinimumWidth(self._page_min_width())
        self.page_splitter.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Expanding)
        QTimer.singleShot(0, self._set_page_sizes)

    def _available_page_width(self) -> int:
        width = self.page_splitter.width()
        if width > 0:
            return width
        width = self.ui.gisaxsFittingPage.width()
        return width if width > 0 else self._page_min_width()

    def _set_page_sizes(self, sizes: Sequence[int] | None = None) -> None:
        try:
            available = max(1, self._available_page_width() - self.page_splitter.handleWidth())
        except RuntimeError:
            return
        left_min = self._control_min_width()
        right_min = self._preview_min_width()
        if available < left_min + right_min:
            self.page_splitter.setSizes([left_min, right_min])
            return

        left_max = self._control_target_width() + scale_value(80, self.profile, 60)
        requested_left = int(sizes[0]) if sizes and len(sizes) == 2 else self._control_target_width()
        left = min(left_max, max(left_min, requested_left))
        right = available - left
        if right < right_min:
            right = right_min
            left = max(left_min, available - right)
        self.page_splitter.setSizes([left, right])

    def restore_sizes(self) -> None:
        sizes = self.preferences.get(self.SETTINGS_KEY, None)
        if isinstance(sizes, dict):
            if sizes.get("profile") != self.profile.key:
                self._set_page_sizes(self.profile.page_sizes)
                self.work_splitter.setSizes(self.DEFAULT_WORK_SIZES)
                return
            page_sizes = sizes.get("page")
            work_sizes = sizes.get("work")
            self._set_page_sizes(
                page_sizes
                if isinstance(page_sizes, (list, tuple)) and len(page_sizes) == 2
                else self.profile.page_sizes
            )
            if self.work_splitter.count() >= 2:
                if isinstance(work_sizes, (list, tuple)) and len(work_sizes) == 2:
                    self.work_splitter.setSizes(
                        [
                            max(self.DEFAULT_WORK_SIZES[0], int(work_sizes[0])),
                            max(self.DEFAULT_WORK_SIZES[1], int(work_sizes[1])),
                        ]
                    )
                else:
                    self.work_splitter.setSizes(self.DEFAULT_WORK_SIZES)
            return
        self._set_page_sizes(self.profile.page_sizes)
        if self.work_splitter.count() >= 2:
            self.work_splitter.setSizes(self.DEFAULT_WORK_SIZES)

    def save_state(self) -> None:
        self.preferences.set(
            self.SETTINGS_KEY,
            {
                "page": self.page_splitter.sizes(),
                "work": self.work_splitter.sizes() if self.work_splitter.count() >= 2 else [],
                "profile": self.profile.key,
            },
        )

    def apply_responsive_profile(self, profile) -> None:
        self.profile = profile
        self.DEFAULT_WORK_SIZES = list(profile.work_sizes)
        self._configure_button_responsiveness()
        fitting_card = self.fixed_controls_stack.findChild(
            FittingControlsCard, "FittingControlsCard"
        )
        if fitting_card is not None:
            fitting_card.apply_responsive_profile(profile)
        self.right_panel.setMinimumWidth(self._preview_min_width())
        self.right_panel.setMaximumWidth(16777215)
        self.preview_scroll_area.setMinimumWidth(self._preview_min_width())
        self.preview_scroll_area.setMaximumWidth(16777215)
        self.work_splitter.setMinimumWidth(self._control_min_width())
        self.ui.gisaxsFittingPageScrollArea.setMinimumWidth(self._control_min_width())
        maximum = self._control_target_width() + scale_value(80, profile, 60)
        self.ui.gisaxsFittingPageScrollArea.setMaximumWidth(maximum)
        self.left_shell.setMinimumWidth(self._control_min_width())
        self.left_shell.setMaximumWidth(maximum)
        self._apply_page_overflow_policy()

        fixed_min = self._fixed_stack_min_height()
        self.fixed_controls_stack.setMinimumHeight(fixed_min)
        if self.work_area_contents.layout() is not None:
            margins = self.work_area_contents.layout().contentsMargins()
            self.work_area_contents.setMinimumHeight(fixed_min + margins.top() + margins.bottom())
            self.work_area_contents.layout().invalidate()
        self.fixed_controls_stack.layout().invalidate()
        self.fixed_controls_stack.adjustSize()
        self.work_area_contents.adjustSize()
        self._set_page_sizes()
        QTimer.singleShot(0, self._refresh_fixed_stack_geometry)
        if self.work_splitter.count() >= 2:
            self.work_splitter.setSizes(self.DEFAULT_WORK_SIZES)

    def _fixed_stack_min_height(self) -> int:
        workflow_stack = getattr(self, "workflow_content_stack", None)
        current = workflow_stack.currentWidget() if workflow_stack is not None else None
        if current is not None:
            if current is getattr(self, "fitting_fit_step_page", None):
                # The fit page contains a stretch below its current mode.  Its
                # preferred size can therefore reflect the previous viewport;
                # the minimum hint is the current tab's real natural content.
                return max(1, current.minimumSizeHint().height())
            return max(current.minimumSizeHint().height(), current.sizeHint().height())
        card_names = ("GisaxsInputCard", "CutLineCard", "FittingControlsCard", "ModelParameterCard")
        heights = [
            max(widget.minimumHeight(), widget.minimumSizeHint().height(), widget.sizeHint().height())
            for name in card_names
            if (widget := self.fixed_controls_stack.findChild(QWidget, name)) is not None
        ]
        if not heights:
            return self.fixed_controls_stack.minimumHeight()
        return sum(heights) + (len(heights) - 1) * CARD_SPACING

    def _refresh_fixed_stack_geometry(self) -> None:
        try:
            self.fixed_controls_stack.setMinimumHeight(0)
            self.work_area_contents.setMinimumHeight(0)
        except RuntimeError:
            return
        fitting_card = self.fixed_controls_stack.findChild(
            FittingControlsCard, "FittingControlsCard"
        )
        if fitting_card is not None:
            fitting_card._sync_mode_tab_height()
        workflow_stack = getattr(self, "workflow_content_stack", None)
        if workflow_stack is not None and hasattr(workflow_stack, "sync_height"):
            workflow_stack.sync_height()
        self.fixed_controls_stack.layout().activate()
        fixed_min = self._fixed_stack_min_height()
        self.fixed_controls_stack.setMinimumHeight(fixed_min)
        layout = self.work_area_contents.layout()
        margins = layout.contentsMargins()
        natural_height = fixed_min + margins.top() + margins.bottom()
        self.work_area_contents.setMinimumHeight(natural_height)
        viewport = self.ui.gisaxsFittingPageScrollArea.viewport()
        content_height = max(natural_height, viewport.height())
        content_width = max(self._control_min_width(), viewport.width())
        self.work_area_contents.resize(content_width, content_height)
        self.fixed_controls_stack.resize(
            max(1, content_width - margins.left() - margins.right()),
            fixed_min,
        )
        self.fixed_controls_stack.updateGeometry()
        self.work_area_contents.updateGeometry()


__all__ = ["FittingWorkspaceResponsivenessMixin"]
