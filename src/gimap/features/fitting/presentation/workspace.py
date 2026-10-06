"""Fitting workspace: Single analysis (one curve, ``single/page.py``) and In-situ series.

Curves come from Analyze (Send to Fitting) or a curve file. The former single-analysis page
(the classic widgets of ``FittingPageView`` arranged into cards below) is still built, hidden:
its binding runs In-situ series, and follows the curve and the model of the new page
(``attach_legacy``) so that series take their set-up from what is fitted here.
"""

from __future__ import annotations

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.ports import UserPreferencesRepository
from src.gimap.app.presentation import install_safe_wheel_behavior
from src.gimap.app.presentation.components import AdvancedSection
from src.gimap.app.presentation.i18n import tr
from src.gimap.app.presentation.layout_metrics import LAYOUT
from src.gimap.app.presentation.layout_primitives import (
    CARD_SPACING,
    INPUT_WIDGET_TYPES,
    normalize_checkbox,
    normalize_input,
)

from .curve_card import CurveSourceCard
from .fit_steps import FitSteps
from .fitting_theme import apply_fitting_style
from .layout_primitives import detach_from_parent_layout as _detach_from_parent_layout
from .layout_primitives import take_widget as _take_widget
from .model_card import ModelParameterCard
from .preview_cards import FittingPlotControlsCard, PlotPreviewCard, StatusCard
from .run_card import FittingControlsCard
from ..application.single_fit import to_manual
from .single.page import FitPage
from .single.series_page import FitSeriesPage
from .workspace_context import FittingContextContainer

LEGACY_Q_VIEW = {"mean": "average", "both": "fold", "positive": "positive", "negative": "negative_abs"}

CONTROLS_MIN_WIDTH = 440
CONTROLS_TARGET_WIDTH = 540
RESULTS_MIN_WIDTH = 460


def _embed_in_section(card) -> None:
    """A card shown inside an AdvancedSection: the section is its title, toggle and frame."""
    card.set_expanded(True)
    card.header_widget.hide()
    card.setProperty("card", False)
    card.body_layout.setContentsMargins(0, 0, 0, 0)
    card.style().unpolish(card)
    card.style().polish(card)


class FittingWorkspace:
    """Single analysis (one curve) and In-situ series (many curves) of Fitting."""

    SETTINGS_KEY = "fitting_splitter_sizes_v3"

    def __init__(
        self,
        ui,
        profile=None,
        *,
        preferences: UserPreferencesRepository,
        view_model,
        quick_fit=None,
    ):
        self.ui = ui
        self.preferences = preferences
        self.view_model = view_model
        self._legacy = None
        self.fit_page = FitPage(view_model, quick_fit=quick_fit, preferences=preferences)
        self.fit_page.setObjectName("fittingSinglePage")
        self.series_page = FitSeriesPage(view_model, self.fit_page, preferences=preferences)
        self.series_page.editModelRequested.connect(lambda: self.show_context("single"))
        self.series_page.openFrameRequested.connect(self._frame_in_single)
        self.profile = profile or LAYOUT
        legacy_scroll_area = ui.gisaxsFittingPageScrollArea
        self.page_splitter = QSplitter(Qt.Horizontal, ui.gisaxsFittingPage)
        self.page_splitter.setObjectName("fittingPageSplitter")
        self.page_splitter.setChildrenCollapsible(False)
        self.page_splitter.setHandleWidth(8)
        apply_fitting_style(self.page_splitter)  # before it is populated: one polish pass

        _take_widget(ui.gridLayout_24, ui.curvePlotControlWidget)
        _take_widget(ui.verticalLayout_19, ui.FittingTextBrowser)
        self._configure_inputs()
        self._build_controls()
        self._build_results()
        self._install_context(legacy_scroll_area)
        legacy_scroll_area.deleteLater()
        self.page_splitter.setStretchFactor(0, 0)
        self.page_splitter.setStretchFactor(1, 1)
        self.restore_sizes()
        install_safe_wheel_behavior(self.page_splitter)

    # -- left: curve, model and fitting controls ------------------------------------------

    def _build_controls(self) -> None:
        self.controls_scroll_area = QScrollArea(self.page_splitter)
        self.controls_scroll_area.setObjectName("fittingControlsScrollArea")
        self.controls_scroll_area.setWidgetResizable(True)
        self.controls_scroll_area.setFrameShape(QFrame.NoFrame)
        self.controls_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.controls_scroll_area.setMinimumWidth(CONTROLS_MIN_WIDTH)
        content = QWidget()
        content.setObjectName("fittingControlsContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(CARD_SPACING)

        self.curve_card = CurveSourceCard(self.ui, content)
        layout.addWidget(self.curve_card)

        self.model_parameters_card = ModelParameterCard(self.ui, self.profile)
        self.fitting_controls_card = FittingControlsCard(
            self.ui,
            self.profile,
            model_parameters_card=self.model_parameters_card,
            preferences=self.preferences,
        )
        fit_section = QFrame(content)
        fit_section.setObjectName("fittingRunSection")
        fit_section.setProperty("gimapSection", True)
        fit_layout = QVBoxLayout(fit_section)
        fit_layout.setContentsMargins(12, 10, 12, 12)
        fit_layout.setSpacing(6)
        title = QLabel("Model and fit", fit_section)
        title.setProperty("gimapSectionTitle", True)
        description = QLabel(
            "Choose the components and global parameters, plot the model, refine it, "
            "or let 1D Predict fit the curve.",
            fit_section,
        )
        description.setProperty("gimapSectionDescription", True)
        description.setWordWrap(True)
        fit_layout.addWidget(title)
        fit_layout.addWidget(description)
        fit_layout.addWidget(self.fitting_controls_card)
        layout.addWidget(fit_section)
        layout.addStretch(1)
        self.controls_scroll_area.setWidget(content)
        self.controls_content = content
        self.steps = FitSteps(self, lambda: getattr(getattr(self.ui, "runtime", None), "fitting", None))

    # -- right: plot, export, plot controls and log ------------------------------------------

    def _build_results(self) -> None:
        self.results_scroll_area = QScrollArea(self.page_splitter)
        self.results_scroll_area.setObjectName("fittingResultsScrollArea")
        self.results_scroll_area.setWidgetResizable(True)
        self.results_scroll_area.setFrameShape(QFrame.NoFrame)
        self.results_scroll_area.setMinimumWidth(RESULTS_MIN_WIDTH)
        panel = QWidget()
        panel.setObjectName("fittingResultsPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(CARD_SPACING)

        self.inline_feedback = QLabel("", panel)
        self.inline_feedback.setObjectName("fittingInlineFeedback")
        self.inline_feedback.setProperty("feedbackKind", "error")
        self.inline_feedback.setWordWrap(True)
        self.inline_feedback.setVisible(False)
        layout.addWidget(self.inline_feedback)

        self.fitting_plot_card = PlotPreviewCard(
            self.ui, self.ui.curvePlotControlWidget, self.ui.fitGraphicsView, self.profile
        )
        self.curve_plot_card = self.fitting_plot_card
        layout.addWidget(self.fitting_plot_card, 1)

        export_row = QHBoxLayout()
        export_row.setSpacing(8)
        export_button = self.ui.FittingExportButton
        plot_button = self.fitting_controls_card.fitExportPlotButton
        for button, text in ((export_button, "Export Data…"), (plot_button, "Export Plot…")):
            _detach_from_parent_layout(button)
            button.setParent(panel)
            button.setText(text)
            export_row.addWidget(button)
        self.ui.fitExportPlotButton = plot_button
        export_row.addStretch(1)
        layout.addLayout(export_row)

        self.plot_controls_section = AdvancedSection(
            "Fit range and plot options",
            "The q range that is fitted, sampling and which components are drawn.",
            panel,
        )
        self.fitting_controls_plot_card = FittingPlotControlsCard(
            self.ui, self.ui.curvePlotControlWidget, self.profile
        )
        _embed_in_section(self.fitting_controls_plot_card)
        self.plot_controls_section.add_widget(self.fitting_controls_plot_card)
        layout.addWidget(self.plot_controls_section)

        self.log_section = AdvancedSection("Run Log", "", panel)
        self.run_log_card = StatusCard(self.ui.FittingTextBrowser, self.profile)
        _embed_in_section(self.run_log_card)
        self.log_section.add_widget(self.run_log_card)
        layout.addWidget(self.log_section)
        self.results_scroll_area.setWidget(panel)
        self.results_panel = panel

        self.ui.fittingPlotCard = self.fitting_plot_card
        self.ui.fittingPlotControlsCard = self.fitting_controls_plot_card
        self.ui.runLogCard = self.run_log_card
        self.ui.fittingInlineFeedback = self.inline_feedback

    # -- page ------------------------------------------------------------------------------

    def _install_context(self, legacy_scroll_area) -> None:
        layout = self.ui.verticalLayout_19
        _take_widget(layout, legacy_scroll_area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.context_container = FittingContextContainer(
            self.fit_page,
            self.view_model.insitu,
            self.ui.gisaxsFittingPage,
            series_page=self.series_page,
        )
        self.context_container.stack.addWidget(self.page_splitter)  # the former page: kept, never shown
        self.context_bar = self.context_container.context_bar
        self.context_stack = self.context_container.stack
        self.context_button_group = self.context_container.button_group
        self.single_context_button = self.context_container.single_button
        self.insitu_context_button = self.context_container.insitu_button
        self.insitu_series_page = self.context_container.insitu_page
        layout.addWidget(self.context_container, 1)
        self.ui.fittingWorkspace = self
        self.ui.fittingContextStack = self.context_stack
        self.ui.fittingInsituSeriesPage = self.insitu_series_page
        self.ui.fittingSingleContextButton = self.single_context_button
        self.ui.fittingInsituContextButton = self.insitu_context_button
        self.ui.fittingSinglePage = self.fit_page
        # Before In-situ series captures its set-up (the binding connects later): the model of this page.
        self.insitu_series_page.capture_recipe_requested.connect(self._sync_legacy_model)
        self.show_context("single")

    def _configure_inputs(self) -> None:
        for widget in self.ui.gisaxsFittingPage.findChildren(INPUT_WIDGET_TYPES):
            normalize_input(widget)
        for checkbox in self.ui.gisaxsFittingPage.findChildren(QCheckBox):
            normalize_checkbox(checkbox)

    def show_context(self, context: str) -> None:
        """Switch between Single analysis and In-situ series (both keep their state)."""
        self.context_container.show_context(context)

    def show_fit_curve(self) -> None:
        """Show Single analysis."""
        self.show_context("single")

    def open_curve(self, path, side: str = "mean") -> bool:
        """A curve from Analyze (``side`` as Analyze names the halves) on the Single analysis page."""
        self.show_context("single")
        return self.fit_page.open_curve(path, side)

    def open_series(self, folder, pattern: str = "*_fit_input.dat") -> bool:
        """A folder of curves (Analyze ▸ Send Series to Fitting) on the In-situ series page."""
        self.show_context("insitu")
        return self.series_page.open_series(folder, pattern)

    def _frame_in_single(self, path: str, model) -> None:
        """A frame of the series, with its fitted model, in Single analysis — on the points the series
        fitted (its halves, range and left-out points), so that the next Start reads the same settings."""
        self.show_context("single")
        settings = getattr(self.series_page, "_settings", None)
        page = self.fit_page
        if not page.open_curve(path, settings.side if settings is not None else page.session.side):
            return
        if settings is not None:
            page.session.excluded = set(settings.excluded)
            page.set_range(settings.q_range, record=False)  # with the curve: Undo then brings back the model
        if model is not None:
            page.set_model(model)
            page._status(lambda: tr("The range and left-out points of the series are kept; the series now starts from "
                                    "this frame's model (Undo brings the previous one back)."))
        else:
            page._status(lambda: tr("The range and left-out points of the series are kept."))  # again after a switch

    def show_solution(self, row: dict) -> bool:
        """A solution of Analyze's automatic analysis as the model of Single analysis."""
        self.show_context("single")
        return self.fit_page.show_solution(row)

    # -- the former page, for In-situ series ----------------------------------------------

    def attach_legacy(self, binding) -> None:
        """The binding of the former page runs In-situ series; it follows this page's curve and model."""
        self._legacy = binding
        self.fit_page.curveOpened.connect(self._legacy_curve)
        self.fit_page.batchRequested.connect(binding.open_ai_fitting_workspace)
        curve = self.fit_page.session.curve
        if curve is not None and curve.path:
            self._legacy_curve(curve.path)

    def _legacy_curve(self, path: str) -> None:
        if self._legacy is None:
            return
        try:
            self._legacy.import_1d_file(path, q_view=LEGACY_Q_VIEW.get(self.fit_page.session.side, "fold"))
        except Exception as exc:  # the series set-up only; the page itself has the curve
            self.fit_page._log(f"In-situ series could not take this curve: {exc}")

    def _sync_legacy_model(self) -> None:
        if self._legacy is None:
            return
        try:
            self._legacy._load_parameter_mapping(to_manual(self.fit_page.session.model))
        except Exception as exc:
            self.fit_page._log(f"In-situ series could not take this model: {exc}")

    # -- sizes -----------------------------------------------------------------------------

    def _set_page_sizes(self, left=None) -> None:
        total = max(self.page_splitter.width(), CONTROLS_MIN_WIDTH + RESULTS_MIN_WIDTH)
        left = int(left) if left else CONTROLS_TARGET_WIDTH
        left = max(CONTROLS_MIN_WIDTH, min(left, total - RESULTS_MIN_WIDTH))
        self.page_splitter.setSizes([left, max(RESULTS_MIN_WIDTH, total - left)])

    def restore_sizes(self) -> None:
        stored = self.preferences.get(self.SETTINGS_KEY, None)
        left = stored.get("left") if isinstance(stored, dict) else None
        QTimer.singleShot(0, lambda: self._set_page_sizes(left))

    def save_state(self) -> None:
        sizes = self.page_splitter.sizes()
        if sizes and sizes[0] > 0:
            self.preferences.set(self.SETTINGS_KEY, {"left": int(sizes[0])})


__all__ = ["FittingWorkspace"]
