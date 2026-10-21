"""The steps of fitting one curve and the controls of each (static layout; behaviour in ``single/``).

Curve → Model → Fit → Results, as in Analyze: each step page has a title, one sentence of where
it stands (``step_intro[key]``, filled by the page), the controls used most, and the rest folded.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.components import AdvancedSection, StepRail

from ..layout_primitives import ScientificDoubleSpinBox
from .fit_results_view import build_results_step
from .fit_step_view import METHODS, QUICK_FAMILIES, build_fit_step
from .step_widgets import fit_to_rows, info_card, muted, read_only_table

STEPS = (("curve", "Curve"), ("model", "Model"), ("fit", "Fit"), ("results", "Results"))
STEP_INTROS = {"fit": "Choose a method and press Fit; values that stop at an edge of their range are dealt with by the "
                      "fit."}
"""The intro of a step whose sentence does not change (the others are written by the page)."""


class FitStepsView:
    """Builds ``self.step_rail`` and ``self.step_stack`` with one page per step."""

    def setup_steps_panel(self, parent: QWidget) -> QFrame:
        panel = QFrame(parent)
        panel.setObjectName("fitProcessPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.step_rail = StepRail(STEPS, panel)
        self.step_rail.setObjectName("fitStepRail")
        layout.addWidget(self.step_rail)
        divider = QFrame(panel)
        divider.setFrameShape(QFrame.HLine)
        divider.setProperty("gimapDivider", True)
        layout.addWidget(divider)
        self.step_stack = QStackedWidget(panel)
        self.step_stack.setObjectName("fitStepStack")
        self.step_pages: dict[str, QWidget] = {}
        self.step_intro: dict[str, QLabel] = {}
        for key, title in STEPS:
            scroll = QScrollArea(self.step_stack)
            scroll.setObjectName(f"fitStep_{key}")
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            content = QWidget(scroll)
            page_layout = QVBoxLayout(content)
            page_layout.setContentsMargins(4, 2, 6, 4)
            page_layout.setSpacing(8)
            heading = QLabel(title, content)
            heading.setProperty("gimapInspectorTitle", True)
            intro = QLabel(STEP_INTROS.get(key, ""), content)
            intro.setProperty("gimapInspectorIntro", True)
            intro.setWordWrap(True)
            intro.setTextInteractionFlags(Qt.TextSelectableByMouse)
            page_layout.addWidget(heading)
            page_layout.addWidget(intro)
            getattr(self, f"_{key}_step")(content, page_layout)
            page_layout.addStretch(1)
            scroll.setWidget(content)
            self.step_stack.addWidget(scroll)
            self.step_pages[key] = scroll
            self.step_intro[key] = intro
        layout.addWidget(self.step_stack, 1)
        self.step_rail.set_current("curve")
        return panel

    # -- 1 curve -----------------------------------------------------------------------

    def _curve_step(self, page: QWidget, layout: QVBoxLayout) -> None:
        self.curve_card, self.curve_info = info_card(page)
        layout.addWidget(self.curve_card)
        self.open_curve_step_button = QPushButton("Open Curve…", page)
        self.open_curve_step_button.setObjectName("fitOpenCurveStep")
        self.open_curve_step_button.setToolTip(
            "A 1D curve: columns q, I and optionally σ (.dat, .txt). Analyze ▸ Send to Fitting opens its cut here."
        )
        self.open_folder_step_button = QPushButton("Fit a Folder of Curves…", page)
        self.open_folder_step_button.setObjectName("fitOpenFolderStep")
        self.open_folder_step_button.setToolTip(
            "In-situ series: every curve of a folder (a series, or many samples), each fitted with the model of this "
            "page — fit one curve here first, or let each curve be predicted")
        opening = QHBoxLayout()
        opening.setSpacing(6)
        opening.addWidget(self.open_curve_step_button)
        opening.addWidget(self.open_folder_step_button)
        opening.addStretch(1)
        layout.addLayout(opening)

        self.side_label = QLabel("Halves of the cut", page)
        self.side_label.setProperty("gimapRole", "strong")
        self.side_combo = QComboBox(page)
        self.side_combo.setObjectName("fitSideCombo")
        self.side_combo.setToolTip(
            "A cut through the beam has q < 0 and q > 0: fit their mean, both on |q|, or one half"
        )
        layout.addWidget(self.side_label)
        layout.addWidget(self.side_combo)

        range_title = QLabel("Fitting range", page)
        range_title.setProperty("gimapRole", "strong")
        layout.addWidget(range_title)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.range_min_spin = ScientificDoubleSpinBox(page)
        self.range_min_spin.setObjectName("fitRangeMin")
        self.range_max_spin = ScientificDoubleSpinBox(page)
        self.range_max_spin.setObjectName("fitRangeMax")
        for spin in (self.range_min_spin, self.range_max_spin):
            spin.setRange(0.0, 1e6)
            spin.setDecimals(6)
            spin.setSuffix(" nm⁻¹")
            spin.setKeyboardTracking(False)
        row.addWidget(self.range_min_spin, 1)
        row.addWidget(QLabel("–", page))
        row.addWidget(self.range_max_spin, 1)
        layout.addLayout(row)
        self.whole_range_button = QPushButton("Whole Curve", page)
        self.whole_range_button.setObjectName("fitWholeRange")
        self.whole_range_button.setToolTip("Fit every point of the curve")
        layout.addWidget(self.whole_range_button, 0, Qt.AlignLeft)
        layout.addWidget(muted("Or drag the orange band on the plot. Only the points inside it are fitted.", page))
        self.excluded_row = QWidget(page)
        excluded = QHBoxLayout(self.excluded_row)
        excluded.setContentsMargins(0, 0, 0, 0)
        self.excluded_label = QLabel("", self.excluded_row)
        self.excluded_label.setObjectName("fitExcludedLabel")
        self.include_all_button = QPushButton("Include All", self.excluded_row)
        self.include_all_button.setObjectName("fitIncludeAll")
        self.include_all_button.setToolTip("Take every left-out point back into the fit")
        excluded.addWidget(self.excluded_label, 1)
        excluded.addWidget(self.include_all_button)
        self.excluded_row.hide()
        layout.addWidget(self.excluded_row)

        more = AdvancedSection("File", "", page)
        more.setObjectName("fitCurveMore")
        unit_row = QHBoxLayout()
        unit_row.addWidget(QLabel("q in the file", more))
        self.unit_combo = QComboBox(more)
        self.unit_combo.setObjectName("fitFileUnit")
        self.unit_combo.addItem("Å⁻¹", "angstrom")
        self.unit_combo.addItem("nm⁻¹", "nm")
        self.unit_combo.setToolTip("Curves from Analyze are in Å⁻¹; other files may be in nm⁻¹")
        unit_row.addWidget(self.unit_combo, 1)
        more.add_layout(unit_row)
        self.sigma_note = muted("", more)
        more.add_widget(self.sigma_note)
        layout.addWidget(more)

    # -- 2 model -----------------------------------------------------------------------

    def _model_step(self, page: QWidget, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.setSpacing(6)
        self.add_component_button = QToolButton(page)
        self.add_component_button.setObjectName("fitAddComponent")
        self.add_component_button.setText("Add Particle")  # the menu arrow is drawn by the button
        self.add_component_button.setPopupMode(QToolButton.InstantPopup)
        self.add_component_button.setToolTip("Add a particle family to the model")
        self.show_ranges_check = QCheckBox("Ranges", page)
        self.show_ranges_check.setObjectName("fitShowRanges")
        self.show_ranges_check.setToolTip(
            "Show the min–max range of every parameter: the fit keeps each value inside it, and "
            "“Search the ranges” searches across it"
        )
        row.addWidget(self.add_component_button)
        row.addStretch(1)
        row.addWidget(self.show_ranges_check)
        layout.addLayout(row)
        self.model_host = QVBoxLayout()
        self.model_host.setSpacing(8)
        layout.addLayout(self.model_host)
        layout.addWidget(muted(
            "Tick “fit” for the values the fit may change. Not sure which particle? Fit ▸ Find the "
            "particle shape tries every family.", page,
        ))

    # -- 3 fit and 4 results (``fit_step_view``, ``fit_results_view``) ------------------------------------

    def _fit_step(self, page: QWidget, layout: QVBoxLayout) -> None:
        build_fit_step(self, page, layout)

    def _results_step(self, page: QWidget, layout: QVBoxLayout) -> None:
        build_results_step(self, page, layout)


__all__ = ["FitStepsView", "METHODS", "QUICK_FAMILIES", "STEPS", "STEP_INTROS", "fit_to_rows", "info_card", "muted",
           "read_only_table"]
