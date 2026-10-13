"""The steps of fitting one curve and the controls of each (static layout; behaviour in ``single/``).

Curve → Model → Fit → Results, as in Analyze: each step page has a title, one sentence of where
it stands (``step_intro[key]``, filled by the page), the controls used most, and the rest folded.
"""

from __future__ import annotations

from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.components import AdvancedSection, FlowLayout, StepRail
from src.gimap.app.presentation.components.table_copy import enable_table_copy

from ..layout_primitives import ScientificDoubleSpinBox

STEPS = (("curve", "Curve"), ("model", "Model"), ("fit", "Fit"), ("results", "Results"))
STEP_INTROS = {"fit": "Choose a method; the button below runs it (Ctrl+Return)."}
"""The intro of a step whose sentence does not change (the others are written by the page)."""
METHODS = (
    ("local", "Refine the current values",
     "Least squares from the values in Model: fast; finds the nearest good fit."),
    ("global", "Search the ranges, then refine",
     "Tries values across the min–max range of every free parameter first; slower, for a poor start."),
    ("shapes", "Find the particle shape (no AI)",
     "Fits sphere, random cylinder and vertical cylinder from several starts and lists the solutions."),
    ("ai", "AI proposal (1D Predict)",
     "The V5 model proposes compositions and parameters, then corrects them numerically."),
)
QUICK_FAMILIES = (("Try every family", "auto"), ("The families in Model", "model"))


def muted(text: str, parent: QWidget) -> QLabel:
    label = QLabel(text, parent)
    label.setWordWrap(True)
    label.setProperty("gimapRole", "muted")
    return label


def info_card(parent: QWidget) -> tuple[QFrame, QLabel]:
    card = QFrame(parent)
    card.setProperty("gimapInfoCard", True)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(10, 8, 10, 8)
    label = QLabel("", card)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    layout.addWidget(label)
    return card, label


def read_only_table(columns, parent: QWidget, name: str,
                    selection=QAbstractItemView.SingleSelection) -> QTableWidget:
    """Rows selected whole; Ctrl+C and the right button copy them with the column names (``enable_table_copy``).
    ``SingleSelection`` where the current row drives something (“Use This Solution”), else ``ExtendedSelection``."""
    table = QTableWidget(0, len(columns), parent)
    table.setObjectName(name)
    table.setHorizontalHeaderLabels(list(columns))
    table.verticalHeader().hide()
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setSelectionMode(selection)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    enable_table_copy(table)
    return table


class _KeepToRows(QObject):
    """Sizes the table again once it is shown or restyled (its header's height is known only then)."""

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API
        if event.type() in (QEvent.Show, QEvent.StyleChange, QEvent.FontChange):
            fit_to_rows(watched)
        return False


def fit_to_rows(table: QTableWidget) -> None:
    """As tall as its header and rows, with no scroll bar of its own: the step's page scrolls instead.
    Call again after the rows change."""
    if not table.property("gimapFitToRows"):
        table.setProperty("gimapFitToRows", True)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        table._gimap_keep_to_rows = _KeepToRows(table)
        table.installEventFilter(table._gimap_keep_to_rows)
    header = table.horizontalHeader()
    top = header.height() if header.isVisible() and header.height() > 0 else header.sizeHint().height()
    scroll = table.horizontalScrollBar()
    bottom = scroll.sizeHint().height() if table.horizontalScrollBarPolicy() != Qt.ScrollBarAlwaysOff \
        and scroll.isVisible() else 0
    table.setFixedHeight(top + table.verticalHeader().length() + bottom + 2 * table.frameWidth())


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
        layout.addWidget(self.open_curve_step_button, 0, Qt.AlignLeft)

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

    # -- 3 fit -------------------------------------------------------------------------

    def _fit_step(self, page: QWidget, layout: QVBoxLayout) -> None:
        self.method_group = QButtonGroup(page)
        self.method_buttons: dict[str, QRadioButton] = {}
        for index, (key, title, description) in enumerate(METHODS):
            button = QRadioButton(title, page)
            button.setObjectName(f"fitMethod_{key}")
            button.setToolTip(description)
            self.method_group.addButton(button, index)
            self.method_buttons[key] = button
            layout.addWidget(button)
            text = muted(description, page)
            text.setContentsMargins(22, 0, 0, 2)
            layout.addWidget(text)
        self.method_buttons["local"].setChecked(True)
        self.families_row = QWidget(page)
        families = QHBoxLayout(self.families_row)
        families.setContentsMargins(22, 0, 0, 0)
        families.addWidget(QLabel("Families", self.families_row))
        self.families_combo = QComboBox(self.families_row)
        self.families_combo.setObjectName("fitShapeFamilies")
        for text, value in QUICK_FAMILIES:
            self.families_combo.addItem(text, value)
        families.addWidget(self.families_combo, 1)
        layout.addWidget(self.families_row)

        self.fit_step_button = QPushButton("Fit", page)
        self.fit_step_button.setObjectName("fitRunStep")
        self.fit_step_button.setProperty("gimapPrimaryAction", True)
        self.fit_step_button.setMinimumHeight(34)
        layout.addWidget(self.fit_step_button)
        self.fit_progress = QProgressBar(page)
        self.fit_progress.setObjectName("fitProgress")
        self.fit_progress.setRange(0, 100)
        self.fit_progress.hide()
        self.fit_progress_text = muted("", page)
        self.stop_step_button = QPushButton("Stop", page)
        self.stop_step_button.setObjectName("fitStopStep")
        self.stop_step_button.setProperty("gimapDangerAction", True)
        self.stop_step_button.setToolTip("Stop the fit; the best values so far are kept")
        self.stop_step_button.hide()
        layout.addWidget(self.fit_progress)
        layout.addWidget(self.fit_progress_text)
        layout.addWidget(self.stop_step_button, 0, Qt.AlignLeft)
        self.weighting_note = muted("", page)
        layout.addWidget(self.weighting_note)

        advanced = AdvancedSection("Advanced", "", page)
        advanced.setObjectName("fitAdvanced")
        row = QHBoxLayout()
        row.addWidget(QLabel("Evaluations at most", advanced))
        self.budget_spin = QSpinBox(advanced)
        self.budget_spin.setObjectName("fitBudget")
        self.budget_spin.setRange(0, 1_000_000)
        self.budget_spin.setSingleStep(500)
        self.budget_spin.setSpecialValueText("automatic")
        self.budget_spin.setToolTip("How many model evaluations a fit may use (automatic: by the method and free parameters)")
        row.addWidget(self.budget_spin, 1)
        advanced.add_layout(row)
        self.batch_button = QPushButton("Fit Many Curves…", advanced)
        self.batch_button.setObjectName("fitManyCurves")
        self.batch_button.setToolTip("1D Predict: a list of curve files fitted one after another, with their results")
        advanced.add_widget(self.batch_button)
        layout.addWidget(advanced)

    # -- 4 results ---------------------------------------------------------------------

    def _results_step(self, page: QWidget, layout: QVBoxLayout) -> None:
        self.quality_card, self.quality_label = info_card(page)
        self.quality_card.setObjectName("fitQualityCard")
        layout.addWidget(self.quality_card)
        self.warnings_label = QLabel("", page)
        self.warnings_label.setObjectName("fitWarnings")
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setProperty("gimapRole", "warning")
        layout.addWidget(self.warnings_label)
        self.parameters_table = read_only_table(("", "value", "±"), page, "fitParametersTable",
                                                QAbstractItemView.ExtendedSelection)
        header = self.parameters_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # a long name is cut short (its tooltip has it whole)
        self.parameters_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        fit_to_rows(self.parameters_table)  # every row shown: only the page scrolls
        layout.addWidget(self.parameters_table)

        self.solutions_title = QLabel("Solutions", page)
        self.solutions_title.setProperty("gimapRole", "strong")
        layout.addWidget(self.solutions_title)
        # One row at a time: “Use This Solution” takes the current one.
        self.solutions_table = read_only_table(("#", "model", "R (nm)", "D (nm)", "χ²"), page, "fitSolutionsTable")
        self.solutions_table.setMinimumHeight(120)
        layout.addWidget(self.solutions_table)
        self.use_solution_button = QPushButton("Use This Solution", page)
        self.use_solution_button.setObjectName("fitUseSolution")
        self.use_solution_button.setToolTip("Put the selected solution into Model (Undo brings the previous model back)")
        layout.addWidget(self.use_solution_button, 0, Qt.AlignLeft)

        exports = FlowLayout()
        self.export_data_step_button = QPushButton("Save Data and Fit…", page)
        self.export_data_step_button.setObjectName("fitExportData")
        self.export_data_step_button.setToolTip(
            "CSV: q, I, σ, the model, the residuals and each term; a JSON record of the model and the fit next to it"
        )
        self.export_plot_step_button = QPushButton("Save Plot…", page)
        self.export_plot_step_button.setObjectName("fitExportPlot")
        self.export_plot_step_button.setToolTip("The plot as shown: PNG (image) or SVG (vector)")
        self.save_model_button = QPushButton("Save Model…", page)
        self.save_model_button.setObjectName("fitSaveModel")
        self.save_model_button.setToolTip("The model (values, fit/fixed, ranges) as JSON, to load for another curve")
        for button in (self.export_data_step_button, self.export_plot_step_button, self.save_model_button):
            exports.addWidget(button)
        layout.addLayout(exports)

        log = AdvancedSection("Run Log", "", page)
        log.setObjectName("fitLogSection")
        self.log_view = QPlainTextEdit(log)
        self.log_view.setObjectName("fitLog")
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(140)
        log.add_widget(self.log_view)
        layout.addWidget(log)


__all__ = ["FitStepsView", "METHODS", "QUICK_FAMILIES", "STEPS", "STEP_INTROS", "fit_to_rows", "info_card", "muted",
           "read_only_table"]
