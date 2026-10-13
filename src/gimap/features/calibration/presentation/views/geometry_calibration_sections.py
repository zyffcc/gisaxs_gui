"""Semantic section builders for the calibration Python view."""

from PyQt5 import QtCore, QtWidgets


class GeometryCalibrationSections:
    """Build independently maintainable calibration layout sections."""

    @staticmethod
    def _compact_combo(combo: QtWidgets.QComboBox, characters: int = 12) -> None:
        """Size a combo by a short minimum, not by its longest item, so a long standard or
        detector name cannot widen the inputs column past its scroll area."""
        combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(characters)

    def _setup_manual_section(self, parent):
        """Manual refinement: a collapsed section in the inputs column (it never takes height
        from the preview). It opens when manual mode is switched on with 'Manual refine'."""
        self.calibration_manual_section = QtWidgets.QFrame(parent)
        self.calibration_manual_section.setProperty("gimapSection", True)
        self.calibration_manual_section.setObjectName("calibration_manual_section")
        self.calibrationManualSectionLayout = QtWidgets.QVBoxLayout(self.calibration_manual_section)
        self.calibrationManualSectionLayout.setContentsMargins(8, 8, 8, 8)
        self.calibrationManualSectionLayout.setSpacing(7)
        self.calibrationManualSectionLayout.setObjectName("calibrationManualSectionLayout")
        self.calibrationManualToggle = QtWidgets.QToolButton(self.calibration_manual_section)
        self.calibrationManualToggle.setProperty("gimapAdvancedToggle", True)
        self.calibrationManualToggle.setCheckable(True)
        self.calibrationManualToggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.calibrationManualToggle.setArrowType(QtCore.Qt.RightArrow)
        self.calibrationManualToggle.setObjectName("calibrationManualToggle")
        self.calibrationManualSectionLayout.addWidget(self.calibrationManualToggle)
        self.calibrationManualDescription = QtWidgets.QLabel(self.calibration_manual_section)
        self.calibrationManualDescription.setVisible(False)
        self.calibrationManualDescription.setProperty("gimapSectionDescription", True)
        self.calibrationManualDescription.setWordWrap(True)
        self.calibrationManualDescription.setObjectName("calibrationManualDescription")
        self.calibrationManualSectionLayout.addWidget(self.calibrationManualDescription)
        self.calibrationManualContent = QtWidgets.QWidget(self.calibration_manual_section)
        self.calibrationManualContent.setVisible(False)
        self.calibrationManualContent.setObjectName("calibrationManualContent")
        self.calibrationManualContentLayout = QtWidgets.QVBoxLayout(self.calibrationManualContent)
        self.calibrationManualContentLayout.setContentsMargins(4, 2, 4, 4)
        self.calibrationManualContentLayout.setSpacing(8)
        self.calibrationManualContentLayout.setObjectName("calibrationManualContentLayout")
        self.manual_group = QtWidgets.QGroupBox(self.calibrationManualContent)
        self.manual_group.setEnabled(False)
        self.manual_group.setMaximumSize(QtCore.QSize(16777215, 40))
        self.manual_group.setCheckable(True)
        self.manual_group.setChecked(False)
        self.manual_group.setObjectName("manual_group")
        self.manualGroupLayout = QtWidgets.QVBoxLayout(self.manual_group)
        self.manualGroupLayout.setObjectName("manualGroupLayout")
        self.manual_panel = QtWidgets.QWidget(self.manual_group)
        self.manual_panel.setVisible(False)
        self.manual_panel.setObjectName("manual_panel")
        self.manualGridLayout = QtWidgets.QGridLayout(self.manual_panel)
        self.manualGridLayout.setContentsMargins(4, 2, 4, 4)
        self.manualGridLayout.setColumnStretch(1, 1)
        self.manualGridLayout.setObjectName("manualGridLayout")
        self.manual_hint = QtWidgets.QLabel(self.manual_panel)
        self.manual_hint.setWordWrap(True)
        self.manual_hint.setObjectName("manual_hint")
        self.manualGridLayout.addWidget(self.manual_hint, 0, 0, 1, 2)
        self.manualXLabel = QtWidgets.QLabel(self.manual_panel)
        self.manualXLabel.setObjectName("manualXLabel")
        self.manualGridLayout.addWidget(self.manualXLabel, 1, 0, 1, 1)
        self.manual_x = QtWidgets.QDoubleSpinBox(self.manual_panel)
        self.manual_x.setDecimals(3)
        self.manual_x.setMinimum(-100000.0)
        self.manual_x.setMaximum(100000.0)
        self.manual_x.setKeyboardTracking(False)
        self.manual_x.setObjectName("manual_x")
        self.manualGridLayout.addWidget(self.manual_x, 1, 1, 1, 1)
        self.manualYLabel = QtWidgets.QLabel(self.manual_panel)
        self.manualYLabel.setObjectName("manualYLabel")
        self.manualGridLayout.addWidget(self.manualYLabel, 2, 0, 1, 1)
        self.manual_y = QtWidgets.QDoubleSpinBox(self.manual_panel)
        self.manual_y.setDecimals(3)
        self.manual_y.setMinimum(-100000.0)
        self.manual_y.setMaximum(100000.0)
        self.manual_y.setKeyboardTracking(False)
        self.manual_y.setObjectName("manual_y")
        self.manualGridLayout.addWidget(self.manual_y, 2, 1, 1, 1)
        self.manualDistanceLabel = QtWidgets.QLabel(self.manual_panel)
        self.manualDistanceLabel.setObjectName("manualDistanceLabel")
        self.manualGridLayout.addWidget(self.manualDistanceLabel, 3, 0, 1, 1)
        self.manual_distance = QtWidgets.QDoubleSpinBox(self.manual_panel)
        self.manual_distance.setDecimals(3)
        self.manual_distance.setMinimum(0.01)
        self.manual_distance.setMaximum(100000.0)
        self.manual_distance.setProperty("value", 1000.0)
        self.manual_distance.setKeyboardTracking(False)
        self.manual_distance.setObjectName("manual_distance")
        self.manualGridLayout.addWidget(self.manual_distance, 3, 1, 1, 1)
        self.detectedRingLabel = QtWidgets.QLabel(self.manual_panel)
        self.detectedRingLabel.setObjectName("detectedRingLabel")
        self.manualGridLayout.addWidget(self.detectedRingLabel, 4, 0, 1, 1)
        self.experimental_ring_combo = QtWidgets.QComboBox(self.manual_panel)
        self._compact_combo(self.experimental_ring_combo, 10)
        self.experimental_ring_combo.setObjectName("experimental_ring_combo")
        self.manualGridLayout.addWidget(self.experimental_ring_combo, 4, 1, 1, 1)
        self.theoryPeakLabel = QtWidgets.QLabel(self.manual_panel)
        self.theoryPeakLabel.setObjectName("theoryPeakLabel")
        self.manualGridLayout.addWidget(self.theoryPeakLabel, 5, 0, 1, 1)
        self.theory_ring_combo = QtWidgets.QComboBox(self.manual_panel)
        self._compact_combo(self.theory_ring_combo, 14)
        self.theory_ring_combo.setObjectName("theory_ring_combo")
        self.manualGridLayout.addWidget(self.theory_ring_combo, 5, 1, 1, 1)
        self.manualActionsLayout = QtWidgets.QHBoxLayout()
        self.manualActionsLayout.setObjectName("manualActionsLayout")
        self.refine_ring_button = QtWidgets.QPushButton(self.manual_panel)
        self.refine_ring_button.setObjectName("refine_ring_button")
        self.manualActionsLayout.addWidget(self.refine_ring_button)
        self.reset_manual_button = QtWidgets.QPushButton(self.manual_panel)
        self.reset_manual_button.setObjectName("reset_manual_button")
        self.manualActionsLayout.addWidget(self.reset_manual_button)
        self.manualActionsLayout.addStretch(1)
        self.manualGridLayout.addLayout(self.manualActionsLayout, 6, 0, 1, 2)
        self.manualGroupLayout.addWidget(self.manual_panel)
        self.calibrationManualContentLayout.addWidget(self.manual_group)
        self.calibrationManualSectionLayout.addWidget(self.calibrationManualContent)

    def _setup_results_section(self, splitter):
        """Selected solution and candidates. The solution rows sit in a two-column grid where
        the width allows; the grid scrolls only when even that does not fit, so the section
        can give height to the preview without its rows overlapping."""
        self.calibration_results_section = QtWidgets.QFrame(splitter)
        self.calibration_results_section.setProperty("gimapSection", True)
        self.calibration_results_section.setObjectName("calibration_results_section")
        self.calibrationResultsSectionLayout = QtWidgets.QVBoxLayout(
            self.calibration_results_section
        )
        self.calibrationResultsSectionLayout.setContentsMargins(12, 10, 12, 12)
        self.calibrationResultsSectionLayout.setSpacing(8)
        self.calibrationResultsSectionLayout.setObjectName("calibrationResultsSectionLayout")
        self.calibrationResultsTitle = QtWidgets.QLabel(self.calibration_results_section)
        self.calibrationResultsTitle.setProperty("gimapSectionTitle", True)
        self.calibrationResultsTitle.setObjectName("calibrationResultsTitle")
        self.calibrationResultsSectionLayout.addWidget(self.calibrationResultsTitle)
        self.calibrationResultsDescription = QtWidgets.QLabel(self.calibration_results_section)
        self.calibrationResultsDescription.setProperty("gimapSectionDescription", True)
        self.calibrationResultsDescription.setWordWrap(True)
        self.calibrationResultsDescription.setObjectName("calibrationResultsDescription")
        self.calibrationResultsSectionLayout.addWidget(self.calibrationResultsDescription)
        self.calibrationResultsContent = QtWidgets.QWidget(self.calibration_results_section)
        self.calibrationResultsContent.setObjectName("calibrationResultsContent")
        self.calibrationResultsContentLayout = QtWidgets.QVBoxLayout(self.calibrationResultsContent)
        self.calibrationResultsContentLayout.setContentsMargins(0, 0, 0, 0)
        self.calibrationResultsContentLayout.setSpacing(8)
        self.calibrationResultsContentLayout.setObjectName("calibrationResultsContentLayout")
        self.results_splitter = QtWidgets.QSplitter(self.calibrationResultsContent)
        self.results_splitter.setOrientation(QtCore.Qt.Horizontal)
        self.results_splitter.setChildrenCollapsible(False)
        self.results_splitter.setObjectName("results_splitter")
        self.result_group = QtWidgets.QGroupBox(self.results_splitter)
        self.result_group.setMinimumSize(QtCore.QSize(240, 0))
        self.result_group.setObjectName("result_group")
        self.resultGroupLayout = QtWidgets.QVBoxLayout(self.result_group)
        self.resultGroupLayout.setContentsMargins(6, 4, 2, 4)
        self.resultGroupLayout.setObjectName("resultGroupLayout")
        self.result_scroll = QtWidgets.QScrollArea(self.result_group)
        self.result_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.result_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.result_scroll.setWidgetResizable(True)
        self.result_scroll.setObjectName("result_scroll")
        self.result_form_widget = QtWidgets.QWidget()
        self.result_form_widget.setObjectName("result_form_widget")
        # A grid: two title/value pairs per row while the section is wide enough, so every row
        # (Confidence and Warning included) shows without scrolling; one pair per row otherwise.
        self.resultForm = QtWidgets.QGridLayout(self.result_form_widget)
        self.resultForm.setContentsMargins(0, 0, 6, 0)
        self.resultForm.setHorizontalSpacing(10)
        self.resultForm.setVerticalSpacing(4)
        self.resultForm.setObjectName("resultForm")
        self.result_rows = []
        for title_name, value_name in (
            ("resultCenterXTitle", "result_center_x"),
            ("resultCenterYTitle", "result_center_y"),
            ("resultDistanceTitle", "result_distance"),
            ("resultRotationTitle", "result_rotation"),
            ("resultRingsTitle", "result_rings"),
            ("resultRmsTitle", "result_rms"),
            ("resultConfidenceTitle", "result_confidence"),
            ("resultWarningTitle", "result_warning"),
        ):
            title = QtWidgets.QLabel(self.result_form_widget)
            title.setObjectName(title_name)
            value = QtWidgets.QLabel(self.result_form_widget)
            value.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            value.setObjectName(value_name)
            setattr(self, title_name, title)
            setattr(self, value_name, value)
            self.result_rows.append((title, value))
        self.result_warning.setWordWrap(True)
        self.place_result_rows(two_columns=True)
        self.result_scroll.setWidget(self.result_form_widget)
        self.resultGroupLayout.addWidget(self.result_scroll)
        self.candidates_group = QtWidgets.QGroupBox(self.results_splitter)
        self.candidates_group.setObjectName("candidates_group")
        self.candidatesLayout = QtWidgets.QVBoxLayout(self.candidates_group)
        self.candidatesLayout.setObjectName("candidatesLayout")
        self.candidate_table = QtWidgets.QTableWidget(self.candidates_group)
        self.candidate_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.candidate_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.candidate_table.setObjectName("candidate_table")
        self.candidate_table.setColumnCount(6)
        self.candidate_table.setRowCount(0)
        for column in range(6):
            self.candidate_table.setHorizontalHeaderItem(column, QtWidgets.QTableWidgetItem())
        self.candidatesLayout.addWidget(self.candidate_table)
        self.calibrationResultsContentLayout.addWidget(self.results_splitter)
        self.calibrationResultsSectionLayout.addWidget(self.calibrationResultsContent, 1)

    def place_result_rows(self, two_columns: bool) -> None:
        """Lay out the selected-solution rows.

        Two columns: Center X | Center Y, Distance | Rotation, Rings | RMS, then Confidence and
        Warning across the full width. One column: the eight rows under each other.
        """
        grid = self.resultForm
        for title, value in self.result_rows:
            grid.removeWidget(title)
            grid.removeWidget(value)
        paired = 6 if two_columns else 0  # the first six rows share a line in two columns
        for index, (title, value) in enumerate(self.result_rows):
            if index < paired:
                row, column, span = index // 2, 2 * (index % 2), 1
            else:
                row, column, span = index - paired // 2, 0, 3
            # Top-aligned only: the cells keep their full width, so the warning wraps in it.
            grid.addWidget(title, row, column, 1, 1, QtCore.Qt.AlignTop)
            grid.addWidget(value, row, column + 1, 1, span, QtCore.Qt.AlignTop)
        grid.setColumnStretch(1, 1 if two_columns else 0)
        grid.setColumnStretch(3, 1)
        grid.setRowStretch(8, 1)  # below every row in both arrangements: rows stay at the top

    def _setup_export_footer(self, GeometryCalibrationDialog):
        """One row of file and dialog actions under both panes (no titled card)."""
        self.calibration_export_section = QtWidgets.QFrame(GeometryCalibrationDialog)
        self.calibration_export_section.setObjectName("calibration_export_section")
        self.calibrationExportSectionLayout = QtWidgets.QVBoxLayout(self.calibration_export_section)
        self.calibrationExportSectionLayout.setContentsMargins(0, 0, 0, 0)
        self.calibrationExportSectionLayout.setSpacing(0)
        self.calibrationExportSectionLayout.setObjectName("calibrationExportSectionLayout")
        # Kept (hidden) for the section contract: title_label / description_label.
        self.calibrationExportTitle = QtWidgets.QLabel(self.calibration_export_section)
        self.calibrationExportTitle.setVisible(False)
        self.calibrationExportTitle.setObjectName("calibrationExportTitle")
        self.calibrationExportSectionLayout.addWidget(self.calibrationExportTitle)
        self.calibrationExportDescription = QtWidgets.QLabel(self.calibration_export_section)
        self.calibrationExportDescription.setVisible(False)
        self.calibrationExportDescription.setObjectName("calibrationExportDescription")
        self.calibrationExportSectionLayout.addWidget(self.calibrationExportDescription)
        self.calibrationExportContent = QtWidgets.QWidget(self.calibration_export_section)
        self.calibrationExportContent.setObjectName("calibrationExportContent")
        self.calibrationExportContentLayout = QtWidgets.QHBoxLayout(self.calibrationExportContent)
        self.calibrationExportContentLayout.setContentsMargins(0, 0, 0, 0)
        self.calibrationExportContentLayout.setObjectName("calibrationExportContentLayout")
        self.import_button = QtWidgets.QPushButton(self.calibrationExportContent)
        self.import_button.setObjectName("import_button")
        self.calibrationExportContentLayout.addWidget(self.import_button)
        self.export_button = QtWidgets.QPushButton(self.calibrationExportContent)
        self.export_button.setObjectName("export_button")
        self.calibrationExportContentLayout.addWidget(self.export_button)
        self.calibrationExportContentLayout.addStretch(1)
        self.apply_button = QtWidgets.QPushButton(self.calibrationExportContent)
        self.apply_button.setObjectName("apply_button")
        self.calibrationExportContentLayout.addWidget(self.apply_button)
        self.close_button = QtWidgets.QPushButton(self.calibrationExportContent)
        self.close_button.setObjectName("close_button")
        self.calibrationExportContentLayout.addWidget(self.close_button)
        self.calibrationExportSectionLayout.addWidget(self.calibrationExportContent)
