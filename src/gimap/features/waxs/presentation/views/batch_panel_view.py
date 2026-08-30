"""Hand-maintained WAXS batch panel layout."""

from PyQt5 import QtCore, QtWidgets


def _section_label(parent, object_name: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(parent)
    label.setObjectName(object_name)
    label.setProperty("gimapSectionTitle", True)
    return label


def _q_spin(parent, object_name: str) -> QtWidgets.QDoubleSpinBox:
    spin = QtWidgets.QDoubleSpinBox(parent)
    spin.setObjectName(object_name)
    spin.setDecimals(4)
    spin.setRange(-100.0, 100.0)
    spin.setSingleStep(0.05)
    return spin


class WaxsBatchPanelView:
    def setupUi(self, panel):
        panel.setObjectName("waxsBatchPanel")
        self.batchPanelLayout = QtWidgets.QVBoxLayout(panel)
        self.batchPanelLayout.setContentsMargins(4, 4, 4, 4)
        self.batchPanelLayout.setSpacing(10)

        self.batchSourcesTitle = _section_label(panel, "batchSourcesTitle")
        self.batchPanelLayout.addWidget(self.batchSourcesTitle)
        self.batchSourcesHint = QtWidgets.QLabel(panel)
        self.batchSourcesHint.setWordWrap(True)
        self.batchSourcesHint.setObjectName("batchSourcesHint")
        self.batchPanelLayout.addWidget(self.batchSourcesHint)

        self.batch_sources_table = QtWidgets.QTableWidget(0, 3, panel)
        self.batch_sources_table.setObjectName("batch_sources_table")
        self.batch_sources_table.setHorizontalHeaderLabels(
            ["Input folder", "Pattern", "Output subfolder"]
        )
        self.batch_sources_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )
        self.batch_sources_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection
        )
        self.batch_sources_table.setAlternatingRowColors(True)
        self.batch_sources_table.verticalHeader().setVisible(False)
        header = self.batch_sources_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.batch_sources_table.setMinimumHeight(150)
        self.batchPanelLayout.addWidget(self.batch_sources_table)

        source_actions = QtWidgets.QHBoxLayout()
        self.batch_add_folder_button = QtWidgets.QPushButton(panel)
        self.batch_add_folder_button.setObjectName("batch_add_folder_button")
        source_actions.addWidget(self.batch_add_folder_button)
        self.batch_remove_folder_button = QtWidgets.QPushButton(panel)
        self.batch_remove_folder_button.setObjectName("batch_remove_folder_button")
        source_actions.addWidget(self.batch_remove_folder_button)
        source_actions.addStretch(1)
        self.batchDefaultPatternLabel = QtWidgets.QLabel(panel)
        source_actions.addWidget(self.batchDefaultPatternLabel)
        self.batch_pattern_edit = QtWidgets.QLineEdit(panel)
        self.batch_pattern_edit.setObjectName("batch_pattern_edit")
        self.batch_pattern_edit.setMaximumWidth(105)
        source_actions.addWidget(self.batch_pattern_edit)
        self.batchPanelLayout.addLayout(source_actions)

        output_layout = QtWidgets.QGridLayout()
        self.batchOutputFolderLabel = QtWidgets.QLabel(panel)
        output_layout.addWidget(self.batchOutputFolderLabel, 0, 0)
        self.batch_output_edit = QtWidgets.QLineEdit(panel)
        self.batch_output_edit.setObjectName("batch_output_edit")
        output_layout.addWidget(self.batch_output_edit, 0, 1)
        self.batch_output_browse_button = QtWidgets.QPushButton(panel)
        self.batch_output_browse_button.setObjectName("batch_output_browse_button")
        output_layout.addWidget(self.batch_output_browse_button, 0, 2)
        self.batchPanelLayout.addLayout(output_layout)

        self.batchExportsTitle = _section_label(panel, "batchExportsTitle")
        self.batchPanelLayout.addWidget(self.batchExportsTitle)
        export_grid = QtWidgets.QGridLayout()
        self.batch_export_pixel_images = QtWidgets.QCheckBox(panel)
        self.batch_export_pixel_images.setObjectName("batch_export_pixel_images")
        export_grid.addWidget(self.batch_export_pixel_images, 0, 0)
        self.batch_export_q_images = QtWidgets.QCheckBox(panel)
        self.batch_export_q_images.setObjectName("batch_export_q_images")
        self.batch_export_q_images.setChecked(True)
        export_grid.addWidget(self.batch_export_q_images, 0, 1)
        self.batch_export_curves = QtWidgets.QCheckBox(panel)
        self.batch_export_curves.setObjectName("batch_export_curves")
        self.batch_export_curves.setChecked(True)
        export_grid.addWidget(self.batch_export_curves, 1, 0)
        self.batch_export_curve_images = QtWidgets.QCheckBox(panel)
        self.batch_export_curve_images.setObjectName("batch_export_curve_images")
        export_grid.addWidget(self.batch_export_curve_images, 1, 1)
        self.batchPanelLayout.addLayout(export_grid)

        self.batchAppearanceTitle = _section_label(panel, "batchAppearanceTitle")
        self.batchPanelLayout.addWidget(self.batchAppearanceTitle)
        appearance = QtWidgets.QFormLayout()
        appearance.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.batch_export_cmap = QtWidgets.QComboBox(panel)
        self.batch_export_cmap.setObjectName("batch_export_cmap")
        self.batch_export_cmap.addItems(
            ["viridis", "magma", "inferno", "plasma", "cividis", "turbo", "gray"]
        )
        appearance.addRow("Colormap", self.batch_export_cmap)
        scale_row = QtWidgets.QWidget(panel)
        scale_layout = QtWidgets.QHBoxLayout(scale_row)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        self.batch_export_log = QtWidgets.QCheckBox(scale_row)
        self.batch_export_log.setObjectName("batch_export_log")
        scale_layout.addWidget(self.batch_export_log)
        self.batch_export_auto_scale = QtWidgets.QCheckBox(scale_row)
        self.batch_export_auto_scale.setObjectName("batch_export_auto_scale")
        self.batch_export_auto_scale.setChecked(True)
        scale_layout.addWidget(self.batch_export_auto_scale)
        scale_layout.addStretch(1)
        appearance.addRow("Intensity", scale_row)
        limits_row = QtWidgets.QWidget(panel)
        limits_layout = QtWidgets.QHBoxLayout(limits_row)
        limits_layout.setContentsMargins(0, 0, 0, 0)
        self.batch_export_vmin = _q_spin(limits_row, "batch_export_vmin")
        self.batch_export_vmin.setRange(-1e12, 1e12)
        limits_layout.addWidget(self.batch_export_vmin)
        self.batch_export_vmax = _q_spin(limits_row, "batch_export_vmax")
        self.batch_export_vmax.setRange(-1e12, 1e12)
        self.batch_export_vmax.setValue(800.0)
        limits_layout.addWidget(self.batch_export_vmax)
        appearance.addRow("Vmin / Vmax", limits_row)
        self.batchPanelLayout.addLayout(appearance)
        appearance_actions = QtWidgets.QHBoxLayout()
        self.batch_preview_style_button = QtWidgets.QPushButton(panel)
        self.batch_preview_style_button.setObjectName("batch_preview_style_button")
        appearance_actions.addWidget(self.batch_preview_style_button)
        self.batch_copy_preview_button = QtWidgets.QPushButton(panel)
        self.batch_copy_preview_button.setObjectName("batch_copy_preview_button")
        appearance_actions.addWidget(self.batch_copy_preview_button)
        self.batchPanelLayout.addLayout(appearance_actions)

        self.batchGeometryTitle = _section_label(panel, "batchGeometryTitle")
        self.batchPanelLayout.addWidget(self.batchGeometryTitle)
        self.batch_geometry_summary = QtWidgets.QLabel(panel)
        self.batch_geometry_summary.setObjectName("batch_geometry_summary")
        self.batch_geometry_summary.setWordWrap(True)
        self.batch_geometry_summary.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        self.batchPanelLayout.addWidget(self.batch_geometry_summary)
        self.batch_limit_q_range = QtWidgets.QCheckBox(panel)
        self.batch_limit_q_range.setObjectName("batch_limit_q_range")
        self.batchPanelLayout.addWidget(self.batch_limit_q_range)
        q_grid = QtWidgets.QGridLayout()
        q_grid.addWidget(QtWidgets.QLabel("qr min / max", panel), 0, 0)
        self.batch_qr_min = _q_spin(panel, "batch_qr_min")
        q_grid.addWidget(self.batch_qr_min, 0, 1)
        self.batch_qr_max = _q_spin(panel, "batch_qr_max")
        self.batch_qr_max.setValue(3.0)
        q_grid.addWidget(self.batch_qr_max, 0, 2)
        q_grid.addWidget(QtWidgets.QLabel("qz min / max", panel), 1, 0)
        self.batch_qz_min = _q_spin(panel, "batch_qz_min")
        q_grid.addWidget(self.batch_qz_min, 1, 1)
        self.batch_qz_max = _q_spin(panel, "batch_qz_max")
        self.batch_qz_max.setValue(3.0)
        q_grid.addWidget(self.batch_qz_max, 1, 2)
        self.batchPanelLayout.addLayout(q_grid)

        self.batchButtonsLayout = QtWidgets.QHBoxLayout()
        self.batch_start_button = QtWidgets.QPushButton(panel)
        self.batch_start_button.setObjectName("batch_start_button")
        self.batchButtonsLayout.addWidget(self.batch_start_button)
        self.batch_pause_button = QtWidgets.QPushButton(panel)
        self.batch_pause_button.setEnabled(False)
        self.batch_pause_button.setObjectName("batch_pause_button")
        self.batchButtonsLayout.addWidget(self.batch_pause_button)
        self.batch_stop_button = QtWidgets.QPushButton(panel)
        self.batch_stop_button.setEnabled(False)
        self.batch_stop_button.setObjectName("batch_stop_button")
        self.batchButtonsLayout.addWidget(self.batch_stop_button)
        self.batchPanelLayout.addLayout(self.batchButtonsLayout)
        self.batchPanelLayout.addStretch(1)

        self.retranslateUi(panel)
        QtCore.QMetaObject.connectSlotsByName(panel)

    def retranslateUi(self, panel):
        _ = QtCore.QCoreApplication.translate
        self.batchSourcesTitle.setText(_("WaxsBatchPanel", "Data sources"))
        self.batchSourcesHint.setText(
            _(
                "WaxsBatchPanel",
                "Add one row per experiment folder. Each row can use its own "
                "file pattern and output subfolder.",
            )
        )
        self.batch_add_folder_button.setText(_("WaxsBatchPanel", "Add folder..."))
        self.batch_remove_folder_button.setText(_("WaxsBatchPanel", "Remove"))
        self.batchDefaultPatternLabel.setText(_("WaxsBatchPanel", "New row pattern"))
        self.batch_pattern_edit.setText(_("WaxsBatchPanel", "*.tif"))
        self.batch_pattern_edit.setToolTip(
            _("WaxsBatchPanel", "Examples: *.tif, *.tiff, *.nxs, *_m*.nxs")
        )
        self.batchOutputFolderLabel.setText(_("WaxsBatchPanel", "Export root"))
        self.batch_output_browse_button.setText(_("WaxsBatchPanel", "Browse"))
        self.batchExportsTitle.setText(_("WaxsBatchPanel", "Outputs"))
        self.batch_export_pixel_images.setText(
            _("WaxsBatchPanel", "2D image · pixel axes")
        )
        self.batch_export_q_images.setText(_("WaxsBatchPanel", "2D image · q axes"))
        self.batch_export_curves.setText(_("WaxsBatchPanel", "1D data · CSV"))
        self.batch_export_curve_images.setText(_("WaxsBatchPanel", "1D figure · PNG"))
        self.batchAppearanceTitle.setText(
            _("WaxsBatchPanel", "Publication appearance")
        )
        self.batch_export_log.setText(_("WaxsBatchPanel", "Log10"))
        self.batch_export_auto_scale.setText(_("WaxsBatchPanel", "Auto limits"))
        self.batch_preview_style_button.setText(
            _("WaxsBatchPanel", "Preview export style")
        )
        self.batch_copy_preview_button.setText(
            _("WaxsBatchPanel", "Use current preview")
        )
        self.batchGeometryTitle.setText(_("WaxsBatchPanel", "q conversion"))
        self.batch_limit_q_range.setText(
            _("WaxsBatchPanel", "Limit the exported q view")
        )
        self.batch_start_button.setText(_("WaxsBatchPanel", "Start"))
        self.batch_pause_button.setText(_("WaxsBatchPanel", "Pause"))
        self.batch_stop_button.setText(_("WaxsBatchPanel", "Stop"))
