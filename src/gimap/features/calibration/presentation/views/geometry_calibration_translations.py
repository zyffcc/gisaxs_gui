"""User-visible translations for the Python-owned Qt view."""

from PyQt5 import QtCore


class GeometryCalibrationTranslations:
    def retranslateUi(self, GeometryCalibrationDialog):
        _translate = QtCore.QCoreApplication.translate
        GeometryCalibrationDialog.setWindowTitle(
            _translate("GeometryCalibrationDialog", "Geometry Calibration")
        )
        self.calibrationInputTitle.setText(_translate("GeometryCalibrationDialog", "Input"))
        self.calibrationInputDescription.setText(
            _translate(
                "GeometryCalibrationDialog",
                "Choose a calibration image and the essential experiment metadata.",
            )
        )
        self.calibration_file_group.setTitle(
            _translate("GeometryCalibrationDialog", "Calibration image")
        )
        self.path_edit.setPlaceholderText(
            _translate("GeometryCalibrationDialog", "Paste a .cbf/.nxs path or use Open...")
        )
        self.open_button.setText(_translate("GeometryCalibrationDialog", "Open..."))
        self.calibration_input_group.setTitle(_translate("GeometryCalibrationDialog", "Input"))
        self.energyLabel.setText(_translate("GeometryCalibrationDialog", "Energy:"))
        self.energy_spin.setSuffix(_translate("GeometryCalibrationDialog", " keV"))
        self.standardLabel.setText(_translate("GeometryCalibrationDialog", "Standard:"))
        self.estimatedDistanceLabel.setText(
            _translate("GeometryCalibrationDialog", "Estimated distance:")
        )
        self.estimated_distance_spin.setSuffix(_translate("GeometryCalibrationDialog", " mm"))
        self.estimated_distance_spin.setSpecialValueText(
            _translate("GeometryCalibrationDialog", "Optional")
        )
        self.rangeLabel.setText(_translate("GeometryCalibrationDialog", "Distance range:"))
        self.range_combo.setItemText(
            0, _translate("GeometryCalibrationDialog", "Auto (30-10000 mm)")
        )
        self.range_combo.setItemText(
            1, _translate("GeometryCalibrationDialog", "SAXS (500-10000 mm)")
        )
        self.range_combo.setItemText(
            2, _translate("GeometryCalibrationDialog", "WAXS (30-1500 mm)")
        )
        self.range_combo.setItemText(3, _translate("GeometryCalibrationDialog", "Custom"))
        self.pixelSizeLabel.setText(_translate("GeometryCalibrationDialog", "Pixel size:"))
        self.pixel_label.setText(_translate("GeometryCalibrationDialog", "Open an image"))
        self.detectorLabel.setText(_translate("GeometryCalibrationDialog", "Detector:"))
        self.detector_label.setText(_translate("GeometryCalibrationDialog", "Open an image"))
        self.detectorModelLabel.setText(_translate("GeometryCalibrationDialog", "Detector model:"))
        self.calibrationAdvancedToggle.setText(
            _translate("GeometryCalibrationDialog", "Advanced configuration")
        )
        self.calibrationAdvancedDescription.setText(
            _translate(
                "GeometryCalibrationDialog",
                "Detector overrides, custom search bounds and preview overlays.",
            )
        )
        self.advanced_group.setTitle(
            _translate("GeometryCalibrationDialog", "Detector and display overrides")
        )
        self.pixelXLabel.setText(_translate("GeometryCalibrationDialog", "Pixel X:"))
        self.pixel_x_spin.setSuffix(_translate("GeometryCalibrationDialog", " µm"))
        self.pixelYLabel.setText(_translate("GeometryCalibrationDialog", "Pixel Y:"))
        self.pixel_y_spin.setSuffix(_translate("GeometryCalibrationDialog", " µm"))
        self.customMinLabel.setText(_translate("GeometryCalibrationDialog", "Custom minimum:"))
        self.custom_min_spin.setSuffix(_translate("GeometryCalibrationDialog", " mm"))
        self.customMaxLabel.setText(_translate("GeometryCalibrationDialog", "Custom maximum:"))
        self.custom_max_spin.setSuffix(_translate("GeometryCalibrationDialog", " mm"))
        self.background_check.setText(
            _translate("GeometryCalibrationDialog", "Subtract slowly varying background")
        )
        self.log_check.setText(_translate("GeometryCalibrationDialog", "Log intensity"))
        self.mask_check.setText(_translate("GeometryCalibrationDialog", "Show invalid-pixel mask"))
        self.rings_check.setText(_translate("GeometryCalibrationDialog", "Show ring overlays"))
        self.calibrationRunTitle.setText(_translate("GeometryCalibrationDialog", "Run"))
        self.calibrationRunDescription.setText(
            _translate(
                "GeometryCalibrationDialog",
                "Run automatic calibration, monitor the current stage, or cancel safely.",
            )
        )
        self.calibrate_button.setText(_translate("GeometryCalibrationDialog", "Auto Calibration"))
        self.cancel_button.setText(_translate("GeometryCalibrationDialog", "Cancel"))
        self.calibrationPreviewTitle.setText(_translate("GeometryCalibrationDialog", "Preview"))
        self.calibrationPreviewDescription.setText(
            _translate(
                "GeometryCalibrationDialog",
                "Inspect the detector image, detected rings and candidate overlays.",
            )
        )
        self.fit_image_button.setText(_translate("GeometryCalibrationDialog", "Reset view"))
        self.clean_preview_button.setText(_translate("GeometryCalibrationDialog", "Clean image"))
        self.expand_preview_button.setText(_translate("GeometryCalibrationDialog", "Focus image"))
        self.manual_refine_button.setText(_translate("GeometryCalibrationDialog", "Manual refine"))
        self.preview_info_label.setText(
            _translate("GeometryCalibrationDialog", "Open a calibration image to begin")
        )
        self.calibrationResultsTitle.setText(_translate("GeometryCalibrationDialog", "Results"))
        self.calibrationResultsDescription.setText(
            _translate(
                "GeometryCalibrationDialog",
                "Review the selected solution and alternative candidates before applying it.",
            )
        )
        self.result_group.setTitle(_translate("GeometryCalibrationDialog", "Results"))
        self.resultCenterXTitle.setText(_translate("GeometryCalibrationDialog", "Beam center X:"))
        self.result_center_x.setText(_translate("GeometryCalibrationDialog", "—"))
        self.resultCenterYTitle.setText(_translate("GeometryCalibrationDialog", "Beam center Y:"))
        self.result_center_y.setText(_translate("GeometryCalibrationDialog", "—"))
        self.resultDistanceTitle.setText(_translate("GeometryCalibrationDialog", "Distance:"))
        self.result_distance.setText(_translate("GeometryCalibrationDialog", "—"))
        self.resultRotationTitle.setText(
            _translate("GeometryCalibrationDialog", "Detector rotation:")
        )
        self.result_rotation.setText(_translate("GeometryCalibrationDialog", "—"))
        self.resultRingsTitle.setText(_translate("GeometryCalibrationDialog", "Matched rings:"))
        self.result_rings.setText(_translate("GeometryCalibrationDialog", "—"))
        self.resultRmsTitle.setText(_translate("GeometryCalibrationDialog", "RMS residual:"))
        self.result_rms.setText(_translate("GeometryCalibrationDialog", "—"))
        self.resultConfidenceTitle.setText(_translate("GeometryCalibrationDialog", "Confidence:"))
        self.result_confidence.setText(_translate("GeometryCalibrationDialog", "—"))
        self.resultWarningTitle.setText(_translate("GeometryCalibrationDialog", "Warning:"))
        self.result_warning.setText(_translate("GeometryCalibrationDialog", "—"))
        self.candidates_group.setTitle(
            _translate("GeometryCalibrationDialog", "Candidate solutions")
        )
        item = self.candidate_table.horizontalHeaderItem(0)
        item.setText(_translate("GeometryCalibrationDialog", "Standard"))
        item = self.candidate_table.horizontalHeaderItem(1)
        item.setText(_translate("GeometryCalibrationDialog", "Distance"))
        item = self.candidate_table.horizontalHeaderItem(2)
        item.setText(_translate("GeometryCalibrationDialog", "Center"))
        item = self.candidate_table.horizontalHeaderItem(3)
        item.setText(_translate("GeometryCalibrationDialog", "Rings"))
        item = self.candidate_table.horizontalHeaderItem(4)
        item.setText(_translate("GeometryCalibrationDialog", "RMS"))
        item = self.candidate_table.horizontalHeaderItem(5)
        item.setText(_translate("GeometryCalibrationDialog", "Confidence"))
        self.calibrationManualToggle.setText(
            _translate("GeometryCalibrationDialog", "Advanced manual refinement")
        )
        self.calibrationManualDescription.setText(
            _translate(
                "GeometryCalibrationDialog",
                "Fine-tune the selected center, distance or ring correspondence.",
            )
        )
        self.manual_group.setTitle(
            _translate(
                "GeometryCalibrationDialog",
                "Manual refinement · drag the center marker or edit values",
            )
        )
        self.manual_hint.setText(
            _translate(
                "GeometryCalibrationDialog",
                "Fine-tune only when the overlay needs correction. Changes are previewed immediately.",
            )
        )
        self.manualXLabel.setText(_translate("GeometryCalibrationDialog", "Center X:"))
        self.manualYLabel.setText(_translate("GeometryCalibrationDialog", "Center Y:"))
        self.manualDistanceLabel.setText(_translate("GeometryCalibrationDialog", "Distance (mm):"))
        self.detectedRingLabel.setText(_translate("GeometryCalibrationDialog", "Detected ring:"))
        self.theoryPeakLabel.setText(_translate("GeometryCalibrationDialog", "Theoretical peak:"))
        self.refine_ring_button.setText(
            _translate("GeometryCalibrationDialog", "Fit selected ring")
        )
        self.calibrationExportTitle.setText(_translate("GeometryCalibrationDialog", "Export"))
        self.calibrationExportDescription.setText(
            _translate(
                "GeometryCalibrationDialog",
                "Import or save a calibration, or apply the selected result to the project.",
            )
        )
        self.import_button.setText(_translate("GeometryCalibrationDialog", "Import Calibration..."))
        self.export_button.setText(_translate("GeometryCalibrationDialog", "Export Calibration..."))
        self.apply_button.setText(_translate("GeometryCalibrationDialog", "Apply"))
        self.close_button.setText(_translate("GeometryCalibrationDialog", "Close"))
