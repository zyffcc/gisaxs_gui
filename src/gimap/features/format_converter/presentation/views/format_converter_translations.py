"""User-visible translations for the Python-owned Qt view."""

from PyQt5 import QtCore


class FormatConverterTranslations:
    def retranslateUi(self, FormatConverterDialog):
        _translate = QtCore.QCoreApplication.translate
        FormatConverterDialog.setWindowTitle(
            _translate("FormatConverterDialog", "Format Converter")
        )
        self.step_input_label.setText(_translate("FormatConverterDialog", "1  Input"))
        self.step_configure_label.setText(
            _translate("FormatConverterDialog", "2  Configure & Preview")
        )
        self.step_output_label.setText(
            _translate("FormatConverterDialog", "3  Run, Results & Export")
        )
        self.formatInputTitle.setText(_translate("FormatConverterDialog", "Input"))
        self.formatInputDescription.setText(
            _translate(
                "FormatConverterDialog",
                "Add detector files or folders. Sources are inspected without changing them.",
            )
        )
        self.input_intro.setText(
            _translate(
                "FormatConverterDialog",
                "Add any combination of NXS, CBF, and TIFF files. GIMaP detects single images and multi-frame inputs automatically.",
            )
        )
        self.add_files_button.setText(_translate("FormatConverterDialog", "Add files"))
        self.add_folder_button.setText(_translate("FormatConverterDialog", "Add folder"))
        self.current_button.setText(
            _translate("FormatConverterDialog", "Use currently opened file")
        )
        self.input_tree.headerItem().setText(0, _translate("FormatConverterDialog", "Source"))
        self.input_tree.headerItem().setText(1, _translate("FormatConverterDialog", "Type"))
        self.input_tree.headerItem().setText(
            2, _translate("FormatConverterDialog", "Images / frames")
        )
        self.input_tree.headerItem().setText(3, _translate("FormatConverterDialog", "Selection"))
        self.input_tree.headerItem().setText(4, _translate("FormatConverterDialog", "Status"))
        self.dataset_label.setText(_translate("FormatConverterDialog", "Dataset:"))
        self.dataset_note.setText(
            _translate("FormatConverterDialog", "Recommended dataset is selected automatically.")
        )
        self.input_note.setText(_translate("FormatConverterDialog", "No input files yet."))
        self.formatConfigureTitle.setText(_translate("FormatConverterDialog", "Configure"))
        self.formatConfigureDescription.setText(
            _translate(
                "FormatConverterDialog",
                "Choose sources and frame ranges. Existing selection semantics are preserved.",
            )
        )
        self.select_all_button.setText(_translate("FormatConverterDialog", "Select all"))
        self.select_none_button.setText(_translate("FormatConverterDialog", "Select none"))
        self.remove_selected_button.setText(_translate("FormatConverterDialog", "Remove selected"))
        self.sort_button.setText(_translate("FormatConverterDialog", "Sort by filename"))
        self.filter_edit.setPlaceholderText(
            _translate("FormatConverterDialog", "Filter by filename…")
        )
        item = self.selection_table.horizontalHeaderItem(0)
        item.setText(_translate("FormatConverterDialog", "Use"))
        item = self.selection_table.horizontalHeaderItem(1)
        item.setText(_translate("FormatConverterDialog", "Source"))
        item = self.selection_table.horizontalHeaderItem(2)
        item.setText(_translate("FormatConverterDialog", "Type"))
        item = self.selection_table.horizontalHeaderItem(3)
        item.setText(_translate("FormatConverterDialog", "Images / frames"))
        item = self.selection_table.horizontalHeaderItem(4)
        item.setText(_translate("FormatConverterDialog", "Selection"))
        self.frameAdvancedToggle.setText(
            _translate("FormatConverterDialog", "Advanced frame selection")
        )
        self.frameAdvancedDescription.setText(
            _translate(
                "FormatConverterDialog",
                "Use ranges, custom expressions or Every N for multi-frame NXS inputs.",
            )
        )
        self.frame_group.setTitle(
            _translate("FormatConverterDialog", "Frame selection for selected NXS")
        )
        self.frameModeLabel.setText(_translate("FormatConverterDialog", "Mode:"))
        self.frame_mode.setItemText(0, _translate("FormatConverterDialog", "All"))
        self.frame_mode.setItemText(1, _translate("FormatConverterDialog", "Current frame"))
        self.frame_mode.setItemText(2, _translate("FormatConverterDialog", "Frame range"))
        self.frame_mode.setItemText(3, _translate("FormatConverterDialog", "Custom"))
        self.frame_mode.setItemText(4, _translate("FormatConverterDialog", "Every Nth frame"))
        self.rangeLabel.setText(_translate("FormatConverterDialog", "Range:"))
        self.rangeToLabel.setText(_translate("FormatConverterDialog", "to"))
        self.customLabel.setText(_translate("FormatConverterDialog", "Custom:"))
        self.custom_frames.setText(_translate("FormatConverterDialog", "1, 5, 8–20"))
        self.nthLabel.setText(_translate("FormatConverterDialog", "Every N:"))
        self.apply_frames.setText(_translate("FormatConverterDialog", "Apply selection"))
        self.formatPreviewTitle.setText(_translate("FormatConverterDialog", "Preview"))
        self.formatPreviewDescription.setText(
            _translate("FormatConverterDialog", "Inspect representative frames before conversion.")
        )
        self.first_preview_caption.setText(_translate("FormatConverterDialog", "First"))
        self.first_preview_label.setText(_translate("FormatConverterDialog", "No preview"))
        self.middle_preview_caption.setText(_translate("FormatConverterDialog", "Middle"))
        self.middle_preview_label.setText(_translate("FormatConverterDialog", "No preview"))
        self.last_preview_caption.setText(_translate("FormatConverterDialog", "Last"))
        self.last_preview_label.setText(_translate("FormatConverterDialog", "No preview"))
        self.preview_stats.setText(
            _translate("FormatConverterDialog", "Select an input to inspect its frames.")
        )
        self.formatOutputTitle.setText(_translate("FormatConverterDialog", "Configure output"))
        self.formatOutputDescription.setText(
            _translate(
                "FormatConverterDialog", "Choose the destination and primary output contract."
            )
        )
        self.output_format_group.setTitle(_translate("FormatConverterDialog", "Output format"))
        self.tiff_format_button.setProperty(
            "format_name", _translate("FormatConverterDialog", "TIFF")
        )
        self.tiff_format_button.setText(_translate("FormatConverterDialog", "TIFF"))
        self.cbf_format_button.setProperty(
            "format_name", _translate("FormatConverterDialog", "CBF")
        )
        self.cbf_format_button.setText(_translate("FormatConverterDialog", "CBF"))
        self.hdf5_format_button.setProperty(
            "format_name", _translate("FormatConverterDialog", "HDF5")
        )
        self.hdf5_format_button.setText(_translate("FormatConverterDialog", "HDF5"))
        self.numpy_format_button.setProperty(
            "format_name", _translate("FormatConverterDialog", "NumPy")
        )
        self.numpy_format_button.setText(_translate("FormatConverterDialog", "NumPy (.npy)"))
        self.destination_group.setTitle(
            _translate("FormatConverterDialog", "Destination and naming")
        )
        self.destinationLabel.setText(_translate("FormatConverterDialog", "Destination:"))
        self.destination_button.setText(_translate("FormatConverterDialog", "Browse…"))
        self.namingLabel.setText(_translate("FormatConverterDialog", "Multi-frame template:"))
        self.naming_combo.setItemText(
            0, _translate("FormatConverterDialog", "{source}_{frame:06d}")
        )
        self.naming_combo.setItemText(
            1, _translate("FormatConverterDialog", "{source}_img_{frame:06d}")
        )
        self.naming_example.setText(
            _translate("FormatConverterDialog", "Example: scan_001_000123.tif")
        )
        self.add_suffix.setText(
            _translate("FormatConverterDialog", "Add suffix automatically when names collide")
        )
        self.formatOutputAdvancedToggle.setText(
            _translate("FormatConverterDialog", "Advanced output options")
        )
        self.formatOutputAdvancedDescription.setText(
            _translate(
                "FormatConverterDialog",
                "Data type conversion, metadata layout and container options.",
            )
        )
        self.values_group.setTitle(
            _translate("FormatConverterDialog", "Pixel values and data type")
        )
        self.preserve_values.setText(
            _translate(
                "FormatConverterDialog", "Preserve original values and data type when supported"
            )
        )
        self.data_mode.setItemText(
            0, _translate("FormatConverterDialog", "Preserve / use loader data type")
        )
        self.data_mode.setItemText(1, _translate("FormatConverterDialog", "Save as 32-bit float"))
        self.data_mode.setItemText(
            2,
            _translate("FormatConverterDialog", "Convert to uint16 using the original data range"),
        )
        self.data_mode.setItemText(
            3, _translate("FormatConverterDialog", "Clip to the uint16 range")
        )
        self.metadata_group.setTitle(_translate("FormatConverterDialog", "Metadata"))
        self.preserve_metadata.setText(
            _translate("FormatConverterDialog", "Preserve metadata where supported")
        )
        self.write_sidecar.setText(
            _translate("FormatConverterDialog", "Write metadata sidecar JSON")
        )
        self.single_json.setText(
            _translate("FormatConverterDialog", "One metadata file for the whole conversion")
        )
        self.per_image_json.setText(
            _translate("FormatConverterDialog", "One JSON file beside each output image")
        )
        self.advanced_group.setTitle(_translate("FormatConverterDialog", "Advanced"))
        self.container_check.setToolTip(
            _translate(
                "FormatConverterDialog",
                "Stores all selected images as compressed datasets in converted_images.h5.",
            )
        )
        self.container_check.setText(
            _translate("FormatConverterDialog", "Export as one NeXus/HDF5 container")
        )
        self.formatRunTitle.setText(_translate("FormatConverterDialog", "Run, Results & Export"))
        self.formatRunDescription.setText(
            _translate(
                "FormatConverterDialog",
                "Review the estimate, then start conversion. The job dialog provides progress and result links.",
            )
        )
        self.back_button.setText(_translate("FormatConverterDialog", "Back"))
        self.cancel_button.setText(_translate("FormatConverterDialog", "Cancel"))
        self.next_button.setText(_translate("FormatConverterDialog", "Next"))
