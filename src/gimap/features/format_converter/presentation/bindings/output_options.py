"""Output Options behavior for Format Converter."""

from __future__ import annotations


from PyQt5.QtWidgets import (
    QFileDialog,
)


from ..display_formatting import _human_bytes


class OutputOptionsMixin:
    """Own output options presentation behavior."""

    def _configure_output_formats(self) -> None:
        visibility = self.view_model.output_format_visibility(
            container=self.container_check.isChecked(),
        )
        for name, button in self.format_buttons.items():
            button.setVisible(visibility[name])
        checked = self.format_group.checkedButton()
        if checked is None or not checked.isVisible():
            for button in self.format_buttons.values():
                if button.isVisible():
                    button.setChecked(True)
                    break
        self._update_output_preview()

    def _selected_format(self) -> str:
        button = self.format_group.checkedButton()
        return str(button.property("format_name")) if button is not None else "TIFF"

    def _choose_destination(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Choose output folder", self.destination_edit.text()
        )
        if folder:
            self.destination_edit.setText(self.view_model.normalize_path(folder))

    def _container_toggled(self, checked: bool) -> None:
        if checked:
            self.format_buttons["HDF5"].setVisible(True)
            self.format_buttons["HDF5"].setChecked(True)
        self.naming_combo.setEnabled(not checked)
        self.add_suffix.setEnabled(not checked)
        if checked:
            self._update_output_preview()
        else:
            self._configure_output_formats()

    def _sidecar_toggled(self, checked: bool) -> None:
        self.single_json.setEnabled(checked)
        self.per_image_json.setEnabled(checked)

    def _preserve_values_toggled(self, checked: bool) -> None:
        if checked:
            self.data_mode.setCurrentIndex(0)
        elif self.data_mode.currentData() == "original":
            self.data_mode.setCurrentIndex(1)
        self._update_output_preview()

    def _update_output_preview(self) -> None:
        try:
            options = self._options()
            preview = self.view_model.output_preview(options)
            self.naming_example.setText(f"Example: {preview.example}")
            self.output_summary.setText(
                f"Estimated output: {preview.image_count:,} image(s) in "
                f"{preview.file_count:,} file(s), approximately "
                f"{_human_bytes(preview.estimated_bytes)}"
            )
            self.dtype_warning.setText(preview.dtype_warning)
        except Exception:
            self.output_summary.setText("")
            self.dtype_warning.setText("")

    def _options(self):
        return self.view_model.make_options(
            output_format=self._selected_format(),
            destination=self.destination_edit.text().strip(),
            naming_template=self.naming_combo.currentText().strip() or "{source}_{frame:06d}",
            add_suffix=self.add_suffix.isChecked(),
            preserve_values=self.preserve_values.isChecked(),
            data_mode=str(self.data_mode.currentData()),
            preserve_metadata=self.preserve_metadata.isChecked(),
            write_sidecar=self.write_sidecar.isChecked(),
            single_metadata_file=self.single_json.isChecked(),
            container=self.container_check.isChecked(),
        )
