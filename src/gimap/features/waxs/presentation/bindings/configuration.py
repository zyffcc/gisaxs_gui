"""WAXS UI configuration snapshot, persistence, and JSON dialogs."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QTableWidgetItem,
)


class WaxsConfigurationMixin:
    _CONFIG_VERSION = 1

    def _setup_configuration_persistence(self) -> None:
        self._applying_waxs_configuration = False
        self._waxs_settings_timer = QTimer(self)
        self._waxs_settings_timer.setSingleShot(True)
        self._waxs_settings_timer.setInterval(250)
        self._waxs_settings_timer.timeout.connect(self._save_waxs_settings)
        if hasattr(self.view_model, "load_settings"):
            stored = self.view_model.load_settings()
            if stored:
                try:
                    self._apply_waxs_configuration(stored)
                except (TypeError, ValueError):
                    # A stale/corrupt settings section must not prevent WAXS startup.
                    pass
        for widget in self._configuration_widgets():
            signal = None
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                signal = widget.valueChanged
            elif isinstance(widget, QCheckBox):
                signal = widget.toggled
            elif isinstance(widget, QComboBox):
                signal = widget.currentIndexChanged
            elif isinstance(widget, QLineEdit):
                signal = widget.textChanged
            if signal is not None:
                signal.connect(self._schedule_waxs_settings_save)
        self.batch_sources_table.cellChanged.connect(
            self._schedule_waxs_settings_save
        )
        model = self.batch_sources_table.model()
        model.rowsInserted.connect(self._schedule_waxs_settings_save)
        model.rowsRemoved.connect(self._schedule_waxs_settings_save)

    def _configuration_widgets(self):
        widgets = []
        for cls in (QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit):
            widgets.extend(self.findChildren(cls))
        return [
            widget
            for widget in widgets
            if widget.objectName() and not widget.objectName().startswith("qt_")
        ]

    def _waxs_configuration(self) -> dict:
        parameters = {}
        for widget in self._configuration_widgets():
            name = widget.objectName()
            if isinstance(widget, QCheckBox):
                parameters[name] = {"type": "bool", "value": widget.isChecked()}
            elif isinstance(widget, QComboBox):
                parameters[name] = {
                    "type": "combo",
                    "data": widget.currentData(),
                    "text": widget.currentText(),
                }
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                parameters[name] = {"type": "number", "value": widget.value()}
            elif isinstance(widget, QLineEdit):
                parameters[name] = {"type": "text", "value": widget.text()}
        sources = []
        for row in range(self.batch_sources_table.rowCount()):
            sources.append(
                [
                    self.batch_sources_table.item(row, column).text()
                    if self.batch_sources_table.item(row, column)
                    else ""
                    for column in range(3)
                ]
            )
        return {
            "version": self._CONFIG_VERSION,
            "parameters": parameters,
            "batch_sources": sources,
        }

    def _apply_waxs_configuration(self, payload: dict) -> None:
        if int(payload.get("version", 1)) != self._CONFIG_VERSION:
            raise ValueError("Unsupported WAXS configuration version.")
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("WAXS configuration parameters must be an object.")
        self._applying_waxs_configuration = True
        try:
            widgets = {widget.objectName(): widget for widget in self._configuration_widgets()}
            for name, state in parameters.items():
                widget = widgets.get(name)
                if widget is None or not isinstance(state, dict):
                    continue
                blocked = widget.blockSignals(True)
                try:
                    if isinstance(widget, QCheckBox):
                        widget.setChecked(bool(state.get("value", False)))
                    elif isinstance(widget, QComboBox):
                        index = widget.findData(state.get("data"))
                        if index < 0:
                            index = widget.findText(str(state.get("text", "")))
                        if index >= 0:
                            widget.setCurrentIndex(index)
                    elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                        widget.setValue(state.get("value", widget.value()))
                    elif isinstance(widget, QLineEdit):
                        widget.setText(str(state.get("value", "")))
                finally:
                    widget.blockSignals(blocked)
            sources = payload.get("batch_sources", [])
            if isinstance(sources, list):
                self.batch_sources_table.blockSignals(True)
                try:
                    self.batch_sources_table.setRowCount(0)
                    for values in sources:
                        if not isinstance(values, list):
                            continue
                        row = self.batch_sources_table.rowCount()
                        self.batch_sources_table.insertRow(row)
                        for column in range(3):
                            value = values[column] if column < len(values) else ""
                            self.batch_sources_table.setItem(
                                row, column, QTableWidgetItem(str(value))
                            )
                finally:
                    self.batch_sources_table.blockSignals(False)
        finally:
            self._applying_waxs_configuration = False
        self._update_batch_preprocessing_enabled()
        self._update_batch_q_range_enabled(self.batch_limit_q_range.isChecked())
        self._update_batch_export_limits_enabled(
            self.batch_export_auto_scale.isChecked()
        )
        self._update_geometry_summaries()
        self.refresh_view()
        # Restore background frame range/state and re-apply to an open image.
        if getattr(self, "background_path_edit", None):
            self._update_background_frame_range_from_edit()
        if getattr(self, "current_file", None):
            self._reload_current_for_background()

    def _schedule_waxs_settings_save(self, *_args) -> None:
        if not self._applying_waxs_configuration:
            self._waxs_settings_timer.start()

    def _save_waxs_settings(self) -> None:
        if hasattr(self.view_model, "save_settings"):
            self.view_model.save_settings(self._waxs_configuration())

    def save_waxs_configuration_dialog(self) -> None:
        default = str(Path(self.view_model.working_directory()) / "waxs_config.json")
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save WAXS Configuration", default, "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            self.view_model.save_configuration(
                Path(self.view_model.normalize_path(path)), self._waxs_configuration()
            )
            self._set_status("WAXS configuration saved")
        except Exception as exc:
            QMessageBox.warning(self, "Save Configuration", str(exc))

    def load_waxs_configuration_dialog(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Load WAXS Configuration",
            self.view_model.working_directory(),
            "JSON Files (*.json)",
        )
        if not path:
            return
        try:
            payload = self.view_model.load_configuration(
                Path(self.view_model.normalize_path(path))
            )
            self._apply_waxs_configuration(payload)
            self._save_waxs_settings()
            self._set_status("WAXS configuration loaded")
        except Exception as exc:
            QMessageBox.warning(self, "Load Configuration", str(exc))

    def closeEvent(self, event) -> None:
        self._save_waxs_settings()
        super().closeEvent(event)
