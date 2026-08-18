"""Curve Files for fitting presentation."""

from __future__ import annotations

import os


from pathlib import Path


from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

from src.gimap.features.fitting.application import (
    LoadCurveRequest,
)


from src.gimap.shared.file_paths import normalize_path


class CurveFilesMixin:
    """Own curve files behavior."""

    def _import_1d_file(self):
        """No description."""
        try:
            fitting_session = self.fitting_view_model.get_setting("fitting", "last_session", {})
            last_1d_directory = fitting_session.get("last_1d_directory")

            if last_1d_directory and os.path.exists(last_1d_directory):
                start_directory = last_1d_directory
            else:
                start_directory = os.getcwd()

            file_path, _ = QFileDialog.getOpenFileName(
                self.main_window,
                "Select 1D SAXS Data File",
                start_directory,
                "Data Files (*.dat *.txt);;All Files (*)",
            )

            if not file_path:
                return
            file_path = normalize_path(file_path)

            self.current_1d_file_path = file_path

            current_directory = os.path.dirname(file_path)
            fitting_session["last_1d_directory"] = current_directory
            self.fitting_view_model.set_setting("fitting", "last_session", fitting_session)

            self._load_1d_data(file_path)

        except Exception as e:
            self.status_updated.emit(f"Failed to import 1D file: {str(e)}")
            QMessageBox.critical(self.main_window, "Error", f"Failed to import 1D file:\n{str(e)}")

    def _on_1d_file_value_changed(self):
        """No description."""
        try:
            if not hasattr(self.ui, "fitImport1dFileValue"):
                return

            file_path_input = self.ui.fitImport1dFileValue.text().strip()

            if not file_path_input:
                self.status_updated.emit("No file path entered")
                return
            if not os.path.isabs(file_path_input):
                fitting_session = self.fitting_view_model.get_setting("fitting", "last_session", {})
                last_1d_directory = fitting_session.get("last_1d_directory")

                if last_1d_directory and os.path.exists(last_1d_directory):
                    file_path_input = os.path.join(last_1d_directory, file_path_input)
                else:
                    file_path_input = os.path.join(os.getcwd(), file_path_input)

            if not os.path.exists(file_path_input):
                QMessageBox.warning(
                    self.main_window, "File Not Found", f"File does not exist:\n{file_path_input}"
                )
                return

            file_ext = os.path.splitext(file_path_input)[1].lower()
            if file_ext not in [".dat", ".txt"]:
                QMessageBox.warning(
                    self.main_window,
                    "Invalid File Type",
                    f"Only .dat and .txt files are supported.\nSelected: {file_ext}",
                )
                return

            self.ui.fitImport1dFileValue.setText(file_path_input)

            self.current_1d_file_path = file_path_input

            current_directory = os.path.dirname(file_path_input)
            fitting_session = self.fitting_view_model.get_setting("fitting", "last_session", {})
            fitting_session["last_1d_directory"] = current_directory
            self.fitting_view_model.set_setting("fitting", "last_session", fitting_session)

            self._load_1d_data(file_path_input)

        except Exception as e:
            self.status_updated.emit(f"Failed to process 1D file path: {str(e)}")
            QMessageBox.critical(
                self.main_window, "Error", f"Failed to process 1D file path:\n{str(e)}"
            )

    def _load_1d_data(self, file_path):
        """No description."""
        try:
            self.status_updated.emit(f"Loading 1D data from {os.path.basename(file_path)}...")
            outcome = self.fitting_view_model.load_curve(
                LoadCurveRequest(
                    path=Path(file_path),
                    q_source_unit=self._imported_1d_q_unit,
                )
            )
            if outcome.error is not None:
                raise RuntimeError(f"[{outcome.error.code}] {outcome.error.message}")
            data = outcome.value

            self.q = data.q
            self.I = data.intensity

            self.current_1d_data = {
                "q": data.q,
                "I": data.intensity,
                "err": data.error,
                "file_path": file_path,
                "q_source_unit": data.q_source_unit,
            }

            self.data_source = "1d"
            self.display_mode = "normal"
            if hasattr(self.ui, "fitCurrentDataCheckBox"):
                self.ui.fitCurrentDataCheckBox.blockSignals(True)
                self.ui.fitCurrentDataCheckBox.setChecked(False)
                self.ui.fitCurrentDataCheckBox.blockSignals(False)

            if hasattr(self.ui, "fitImport1dFileValue"):
                self.ui.fitImport1dFileValue.setText(file_path)

            try:
                self._initialize_roi_from_current_q(force_full=True)
            except Exception:
                pass
            self._apply_roi_to_data_and_refresh()
            self._update_GUI_image("normal")
            self._update_outside_window("normal")

            self.status_updated.emit(
                f"Successfully loaded 1D data: {os.path.basename(file_path)} ({len(self.q)} points)"
            )

        except Exception as e:
            self.status_updated.emit(f"Failed to load 1D data: {str(e)}")
            QMessageBox.critical(
                self.main_window,
                "Error",
                f"Failed to load 1D data from {os.path.basename(file_path)}:\n{str(e)}",
            )
