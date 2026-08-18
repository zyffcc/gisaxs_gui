"""Scattering Files for fitting presentation."""

from __future__ import annotations

import os


from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
)


from src.gimap.shared.file_paths import normalize_path

from ..binding_primitives import (
    is_matplotlib_available,
)


class ScatteringFilesMixin:
    """Own scattering files behavior."""

    def _import_gisaxs_file(self):
        """GISAXS"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Import GISAXS",
            "",
            "GISAXS Files (*.nxs *.cbf *.tif *.tiff *.dat *.txt *.h5 *.hdf5 *.jpg *.png *.bmp);;NXS Files (*.nxs);;CBF Files (*.cbf);;TIF Files (*.tif *.tiff);;Data Files (*.dat *.txt);;HDF5 Files (*.h5 *.hdf5);;Image Files (*.jpg *.png *.bmp);;All Files (*)",
        )

        if file_path:
            auto_show = bool(
                hasattr(self.ui, "gisaxsInputAutoShowCheckBox")
                and self.ui.gisaxsInputAutoShowCheckBox.isChecked()
            )
            self._apply_imported_gisaxs_file(file_path, show_image=auto_show)

    def _apply_imported_gisaxs_file(self, file_path, show_image=False):
        """Apply a dialog, text-entry, or dropped detector image through one path."""
        file_path = normalize_path(file_path)
        if not self._validate_imported_file(file_path):
            return False
        self.current_parameters["imported_gisaxs_file"] = file_path
        self._set_nxs_frame_state(file_path, 0)
        if hasattr(self.ui, "gisaxsInputImportButtonValue"):
            self.ui.gisaxsInputImportButtonValue.setText(os.path.basename(file_path))
        self._scan_folder_images_for_file(file_path)
        self._update_stack_display()
        self.status_updated.emit(f"Imported GISAXS file: {os.path.basename(file_path)}")
        self.parameters_changed.emit(self.current_parameters)
        if hasattr(self.parent, "save_current_session"):
            self.parent.save_current_session()
        if show_image:
            self._show_image()
        return True

    def _validate_imported_file(self, file_path):
        """ISAXS"""
        try:
            file_path = normalize_path(file_path)
            if self.fitting_view_model.storage.is_remote_source(file_path):
                self.status_updated.emit(
                    "Remote/cloud file selected. Validation will run in the background loader."
                )
                return True

            if not os.path.exists(file_path):
                QMessageBox.warning(
                    self.main_window, "File Error", f"File does not exist: {file_path}"
                )
                return False

            file_size = os.path.getsize(file_path)
            if file_size == 0:
                QMessageBox.warning(self.main_window, "File Error", "File is empty")
                return False

            file_ext = os.path.splitext(file_path)[1].lower()
            supported_extensions = [
                ".nxs",
                ".cbf",
                ".tif",
                ".tiff",
                ".dat",
                ".txt",
                ".h5",
                ".hdf5",
                ".jpg",
                ".png",
                ".bmp",
            ]

            if file_ext not in supported_extensions:
                reply = QMessageBox.question(
                    self.main_window,
                    "File Format Warning",
                    f"The file extension '{file_ext}' may not be supported.\nSupported formats: {', '.join(supported_extensions)}\n\nContinue import?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply == QMessageBox.No:
                    return False

            self.status_updated.emit(f"File validation passed - {os.path.basename(file_path)}")
            return True

        except Exception as e:
            QMessageBox.critical(
                self.main_window, "File Validation Error", f"Error validating file:\n{str(e)}"
            )
            return False

    def _on_import_value_changed(self):
        """No description."""
        try:
            if not hasattr(self.ui, "gisaxsInputImportButtonValue"):
                return

            file_path_input = self.ui.gisaxsInputImportButtonValue.text().strip()

            if not file_path_input:
                self.status_updated.emit("Please enter a valid file path")
                return

            if not os.path.isabs(file_path_input):
                current_file = self.current_parameters.get("imported_gisaxs_file", "")
                if current_file and (
                    self.fitting_view_model.storage.is_remote_source(current_file)
                    or os.path.exists(current_file)
                ):
                    current_dir = os.path.dirname(current_file)
                    file_path_input = os.path.join(current_dir, file_path_input)
                else:
                    file_path_input = os.path.abspath(file_path_input)

            if not self.fitting_view_model.storage.is_remote_source(
                file_path_input
            ) and not os.path.exists(file_path_input):
                self.status_updated.emit(
                    f"File does not exist: {os.path.basename(file_path_input)}"
                )
                QMessageBox.warning(
                    self.main_window, "File Error", f"File does not exist:\n{file_path_input}"
                )
                return

            self.current_parameters["imported_gisaxs_file"] = file_path_input
            self._set_nxs_frame_state(file_path_input, 0)

            file_name = os.path.basename(file_path_input)
            self.ui.gisaxsInputImportButtonValue.setText(file_name)

            if self._validate_imported_file(file_path_input):
                self._scan_folder_images_for_file(file_path_input)
                self.status_updated.emit(f"Updated GISAXS file: {file_name}")
                self.parameters_changed.emit(self.current_parameters)

                if hasattr(self.parent, "save_current_session"):
                    self.parent.save_current_session()

                self._update_stack_display()
                self._refresh_vmin_vmax_display()

                if (
                    hasattr(self.ui, "gisaxsInputAutoShowCheckBox")
                    and self.ui.gisaxsInputAutoShowCheckBox.isChecked()
                ):
                    self._show_image()
                else:
                    self.status_updated.emit("File updated. Click 'Show' to display the image")

        except Exception as e:
            self.status_updated.emit(f"Import value processing error: {str(e)}")
            QMessageBox.critical(
                self.main_window,
                "Processing Error",
                f"Error handling the imported file path:\n{str(e)}",
            )

    def _on_stack_value_changed(self):
        """No description."""
        try:
            stack_text = (
                self.ui.gisaxsInputStackValue.text()
                if hasattr(self.ui, "gisaxsInputStackValue")
                else "1"
            )

            try:
                stack_count = int(stack_text)
            except ValueError:
                if hasattr(self.ui, "gisaxsInputStackValue"):
                    self.ui.gisaxsInputStackValue.setText("1")
                stack_count = 1

            if stack_count < 1:
                if hasattr(self.ui, "gisaxsInputStackDisplayLabel"):
                    self.ui.gisaxsInputStackDisplayLabel.setText("At least 1")
                return

            self.current_parameters["stack_count"] = stack_count
            if getattr(self, "load_mode", "Single") == "Stack":
                stack_count = self._clamp_stack_count(stack_count)
            self._update_stack_display()
            self._refresh_vmin_vmax_display()

            should_reload_image = False

            if (
                hasattr(self.ui, "gisaxsInputAutoShowCheckBox")
                and self.ui.gisaxsInputAutoShowCheckBox.isChecked()
            ):
                should_reload_image = True
            elif self.current_stack_data is not None:
                imported_file = self.current_parameters.get("imported_gisaxs_file", "")
                if imported_file and os.path.splitext(imported_file)[1].lower() in {
                    ".cbf",
                    ".nxs",
                    ".tif",
                    ".tiff",
                }:
                    should_reload_image = True

            if should_reload_image:
                self._show_image()
            else:
                self.status_updated.emit(f"Stack count updated to {stack_count}")

        except Exception as e:
            self.status_updated.emit(f"Stack value processing error: {str(e)}")

    def _update_stack_display(self):
        """No description."""
        try:
            imported_file = self.current_parameters.get("imported_gisaxs_file", "")
            if not imported_file:
                return

            file_ext = os.path.splitext(imported_file)[1].lower()
            mode = getattr(self, "load_mode", "Single")
            stack_count = self.current_parameters.get("stack_count", 1)

            if mode == "Stack":
                stack_count = self._clamp_stack_count(stack_count)

            if file_ext == ".nxs":
                if hasattr(self.ui, "gisaxsInputStackDisplayLabel"):
                    if mode == "Stack":
                        end_frame = self._nxs_frame_index + stack_count
                        self.ui.gisaxsInputStackDisplayLabel.setText(
                            f"NXS Stack: frames {self._nxs_frame_index + 1} - {end_frame} / {self._nxs_frame_count}"
                        )
                    else:
                        self.ui.gisaxsInputStackDisplayLabel.setText(
                            f"NXS frame: {self._nxs_frame_index + 1} / {self._nxs_frame_count}"
                        )
                return

            if file_ext not in {".cbf", ".tif", ".tiff"}:
                if hasattr(self.ui, "gisaxsInputStackDisplayLabel"):
                    self.ui.gisaxsInputStackDisplayLabel.setText(
                        f"File: {os.path.basename(imported_file)}"
                    )
                return

            if mode == "Single" or stack_count == 1:
                if hasattr(self.ui, "gisaxsInputStackDisplayLabel"):
                    self.ui.gisaxsInputStackDisplayLabel.setText(
                        f"Single: {os.path.basename(imported_file)}"
                    )
                return

            if mode == "Stack":
                sequence = self._ordinary_stack_sequence(imported_file)
                selected = sequence[:stack_count]
                if selected and hasattr(self.ui, "gisaxsInputStackDisplayLabel"):
                    start_name = os.path.splitext(os.path.basename(selected[0]))[0]
                    end_name = os.path.splitext(os.path.basename(selected[-1]))[0]
                    self.ui.gisaxsInputStackDisplayLabel.setText(
                        f"Stack: {start_name} - {end_name}"
                    )
                return

            if mode == "In-situ":
                if file_ext != ".cbf":
                    if hasattr(self.ui, "gisaxsInputStackDisplayLabel"):
                        self.ui.gisaxsInputStackDisplayLabel.setText(
                            f"File: {os.path.basename(imported_file)}"
                        )
                    return
                dir_path = os.path.dirname(imported_file)
                sv = ""
                try:
                    if hasattr(self.ui, "gisaxsInputStackValue"):
                        sv = self.ui.gisaxsInputStackValue.text().strip()
                except Exception:
                    sv = ""
                latest = ""
                if self.fitting_view_model.storage.is_remote_source(dir_path):
                    cached_paths = self._folder_image_scan_cache.get(normalize_path(dir_path))
                    if cached_paths:
                        latest = cached_paths[-1]
                    else:
                        self._scan_folder_images_for_file(imported_file)
                else:
                    latest = self._find_latest_cbf(dir_path)
                if hasattr(self.ui, "gisaxsInputStackDisplayLabel"):
                    if sv == "" or sv.endswith("-"):
                        self.ui.gisaxsInputStackDisplayLabel.setText(
                            f"In-situ: latest -> {os.path.splitext(os.path.basename(latest or 'scanning...'))[0]}"
                        )
                    elif "-" in sv:
                        self.ui.gisaxsInputStackDisplayLabel.setText(f"In-situ range: {sv}")
                    else:
                        self.ui.gisaxsInputStackDisplayLabel.setText(f"In-situ index: {sv}")
                return

        except Exception as e:
            self.status_updated.emit(f"Display update error: {str(e)}")

    def _sync_ui_to_parameters(self):
        """UI"""
        try:
            if hasattr(self.ui, "gisaxsInputImportButtonValue"):
                file_input = self.ui.gisaxsInputImportButtonValue.text().strip()
                if file_input:
                    if os.path.isabs(file_input) and (
                        self.fitting_view_model.storage.is_remote_source(file_input)
                        or os.path.exists(file_input)
                    ):
                        self.current_parameters["imported_gisaxs_file"] = file_input
                    elif not os.path.isabs(file_input):
                        file_found = False

                        current_file = self.current_parameters.get("imported_gisaxs_file", "")
                        if current_file and os.path.dirname(current_file):
                            new_path = os.path.join(os.path.dirname(current_file), file_input)
                            if self.fitting_view_model.storage.is_remote_source(
                                new_path
                            ) or os.path.exists(new_path):
                                self.current_parameters["imported_gisaxs_file"] = new_path
                                file_found = True

                        if not file_found:
                            experiment_dir = os.path.join(
                                os.path.dirname(os.path.dirname(__file__)), "Experiment_data"
                            )
                            if os.path.exists(experiment_dir):
                                new_path = os.path.join(experiment_dir, file_input)
                                if os.path.exists(new_path):
                                    self.current_parameters["imported_gisaxs_file"] = new_path
                                    file_found = True

                        if not file_found:
                            self.status_updated.emit(
                                f"Error: File '{file_input}' not found in any expected location"
                            )
                            return

                    elif (
                        os.path.isabs(file_input)
                        and not self.fitting_view_model.storage.is_remote_source(file_input)
                        and not os.path.exists(file_input)
                    ):
                        self.status_updated.emit(f"Error: File '{file_input}' does not exist")
                        return

            if hasattr(self.ui, "gisaxsInputStackValue"):
                sv = self.ui.gisaxsInputStackValue.text().strip()
                if getattr(self, "load_mode", "Single") == "Single":
                    self.current_parameters["stack_count"] = 1
                elif self.load_mode == "Stack":
                    try:
                        self.current_parameters["stack_count"] = max(1, int(sv or "1"))
                    except Exception:
                        self.current_parameters["stack_count"] = 1
                elif self.load_mode == "In-situ":
                    self.current_parameters["insitu_range"] = sv

        except Exception as e:
            self.status_updated.emit(f"Failed to sync UI parameters: {str(e)}")

    def _show_image(self):
        """No description."""
        try:
            self._sync_ui_to_parameters()

            imported_file = self.current_parameters.get("imported_gisaxs_file", "")
            if not imported_file:
                self.status_updated.emit("No file imported to show")
                return

            if not self.fitting_view_model.storage.is_remote_source(
                imported_file
            ) and not os.path.exists(imported_file):
                self.status_updated.emit("File does not exist")
                QMessageBox.warning(
                    self.main_window, "File Error", f"File does not exist:\n{imported_file}"
                )
                self._scan_folder_images_for_file(imported_file)
                return

            self._scan_folder_images_for_file(imported_file)

            file_ext = os.path.splitext(imported_file)[1].lower()
            if file_ext == ".cbf" and not self.fitting_view_model.storage.dependency_available(
                "fabio"
            ):
                QMessageBox.warning(
                    self.main_window,
                    "Missing Library",
                    "fabio library is required for CBF file processing.\nPlease install it using: pip install fabio",
                )
                return

            if not is_matplotlib_available():
                QMessageBox.warning(
                    self.main_window,
                    "Missing Library",
                    "matplotlib library is required for image display.\nPlease install it using: pip install matplotlib",
                )
                return

            if file_ext not in {".cbf", ".nxs", ".tif", ".tiff"}:
                self.status_updated.emit(
                    "Image display supports CBF, NXS, and TIFF detector images"
                )
                return

            self.fitting_view_model.begin_image_load(os.path.basename(imported_file))

            mode = getattr(self, "load_mode", "Single")
            if file_ext == ".nxs":
                frame_index = self._nxs_frame_index
                stack_count = self._clamp_stack_count() if mode == "Stack" else 1
                if stack_count > 1:
                    self.status_updated.emit(
                        f"Please wait while stacking NXS frames {frame_index + 1}-{frame_index + stack_count}..."
                    )
                else:
                    self.status_updated.emit(
                        f"Please wait while loading {os.path.basename(imported_file)} "
                        f"frame {frame_index + 1}/{self._nxs_frame_count}..."
                    )
                self.async_image_loader.load_image(
                    imported_file,
                    stack_count,
                    frame_index=frame_index,
                )
                return

            if file_ext in {".tif", ".tiff"}:
                stack_count = self._clamp_stack_count() if mode == "Stack" else 1
                self.status_updated.emit(f"Please wait while loading {stack_count} TIFF file(s)...")
                self.async_image_loader.load_image(imported_file, stack_count)
                return

            if mode == "Single":
                self.status_updated.emit("Please wait while the image starts loading (Single)...")
                self.async_image_loader.load_image(imported_file, 1)
            elif mode == "Stack":
                stack_count = self._clamp_stack_count()
                self.status_updated.emit(f"Please wait while stacking {stack_count} files...")
                self.async_image_loader.load_image(imported_file, stack_count)
            else:
                sv = ""
                try:
                    if hasattr(self.ui, "gisaxsInputStackValue"):
                        sv = self.ui.gisaxsInputStackValue.text().strip()
                except Exception:
                    sv = ""
                dir_path = os.path.dirname(imported_file)
                target = self._resolve_insitu_target(dir_path, imported_file, sv)
                if not target:
                    self.status_updated.emit("No CBF file found for In-situ mode")
                    return
                self._insitu_last_file = target
                self._show_image_insitu(target)

        except Exception as e:
            self.status_updated.emit(f"Show image error: {str(e)}")

    def _on_load_mode_changed(self, text: str):
        """No description."""
        try:
            self.load_mode = text or "Single"
            try:
                self.fitting_view_model.set_setting(
                    "fitting", "gisaxs_input.load_mode", self.load_mode
                )
            except Exception:
                pass
            self._update_stack_controls_visibility()
            if self.load_mode == "In-situ":
                if self._is_auto_show_enabled():
                    self._start_insitu_timer()
            else:
                self._stop_insitu_timer()
                self._stop_insitu_workflow()
                dialog = getattr(self, "_insitu_workflow_dialog", None)
                if dialog is not None and dialog.isVisible():
                    dialog.close()
            self._update_stack_display()
            self._update_insitu_workflow_button_visibility()
        except Exception:
            pass
