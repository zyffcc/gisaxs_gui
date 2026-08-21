"""Image Navigation coordination for the fitting workspace."""

from __future__ import annotations

import os
import re
from pathlib import Path


from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QHBoxLayout,
    QWidget,
    QSizePolicy,
    QLabel,
    QDoubleSpinBox,
    QCheckBox,
    QPushButton,
    QLineEdit,
)


from src.gimap.shared.file_paths import normalize_path


from ..binding_primitives import (
    FolderImageScanWorker,
)


class ImageNavigationMixin:
    """Own image navigation presentation behavior."""

    def _setup_folder_navigation_ui(self):
        """Add Previous/Next buttons beside the current GISAXS file field."""
        try:
            if self._previous_image_button is not None:
                return

            parent = getattr(self.ui, "gisaxsInputBox", None)
            self._previous_image_button = QPushButton("Previous", parent)
            self._next_image_button = QPushButton("Next", parent)
            self._previous_image_button.setObjectName("gisaxsInputPreviousImageButton")
            self._next_image_button.setObjectName("gisaxsInputNextImageButton")
            self._previous_image_button.setToolTip("No previous image")
            self._next_image_button.setToolTip("No next image")
            self._previous_image_button.setEnabled(False)
            self._next_image_button.setEnabled(False)
            self._image_position_label = QLabel("— / —", parent)
            self._image_position_label.setObjectName("gisaxsInputImagePositionLabel")
            self._image_position_label.setToolTip("Current file position in this folder")

            nav_layout = getattr(self.ui, "gisaxsInputFileNavigationLayout", None)
            if nav_layout is not None:
                self._previous_image_button.setParent(
                    getattr(self.ui, "gisaxsInputFileNavigationWidget", parent)
                )
                self._next_image_button.setParent(
                    getattr(self.ui, "gisaxsInputFileNavigationWidget", parent)
                )
                show_button = getattr(self.ui, "gisaxsInputShowButton", None)
                show_index = nav_layout.indexOf(show_button) if show_button is not None else -1
                insert_at = show_index if show_index >= 0 else nav_layout.count()
                nav_layout.insertWidget(insert_at, self._previous_image_button)
                nav_layout.insertWidget(insert_at + 1, self._next_image_button)
                nav_layout.insertWidget(insert_at + 2, self._image_position_label)
            elif hasattr(self.ui, "gridLayout_23"):
                self.ui.gridLayout_23.addWidget(self._previous_image_button, 0, 5, 1, 1)
                self.ui.gridLayout_23.addWidget(self._next_image_button, 0, 6, 1, 1)
        except Exception as e:
            self.status_updated.emit(f"Failed to set up image navigation: {str(e)}")

    def _load_remote_cache_settings(self):
        try:
            self._remote_copy_enabled = bool(self.preferences.get("remote.copy_to_cache", True))
            self._remote_cache_dir = self.fitting_view_model.storage.display_remote_cache_directory(
                self.preferences.get("remote.cache_dir", "")
            )
            self._remote_cache_limit_gb = float(self.preferences.get("remote.cache_limit_gb", 3.0))
        except Exception:
            self._remote_copy_enabled = True
            self._remote_cache_dir = (
                self.fitting_view_model.storage.default_remote_cache_directory()
            )
            self._remote_cache_limit_gb = 3.0
        self._configure_async_loader_remote_cache()

    def _save_remote_cache_settings(self):
        try:
            self.preferences.set("remote.copy_to_cache", bool(self._remote_copy_enabled))
            self.preferences.set(
                "remote.cache_dir",
                self.fitting_view_model.storage.display_remote_cache_directory(
                    self._remote_cache_dir
                    or self.fitting_view_model.storage.default_remote_cache_directory()
                ),
            )
            self.preferences.set("remote.cache_limit_gb", float(self._remote_cache_limit_gb or 3.0))
            self.preferences.save()
        except Exception:
            pass

    def _configure_async_loader_remote_cache(self):
        try:
            self.async_image_loader.configure_remote_cache(
                enabled=bool(self._remote_copy_enabled),
                cache_dir=self._remote_cache_dir,
                max_gb=self._remote_cache_limit_gb,
            )
        except Exception:
            pass

    def _setup_remote_cache_controls(self):
        try:
            if getattr(self, "_remote_cache_controls", None):
                return
            parent = getattr(
                self.ui,
                "gisaxsRemoteCacheControlsHost",
                getattr(self.ui, "gisaxsInputBox", None),
            )
            target_layout = getattr(
                self.ui,
                "gisaxsRemoteCacheControlsLayout",
                parent.layout() if parent is not None else None,
            )
            if parent is None or target_layout is None:
                return
            row = QWidget(parent)
            row.setObjectName("gisaxsRemoteCacheRow")
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 2, 0, 0)
            layout.setSpacing(6)

            copy_cb = QCheckBox("Copy cloud/network files to local cache", row)
            copy_cb.setChecked(bool(self._remote_copy_enabled))
            path_edit = QLineEdit(row)
            path_edit.setText(self._remote_cache_dir)
            path_edit.setPlaceholderText(".gimap_cache/remote_files")
            path_edit.setMinimumWidth(180)
            browse_btn = QPushButton("...", row)
            browse_btn.setFixedWidth(32)
            limit_spin = QDoubleSpinBox(row)
            limit_spin.setRange(0.25, 100.0)
            limit_spin.setDecimals(2)
            limit_spin.setSingleStep(0.5)
            limit_spin.setValue(float(self._remote_cache_limit_gb or 3.0))
            limit_spin.setSuffix(" GB")
            clear_btn = QPushButton("Clear Cache", row)

            layout.addWidget(copy_cb)
            layout.addWidget(QLabel("Cache:", row))
            layout.addWidget(path_edit, 1)
            layout.addWidget(browse_btn)
            layout.addWidget(QLabel("Max:", row))
            layout.addWidget(limit_spin)
            layout.addWidget(clear_btn)
            row.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
            target_layout.addWidget(row)

            self._remote_cache_controls = {
                "row": row,
                "copy": copy_cb,
                "path": path_edit,
                "browse": browse_btn,
                "limit": limit_spin,
                "clear": clear_btn,
            }
            copy_cb.toggled.connect(self._on_remote_cache_setting_changed)
            path_edit.editingFinished.connect(self._on_remote_cache_setting_changed)
            limit_spin.valueChanged.connect(self._on_remote_cache_setting_changed)
            browse_btn.clicked.connect(self._browse_remote_cache_folder)
            clear_btn.clicked.connect(self._clear_remote_file_cache)
        except Exception as exc:
            self.status_updated.emit(f"Remote cache controls failed: {exc}")

    def _on_remote_cache_setting_changed(self):
        try:
            widgets = getattr(self, "_remote_cache_controls", {})
            copy_cb = widgets.get("copy")
            path_edit = widgets.get("path")
            limit_spin = widgets.get("limit")
            self._remote_copy_enabled = bool(copy_cb.isChecked()) if copy_cb is not None else True
            self._remote_cache_dir = (
                self.fitting_view_model.storage.display_remote_cache_directory(
                    path_edit.text().strip()
                )
                if path_edit is not None
                else self.fitting_view_model.storage.default_remote_cache_directory()
            )
            self._remote_cache_limit_gb = (
                float(limit_spin.value()) if limit_spin is not None else 3.0
            )
            self._configure_async_loader_remote_cache()
            self._save_remote_cache_settings()
        except Exception as exc:
            self.status_updated.emit(f"Remote cache setting failed: {exc}")

    def _browse_remote_cache_folder(self):
        try:
            start = self.fitting_view_model.storage.resolve_remote_cache_directory(
                self._remote_cache_dir
                or self.fitting_view_model.storage.default_remote_cache_directory()
            )
            folder = QFileDialog.getExistingDirectory(
                self.main_window, "Select Remote File Cache Folder", start
            )
            if not folder:
                return
            widgets = getattr(self, "_remote_cache_controls", {})
            edit = widgets.get("path")
            if edit is not None:
                edit.setText(normalize_path(folder))
            self._on_remote_cache_setting_changed()
        except Exception as exc:
            self.status_updated.emit(f"Select cache folder failed: {exc}")

    def _clear_remote_file_cache(self):
        try:
            removed = self.fitting_view_model.storage.clear_remote_cache(
                self._remote_cache_dir
                or self.fitting_view_model.storage.default_remote_cache_directory()
            )
            self.status_updated.emit(f"Remote raw file cache cleared: {removed} file(s)")
        except Exception as exc:
            self.status_updated.emit(f"Clear remote cache failed: {exc}")

    def _on_remote_file_detected(self, source_path: str):
        message = "This file appears to be in a cloud or network folder. Copying to local cache before processing..."
        self.status_updated.emit(message)
        try:
            self._add_fitting_message(f"{message} {source_path}", "INFO")
        except Exception:
            pass

    def _on_remote_copy_started(self, source_path: str, target_path: str):
        self.status_updated.emit(f"Copying remote file to cache: {os.path.basename(source_path)}")

    def _on_remote_copy_finished(self, source_path: str, target_path: str):
        self.status_updated.emit(f"Remote file cached: {os.path.basename(source_path)}")

    def _on_remote_load_started(self, source_path: str):
        self.status_updated.emit(f"Load started: {os.path.basename(source_path)}")

    def _on_remote_load_finished(self, source_path: str):
        self.status_updated.emit(f"Load finished: {os.path.basename(source_path)}")

    def _supported_folder_image_extensions(self, file_path=""):
        suffix = os.path.splitext(file_path or "")[1].lower()
        if suffix == ".nxs":
            return (".nxs",)
        if suffix in {".tif", ".tiff"}:
            return (".tif", ".tiff")
        return (".cbf",)

    def _folder_image_cache_key(self, file_path):
        normalized = normalize_path(file_path)
        folder = (
            normalized
            if not os.path.splitext(normalized)[1]
            else normalize_path(os.path.dirname(normalized))
        )
        extensions = self._supported_folder_image_extensions(normalized)
        return folder if extensions == (".cbf",) else (folder, extensions)

    def _navigation_file_key(self, file_path):
        """Map every P03 module member to the first file of its logical NXS source."""
        normalized = normalize_path(file_path)
        if os.path.splitext(normalized)[1].lower() == ".nxs":
            try:
                info = self.fitting_view_model.storage.inspect_scattering_sequence(Path(normalized))
                normalized = normalize_path(str(info.logical_path))
            except Exception:
                pass
        return os.path.normcase(os.path.abspath(normalized))

    def _logical_navigation_files(self, files):
        logical_files = []
        seen = set()
        for file_path in files or []:
            normalized = normalize_path(file_path)
            key = self._navigation_file_key(normalized)
            if key in seen:
                continue
            seen.add(key)
            logical_file = normalized
            if os.path.splitext(normalized)[1].lower() == ".nxs":
                try:
                    info = self.fitting_view_model.storage.inspect_scattering_sequence(
                        Path(normalized)
                    )
                    logical_file = normalize_path(str(info.logical_path))
                except Exception:
                    pass
            logical_files.append(logical_file)
        return logical_files

    def _set_nxs_frame_state(self, file_path, frame_index=0):
        """Update NXS frame navigation only; detector geometry is intentionally untouched."""
        if os.path.splitext(file_path or "")[1].lower() != ".nxs":
            self._nxs_frame_index = 0
            self._nxs_frame_count = 1
            self.current_parameters.pop("nxs_frame_index", None)
            return
        try:
            info = self.fitting_view_model.storage.inspect_scattering_sequence(Path(file_path))
            frame_count = max(1, int(info.frame_count))
        except Exception as exc:
            frame_count = 1
            self.status_updated.emit(f"Could not inspect NXS frame count: {exc}")
        self._nxs_frame_count = frame_count
        self._nxs_frame_index = max(0, min(int(frame_index or 0), frame_count - 1))
        self.current_parameters["nxs_frame_index"] = self._nxs_frame_index

    def _nxs_uses_internal_frames(self, file_path):
        """Only stitched/module NXS sources expose frames to GISAXS navigation."""
        if os.path.splitext(file_path or "")[1].lower() != ".nxs":
            return False
        try:
            info = self.fitting_view_model.storage.inspect_scattering_sequence(Path(file_path))
            return bool(info.uses_internal_frames)
        except Exception:
            return False

    def _ordinary_stack_sequence(self, file_path):
        """Return ordinary detector files from the selected file to the series end."""
        suffix = os.path.splitext(file_path or "")[1].lower()
        if suffix == ".cbf":
            extensions = {".cbf"}
        elif suffix in {".tif", ".tiff"}:
            extensions = {".tif", ".tiff"}
        else:
            return []
        cached = self._folder_image_scan_cache.get(self._folder_image_cache_key(file_path))
        if cached is not None:
            paths = [
                normalize_path(path)
                for path in cached
                if os.path.splitext(path)[1].lower() in extensions
            ]
        else:
            folder = os.path.dirname(file_path)
            try:
                paths = [
                    normalize_path(entry.path)
                    for entry in os.scandir(folder)
                    if entry.is_file() and os.path.splitext(entry.name)[1].lower() in extensions
                ]
            except Exception:
                paths = []
        paths.sort(key=self._natural_sort_key)
        current_key = os.path.normcase(os.path.abspath(file_path))
        keys = [os.path.normcase(os.path.abspath(path)) for path in paths]
        try:
            return paths[keys.index(current_key) :]
        except ValueError:
            return [normalize_path(file_path)] if file_path else []

    def _maximum_stack_count(self, file_path=None):
        file_path = file_path or self.current_parameters.get("imported_gisaxs_file", "")
        suffix = os.path.splitext(file_path or "")[1].lower()
        if suffix == ".nxs":
            return max(1, int(self._nxs_frame_count) - int(self._nxs_frame_index))
        if suffix in {".cbf", ".tif", ".tiff"}:
            return max(1, len(self._ordinary_stack_sequence(file_path)))
        return 1

    def _clamp_stack_count(self, requested=None, notify=True):
        """Clamp Stack mode to the number of images available from the current start."""
        if getattr(self, "load_mode", "Single") != "Stack":
            return 1
        if requested is None:
            requested = self.current_parameters.get("stack_count", 1)
        try:
            requested = max(1, int(requested))
        except Exception:
            requested = 1
        maximum = self._maximum_stack_count()
        clamped = min(requested, maximum)
        self.current_parameters["stack_count"] = clamped
        stack_edit = getattr(self.ui, "gisaxsInputStackValue", None)
        if stack_edit is not None and stack_edit.text().strip() != str(clamped):
            stack_edit.setText(str(clamped))
        if notify and clamped != requested:
            self.status_updated.emit(
                f"Stack count adjusted from {requested} to maximum available {maximum}"
            )
        return clamped

    def _natural_sort_key(self, path):
        name = os.path.basename(path)
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]

    def _scan_folder_images_for_file(self, file_path):
        """Scan the selected file's folder and update navigation state."""
        try:
            file_path = normalize_path(file_path)
            if not file_path:
                self._folder_image_files = []
                self._folder_image_index = -1
                self._update_folder_navigation_buttons()
                return

            cached = self._folder_image_scan_cache.get(self._folder_image_cache_key(file_path))
            if cached:
                self._apply_folder_image_scan_result(file_path, cached)
                return

            worker = getattr(self, "_folder_image_scan_worker", None)
            if worker is not None and worker.isRunning():
                try:
                    worker.requestInterruption()
                except Exception:
                    pass

            self.status_updated.emit("Scanning image folder...")
            worker = FolderImageScanWorker(
                file_path,
                self._supported_folder_image_extensions(file_path),
                max_files=5000,
                fitting_view_model=self.fitting_view_model,
            )
            worker.status_updated.connect(self.status_updated.emit)
            worker.scan_finished.connect(self._on_folder_image_scan_finished)
            worker.error_occurred.connect(self._on_folder_image_scan_error)
            worker.finished.connect(lambda: setattr(self, "_folder_image_scan_worker", None))
            self._folder_image_scan_worker = worker
            worker.start()
        except Exception as e:
            self._folder_image_files = []
            self._folder_image_index = -1
            self._update_folder_navigation_buttons()
            self.status_updated.emit(f"Folder scan failed: {str(e)}")

    def _apply_folder_image_scan_result(self, file_path: str, files: list):
        try:
            file_path = normalize_path(file_path)
            current_norm = self._navigation_file_key(file_path)
            files = self._logical_navigation_files(files)
            self._folder_image_files = files
            norm_files = [self._navigation_file_key(p) for p in files]
            self._folder_image_index = (
                norm_files.index(current_norm) if current_norm in norm_files else -1
            )
            if self._folder_image_index < 0 and files:
                self.status_updated.emit("Current image is not in the scanned folder list")
            self._update_folder_navigation_buttons()
        except Exception as exc:
            self.status_updated.emit(f"Folder scan apply failed: {exc}")

    def _on_folder_image_scan_finished(self, file_path: str, files: list):
        try:
            self._folder_image_scan_cache[self._folder_image_cache_key(file_path)] = [
                normalize_path(p) for p in files
            ]
            self._apply_folder_image_scan_result(file_path, files)
            self.status_updated.emit(f"Folder scan complete: {len(files)} image file(s)")
        except Exception as exc:
            self.status_updated.emit(f"Folder scan result failed: {exc}")

    def _on_folder_image_scan_error(self, file_path: str, message: str):
        self._folder_image_files = []
        self._folder_image_index = -1
        self._update_folder_navigation_buttons()
        self.status_updated.emit(f"Folder scan failed: {message}")

    def _update_folder_navigation_buttons(self):
        try:
            count = len(self._folder_image_files)
            index = self._folder_image_index
            current_file = self.current_parameters.get("imported_gisaxs_file", "")
            uses_internal_navigation = self._nxs_uses_internal_frames(current_file)
            has_previous = (uses_internal_navigation and self._nxs_frame_index > 0) or (
                count > 1 and index > 0
            )
            has_next = (
                uses_internal_navigation and self._nxs_frame_index < self._nxs_frame_count - 1
            ) or (count > 1 and 0 <= index < count - 1)
            if self._previous_image_button is not None:
                self._previous_image_button.setEnabled(has_previous)
                self._previous_image_button.setToolTip(
                    "Previous" if has_previous else "No previous image"
                )
            if self._next_image_button is not None:
                self._next_image_button.setEnabled(has_next)
                self._next_image_button.setToolTip("Next" if has_next else "No next image")
            label = getattr(self, "_image_position_label", None)
            if label is not None:
                if uses_internal_navigation:
                    label.setText(f"{self._nxs_frame_index + 1} / {self._nxs_frame_count}")
                elif count and index >= 0:
                    label.setText(f"{index + 1} / {count}")
                else:
                    label.setText("— / —")
        except Exception:
            pass

    def _show_previous_folder_image(self):
        self._show_folder_image_at_offset(-1)

    def _show_next_folder_image(self):
        self._show_folder_image_at_offset(1)

    def _show_folder_image_at_offset(self, offset):
        try:
            current_file = self.current_parameters.get("imported_gisaxs_file", "")
            if os.path.splitext(current_file)[
                1
            ].lower() == ".nxs" and self._nxs_uses_internal_frames(current_file):
                target_frame = self._nxs_frame_index + offset
                if 0 <= target_frame < self._nxs_frame_count:
                    self._nxs_frame_index = target_frame
                    self.current_parameters["nxs_frame_index"] = target_frame
                    self._update_stack_display()
                    self._update_folder_navigation_buttons()
                    self.parameters_changed.emit(self.current_parameters)
                    self.status_updated.emit(
                        f"Current NXS frame: {target_frame + 1}/{self._nxs_frame_count}"
                    )
                    self._show_image()
                    return

            if not self._folder_image_files:
                self.status_updated.emit("No previous image" if offset < 0 else "No next image")
                self._update_folder_navigation_buttons()
                return

            if self._folder_image_index < 0 and current_file:
                self._scan_folder_images_for_file(current_file)
                self.status_updated.emit(
                    "Image list is still scanning; try navigation again in a moment"
                )
                return

            target_index = self._folder_image_index + offset
            if target_index < 0:
                self.status_updated.emit("No previous image")
                self._update_folder_navigation_buttons()
                return
            if target_index >= len(self._folder_image_files):
                self.status_updated.emit("No next image")
                self._update_folder_navigation_buttons()
                return

            target_file = self._folder_image_files[target_index]
            target_frame = 0
            if (
                offset < 0
                and os.path.splitext(target_file)[1].lower() == ".nxs"
                and self._nxs_uses_internal_frames(target_file)
            ):
                try:
                    info = self.fitting_view_model.storage.inspect_scattering_sequence(
                        Path(target_file)
                    )
                    target_frame = max(0, int(info.frame_count) - 1)
                except Exception:
                    target_frame = 0
            self._select_folder_image(target_file, frame_index=target_frame)
        except Exception as e:
            self.status_updated.emit(f"Image navigation failed: {str(e)}")

    def _select_folder_image(self, file_path, frame_index=0):
        try:
            file_path = normalize_path(file_path)
            if not self.fitting_view_model.storage.is_remote_source(
                file_path
            ) and not os.path.exists(file_path):
                QMessageBox.warning(
                    self.main_window, "File Error", f"File does not exist:\n{file_path}"
                )
                self._scan_folder_images_for_file(
                    self.current_parameters.get("imported_gisaxs_file", "")
                )
                return

            self.current_parameters["imported_gisaxs_file"] = file_path
            self._set_nxs_frame_state(file_path, frame_index)
            if hasattr(self.ui, "gisaxsInputImportButtonValue"):
                self.ui.gisaxsInputImportButtonValue.setText(os.path.basename(file_path))

            cached = self._folder_image_scan_cache.get(self._folder_image_cache_key(file_path))
            if cached:
                self._apply_folder_image_scan_result(file_path, cached)
            else:
                self._scan_folder_images_for_file(file_path)
            self._update_stack_display()
            self.parameters_changed.emit(self.current_parameters)
            if hasattr(self.parent, "save_current_session"):
                self.parent.save_current_session()

            self.status_updated.emit(f"Current image: {os.path.basename(file_path)}")
            self._show_image()
        except Exception as e:
            QMessageBox.warning(
                self.main_window, "Image Navigation Error", f"Failed to load image:\n{str(e)}"
            )
