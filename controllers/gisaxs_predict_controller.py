"""GISAXS Predict controller responsible for displaying GISAXS data."""

from __future__ import annotations

import os
import re
import json
import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, Qt, QSignalBlocker, QRectF
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QGraphicsScene

try:  # matplotlib is optional for colormap rendering
    from matplotlib import cm as mpl_cm
except Exception:  # pragma: no cover - optional dependency
    mpl_cm = None

from core.global_params import global_params
from .fitting_controller import AsyncImageLoader, is_matplotlib_available, is_fabio_available


class GisaxsPredictController(QObject):
    """GISAXS prediction controller handling data import, display, and prediction."""

    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    parameters_changed = pyqtSignal(dict)
    prediction_completed = pyqtSignal(dict)

    _DEFAULT_COLORMAPS = ["viridis", "plasma", "magma", "cividis", "gray"]

    def __init__(self, ui, parent=None) -> None:
        super().__init__(parent)
        self.ui = ui
        self.parent = parent
        self.main_window = parent.parent if hasattr(parent, "parent") else None

        self.current_parameters: Dict[str, Optional[str]] = {}
        self.prediction_results: Dict[str, object] = {}

        # 初始化运行时状态
        self._initialized = False
        self._ui_updating = False
        self._synchronizing = False

        self._graphics_scene: Optional[QGraphicsScene] = None
        self._current_pixmap: Optional[QPixmap] = None
        self._current_image: Optional[np.ndarray] = None
        self._view_zoom_steps = 0

        self._index_to_file: Dict[int, str] = {}
        self._available_indices: List[int] = []
        self._sequence_indices: List[int] = []
        self._current_file_index: Optional[int] = None
        self._folder_entries: List[Tuple[str, int]] = []

        self._load_request_seq = 0
        self._latest_display_request = 0
        self._active_loaders: Dict[int, AsyncImageLoader] = {}
        self._pending_contexts: Dict[int, Dict[str, object]] = {}

        # 读取全局参数
        self._set_default_parameters()
        self._load_saved_parameters()

    # ------------------------------------------------------------------
    # 初始化 & UI
    # ------------------------------------------------------------------
    def _load_saved_parameters(self) -> None:
        try:
            saved = global_params.get_module_parameters("gisaxs_predict")
            if saved:
                self.current_parameters.update(saved)
        except Exception:
            pass

    def initialize(self) -> None:
        if self._initialized:
            return

        self._setup_display_resources()
        self._setup_connections()
        self._initialize_ui()
        self._initialized = True

    def _setup_display_resources(self) -> None:
        view = getattr(self.ui, "gisaxsImageGraphicsView", None)
        if view is None:
            return
        self._graphics_scene = QGraphicsScene(view)
        view.setScene(self._graphics_scene)
        view.setTransformationAnchor(view.AnchorUnderMouse)
        view.setDragMode(view.ScrollHandDrag)

        combo = getattr(self.ui, "gisaxsImageColormapCombox", None)
        if combo is not None and combo.count() == 0:
            combo.addItems(self._DEFAULT_COLORMAPS)

    def _initialize_ui(self) -> None:
        self._ui_updating = True
        try:
            framework_combo = getattr(self.ui, "gisaxsPredictFrameworkCombox", None)
            if framework_combo is not None:
                idx = framework_combo.findText(self.current_parameters.get("framework", ""))
                framework_combo.setCurrentIndex(idx if idx >= 0 else 0)

            mode = self.current_parameters.get("mode", "single_file")
            single_btn = getattr(self.ui, "gisaxsPredictSingleFileRadioButton", None)
            multi_btn = getattr(self.ui, "gisaxsPredictMultiFilesRadioButton", None)
            if single_btn is not None and multi_btn is not None:
                if mode == "multi_files":
                    multi_btn.setChecked(True)
                else:
                    single_btn.setChecked(True)

            self._set_line_edit("gisaxsPredictChooseGisaxsFileValue", os.path.basename(self.current_parameters.get("input_file", "")))
            self._set_line_edit("gisaxsPredictChooseFolderValue", self.current_parameters.get("input_folder", ""))
            self._set_line_edit("gisaxsPredictExportFolderValue", self.current_parameters.get("export_path", ""))

            stack_text = self.current_parameters.get("stack_value", "1")
            if mode == "multi_files":
                stack_text = self.current_parameters.get("range_value", "") or stack_text
            self._set_line_edit("gisaxsPredictStackValue", stack_text)
            self._set_line_edit("gisaxsImageShowingValue", self.current_parameters.get("showing_value", ""))

            auto_scale = bool(self.current_parameters.get("auto_scale", True))
            self._set_checkbox("gisaxsImageAutoScaleCheckBox", auto_scale)
            self._set_double_spin("gisaxsImageVminValue", self.current_parameters.get("vmin"))
            self._set_double_spin("gisaxsImageVmaxValue", self.current_parameters.get("vmax"))

            colormap = self.current_parameters.get("colormap") or self._DEFAULT_COLORMAPS[0]
            self._set_combobox_text("gisaxsImageColormapCombox", colormap)

            self._update_mode_controls(mode)

        finally:
            self._ui_updating = False

    def _setup_connections(self) -> None:
        btn = getattr(self.ui, "gisaxsPredictChooseFolderButton", None)
        if btn:
            btn.clicked.connect(self._choose_gisaxs_folder)

        btn = getattr(self.ui, "gisaxsPredictChooseGisaxsFileButton", None)
        if btn:
            btn.clicked.connect(self._choose_gisaxs_file)

        btn = getattr(self.ui, "gisaxsPredictExportFolderButton", None)
        if btn:
            btn.clicked.connect(self._choose_export_folder)

        btn = getattr(self.ui, "gisaxsPredictPredictButton", None)
        if btn:
            btn.clicked.connect(self._run_gisaxs_predict)

        single_btn = getattr(self.ui, "gisaxsPredictSingleFileRadioButton", None)
        multi_btn = getattr(self.ui, "gisaxsPredictMultiFilesRadioButton", None)
        if single_btn:
            single_btn.toggled.connect(self._on_mode_changed)
        if multi_btn:
            multi_btn.toggled.connect(self._on_mode_changed)

        self._connect_line_edit("gisaxsPredictChooseGisaxsFileValue", self._handle_file_line_edit_committed)
        self._connect_line_edit("gisaxsPredictStackValue", self._on_stack_field_committed)
        self._connect_line_edit("gisaxsImageShowingValue", self._on_showing_value_committed)

        cb = getattr(self.ui, "gisaxsImageAutoScaleCheckBox", None)
        if cb:
            cb.toggled.connect(self._on_auto_scale_toggled)

        btn = getattr(self.ui, "gisaxsImageAutoScaleResetButton", None)
        if btn:
            btn.clicked.connect(self._on_auto_scale_reset)

        self._connect_double_spin("gisaxsImageVminValue", self._on_vmin_changed)
        self._connect_double_spin("gisaxsImageVmaxValue", self._on_vmax_changed)

        combo = getattr(self.ui, "gisaxsImageColormapCombox", None)
        if combo:
            combo.currentTextChanged.connect(self._on_colormap_changed)

        zoom_in = getattr(self.ui, "gisaxsImageZoomInButton", None)
        zoom_out = getattr(self.ui, "gisaxsImageZoomOutButton", None)
        zoom_reset = getattr(self.ui, "gisaxsImageZoomResetButton", None)
        if zoom_in:
            zoom_in.clicked.connect(self._zoom_in)
        if zoom_out:
            zoom_out.clicked.connect(self._zoom_out)
        if zoom_reset:
            zoom_reset.clicked.connect(self._zoom_reset)

    def _connect_line_edit(self, name: str, slot) -> None:
        widget = getattr(self.ui, name, None)
        if widget is None:
            return
        widget.returnPressed.connect(slot)
        widget.editingFinished.connect(slot)

    def _connect_double_spin(self, name: str, slot) -> None:
        widget = getattr(self.ui, name, None)
        if widget is None:
            return
        widget.editingFinished.connect(slot)

    # ------------------------------------------------------------------
    # 参数与持久化
    # ------------------------------------------------------------------
    def _set_default_parameters(self) -> None:
        self.current_parameters = {
            "framework": "tensorflow 2.15.0",
            "mode": "single_file",
            "input_file": "",
            "input_folder": "",
            "export_path": "",
            "stack_value": "1",
            "range_value": "",
            "showing_value": "",
            "auto_scale": True,
            "vmin": None,
            "vmax": None,
            "colormap": self._DEFAULT_COLORMAPS[0],
        }

    def _persist_parameters(self) -> None:
        if self._synchronizing:
            return
        self._synchronizing = True
        try:
            global_params.set_module_parameters("gisaxs_predict", dict(self.current_parameters))
            self.parameters_changed.emit(dict(self.current_parameters))
        finally:
            self._synchronizing = False

    def get_parameters(self) -> Dict[str, object]:
        return dict(self.current_parameters)

    def set_parameters(self, parameters: Dict[str, object]) -> None:
        if not parameters:
            return
        self.current_parameters.update(parameters)
        if self._initialized:
            self._initialize_ui()

    # ------------------------------------------------------------------
    # 文件与模式处理
    # ------------------------------------------------------------------
    def _choose_gisaxs_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self.main_window, "Select GISAXS Folder", "")
        if not folder:
            return
        self.current_parameters["input_folder"] = folder
        self._set_line_edit("gisaxsPredictChooseFolderValue", folder)
        self._scan_directory_for_cbf(folder)
        self._persist_parameters()

    def _choose_gisaxs_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Select GISAXS File",
            self.current_parameters.get("input_folder", ""),
            "GISAXS Files (*.cbf);;All Files (*)",
        )
        if file_path:
            self._handle_new_file_selection(file_path)

    def _choose_export_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self.main_window,
            "Select Export Folder",
            self.current_parameters.get("export_path", ""),
        )
        if not folder:
            return
        self.current_parameters["export_path"] = folder
        self._set_line_edit("gisaxsPredictExportFolderValue", folder)
        self._persist_parameters()
        self._append_status_message(f"Export folder selected: {folder}")

    def _handle_file_line_edit_committed(self) -> None:
        widget = getattr(self.ui, "gisaxsPredictChooseGisaxsFileValue", None)
        if not widget:
            return
        text = widget.text().strip()
        if not text:
            return
        if os.path.isabs(text) and os.path.exists(text):
            self._handle_new_file_selection(text)
            return

        folder = self.current_parameters.get("input_folder", "")
        candidate = os.path.join(folder, text) if folder else text
        if os.path.exists(candidate):
            self._handle_new_file_selection(candidate)
            return
        self._append_status_message(f"Unable to locate file: {text}", level="WARN")

    def _handle_new_file_selection(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            QMessageBox.warning(self.main_window, "File Not Found", file_path)
            return
        if not file_path.lower().endswith(".cbf"):
            QMessageBox.warning(self.main_window, "Unsupported Format", "Only CBF files are supported.")
            return

        folder = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        index = self._extract_index(base_name)

        self.current_parameters.update(
            {
                "input_file": file_path,
                "input_folder": folder,
                "showing_value": str(index or ""),
            }
        )

        self._set_line_edit("gisaxsPredictChooseGisaxsFileValue", base_name)
        self._set_line_edit("gisaxsPredictChooseFolderValue", folder)
        self._set_line_edit("gisaxsImageShowingValue", self.current_parameters.get("showing_value", ""))

        self._scan_directory_for_cbf(folder)
        if index is not None:
            self._current_file_index = index
        elif self._available_indices:
            self._current_file_index = self._available_indices[0]

        mode = self.current_parameters.get("mode", "single_file")
        if mode == "multi_files":
            default_range = f"{self._current_file_index}-{self._current_file_index}"
            self.current_parameters["range_value"] = default_range
            self._set_line_edit("gisaxsPredictStackValue", default_range)
        else:
            stack_text = self.current_parameters.get("stack_value", "1")
            self._set_line_edit("gisaxsPredictStackValue", stack_text or "1")

        self._update_range_tooltip()
        self._persist_parameters()
        self._trigger_data_reload()

    def _scan_directory_for_cbf(self, folder: str) -> None:
        entries: List[Tuple[str, int]] = []
        index_to_file: Dict[int, str] = {}

        try:
            for name in sorted(os.listdir(folder)):
                if not name.lower().endswith(".cbf"):
                    continue
                idx = self._extract_index(name)
                if idx is None:
                    continue
                full = os.path.join(folder, name)
                entries.append((name, idx))
                index_to_file[idx] = full

            if not entries:
                self._append_status_message("No numbered CBF files detected in the current folder", level="WARN")

            self._folder_entries = entries
            self._index_to_file = index_to_file
            self._available_indices = sorted(index_to_file.keys())
            self._update_range_tooltip()
        except Exception as exc:
            self._append_status_message(f"Failed to scan folder: {exc}", level="ERROR")

    def _extract_index(self, file_name: str) -> Optional[int]:
        match = re.search(r"(\d+)(?=\.cbf$)", file_name, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    def _update_range_tooltip(self) -> None:
        if not self._available_indices:
            tooltip = "No valid indices detected yet"
        else:
            tooltip = f"Available index range: {self._available_indices[0]} - {self._available_indices[-1]}"

        label = getattr(self.ui, "gisaxsPredictStackLabel", None)
        line_edit = getattr(self.ui, "gisaxsPredictStackValue", None)
        if label:
            label.setToolTip(tooltip)
        if line_edit:
            line_edit.setToolTip(tooltip)

    def _on_mode_changed(self) -> None:
        if self._ui_updating:
            return
        single_btn = getattr(self.ui, "gisaxsPredictSingleFileRadioButton", None)
        if single_btn is not None and single_btn.isChecked():
            self.current_parameters["mode"] = "single_file"
        else:
            self.current_parameters["mode"] = "multi_files"

        self._update_mode_controls(self.current_parameters["mode"])
        self._persist_parameters()
        self._trigger_data_reload()

    def _update_mode_controls(self, mode: str) -> None:
        label = getattr(self.ui, "gisaxsPredictStackLabel", None)
        stack_edit = getattr(self.ui, "gisaxsPredictStackValue", None)
        showing = getattr(self.ui, "gisaxsImageShowingValue", None)

        if label:
            label.setText("Range:" if mode == "multi_files" else "Stack:")
        if stack_edit:
            text = (
                self.current_parameters.get("range_value", "")
                if mode == "multi_files"
                else self.current_parameters.get("stack_value", "1")
            )
            self._set_line_edit("gisaxsPredictStackValue", text or ("1" if mode == "single_file" else ""))
        if showing:
            showing.setEnabled(mode == "multi_files")

    def _on_stack_field_committed(self) -> None:
        if self._ui_updating:
            return
        mode = self.current_parameters.get("mode", "single_file")
        text = self._get_line_edit_text("gisaxsPredictStackValue")
        if mode == "multi_files":
            self.current_parameters["range_value"] = text
            self._persist_parameters()
            self._trigger_data_reload()
            return

        try:
            count = max(1, int(text or "1"))
        except ValueError:
            count = 1
        self.current_parameters["stack_value"] = str(count)
        self._set_line_edit("gisaxsPredictStackValue", str(count))
        self._persist_parameters()
        self._trigger_data_reload()

    def _on_showing_value_committed(self) -> None:
        if self._ui_updating:
            return
        mode = self.current_parameters.get("mode", "single_file")
        if mode != "multi_files":
            return
        text = self._get_line_edit_text("gisaxsImageShowingValue")
        try:
            index = int(text)
        except ValueError:
            self._append_status_message("Showing Value must be a valid index", level="WARN")
            return
        if index not in self._index_to_file:
            self._append_status_message("Index is outside the available range", level="WARN")
            return
        self.current_parameters["showing_value"] = str(index)
        self._persist_parameters()
        self._start_image_loading(self._index_to_file[index], 1, {"mode": "multi_files", "index": index})

    def _parse_range_text(self, text: str) -> List[int]:
        match = re.match(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", text)
        if not match:
            return []
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if end < start:
            start, end = end, start
        return list(range(start, end + 1))

    def _trigger_data_reload(self) -> None:
        if self.current_parameters.get("mode", "single_file") == "single_file":
            self._load_single_stack()
        else:
            self._load_multi_sequence()

    def _load_single_stack(self) -> None:
        file_path = self.current_parameters.get("input_file")
        if not file_path:
            return
        try:
            stack = max(1, int(self.current_parameters.get("stack_value", "1")))
        except ValueError:
            stack = 1
            self.current_parameters["stack_value"] = "1"
        self._start_image_loading(file_path, stack, {"mode": "single_file", "stack": stack})

    def _load_multi_sequence(self) -> None:
        if not self._index_to_file:
            if self.current_parameters.get("input_folder"):
                self._scan_directory_for_cbf(self.current_parameters["input_folder"])
            if not self._index_to_file:
                return

        range_text = self.current_parameters.get("range_value") or self._get_line_edit_text("gisaxsPredictStackValue")
        indices = [idx for idx in self._parse_range_text(range_text) if idx in self._index_to_file]
        if not indices:
            if self._available_indices:
                indices = [self._available_indices[0]]
            else:
                self._append_status_message("No Multi File indices available", level="WARN")
                return

        self._sequence_indices = indices
        first = indices[0]
        self.current_parameters["range_value"] = range_text
        self.current_parameters["showing_value"] = str(first)
        self._set_line_edit("gisaxsImageShowingValue", str(first))
        self._persist_parameters()
        self._start_image_loading(self._index_to_file[first], 1, {"mode": "multi_files", "index": first})

    # ------------------------------------------------------------------
    # 图像加载与显示
    # ------------------------------------------------------------------
    def _start_image_loading(self, file_path: str, stack_count: int, context: Dict[str, object]) -> None:
        if not os.path.exists(file_path):
            QMessageBox.warning(self.main_window, "File Not Found", file_path)
            return
        if not is_fabio_available():
            QMessageBox.warning(
                self.main_window,
                "Missing Dependency",
                "Install the fabio package to read CBF files (pip install fabio)",
            )
            return

        self._load_request_seq += 1
        request_id = self._load_request_seq
        loader = AsyncImageLoader()
        self._active_loaders[request_id] = loader
        self._pending_contexts[request_id] = dict(context, file=file_path, stack=stack_count)

        loader.image_loaded.connect(lambda data, path, rid=request_id: self._on_image_loaded(rid, data, path))
        loader.progress_updated.connect(lambda progress, msg, rid=request_id: self._on_loader_progress(rid, progress, msg))
        loader.error_occurred.connect(lambda err, rid=request_id: self._on_loader_error(rid, err))
        loader.finished.connect(lambda rid=request_id: self._cleanup_loader(rid))

        loader.load_image(file_path, stack_count)
        self._latest_display_request = request_id
        self._append_status_message(f"Loading {os.path.basename(file_path)} (Stack={stack_count}) ...")

    def _on_loader_progress(self, request_id: int, progress: int, message: str) -> None:
        if request_id != self._latest_display_request:
            return
        self.status_updated.emit(f"Image loading {progress}%: {message}")
        self.progress_updated.emit(progress)

    def _on_loader_error(self, request_id: int, error: str) -> None:
        if request_id == self._latest_display_request:
            QMessageBox.critical(self.main_window, "Image Load Failed", error)
            self.status_updated.emit(error)
        self._cleanup_loader(request_id)

    def _cleanup_loader(self, request_id: int) -> None:
        loader = self._active_loaders.pop(request_id, None)
        if loader:
            loader.deleteLater()
        self._pending_contexts.pop(request_id, None)

    def _on_image_loaded(self, request_id: int, image_data: np.ndarray, file_path: str) -> None:
        context = self._pending_contexts.get(request_id)
        if context is None:
            return
        if request_id != self._latest_display_request:
            return

        self._current_image = image_data.astype(np.float32, copy=False)
        self._append_status_message(f"Image loaded: {os.path.basename(file_path)}")

        if context.get("mode") == "multi_files" and context.get("index") is not None:
            self.current_parameters["showing_value"] = str(context["index"])
            self._set_line_edit("gisaxsImageShowingValue", str(context["index"]))

        self._update_image_display()

    def _update_image_display(self) -> None:
        if self._current_image is None or self._graphics_scene is None:
            return

        auto_scale = bool(self.current_parameters.get("auto_scale", True))
        vmin = self.current_parameters.get("vmin")
        vmax = self.current_parameters.get("vmax")

        if auto_scale or vmin is None or vmax is None:
            vmin, vmax = self._auto_scale_values(self._current_image)
            self.current_parameters["vmin"] = vmin
            self.current_parameters["vmax"] = vmax
            self._set_double_spin("gisaxsImageVminValue", vmin)
            self._set_double_spin("gisaxsImageVmaxValue", vmax)

        pixmap = self._create_pixmap_from_array(
            self._current_image,
            vmin,
            vmax,
            self.current_parameters.get("colormap", self._DEFAULT_COLORMAPS[0]),
        )
        if pixmap is None:
            return

        self._graphics_scene.clear()
        self._graphics_scene.addPixmap(pixmap)
        self._graphics_scene.setSceneRect(QRectF(pixmap.rect()))
        self._current_pixmap = pixmap
        self._zoom_reset()

        cmap_name = self.current_parameters.get("colormap", "")
        self.status_updated.emit(f"Display complete (vmin={vmin:.3f}, vmax={vmax:.3f}, cmap={cmap_name})")
        self._persist_parameters()

    def _auto_scale_values(self, image: np.ndarray) -> Tuple[float, float]:
        finite = np.isfinite(image)
        if not np.any(finite):
            return 0.0, 1.0
        data = image[finite]
        vmin = float(np.percentile(data, 2))
        vmax = float(np.percentile(data, 98))
        if vmin == vmax:
            vmax = vmin + 1.0
        return vmin, vmax

    def _create_pixmap_from_array(self, image: np.ndarray, vmin: float, vmax: float, cmap_name: str) -> Optional[QPixmap]:
        data = np.clip(image, vmin, vmax)
        norm = (data - vmin) / max(vmax - vmin, 1e-9)
        norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)

        if mpl_cm is None or not is_matplotlib_available():
            gray = (norm * 255).astype(np.uint8)
            rgba = np.dstack([gray, gray, gray, np.full_like(gray, 255)])
        else:
            cmap = mpl_cm.get_cmap(cmap_name or self._DEFAULT_COLORMAPS[0])
            rgba = (cmap(norm) * 255).astype(np.uint8)

        height, width = rgba.shape[:2]
        bytes_per_line = rgba.strides[0]
        image_q = QImage(rgba.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
        return QPixmap.fromImage(image_q.copy())

    # ------------------------------------------------------------------
    # 显示控制
    # ------------------------------------------------------------------
    def _zoom_in(self) -> None:
        self._view_zoom_steps += 1
        self._apply_zoom()

    def _zoom_out(self) -> None:
        self._view_zoom_steps -= 1
        self._apply_zoom()

    def _zoom_reset(self) -> None:
        self._view_zoom_steps = 0
        self._apply_zoom(reset=True)

    def _apply_zoom(self, reset: bool = False) -> None:
        view = getattr(self.ui, "gisaxsImageGraphicsView", None)
        if view is None or self._current_pixmap is None:
            return
        view.resetTransform()
        if reset:
            view.fitInView(QRectF(self._current_pixmap.rect()), Qt.KeepAspectRatio)
            return
        factor = 1.15 ** self._view_zoom_steps
        view.scale(factor, factor)

    def _on_auto_scale_toggled(self) -> None:
        if self._ui_updating:
            return
        auto = getattr(self.ui, "gisaxsImageAutoScaleCheckBox", None)
        checked = bool(auto.isChecked()) if auto else True
        self.current_parameters["auto_scale"] = checked
        self._persist_parameters()
        if checked:
            self._update_image_display()

    def _on_auto_scale_reset(self) -> None:
        self._update_image_display()

    def _on_vmin_changed(self) -> None:
        if self._ui_updating or self.current_parameters.get("auto_scale"):
            return
        value = self._get_double_spin_value("gisaxsImageVminValue")
        if value is None:
            return
        self.current_parameters["vmin"] = value
        self._update_image_display()

    def _on_vmax_changed(self) -> None:
        if self._ui_updating or self.current_parameters.get("auto_scale"):
            return
        value = self._get_double_spin_value("gisaxsImageVmaxValue")
        if value is None:
            return
        self.current_parameters["vmax"] = value
        self._update_image_display()

    def _on_colormap_changed(self, text: str) -> None:
        if self._ui_updating:
            return
        self.current_parameters["colormap"] = text or self._DEFAULT_COLORMAPS[0]
        self._update_image_display()

    # ------------------------------------------------------------------
    # 预测逻辑（当前为占位实现，无TensorFlow依赖）
    # ------------------------------------------------------------------
    def _run_gisaxs_predict(self) -> None:
        self._execute_prediction()

    def _execute_prediction(self) -> None:
        self._update_parameters_from_ui()
        if not self._validate_parameters():
            return
        try:
            self.status_updated.emit("Starting GISAXS prediction...")
            self.progress_updated.emit(0)
            mode = self.current_parameters.get("mode", "single_file")
            results = self._predict_single_file() if mode == "single_file" else self._predict_multi_files()
            if results:
                self._save_results(results)
                self.prediction_completed.emit(results)
                self.progress_updated.emit(100)
                self.status_updated.emit("GISAXS prediction finished!")
        except Exception as exc:  # pragma: no cover - runtime safety
            QMessageBox.critical(self.main_window, "Prediction Error", str(exc))
            self.status_updated.emit(f"GISAXS prediction error: {exc}")

    def _update_parameters_from_ui(self) -> None:
        combo = getattr(self.ui, "gisaxsPredictFrameworkCombox", None)
        if combo is not None:
            self.current_parameters["framework"] = combo.currentText()
        export_edit = getattr(self.ui, "gisaxsPredictExportFolderValue", None)
        if export_edit is not None:
            text = export_edit.text().strip()
            if text:
                self.current_parameters["export_path"] = text

    def _validate_parameters(self) -> bool:
        export_path = self.current_parameters.get("export_path")
        if not export_path:
            QMessageBox.warning(self.main_window, "Invalid Parameters", "Please choose an export folder")
            return False
        if not os.path.exists(export_path):
            QMessageBox.warning(self.main_window, "Path Error", f"Export path does not exist: {export_path}")
            return False

        mode = self.current_parameters.get("mode", "single_file")
        if mode == "single_file":
            file_path = self.current_parameters.get("input_file")
            if not file_path or not os.path.exists(file_path):
                QMessageBox.warning(self.main_window, "Invalid Parameters", "Please select a valid input file")
                return False
        else:
            folder = self.current_parameters.get("input_folder")
            if not folder or not os.path.exists(folder):
                QMessageBox.warning(self.main_window, "Invalid Parameters", "Please select a valid folder")
                return False
        return True

    def _predict_single_file(self) -> Optional[Dict[str, object]]:
        file_path = self.current_parameters.get("input_file")
        if not file_path:
            return None
        self.status_updated.emit(f"Processing file: {os.path.basename(file_path)}")
        self.progress_updated.emit(25)
        results = {
            "file": file_path,
            "predictions": [],
            "confidence": 0.95,
            "processing_time": 1.5,
        }
        self.progress_updated.emit(75)
        return results

    def _predict_multi_files(self) -> Optional[Dict[str, object]]:
        folder = self.current_parameters.get("input_folder")
        if not folder:
            return None
        files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(".cbf")]
        if not files:
            self._append_status_message("No CBF files available for prediction", level="WARN")
            return None
        results = {
            "folder": folder,
            "total_files": len(files),
            "processed_files": len(files),
            "predictions": [],
            "average_confidence": 0.92,
            "total_processing_time": len(files) * 1.2,
        }
        self.progress_updated.emit(80)
        return results

    def _save_results(self, results: Dict[str, object]) -> None:
        export_path = self.current_parameters.get("export_path")
        if not export_path:
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(export_path, f"gisaxs_prediction_results_{timestamp}.json")
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        self._append_status_message(f"Results saved: {file_path}")

    # ------------------------------------------------------------------
    # UI 辅助方法
    # ------------------------------------------------------------------
    def _set_line_edit(self, name: str, text: Optional[str]) -> None:
        widget = getattr(self.ui, name, None)
        if widget is None:
            return
        blocker = QSignalBlocker(widget)
        widget.setText(text or "")
        del blocker

    def _get_line_edit_text(self, name: str) -> str:
        widget = getattr(self.ui, name, None)
        return widget.text().strip() if widget else ""

    def _set_checkbox(self, name: str, checked: bool) -> None:
        widget = getattr(self.ui, name, None)
        if widget is None:
            return
        blocker = QSignalBlocker(widget)
        widget.setChecked(bool(checked))
        del blocker

    def _set_double_spin(self, name: str, value: Optional[float]) -> None:
        widget = getattr(self.ui, name, None)
        if widget is None or value is None:
            return
        blocker = QSignalBlocker(widget)
        widget.setValue(float(value))
        del blocker

    def _set_combobox_text(self, name: str, text: str) -> None:
        widget = getattr(self.ui, name, None)
        if widget is None or text is None:
            return
        blocker = QSignalBlocker(widget)
        index = widget.findText(text)
        widget.setCurrentIndex(index if index >= 0 else 0)
        del blocker

    def _get_double_spin_value(self, name: str) -> Optional[float]:
        widget = getattr(self.ui, name, None)
        return float(widget.value()) if widget is not None else None

    def _append_status_message(self, message: str, level: str = "INFO") -> None:
        self.status_updated.emit(message)
        browser = getattr(self.ui, "predictStatusTextBrowser", None)
        if browser is not None:
            browser.append(f"[{level}] {message}")
