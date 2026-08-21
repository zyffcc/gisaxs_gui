"""Image Loading coordination for Prediction."""

from __future__ import annotations

import os

import datetime


from pathlib import Path

from typing import Dict, List, Optional, Tuple

import numpy as np

from PyQt5.QtCore import QRectF

from PyQt5.QtGui import QImage, QPixmap

from PyQt5.QtWidgets import (
    QMessageBox,
)


from src.gimap.features.prediction.presentation.image_worker import PredictionImageLoader


from ..dependency_detection import dependency_available as _dependency_available


class ImageLoadingMixin:
    """Own image loading presentation behavior."""

    def _start_image_loading(
        self, file_path: str, stack_count: int, context: Dict[str, object]
    ) -> None:
        if not os.path.exists(file_path):
            QMessageBox.warning(self.main_window, "File Not Found", file_path)
            return
        if not _dependency_available("fabio"):
            QMessageBox.warning(
                self.main_window,
                "Missing Dependency",
                "Install the fabio package to read CBF files (pip install fabio)",
            )
            return

        self._load_request_seq += 1
        request_id = self._load_request_seq
        loader = PredictionImageLoader(self.prediction_view_model)
        self._active_loaders[request_id] = loader

        # Precompute stack file names for logging so we can show per-file progress
        stack_files: List[str] = []
        if stack_count > 1:
            try:
                directory = os.path.dirname(file_path)
                start_name = os.path.basename(file_path)
                cbf_files = [
                    path.name
                    for path in self.prediction_view_model.files.discover_files(
                        Path(directory), (".cbf",)
                    )
                ]
                start_idx = cbf_files.index(start_name)
                stack_files = cbf_files[start_idx : start_idx + stack_count]
            except Exception:
                stack_files = []

        self._pending_contexts[request_id] = {
            **context,
            "file": file_path,
            "stack": stack_count,
            "stack_files": stack_files,
            "_last_progress_file": None,
        }

        loader.image_loaded.connect(
            lambda data, path, rid=request_id: self._on_image_loaded(rid, data, path)
        )
        loader.progress_updated.connect(
            lambda progress, msg, rid=request_id: self._on_loader_progress(rid, progress, msg)
        )
        loader.error_occurred.connect(lambda err, rid=request_id: self._on_loader_error(rid, err))
        loader.finished.connect(lambda rid=request_id: self._cleanup_loader(rid))

        loader.load_image(file_path, stack_count)
        self._latest_display_request = request_id
        self._append_status_message(
            f"Loading {os.path.basename(file_path)} (Stack={stack_count}) ..."
        )

    def _on_loader_progress(self, request_id: int, progress: int, message: str) -> None:
        if request_id != self._latest_display_request:
            return

        context = self._pending_contexts.get(request_id, {})
        stack_files = context.get("stack_files") or []
        if stack_files and "Processing file" in message:
            # Example message: "Processing file 2/5: foo.cbf"
            parts = message.split(":", 1)
            fname = parts[1].strip() if len(parts) == 2 else ""
            last_file = context.get("_last_progress_file")
            if fname and fname != last_file:
                self._append_status_message(f"Loading {fname} ...")
                context["_last_progress_file"] = fname
                self._pending_contexts[request_id] = context
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
        self._current_image_path = file_path

        stack_files = context.get("stack_files") or []
        if context.get("stack", 1) and context.get("stack", 1) > 1 and stack_files:
            first = stack_files[0]
            last = stack_files[-1]
            self._append_status_message(f"Image loaded: {first} - {last}")
        else:
            self._append_status_message(f"Image loaded: {os.path.basename(file_path)}")

        if context.get("mode") == "multi_files" and context.get("index") is not None:
            self.current_parameters["showing_value"] = str(context["index"])
            self._set_line_edit("gisaxsImageShowingValue", str(context["index"]))

        self._update_image_display()
        self._set_predict_main_tab("input")
        self._refresh_predict_readiness()

    def _maybe_log_scale(self, image: np.ndarray, enabled: bool) -> np.ndarray:
        if not enabled:
            return image
        img = np.array(image, dtype=np.float32, copy=False)
        finite = np.isfinite(img)
        if not finite.any():
            return img
        positives = img[finite & (img > 0)]
        floor = float(np.min(positives)) if positives.size else 1e-6
        floor = max(floor, 1e-6)
        return np.log10(np.maximum(img, floor))

    def _on_gisaxs_log_scale_toggled(self, checked: bool) -> None:
        if self._ui_updating:
            return
        self.current_parameters["gisaxs_log_scale"] = bool(checked)
        self._persist_parameters()
        self._update_image_display()

    def _export_gisaxs_image(self) -> None:
        if self._current_pixmap is None:
            QMessageBox.information(
                self.main_window,
                "Export",
                "Import a detector image before exporting the input preview.",
            )
            self._append_status_message("No input preview to export", level="WARN")
            return
        export_path = self._prompt_export_folder("Save GISAXS Image To")
        if not export_path:
            return
        if not os.path.isdir(export_path):
            QMessageBox.warning(
                self.main_window, "Export Path", f"Export folder not found: {export_path}"
            )
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(export_path, f"gisaxs_{timestamp}.jpg")
        try:
            if not self._current_pixmap.save(file_path, "JPG"):
                raise IOError("Save returned False")
            self._append_status_message(f"GISAXS image exported: {file_path}")
        except Exception as exc:
            self._append_status_message(f"Export failed: {exc}", level="ERROR")

    def _update_image_display(self) -> None:
        if self._current_image is None or self._graphics_scene is None:
            return

        display_img = self._maybe_log_scale(
            self._current_image, bool(self.current_parameters.get("gisaxs_log_scale", False))
        )

        auto_scale = bool(self.current_parameters.get("auto_scale", True))
        vmin = self.current_parameters.get("vmin")
        vmax = self.current_parameters.get("vmax")

        if auto_scale or vmin is None or vmax is None:
            vmin, vmax = self._auto_scale_percentiles(display_img, 0.5, 99.5)
            self.current_parameters["vmin"] = vmin
            self.current_parameters["vmax"] = vmax
            self._set_double_spin("gisaxsImageVminValue", vmin)
            self._set_double_spin("gisaxsImageVmaxValue", vmax)
        self._set_checkbox("gisaxsImageAutoScaleCheckBox", auto_scale)

        pixmap = self._create_pixmap_from_array(
            display_img,
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
        self.status_updated.emit(
            f"Display complete (vmin={vmin:.3f}, vmax={vmax:.3f}, cmap={cmap_name})"
        )
        self._persist_parameters()

    def _auto_scale_values(self, image: np.ndarray) -> Tuple[float, float]:
        finite = np.isfinite(image)
        if not np.any(finite):
            return 0.0, 1.0
        data = image[finite]
        vmin = float(np.min(data))
        vmax = float(np.max(data))
        if vmin == vmax:
            vmax = vmin + 1.0
        return vmin, vmax

    def _auto_scale_percentiles(
        self, image: np.ndarray, low: float, high: float
    ) -> Tuple[float, float]:
        finite = np.isfinite(image)
        if not np.any(finite):
            return 0.0, 1.0
        data = image[finite]
        vmin = float(np.percentile(data, low))
        vmax = float(np.percentile(data, high))
        if vmin == vmax:
            vmax = vmin + 1.0
        return vmin, vmax

    def _create_pixmap_from_array(
        self, image: np.ndarray, vmin: float, vmax: float, cmap_name: str
    ) -> Optional[QPixmap]:
        data = np.clip(image, vmin, vmax)
        norm = (data - vmin) / max(vmax - vmin, 1e-9)
        norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)

        mpl_cm = self._get_mpl_cm()
        if mpl_cm is None:
            gray = (norm * 255).astype(np.uint8)
            rgba = np.dstack([gray, gray, gray, np.full_like(gray, 255)])
        else:
            cmap = mpl_cm.get_cmap(cmap_name or self._DEFAULT_COLORMAPS[0])
            rgba = (cmap(norm) * 255).astype(np.uint8)

        height, width = rgba.shape[:2]
        bytes_per_line = rgba.strides[0]
        image_q = QImage(rgba.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
        return QPixmap.fromImage(image_q.copy())

    def _get_mpl_cm(self):
        if not _dependency_available("matplotlib"):
            return None
        if self.__class__._mpl_cm is None:
            try:
                from matplotlib import cm as mpl_cm  # type: ignore

                self.__class__._mpl_cm = mpl_cm
            except Exception:
                self.__class__._mpl_cm = False
        return None if self.__class__._mpl_cm is False else self.__class__._mpl_cm
