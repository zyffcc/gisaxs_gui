"""Export input… and Export result… of 2D Prediction: a PNG named after the input, the data beside it with
the same stem, and a JSON record of what produced them; a toast with Open Folder when written."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.features.prediction.application import (
    PredictionArrayExportRequest,
    PredictionRecordRequest,
)
from src.gimap.shared.file_paths import normalize_path

IMAGE_KINDS = ("hr", "array", "steps")


class ExportFilesMixin:
    """Own the single-file exports of the input preview and of the prediction result."""

    # -- what is asked ---------------------------------------------------------------------------

    def _ask_save_path(self, title: str, suggested_name: str) -> str:
        """The PNG path the user chose ("" when cancelled); the dialog opens in the last export folder."""
        folder = self.current_parameters.get("export_path") or str(Path(self._export_source() or ".").parent)
        path, _ = QFileDialog.getSaveFileName(
            self.main_window, title, str(Path(folder) / suggested_name), tr("PNG image (*.png)")
        )
        return path or ""

    def _export_source(self) -> str:
        return str(self._current_image_path or self.current_parameters.get("input_file") or "")

    def _export_stem(self, suffix: str) -> str:
        source = self._export_source()
        return f"{Path(source).stem if source else 'prediction'}{suffix}"

    def _chosen_png(self, title: str, suffix: str) -> Optional[Path]:
        chosen = self._ask_save_path(title, f"{self._export_stem(suffix)}.png")
        if not chosen:
            return None
        path = Path(normalize_path(chosen))
        if path.suffix.lower() != ".png":
            path = path.with_name(path.name + ".png")
        self.current_parameters["export_path"] = str(path.parent)  # the next export opens here
        self._persist_parameters()
        return path

    # -- the two exports ---------------------------------------------------------------------------

    def _export_gisaxs_image(self) -> None:
        """Export input…: the detector frame as displayed (PNG) and its record."""
        if self._current_pixmap is None:
            QMessageBox.information(
                self.main_window, tr("Export"), tr("Import a detector image before exporting the input preview.")
            )
            self._append_status_message("No input preview to export", level="WARN")
            return
        path = self._chosen_png(tr("Save Input Preview"), "_input")
        if path is None:
            return
        written: List[Path] = []
        try:
            if not self._current_pixmap.save(str(path), "PNG"):
                raise OSError(trf("The image {name} could not be written.", name=path.name))
            written.append(path)
            written.append(self._write_export_record(path, written, kind="input preview"))
        except Exception as exc:  # noqa: BLE001 - every failure is shown, nothing is half-reported
            self._export_failed(exc)
            return
        self._export_written(written)

    def _on_predict_export_clicked(self) -> None:
        """Export result…: the shown result (PNG), its data (ASCII) and the record, all with one stem."""
        mode = self.current_parameters.get("mode", "single_file")
        if mode == "multi_files" and self._multifile_results_widget:
            self._multifile_results_widget.onExportClicked()  # the batch export dialog
            return
        if not self.prediction_results:
            QMessageBox.information(
                self.main_window, tr("Export"), tr("Run a prediction before exporting the current result.")
            )
            self._append_status_message("No prediction result to export", level="WARN")
            return
        kind = self._shown_result_kind()
        if kind is None:
            self._append_status_message("No prediction output to export", level="WARN")
            return
        if self._predict_pixmap is None:
            self._append_status_message("No predict view image to export", level="WARN")
            return
        path = self._chosen_png(tr("Save Prediction Result"), "_prediction")
        if path is None:
            return
        written: List[Path] = []
        try:
            if not self._predict_pixmap.save(str(path), "PNG"):
                raise OSError(trf("The image {name} could not be written.", name=path.name))
            written.append(path)
            data = self._export_result_data(kind, path.with_suffix(".txt"))
            if data is not None:
                written.append(data)
            written.append(self._write_export_record(path, written, kind="prediction", shown=self._shown_label(kind)))
        except Exception as exc:  # noqa: BLE001
            self._export_failed(exc)
            return
        self._export_written(written)

    def _shown_result_kind(self) -> Optional[str]:
        spec = None
        tabs = getattr(self, "_predict_tabs", None)
        try:
            if tabs is not None and 0 <= tabs.currentIndex() < len(self._predict_tab_specs):
                spec = self._predict_tab_specs[tabs.currentIndex()]
        except Exception:  # noqa: BLE001 - a tab widget being rebuilt
            spec = None
        if spec is None and self._predict_tab_specs:
            spec = self._predict_tab_specs[0]
        if spec is None:
            return None
        kind = self._predict_current_kind
        if kind is None and isinstance(spec, dict):
            kind = spec.get("kind")
        return str(kind or "view")

    def _shown_label(self, kind: str) -> str:
        """The exported view for the record; a preprocessing step also by its label ("steps: crop")."""
        if kind != "steps":
            return kind
        steps = getattr(self, "_step_snapshots", None) or []
        index = int(getattr(self, "_current_step_index", 0) or 0)
        label = steps[index].get("label") if 0 <= index < len(steps) and isinstance(steps[index], dict) else None
        return f"steps: {label}" if label else kind

    def _export_result_data(self, kind: str, path: Path) -> Optional[Path]:
        """The numbers of the shown result beside the image: a curve as x y columns, a map as a matrix."""
        if kind == "curve" and isinstance(self._predict_current_curve, np.ndarray):
            curve = np.array(self._predict_current_curve, dtype=np.float32)
            x = getattr(self, "_predict_current_curve_x", None)
            if not isinstance(x, np.ndarray) or x.shape != curve.shape:
                x = np.arange(len(curve), dtype=np.float32)
            request = PredictionArrayExportRequest(path, np.column_stack([x, curve]), fmt="%.6g", header="x y", comments="")
        elif kind in IMAGE_KINDS and isinstance(self._predict_current_image, np.ndarray):
            request = PredictionArrayExportRequest(path, np.array(self._predict_current_image, dtype=np.float32), fmt="%.6g")
        else:
            self._append_status_message("No data available to export", level="WARN")
            return None
        exported = self.prediction_view_model.export_array(request)
        if exported is None:
            raise OSError(self.prediction_view_model.state.error_message or tr("Predict data export failed"))
        self._append_status_message(trf("Prediction data exported: {path}", path=exported))
        return Path(exported)

    # -- the record ---------------------------------------------------------------------------------

    def _write_export_record(self, image: Path, written: List[Path], *, kind: str, shown: str = "") -> Path:
        module = self._current_module if isinstance(self._current_module, dict) else {}
        typed = module.get("_prediction_module")
        preprocess = getattr(typed, "preprocess", None)
        runtime = getattr(self, "_latest_runtime", None) or self._current_model
        inputs = getattr(self, "_current_input_files", None) or ([self._export_source()] if self._export_source() else [])
        request = PredictionRecordRequest(
            path=image.with_suffix(".json"),
            outputs=tuple(written) + (image.with_suffix(".json"),),
            kind=kind,
            shown=shown,
            mode=str(self.current_parameters.get("mode", "single_file")),
            input_files=tuple(str(path) for path in inputs),
            stack=int(getattr(self, "_current_input_stack", 1) or 1),
            module={
                "name": str(module.get("name") or self.current_parameters.get("module_name") or ""),
                "id": str(getattr(typed, "id", "") or ""),
                "version": str(getattr(typed, "version", "") or ""),
                "file": str(getattr(typed, "yaml_path", "") or ""),
            },
            model_path=str(self.current_parameters.get("module_model_path") or module.get("model_path") or ""),
            framework=str(self.current_parameters.get("framework") or ""),
            runtime_name=str(getattr(runtime, "runtime_name", "") or ""),
            runtime_version=str(getattr(runtime, "runtime_version", "") or ""),
            preprocess_entry=str(getattr(preprocess, "entry", "") or ""),
            preprocess_steps=tuple(str(step) for step in getattr(preprocess, "steps", ()) or ()),
            applied_steps=tuple(step for step in self._latest_preprocess_steps if isinstance(step, dict))
            if kind == "prediction" else (),
            display=self._export_display(kind),
        )
        record = self.prediction_view_model.exports.export_record(request)
        if record is None:
            raise OSError(self.prediction_view_model.state.error_message or tr("The export record could not be written."))
        return Path(record)

    def _export_display(self, kind: str) -> dict:
        prefix = "" if kind == "input preview" else "predict_"
        keys = ("auto_scale", "vmin", "vmax") if kind == "input preview" else ("predict_auto_scale", "predict_vmin", "predict_vmax")
        display = {"colormap": self.current_parameters.get("colormap")}
        for key in keys:
            display[key[len(prefix):]] = self.current_parameters.get(key)
        display["log_scale"] = bool(
            self.current_parameters.get("gisaxs_log_scale" if kind == "input preview" else "predict_log_scale", False)
        )
        return display

    # -- feedback -----------------------------------------------------------------------------------

    def _toast_parent(self):
        return getattr(self.ui, "gisaxsPredictPage", None) or self.main_window

    def _export_written(self, written: List[Path]) -> None:
        image = written[0]
        self._append_status_message(trf("Exported {files}", files=", ".join(path.name for path in written)))
        folder = image.parent
        parent = self._toast_parent()
        if parent is not None:
            show_toast(
                parent,
                trf("Saved {name} with its data and record", name=image.name)
                if len(written) > 2
                else trf("Saved {name} with its record", name=image.name),
                level="ok",
                action=(tr("Open Folder"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))),
            )

    def _export_failed(self, error) -> None:
        message = trf("Export failed: {error}", error=error)
        self._append_status_message(message, level="ERROR")
        parent = self._toast_parent()
        if parent is not None:
            show_toast(parent, message, level="error")


__all__ = ["ExportFilesMixin"]
