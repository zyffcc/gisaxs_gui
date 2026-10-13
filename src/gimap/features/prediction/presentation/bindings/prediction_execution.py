"""Prediction Execution coordination for Prediction."""

from __future__ import annotations

import os
import time

from pathlib import Path

from typing import Dict, List, Optional


from PyQt5.QtWidgets import (
    QMessageBox,
)

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.shared.file_paths import normalize_path

from ..prediction_worker import PredictionWorker

BUSY_TEXT = "Predicting… (the model starts in a separate process)"
# The model process cannot be interrupted: Predict stays disabled until it ends (one run at a time).
STOPPING_TEXT = "Stopping… (waiting for the model process to end)"
INPUT_CHANGED_TEXT = "The input, module or model changed during the prediction; its result was discarded."
# At quit a worker ends within its preprocessing plus ~1 s; this only bounds a step that hangs.
QUIT_WAIT_LIMIT_S = 60.0


def _path_key(path) -> str:
    return os.path.normcase(os.path.abspath(normalize_path(path)))


def _module_changed(current, started_with) -> bool:
    """Whether the module differs in value from the one the run was started with.

    Clicking or focusing the module list re-reads every module.yaml into new (equal) objects,
    so identity is not the question; a YAML edit saved during the run is. A comparison that
    cannot be made counts as changed.
    """
    if current is started_with:
        return False
    try:
        return bool(current != started_with)
    except Exception:  # noqa: BLE001 - e.g. an array in the YAML parameters
        return True


class PredictionExecutionMixin:
    """Own prediction execution presentation behavior."""

    def _execute_prediction(self) -> None:
        if self._prediction_active:
            return  # a single-file prediction is still running
        self._update_parameters_from_ui()
        if not self._validate_parameters():
            return
        if self.current_parameters.get("mode", "single_file") == "single_file":
            self._start_single_prediction()
            return
        self._prediction_active = True
        self._refresh_predict_readiness()
        try:
            self.status_updated.emit("Starting GISAXS prediction...")
            self.progress_updated.emit(0)
            # Multi-files: queue-based processing; progress and completion come from the manager.
            results = self._predict_multi_files()
            if not (results and results.get("processing_started")):
                self.progress_updated.emit(0)
                self.status_updated.emit("Failed to start multi-file prediction")
        except Exception as exc:  # pragma: no cover - runtime safety
            QMessageBox.critical(self.main_window, "Prediction Error", str(exc))
            self.status_updated.emit(trf("GISAXS prediction error: {error}", error=exc))
            # 重置多文件预测状态
            if self._multifile_prediction_active:
                self._on_multifile_prediction_completed()
        finally:
            self._prediction_active = False
            self._refresh_predict_readiness()

    # -- single file: the model runs in a PredictionWorker, the window stays responsive ------

    def _start_single_prediction(self) -> None:
        self.status_updated.emit("Starting GISAXS prediction...")
        self.progress_updated.emit(0)
        if self._current_image is None:
            self._append_status_message("No image loaded for prediction", level="WARN")
            return
        typed_module = self._typed_prediction_module()
        if typed_module is None:
            self._append_status_message("Selected module has no typed prediction contract", level="ERROR")
            self._append_status_message("Preprocessing failed", level="ERROR")
            return
        model_path = str(self.current_parameters.get("module_model_path") or "")
        if self._current_model is None or not model_path:
            if not model_path:
                self._append_status_message(
                    "Selected module has no typed prediction contract or model path", level="ERROR"
                )
            self._append_status_message("Prediction failed", level="ERROR")
            return
        self._prediction_run_id += 1
        worker = PredictionWorker(
            self.prediction_view_model,
            self._current_image,
            typed_module,
            Path(model_path),
            self._prediction_run_id,
            self,
        )
        worker.prediction_finished.connect(self._on_single_prediction_finished)
        worker.finished.connect(worker.deleteLater)
        self._prediction_workers[worker.run_id] = worker
        self._prediction_active = True  # until the result slot runs: Predict stays disabled
        self._set_prediction_busy(BUSY_TEXT)  # English: the canvas shows it translated
        self._refresh_predict_readiness()
        self.progress_updated.emit(10)
        worker.start()

    def _on_single_prediction_finished(self, run_id: int, prepared, result, error: str) -> None:
        """GUI thread: log, keep the preprocessing snapshots and show the result.

        Only here does the run end (``_prediction_active`` false, Predict enabled again), also
        after Stop, so at most one model process runs at a time.
        """
        worker = self._prediction_workers.pop(run_id, None)
        try:
            if worker is None or worker.discard:
                self._append_status_message("Ignored the result of a stopped prediction.", level="INFO")
                return
            if self._prediction_inputs_changed(worker):
                # One result view must not mix two frames, modules or models (data lineage).
                message = tr(INPUT_CHANGED_TEXT)
                self._append_status_message(message, level="WARN")
                self.progress_updated.emit(0)
                parent = getattr(self.ui, "gisaxsPredictPage", None) or self.main_window
                if parent is not None:
                    show_toast(parent, message, level="warning")
                return
            if prepared is None:
                self._append_status_message(error or "Module preprocessing failed", level="ERROR")
                self._append_status_message("Preprocessing failed", level="ERROR")
                self.progress_updated.emit(0)
                return
            self._latest_preprocess_steps = list(prepared.steps)
            self._latest_model_input = prepared.values
            self._latest_preprocess_source = worker.image
            self._append_status_message(trf("Module preprocess output shape {shape}", shape=prepared.values.shape))
            if result is None:
                self._append_status_message(error or "Isolated prediction failed", level="ERROR")
                self._append_status_message("Prediction failed", level="ERROR")
                self.progress_updated.emit(0)
                return
            self.progress_updated.emit(70)
            self._latest_runtime = getattr(result, "runtime", None)  # the export record names it
            self._display_prediction(dict(result.outputs))
            self.progress_updated.emit(100)
            self.status_updated.emit("GISAXS prediction finished!")
        except Exception as exc:  # pragma: no cover - runtime safety
            QMessageBox.critical(self.main_window, "Prediction Error", str(exc))
            self.status_updated.emit(trf("GISAXS prediction error: {error}", error=exc))
        finally:
            if not self._prediction_workers:
                self._prediction_active = False
                self._set_prediction_busy(None)
                self._refresh_predict_readiness()

    def _prediction_inputs_changed(self, worker) -> bool:
        """Whether the frame, module or model differs from what the run was started with."""
        module = self._current_module
        typed_module = module.get("_prediction_module") if isinstance(module, dict) else None
        model_path = str(self.current_parameters.get("module_model_path") or "")
        return bool(
            worker.image is not self._current_image
            or _module_changed(typed_module, worker.module)
            or self._current_model is None
            or not model_path
            or _path_key(model_path) != _path_key(worker.model_path)
        )

    def _stop_single_prediction(self) -> bool:
        """Stop the running single-file prediction: its result will be discarded.

        The model process cannot be interrupted, so the run stays active ('Stopping…', Predict
        disabled) until the worker reports back; only then can the next prediction start.
        """
        if not self._prediction_workers:
            return False
        running = [worker for worker in self._prediction_workers.values() if not worker.discard]
        if not running:
            return True  # already stopping
        for worker in running:
            worker.discard = True
        self._set_prediction_busy(STOPPING_TEXT)
        self._append_status_message(
            "Prediction stopped. The model process finishes in the background; its result is discarded.",
            level="WARN",
        )
        self.progress_updated.emit(0)
        self._refresh_predict_readiness()
        return True

    def _prediction_stopping(self) -> bool:
        """A stopped single-file run whose model process has not ended yet."""
        workers = getattr(self, "_prediction_workers", None) or {}
        return bool(workers) and all(worker.discard for worker in workers.values())

    def prediction_running(self) -> bool:
        """A 2D prediction the user has not stopped is still running (for the question on quit)."""
        workers = getattr(self, "_prediction_workers", None) or {}
        if any(not worker.discard for worker in workers.values()):
            return True
        manager = getattr(self, "_multifile_manager", None)
        return bool(getattr(manager, "is_running", False))

    def stop_predictions(self) -> None:
        """At quit, before the job runner cancels its jobs: discard every single-file result, so
        a worker still preprocessing starts no model process, and end the batch after its file."""
        for worker in list((getattr(self, "_prediction_workers", None) or {}).values()):
            worker.discard = True
        manager = getattr(self, "_multifile_manager", None)
        if manager is not None and getattr(manager, "is_running", False):
            manager.cancel_prediction()

    def _wait_for_prediction_workers(self, limit_s: float = QUIT_WAIT_LIMIT_S) -> None:
        """aboutToQuit: wait until every worker has ended, since a QThread destroyed while
        running aborts the process.

        A discarded worker ends after its preprocessing, or about 1 s after its model job is
        cancelled (process terminated and joined). A model job started after the window shut
        the job runner down is cancelled here, so the wait has no fixed short cap; the limit
        only guards against a preprocessing step that never returns.
        """
        self.stop_predictions()
        deadline = time.monotonic() + limit_s
        for worker in list(self._prediction_workers.values()):
            try:
                while not worker.wait(100):
                    self._cancel_model_jobs()
                    if time.monotonic() >= deadline:
                        return
            except RuntimeError:
                continue  # already finished and deleted

    def _cancel_model_jobs(self) -> None:
        """Cancel the model processes still registered with the job runner (quitting only)."""
        context = getattr(self.prediction_view_model, "context", None)
        jobs = getattr(context, "jobs", None)
        shutdown = getattr(jobs, "shutdown", None)
        if shutdown is not None:
            try:
                shutdown()
            except Exception:  # noqa: BLE001 - quitting goes on
                pass

    def _set_prediction_busy(self, text) -> None:
        workbench = getattr(self.ui, "predictionWorkbenchLayout", None)
        if workbench is not None and hasattr(workbench, "set_busy"):
            workbench.set_busy(text)

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
        mode = self.current_parameters.get("mode", "single_file")
        if mode == "single_file":
            file_path = self.current_parameters.get("input_file")
            if not file_path or not os.path.exists(file_path):
                QMessageBox.warning(
                    self.main_window, "Invalid Parameters", "Please select a valid input file"
                )
                return False
        else:
            folder = self.current_parameters.get("input_folder")
            if not folder or not os.path.exists(folder):
                QMessageBox.warning(
                    self.main_window, "Invalid Parameters", "Please select a valid folder"
                )
                return False
        if not self._framework_ready():
            QMessageBox.warning(
                self.main_window,
                "Framework",
                "The selected model requires a compatible installed framework.",
            )
            return False
        if not self._model_ready():
            QMessageBox.warning(
                self.main_window, "Model", "Please import a model before running prediction."
            )
            return False
        return True

    def _predict_single_file(self) -> Optional[Dict[str, object]]:
        file_path = self.current_parameters.get("input_file")
        if not file_path:
            return None
        self.status_updated.emit(trf("Processing file: {name}", name=os.path.basename(file_path)))
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
        """多文件预测 - 使用新的队列处理系统"""
        folder = self.current_parameters.get("input_folder")
        if not folder:
            self._append_status_message("No input folder selected", level="WARN")
            return None

        files = [
            str(path)
            for path in self.prediction_view_model.files.discover_files(
                Path(folder), (".cbf", ".tif", ".tiff")
            )
        ]
        if not files and self.prediction_view_model.state.error_message:
            self._append_status_message(
                trf("Error scanning folder: {error}", error=self.prediction_view_model.state.error_message),
                level="ERROR",
            )
            return None

        if not files:
            self._append_status_message("No compatible image files found in folder", level="WARN")
            return None

        # 应用范围过滤
        range_text = self.current_parameters.get("range_value", "")
        if range_text:
            try:
                indices = self._parse_range_text(range_text)
                if indices:
                    self._scan_directory_for_cbf(folder)
                    missing = [idx for idx in indices if idx not in self._index_to_file]
                    files = [
                        self._index_to_file[idx] for idx in indices if idx in self._index_to_file
                    ]
                    if missing:
                        missing_text = ", ".join(f"{idx:05d}" for idx in missing[:10])
                        if len(missing) > 10:
                            missing_text += ", ..."
                        self._append_status_message(
                            trf("Range skipped missing CBF indices: {indices}", indices=missing_text), level="WARN"
                        )
            except Exception as e:
                self._append_status_message(trf("Error parsing range: {error}", error=e), level="WARN")

        if not files:
            self._append_status_message("No files selected by range", level="WARN")
            return None

        try:
            every = max(1, int(self._get_line_edit_text("gisaxsPredictEveryValue") or "1"))
        except ValueError:
            every = 1
            self._set_line_edit("gisaxsPredictEveryValue", "1")
            self._append_status_message("Every must be a positive integer; using 1.", level="WARN")

        if every > 1:
            batches = [
                list(batch)
                for batch in self.prediction_view_model.files.complete_batches(files, every)
            ]
            skipped = len(files) - (len(batches) * every)
            if skipped:
                self._append_status_message(
                    trf(
                        "Skipped {count} trailing file(s) that do not make a full Every={every} stack.",
                        count=skipped, every=every,
                    ),
                    level="WARN",
                )
        else:
            batches = [[file_path] for file_path in files]
        self._multifile_batch_map = {batch[0]: batch for batch in batches if batch}
        files_to_process = list(self._multifile_batch_map.keys())
        if not files_to_process:
            self._append_status_message(
                "No complete multi-file stacks selected by range/every.", level="WARN"
            )
            return None
        if every > 1:
            self._append_status_message(
                trf(
                    "Multi-file range grouped into {count} batch(es), Every={every}.",
                    count=len(files_to_process), every=every,
                ),
                level="INFO",
            )

        # 清空现有结果并添加新的待处理项目
        if self._multifile_results_widget:
            self._multifile_results_widget.clearResults()

            # 添加所有文件到结果列表
            for file_path in files_to_process:
                row = self._multifile_results_widget.addPredictResult(file_path)
                batch = self._multifile_batch_map.get(file_path, [])
                if len(batch) > 1:
                    result = self._multifile_results_widget.table_model.getResult(row)
                    if result is not None:
                        result.file_name = (
                            f"{os.path.basename(batch[0])} - {os.path.basename(batch[-1])}"
                        )
                        result.file_path = "\n".join(batch)
                        result.stack_count = len(batch)
                        self._multifile_results_widget.table_model.updateResult(row, result)
                        self._append_status_message(
                            trf(
                                "Queued stack: {first} - {last} ({count} files)",
                                first=os.path.basename(batch[0]), last=os.path.basename(batch[-1]), count=len(batch),
                            ),
                            level="INFO",
                        )
                elif batch:
                    result = self._multifile_results_widget.table_model.getResult(row)
                    if result is not None:
                        result.stack_count = 1
                        self._multifile_results_widget.table_model.updateResult(row, result)

        # 开始批量预测
        if self._multifile_manager:
            self._multifile_prediction_active = True
            self._show_multifile_results_window()
            self._multifile_manager.start_batch_prediction(
                files_to_process, self._predict_single_file_for_batch
            )

        # 立即返回，实际处理将在后台进行
        return {"folder": folder, "total_files": len(files_to_process), "processing_started": True}

    def _predict_single_file_for_batch(self, file_path: str) -> Dict[str, object]:
        """为批量处理执行单文件预测"""
        try:
            # 临时设置当前文件用于预测
            old_file = self.current_parameters.get("input_file", "")
            self.current_parameters["input_file"] = file_path
            batch = self._multifile_batch_map.get(file_path) or [file_path]
            if len(batch) > 1:
                self.status_updated.emit(
                    trf(
                        "Predicting stack ({count} files): {first} - {last}",
                        count=len(batch), first=os.path.basename(batch[0]), last=os.path.basename(batch[-1]),
                    )
                )
            else:
                self.status_updated.emit(trf("Predicting file: {name}", name=os.path.basename(file_path)))

            # 执行实际预测逻辑（这里需要调用真正的预测代码）
            result = self._execute_single_file_prediction(file_path, batch)

            # 恢复原来的文件设置
            self.current_parameters["input_file"] = old_file

            return result

        except Exception as e:
            # 恢复原来的文件设置
            if "old_file" in locals():
                self.current_parameters["input_file"] = old_file
            raise e

    def _execute_single_file_prediction(
        self, file_path: str, stack_files: Optional[List[str]] = None
    ) -> Dict[str, object]:
        """执行单个文件的预测逻辑 - 真正调用预测流程"""
        typed_module = (
            self._current_module.get("_prediction_module")
            if isinstance(self._current_module, dict)
            else None
        )
        model_path = str(self.current_parameters.get("module_model_path") or "")
        if typed_module is not None and model_path:
            paths = tuple(stack_files or [file_path])
            item = self.prediction_view_model.predict_file_batch(
                paths,
                typed_module,
                Path(model_path),
            )
            if item.status != "succeeded" or item.prediction is None:
                raise RuntimeError(item.error_message or "Prediction failed")
            return {
                "file": file_path,
                "stack_count": len(paths),
                "stack_files": list(paths),
                "prediction_data": dict(item.prediction.outputs),
            }
        try:
            # 保存原有参数和状态
            old_input_file = self.current_parameters.get("input_file", "")
            old_mode = self.current_parameters.get("mode", "single_file")
            old_current_image = self._current_image

            # 临时设置为单文件模式
            self.current_parameters["input_file"] = file_path
            self.current_parameters["mode"] = "single_file"

            # 加载图像（使用同步方法）
            image_data = (
                self._load_cbf_stack_sync(stack_files)
                if stack_files and len(stack_files) > 1
                else self._load_cbf_file_sync(file_path)
            )

            if image_data is None:
                raise Exception(f"Failed to load image: {file_path}")

            # 设置当前图像
            self._current_image = image_data

            # 执行真正的预测流程（与单文件相同）
            # 1. 预处理
            inp = self._preprocess_for_module(self._current_image)
            if inp is None:
                raise Exception("Preprocessing failed")

            # 2. 模型预测
            outs = self._predict_with_current_model(inp)
            if not outs:
                raise Exception("Prediction failed")

            # 恢复原有参数和状态
            self.current_parameters["input_file"] = old_input_file
            self.current_parameters["mode"] = old_mode
            self._current_image = old_current_image

            # 返回结果（只包含预测数据，预处理步骤按需计算）
            return {
                "file": file_path,
                "stack_count": len(stack_files) if stack_files else 1,
                "stack_files": list(stack_files) if stack_files else [file_path],
                "prediction_data": outs,  # 真正的预测结果
            }

        except Exception as e:
            # 确保恢复原有参数和状态
            if "old_input_file" in locals():
                self.current_parameters["input_file"] = old_input_file
            if "old_mode" in locals():
                self.current_parameters["mode"] = old_mode
            if "old_current_image" in locals():
                self._current_image = old_current_image
            raise e
