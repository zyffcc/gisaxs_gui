"""Insitu Workers primitives for fitting presentation."""

from __future__ import annotations


from pathlib import Path

import numpy as np

from PyQt5.QtCore import pyqtSignal, QThread


from src.gimap.features.fitting.application import (
    LoadScatteringFileRequest,
)


from src.gimap.shared.file_paths import normalize_path


from src.gimap.features.fitting.application import (
    LoadScatteringFileRequest,
)


from src.gimap.shared.file_paths import normalize_path


from .scientific_commands import (
    _create_default_fitting_view_model,
)


class InsituBatchImageLoader(QThread):
    """Load and sum an explicit list of detector images without blocking the UI."""

    image_loaded = pyqtSignal(object, str)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int, str)
    remote_file_detected = pyqtSignal(str)
    copy_started = pyqtSignal(str, str)
    copy_finished = pyqtSignal(str, str)

    # 函数说明：初始化对象状态和相关资源。
    def __init__(
        self,
        file_paths,
        fitting_view_model=None,
        copy_remote_to_cache=True,
        cache_dir=None,
        cache_limit_gb=3.0,
    ):
        super().__init__()
        self.file_paths = list(file_paths or [])
        self.fitting_view_model = fitting_view_model or _create_default_fitting_view_model()
        self.copy_remote_to_cache = bool(copy_remote_to_cache)
        self.cache_dir = (
            cache_dir or self.fitting_view_model.storage.default_remote_cache_directory()
        )
        self.cache_limit_gb = float(cache_limit_gb or 3.0)

    # 函数说明：实现 prepare 文件 for read 相关逻辑。
    def _prepare_file_for_read(self, source_path: str) -> str:
        source_path = normalize_path(source_path)
        if self.copy_remote_to_cache and self.fitting_view_model.storage.is_remote_source(
            source_path
        ):
            self.remote_file_detected.emit(source_path)
            self.progress_updated.emit(5, "Copying remote in-situ file to local cache...")
            target = self.fitting_view_model.storage.remote_cache_target(
                source_path, self.cache_dir
            )
            self.copy_started.emit(source_path, target)
            cached = self.fitting_view_model.storage.prepare_remote_source(
                source_path,
                self.cache_dir,
                self.cache_limit_gb,
                on_progress=self.progress_updated.emit,
                is_cancelled=self.isInterruptionRequested,
            )
            self.copy_finished.emit(source_path, cached)
            return cached
        return source_path

    # 函数说明：执行后台任务或工作流程。
    def run(self):
        try:
            if not self.file_paths:
                raise RuntimeError("No files to load")
            summed = None
            for index, path in enumerate(self.file_paths):
                if self.isInterruptionRequested():
                    raise RuntimeError("Batch loading stopped")
                self.progress_updated.emit(
                    int(10 + (index / max(1, len(self.file_paths))) * 70),
                    f"Loading in-situ file {index + 1}/{len(self.file_paths)}",
                )
                outcome = self.fitting_view_model.storage.load_scattering_background(
                    LoadScatteringFileRequest(Path(normalize_path(path))),
                    prepare_path=self._prepare_file_for_read,
                    on_progress=self.progress_updated.emit,
                )
                if outcome.error is not None:
                    raise RuntimeError(f"[{outcome.error.code}] {outcome.error.message}")
                data = outcome.value.image.astype(np.float32, copy=False)
                if summed is None:
                    summed = data.copy()
                else:
                    if summed.shape != data.shape:
                        raise RuntimeError(f"Image shape mismatch in batch: {path}")
                    summed += data
            self.image_loaded.emit(summed, self.file_paths[0])
        except Exception as exc:
            self.error_occurred.emit(str(exc))


class InsituCutWorker(QThread):
    """Run the injected in-situ cut command off the UI thread."""

    cut_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    # 函数说明：初始化对象状态和相关资源。
    def __init__(self, payload: dict, compute_cut=None):
        super().__init__()
        self.payload = dict(payload or {})
        self.compute_cut = compute_cut or _create_default_fitting_view_model().science.insitu_cut

    # 函数说明：执行后台任务或工作流程。
    def run(self):
        try:
            self.cut_finished.emit(self.compute_cut.execute(self.payload))
        except Exception as exc:
            self.error_occurred.emit(str(exc))
