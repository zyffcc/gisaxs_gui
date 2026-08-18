"""Image Loading Workers primitives for fitting presentation."""

from __future__ import annotations

import os


import re

import time


from collections import OrderedDict

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


class FolderImageScanWorker(QThread):
    """Scan one folder level for detector-image navigation without blocking the UI."""

    scan_finished = pyqtSignal(str, list)
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str, str)

    # 函数说明：初始化对象状态和相关资源。
    def __init__(
        self,
        file_path: str,
        extensions: tuple,
        max_files: int = 5000,
        fitting_view_model=None,
    ):
        super().__init__()
        self.fitting_view_model = fitting_view_model or _create_default_fitting_view_model()
        self.file_path = normalize_path(file_path or "")
        self.extensions = tuple(extensions or (".cbf",))
        self.max_files = int(max(100, max_files or 5000))

    # 函数说明：实现 natural 排序 key 相关逻辑。
    def _natural_sort_key(self, path):
        name = os.path.basename(path)
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]

    # 函数说明：执行后台任务或工作流程。
    def run(self):
        try:
            if not self.file_path:
                self.scan_finished.emit("", [])
                return
            remote = self.fitting_view_model.storage.is_remote_source(self.file_path)
            if remote:
                self.status_updated.emit(
                    "Remote/cloud folder detected. Scanning current directory only..."
                )
            if os.path.isdir(self.file_path):
                folder = self.file_path
            else:
                folder = os.path.dirname(self.file_path)
            if not folder or not os.path.isdir(folder):
                self.error_occurred.emit(self.file_path, "Folder does not exist")
                return
            files = []
            with os.scandir(folder) as it:
                for entry in it:
                    if self.isInterruptionRequested():
                        return
                    if len(files) >= self.max_files:
                        self.status_updated.emit(f"Folder scan limited to {self.max_files} files")
                        break
                    try:
                        if (
                            entry.is_file()
                            and os.path.splitext(entry.name)[1].lower() in self.extensions
                        ):
                            files.append(normalize_path(entry.path))
                    except Exception:
                        continue
            files.sort(key=self._natural_sort_key)
            self.scan_finished.emit(self.file_path, files)
        except Exception as exc:
            self.error_occurred.emit(self.file_path, str(exc))


class AsyncImageLoader(QThread):
    """No description."""

    image_loaded = pyqtSignal(np.ndarray, str)
    progress_updated = pyqtSignal(int, str)
    error_occurred = pyqtSignal(str)
    remote_file_detected = pyqtSignal(str)
    copy_started = pyqtSignal(str, str)
    copy_finished = pyqtSignal(str, str)
    load_started = pyqtSignal(str)
    load_finished = pyqtSignal(str)

    # 函数说明：初始化对象状态和相关资源。
    def __init__(self, fitting_view_model=None):
        super().__init__()
        self.fitting_view_model = fitting_view_model or _create_default_fitting_view_model()
        self.file_path = None
        self.stack_count = 1
        self.frame_index = 0
        self._image_cache = OrderedDict()
        self._image_cache_limit = 8
        self.copy_remote_to_cache = True
        self.remote_cache_dir = self.fitting_view_model.storage.default_remote_cache_directory()
        self.remote_cache_limit_gb = 3.0
        self._last_source_files = []
        self._last_effective_files = []

    # 函数说明：实现 configure 远程 缓存 相关逻辑。
    def configure_remote_cache(self, enabled=True, cache_dir=None, max_gb=3.0):
        self.copy_remote_to_cache = bool(enabled)
        self.remote_cache_dir = self.fitting_view_model.storage.display_remote_cache_directory(
            cache_dir or self.fitting_view_model.storage.default_remote_cache_directory()
        )
        try:
            self.remote_cache_limit_gb = float(max_gb)
        except Exception:
            self.remote_cache_limit_gb = 3.0

    @staticmethod
    def _natural_sort_key(path):
        name = os.path.basename(path)
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]

    # 函数说明：加载图像。
    def load_image(self, file_path, stack_count=1, frame_index=0):
        """No description."""
        if self.isRunning():
            self.requestInterruption()
            self.progress_updated.emit(
                0,
                "Previous image load is still running; please wait or cancel the current workflow.",
            )
            return
        self.file_path = file_path
        self.stack_count = stack_count
        self.frame_index = max(0, int(frame_index or 0))
        self.start()

    # 函数说明：实现 prepare 文件 for read 相关逻辑。
    def _prepare_file_for_read(self, source_path: str) -> str:
        source_path = normalize_path(source_path)
        if self.copy_remote_to_cache and self.fitting_view_model.storage.is_remote_source(
            source_path
        ):
            self.remote_file_detected.emit(source_path)
            self.progress_updated.emit(
                5,
                "This file appears to be in a cloud or network folder. Copying to local cache before processing...",
            )
            target = self.fitting_view_model.storage.remote_cache_target(
                source_path, self.remote_cache_dir
            )
            self.copy_started.emit(source_path, target)
            cached = self.fitting_view_model.storage.prepare_remote_source(
                source_path,
                self.remote_cache_dir,
                self.remote_cache_limit_gb,
                on_progress=self.progress_updated.emit,
                is_cancelled=self.isInterruptionRequested,
            )
            self.copy_finished.emit(source_path, cached)
            return cached
        return source_path

    # 函数说明：执行后台任务或工作流程。
    def run(self):
        """No description."""
        try:
            file_ext = os.path.splitext(self.file_path)[1].lower()
            if file_ext == ".cbf" and not self.fitting_view_model.storage.dependency_available(
                "fabio"
            ):
                self.error_occurred.emit("fabio library is required for CBF file processing")
                return

            self.progress_updated.emit(10, "Loading file...")
            self.load_started.emit(self.file_path)

            if file_ext not in {".cbf", ".nxs", ".tif", ".tiff"}:
                self.error_occurred.emit(
                    "Only CBF, NXS, and TIFF detector images are supported currently"
                )
                return

            effective_stack_count = max(1, int(self.stack_count))
            cache_key = (
                normalize_path(self.file_path),
                effective_stack_count,
                int(self.frame_index),
            )
            cached = self._image_cache.get(cache_key)
            if cached is not None:
                self._image_cache.move_to_end(cache_key)
                self.progress_updated.emit(90, "Using cached image data...")
                print(
                    f"[Timing] fabio read: 0.00 ms (cache hit: {os.path.basename(self.file_path)})"
                )
                self.image_loaded.emit(cached, self.file_path)
                self.progress_updated.emit(100, "Done")
                return

            read_start = time.perf_counter()
            image_data = self._load_scattering_file_compat(
                self.file_path,
                frame_index=self.frame_index,
                stack_count=self.stack_count,
            )
            print(
                f"[Timing] fabio read: {(time.perf_counter() - read_start) * 1000:.2f} ms ({os.path.basename(self.file_path)})"
            )

            if image_data is not None:
                self._image_cache[cache_key] = image_data
                self._image_cache.move_to_end(cache_key)
                while len(self._image_cache) > self._image_cache_limit:
                    self._image_cache.popitem(last=False)
                self.progress_updated.emit(90, "Processing image data...")
                self.image_loaded.emit(image_data, self.file_path)
                self.progress_updated.emit(100, "Done")
                self.load_finished.emit(self.file_path)
            else:
                self.error_occurred.emit("Failed to load image data")

        except Exception as e:
            self.error_occurred.emit(f"Error loading image: {str(e)}")

    def _load_scattering_file_compat(self, file_path, frame_index=0, stack_count=1):
        result = self.fitting_view_model.storage.load_scattering_background(
            LoadScatteringFileRequest(
                path=Path(normalize_path(file_path)),
                frame_index=int(frame_index),
                stack_count=int(stack_count),
            ),
            prepare_path=self._prepare_file_for_read,
            on_progress=self.progress_updated.emit,
        )
        if result.error is not None:
            raise RuntimeError(f"[{result.error.code}] {result.error.message}")
        loaded = result.value
        self._last_source_files = [str(path) for path in loaded.source_files]
        self._last_effective_files = list(loaded.metadata.get("effective_files", ()))
        return loaded.image

    def _load_detector_file(self, file_path, frame_index=0):
        """Legacy loader entry delegated to the fitting file use case."""
        return self._load_scattering_file_compat(
            file_path,
            frame_index=frame_index,
            stack_count=1,
        )

    def _load_multiple_nxs_frames(self, file_path, frame_index, stack_count):
        """Legacy NXS stack entry delegated to the fitting file use case."""
        return self._load_scattering_file_compat(
            file_path,
            frame_index=frame_index,
            stack_count=stack_count,
        )

    def _load_multiple_detector_files(self, start_file, stack_count, extensions):
        """Legacy ordinary-stack entry; extensions remain validated by the adapter."""
        del extensions
        return self._load_scattering_file_compat(
            start_file,
            frame_index=0,
            stack_count=stack_count,
        )

    def _load_single_cbf_file(self, cbf_file):
        """Legacy CBF entry preserves the historical None-on-error contract."""
        try:
            return self._load_scattering_file_compat(cbf_file, stack_count=1)
        except Exception:
            return None

    def _load_multiple_cbf_files(self, start_file, stack_count):
        """Legacy CBF stack entry preserves the historical None-on-error contract."""
        try:
            return self._load_scattering_file_compat(
                start_file,
                stack_count=stack_count,
            )
        except Exception:
            return None
