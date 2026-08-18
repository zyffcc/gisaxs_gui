"""Batch Manager for multi-file prediction."""

from __future__ import annotations


from typing import List, Callable

from concurrent.futures import ThreadPoolExecutor


from PyQt5.QtCore import QObject, pyqtSignal


from .result_types import (
    PredictStatus,
)


class MultiFilePredictManager(QObject):
    """多文件预测管理器"""

    # 信号
    prediction_started = pyqtSignal()
    prediction_completed = pyqtSignal()
    result_updated = pyqtSignal(int, dict)  # index, update_data
    progress_updated = pyqtSignal(int, int)  # completed, total

    def __init__(self, parent=None):
        super().__init__(parent)
        self.executor = ThreadPoolExecutor(max_workers=1)  # 单线程顺序处理
        self.current_futures = []
        self.is_running = False
        self.cancelled = False

    def start_batch_prediction(self, file_paths: List[str], predict_func: Callable) -> None:
        """开始批量预测"""
        if self.is_running:
            return

        self.is_running = True
        self.cancelled = False
        self.current_futures.clear()

        self.prediction_started.emit()

        # 提交批量任务
        future = self.executor.submit(self._batch_predict_worker, file_paths, predict_func)
        self.current_futures.append(future)

    def cancel_prediction(self) -> None:
        """取消预测"""
        self.cancelled = True
        for future in self.current_futures:
            future.cancel()

    def _batch_predict_worker(self, file_paths: List[str], predict_func: Callable) -> None:
        """批量预测工作线程"""
        total = len(file_paths)
        completed = 0

        try:
            for i, file_path in enumerate(file_paths):
                if self.cancelled:
                    for cancel_index in range(i, total):
                        self.result_updated.emit(cancel_index, {"status": PredictStatus.CANCELLED})
                    break

                # 更新状态为运行中
                self.result_updated.emit(i, {"status": PredictStatus.RUNNING})

                try:
                    # 执行预测
                    result_data = predict_func(file_path)

                    # 更新完成状态
                    self.result_updated.emit(
                        i, {"status": PredictStatus.COMPLETED, "prediction_data": result_data}
                    )
                    completed += 1

                except Exception as e:
                    # 更新失败状态
                    self.result_updated.emit(
                        i, {"status": PredictStatus.FAILED, "error_message": str(e)}
                    )

                # 更新进度
                self.progress_updated.emit(completed, total)

        finally:
            self.is_running = False
            self.prediction_completed.emit()
