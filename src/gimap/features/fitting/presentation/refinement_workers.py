"""Refinement Workers primitives for fitting presentation."""

from __future__ import annotations


from PyQt5.QtCore import QObject, pyqtSignal


class ManualAutoRefineWorker(QObject):
    """Run manual Auto Refine outside the GUI thread."""

    started = pyqtSignal()
    progress = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    # 函数说明：初始化对象状态和相关资源。
    def __init__(self, controller, setup, selected, options):
        super().__init__()
        self.controller = controller
        self.setup = setup
        self.selected = selected
        self.options = options
        self._stop_requested = False

    # 函数说明：实现 request stop 相关逻辑。
    def request_stop(self):
        self._stop_requested = True

    # 函数说明：执行后台任务或工作流程。
    def run(self):
        try:
            self.started.emit()
            result = self.controller._run_manual_auto_refine(
                self.setup,
                self.selected,
                self.options,
                progress_callback=self.progress.emit,
                stop_callback=lambda: self._stop_requested,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class RefineUiBridge(QObject):
    """Relay refine worker signals to slots owned by the GUI thread."""

    progress = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)
