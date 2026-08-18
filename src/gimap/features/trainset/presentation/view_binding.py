"""Feature-owned binding between the Trainset page and ViewModel."""

from __future__ import annotations


from pathlib import Path

from typing import Any, Dict, Optional

import numpy as np

from PyQt5.QtCore import QObject, QThreadPool, QTimer, pyqtSignal


from src.gimap.features.trainset.application.ports import SimulationPort

from .page import TrainsetBuildPage

from .view_model import TrainsetViewModel

from .bindings.page_connections import PageConnectionsMixin
from .bindings.configuration import ConfigurationMixin
from .bindings.detector_design import DetectorDesignMixin
from .bindings.preview_simulation import PreviewSimulationMixin
from .bindings.validation_files import ValidationFilesMixin
from .bindings.local_jobs import LocalJobsMixin
from .bindings.hpc_jobs import HpcJobsMixin
from .bindings.public_api import PublicApiMixin
from .background_tasks import _FunctionWorker

__all__ = ["TrainsetViewBinding"]


class TrainsetViewBinding(
    PageConnectionsMixin,
    ConfigurationMixin,
    DetectorDesignMixin,
    PreviewSimulationMixin,
    ValidationFilesMixin,
    LocalJobsMixin,
    HpcJobsMixin,
    PublicApiMixin,
    QObject,
):
    """Bind Trainset page events and rendering to framework-neutral commands.

    This is presentation glue rather than an application controller: scientific
    work, persistence, local execution and Slurm operations are delegated to the
    injected ViewModel and its use cases/adapters.
    """

    parameters_changed = pyqtSignal(str, dict)

    generation_started = pyqtSignal()

    generation_finished = pyqtSignal()

    generation_error = pyqtSignal(str)

    progress_updated = pyqtSignal(int)

    status_updated = pyqtSignal(str)

    prediction_module_registered = pyqtSignal(str)

    def __init__(
        self,
        ui,
        parent=None,
        *,
        simulation_port: SimulationPort | None = None,
        trainset_view_model: TrainsetViewModel | None = None,
        page: TrainsetBuildPage | None = None,
        project_root: Path | None = None,
    ):
        super().__init__(parent)
        self.ui = ui
        self.window = getattr(parent, "parent", None)
        if simulation_port is None:
            raise ValueError("TrainsetViewBinding requires SimulationPort")
        if page is None:
            raise ValueError("TrainsetViewBinding requires an injected TrainsetBuildPage")
        if trainset_view_model is None:
            raise ValueError("TrainsetViewBinding requires TrainsetViewModel")
        if project_root is None:
            raise ValueError("TrainsetViewBinding requires an explicit project root")
        self.simulation_port = simulation_port
        self.trainset_view_model = trainset_view_model
        self.catalog = getattr(trainset_view_model, "catalog", page.catalog)
        self.page = page
        self.project_root = Path(project_root)
        self.config: Dict[str, Any] = self.trainset_view_model.default_config()
        self.reference_image: Optional[np.ndarray] = None
        self.package_dir: Optional[Path] = None
        self._pending_local_arguments = None
        self._local_paused = False
        self.thread_pool = QThreadPool.globalInstance()
        self._initialized = False
        self._remote_refresh_running = False
        self._result_sync_started = False
        self._preview_realization = 0
        self._preview_busy = False
        self._preview_worker: Optional[_FunctionWorker] = None
        self._random_mask_example: Optional[np.ndarray] = None
        self._applying_config = False
        self._what_if_busy = False
        self._what_if_worker: Optional[_FunctionWorker] = None
        self._pending_what_if_values: Optional[Dict[str, float]] = None
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(900)
        self._autosave_timer.timeout.connect(self._persist_current_config)
        self.monitor_timer = QTimer(self)
        self.monitor_timer.setInterval(15000)
        self.monitor_timer.timeout.connect(self._refresh_job)
        self._connect_page()
