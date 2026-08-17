"""不依赖 PyQt widget 的 Geometry Calibration ViewModel。"""

from __future__ import annotations

from pathlib import Path

from src.gimap.app import AppContext

from ..application import (
    ApplyCalibration,
    ExportCalibration,
    ImportCalibration,
    LoadCalibrationImage,
    LoadDetectorCatalog,
    RunCalibration,
)
from ..application.ports import CancellationCheck, ProgressCallback
from ..domain import CalibrationRequest, CalibrationResult, DetectorImage
from .state import CalibrationState


class CalibrationViewModel:
    """持有 dialog state，并把 commands 委托给 application use cases。"""

    def __init__(
        self,
        *,
        app_context: AppContext,
        load_image: LoadCalibrationImage,
        run_calibration: RunCalibration,
        export_calibration: ExportCalibration,
        import_calibration: ImportCalibration,
        apply_calibration: ApplyCalibration,
        load_detector_catalog: LoadDetectorCatalog,
    ):
        self.app_context = app_context
        self.state = app_context.project_state.feature_state(
            "calibration",
            CalibrationState,
        )
        self.image: DetectorImage | None = None
        self.result: CalibrationResult | None = None
        self._load_image = load_image
        self._run_calibration = run_calibration
        self._export_calibration = export_calibration
        self._import_calibration = import_calibration
        self._apply_calibration = apply_calibration
        self.detector_models = load_detector_catalog()

    def load_image(
        self,
        path: str | Path,
        dataset_path: str | None = None,
    ) -> DetectorImage:
        self.image = self._load_image(path, dataset_path)
        self.state.last_image_path = str(path)
        self.result = None
        return self.image

    def run_calibration(
        self,
        options: dict,
        progress: ProgressCallback | None = None,
        cancelled: CancellationCheck | None = None,
    ) -> CalibrationResult:
        if self.image is None:
            raise ValueError("No calibration image is loaded.")
        request = CalibrationRequest(image=self.image, **options)
        self.result = self._run_calibration(request, progress, cancelled)
        self.state.last_result_source = self.result.source_image
        return self.result

    def current_geometry(self, defaults: dict[str, float]) -> dict[str, float]:
        return self._apply_calibration.current_geometry(defaults)

    def apply_result(self) -> dict[str, float]:
        if self.result is None:
            raise ValueError("No calibration result is available.")
        return self._apply_calibration(self.result)

    def export_result(self, path: str | Path) -> None:
        if self.result is None:
            raise ValueError("No calibration result is available.")
        self._export_calibration(self.result, path)

    def import_result(self, path: str | Path) -> CalibrationResult:
        imported = self._import_calibration(path)
        self.result = imported.result
        self.state.last_result_source = imported.result.source_image
        if imported.image is not None:
            self.image = imported.image
            self.state.last_image_path = imported.result.source_image
        return imported.result
