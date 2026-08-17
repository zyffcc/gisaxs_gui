"""Geometry Calibration composition root。"""

from src.gimap.app import AppContext

from .application import (
    ApplyCalibration,
    ExportCalibration,
    ImportCalibration,
    LoadCalibrationImage,
    LoadDetectorCatalog,
    RunCalibration,
)
from .infrastructure.adapters import (
    JsonCalibrationStorageAdapter,
    JsonDetectorCatalogAdapter,
    LegacyCalibrationRunnerAdapter,
    LocalCalibrationImageAdapter,
    SettingsGeometryAdapter,
)
from .presentation import CalibrationViewModel


def create_calibration_view_model(app_context: AppContext) -> CalibrationViewModel:
    images = LocalCalibrationImageAdapter()
    storage = JsonCalibrationStorageAdapter()
    return CalibrationViewModel(
        app_context=app_context,
        load_image=LoadCalibrationImage(images),
        run_calibration=RunCalibration(LegacyCalibrationRunnerAdapter()),
        export_calibration=ExportCalibration(storage),
        import_calibration=ImportCalibration(storage, images),
        apply_calibration=ApplyCalibration(SettingsGeometryAdapter(app_context.settings)),
        load_detector_catalog=LoadDetectorCatalog(JsonDetectorCatalogAdapter()),
    )
