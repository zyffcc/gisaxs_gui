"""Geometry Calibration composition root。"""

from src.gimap.app import AppContext

from .application import (
    ApplyCalibration,
    ExportCalibration,
    HeadlessCalibration,
    ImportCalibration,
    LoadCalibrationImage,
    LoadDetectorCatalog,
    NormalizeCalibrationPath,
    RecordInstrumentProfile,
    RunCalibration,
)
from .infrastructure.adapters import (
    JsonCalibrationStorageAdapter,
    JsonDetectorCatalogAdapter,
    LegacyCalibrationRunnerAdapter,
    LocalCalibrationImageAdapter,
    LocalCalibrationPathAdapter,
    SettingsGeometryAdapter,
)
from .presentation import CalibrationViewModel


def create_calibration_view_model(app_context: AppContext) -> CalibrationViewModel:
    images = LocalCalibrationImageAdapter()
    storage = JsonCalibrationStorageAdapter()
    profiles = getattr(app_context, "instrument_profiles", None)
    return CalibrationViewModel(
        app_context=app_context,
        load_image=LoadCalibrationImage(images),
        run_calibration=RunCalibration(LegacyCalibrationRunnerAdapter()),
        export_calibration=ExportCalibration(storage),
        import_calibration=ImportCalibration(storage, images),
        apply_calibration=ApplyCalibration(SettingsGeometryAdapter(app_context.settings)),
        load_detector_catalog=LoadDetectorCatalog(JsonDetectorCatalogAdapter()),
        normalize_path=NormalizeCalibrationPath(LocalCalibrationPathAdapter()),
        record_profile=RecordInstrumentProfile(profiles) if profiles is not None else None,
    )


def create_headless_calibration() -> HeadlessCalibration:
    """Calibration for another component (the assistant): fit or read, never save."""
    images = LocalCalibrationImageAdapter()
    return HeadlessCalibration(
        load_image=LoadCalibrationImage(images),
        run_calibration=RunCalibration(LegacyCalibrationRunnerAdapter()),
        import_calibration=ImportCalibration(JsonCalibrationStorageAdapter(), images),
    )
