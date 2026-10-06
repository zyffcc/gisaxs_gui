"""不依赖 PyQt widget 的 Geometry Calibration ViewModel。"""

from __future__ import annotations

from pathlib import Path

from src.gimap.app import AppContext
from src.gimap.shared.geometry import InstrumentProfile

from ..application import (
    ApplyCalibration,
    ExportCalibration,
    ImportCalibration,
    LoadCalibrationImage,
    LoadDetectorCatalog,
    NormalizeCalibrationPath,
    RecordInstrumentProfile,
    RunCalibration,
)
from ..application.ports import CalibrationInputFolderPort, CancellationCheck, ProgressCallback
from ..application import (
    CalibrationCandidate,
    CalibrationRequest,
    CalibrationResult,
    DetectorImage,
    commit_manual_refinement,
    detect_standard_keys,
    manual_ring_distance,
    preview_manual_candidate,
    profile_change_is_significant,
    select_calibration_candidate,
    standard_display_name,
    standard_options,
    standard_q_values,
    theoretical_ring_overlays,
)
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
        normalize_path: NormalizeCalibrationPath,
        record_profile: RecordInstrumentProfile | None = None,
        input_folders: CalibrationInputFolderPort | None = None,
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
        self._normalize_path = normalize_path
        self._record_profile = record_profile
        self._input_folders = input_folders
        self.last_profile = None
        self.detector_models = load_detector_catalog()

    def normalize_path(self, path: str | Path) -> str:
        return self._normalize_path(path)

    def last_folder(self) -> str:
        """The folder of the last calibration image read ('' when none is known)."""
        return self._input_folders.last_folder() if self._input_folders is not None else ""

    def remember_folder(self, path: str | Path) -> None:
        if self._input_folders is not None and path:
            self._input_folders.remember(path)

    @staticmethod
    def source_name(path: str | Path) -> str:
        return Path(path).name

    @staticmethod
    def default_export_path(source_path: str | Path) -> str:
        return str(Path(source_path).with_suffix(".gimap-calibration.json"))

    @staticmethod
    def standard_options():
        return standard_options()

    @staticmethod
    def detected_standard_keys(source_path: str | Path) -> tuple[str, ...]:
        return detect_standard_keys(source_path)

    @staticmethod
    def standard_display_name(key: str) -> str:
        return standard_display_name(key)

    @staticmethod
    def standard_q_values(key: str) -> tuple[float, ...]:
        return standard_q_values(key)

    def load_image(
        self,
        path: str | Path,
        dataset_path: str | None = None,
    ) -> DetectorImage:
        normalized_path = self._normalize_path(path)
        self.image = self._load_image(normalized_path, dataset_path)
        self.state.last_image_path = normalized_path
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
        geometry = self._apply_calibration(self.result)
        if self._record_profile is not None:
            self.last_profile = self._record_profile(self.result)
        return geometry

    def select_candidate(self, index: int) -> CalibrationCandidate:
        if self.result is None:
            raise ValueError("No calibration result is available.")
        return select_calibration_candidate(self.result, index)

    def display_candidate(
        self,
        *,
        manual_enabled: bool,
        center_x_px: float,
        center_y_px: float,
        distance_mm: float,
    ) -> CalibrationCandidate | None:
        if self.result is None:
            return None
        return preview_manual_candidate(
            self.result.selected_candidate,
            enabled=manual_enabled,
            center_x_px=center_x_px,
            center_y_px=center_y_px,
            distance_mm=distance_mm,
        )

    def commit_manual_refinement(
        self,
        *,
        manual_enabled: bool,
        center_x_px: float,
        center_y_px: float,
        distance_mm: float,
    ) -> CalibrationCandidate:
        if self.result is None:
            raise ValueError("No calibration result is available.")
        return commit_manual_refinement(
            self.result,
            enabled=manual_enabled,
            center_x_px=center_x_px,
            center_y_px=center_y_px,
            distance_mm=distance_mm,
        )

    def theoretical_ring_overlays(self, candidate: CalibrationCandidate):
        if self.result is None:
            return ()
        return theoretical_ring_overlays(candidate, self.result)

    def manual_ring_distance(
        self,
        experimental_radius_px: float,
        theoretical_q_inv_angstrom: float,
    ) -> float:
        if self.result is None:
            raise ValueError("No calibration result is available.")
        return manual_ring_distance(
            self.result,
            experimental_radius_px,
            theoretical_q_inv_angstrom,
        )

    def significantly_changed_profile(self) -> InstrumentProfile | None:
        """The saved instrument profile that applying the result would change a lot."""
        if self.result is None or self._record_profile is None:
            return None
        existing = self._record_profile.existing(self.result)
        if existing is None:
            return None
        if profile_change_is_significant(existing, self.result.selected_candidate):
            return existing
        return None

    def export_result(self, path: str | Path) -> None:
        if self.result is None:
            raise ValueError("No calibration result is available.")
        self._export_calibration(self.result, self._normalize_path(path))

    def import_result(self, path: str | Path) -> CalibrationResult:
        imported = self._import_calibration(self._normalize_path(path))
        self.result = imported.result
        self.state.last_result_source = imported.result.source_image
        if imported.image is not None:
            self.image = imported.image
            self.state.last_image_path = imported.result.source_image
        return imported.result
