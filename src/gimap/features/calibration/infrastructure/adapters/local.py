"""Calibration ports 的本地及 legacy adapters。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...domain.engine import (
    CalibrationCancelled as LegacyCalibrationCancelled,
    CalibrationEngine,
)
from src.gimap.shared.detector_io import (
    AmbiguousDatasetError as LegacyAmbiguousDatasetError,
    load_detector_image,
)
from src.gimap.app import SettingsRepository
from src.gimap.shared.file_paths import normalize_path

from ...application.errors import (
    AmbiguousImageDatasetError,
    CalibrationCancelledError,
)
from ...application.ports import CancellationCheck, ProgressCallback
from ...domain import CalibrationRequest, CalibrationResult, DetectorImage


class LocalCalibrationImageAdapter:
    """通过现有共享 detector loader 读取 CBF / NXS。"""

    def load(
        self,
        path: str | Path,
        dataset_path: str | None = None,
    ) -> DetectorImage:
        try:
            return load_detector_image(path, dataset_path=dataset_path)
        except LegacyAmbiguousDatasetError as exc:
            raise AmbiguousImageDatasetError(exc.paths) from exc

    def exists(self, path: str | Path) -> bool:
        return Path(path).is_file()


class LocalCalibrationPathAdapter:
    """Normalize user-selected paths at the filesystem boundary."""

    def normalize(self, path: str | Path) -> str:
        return normalize_path(path)


class LegacyCalibrationRunnerAdapter:
    """在 port 后复用已建立数值回归基线的 CalibrationEngine。"""

    def calibrate(
        self,
        request: CalibrationRequest,
        progress: ProgressCallback | None = None,
        cancelled: CancellationCheck | None = None,
    ) -> CalibrationResult:
        engine = CalibrationEngine(progress=progress, cancelled=cancelled)
        try:
            return engine.calibrate(request.image, **request.algorithm_options())
        except LegacyCalibrationCancelled as exc:
            raise CalibrationCancelledError(str(exc)) from exc


class JsonCalibrationStorageAdapter:
    """GIMaP calibration JSON v1 的本地存储实现。"""

    def save(self, result: CalibrationResult, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> CalibrationResult:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "gimap-geometry-calibration":
            raise ValueError("This file is not a GIMaP geometry calibration.")
        if int(payload.get("format_version", 0)) != 1:
            raise ValueError("Unsupported geometry calibration file version.")
        return CalibrationResult.from_dict(payload)


class JsonDetectorCatalogAdapter:
    """从现有 config/detectors.json 读取 detector definitions。"""

    def __init__(self, path: str | Path | None = None):
        project_root = Path(__file__).resolve().parents[6]
        self.path = Path(path) if path is not None else project_root / "config" / "detectors.json"

    def load(self) -> dict[str, dict[str, Any]]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}


class SettingsGeometryAdapter:
    """通过 SettingsRepository 读取并保存共享 geometry settings。"""

    def __init__(self, settings: SettingsRepository):
        self.settings = settings

    def current_geometry(self, defaults: dict[str, float]) -> dict[str, float]:
        return {
            "distance": float(
                self.settings.get(
                    "fitting",
                    "detector.distance",
                    defaults["distance"],
                )
            ),
            "beam_center_x": float(
                self.settings.get(
                    "fitting",
                    "detector.beam_center_x",
                    defaults["beam_center_x"],
                )
            ),
            "beam_center_y": float(
                self.settings.get(
                    "fitting",
                    "detector.beam_center_y",
                    defaults["beam_center_y"],
                )
            ),
        }

    def apply(self, result: CalibrationResult) -> dict[str, float]:
        candidate = result.selected_candidate
        geometry = {
            "distance": float(candidate.distance_mm),
            "pixel_size_x": float(result.pixel_size_x_m * 1e6),
            "pixel_size_y": float(result.pixel_size_y_m * 1e6),
            "beam_center_x": float(candidate.center_x_px),
            "beam_center_y": float(candidate.center_y_px),
        }
        for key, value in geometry.items():
            self.settings.set("detector", key, value)
            self.settings.set("fitting", f"detector.{key}", value)
        self.settings.set(
            "detector",
            "rotation_deg",
            float(candidate.detector_rotation_deg),
        )
        self.settings.set(
            "beam",
            "wavelength",
            float(result.wavelength_angstrom / 10.0),
        )
        self.settings.set("beam", "energy_kev", float(result.energy_kev))
        self.settings.set(
            "system",
            "geometry_calibration",
            {
                "source_image": result.source_image,
                "timestamp": result.calibration_timestamp,
                "standard": candidate.standard_key,
                "confidence": candidate.confidence,
                "residual_px": candidate.rms_residual_px,
            },
        )
        return geometry

    def save(self) -> None:
        self.settings.save()
