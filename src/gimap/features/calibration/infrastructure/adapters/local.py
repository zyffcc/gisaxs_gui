"""Calibration ports 的本地及 legacy adapters。"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ...domain.engine import (
    CalibrationCancelled as LegacyCalibrationCancelled,
    CalibrationEngine,
)
from src.gimap.shared.detector_io import (
    AmbiguousDatasetError as LegacyAmbiguousDatasetError,
    is_hdf5,
    load_detector_image,
    open_nexus,
    select_nxs_dataset,
)
from src.gimap.app import SettingsRepository
from src.gimap.shared.file_paths import normalize_path

from ...application.errors import (
    AmbiguousImageDatasetError,
    CalibrationCancelledError,
)
from ...application.ports import CancellationCheck, FrameProgress, ProgressCallback
from ...domain import CalibrationRequest, CalibrationResult, DetectorImage


class LocalCalibrationImageAdapter:
    """通过现有共享 detector loader 读取 CBF / NeXus / HDF5 / TIFF / EDF。

    ``ask_dataset``: a file with several equally plausible image datasets (two scans in one file) raises
    ``AmbiguousImageDatasetError`` so that the dialog asks which one is the standard. Without it (no one
    to ask: the assistant) the dataset every other reader shows is read, and ``metadata["layout_warnings"]``
    says that there are others.
    """

    def __init__(self, ask_dataset: bool = True):
        self.ask_dataset = ask_dataset

    def load(
        self,
        path: str | Path,
        dataset_path: str | None = None,
        *,
        frame: int = 0,
        sum_frames: int = 1,
        progress: FrameProgress | None = None,
        cancelled: CancellationCheck | None = None,
    ) -> DetectorImage:
        if dataset_path is None and self.ask_dataset and is_hdf5(path):
            self._ask_when_ambiguous(path)
        first = max(0, int(frame))
        image = load_detector_image(path, frame_idx=first, dataset_path=dataset_path)
        total = max(1, int(image.metadata.get("frame_count") or 1))
        last = min(total, first + max(1, int(sum_frames))) - 1
        metadata = {**image.metadata, "frame_index": first, "summed_frames": last - first + 1}
        if last == first:
            return replace(image, metadata=metadata)
        # Photon counts add. A pixel without a value in any frame has none in the sum (NaN, and masked).
        data = np.array(image.data, dtype=np.float64)
        mask = None if image.mask is None else np.array(image.mask, dtype=bool)
        if progress is not None:
            progress(1, last - first + 1)
        for index in range(first + 1, last + 1):
            if cancelled is not None and cancelled():
                raise CalibrationCancelledError("Reading cancelled.")
            other = load_detector_image(path, frame_idx=index, dataset_path=dataset_path)
            data += other.data
            if other.mask is not None:
                mask = np.array(other.mask, dtype=bool) if mask is None else mask | np.asarray(other.mask, dtype=bool)
            if progress is not None:
                progress(index - first + 1, last - first + 1)
        metadata["summed_frame_indices"] = [first, last]
        return replace(image, data=data.astype(np.float32), mask=mask, metadata=metadata)

    @staticmethod
    def _ask_when_ambiguous(path: str | Path) -> None:
        try:
            with open_nexus(path) as handle:
                select_nxs_dataset(handle)
        except LegacyAmbiguousDatasetError as exc:
            raise AmbiguousImageDatasetError(exc.paths) from exc
        except Exception:  # noqa: BLE001 - a file that cannot be read: reading it says why
            return

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
    """GIMaP calibration JSON v1 的本地存储实现。

    The beam centres in the file are pixel-centre indices (the centre of the first pixel is 0.0), as in every
    earlier export; ``metadata["center_convention"]`` names that convention (``CalibrationResult.to_dict``).
    """

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


def _image_shape(result: CalibrationResult) -> list[int] | None:
    shape = (result.metadata or {}).get("image_shape")
    try:
        rows, columns = (int(value) for value in shape)
    except (TypeError, ValueError):
        return None
    return [rows, columns]


def _stored_number(value: Any, default: float) -> float:
    """A number of the settings file; ``default`` for what a damaged file holds in its place (null, text, inf)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


class SettingsGeometryAdapter:
    """Keep the last applied calibration in the shared settings.

    ``detector.*`` holds the geometry in calibration convention (numpy pixel
    indices, row 0 at the top) and ``beam.*`` the energy; the instrument
    profile Analyze uses is recorded separately (see ``RecordInstrumentProfile``).
    """

    def __init__(self, settings: SettingsRepository):
        self.settings = settings

    def current_geometry(self, defaults: dict[str, float]) -> dict[str, float]:
        """The last applied geometry; ``defaults`` when no calibration was ever applied.

        ``detector.*`` holds numbers on every installation: the settings store fills in the built-in
        geometry of the simulation pages (2000 mm, centre 737), which nobody measured. ``apply`` writes
        ``system.geometry_calibration`` together with the geometry, so that entry tells an applied
        geometry from those numbers (the XRR window reads the settings the same way). Settings carried
        over from an earlier version can hold an applied geometry without the entry: it counts as never
        applied, since nothing says any more which image those numbers belong to.
        """
        keys = ("distance", "beam_center_x", "beam_center_y")
        if not isinstance(self.settings.get("system", "geometry_calibration", None), dict):
            return {key: float(defaults[key]) for key in keys}
        return {key: _stored_number(self.settings.get("detector", key, None), float(defaults[key])) for key in keys}

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
                # Which frames the geometry belongs to (rows, columns), for tools that reuse it.
                "image_shape": _image_shape(result),
                "detector": result.detector_name or "",
            },
        )
        return geometry

    def save(self) -> None:
        self.settings.save()


class PreferencesCalibrationFolderAdapter:
    """The folder of the last calibration image, in the user preferences."""

    KEY = "calibration.last_folder"

    def __init__(self, preferences, key: str = KEY):
        self.preferences = preferences
        self.key = key

    def last_folder(self) -> str:
        try:
            folder = str(self.preferences.get(self.key, "") or "")
        except Exception:
            return ""
        return folder if folder and Path(folder).is_dir() else ""

    def remember(self, path: str | Path) -> None:
        if not path:
            return
        location = Path(path)
        folder = location if location.is_dir() else location.parent
        try:
            self.preferences.set(self.key, str(folder))
            self.preferences.save()
        except OSError:
            pass
