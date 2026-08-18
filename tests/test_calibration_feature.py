from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from src.gimap.app import AppContext
from src.gimap.features.calibration.application import (
    ApplyCalibration,
    ExportCalibration,
    ImportCalibration,
    LoadCalibrationImage,
    LoadDetectorCatalog,
    NormalizeCalibrationPath,
    RunCalibration,
)
from src.gimap.features.calibration.domain import (
    MANUAL_REFINEMENT_WARNING,
    CalibrationCandidate,
    CalibrationRequest,
    CalibrationResult,
    DetectorImage,
    commit_manual_refinement,
    detect_standard_keys,
    energy_to_wavelength,
    geometry_change_is_significant,
    manual_ring_distance,
    preview_manual_candidate,
    standard_q_values,
    theoretical_ring_overlays,
)
from src.gimap.features.calibration.bootstrap import create_calibration_view_model
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)


def _result(source_path: Path) -> CalibrationResult:
    candidate = CalibrationCandidate(
        "agbh",
        123.5,
        234.5,
        1456.7,
        matched_ring_count=3,
    )
    return CalibrationResult(
        str(source_path),
        10,
        20,
        "abc",
        12.0,
        energy_to_wavelength(12.0),
        "Detector",
        172e-6,
        172e-6,
        candidate,
        [candidate],
        datetime.now(timezone.utc).isoformat(),
    )


class ImagePortStub:
    def __init__(self, image: DetectorImage, exists: bool = True):
        self.image = image
        self.source_exists = exists
        self.loaded: list[tuple[str, str | None]] = []

    def load(self, path, dataset_path=None):
        self.loaded.append((str(path), dataset_path))
        return self.image

    def exists(self, _path):
        return self.source_exists


class RunnerPortStub:
    def __init__(self, result: CalibrationResult):
        self.result = result
        self.request = None

    def calibrate(self, request, progress=None, cancelled=None):
        self.request = request
        if progress:
            progress(100, "done")
        assert cancelled is None or not cancelled()
        return self.result


class StoragePortStub:
    def __init__(self, result: CalibrationResult):
        self.result = result
        self.saved = None

    def save(self, result, path):
        self.saved = (result, str(path))

    def load(self, _path):
        return self.result


class GeometryPortStub:
    def __init__(self):
        self.applied = None
        self.saved = False

    def current_geometry(self, defaults):
        return dict(defaults)

    def apply(self, result):
        self.applied = result
        return {"distance": result.selected_candidate.distance_mm}

    def save(self):
        self.saved = True


class CatalogPortStub:
    def load(self):
        return {"Pilatus 2M": {"pixel_size_x": 172.0}}


class PathPortStub:
    def __init__(self):
        self.received = None

    def normalize(self, path):
        self.received = path
        return f"normalized/{Path(path).name}"


def test_load_and_run_calibration_use_cases_are_port_driven(tmp_path: Path) -> None:
    image = DetectorImage(
        np.ones((8, 10), dtype=np.float32),
        None,
        tmp_path / "source.cbf",
    )
    result = _result(image.source_path)
    images = ImagePortStub(image)
    runner = RunnerPortStub(result)

    loaded = LoadCalibrationImage(images)(image.source_path, "/entry/data")
    request = CalibrationRequest(image=loaded, energy_kev=12.0)
    progress = []
    actual = RunCalibration(runner)(request, lambda value, stage: progress.append((value, stage)))

    assert actual is result
    assert images.loaded == [(str(image.source_path), "/entry/data")]
    assert runner.request is request
    assert progress == [(100, "done")]


def test_import_and_export_calibration_use_cases_use_storage_ports(tmp_path: Path) -> None:
    source_path = tmp_path / "source.cbf"
    image = DetectorImage(np.ones((4, 4), dtype=np.float32), None, source_path)
    result = _result(source_path)
    storage = StoragePortStub(result)
    images = ImagePortStub(image)
    output = tmp_path / "result.json"

    ExportCalibration(storage)(result, output)
    imported = ImportCalibration(storage, images)(output)

    assert storage.saved == (result, str(output))
    assert imported.result is result
    assert imported.image is image
    assert images.loaded == [(str(source_path), None)]


def test_apply_calibration_reads_writes_and_saves_through_port(tmp_path: Path) -> None:
    result = _result(tmp_path / "source.cbf")
    parameters = GeometryPortStub()
    use_case = ApplyCalibration(parameters)
    defaults = {"distance": 1.0, "beam_center_x": 2.0, "beam_center_y": 3.0}

    assert use_case.current_geometry(defaults) == defaults
    assert use_case(result) == {"distance": 1456.7}
    assert parameters.applied is result
    assert parameters.saved


def test_load_detector_catalog_uses_catalog_port() -> None:
    assert LoadDetectorCatalog(CatalogPortStub())() == {
        "Pilatus 2M": {"pixel_size_x": 172.0}
    }


def test_normalize_calibration_path_is_port_driven(tmp_path: Path) -> None:
    paths = PathPortStub()
    source = tmp_path / "image.cbf"

    normalized = NormalizeCalibrationPath(paths)(source)

    assert paths.received == source
    assert normalized == "normalized/image.cbf"


def test_standard_detection_preserves_legacy_filename_aliases() -> None:
    assert detect_standard_keys("scan_silver_behenate_001.cbf") == ("agbh",)
    assert detect_standard_keys("scan_lab6_ceo2_001.nxs") == ("lab6", "ceo2")
    assert detect_standard_keys("unknown.cbf") == ()


def test_ring_geometry_preserves_overlay_and_manual_distance_results(tmp_path: Path) -> None:
    result = _result(tmp_path / "source.cbf")
    overlays = theoretical_ring_overlays(result.selected_candidate, result)

    assert len(overlays) == len(standard_q_values("agbh"))
    radius_px = overlays[0].width_px / 2.0
    recovered = manual_ring_distance(result, radius_px, standard_q_values("agbh")[0])
    assert recovered == pytest.approx(result.selected_candidate.distance_mm)


def test_manual_refinement_preview_commit_and_significant_change_are_stable(
    tmp_path: Path,
) -> None:
    result = _result(tmp_path / "source.cbf")
    original = result.selected_candidate

    preview = preview_manual_candidate(
        original,
        enabled=True,
        center_x_px=130.0,
        center_y_px=240.0,
        distance_mm=1500.0,
    )

    assert preview is not original
    assert original.center_x_px == 123.5
    assert preview.center_x_px == 130.0
    committed = commit_manual_refinement(
        result,
        enabled=True,
        center_x_px=130.0,
        center_y_px=240.0,
        distance_mm=1500.0,
    )
    commit_manual_refinement(
        result,
        enabled=True,
        center_x_px=130.0,
        center_y_px=240.0,
        distance_mm=1500.0,
    )
    assert committed is original
    assert committed.warnings.count(MANUAL_REFINEMENT_WARNING) == 1
    assert geometry_change_is_significant(
        {"distance": 1000.0, "beam_center_x": 130.0, "beam_center_y": 240.0},
        committed,
    )


def test_calibration_applies_with_in_memory_context_and_no_qapplication(tmp_path: Path) -> None:
    settings = InMemorySettingsRepository(
        {
            "fitting": {
                "detector": {
                    "distance": 2000.0,
                    "beam_center_x": 100.0,
                    "beam_center_y": 200.0,
                }
            }
        }
    )
    context = AppContext(
        settings=settings,
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
    )
    view_model = create_calibration_view_model(context)
    view_model.result = _result(tmp_path / "source.cbf")

    geometry = view_model.apply_result()

    assert geometry["distance"] == 1456.7
    assert settings.get("fitting", "detector.beam_center_x") == 123.5
