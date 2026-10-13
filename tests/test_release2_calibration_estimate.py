"""Geometry Calibration never offers a distance nobody measured as the estimate.

"Estimated distance" weighs heavily in the ranking of the solutions. Until 0.1.0 the window filled it, on a new
installation, with the 2000 mm of the built-in simulation geometry: on the public silver behenate frame that
alone put a solution near 1830 mm first, and the right one (about 1630 mm) was not in the table.

When an image is opened the field now takes, in this order:

- the distance the image file states (a NeXus ``/entry/instrument/detector/distance``; the distances of CBF
  and EDF headers are often stale and are not used, see ``shared/detector_io/metadata.py``), when the field
  can show it (up to 100 m: it is never turned into the field's maximum);
- the distance of the calibration applied last. The settings store holds ``detector.distance`` on every
  installation (the built-in 2000 mm is merged in and written back with every save), so the number alone
  says nothing: ``system.geometry_calibration``, which Apply writes with the geometry, tells the two apart.
  Settings of an earlier version that lost this record are treated as never applied;
- with neither, a value the user typed stays, and a value the window filled in for an earlier image goes
  back to 0, which the field shows as "Optional" and the search reads as "no estimate".

The search and its ranking are not part of this: what Auto Detect does on that frame is pinned in
``tests/test_release2_docs_decisions.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest
from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtWidgets import QApplication

from src.gimap.app import AppContext
from src.gimap.app.presentation import i18n
from src.gimap.features.calibration.application import ApplyCalibration
from src.gimap.features.calibration.bootstrap import create_calibration_view_model
from src.gimap.features.calibration.domain import CalibrationCandidate, CalibrationResult, DetectorImage
from src.gimap.features.calibration.infrastructure.adapters import SettingsGeometryAdapter
from src.gimap.features.calibration.presentation.dialog import GeometryCalibrationDialog
from src.gimap.integrations.state import (
    InMemoryInstrumentProfileRepository,
    InMemorySessionRepository,
    StorePreferencesRepository,
    StoreSettingsRepository,
    UserStore,
    migrate_legacy_files,
)

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "tests" / "data" / "external"
AGBH = EXTERNAL / "saxs_agbh_pyfai" / "Pilatus1M.edf"
UNKNOWN = {"distance": 0.0, "beam_center_x": 0.0, "beam_center_y": 0.0}
APPLIED_MM = 1456.7  # the made-up distance of ``_result``


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _settle(app, seconds: float = 0.05) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def _wait(app, done, seconds: float) -> None:
    end = time.monotonic() + seconds
    while not done():
        assert time.monotonic() < end, "timed out"
        _settle(app)


def _dispose(dialog) -> None:
    """Close the window and delete it now, by the event loop (not inside a garbage collection)."""
    for name in ("_load_thread", "_cal_thread"):
        thread = getattr(dialog, name, None)
        if thread is not None:
            thread.wait(10000)
    dialog.close()
    dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def _settings(home: Path) -> StoreSettingsRepository:
    """The settings as the application opens them: the user's file over the built-in values."""
    return StoreSettingsRepository(UserStore(home / "settings.json"))


def _installation(home: Path) -> AppContext:
    """The context of an installation whose user data folder is ``home`` (empty: a new installation)."""
    store = UserStore(home / "settings.json")
    return AppContext(
        settings=StoreSettingsRepository(store),
        preferences=StorePreferencesRepository(store),
        session=InMemorySessionRepository(),
        instrument_profiles=InMemoryInstrumentProfileRepository(),
    )


def _result(source: Path) -> CalibrationResult:
    """A made-up result (no detector data): what Apply stores is what is looked at."""
    candidate = CalibrationCandidate("agbh", 123.5, 234.5, APPLIED_MM, matched_ring_count=3, rms_residual_px=1.25)
    return CalibrationResult(
        str(source), 10, 20, "abc", 12.0, 12.398419843320026 / 12.0, "Pilatus 1M", 172e-6, 172e-6,
        candidate, [candidate], datetime.now(timezone.utc).isoformat(), metadata={"image_shape": [96, 120]},
    )


def _frame(name: str = "standard_frame.cbf", distance_m: float | None = None) -> DetectorImage:
    """A frame as the loader hands it over; ``distance_m`` is what its file states."""
    data = np.ones((96, 120), dtype=np.float32)
    return DetectorImage(data, np.zeros(data.shape, dtype=bool), Path(name), distance_m=distance_m)


def _shown(dialog) -> tuple[float, str]:
    spin = dialog.estimated_distance_spin
    return spin.value(), spin.text()


@pytest.fixture
def window(tmp_path):
    """Geometry Calibration on a new installation."""
    _app()
    dialog = GeometryCalibrationDialog(app_context=_installation(tmp_path / "home"))
    yield dialog
    _dispose(dialog)


# -- which geometry is one somebody applied ------------------------------------------------------------


def test_the_built_in_geometry_is_not_a_geometry_in_use(tmp_path: Path) -> None:
    home = tmp_path / "home"
    settings = _settings(home)
    assert settings.get("detector", "distance") == 2000  # the simulation default: every installation has it
    assert settings.get("system", "geometry_calibration") is None
    assert SettingsGeometryAdapter(settings).current_geometry(UNKNOWN) == UNKNOWN
    # The caller's own fallback comes back, whatever it is.
    fallback = {"distance": 1.0, "beam_center_x": 2.0, "beam_center_y": 3.0}
    assert SettingsGeometryAdapter(settings).current_geometry(fallback) == fallback

    # Anything that saves the settings writes the built-in number into the user's file: from then on it is a
    # stored value like any other, which is why the number itself cannot be asked whether it was measured.
    settings.save()
    assert json.loads((home / "settings.json").read_text(encoding="utf-8"))["settings"]["detector"]["distance"] == 2000
    assert SettingsGeometryAdapter(_settings(home)).current_geometry(UNKNOWN) == UNKNOWN


def test_an_applied_calibration_is_the_geometry_in_use_in_the_next_session_too(tmp_path: Path) -> None:
    home = tmp_path / "home"
    ApplyCalibration(SettingsGeometryAdapter(_settings(home)))(_result(tmp_path / "standard_frame.cbf"))

    reopened = _settings(home)
    assert isinstance(reopened.get("system", "geometry_calibration"), dict)  # written by Apply, with the geometry
    assert SettingsGeometryAdapter(reopened).current_geometry(UNKNOWN) == {
        "distance": APPLIED_MM, "beam_center_x": 123.5, "beam_center_y": 234.5,
    }


def test_a_calibration_applied_at_the_built_in_distance_is_still_one(tmp_path: Path) -> None:
    """Told by the record of the Apply, not by comparing the number with the default."""
    result = _result(tmp_path / "standard_frame.cbf")
    result.selected_candidate.distance_mm = 2000.0
    settings = _settings(tmp_path / "home")
    ApplyCalibration(SettingsGeometryAdapter(settings))(result)
    assert SettingsGeometryAdapter(settings).current_geometry(UNKNOWN)["distance"] == 2000.0


def test_settings_of_an_earlier_version_without_the_record_of_the_apply_are_not_a_geometry_in_use(tmp_path: Path) -> None:
    """``config/user_parameters.json`` of a version before the user folder, imported as the application does.

    Such a file can hold what an Apply wrote to ``detector`` (also ``rotation_deg``, which only Apply writes)
    while ``system`` no longer has the record of it. Without the record nothing says which image, detector or
    day the numbers belong to, so they are not offered as the estimate (made-up values).
    """
    program = tmp_path / "program"
    (program / "config").mkdir(parents=True)
    legacy = {
        "detector": {"preset": "Pilatus 2M", "distance": 3210.5, "pixel_size_x": 172, "pixel_size_y": 172,
                     "beam_center_x": 611.2, "beam_center_y": 388.9, "rotation_deg": 0.0},
        "beam": {"wavelength": 0.1, "energy_kev": 12.0},
        "system": {"calculation_method": "DWBA", "num_threads": 4},
    }
    (program / "config" / "user_parameters.json").write_text(json.dumps(legacy), encoding="utf-8")
    home = tmp_path / "home"
    assert migrate_legacy_files(home, project_root=program)  # the first start of this version

    settings = _settings(home)
    assert settings.get("detector", "distance") == 3210.5 and settings.get("detector", "rotation_deg") == 0.0
    assert SettingsGeometryAdapter(settings).current_geometry(UNKNOWN) == UNKNOWN
    _app()
    dialog = GeometryCalibrationDialog(app_context=_installation(home))
    try:
        dialog._image_loaded(_frame())
        assert _shown(dialog) == (0.0, "Optional")
        # The next Apply writes the record: from then on its distance is the estimate.
        dialog.view_model.result = _result(tmp_path / "standard_frame.cbf")
        dialog.view_model.apply_result()
        dialog._image_loaded(_frame("second.cbf"))
        assert _shown(dialog) == (APPLIED_MM, "1456.70 mm")
    finally:
        _dispose(dialog)


@pytest.mark.parametrize("damaged", [None, "far", float("inf"), float("nan"), -3.0, 0, 1e9, {"mm": 1456.7}])
def test_a_damaged_distance_of_the_applied_geometry_is_no_estimate_and_does_not_stop_the_window(
    tmp_path: Path, damaged
) -> None:
    """A settings file edited by hand or cut off: the image is still shown and the window stays usable."""
    home = tmp_path / "home"
    settings = _settings(home)
    ApplyCalibration(SettingsGeometryAdapter(settings))(_result(tmp_path / "standard_frame.cbf"))
    settings.set("detector", "distance", damaged)
    settings.set("detector", "beam_center_x", "left")
    settings.save()
    geometry = SettingsGeometryAdapter(_settings(home)).current_geometry(UNKNOWN)
    assert geometry["beam_center_x"] == 0.0 and geometry["beam_center_y"] == 234.5  # what is readable is read
    numeric = isinstance(damaged, (int, float)) and np.isfinite(damaged)
    assert geometry["distance"] == (float(damaged) if numeric else 0.0)

    _app()
    dialog = GeometryCalibrationDialog(app_context=_installation(home))
    try:
        dialog._set_running(True)  # as ``load_image`` leaves the window while the file is read
        dialog._image_loaded(_frame())
        assert _shown(dialog) == (0.0, "Optional")
        assert dialog.open_button.isEnabled() and dialog.calibrate_button.isEnabled()
    finally:
        _dispose(dialog)


# -- what the window puts in the field -----------------------------------------------------------------


def test_a_new_installation_leaves_the_estimate_open(window) -> None:
    assert _shown(window) == (0.0, "Optional")  # before any image
    window._image_loaded(_frame())
    assert _shown(window) == (0.0, "Optional")
    # 0 is "no estimate" for the search, not a distance of 0 mm.
    assert (window.estimated_distance_spin.value() or None) is None
    source = (ROOT / "src/gimap/features/calibration/presentation/bindings/calibration_run.py").read_text("utf-8")
    assert '"estimated_distance_mm": self.estimated_distance_spin.value() or None' in source


def test_the_open_estimate_reads_the_same_in_chinese(window) -> None:
    window._image_loaded(_frame())
    try:
        i18n.apply_language("zh", roots=[window])
        assert window.estimatedDistanceLabel.text() == "估计距离："
        assert window.estimated_distance_spin.text() == "可选"
        window.estimated_distance_spin.setValue(1600.0)
        assert window.estimated_distance_spin.text() == "1600.00 mm"  # a number and its unit are not translated
    finally:
        i18n.apply_language("en", roots=[window])
    window.estimated_distance_spin.setValue(0.0)
    assert window.estimated_distance_spin.text() == "Optional"


def test_the_distance_a_file_states_fills_the_field_and_does_not_stay_for_the_next_file(window, tmp_path: Path) -> None:
    """Through the loader: a NeXus file with ``/entry/instrument/detector/distance`` (made-up values)."""
    path = tmp_path / "standard_frame.nxs"
    with h5py.File(path, "w") as handle:
        detector = handle.create_group("entry/instrument/detector")
        detector.create_dataset("data", data=np.ones((96, 120), dtype=np.float32))
        detector.create_dataset("distance", data=1250.0).attrs["units"] = "mm"
    stated = window.view_model.load_image(path)
    assert stated.distance_m == pytest.approx(1.25)

    window._image_loaded(stated)
    assert _shown(window) == (1250.0, "1250.00 mm")
    # The next file states none: 1250 mm belonged to the other file and is not offered for this one.
    window._image_loaded(_frame())
    assert _shown(window) == (0.0, "Optional")


def test_a_distance_the_user_typed_stays_until_a_file_states_its_own(window) -> None:
    window.estimated_distance_spin.setValue(1600.0)  # typed before the first image is opened
    window._image_loaded(_frame("first.cbf"))
    assert _shown(window) == (1600.0, "1600.00 mm")
    window._image_loaded(_frame("second.cbf"))
    assert _shown(window) == (1600.0, "1600.00 mm")
    window._image_loaded(_frame("third.nxs", distance_m=0.85))  # the file's own distance comes first
    assert _shown(window) == (850.0, "850.00 mm")
    # Corrected by hand, the number is the user's again; so is a field the user emptied.
    window.estimated_distance_spin.setValue(900.0)
    window._image_loaded(_frame("fourth.cbf"))
    assert _shown(window) == (900.0, "900.00 mm")
    window.estimated_distance_spin.setValue(0.0)
    window._image_loaded(_frame("fifth.cbf"))
    assert _shown(window) == (0.0, "Optional")


@pytest.mark.parametrize("stated", [0.0, -1.0, float("nan")])
def test_a_distance_that_is_none_is_no_estimate(window, stated) -> None:
    window._image_loaded(_frame(distance_m=stated))
    assert _shown(window) == (0.0, "Optional")


@pytest.mark.parametrize("stated_m", [1250.0, 100.01, float("inf"), 4e-6])
def test_a_stated_distance_the_field_cannot_show_is_not_offered(window, tmp_path: Path, stated_m) -> None:
    """The field ends at 100 m and has two decimals. It used to show its maximum, "100000.00 mm", for more:
    a distance that stands in no file, sent to the search as the estimate."""
    assert window.estimated_distance_spin.maximum() == 100000.0
    window._image_loaded(_frame(distance_m=stated_m))
    assert _shown(window) == (0.0, "Optional")
    window.estimated_distance_spin.setValue(1600.0)  # what the user typed stays, as with a file that states none
    window._image_loaded(_frame("second.nxs", distance_m=stated_m))
    assert _shown(window) == (1600.0, "1600.00 mm")
    window._image_loaded(_frame("third.nxs", distance_m=100.0))  # the largest distance the field shows
    assert _shown(window) == (100000.0, "100000.00 mm")
    # The distance of the applied calibration is the next source, as for a file that states none.
    window.view_model.result = _result(tmp_path / "standard_frame.cbf")
    window.view_model.apply_result()
    window._image_loaded(_frame("fourth.nxs", distance_m=stated_m))
    assert _shown(window) == (APPLIED_MM, "1456.70 mm")


def test_a_nexus_distance_without_its_unit_is_not_turned_into_100_m(window, tmp_path: Path) -> None:
    """Through the loader, which reads a NeXus distance without ``units`` as metres (made-up values)."""
    path = tmp_path / "standard_frame.nxs"
    with h5py.File(path, "w") as handle:
        detector = handle.create_group("entry/instrument/detector")
        detector.create_dataset("data", data=np.ones((96, 120), dtype=np.float32))
        detector.create_dataset("distance", data=1250.0)  # millimetres were meant; the file does not say so
    stated = window.view_model.load_image(path)
    assert stated.distance_m == pytest.approx(1250.0)
    window._image_loaded(stated)
    assert _shown(window) == (0.0, "Optional")
    assert (window.estimated_distance_spin.value() or None) is None  # no estimate goes to the search


def test_a_user_who_applied_a_calibration_gets_its_distance(tmp_path: Path) -> None:
    _app()
    home = tmp_path / "home"
    first = create_calibration_view_model(_installation(home))
    first.result = _result(tmp_path / "standard_frame.cbf")
    first.apply_result()

    dialog = GeometryCalibrationDialog(app_context=_installation(home))  # the next session
    try:
        assert _shown(dialog) == (0.0, "Optional")  # nothing is filled in before an image is opened
        dialog._image_loaded(_frame("first.cbf"))
        assert _shown(dialog) == (APPLIED_MM, "1456.70 mm")
        dialog._image_loaded(_frame("second.nxs", distance_m=0.85))  # the file's own distance comes first
        assert _shown(dialog) == (850.0, "850.00 mm")
        dialog._image_loaded(_frame("third.cbf"))
        assert _shown(dialog) == (APPLIED_MM, "1456.70 mm")
        # As before this change, each image starts from the applied distance, also after the user typed another.
        dialog.estimated_distance_spin.setValue(900.0)
        dialog._image_loaded(_frame("fourth.cbf"))
        assert _shown(dialog) == (APPLIED_MM, "1456.70 mm")
    finally:
        _dispose(dialog)


def test_applying_in_the_window_makes_its_distance_the_estimate_of_the_next_image(window, tmp_path: Path) -> None:
    window._image_loaded(_frame("first.cbf"))
    assert _shown(window) == (0.0, "Optional")
    window.view_model.result = _result(tmp_path / "first.cbf")
    window.view_model.apply_result()
    window._image_loaded(_frame("second.cbf"))
    assert _shown(window) == (APPLIED_MM, "1456.70 mm")


# -- the case that was reported -------------------------------------------------------------------------


def test_the_reported_frame_is_solved_on_a_new_installation(window, tmp_path: Path) -> None:
    """The public pyFAI silver behenate frame, in the window: the standard chosen by hand, the field never touched.

    With the 2000 mm the window used to fill in, a solution near 1830 mm came first (that part of the search
    is pinned in ``tests/test_release2_docs_decisions.py``). The name of this test names no standard on
    purpose: the folder pytest makes for it is part of the image's path, where the window looks for one.
    """
    if not AGBH.is_file():
        pytest.skip("the public silver behenate image is not in this checkout")
    manifest = json.loads((EXTERNAL / "manifest.json").read_text(encoding="utf-8"))["saxs_agbh_pyfai"]
    expected_mm = float(manifest["ring_distance_mm_at_stated_wavelength"])
    expected_x, expected_y = manifest["beam_center_px"]
    frame = tmp_path / "standard_image.edf"
    shutil.copyfile(AGBH, frame)

    app = _app()
    window.load_image(str(frame))
    _wait(app, lambda: window._load_thread is None and window.image is not None, 60)
    assert window.image.distance_m is None  # an EDF header's distance is not used
    assert _shown(window) == (0.0, "Optional")
    assert window.standard_combo.currentData() == "auto"  # the name gives no standard: chosen by hand below
    window.energy_spin.setValue(12.398419843 / float(manifest["wavelength_angstrom"]))
    window.standard_combo.setCurrentIndex(window.standard_combo.findData("agbh"))
    window.start_calibration()
    _wait(app, lambda: window._cal_thread is None and window.result is not None, 240)

    assert window.result.metadata["estimated_distance_mm"] is None  # the search was given no estimate
    best = window.result.selected_candidate
    assert best.standard_key == "agbh"
    assert abs(best.distance_mm - expected_mm) <= 0.03 * expected_mm
    # The table's first row, in the pixels the window shows (0.5 more than the fit's pixel-centre indices).
    assert abs(best.center_x_px + 0.5 - expected_x) < 5.0 and abs(best.center_y_px + 0.5 - expected_y) < 5.0
    assert window.candidate_table.item(0, 1).text() == f"{best.distance_mm:.2f} mm"
