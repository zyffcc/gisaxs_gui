"""SEM integration boundaries: explicit units, immutable handoffs and auditable exports.

Only synthetic images and measurements are used. A model runtime or weight file
must never be needed by the standalone SEM tool.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from PyQt5.QtWidgets import QFileDialog, QWidget

from src.gimap.app.menus import ApplicationMenus
from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.menu_bar import ToolWindows
from src.gimap.features.prediction.domain import PredictionModule, PreprocessSpec
from src.gimap.features.prediction.presentation.bindings.sem_comparison import SemComparisonMixin
from src.gimap.features.sem.application import (
    SemComparisonSettings,
    SemCsvSettings,
    SemExtractionSettings,
)
from src.gimap.features.sem.bootstrap import create_sem_service, create_sem_window


def _synthetic_image(path: Path) -> np.ndarray:
    values = np.zeros((40, 48), dtype=np.uint8)
    values[6:10, 7:11] = 200
    values[20:23, 28:31] = 200
    Image.fromarray(values).save(path)
    return values


def _wait(app, condition, timeout=8):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "SEM background operation timed out"
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()


def test_standalone_image_pipeline_exports_science_and_source_without_model(tmp_path, monkeypatch):
    import sys

    # Importing the optional model runtime here is an integration error.
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    source = tmp_path / "synthetic_sem.png"
    values = _synthetic_image(source)
    original_bytes = source.read_bytes()
    service = create_sem_service()
    image = service.load_image(source)
    np.testing.assert_array_equal(image.image, values)

    measurements = service.extract(
        image, SemExtractionSettings(nm_per_pixel=0.5, threshold=100, min_area_px=1),
    )
    np.testing.assert_allclose(
        np.sort(measurements.radius_nm), np.sqrt(np.array([9, 16]) / np.pi) * 0.5,
    )
    assert [(p.centroid_x_px, p.centroid_y_px) for p in measurements.particles] == [
        (9.0, 8.0), (29.5, 21.5),
    ]
    comparison = service.compare(measurements, None, SemComparisonSettings(bins=8))
    assert comparison.metadata["prediction_included"] is False
    assert comparison.metadata["sem_radius_scale"] == 1.0
    assert comparison.prediction_probability is None
    assert comparison.sem_probability.sum() == pytest.approx(1.0)

    output = tmp_path / "export"
    service.export(measurements, comparison, output)
    records = list(output.glob("*.json"))
    assert records, "SEM exports must include a machine-readable source/settings record"
    record_text = "\n".join(path.read_text(encoding="utf-8") for path in records)
    assert json.dumps(str(source.resolve())) in record_text  # as JSON writes it (a Windows path escapes its \)
    assert hashlib.sha256(original_bytes).hexdigest() in record_text
    assert "nm_per_pixel" in record_text and "radius_definition" in record_text
    assert "prediction_included" in record_text and "sem_radius_scale" in record_text
    assert list(output.glob("*.csv")), "Export the measured numerical values as well as the figure"
    assert list(output.glob("*.png")) and list(output.glob("*.pdf"))
    assert source.read_bytes() == original_bytes
    assert set(tmp_path.iterdir()) == {source, output}, "Never write beside the input image implicitly"


def test_csv_pixel_diameters_require_calibration_and_convert_once(tmp_path):
    source = tmp_path / "sizes.csv"
    source.write_text("diameter_px,unrelated\n4,100\n8,200\n", encoding="utf-8")
    service = create_sem_service()
    with pytest.raises(ValueError, match="calibration|nm/pixel"):
        service.import_csv(source, SemCsvSettings("diameter_px", quantity="diameter", unit="px"))
    measurements = service.import_csv(
        source, SemCsvSettings("diameter_px", quantity="diameter", unit="px", nm_per_pixel=0.25),
    )
    np.testing.assert_array_equal(measurements.radius_nm, [0.5, 1.0])
    comparison = service.compare(measurements)
    assert comparison.metadata["prediction_included"] is False
    assert comparison.metadata["sem_radius_scale"] == 1.0
    assert measurements.source["path"] == str(source.resolve())


def test_gui_standalone_image_needs_calibration_and_exports_without_prediction(
    qt_application, tmp_path, monkeypatch,
):
    source = tmp_path / "synthetic_sem.png"
    _synthetic_image(source)
    window = create_sem_window()
    try:
        window._open_source(str(source), "image")
        _wait(qt_application, lambda: not window._runner.is_busy())
        assert window.view_model.state.image is not None
        assert window.view_model.state.prediction is None
        assert window.nm_per_pixel_spin.value() == 0
        assert not window.run_button.isEnabled()
        window.nm_per_pixel_spin.setValue(0.5)
        assert window.run_button.isEnabled()
        window.run_button.click()
        _wait(qt_application, lambda: not window._runner.is_busy())
        state = window.view_model.state
        assert state.measurements.radius_nm.size == 2
        assert state.comparison.metadata["prediction_included"] is False
        assert window.export_button.isEnabled()

        output = tmp_path / "gui_export"
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_a, **_k: str(output))
        window.export_button.click()
        _wait(qt_application, lambda: not window._runner.is_busy())
        assert list(output.rglob("*.json"))
        assert list(output.rglob("*.png")) and list(output.rglob("*.pdf"))
    finally:
        window.dispose()
        window.close()


def _prediction_bridge():
    module = PredictionModule(
        id="synthetic", name="Synthetic model", version="1",
        preprocess=PreprocessSpec(steps=("crop",), params={"crop": {"left": 2}}),
    )
    outputs = {
        "distribution_axes": {"row": {"key": "r", "unit": "nm", "label": "R"}},
        "r_x": np.array([1.0, 2.0, 3.0]), "r": np.array([0.2, 0.5, 0.3]),
    }
    bridge = SimpleNamespace(
        _current_image=np.ones((5, 7)), _typed_prediction_module=lambda: bridge.module,
        module=module, current_parameters={"module_model_path": "synthetic_model"},
        _current_input_files=["synthetic.cbf"], _current_input_stack=1,
        prediction_results=outputs,
    )
    SemComparisonMixin._capture_sem_prediction(bridge, outputs)
    return bridge, outputs


@pytest.mark.parametrize("change", ["image", "module", "model_path", "outputs", "nested_preprocess"])
def test_prediction_comparison_cannot_relabel_an_old_result_as_current(change):
    bridge, _outputs = _prediction_bridge()
    assert SemComparisonMixin._sem_prediction_for_comparison(bridge) is not None
    if change == "image":
        bridge._current_image = bridge._current_image.copy()
    elif change == "module":
        bridge.module = PredictionModule(id="synthetic", name="Synthetic model", version="2")
    elif change == "model_path":
        bridge.current_parameters["module_model_path"] = "another_model"
    elif change == "outputs":
        bridge.prediction_results = dict(bridge.prediction_results)
    else:
        bridge.module.preprocess.params["crop"]["left"] = 3
    assert SemComparisonMixin._sem_prediction_for_comparison(bridge) is None


def test_prediction_comparison_captures_values_and_source_at_inference_time():
    bridge, outputs = _prediction_bridge()
    outputs["r"][:] = 0
    outputs["r_x"][:] = 20
    bridge._current_input_files[0] = "later.cbf"
    snapshot = SemComparisonMixin._sem_prediction_for_comparison(bridge)
    np.testing.assert_array_equal(snapshot["radius_nm"], [1, 2, 3])
    np.testing.assert_array_equal(snapshot["probability"], [0.2, 0.5, 0.3])
    assert snapshot["metadata"]["input_files"] == ["synthetic.cbf"]
    snapshot["metadata"]["preprocessing_settings"]["crop"]["left"] = 100
    assert SemComparisonMixin._sem_prediction_for_comparison(bridge)["metadata"]["preprocessing_settings"]["crop"]["left"] == 2


def test_independent_menu_entry_removes_prediction_from_a_reused_tool(qt_application, tmp_path):
    owner = QWidget()
    menus = object.__new__(ApplicationMenus)
    menus.tools = ToolWindows(owner)
    payload = {
        "radius_nm": np.array([1, 2, 3]), "probability": np.array([0.2, 0.5, 0.3]),
        "source": "synthetic prediction", "metadata": {"input_files": ["synthetic.cbf"]},
    }
    window = menus.open_sem(payload)
    try:
        assert window.view_model.state.prediction is not None
        source = tmp_path / "sem_radii.csv"
        source.write_text("radius_nm\n1\n2\n3\n", encoding="utf-8")
        window._open_source(str(source), "csv")
        _wait(qt_application, lambda: not window._runner.is_busy())
        window.csv_column_combo.setCurrentText("radius_nm")
        window.csv_quantity_combo.setCurrentIndex(1)  # Explicitly map the column as radius.
        window.run_button.click()
        _wait(qt_application, lambda: not window._runner.is_busy())
        measurements = window.view_model.state.measurements
        assert window.view_model.state.comparison.metadata["prediction_included"] is True

        # Tools > SEM is independent even when the existing tool holds a prediction.
        standalone = menus.open_sem()
        assert standalone is window
        _wait(qt_application, lambda: not window._runner.is_busy())
        assert window.view_model.state.prediction is None
        assert window.view_model.state.measurements is measurements
        assert window.view_model.state.comparison.metadata["prediction_included"] is False
        assert window.export_button.isEnabled()
    finally:
        window.dispose()
        window.close()
        owner.close()


def test_prediction_handoff_copies_arrays_and_nested_provenance(qt_application):
    window = create_sem_window()
    radius, probability = np.array([1.0, 2.0, 3.0]), np.array([0.2, 0.5, 0.3])
    metadata = {"sources": [{"path": "synthetic.cbf"}], "model": {"id": "synthetic"}}
    try:
        window.set_prediction(radius, probability, "synthetic prediction", metadata)
        attached = window.view_model.state.prediction
        radius[:] = 99
        probability[:] = 0
        metadata["sources"][0]["path"] = "another.cbf"
        metadata["model"]["id"] = "another"
        np.testing.assert_array_equal(attached.radius_nm, [1, 2, 3])
        np.testing.assert_array_equal(attached.probability, [0.2, 0.5, 0.3])
        assert attached.metadata["sources"][0]["path"] == "synthetic.cbf"
        assert attached.metadata["model"]["id"] == "synthetic"
        assert not attached.radius_nm.flags.writeable
        assert not attached.probability.flags.writeable
        window.set_prediction(None)
        assert window.view_model.state.prediction is None
        assert window.view_model.state.comparison is None
    finally:
        window.dispose()
        window.close()


def test_new_source_discards_late_old_image_load(qt_application, tmp_path, monkeypatch):
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    _synthetic_image(first)
    _synthetic_image(second)
    window = create_sem_window()
    reader = window.view_model.service.load_image
    started, release = threading.Event(), threading.Event()

    def load(path):
        if Path(path) == first:
            started.set()
            assert release.wait(5), "Test failed to release the old SEM image load"
        return reader(path)

    monkeypatch.setattr(window.view_model.service, "load_image", load)
    try:
        window._open_source(str(first), "image")
        _wait(qt_application, started.is_set)
        window._open_source(str(second), "image")
        release.set()
        _wait(qt_application, lambda: not window._runner.is_busy())
        state = window.view_model.state
        assert Path(state.source_path) == second
        assert Path(state.image.source_path) == second
        assert state.measurements is None and state.comparison is None
        assert window.sem_source_edit.text() == str(second)
    finally:
        release.set()
        window._runner.wait(5)
        window.dispose()
        window.close()


def test_all_rejected_particles_remain_reviewable_and_exportable(qt_application, tmp_path):
    source = tmp_path / "all_rejected.png"
    _synthetic_image(source)
    window = create_sem_window()
    try:
        window._open_source(str(source), "image")
        _wait(qt_application, lambda: not window._runner.is_busy())
        window.nm_per_pixel_spin.setValue(0.5)
        window.min_area_spin.setValue(100)
        window.run_button.click()
        _wait(qt_application, lambda: not window._runner.is_busy())
        state = window.view_model.state
        assert state.measurements is not None, "Keep rejected candidates so the scientist can review the filters"
        assert state.measurements.radius_nm.size == 0
        assert len(state.measurements.particles) == 2
        assert all(not particle.accepted for particle in state.measurements.particles)
        assert state.comparison is None
        assert window.particles_table.rowCount() == 2
        assert window.export_button.isEnabled()
        assert tuple(window.plotter.image_axis.images[0].get_extent()) == (0, 48, 40, 0)

        output = tmp_path / "rejected_export"
        window._export_to(output)
        _wait(qt_application, lambda: not window._runner.is_busy())
        assert (output / "sem_analysis.json").is_file()
        assert (output / "sem_particles.csv").is_file()
        assert (output / "sem_overlay.png").is_file()
        assert not (output / "sem_distribution.png").exists()
    finally:
        window.dispose()
        window.close()


def test_tool_reopened_while_a_cancelled_read_finishes_stays_usable(qt_application, tmp_path, monkeypatch):
    owner = QWidget()
    menus = object.__new__(ApplicationMenus)
    menus.tools = ToolWindows(owner)
    window = menus.open_sem()
    source = tmp_path / "slow_sem.png"
    _synthetic_image(source)
    read = window.view_model.service.load_image
    started, release = threading.Event(), threading.Event()

    def slow_read(path):
        started.set()
        assert release.wait(5), "Test failed to release the closing SEM image read"
        return read(path)

    monkeypatch.setattr(window.view_model.service, "load_image", slow_read)
    try:
        window._open_source(str(source), "image")
        _wait(qt_application, started.is_set)
        window.close()
        assert not window.isVisible()
        assert window._runner.is_busy()

        reopened = menus.open_sem()
        assert reopened is window
        assert reopened.isVisible()
        release.set()
        _wait(qt_application, lambda: not reopened._runner.is_busy())
        assert reopened.isVisible(), "A previously cancelled read must not close a newly reopened tool"
        assert reopened.view_model.state.image is None, "Closing cancelled the old source read"

        monkeypatch.setattr(window.view_model.service, "load_image", read)
        reopened._open_source(str(source), "image")
        _wait(qt_application, lambda: not reopened._runner.is_busy())
        reopened.nm_per_pixel_spin.setValue(0.5)
        reopened.run_button.click()
        _wait(qt_application, lambda: not reopened._runner.is_busy())
        assert reopened.view_model.state.measurements.radius_nm.size == 2
        assert reopened.export_button.isEnabled()
    finally:
        release.set()
        window._runner.wait(5)
        window.dispose()
        window.close()
        owner.close()


def test_language_switch_keeps_csv_source_column_names_and_import_mapping(qt_application, tmp_path):
    source = tmp_path / "source_column_names.csv"
    source.write_text("Radius,Diameter\n1,200\n2,400\n", encoding="utf-8")
    previous_language = i18n.current_language()
    i18n.apply_language("en")
    window = create_sem_window()
    try:
        window.show()
        window._open_source(str(source), "csv")
        _wait(qt_application, lambda: not window._runner.is_busy())
        window.csv_column_combo.setCurrentText("Radius")
        window.csv_quantity_combo.setCurrentIndex(1)
        for language in ("zh", "en", "zh", "en"):
            i18n.apply_language(language, [window])
            qt_application.processEvents()
            assert [window.csv_column_combo.itemText(i) for i in range(2)] == ["Radius", "Diameter"]
            assert window.csv_column_combo.currentText() == "Radius"
            assert window.run_button.isEnabled()
            window.run_button.click()
            _wait(qt_application, lambda: not window._runner.is_busy())
            measurements = window.view_model.state.measurements
            np.testing.assert_array_equal(measurements.radius_nm, [1, 2])
            assert measurements.settings["value_column"] == "Radius"
    finally:
        window.dispose()
        window.close()
        i18n.apply_language(previous_language)


def test_main_window_keeps_parent_alive_when_sem_shutdown_is_pending(qt_application, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox

    from main import MainWindow
    from src.gimap.app import AppContext
    from src.gimap.integrations.jobs import LocalProcessJobRunner
    from src.gimap.integrations.state import (
        InMemorySessionRepository,
        InMemorySettingsRepository,
        InMemoryUserPreferencesRepository,
    )

    context = AppContext(
        settings=InMemorySettingsRepository(), session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(), jobs=LocalProcessJobRunner(),
    )
    window = MainWindow(context)
    window.show()
    _wait(qt_application, lambda: window._initialization_completed, timeout=30)
    _wait(qt_application, lambda: getattr(window.runtime.prediction, "_initialized", False), timeout=30)
    ready = False
    monkeypatch.setattr(window.menus, "sem_running", lambda: True)
    monkeypatch.setattr(window.menus, "shutdown_sem", lambda: ready)
    questions, saved = [], []

    def confirm(_parent, _title, message, *_args, **_kwargs):
        questions.append(message)
        return QMessageBox.Yes

    save_session = context.save_session
    monkeypatch.setattr(QMessageBox, "question", confirm)
    monkeypatch.setattr(context, "save_session", lambda: saved.append(True) or save_session())
    try:
        job_name = i18n.tr("SEM Particle Analysis")
        assert job_name in window.components.running_jobs()

        # The user agreed to quit, but the SEM worker still owns its parent signals.
        assert window.close() is False
        qt_application.processEvents()
        assert window.isVisible() and window.isEnabled()
        assert not getattr(window, "_closed", False)
        assert not saved, "Do not save/dispose the rest of the app before SEM has stopped"
        assert len(questions) == 1 and job_name in questions[0]

        ready = True
        assert window.close() is True
        qt_application.processEvents()
        assert getattr(window, "_closed", False)
        assert not window.isVisible()
        assert len(saved) == 1
        assert len(questions) == 2
    finally:
        ready = True
        window.close()
