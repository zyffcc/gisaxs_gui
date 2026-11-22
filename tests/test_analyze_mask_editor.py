"""Exercise the mask editor through its Qt controls, detector viewport and file dialogs' formats."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from src.gimap.features.analyze.infrastructure.adapters import DetectorIoFrameSource
from src.gimap.shared.detector_io.mask_files import read_mask_image, write_mask_image
from tests.test_giwaxs_workspace import page  # noqa: F401 - shared GIWAXS fixture


def _tool(page, kind):
    page.mask_tool.setCurrentIndex(page.mask_tool.findData(kind))
    QApplication.processEvents()


def _draw(page, kind, points):
    _tool(page, kind)
    page.mask_draw.click()
    graphics = page.detector_view.graphics
    viewport = graphics.viewport()
    positions = [graphics.mapFromScene(page.detector_view.view_box.mapViewToScene(QPointF(x, y))) for x, y in points]
    if kind in ("brush", "eraser"):
        QTest.mousePress(viewport, Qt.LeftButton, pos=positions[0])
        for position in positions[1:]:
            QTest.mouseMove(viewport, position)
        QTest.mouseRelease(viewport, Qt.LeftButton, pos=positions[-1])
    else:
        for position in positions:
            QTest.mouseClick(viewport, Qt.LeftButton, pos=position)
            QApplication.processEvents()
    assert page.tasks.wait(20)
    assert page.mask_apply.isVisible(), page.status_text()
    count = page._mask_count
    page.mask_apply.click()
    assert page.tasks.wait(20)
    mask = page.view_model.state.analysis.drawn_mask
    assert np.count_nonzero(mask) == count
    return mask.copy()


def test_circle_annulus_drawn_with_mouse_and_undone_separately(page):
    page.show_step("mask")
    circle = _draw(page, "circle", [(90, 80), (100, 80)])
    assert circle[80, 90] and not circle[80, 110]
    both = _draw(page, "annulus", [(150, 90), (157, 90), (165, 90)])
    assert both[90, 160] and not both[90, 150] and not both[90, 168]
    assert np.all(both[circle])
    assert page.undo_setup() and page.tasks.wait()
    np.testing.assert_array_equal(page.view_model.state.analysis.drawn_mask, circle)
    assert page.undo_setup() and page.tasks.wait()
    assert page.view_model.state.analysis.drawn_mask is None
    assert page.redo_setup() and page.tasks.wait()
    assert page.redo_setup() and page.tasks.wait()
    np.testing.assert_array_equal(page.view_model.state.analysis.drawn_mask, both)


def test_brush_eraser_and_value_preview(page):
    page.show_step("mask")
    stroke = _draw(page, "brush", [(105, 70), (135, 70)])
    assert stroke[70, 110] and stroke[70, 130]
    erased = _draw(page, "eraser", [(120, 65), (120, 75)])
    assert erased[70, 110] and not erased[70, 120]
    _tool(page, "above")
    page.mask_threshold.setValue(10)
    page.mask_preview.click()
    assert page.tasks.wait()
    assert page.mask_apply.isVisible() and page._mask_count > np.count_nonzero(erased)
    np.testing.assert_array_equal(page.view_model.state.analysis.drawn_mask, erased)
    page.mask_threshold.setValue(15)
    assert not page.mask_apply.isVisible()
    page.mask_preview.click()
    assert page.tasks.wait()
    count = page._mask_count
    page.mask_apply.click()
    assert page.tasks.wait() and np.count_nonzero(page.view_model.state.analysis.drawn_mask) == count


@pytest.mark.parametrize("extension", [".edf", ".tif", ".tiff", ".npy", ".msk"])
def test_mask_bitmap_round_trip_and_orientation(tmp_path, extension):
    expected = np.zeros((7, 37), dtype=bool)  # asymmetric, with Fit2D's last byte only partly occupied
    expected[0, 35:] = expected[2:5, 2] = expected[6, 0] = True
    path = tmp_path / ("user_mask" + extension)
    write_mask_image(path, expected)
    np.testing.assert_array_equal(read_mask_image(path, DetectorIoFrameSource()), expected)
    if extension == ".npy":
        stored = np.load(path, allow_pickle=False)
    else:
        import fabio
        with fabio.open(str(path)) as image:
            stored = image.data
    np.testing.assert_array_equal(stored, expected.astype(np.uint8))


def test_composite_files_recipe_and_export_record(page, tmp_path):
    state = page.view_model.state
    shape = state.analysis.shape
    paths = []
    for name, y, x in (("first", 5, 7), ("second", 3, 1)):
        bitmap = np.zeros(shape, dtype=np.int16)
        bitmap[y, x] = -3
        path = tmp_path / (name + ".npy")
        np.save(path, bitmap)
        assert page.load_masks(path) and page.tasks.wait()
        paths.append(path)
    expected = state.analysis.drawn_mask.copy()
    assert expected.sum() == 2 and expected[5, 7] and expected[3, 1]
    saved = page.save_masks(tmp_path / "composite.edf")
    np.testing.assert_array_equal(read_mask_image(saved, DetectorIoFrameSource()), expected)
    recipe = saved.with_suffix(".edf.json")
    record = json.loads(recipe.read_text(encoding="utf-8"))
    assert record["analysis"]["corrections"]["mask_sources"] == list(map(str, paths))
    digests = {item["file"]: item["sha256"] for item in record["analysis"]["input_sha256"]}
    assert all(digests[path.name] == sha256(path.read_bytes()).hexdigest() for path in paths)
    page._clear_masks()
    assert page.tasks.wait() and state.analysis.drawn_mask is None
    assert page.load_masks(recipe) and page.tasks.wait()
    np.testing.assert_array_equal(state.analysis.drawn_mask, expected)
    from src.gimap.features.analyze.application import analysis_metadata
    assert set(analysis_metadata(state.analysis)["corrections"]["mask_sources"]) == set(map(str, [recipe, *paths]))
    assert page.undo_setup() and page.tasks.wait() and state.analysis.drawn_mask is None


def test_reject_non_image_numpy_masks(tmp_path: Path):
    path = tmp_path / "one_dimension.npy"
    np.save(path, np.array([1, 0]))
    with pytest.raises(ValueError, match="two-dimensional"):
        read_mask_image(path, DetectorIoFrameSource())


@pytest.mark.parametrize("suffix", [".npy", ".msk"])
def test_truncated_mask_file_is_reported(tmp_path, suffix):
    path = tmp_path / ("incomplete" + suffix)
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="Could not decode"):
        read_mask_image(path, DetectorIoFrameSource())


def test_q_rule_preview_cancel_clear_and_redo(page):
    page.show_step("mask")
    state = page.view_model.state
    _tool(page, "q_annulus")
    page.mask_q_min.setValue(0.12)
    page.mask_q_max.setValue(0.2)
    page.mask_preview.click()
    assert page.tasks.wait() and page._mask_count > 0
    page.mask_cancel.click()
    assert not page.mask_apply.isVisible() and state.corrections.mask_shapes == ()
    page.mask_preview.click()
    assert page.tasks.wait()
    page.mask_apply.click()
    assert page.tasks.wait()
    annulus = state.analysis.drawn_mask.copy()
    _tool(page, "invert")
    page.mask_preview.click()
    assert page.tasks.wait() and page._mask_count == np.count_nonzero(~annulus)
    page.mask_apply.click()
    assert page.tasks.wait()
    np.testing.assert_array_equal(state.analysis.drawn_mask, ~annulus)
    page.mask_clear_button.click()
    assert page.tasks.wait() and state.analysis.drawn_mask is None
    assert page.undo_setup() and page.tasks.wait()
    np.testing.assert_array_equal(state.analysis.drawn_mask, ~annulus)
    assert page.redo_setup() and page.tasks.wait() and state.analysis.drawn_mask is None


def test_failed_bitmap_export_preserves_previous_record(tmp_path):
    from src.gimap.shared.detector_io.mask_files import write_mask_export

    destination = tmp_path / "occupied.edf"
    destination.mkdir()
    sidecar = tmp_path / "occupied.edf.json"
    sidecar.write_text("previous record", encoding="utf-8")
    with pytest.raises(OSError):
        write_mask_export(destination, {"masks": []}, np.ones((2, 3), dtype=bool))
    assert destination.is_dir() and sidecar.read_text(encoding="utf-8") == "previous record"
