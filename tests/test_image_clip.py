"""A large picture is given to the image item a part at a time (``components/image_clip.py``): what is drawn is,
pixel for pixel, the same part of what the whole picture gives, at every zoom; the limits, the histogram of the colour
bar and Fit view are those of the whole picture; a small picture is given whole, as ever. Synthetic pictures."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from PyQt5.QtCore import QRectF
from PyQt5.QtWidgets import QApplication

from src.gimap.app.presentation.components import image_clip
from src.gimap.app.presentation.components.detector_view import DetectorView

ROWS, COLUMNS = 3000, 400
RECT = (-0.12, 0.0, 0.24, float(ROWS))  # (a Series map: q across, rows up)


def _picture(rows: int = ROWS, columns: int = COLUMNS) -> np.ndarray:
    rng = np.random.default_rng(4)
    picture = rng.random((rows, columns)) * 100 + np.linspace(0, 50, columns)
    picture[rng.integers(0, rows, 200), rng.integers(0, COLUMNS, 200)] = np.nan  # (pixels without a value)
    return picture


def _rendered(item) -> np.ndarray:
    item._renderRequired = True
    item.qimage = None
    item.render()
    return pg.functions.ndarray_from_qimage(item.qimage).copy()


def _view(*, frame: bool = False):
    app = QApplication.instance() or QApplication([])
    widget = DetectorView()
    widget.resize(900, 700)
    widget.show()
    app.processEvents()
    if frame:  # (a detector frame: pixels, row 0 at the top, square pixels)
        widget.set_image(_picture(1000, 800), y_down=True, title="frame")
    else:  # (a Series map: q across, rows up, as the map view shows it)
        widget.set_aspect_locked(False)
        widget.set_image(_picture(), rect=RECT, y_down=False, title="map")
    app.processEvents()
    return widget


@pytest.fixture
def view(monkeypatch):
    monkeypatch.setattr(image_clip, "CLIP_CELLS", 200_000)  # (a small picture stands for a large one)
    widget = _view()
    yield widget
    widget.dispose()
    widget.deleteLater()


@pytest.mark.parametrize("frame", [False, True])
def test_the_part_drawn_is_the_whole_picture_drawn_pixel_for_pixel(monkeypatch, frame) -> None:
    monkeypatch.setattr(image_clip, "CLIP_CELLS", 200_000)
    app = QApplication.instance() or QApplication([])
    view = _view(frame=frame)
    whole = view._clip.whole
    rows, columns = whole.shape
    rect = QRectF(0.0, 0.0, columns, rows) if frame else QRectF(*RECT)
    # The whole picture in the viewer's own kind of item: pyqtgraph 0.13 draws the NaN of a plain item where they
    # were at its first down-sampling factor (the viewer's item finds them again at each render).
    reference = type(view.image_item)(axisOrder="row-major")
    reference.setAutoDownsample(True)
    view.plot.addItem(reference)
    reference.hide()
    reference.setImage(whole, autoLevels=False, levels=view.image_item.getLevels())
    reference.setLookupTable(view.image_item.lut)
    reference.setRect(rect)
    required = 0
    for shown in (rows, rows // 2, 300, 100, 25):
        for top in (rows - shown, (rows - shown) // 2, 0):
            for part in (1.0, 0.3):
                left, width = rect.x(), rect.width() * part
                view.view_box.setRange(xRange=(left, left + width), yRange=(top, top + shown), padding=0)
                app.processEvents()
                view._clip.show(whole, rect, view.image_item.getLevels())  # (as a new picture kept in view)
                factors = reference._computeDownsampleFactors()
                if view._clip.part is None:
                    assert view.image_item.image is whole
                    continue
                low_r, _high_r, low_c, _high_c, part_factors = view._clip.part
                assert part_factors == factors
                drawn, full = _rendered(view.image_item), _rendered(reference)
                fx, fy = factors
                expected = full[low_r // fy: low_r // fy + drawn.shape[0], low_c // fx: low_c // fx + drawn.shape[1]]
                np.testing.assert_array_equal(drawn, expected)
                # The view and as much again on each side (a block more at each end) leave part of the picture
                # out: the item holds a part. (How large the view is depends on the window's geometry.)
                (x0, x1), (y0, y1) = view.view_box.viewRange()
                seen_rows, seen_columns = (y1 - y0) / rect.height() * rows, (x1 - x0) / rect.width() * columns
                if 3 * seen_rows + 2 * fy < rows or 3 * seen_columns + 2 * fx < columns:
                    required += 1
                    assert view.image_item.image.size < whole.size
    view.dispose()
    view.deleteLater()
    assert required >= 6  # (zoomed in: the views of a small part of the picture)


def test_limits_histogram_and_fit_view_are_of_the_whole_picture(view) -> None:
    app = QApplication.instance()
    whole = view._clip.whole
    view.view_box.setRange(xRange=(-0.05, 0.0), yRange=(100, 160), padding=0)
    app.processEvents()
    view._clip.follow()
    assert view.image_item.image.size < whole.size and view.image_item.full_image is whole
    reference = pg.ImageItem(axisOrder="row-major")
    reference.setImage(whole, autoLevels=False, levels=view.image_item.getLevels())
    for got, want in zip(view.image_item.getHistogram(bins=64, targetImageSize=400),
                         reference.getHistogram(bins=64, targetImageSize=400)):
        np.testing.assert_array_equal(got, want)
    view.fit_view()
    (x0, x1), (y0, y1) = view.view_box.viewRange()
    assert x0 <= RECT[0] and x1 >= RECT[0] + RECT[2] and y0 <= 0 and y1 >= ROWS
    view._clip.follow()
    assert view._clip.part[:4] == (0, ROWS, 0, COLUMNS)


def test_a_new_picture_kept_in_view_is_given_as_the_part_in_view(view) -> None:
    app = QApplication.instance()
    view.view_box.setRange(xRange=(-0.12, 0.12), yRange=(ROWS - 100, ROWS), padding=0)
    app.processEvents()
    view.set_image(_picture(ROWS + 500), rect=(-0.12, 0.0, 0.24, float(ROWS + 500)), y_down=False, keep_view=True)
    assert view.image_item.image.shape[0] < 1000 and view._clip.whole.shape[0] == ROWS + 500
    assert view.view_box.viewRange()[1][1] == pytest.approx(ROWS)


def test_a_small_picture_is_given_whole(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    widget = DetectorView()
    widget.resize(900, 700)
    widget.show()
    try:
        picture = _picture(400)
        widget.set_image(picture, y_down=True)
        widget.view_box.setRange(xRange=(10, 20), yRange=(10, 20), padding=0)
        app.processEvents()
        widget._clip.follow()
        assert widget._clip.part is None and widget.image_item.full_image is None
        assert widget.image_item.image.shape == picture.shape
    finally:
        widget.dispose()
        widget.deleteLater()
