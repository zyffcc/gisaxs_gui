"""Compare reads a series from a results database (Add Series ▸ Results Database…): the curves Analyze recorded, one
curve of one series, without the frames; with several, the person chooses; what cannot be read is said."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtWidgets import QInputDialog

from src.gimap.features.analyze.application import store_rows
from src.gimap.features.analyze.application.series_map import RowLabel
from src.gimap.features.compare.bootstrap import create_compare_service
from src.gimap.shared.series_store import SeriesStore, store_path
from tests.test_compare import Q, _close, _run, _wait


def _record(folder: Path, name: str, image: np.ndarray, curve: str = "radial") -> Path:
    rows = [(Q, image[row], "q (Å⁻¹)", RowLabel(f"{name}_{row:03d}", source=f"C:/data/{name}_{row:03d}.tif"),
             (Path(f"C:/data/{name}_{row:03d}.tif"), 0)) for row in range(image.shape[0])]
    path = store_path(folder)
    with SeriesStore(path) as store:
        store_rows(store, name, rows, curve)
    return path


def test_the_service_reads_one_curve_of_one_series_as_it_was_recorded(tmp_path):
    image = _run(30, seed=4)
    path = _record(tmp_path, "film", image)
    _record(tmp_path, "film", image[:, ::-1], curve="azimuthal")
    service = create_compare_service()
    assert service.store_contents(path) == [("film", "radial", 30), ("film", "azimuthal", 30)]
    data = service.series_from_store(path, "film", "radial")
    assert data.name == "film · radial" and data.rows == 30 and data.source == str(path)
    np.testing.assert_allclose(data.image, image)
    assert data.labels[0] == "film_000" and data.paths[-1].endswith("film_029.tif")
    with pytest.raises(ValueError, match="no curve"):
        service.series_from_store(path, "film", "vertical")


def test_the_page_adds_a_series_from_a_database_and_asks_when_it_holds_several(tmp_path, monkeypatch):
    from src.gimap.features.compare.presentation.page import ComparePage
    from tests.test_assistant_gui import _app

    path = _record(tmp_path, "film", _run(40, seed=1))
    _record(tmp_path, "other", _run(40, seed=2, extra=1.0))
    _app()
    page = ComparePage(create_compare_service())
    try:
        asked = []
        monkeypatch.setattr(QInputDialog, "getItem", staticmethod(
            lambda _parent, _title, text, items, current, _editable: asked.append((text, items)) or (items[0], True)))
        first = page.add_store(path)
        assert asked and len(asked[0][1]) == 2 and first.name == "film · radial"
        second = page.add_store(path, "other", "radial")
        assert second.name == "other · radial" and len(asked) == 1  # named: not asked
        _wait(page, lambda: page.comparison is not None and len(page.comparison.results) == 2)
        junk = tmp_path / "junk.sqlite"
        junk.write_text("no", encoding="utf-8")
        assert page.add_store(junk) is None and "not a GIMaP results database" in page.status_label.text()
    finally:
        _close(page)
