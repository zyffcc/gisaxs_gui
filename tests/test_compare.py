"""Compare: several series side by side — groups, differences, odd frames, files and the page."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.gimap.features.compare.application import CompareSettings, SeriesData
from src.gimap.features.compare.bootstrap import create_compare_service
from src.gimap.features.compare.domain import common_grid, compare, on_grid

Q = np.linspace(0.2, 3.0, 300)


def _peak(centre: float, width: float = 0.05) -> np.ndarray:
    return np.exp(-0.5 * ((Q - centre) / width) ** 2)


def _run(rows: int, *, extra: float = 0.0, seed: int = 0, every: int = 1, odd: int = -1, q=Q) -> np.ndarray:
    """A peak at 1.0 that grows and saturates (in log I); ``extra``: a second peak at 2.0 at the end."""
    t = np.arange(0, rows * every, every, dtype=float) / (rows * every)
    growth = 1.0 - np.exp(-t / 0.25)
    log = np.log10(40.0 * Q ** -2)[None, :] + growth[:, None] * _peak(1.0) + extra * t[:, None] * _peak(2.0)
    image = 10 ** log * (1.0 + 0.01 * np.random.default_rng(seed).normal(size=log.shape))
    if odd >= 0:
        image[odd] *= np.exp(-Q)
    if q is not Q:
        image = np.vstack([np.interp(q, Q, row) for row in image])
    return image


def _series(name, image, q=Q, **kw) -> SeriesData:
    return SeriesData(name, q, image, tuple(f"{name}_{i:03d}" for i in range(len(image))), **kw)


def test_alike_series_group_together_and_the_different_one_stands_apart() -> None:
    a = _series("A", _run(120, seed=1, odd=30))
    b = _series("B", _run(60, seed=2, every=2))  # the same run, every second frame
    c = _series("C", _run(120, seed=3, extra=1.0))  # a second peak grows: ends elsewhere
    d = _series("D", _run(120, seed=4))
    comparison = compare([a, b, c, d])
    assert comparison.names == ["A", "B", "C", "D"]
    assert comparison.groups == (1, 1, 2, 1) and comparison.group_text() == "A, B, D; C"
    assert comparison.distance[0, 1] < 5 and comparison.distance[0, 2] > 4 * comparison.distance[0, 1]
    assert [frame.row for frame in comparison.results[0].odd] == [30]
    assert not comparison.results[1].odd
    half_a, half_b = comparison.results[0].half_row, comparison.results[1].half_row
    assert abs(half_a / 120 - half_b / 60) < 0.05  # the same pace on the share of the series
    assert comparison.results[0].stages is not None and comparison.results[0].scores.shape[0] == 120


def test_two_alike_series_are_not_split_and_a_different_axis_is_refused() -> None:
    comparison = compare([_series("A", _run(80, seed=1)), _series("B", _run(80, seed=2))])
    assert comparison.groups == (1, 1) and comparison.group_text() == ""
    with pytest.raises(ValueError, match="different x axes"):
        compare([_series("A", _run(20)), _series("B", _run(20), x_label="χ (°)")])
    with pytest.raises(ValueError, match="Add a series"):
        compare([])


def test_the_common_grid_is_where_every_series_has_data() -> None:
    narrow_q = np.linspace(0.5, 2.5, 150)
    a = _series("A", _run(30, seed=1))
    b = _series("B", _run(30, seed=2, q=narrow_q), q=narrow_q)
    grid = common_grid([a, b])
    assert grid.min() >= 0.5 and grid.max() <= 2.5 and np.all(np.isin(grid, Q))
    moved = on_grid(b, Q)
    assert np.isnan(moved[:, 0]).all() and np.isfinite(moved[:, 100]).all()  # never extrapolated
    comparison = compare([a, b])
    assert comparison.compared.min() >= 0.5 and comparison.compared.max() <= 2.5


def _write_curves(folder: Path, image, *, unit: str = "1/A", q=Q) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, row in enumerate(image):
        path = folder / f"run_{index + 1}_fit_input.dat"  # run_1 … run_10: counting order, not text order
        x = q * (10.0 if unit == "nm^-1" else 1.0)
        lines = ["# GIMaP Analyze fit input", f"# columns: q ({unit})  I  sigma"]
        lines += [f"{a:.8g} {b:.8g} {np.sqrt(b):.4g}" for a, b in zip(x, row)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def test_curve_folders_are_read_in_counting_order_and_the_tables_written(tmp_path: Path) -> None:
    service = create_compare_service()
    _write_curves(tmp_path / "sample_A", _run(12, seed=1))
    _write_curves(tmp_path / "sample_B", _run(12, seed=2, extra=1.0), unit="nm^-1")
    a = service.series_from_folder(tmp_path / "sample_A")
    b = service.series_from_folder(tmp_path / "sample_B")
    assert a.name == "sample_A" and a.rows == 12 and a.labels[:3] == ("run_1_fit_input.dat", "run_2_fit_input.dat",
                                                                       "run_3_fit_input.dat")
    assert a.x_label == "q (Å⁻¹)" and b.x.max() == pytest.approx(3.0)  # nm⁻¹ converted
    with pytest.raises(ValueError, match="Fewer than two"):
        service.series_from_folder(tmp_path)
    comparison = service.compare([a, b], CompareSettings(q_range=(0.5, 2.5)))
    assert comparison.compared.min() >= 0.5
    table = service.export_series_table([a, b], comparison, tmp_path / "out" / "series.csv")
    rows = [line for line in table.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    assert rows[0].startswith("series,frames,odd_frames,stages,half_of_change_frame") and len(rows) == 3
    record = json.loads(table.with_suffix(".json").read_text(encoding="utf-8"))
    assert record["series"][1]["name"] == "sample_B" and "Ward" in record["method"]
    frames = service.export_frames_table([a, b], comparison, tmp_path / "out" / "frames.csv")
    assert len([line for line in frames.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]) == 25


def _wait(page, condition, timeout: float = 60.0) -> None:
    from PyQt5.QtWidgets import QApplication

    end = time.monotonic() + timeout
    while not condition() and time.monotonic() < end:
        page.tasks.wait(0.1)
        QApplication.processEvents()
    assert condition()


def test_the_page_compares_on_every_change_and_reopens_from_a_project(tmp_path: Path) -> None:
    from src.gimap.features.compare.presentation.page import ComparePage
    from tests.test_assistant_gui import _app

    _app()
    page = ComparePage(create_compare_service())
    page.resize(1400, 900)
    try:
        assert page.step_rail.state("series") == "pending" and page.save_button.isHidden()
        maps = [SimpleNamespace(x=Q, image=_run(60, seed=seed, extra=extra), labels=tuple(f"f{i}" for i in range(60)),
                                x_label="q (Å⁻¹)") for seed, extra in ((1, 0.0), (2, 0.0), (3, 1.0))]
        page.set_analyze_source(lambda: (maps[0], "run"))
        assert page.add_from_analyze().name == "run"
        page.add_map(maps[1], "run")  # the same name: made unique
        page.add_map(maps[2], "hot")
        _wait(page, lambda: page.comparison is not None and len(page.comparison.results) == 3)
        assert [item.name for item in page.series] == ["run", "run (2)", "hot"]
        assert page.results_table.rowCount() == 3 and page.distance_table.columnCount() == 4
        assert page.comparison.groups == (1, 1, 2) and "hot differs most" in page.summary_label.text()
        assert not page.save_button.isHidden() and page.step_rail.current() == "results"
        assert page.change_plot.curve_count() == 3 and page.end_plot.curve_count() == 3
        page.x_axis_combo.setCurrentIndex(1)
        assert page.change_plot.figure_state()["curves"][0][1][-1] == pytest.approx(1.0)
        # Rename, a narrower q range, and Whole Range again.
        page.series_table.item(2, 0).setText("annealed")
        page.q_low_spin.setValue(0.6)
        page.q_high_spin.setValue(2.4)
        _wait(page, lambda: page.comparison.names[2] == "annealed" and page.comparison.compared.min() >= 0.6)
        page.whole_range()
        _wait(page, lambda: page.comparison.compared.min() < 0.3)
        # Save and reopen.
        assert page.save_series_table(tmp_path / "series.csv").exists()
        state = json.loads(json.dumps(page.project_state(tmp_path / "sample.gimap")))
        assert (tmp_path / "sample.compare.npz").exists() and state["series"][2]["name"] == "annealed"
        page.series_table.selectRow(0)
        page.remove_selected()
        _wait(page, lambda: len(page.comparison.results) == 2)
        notes = page.apply_project_state(state, tmp_path / "sample.gimap")
        assert not notes and [item.name for item in page.series] == ["run", "run (2)", "annealed"]
        _wait(page, lambda: page.comparison is not None and len(page.comparison.results) == 3)
        page.clear_all()
        assert page.comparison is None and page.results_table.rowCount() == 0 and page.save_button.isHidden()
    finally:
        page.tasks.wait(10)
        page.dispose()
