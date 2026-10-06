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
    assert comparison.groups == (1, 1, 2, 1) and comparison.group_text() == "A, B, D alike; C different"
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
        assert "Group 1: run, run (2)" in page.summary_label.text() and page.step_rail.detail("results") == "2 groups"
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
        summary = page.summary_label.text()
        assert "differs most" not in summary and "run (2) and annealed: end states" in summary
        assert page.results_table.isColumnHidden(1) and not page.distance_table.isHidden()
        notes = page.apply_project_state(state, tmp_path / "sample.gimap")
        assert not notes and [item.name for item in page.series] == ["run", "run (2)", "annealed"]
        _wait(page, lambda: page.comparison is not None and len(page.comparison.results) == 3)
        page.clear_all()
        assert page.comparison is None and page.results_table.rowCount() == 0 and page.save_button.isHidden()
    finally:
        page.tasks.wait(10)
        page.dispose()


# -- names, words and notices (the audit of the Compare page) ------------------------------------------


def test_display_names_drop_the_parts_every_name_shares() -> None:
    from src.gimap.features.compare.presentation.render import display_names

    long = ["lyx_cu_peo1_20pl_2p5fr_116_00002", "lyx_cu_peo2_20pl_2p5fr_120_00002"]
    assert display_names(long) == ["peo1_116", "peo2_120"]
    assert display_names(long + ["lyx_cu_peo1_20pl_2p5fr_116_00002 (2)"]) == ["peo1_116", "peo2_120", "peo1_116 (2)"]
    assert display_names(["a_x", "a_x_b", "a_b"]) == ["x", "x_b", "b"]
    assert display_names(["s_1_t", "s_1_u", "s_1_t (2)"]) == ["t", "u", "t (2)"]
    assert display_names(["a_b_c", "a_c_b"]) == ["a_b_c", "a_c_b"]  # both would be empty: the full names
    assert display_names(["x_a_b", "y_b_a", "x_b_a"]) == ["x_a_b", "y_b_a", "x_b_a"]  # “x” twice: the full names
    assert display_names(["run", "run (2)", "hot"]) == ["run", "run (2)", "hot"]
    assert display_names(["one_long_name"]) == ["one_long_name"]  # nothing to tell apart
    assert display_names([]) == []
    assert display_names(["x_a\nb", "x_c"]) == ["a\nb", "c"]  # a line break is part of the name, not a crash
    assert display_names(["x_a\nb (2)", "x_c"]) == ["a\nb (2)", "c"]


def _fake(names, distance, start, groups):
    from src.gimap.features.compare.domain import Comparison

    results = tuple(SimpleNamespace(name=name, odd=()) for name in names)
    return Comparison(q=Q, compared=Q, x_label="q (Å⁻¹)", explained=np.array([0.9, 0.1]), results=results,
                      distance=np.asarray(distance, float), start_distance=np.asarray(start, float), groups=tuple(groups),
                      end_frames=10, shape_only=True)


def test_the_summary_says_what_the_distances_support() -> None:
    from src.gimap.features.compare.presentation.page import ComparePage
    from tests.test_assistant_gui import _app

    _app()
    page = ComparePage(create_compare_service())
    try:
        page.comparison = _fake(["A", "B"], [[0, 88.2], [88.2, 0]], [[0, 58], [58, 0]], (1, 1))
        summary = page._summary()
        assert summary.splitlines()[0] == "A and B: end states 88 % apart (58 % at the start)."
        assert "differs most" not in summary and "–" not in summary
        equal = [[0, 5.1, 4.9], [5.1, 0, 5.0], [4.9, 5.0, 0]]
        page.comparison = _fake(["A", "B", "C"], equal, equal, (1, 1, 1))
        summary = page._summary()
        assert summary.splitlines()[0] == "No clear groups: the end states are all alike."
        assert "differs most" not in summary and "–" not in summary and "every two series are 5 % apart" in summary
        page.comparison = _fake(["A", "B", "C"], [[0, 2, 40], [2, 0, 40.3], [40, 40.3, 0]], equal, (1, 1, 2))
        lines = page._summary().splitlines()
        assert lines[:3] == ["Group 1: A, B", "Group 2: C", "C differs most from the others (end states 40 % apart)."]
        assert page._results_detail(page.comparison) == "2 groups"
        assert page._results_detail(_fake(["A", "B", "C"], equal, equal, (1, 1, 1))) == "All alike"
        assert page.comparison.group_text() == "A, B alike; C different"
    finally:
        page.dispose()


def test_stage_texts_name_the_axis_and_keep_q_by_default() -> None:
    from src.gimap.app.presentation.stage_text import change_text, odd_reason
    from src.gimap.shared.series_stages.comparable import OddFrame
    from src.gimap.shared.series_stages.stages import StageChange

    change = StageChange(stage=2, rises=((0.512, 12.0),), falls=((1.2, -8.0),))
    assert change_text(change) == change.text() == "Stage 1 → 2: grows most at q 0.512 (+12 %); falls most at q 1.2 (-8 %)"
    assert change_text(change, "χ") == "Stage 1 → 2: grows most at χ 0.512 (+12 %); falls most at χ 1.2 (-8 %)"
    for frame in (OddFrame(row=3, z=7.0, q=0.75, narrow=False), OddFrame(row=4, z=9.0, q=1.5, narrow=True)):
        assert odd_reason(frame) == frame.reason
        assert " χ " in odd_reason(frame, "χ") and " q " not in odd_reason(frame, "χ")


def _map(seed: int, *, extra: float = 0.0, rows: int = 60, x=Q, x_label: str = "q (Å⁻¹)"):
    return SimpleNamespace(x=x, image=_run(rows, seed=seed, extra=extra), labels=tuple(f"f{i}" for i in range(rows)),
                           x_label=x_label)


def _process() -> None:
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()


def _shown_page(width: int = 1400, height: int = 900):
    from PyQt5.QtWidgets import QApplication

    from src.gimap.features.compare.presentation.page import ComparePage
    from tests.test_assistant_gui import _app

    _app()
    page = ComparePage(create_compare_service())
    page.resize(width, height)
    page.show()
    QApplication.setActiveWindow(page)
    _process()
    return page


def _idle(page) -> bool:
    return not page._running and not page._timer.isActive()


def _close(page) -> None:
    page.tasks.wait(10)
    page.dispose()
    page.close()


def test_one_toast_per_action_and_honest_saves(tmp_path: Path, monkeypatch) -> None:
    from src.gimap.features.compare.presentation import page as page_module

    toasts = []
    monkeypatch.setattr(page_module, "show_toast", lambda parent, text, **kw: toasts.append((text, kw)))
    page = _shown_page()
    try:
        for index, (seed, extra) in enumerate(((1, 0.0), (2, 0.0), (3, 1.0))):
            page.add_map(_map(seed, extra=extra), f"run {index + 1}")
            _wait(page, lambda: _idle(page) and page.comparison is not None and len(page.comparison.results) == index + 1)
        assert len(toasts) == 3 and toasts[-1][0] == "3 series compared: 2 groups."
        assert page.status_label.text() == "3 series compared: 2 groups."
        for _step in range(10):  # settings: the status line only
            page.q_low_spin.stepUp()
            _process()
        _wait(page, lambda: _idle(page) and page.comparison.compared.min() >= 0.25)
        assert len(toasts) == 3
        # The same map again: refused, one warning.
        page.add_map(_map(1), "run 1 again")
        assert len(page.series) == 3 and toasts[-1] == ("run 1 is already in Compare.", {"level": "warning"})
        # Saving: a folder that does not exist is an error; a written file offers its folder.
        assert page._save_plot(page.change_plot, "change", path=str(tmp_path / "missing" / "change.png")) is None
        assert toasts[-1][1]["level"] == "error" and page.status_label.text().startswith("Could not save:")
        assert not any(text.startswith("Saved") for text, _kw in toasts)
        written = page._save_plot(page.change_plot, "change", path=str(tmp_path / "change.png"))
        assert written is not None and written.exists() and toasts[-1][0] == "Saved change.png."
        assert toasts[-1][1]["level"] == "ok" and toasts[-1][1]["action"][0] == "Open Folder"
        assert page._save_plot(page.end_plot, "end", path=str(tmp_path / "missing" / "end.svg")) is None
        assert toasts[-1][1]["level"] == "error"
        table = page.save_series_table(tmp_path / "series.csv")
        assert table is not None and toasts[-1][1]["action"][0] == "Open Folder"
        record = json.loads(table.with_suffix(".json").read_text(encoding="utf-8"))
        assert record["groups"] == "run 1, run 2 alike; run 3 different"  # the full names in the record
        # A failed comparison: one error toast.
        count = len(toasts)
        page.add_map(_map(4, x=np.linspace(-80, 80, Q.size), x_label="χ (°)"), "chi")
        _wait(page, lambda: _idle(page) and page.comparison is None)
        assert len(toasts) == count + 1 and toasts[-1][1]["level"] == "error"
        assert toasts[-1][0].startswith("Could not compare:")
    finally:
        _close(page)


def test_long_names_are_shortened_where_they_are_shown_and_kept_in_the_files(tmp_path: Path) -> None:
    page = _shown_page(1600, 1000)
    names = ["lyx_cu_peo1_20pl_2p5fr_116_00002", "lyx_cu_peo2_20pl_2p5fr_117_00002", "lyx_cu_peo2_20pl_2p5fr_120_00002",
             "lyx_cu_peo2_20pl_2p5fr_123_00002", "lyx_cu_peo1_20pl_2p5fr_116_00002"]
    try:
        for index, name in enumerate(names):
            page.add_map(_map(index + 1, extra=1.0 if index == 2 else 0.0), name)
        _wait(page, lambda: _idle(page) and page.comparison is not None and len(page.comparison.results) == 5)
        legend = [curve[0] for curve in page.change_plot.figure_state()["curves"]]
        assert legend == ["peo1_116", "peo2_117", "peo2_120", "peo2_123", "peo1_116 (2)"]
        assert all(len(name) <= 12 for name in legend)
        assert [curve[0] for curve in page.end_plot.figure_state()["curves"]] == legend
        results = page.results_table
        assert results.item(0, 0).text() == "peo1_116" and results.item(0, 0).toolTip() == names[0]
        assert page.series_table.item(0, 0).text() == names[0]  # the editable table keeps the full name
        assert page.distance_table.horizontalHeaderItem(1).text() == "peo1_116"
        assert page.distance_table.item(0, 1).text() == "—"
        assert "peo2_120" in page.summary_label.text() and "lyx_cu" not in page.summary_label.text()
        assert "lyx_cu" not in page.step_rail.detail("results")
        text = page.save_series_table(tmp_path / "series.csv").read_text(encoding="utf-8")
        assert names[0] in text and "lyx_cu_peo1_20pl_2p5fr_116_00002 (2)" in text
        # Group: the second column, in view without scrolling.
        assert results.horizontalHeaderItem(1).text() == "Group" and not results.isColumnHidden(1)
        assert results.columnViewportPosition(1) + results.columnWidth(1) <= results.viewport().width()
        assert results.columnWidth(0) <= 160
    finally:
        _close(page)


def test_the_empty_state_the_layout_and_the_wording_follow_the_data() -> None:
    page = _shown_page(1400, 900)
    try:
        assert page.plot_stack.currentWidget() is page.empty_page
        assert not page.x_axis_combo.isEnabled() and not page.component_combo.isEnabled()
        assert page.method_label.text() and page.results_table.isHidden()
        page.empty_state.actionRequested.emit()  # its Add Series opens the Add menu
        _process()
        assert page.add_button.menu().isVisible()
        page.add_button.menu().hide()
        page.add_map(_map(1), "one")
        _wait(page, lambda: _idle(page) and page.comparison is not None)
        assert page.plot_stack.currentWidget() is page.plots_page and page.x_axis_combo.isEnabled()
        assert page.distance_row.isHidden() and page.distance_table.isHidden()  # one series: nothing to compare with
        assert page.results_table.isColumnHidden(1)
        page.add_map(_map(2, extra=1.0), "two")
        _wait(page, lambda: _idle(page) and len(page.comparison.results) == 2)
        assert not page.distance_row.isHidden() and page.distance_table.rowCount() == 2
        assert page.distance_label.text() == "How different (percent of intensity, shape):"
        assert page.results_table.height() < 140  # as tall as its rows, not a fixed block
        page.show_step("compare")
        for width in (1400, 1100):
            page.resize(width, 900)
            _process()
            scroll = page.step_pages["compare"]
            assert scroll.widget().minimumSizeHint().width() <= scroll.viewport().width(), width
        page.show_step("results")
        page.shape_check.setChecked(False)
        _wait(page, lambda: _idle(page) and not page.comparison.shape_only)
        assert page.distance_label.text() == "How different (percent of intensity, shape and level):"
        assert page.range_label.text() == "q range" and page.step_rail.detail("compare").startswith("q ")
        page.clear_all()
        assert page.plot_stack.currentWidget() is page.empty_page and page.distance_table.columnCount() == 0
        assert not page.component_combo.isEnabled() and page._q_range is None
        # An I(χ) map is described in χ, never in Å⁻¹.
        page.add_map(_map(3, x=np.linspace(-80, 80, Q.size), x_label="χ (°)"), "chi")
        _wait(page, lambda: _idle(page) and page.comparison is not None)
        assert page.range_label.text() == "χ range" and page.step_rail.detail("compare").startswith("χ ")
        tip = page.q_low_spin.toolTip()
        assert "χ" in tip and "Å⁻¹" not in tip and "Å⁻¹" not in page.range_label.text()
    finally:
        _close(page)


def test_the_delete_key_removes_the_selected_series() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    page = _shown_page()
    try:
        page.remove_selected()
        assert page.status_label.text() == "Select a series to remove"
        page.add_map(_map(1), "a")
        page.add_map(_map(2), "b")
        page.series_table.setFocus()
        page.series_table.selectRow(0)
        QTest.keyClick(page.series_table, Qt.Key_Delete)
        assert [item.name for item in page.series] == ["b"]
        # While a name is edited, Delete edits the name.
        page.series_table.editItem(page.series_table.item(0, 0))
        _process()
        editor = page.series_table.focusWidget()
        assert editor is not None and editor is not page.series_table
        QTest.keyClick(editor, Qt.Key_Delete)
        assert [item.name for item in page.series] == ["b"]
    finally:
        _close(page)


def test_pasted_names_fonts_ranges_and_stale_comparisons_keep_the_page_right() -> None:
    from src.gimap.app.presentation.theme import apply_theme
    from src.gimap.app.presentation.theme.engine import theme_manager

    mode, font_pt = theme_manager().mode, theme_manager().font_pt
    page = _shown_page()
    real = page.service.compare
    try:
        page.add_map(_map(1), "x_a")
        page.add_map(_map(2, extra=1.0), "x_c")
        _wait(page, lambda: _idle(page) and page.comparison is not None and len(page.comparison.results) == 2)
        # A name pasted with a line break is one line; an emptied name gets the old one back.
        page.series_table.item(0, 0).setText("x_a\nb")
        assert page.series[0].name == "x_a b" and page.series_table.item(0, 0).text() == "x_a b"
        _wait(page, lambda: _idle(page) and page.comparison.names[0] == "x_a b")
        assert page.results_table.item(0, 0).text() == "a b"
        page.series_table.item(1, 0).setText(" \n ")
        assert page.series[1].name == "x_c" and page.series_table.item(1, 0).text() == "x_c"
        # A bigger font (Settings ▸ font size): the result tables grow with their header, no row cut off.
        page.show_step("results")
        table = page.distance_table
        header = table.horizontalHeader()
        before, name_width = header.sizeHint().height(), page.results_table.columnWidth(0)
        apply_theme(mode, font_pt + 3.0)
        _process()
        _process()
        rows = sum(table.rowHeight(row) for row in range(table.rowCount()))
        assert header.sizeHint().height() > before and page.results_table.columnWidth(0) > name_width
        assert table.height() >= header.sizeHint().height() + rows + 2 * table.frameWidth()
        apply_theme(mode, font_pt)
        _process()
        # A narrower range, then the last series removed: the next series start from their whole range.
        page.q_low_spin.setValue(0.4)
        _wait(page, lambda: _idle(page) and page.comparison.compared.min() >= 0.4)
        page.series_table.selectAll()
        page.remove_selected()
        _wait(page, lambda: _idle(page) and page.comparison is None)
        assert page._q_range is None
        # An I(χ) series over ±180°: the spins reach its ends, and one step does not cut it to ±100.
        page.add_map(_map(3, x=np.linspace(-180, 180, Q.size), x_label="χ (°)"), "chi")
        _wait(page, lambda: _idle(page) and page.comparison is not None)
        assert (page.q_low_spin.value(), page.q_high_spin.value()) == (-180.0, 180.0)
        page.q_high_spin.stepDown()
        assert page._q_range[0] == -180.0 and 179.0 <= page._q_range[1] < 180.0
        _wait(page, lambda: _idle(page) and page.comparison.compared.max() < 180.0)
        assert page.comparison.compared.min() == -180.0
        # Remove All while a comparison runs: the steps do not stay at “Comparing …”.
        page.service.compare = lambda series, settings: (time.sleep(0.5), real(series, settings))[1]
        page.whole_range()
        _wait(page, lambda: page._running)
        page.clear_all()
        _wait(page, lambda: _idle(page))
        assert page.comparison is None and page.step_rail.state("results") == "pending"
        assert page.step_rail.detail("results") == "After a series is added"
        # A comparison that fails after Remove All is about series no longer there: no error.
        def broken(series, settings):
            time.sleep(0.5)
            raise ValueError("gone")

        page.service.compare = broken
        page.add_map(_map(4), "late")
        _wait(page, lambda: page._running)
        page.clear_all()
        _wait(page, lambda: _idle(page))
        assert page.status_label.text() == "" and page.step_rail.state("results") == "pending"
    finally:
        page.service.compare = real
        if (theme_manager().mode, theme_manager().font_pt) != (mode, font_pt):
            apply_theme(mode, font_pt)
        _close(page)


def test_stage_marks_stand_out_on_any_colour() -> None:
    import pyqtgraph as pg

    from src.gimap.app.presentation.components import DetectorView, RowGroups
    from tests.test_assistant_gui import _app

    _app()
    view = DetectorView()
    view.set_image(np.random.default_rng(0).random((30, 50)), rect=(0.0, 0.0, 1.0, 30.0), y_down=True)
    groups = RowGroups(view)
    groups.show([0, 10, 30], (0.0, 1.0), odd_rows=[5])
    labels = [item for item in groups._items if isinstance(item, pg.TextItem)]
    lines = [item for item in groups._items if isinstance(item, pg.InfiniteLine)]
    assert [label.toPlainText() for label in labels] == ["1", "2"]
    assert all(label.fill.color().alpha() == 150 and label.textItem.font().bold() for label in labels)
    assert sorted(line.pen.widthF() for line in lines) == [1.2, 3.0]  # a dark line under the white dashes
    assert lines[0].zValue() < lines[1].zValue()
    groups.clear()
    assert not groups.shown
    view.deleteLater()
