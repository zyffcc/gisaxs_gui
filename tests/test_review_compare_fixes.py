"""Compare after the review: saving waits for the series it describes, short series, a failed comparison,
an empty project, unique names and the axis of the stage texts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.gimap.features.compare.domain import compare
from tests.test_compare import Q, _close, _idle, _process, _run, _series, _shown_page, _wait


def _named_map(name: str, seed: int, rows: int = 60, *, extra: float = 0.0, x=Q, x_label: str = "q (Å⁻¹)"):
    """A map whose frame labels carry its name (“A_17”), so a label of the wrong series shows."""
    return SimpleNamespace(x=x, image=_run(rows, seed=seed, extra=extra), labels=tuple(f"{name}_{i}" for i in range(rows)),
                           x_label=x_label)


def _toasts(monkeypatch) -> list:
    from src.gimap.features.compare.presentation import page as page_module

    found = []
    monkeypatch.setattr(page_module, "show_toast", lambda parent, text, **kw: found.append((text, kw.get("level"))))
    return found


def _csv_rows(path: Path) -> list[list[str]]:
    return [line.split(",") for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]


def test_series_too_short_for_stages_are_compared_without_them() -> None:
    for rows in (3, 4):
        comparison = compare([_series("A", _run(rows)), _series("B", _run(40, seed=2))])
        assert comparison.results[0].stages is None and comparison.results[1].stages is not None, rows
        assert comparison.results[0].rows == rows
    assert compare([_series("A", _run(3))]).results[0].stages is None
    assert len(compare([_series("A", _run(4)), _series("B", _run(3, seed=2)), _series("C", _run(4, seed=3))]).results) == 3


def test_a_short_series_on_the_page_is_compared_not_an_error(monkeypatch) -> None:
    toasts = _toasts(monkeypatch)
    page = _shown_page()
    try:
        page.add_map(_named_map("short", 1, rows=3), "short")
        page.add_map(_named_map("long", 2, rows=40), "long")
        _wait(page, lambda: _idle(page) and page.comparison is not None and len(page.comparison.results) == 2)
        assert page.results_table.item(0, 4).text() == "—" and not any(level == "error" for _text, level in toasts)
    finally:
        _close(page)


def test_saving_waits_until_the_changed_series_are_compared_again(tmp_path: Path, monkeypatch) -> None:
    toasts = _toasts(monkeypatch)
    page = _shown_page()
    try:
        for seed, (name, rows) in enumerate((("A", 80), ("B", 40), ("C", 60)), start=1):
            page.add_map(_named_map(name, seed, rows, extra=1.0 if name == "B" else 0.0), name)
        _wait(page, lambda: _idle(page) and page.comparison is not None and len(page.comparison.results) == 3)
        assert page.save_button.isEnabled()
        page.series_table.selectRow(0)
        page.remove_selected()  # the comparison of A, B and C is still shown until the next one is ready
        assert page.comparison.names == ["A", "B", "C"] and not page.save_button.isEnabled()
        assert page._label(1, 5) == "B_5"  # the frames of the compared series, not of the page's row 1 (C)
        assert page.save_series_table(tmp_path / "early.csv") is None and not (tmp_path / "early.csv").exists()
        assert page.save_frames_table(tmp_path / "early_frames.csv") is None
        assert toasts[-1] == ("The series changed: save when they are compared again.", "warning")
        _wait(page, lambda: _idle(page) and len(page.comparison.results) == 2)
        assert page.save_button.isEnabled()
        table = page.save_series_table(tmp_path / "series.csv")
        rows = _csv_rows(table)[1:]
        assert [row[0] for row in rows] == ["B", "C"] and [row[1] for row in rows] == ["40", "60"]
        for row in rows:  # half and 90 % of the change: frames of that series
            assert all(label.startswith(row[0] + "_") for label in row[4:6] if label), row
        record = json.loads(table.with_suffix(".json").read_text(encoding="utf-8"))
        assert [item["name"] for item in record["series"]] == ["B", "C"]
        frames = _csv_rows(page.save_frames_table(tmp_path / "frames.csv"))[1:]
        assert len(frames) == 100 and all(row[2].startswith(row[0] + "_") for row in frames)
        # A rename: Save waits for the new name too.
        page.series_table.item(0, 0).setText("B renamed")
        assert not page.save_button.isEnabled() and page.save_series_table(tmp_path / "renamed.csv") is None
        _wait(page, lambda: _idle(page) and page.comparison.names[0] == "B renamed")
        assert page.save_button.isEnabled()
    finally:
        _close(page)


def test_a_failed_comparison_says_why_where_the_plots_were(monkeypatch) -> None:
    _toasts(monkeypatch)
    page = _shown_page()
    try:
        page.add_map(_named_map("q", 1), "q one")
        _wait(page, lambda: _idle(page) and page.comparison is not None)
        page.add_map(_named_map("chi", 2, x=np.linspace(-80, 80, Q.size), x_label="χ (°)"), "chi two")
        _wait(page, lambda: _idle(page) and page.comparison is None)
        assert page.plot_stack.currentWidget() is page.empty_page
        assert page.empty_state.title_label.text() == "Could not compare these series"
        assert "different x axes" in page.empty_state.message_label.text()
        assert page.empty_state.action_button.isHidden()  # not “add a series”: two are listed
        assert page.summary_label.text().startswith("Could not compare: The series have different x axes")
        for key in ("compare", "results"):
            assert (page.step_rail.state(key), page.step_rail.detail(key)) == ("error", "Could not compare")
        assert page.chip.text() == "2 series · 120 frames"
        # The series that does not fit removed: no stale error while the rest are compared again.
        page.series_table.selectRow(1)
        page.remove_selected()
        assert page.failure_reason() == "" and page.step_rail.state("results") == "busy"
        assert page.empty_state.title_label.text() == "Comparing …"
        _wait(page, lambda: _idle(page) and page.comparison is not None)
        assert page.plot_stack.currentWidget() is page.plots_page and page.step_rail.state("results") == "ok"
        page.clear_all()
        assert page.empty_state.title_label.text() == "Nothing to compare yet"
        assert not page.empty_state.action_button.isHidden()
        assert page.step_rail.detail("results") == "After a series is added"
    finally:
        _close(page)


def test_an_empty_project_leaves_nothing_of_the_last_comparison(monkeypatch) -> None:
    from src.gimap.features.compare.presentation.views.compare_page_view import RANGE_LIMITS

    _toasts(monkeypatch)
    page = _shown_page()
    try:
        for seed, extra in ((1, 0.0), (2, 1.0), (3, 0.0)):
            page.add_map(_named_map(f"s{seed}", seed, extra=extra), f"s{seed}")
        _wait(page, lambda: _idle(page) and page.comparison is not None and len(page.comparison.results) == 3)
        page.q_low_spin.setValue(0.5)
        _wait(page, lambda: _idle(page) and page.comparison.compared.min() >= 0.5)
        notes = page.apply_project_state({"q_range": [0.4, 2.0], "shape_only": False, "end_frames": 5, "series": []})
        _process()
        assert not notes and page.comparison is None and page.status_label.text() == ""
        assert (page.q_low_spin.value(), page.q_high_spin.value()) == (0.0, 0.0)
        assert (page.q_low_spin.minimum(), page.q_low_spin.maximum()) == RANGE_LIMITS
        assert not page._announce and page._q_range == (0.4, 2.0)
        assert not page.shape_check.isChecked() and page.end_spin.value() == 5
        assert page.empty_state.title_label.text() == "Nothing to compare yet" and page.save_button.isHidden()
    finally:
        _close(page)


def test_a_rename_to_another_series_name_is_made_unique(tmp_path: Path, monkeypatch) -> None:
    _toasts(monkeypatch)
    page = _shown_page()
    try:
        for seed, (name, extra) in enumerate((("a", 0.0), ("b", 1.0), ("c", 0.0)), start=1):
            page.add_map(_named_map(name, seed, extra=extra), name)
        _wait(page, lambda: _idle(page) and page.comparison is not None and len(page.comparison.results) == 3)
        page.series_table.item(2, 0).setText(" a ")
        assert [item.name for item in page.series] == ["a", "b", "a (2)"]
        assert page.series_table.item(2, 0).text() == "a (2)"
        assert page.status_label.text() == "a is already a series: this one is a (2)."
        _wait(page, lambda: _idle(page) and page.comparison.names == ["a", "b", "a (2)"])
        record = json.loads(page.save_series_table(tmp_path / "series.csv").with_suffix(".json").read_text(encoding="utf-8"))
        assert all(len(item["differs_at_the_end_percent"]) == 3 for item in record["series"])
        page.series_table.item(1, 0).setText("b")  # its own name: nothing changes
        assert [item.name for item in page.series] == ["a", "b", "a (2)"] and page.save_button.isEnabled()
    finally:
        _close(page)


def test_axis_symbol_is_shared_with_the_stage_texts() -> None:
    from src.gimap.app.presentation.stage_text import axis_symbol, change_text
    from src.gimap.shared.series_stages.stages import StageChange

    assert [axis_symbol(label) for label in ("q (Å⁻¹)", "χ (°)", "|q| (nm⁻¹)", "qz or qr (Å⁻¹)", "")] == ["q", "χ", "q", "qz", "x"]
    change = StageChange(stage=2, rises=((35.2, 12.0),), falls=())
    assert change_text(change, axis_symbol("χ (°)")) == "Stage 1 → 2: grows most at χ 35.2 (+12 %)"
