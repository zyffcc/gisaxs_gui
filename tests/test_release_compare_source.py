"""Release 0.1.0, Compare: the From column of the Series table names the Analyze workspace in the interface
language; the series itself (the record, the project) keeps the English word, and a folder's name stays as it is."""

from __future__ import annotations

from pathlib import Path

from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language, translate
from tests.test_compare import _close, _idle, _map, _run, _shown_page, _wait, _write_curves


def test_the_from_column_names_the_workspace_in_the_interface_language(tmp_path: Path) -> None:
    page = _shown_page()
    try:
        page.add_map(_map(1), "a")
        _write_curves(tmp_path / "run_A", _run(60, seed=2))
        page.add_folder(tmp_path / "run_A")
        table = page.series_table
        assert table.item(0, 2).text() == "Analyze" and table.item(1, 2).text() == "run_A"
        apply_language("zh", [page])
        page.refresh_language()
        workspace = translate("Analyze", "zh")
        assert workspace == "分析" and table.horizontalHeaderItem(2).text() == translate("From", "zh")
        assert [table.item(0, column).text() for column in range(3)] == ["a", "60", workspace]
        assert {table.item(0, column).toolTip() for column in range(3)} == {workspace}
        # A folder: its name and its path are data.
        assert table.item(1, 2).text() == "run_A" and table.item(1, 2).toolTip() == str(tmp_path / "run_A")
        # The series keeps the English word: the record and the project hold it, and saving looks for a folder.
        assert page.series[0].source == "Analyze" and page._folder() == str(tmp_path / "run_A")
        _wait(page, lambda: _idle(page) and page.comparison is not None and len(page.comparison.results) == 2)
        assert page.service.record(page.series, page.comparison)["series"][0]["source"] == "Analyze"
        assert page.project_state(tmp_path / "project.gimap")["series"][0]["source"] == "Analyze"
        apply_language(DEFAULT_LANGUAGE, [page])
        page.refresh_language()
        assert table.item(0, 2).text() == "Analyze" == table.item(0, 2).toolTip()
    finally:
        apply_language(DEFAULT_LANGUAGE)
        _close(page)
