"""Fitting tables and names: copy (fitting-11), the parameters table without its own scroll bar, short names
for a model of one particle and the Fit step's intro (fitting-13)."""

from __future__ import annotations

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QAbstractItemView, QApplication

from src.gimap.features.fitting.application.single_fit import FitModel, new_component
from tests.test_fit_page import _curve_file, _page
from tests.test_fit_series import _model, _pages, _series, _wait
from tests.test_wave2b_fitting_series_list import _close


def _ctrl_c(table) -> str:
    QApplication.clipboard().clear()
    QApplication.sendEvent(table, QKeyEvent(QEvent.KeyPress, Qt.Key_C, Qt.ControlModifier))
    return QApplication.clipboard().text()


def test_the_tables_copy_their_rows_and_the_solutions_keep_one_current_row(tmp_path) -> None:
    page = _page()
    page.open_curve(_curve_file(tmp_path))
    table = page.parameters_table
    assert table.selectionMode() == QAbstractItemView.ExtendedSelection
    assert page.solutions_table.selectionMode() == QAbstractItemView.SingleSelection  # “Use This Solution”
    assert [action.text() for action in table.actions()] == ["Copy Rows", "Copy Table"]
    assert page.solutions_table.actions() and table.contextMenuPolicy() == Qt.ActionsContextMenu
    whole = _ctrl_c(table)  # nothing selected: the whole table
    lines = whole.splitlines()
    assert lines[0] == "\tvalue\t±" and len(lines) == table.rowCount() + 1
    assert any(line.startswith("R\t") and "nm" in line for line in lines)
    table.selectRow(0)
    table.selectionModel().select(table.model().index(2, 0), table.selectionModel().Select | table.selectionModel().Rows)
    assert len(_ctrl_c(table).splitlines()) == 3  # the header and the two rows chosen
    _close(page)


def test_the_parameters_table_shows_every_row_and_only_the_page_scrolls(tmp_path) -> None:
    page = _page()
    page.open_curve(_curve_file(tmp_path))
    page.show_step("results")
    QApplication.processEvents()
    table = page.parameters_table
    assert table.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert table.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    rows = table.verticalHeader().length()
    assert table.rowCount() >= 5 and table.height() == table.horizontalHeader().height() + rows + 2 * table.frameWidth()
    assert table.viewport().height() >= rows  # no row hidden under the edge
    two = FitModel((new_component("sphere"), new_component("cylinder")))
    page.set_model(two)
    QApplication.processEvents()
    assert table.verticalHeader().length() > rows and table.height() > table.horizontalHeader().height() + rows
    _close(page)


def test_a_model_of_one_particle_names_its_values_short_and_the_full_name_is_on_hover(tmp_path) -> None:
    from src.gimap.features.fitting.presentation.single.session import path_name

    one = FitModel((new_component("sphere"),))
    two = FitModel((new_component("sphere"), new_component("cylinder")))
    assert path_name(one, (0, "R"), unit=True) == "R (nm)"
    assert path_name(one, (0, "R"), unit=True, full=True) == "1·Sphere R (nm)"
    assert path_name(two, (1, "R")) == "2·Random cylinder R" and path_name(two, ("globals", "background")) == "Background"
    page = _page()
    page.open_curve(_curve_file(tmp_path))
    table = page.parameters_table
    names = {table.item(line, 0).text(): table.item(line, 0).toolTip() for line in range(table.rowCount())}
    assert names["R"] == "1·Sphere R" and "1·Sphere R" not in names
    page.set_model(two)
    names = [table.item(line, 0).text() for line in range(table.rowCount())]
    assert "1·Sphere R" in names and "2·Random cylinder R" in names
    _close(page)


def test_the_series_table_copies_and_its_headers_are_short_with_the_full_name_on_hover(tmp_path) -> None:
    single, series = _pages()
    folder = _series(tmp_path)
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    series.open_series(folder)
    combo = series.trend_combo
    assert combo.itemText(combo.currentIndex()) == "R (nm)"
    assert combo.itemData(combo.currentIndex(), Qt.ToolTipRole) == "1·Sphere R (nm)"
    series.start()
    _wait(series)
    table = series.results_table
    assert table.selectionMode() == QAbstractItemView.ExtendedSelection
    header = [table.horizontalHeaderItem(column).text() for column in range(table.columnCount())]
    assert header[:4] == ["#", "χ²ᵣ", "Scale", "R (nm)"]
    assert table.horizontalHeaderItem(3).toolTip() == "1·Sphere R (nm)"
    text = _ctrl_c(table)
    assert text.splitlines()[0].split("\t")[:4] == header[:4] and len(text.splitlines()) == 6
    path = series.save_table(str(tmp_path / "series.csv"))
    first = open(path, encoding="utf-8").readline()
    assert "1_Sphere_R_nm" in first  # the saved table names every value in full
    _close(series, single)


def test_the_fit_step_says_what_to_do() -> None:
    page = _page()
    assert page.step_intro["fit"].text() == "Choose a method; the button below runs it (Ctrl+Return)."
    _close(page)
