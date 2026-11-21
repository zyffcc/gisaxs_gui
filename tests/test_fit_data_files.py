"""Fitting opens every curve GIMaP writes, CSV among them, with q, I and σ chosen by the header's names (Fitting's own
Save ▸ Data and Fit table has q twice: in nm⁻¹, then in Å⁻¹), asks for the columns when the header does not say,
and Save Plot writes the curve with its residuals and error bars as PNG, SVG or PDF."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.gimap.features.fitting.application import COLUMNS_TO_CHOOSE, LoadCurve, LoadCurveRequest
from src.gimap.features.fitting.application.curve_sources import ColumnMapping
from src.gimap.features.fitting.infrastructure.adapters import LocalCurveRepository
from src.gimap.features.fitting.infrastructure.adapters.curve_columns import header_columns
from src.gimap.features.fitting.infrastructure.adapters.curve_tables import inspect_curve_table
from tests.test_fit_page import Q_NM, _curve_file, _page, _truth

Q_A = Q_NM / 10.0


def _measured():
    from src.gimap.features.fitting.application.single_fit import evaluate

    exact = evaluate(_truth(), Q_NM)
    sigma = np.sqrt(exact) + 0.5
    return exact + np.random.default_rng(3).normal(0.0, sigma), sigma


def _agent_csv(path):
    """A cut as the command line writes it: ``#`` lines of provenance, then the names in the last one."""
    intensity, sigma = _measured()
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        np.savetxt(stream, np.column_stack([Q_A, intensity, sigma]), delimiter=",",
                   header="GIMaP 0.0.0 (command line); settings and decisions in report.json\nqy (A^-1), I, sigma",
                   comments="# ")
    return path


def _load(path, unit="angstrom"):
    return LoadCurve(LocalCurveRepository()).execute(LoadCurveRequest(path, unit))


def test_header_names_choose_the_columns_and_the_unit_of_q(tmp_path) -> None:
    intensity, sigma = _measured()
    agent = _load(_agent_csv(tmp_path / "horizontal.csv"))
    assert agent.succeeded and agent.value.q_source_unit == "angstrom" and agent.value.q_unit_from == "stated in the file"
    assert np.allclose(agent.value.q, Q_A) and np.allclose(agent.value.error, sigma)
    # Fitting's own table: q in nm⁻¹ first, q in Å⁻¹ second — the first q, read in nm⁻¹, never Å⁻¹ as I.
    table = tmp_path / "fit.csv"
    columns = np.column_stack([Q_NM, Q_A, intensity, sigma, intensity * 1.1, np.zeros_like(Q_NM)])
    np.savetxt(table, columns, delimiter=",", header="q_nm^-1,q_A^-1,I,sigma,model,residual", comments="", fmt="%.8g")
    loaded = _load(table)
    assert loaded.succeeded and loaded.value.q_source_unit == "nm"
    assert np.allclose(loaded.value.q, Q_NM) and np.allclose(loaded.value.intensity, intensity, rtol=1e-7)
    assert loaded.value.observation["curve_source"]["columns"] == {
        "q": 0, "intensity": 2, "sigma": 3, "column_names": ["q_nm^-1", "q_A^-1", "I", "sigma", "model", "residual"],
        "column_count": 6} and loaded.value.observation["curve_source"]["chosen_by"] == "header"
    bare = tmp_path / "bare.csv"  # no header, three columns: q, I, σ by their place, as a .dat file
    np.savetxt(bare, np.column_stack([Q_A, intensity, sigma]), delimiter=",")
    assert _load(bare).succeeded and _load(bare).value.q_unit_from.startswith("not stated")
    for name, header in (("two_i.csv", "q,I1,I2,sigma"), ("axes.csv", "qy (A^-1),qz (A^-1),I")):
        path = tmp_path / name
        np.savetxt(path, np.column_stack([Q_A, intensity, intensity, sigma])[:, :3 if "axes" in name else 4],
                   delimiter=",", header=header, comments="")
        outcome = _load(path)
        assert outcome.error is not None and outcome.error.message == COLUMNS_TO_CHOOSE
    wide = tmp_path / "wide.csv"  # no header, four columns: not to be guessed
    np.savetxt(wide, np.column_stack([Q_A, intensity, intensity, sigma]), delimiter=",")
    assert _load(wide).error.message == COLUMNS_TO_CHOOSE
    assert header_columns(inspect_curve_table(table)).q == 0


def test_the_page_opens_every_curve_file_gimap_writes(tmp_path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QInputDialog

    # Save ▸ Data and Fit as NXcanSAS holds the points fitted and the model: the first is the curve.
    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(lambda _parent, _title, _label, items, *_rest: (items[0], True)))
    page = _page()
    other = _page()
    try:
        assert page.open_curve(_curve_file(tmp_path)), page.status_label.text()  # Analyze's _fit_input-like .dat
        written = {suffix: page.export_data_dialog(str(tmp_path / f"saved{suffix}")) for suffix in (".csv", ".h5")}
        data = page._data()
        for suffix, path in written.items():
            assert path and other.open_curve_file(path), (suffix, other.status_label.text())
            mine = other._data()
            assert np.allclose(mine.q, data.q) and np.allclose(mine.intensity, data.intensity, rtol=1e-6)
            assert np.allclose(mine.sigma, data.sigma, rtol=1e-6)
        record = json.loads((tmp_path / "saved.json").read_text(encoding="utf-8"))
        assert record["curve"]["columns_read"] is None  # a .dat read by place
        assert other.open_curve_file(str(_agent_csv(tmp_path / "horizontal.csv"))), other.status_label.text()
        assert np.allclose(other.session.curve.q, Q_NM)  # Å⁻¹ in the file, nm⁻¹ in Fitting
        assert other.export_data_dialog(str(tmp_path / "again.csv"))
        record = json.loads((tmp_path / "again.json").read_text(encoding="utf-8"))
        assert record["curve"]["columns_read"]["column_names"] == ["qy (A^-1)", "I", "sigma"]
    finally:
        page.dispose()
        other.dispose()


def test_a_table_whose_header_does_not_say_asks_for_its_columns(tmp_path, monkeypatch) -> None:
    from src.gimap.features.fitting.presentation.curve_columns_dialog import CurveColumnsDialog
    from src.gimap.features.fitting.presentation.single import page as page_module

    intensity, sigma = _measured()
    path = tmp_path / "two.csv"
    np.savetxt(path, np.column_stack([Q_NM, intensity * 3, intensity, sigma]), delimiter=",",
               header="q (1/nm),I1,I2,sigma", comments="")
    asked = []

    def choose(parent, where, inspect, unit, stated_unit=None):
        preview = inspect(where)
        dialog = CurveColumnsDialog(preview, unit, lambda mapping: parent.view_model.stated_q_unit(
            page_module.InSituSourceFrame(where, columns=mapping).token), parent)
        dialog.intensity_column_combo.setCurrentIndex(2)  # I2
        asked.append((dialog.columns_note.text(), dialog.columns_unit_combo.isEnabled(), dialog.columns_preview.rowCount()))
        dialog.q_column_combo.setCurrentIndex(2)  # q and I the same column: refused before Open
        assert not dialog.columns_buttons.button(dialog.columns_buttons.Ok).isEnabled()
        assert dialog.columns_problem.text() == "q, I and σ must be three different columns."
        dialog.q_column_combo.setCurrentIndex(0)
        chosen = dialog.chosen()
        dialog.deleteLater()
        return chosen

    monkeypatch.setattr(page_module, "choose_columns", choose)
    page = _page()
    try:
        assert page.open_curve_file(str(path)), page.status_label.text()
        assert asked[0][0] == COLUMNS_TO_CHOOSE and asked[0][1] is False and asked[0][2] == 10  # unit in the name
        assert np.allclose(page.session.curve.q, Q_NM) and np.allclose(page.session.curve.intensity, intensity)
        assert page.session.curve.columns["intensity"] == 2 and page.session.curve.columns["sigma"] == 3
        monkeypatch.setattr(page_module, "choose_columns", lambda *_args, **_kwargs: None)
        assert page.open_curve_file(str(path)) is False
        assert page.status_label.text() == "two.csv was not opened: its columns were not chosen."
    finally:
        page.dispose()


def test_a_chosen_q_column_is_read_in_the_unit_its_name_states(tmp_path) -> None:
    intensity, sigma = _measured()
    path = tmp_path / "fit.csv"
    np.savetxt(path, np.column_stack([Q_NM, Q_A, intensity, sigma]), delimiter=",",
               header="q_nm^-1,q_A^-1,I,sigma", comments="")
    names = ("q_nm^-1", "q_A^-1", "I", "sigma")
    repository = LocalCurveRepository()
    for q, unit in ((0, "nm"), (1, "angstrom")):
        token = LocalCurveRepository().inspect_table(path).path
        from src.gimap.features.fitting.application.models import InSituSourceFrame

        source = InSituSourceFrame(token, columns=ColumnMapping(q, 2, 3, names, 4)).token
        assert repository.stated_q_unit(source) == unit
        loaded = _load(source)
        assert loaded.succeeded and np.allclose(loaded.value.q * (10 if unit == "angstrom" else 1), Q_NM)


@pytest.mark.parametrize("suffix", [".png", ".svg", ".pdf"])
def test_save_plot_writes_the_curve_with_its_residuals(tmp_path, suffix) -> None:
    page = _page()
    try:
        assert page.open_curve(_curve_file(tmp_path))
        request = page.figure_request(tmp_path / f"plot{suffix}")
        assert request.residuals and request.residual_label == "Δ/σ" and request.band is not None
        measured = next(series for series in request.series if series.label == "measured")
        assert measured.error is not None and measured.error.size == measured.x.size  # error bars
        saved = page.export_plot_dialog(str(tmp_path / f"plot{suffix}"))
        assert saved and (tmp_path / f"plot{suffix}").stat().st_size > 1000, page.status_label.text()
        if suffix == ".pdf":
            assert (tmp_path / "plot.pdf").read_bytes()[:4] == b"%PDF"
        assert page.export_plot_dialog(str(tmp_path / "plot.jpg2")) is None
        assert page.status_label.text() == "Choose a .png, .svg or .pdf file: plot.jpg2 is none of them."
    finally:
        page.dispose()


def test_a_long_status_line_is_shortened_and_kept_off_the_notices(tmp_path) -> None:
    from PyQt5.QtWidgets import QApplication

    page = _page()
    try:
        long = "A very long message of the fit " * 20
        page._status(long)
        QApplication.processEvents()
        label = page.status_label
        assert label.text() == long and label.toolTip() == long  # the whole text: there and in the tooltip
        assert QLabel_text(label) != long and QLabel_text(label).endswith("…")
        assert label.maximumWidth() <= int(0.6 * page.width()) + 1
    finally:
        page.dispose()


def QLabel_text(label) -> str:  # noqa: N802 - the text Qt draws (not the whole one ``text()`` gives)
    from PyQt5.QtWidgets import QLabel

    return QLabel.text(label)
