"""Single analysis and In-situ series read a curve file in the unit of q its header states (release 3).

One rule for all of Fitting (``application/curve_units.q_unit_read``, the one 1D Predict follows): the unit the
file's header states wins (Analyze's ``_fit_input.dat``: 1/A); the setting Curve ▸ File ▸ “q unit of files without
one in their header” is for the others, in Single analysis and in In-situ series. A header that states both units:
the file is not opened, its frame is not fitted. The Curve step and the In-situ Curves step say where the unit came
from; the fit's record, the series record and projects name it. All data here is invented.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtWidgets import QLabel

from src.gimap.features.analyze.domain.models import Curve as AnalyzeCurve
from src.gimap.features.analyze.infrastructure.adapters import CsvCurveWriter
from src.gimap.features.fitting.application import LoadCurve, LoadCurveRequest
from src.gimap.features.fitting.application.curve_units import FROM_FILE, FROM_SETTING
from src.gimap.features.fitting.application.single_fit import evaluate, fit_record, model_to_dict
from src.gimap.features.fitting.infrastructure.adapters.curve_units import BOTH_UNITS
from src.gimap.features.fitting.infrastructure.adapters.local_files import LocalCurveRepository
from tests.test_fit_series import Q_NM, RADII, _model, _pages, _wait


def _measured(radius: float, seed: int):
    exact = evaluate(_model(radius), Q_NM)
    sigma = np.sqrt(exact) + 0.5
    return exact + np.random.default_rng(seed).normal(0.0, sigma), sigma


def _analyze_file(path: Path, radius: float = 5.0, seed: int = 0) -> Path:
    """Analyze's curve for Fitting, by Analyze's writer: q in Å⁻¹, said in its header."""
    intensity, sigma = _measured(radius, seed)
    curve = AnalyzeCurve("fit_input", "Horizontal cut I(qy)", Q_NM / 10.0, intensity, sigma,
                         np.full(Q_NM.size, 5, dtype=np.int64), "qy (Å⁻¹)")
    return CsvCurveWriter().write_xy(curve, path, {"source_file": "invented_frame.cbf"})


def _plain_nm_file(path: Path, radius: float = 5.0, seed: int = 0) -> Path:
    """Another program's curve: q in nm⁻¹, not a word about it."""
    intensity, sigma = _measured(radius, seed)
    np.savetxt(path, np.column_stack([Q_NM, intensity, sigma]))
    return path


def _contradicting_file(path: Path) -> Path:
    intensity, sigma = _measured(5.0, 0)
    rows = "\n".join(f"{q:.8g} {i:.8g} {s:.8g}" for q, i, s in zip(Q_NM / 10.0, intensity, sigma))
    path.write_text(f"# q in 1/A\nq (nm^-1) I sigma\n{rows}\n", encoding="utf-8")
    return path


def _dispose(*pages) -> None:
    for page in pages:
        page.dispose()


# -- the rule, where every page reads curves ------------------------------------------------------------


def test_load_curve_reads_the_unit_the_header_states_else_the_setting(tmp_path) -> None:
    load = LoadCurve(LocalCurveRepository())
    stated = load.execute(LoadCurveRequest(_analyze_file(tmp_path / "film_C2_fit_input.dat"), "nm")).value
    assert (stated.q_source_unit, stated.q_unit_from) == ("angstrom", FROM_FILE)
    plain = load.execute(LoadCurveRequest(_plain_nm_file(tmp_path / "other.dat"), "nm")).value
    assert (plain.q_source_unit, plain.q_unit_from) == ("nm", FROM_SETTING)
    refused = load.execute(LoadCurveRequest(_contradicting_file(tmp_path / "contradicting.dat"), "nm"))
    assert refused.value is None and refused.error.message == BOTH_UNITS and refused.error.code == "invalid_data"
    missing = load.execute(LoadCurveRequest(tmp_path / "gone.dat", "nm"))
    assert missing.error.code == "not_found" and "gone.dat" in missing.error.message  # as before: the path said


# -- Single analysis --------------------------------------------------------------------------------------


def test_single_analysis_reads_an_analyze_curve_in_angstrom_with_the_setting_on_nm(tmp_path) -> None:
    single, series = _pages()
    try:
        single.set_unit_setting("nm")
        assert single.open_curve(_analyze_file(tmp_path / "film_C2_fit_input.dat"))
        curve = single.session.curve
        assert (curve.source_unit, curve.unit_from) == ("angstrom", FROM_FILE)
        np.testing.assert_allclose(np.sort(np.abs(curve.q)), Q_NM)  # nm⁻¹ for the model: ten times the file's Å⁻¹
        assert single.unit_setting() == "nm"  # the setting stays: it is for files without a unit
        assert single.curve_info.text().splitlines()[-1] == "q read in Å⁻¹: stated in the file"
        record = fit_record(single.session.model, single._data(), curve, side=single.session.side, q_range=None,
                            result=None)
        assert record["curve"]["q_unit_of_file"] == "angstrom" and record["curve"]["q_unit_from"] == FROM_FILE
        index = single.unit_combo.findData("angstrom")
        single.unit_combo.setCurrentIndex(index)
        single.unit_combo.activated.emit(index)  # the person changes the setting: this curve stays as it is read
        assert single.session.curve is curve
        assert single.status_label.text() == (
            "film_C2_fit_input.dat states q in Å⁻¹ in its header: the setting is for files without one.")
    finally:
        _dispose(series, single)


def test_single_analysis_reads_a_file_without_a_unit_in_the_setting(tmp_path) -> None:
    single, series = _pages()
    try:
        single.set_unit_setting("nm")
        assert single.open_curve(_plain_nm_file(tmp_path / "other.dat"))
        assert (single.session.curve.source_unit, single.session.curve.unit_from) == ("nm", FROM_SETTING)
        np.testing.assert_allclose(single.session.curve.q, Q_NM)
        assert single.curve_info.text().splitlines()[-1] == (
            "q read in nm⁻¹: not stated in the file: the setting for files without one")
        index = single.unit_combo.findData("angstrom")
        single.unit_combo.setCurrentIndex(index)
        single.unit_combo.activated.emit(index)  # this file follows the setting: opened again in Å⁻¹
        assert single.session.curve.source_unit == "angstrom"
        np.testing.assert_allclose(single.session.curve.q, Q_NM * 10)
    finally:
        _dispose(series, single)


def test_a_file_that_states_both_units_is_not_opened(tmp_path) -> None:
    single, series = _pages()
    try:
        assert not single.open_curve(_contradicting_file(tmp_path / "contradicting.dat"))
        assert single.session.curve is None
        assert single.status_label.text() == f"Could not open contradicting.dat: {BOTH_UNITS}"
    finally:
        _dispose(series, single)


def test_projects_saved_before_open_with_the_unit_they_were_read_in(tmp_path) -> None:
    """A project kept the unit its curve was read in (the setting then). A file without a unit opens in it, as
    before; a file whose header states another one is read in the stated unit, and the project's notes say so."""
    single, series = _pages()
    try:
        plain, analyze = _plain_nm_file(tmp_path / "other.dat"), _analyze_file(tmp_path / "film_C2_fit_input.dat")
        model = model_to_dict(single.session.model)
        notes = single.apply_project_state({"curve": {"path": str(plain), "unit": "nm"}, "model": model,
                                            "method": single.method()})
        assert notes == [] and single.session.curve.source_unit == "nm" and single.unit_setting() == "nm"
        state = single.project_state()
        assert state["curve"]["unit"] == "nm" and state["unit_setting"] == "nm"
        notes = single.apply_project_state({"curve": {"path": str(analyze), "unit": "nm"}, "model": model,
                                            "method": single.method()})
        said = "film_C2_fit_input.dat states q in Å⁻¹ in its header: read in Å⁻¹, not in nm⁻¹ as saved before."
        assert single.session.curve.source_unit == "angstrom" and notes == [said]
        assert single.project_state()["curve"]["unit"] == "angstrom"  # the unit used, from now on
        single.set_unit_setting("angstrom")
        single.apply_project_state({"curve": None, "model": model, "method": single.method(), "unit_setting": "nm"})
        assert single.unit_setting() == "nm"  # a project keeps the setting too
    finally:
        _dispose(series, single)


# -- In-situ series ------------------------------------------------------------------------------------------


def test_a_series_reads_each_file_in_its_own_unit_and_records_it(tmp_path) -> None:
    single, series = _pages()
    folder = tmp_path / "gimap_analysis"
    folder.mkdir()
    radii = RADII[:3]
    _analyze_file(folder / "run_00001_fit_input.dat", radii[0], 0)
    _plain_nm_file(folder / "run_00002_fit_input.dat", radii[1], 1)  # q in nm⁻¹, nothing said: the setting
    _analyze_file(folder / "run_00003_fit_input.dat", radii[2], 2)
    try:
        single.set_unit_setting("nm")
        single.open_curve(folder / "run_00001_fit_input.dat")
        single.set_model(_model(5.2))
        assert series.open_series(folder)
        assert series.units_note.text().splitlines() == [
            "2 curves: q in Å⁻¹, as their header states",
            "1 curves: q in nm⁻¹, no unit in their header (Single analysis ▸ Curve ▸ File)",
        ] and not series.units_note.isHidden()
        series.start()
        _wait(series)
        assert len(series.fits) == 3 and all(frame.ok for frame in series.fits.values())
        fitted = [series.fits[index].result.model.get((0, "R")).value for index in range(3)]
        assert fitted == pytest.approx(radii, rel=0.05)  # every file in its own unit: one trend
        series.save_table(str(tmp_path / "series.csv"))
        record = json.loads((tmp_path / "series.json").read_text(encoding="utf-8"))
        assert record["schema"] == "gimap_series_fit_v2" and record["q_unit_of_files"] == "nm"  # the setting
        assert [(frame["q_unit"], frame["q_unit_from"]) for frame in record["per_frame"]] == [
            ("angstrom", FROM_FILE), ("nm", FROM_SETTING), ("angstrom", FROM_FILE)]
    finally:
        _dispose(series, single)


def test_a_series_does_not_fit_a_file_that_states_both_units(tmp_path) -> None:
    single, series = _pages()
    folder = tmp_path / "gimap_analysis"
    folder.mkdir()
    for index in range(3):
        _analyze_file(folder / f"run_{index + 1:05d}_fit_input.dat", RADII[index], index)
    _contradicting_file(folder / "run_00004_fit_input.dat")
    try:
        single.open_curve(folder / "run_00001_fit_input.dat")
        single.set_model(_model(5.2))
        assert series.open_series(folder)
        assert series.units_note.text().splitlines()[-1] == (
            f"1 curves are not fitted, the unit of q cannot be read: run_00004_fit_input.dat — {BOTH_UNITS}")
        series.skip_odd_check.setChecked(False)
        series.start()
        _wait(series)
        assert series.fits[3].error == BOTH_UNITS and not series.fits[3].ok
        assert all(series.fits[index].ok for index in range(3))
    finally:
        _dispose(series, single)


# -- in Chinese ---------------------------------------------------------------------------------------------


def test_the_unit_lines_and_the_setting_in_chinese(tmp_path) -> None:
    from src.gimap.app.presentation import i18n

    single, series = _pages()
    folder = tmp_path / "gimap_analysis"
    folder.mkdir()
    for index in range(2):
        _analyze_file(folder / f"run_{index + 1:05d}_fit_input.dat", RADII[index], index)
    try:
        single.open_curve(folder / "run_00001_fit_input.dat")
        series.open_series(folder)
        i18n.apply_language("zh")
        i18n.apply_to(single, "zh")
        single.refresh_language()
        series.refresh_language()
        assert "文件头未注明时的 q 单位" in [label.text() for label in single.findChildren(QLabel)]
        assert single.unit_combo.toolTip().startswith("文件头注明了 q 单位的文件按注明的单位读取")
        assert single.curve_info.text().splitlines()[-1] == "q 按 Å⁻¹ 读取：文件中注明"
        assert series.units_note.text() == "2 条曲线：q 按文件头注明的 Å⁻¹"
        i18n.apply_language("en")
        single.refresh_language()
        series.refresh_language()
        assert single.curve_info.text().splitlines()[-1] == "q read in Å⁻¹: stated in the file"
        assert series.units_note.text() == "2 curves: q in Å⁻¹, as their header states"
    finally:
        i18n.apply_language("en")
        i18n.apply_to(single, "en")
        _dispose(series, single)
