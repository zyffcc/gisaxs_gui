"""What Fitting says of a curve as it takes it in — one rule on the Fit page, in the 1D Predict job and in the fit of
one cut (``fitting/application/curve_intake.py``):

* one default unit of q for files whose header states none (Å⁻¹), and a warning next to the result when the q range
  that gives is implausible for a GISAXS / SAXS cut;
* one σ for a file without one (Predict parameters' estimate from the scatter of ln I), said in the same words;
* one reader (the Fit page's): a title line above the column names is a header in both places;
* negative and zero intensities counted, and said with what was done with them.

All curves are invented.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from src.gimap.features.fitting.application import curve_intake
from src.gimap.features.fitting.application.curve_units import FROM_FILE, FROM_SETTING
from src.gimap.features.fitting.application.workflow_v5 import default_options, validate_options
from src.gimap.features.fitting.domain.posterior.infer import Cut
from src.gimap.features.fitting.infrastructure.adapters import workflow_v5
from tests.test_fit_page import _page

Q_NM = np.linspace(0.05, 2.5, 120)
"""q of the invented cut in nm⁻¹."""


def _intensity(q=Q_NM, seed: int = 3):
    exact = 4e4 / (1 + (q * 5.0) ** 4) + 6.0
    return exact * np.exp(np.random.default_rng(seed).normal(0.0, 0.03, q.size))


def _file(folder, name, columns, header: str = ""):
    path = folder / name
    np.savetxt(path, np.column_stack(columns), fmt="%.6g", header=header, comments="")
    return path


def _run(tmp_path, monkeypatch, files, **options):
    """The 1D Predict job with the fit replaced by a recorder of what it was given."""
    from src.gimap.features.fitting.infrastructure.adapters import posterior_job

    given = []

    def fit(_output, item, _options, _report, _cancelled):
        given.append(item)
        return [dict(side=item["side"], best_log_rmse=0.1)]

    monkeypatch.setattr(posterior_job, "fit_and_write", fit)
    output = tmp_path / "run"
    result = workflow_v5.run_workflow_job(
        dict(output_dir=str(output), files=[str(path) for path in files],
             options=dict(method="posterior", side="positive", **options)), lambda *_args: None, lambda: False)
    return result, given, output


# -- B: one default unit, and the warning ----------------------------------------------------------------------


def test_one_default_unit_for_the_fit_page_and_the_1d_predict_window(tmp_path) -> None:
    from src.gimap.features.fitting.presentation.workflow_v5_dialog import WorkflowV5Dialog

    assert default_options()["q_unit"] == "A^-1" and validate_options({})["q_unit"] == "A^-1"
    assert validate_options({"q_unit": "nm^-1"})["q_unit"] == "nm^-1"  # a saved setting keeps its unit
    page = _page()
    dialog = WorkflowV5Dialog(options={})
    try:
        assert page.unit_setting() == "angstrom" and dialog.unit.currentData() == "A^-1"
    finally:
        dialog.close()
        dialog.deleteLater()
        page.dispose()


@pytest.mark.parametrize("q_max_nm, kind", [(25.0, curve_intake.UNIT_TOO_LARGE), (0.01, curve_intake.UNIT_TOO_SMALL),
                                            (2.5, None), (9.9, None), (0.03, None)])
def test_an_unstated_unit_is_doubted_when_the_q_range_is_implausible(q_max_nm, kind) -> None:
    q = np.linspace(0.02, 1.0, 50) * q_max_nm
    note = curve_intake.unit_note(q, "angstrom", FROM_SETTING)
    # a plausible range: the unit is still said to be assumed (nothing in the numbers tells Å⁻¹ from nm⁻¹)
    assert note.kind == "q_unit" and note.template == (kind or curve_intake.UNIT_ASSUMED)
    assert curve_intake.unit_note(q, "angstrom", FROM_FILE) is None  # a stated unit is the file's own word
    if note is not None:
        assert note.values["unit"] == "Å⁻¹" and note.values["other"] == "nm⁻¹"
        assert "q unit of files without one in their header" in note.text and "# q in 1/nm" in note.text
        assert curve_intake.note_from_dict(note.as_dict()) == note  # a record's entry can be said in any language


def test_the_fit_page_warns_next_to_the_result_of_a_file_read_in_the_wrong_unit(tmp_path) -> None:
    intensity = _intensity()
    plain = _file(tmp_path, "cut_in_inverse_nm.txt", [Q_NM, intensity, 0.03 * intensity])
    stated = _file(tmp_path, "stated.txt", [Q_NM, intensity, 0.03 * intensity], header="q(nm^-1) I err")
    page = _page()
    try:
        assert page.open_curve(plain)  # no unit in the file: read in Å⁻¹, |q| up to 25 nm⁻¹
        notes = page.curve_notes()
        assert [note.kind for note in notes] == ["q_unit"] and notes[0].template == curve_intake.UNIT_TOO_LARGE
        for label in (page.sigma_warning, page.weighting_note, page.curve_notes_label):  # Curve, Fit and Results
            assert not label.isHidden() and label.property("gimapRole") == "warning"
            assert "read in Å⁻¹" in label.text() and "10× too small" in label.text() and "to nm⁻¹" in label.text()
        results = page.step_pages["results"]
        assert results.isAncestorOf(page.curve_notes_label) and not page.fit_notes_dialog.isAncestorOf(page.curve_notes_label)
        assert "read in Å⁻¹" in page.status_label.text()
        page.set_unit_setting("nm")
        page._unit_setting_changed()  # the person follows the message: the file is read again, in a plausible range
        assumed = page.curve_notes()  # the unit is still the setting's, not the file's: said, no longer doubted
        assert page.session.curve.source_unit == "nm" and [note.template for note in assumed] == [curve_intake.UNIT_ASSUMED]
        assert page.curve_notes_label.text().startswith("⚠ Unit of q assumed: nm⁻¹ (not stated in the file).")
        assert "10× too" not in page.sigma_warning.text()
        assert page.open_curve(stated) and page.curve_notes() == []
    finally:
        page.dispose()


def test_the_1d_predict_job_records_the_doubt_and_the_window_shows_it(tmp_path, monkeypatch) -> None:
    from src.gimap.features.fitting.presentation.workflow_v5_dialog import WorkflowV5Dialog

    intensity = _intensity()
    plain = _file(tmp_path, "cut_in_inverse_nm.txt", [Q_NM, intensity, 0.03 * intensity])
    result, _given, output = _run(tmp_path, monkeypatch, [plain])  # the default: Å⁻¹, as on the Fit page
    row = result["candidates"][0]
    assert row["q_unit_read"] == "A^-1" and [note["kind"] for note in row["curve_notes"]] == ["q_unit"]
    assert "10× too small" in row["curve_notes"][0]["text"]
    assert json.loads((output / "batch_summary.json").read_text(encoding="utf-8"))[0]["curve_notes"] == row["curve_notes"]
    row.update(file=plain.name, rank=1, best_source="posterior", combination="sphere", signed_weighted_rms=1.0,
               native_q=[0.1, 0.2], observed=[2.0, 1.0], sigma=[0.1, 0.1], display_q=[0.1, 0.2], display_fit=[2.0, 1.0])
    dialog = WorkflowV5Dialog(options={})
    try:
        dialog.set_results([row], output)
        label = dialog.curve_notes_label
        assert not label.isHidden() and label.property("gimapRole") == "warning"
        assert label.text().startswith(f"⚠ {plain.name} / positive: The file states no unit of q")
        assert "The file states no unit of q" in dialog.table.item(0, 0).toolTip()
        assert dialog.parameters.toPlainText().startswith("⚠ The file states no unit of q")
        dialog.set_results([{**row, "curve_notes": []}], output)
        assert label.isHidden()
    finally:
        dialog.close()
        dialog.deleteLater()


# -- G(i): one σ for a file without one --------------------------------------------------------------------------


def test_a_file_without_sigma_gets_the_same_sigma_on_the_fit_page_and_in_the_job(tmp_path, monkeypatch) -> None:
    path = _file(tmp_path, "no_sigma.txt", [Q_NM, _intensity()], header="q(nm^-1) I")
    q, intensity, _none = workflow_v5.read_curve(path)  # the values as the file has them
    network = Cut.prepare(q, intensity, None)  # what Predict parameters makes of the Fit page's payload (σ: None)
    level = curve_intake.sigma_level(q, intensity)
    np.testing.assert_allclose(curve_intake.relative_sigma(q, intensity)[-network.q.size:], network.sigma)
    assert 0.01 <= level < 0.1

    result, given, _output = _run(tmp_path, monkeypatch, [path])
    item = given[0]
    np.testing.assert_allclose(item["sigma"][-network.q.size:], network.sigma)  # not 10 % of I any more
    row = result["candidates"][0]
    assert row["sigma_source"].startswith("estimated: a relative sigma from the scatter of ln I")
    assert row["sigma_relative"] == pytest.approx(level)
    said = [note for note in row["curve_notes"] if note["kind"] == "sigma_estimated"]
    assert said and said[0]["text"] == curve_intake.SIGMA_ESTIMATED.format(percent=f"{100 * level:.3g}")

    page = _page()
    try:
        assert page.open_curve(path)
        assert page._data().sigma is None  # Predict's job gets σ = None and estimates it by the same rule
        card = page.sigma_warning.text()
        assert f"a relative σ of {100 * level:.3g} %, estimated from the scatter of ln I about its trend" in card
        assert "every point weighted alike (ln I fitted)" not in card and "Refine and Search: ln I is fitted" in card
        page.set_method("predict")  # the Fit step says what its method does, in the job's words
        assert page.weighting_note.text() == "⚠ " + said[0]["text"]
        page.set_method("local")
        assert page.weighting_note.text() == "⚠ No σ in the file: every point weighted alike"
    finally:
        page.dispose()


# -- G(ii): one reader -------------------------------------------------------------------------------------------


def test_a_title_line_above_the_column_names_opens_in_both_places(tmp_path, monkeypatch) -> None:
    intensity = _intensity()
    path = tmp_path / "two_line_header.txt"
    np.savetxt(path, np.column_stack([Q_NM, intensity, 0.03 * intensity]), fmt="%.6g", delimiter=", ", comments="",
               header="Sample NP_07 horizontal cut 2031-02-03\nqy [1/nm], intensity, error")
    page = _page()
    try:
        assert page.open_curve(path)
        on_page = page.session.curve
        assert on_page.source_unit == "nm" and on_page.unit_from == FROM_FILE
    finally:
        page.dispose()
    result, given, _output = _run(tmp_path, monkeypatch, [path])
    assert result["records"][0]["status"] == "complete" and result["records"][0]["q_unit_read"] == "nm^-1"
    np.testing.assert_allclose(given[0]["q"], np.abs(on_page.q))
    np.testing.assert_allclose(given[0]["observed"], on_page.intensity)
    np.testing.assert_allclose(given[0]["sigma"], on_page.sigma)
    q, y, sigma = workflow_v5.read_curve(path)
    assert q.size == Q_NM.size and sigma is not None


# -- G(iii): negative and zero intensities -------------------------------------------------------------------------


def test_negative_and_zero_intensities_are_counted_and_said(tmp_path, monkeypatch) -> None:
    intensity = _intensity()
    low = intensity.copy()
    low[-12:] -= 30.0  # a background subtracted too far: negative points
    low[40:43] = 0.0
    negative, zero = int((low < 0).sum()), 3
    assert negative > 5
    with_sigma = _file(tmp_path, "with_sigma.txt", [Q_NM, low, 0.03 * intensity], header="q(nm^-1) I err")
    without = _file(tmp_path, "without_sigma.txt", [Q_NM, low], header="q(nm^-1) I")

    result, given, _output = _run(tmp_path, monkeypatch, [with_sigma, without])
    kept, left_out = given
    assert len(kept["q"]) == Q_NM.size and len(left_out["q"]) == Q_NM.size - negative - zero
    notes = [next(note for note in row["curve_notes"] if note["kind"] == "nonpositive") for row in result["candidates"]]
    assert notes[0] == dict(kind="nonpositive", text=curve_intake.NONPOSITIVE_KEPT.format(
        count=negative + zero, points=Q_NM.size, negative=negative, zero=zero), count=negative + zero,
        negative=negative, zero=zero, points=Q_NM.size, left_out=False)
    assert notes[1]["left_out"] is True and notes[1]["text"].startswith(f"{negative + zero} points with a negative or zero")

    page = _page()
    try:
        assert page.open_curve(with_sigma)  # fitted like any point: said in the card and the status line, no warning
        said = f"{negative + zero} of the {Q_NM.size} fitted points have a negative or zero intensity"
        assert said in page.curve_info.text() and said in page.status_label.text()
        assert page.sigma_warning.isHidden() and page.curve_notes_label.isHidden()
        note = next(note for note in page.curve_notes() if note.kind == "nonpositive")
        assert (note.values["negative"], note.values["zero"], note.values["left_out"]) == (negative, zero, False)
        assert page.open_curve(without)  # ln I has no value there: left out, a warning
        said = f"{negative + zero} points with a negative or zero intensity ({negative} negative, {zero} zero) were left out"
        assert said in page.sigma_warning.text() and said in page.status_label.text()
        assert said in page.curve_notes_label.text() and page._data().q.size == Q_NM.size - negative - zero
    finally:
        page.dispose()


def test_the_fit_of_one_cut_says_it_in_its_rows_and_the_report(monkeypatch) -> None:
    from src.gimap.features.assistant.application.gisaxs_report import gisaxs_sections

    item = workflow_v5.prepare_sides(Q_NM, np.r_[_intensity()[:-2], -1.0, 0.0], None,
                                     {**default_options(), "q_unit": "nm^-1", "side": "positive"})[0]
    notes = workflow_v5.side_notes(item, {})
    assert [note["kind"] for note in notes] == ["sigma_estimated", "nonpositive"] and notes[1]["count"] == 2
    fit = {"curve": "mean of both halves", "points": 118, "q_range_inv_angstrom": [0.005, 0.25], "curve_notes": notes,
           "solutions": []}
    text = "\n".join(gisaxs_sections({"gisaxs": {"fit": fit}}))
    assert f"- {notes[0]['text']}" in text and "2 points with a negative or zero intensity" in text
