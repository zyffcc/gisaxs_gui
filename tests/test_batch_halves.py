"""The halves of the signed GISAXS cut in Batch Export and Send Series to Fitting.

A fit takes one curve of the horizontal cut: the mean of its two halves, or one half — never both halves
together. The choice is made in the dialog (one combo, started from the Cuts step), goes to the particle-model
fit of the batch and to its trial on the frame on screen, is written into every fit input (the file itself
keeps both halves, signed) and into the table and the record of the batch. A frame whose fit fails does not
end the batch as a success.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.gimap.features.analyze.application import (
    FIT_HALVES,
    AnalyzeSettings,
    BatchChoices,
    FitStarts,
    FitTable,
    fit_halves,
    fit_input_observation,
    fit_model_curve,
    model_fit_curve,
    model_fit_frame,
    model_row,
    readme_text,
    settings_from_record,
    settings_record,
)
from src.gimap.features.analyze.domain import Corrections, Curve
from src.gimap.shared.signed_cut import BAD_SIGMA, MISSING_HALF
from tests.test_analyze_workspace import CBF, _app, _context, _pilatus_profile, requires_data
from tests.test_giwaxs_workspace import _settle

CBFS = sorted(CBF.parent.glob("*.cbf"))


# -- the choice -------------------------------------------------------------------------------------------------


def test_the_choice_is_one_of_three_and_a_stored_display_choice_becomes_the_mean() -> None:
    assert FIT_HALVES == ("mean", "positive", "negative")
    assert BatchChoices().fit_halves == "mean"
    for halves in FIT_HALVES:
        choices = BatchChoices(fit_halves=halves)
        assert choices.to_dict()["fit_halves"] == halves
        assert BatchChoices.from_dict(choices.to_dict()) == choices
    for stored in ("both", "both_abs", "left", None, 3, ""):
        assert BatchChoices.from_dict({"fit_halves": stored}).fit_halves == "mean", stored
    assert BatchChoices.from_dict({"tables": True}).fit_halves == "mean"  # a record written before the choice
    # … and it comes back with a settings file (Load / Save Settings…)
    settings = AnalyzeSettings(mode="gisaxs", export=BatchChoices(fit="model", fit_halves="negative").to_dict())
    loaded = settings_from_record(json.loads(json.dumps(settings_record(settings))))
    assert BatchChoices.from_dict(loaded.export).fit_halves == "negative"
    assert fit_halves("both_abs") == "mean" and fit_halves("negative") == "negative"


# -- the fit input says the halves; the file keeps both -----------------------------------------------------------


def _frame(kind: str = "gisaxs"):
    """Just what ``fit_input_observation`` reads of a frame."""
    return SimpleNamespace(kind=kind, corrections=Corrections(), path=Path("frame_0001.cbf"), metadata={},
                           intensity_scale=None, frame_total=1)


def test_the_observation_names_the_halves_only_when_a_fit_choice_is_given() -> None:
    plain = fit_input_observation(_frame())
    assert "halves" not in plain and plain["source"] == "native_detector_columns"
    for halves in FIT_HALVES:
        record = fit_input_observation(_frame(), halves)
        assert record["halves"] == halves and {key: record[key] for key in plain} == plain
    # the display choice of the Cuts step is no choice of a fit: nothing is written, Fitting decides
    for other in ("both_abs", "both", "", None):
        assert "halves" not in fit_input_observation(_frame(), other)
    assert "halves" not in fit_input_observation(_frame("giwaxs"), "negative")  # I(q) has no halves


def _signed_cut(points_per_half: int = 60, *, positive: bool = True, negative: bool = True, scale: float = 1.0):
    """A signed cut I(qy) of a sphere model, q in Å⁻¹ (the q > 0 half ``scale`` times the other)."""
    from src.gimap.features.fitting.application.single_fit import FitModel, evaluate, new_component

    model = FitModel((new_component("sphere", radius=4.0, distance=40.0),)).with_values(
        {(0, "Int"): 2000.0, (0, "sigma_R"): 0.15, (0, "sigma_D"): 0.3, ("globals", "background"): 3.0})
    q = np.linspace(0.004, 0.2, points_per_half)
    rng = np.random.default_rng(5)
    parts = []
    for sign, wanted, factor in ((-1.0, negative, 1.0), (1.0, positive, scale)):
        if not wanted:
            continue
        ideal = evaluate(model, q * 10.0) * factor  # the model is a function of q in nm⁻¹
        pixels = np.full(q.size, 6.0)
        counts = rng.poisson(ideal * pixels) / pixels
        parts.append((sign * q, counts, np.sqrt(np.maximum(counts * pixels, 1.0)) / pixels, pixels))
    x, y, s, n = (np.concatenate(column) for column in zip(*parts))
    order = np.argsort(x)
    return Curve("fit_input", "Horizontal cut I(qy)", x[order], y[order], s[order], n[order], "qy (Å⁻¹)")


def test_the_fit_input_carries_the_halves_into_fitting_and_keeps_both_halves(tmp_path: Path) -> None:
    from src.gimap.features.analyze.infrastructure.adapters import CsvCurveWriter
    from src.gimap.features.fitting.application import LoadCurveRequest
    from src.gimap.features.fitting.infrastructure.adapters.local_files import LocalCurveRepository

    curve = _signed_cut()
    for halves in FIT_HALVES:
        path = CsvCurveWriter().write_xy(
            curve, tmp_path / f"{halves}_fit_input.dat",
            {"source_file": "frame_0001.cbf", "observation": fit_input_observation(_frame(), halves)})
        loaded = LocalCurveRepository().load(LoadCurveRequest(path=path, q_source_unit="angstrom"))
        assert loaded.observation["halves"] == halves  # the key Fitting ▸ In-situ series starts from
        table = np.loadtxt(path)
        assert table.shape == (curve.x.size, 4)  # q, I, sigma, pixels: every point of both halves, whatever is fitted
        assert np.count_nonzero(table[:, 0] < 0) == np.count_nonzero(table[:, 0] > 0) == 60
        np.testing.assert_allclose(table[:, 0], curve.x, rtol=1e-7)


# -- the fit of one frame -----------------------------------------------------------------------------------------


class _Fitter:
    """Stands in for the quick physical fit: records what it was asked, fails where told."""

    def __init__(self, fail=()):
        self.calls, self.fail = [], dict(fail)

    def __call__(self, q, intensity, sigma, *, components=(), distance_nm=None, cancelled=None, halves=None):
        index = len(self.calls)
        self.calls.append({"q": np.asarray(q), "halves": halves, "components": components, "distance": distance_nm})
        if index in self.fail:
            raise self.fail[index]
        radius = 5.0 + index
        return [{
            "combination": "sphere", "best_chi2_weighted": 1.1, "best_log_rmse": 0.1,
            "components": [{"type": "sphere", "weight": 1.0, "params": {"R": radius, "D": 50.0 + index}}],
            "warnings": [], "prepared": f"the {halves} of the cut", "preparation": {
                "halves": halves, "points": 59, "averaged": 59 if halves == "mean" else 0, "one_half": 0, "merged": 1},
        }]


def test_the_fit_of_a_frame_is_asked_for_the_chosen_halves_and_the_row_says_what_was_fitted() -> None:
    from src.gimap.shared.signed_cut import MissingHalf

    curve = _signed_cut()
    for halves in FIT_HALVES:
        fitter = _Fitter()
        rows = fit_model_curve(curve, fitter, "sphere", halves=halves)
        assert fitter.calls[0]["halves"] == halves
        np.testing.assert_array_equal(fitter.calls[0]["q"], curve.x)  # the whole signed cut: the fit prepares it
        row = model_row("frame 1", rows, halves=halves)
        assert list(row)[:3] == ["frame", "halves", "points"]  # right after the frame: what the values are of
        assert row["halves"] == halves and row["points"] == 59 and row["c1 sphere R"] == 5.0
    fitter = _Fitter()
    fit_model_curve(curve, fitter, "sphere")  # nothing said: the mean …
    fit_model_curve(curve, fitter, "sphere", halves="both_abs")  # … and both halves together are never fitted
    assert [call["halves"] for call in fitter.calls] == ["mean", "mean"]

    # a frame without the chosen half: its row says why, and the next frame starts from the last good result
    starts, table = FitStarts("previous"), FitTable("model")
    fitter = _Fitter(fail={1: MissingHalf(MISSING_HALF["positive"])})
    frames = []
    for index in range(3):
        rows, error = [], ""
        try:
            rows = fit_model_curve(curve, fitter, "auto", starts, halves="positive")
        except ValueError as exc:
            error = str(exc)
        table.add(model_row(f"frame {index + 1}", rows, error=error, halves="positive"))
        frames.append(model_fit_frame(f"frame {index + 1}", rows, error=error, halves="positive"))
    assert table.rows[1] == {"frame": "frame 2", "halves": "positive", "points": 0,
                             "note": "No points on the q > 0 half"}
    assert fitter.calls[2]["components"] == (1,) and fitter.calls[2]["distance"] == 50.0  # of frame 1, the last good one
    assert table.columns()[:3] == ["frame", "halves", "points"]
    # a first frame without a fit has only its note: the note is still the last column, after the model's values
    late = FitTable("model")
    late.add(model_row("frame 1", [], error=MISSING_HALF["negative"], halves="negative"))
    late.add(model_row("frame 2", _Fitter()(curve.x, None, None, halves="negative"), halves="negative"))
    assert late.columns()[:5] == ["frame", "halves", "points", "model", "chi2"] and late.columns()[-1] == "note"
    assert table.columns() == late.columns()  # the same table whichever frame fails
    # a fit that does not return the points it took: its curve is the model alone, as before
    assert model_fit_curve("frame 1", _Fitter()(curve.x, None, None, halves="mean"))[1] == ["q (1/nm)", "fit"]
    assert [frame["state"] for frame in frames] == ["ok", "failed", "ok"]
    assert frames[0] == {"frame": "frame 1", "state": "ok", "halves": "positive", "points": 59,
                         "prepared": "the positive of the cut", "averaged": 0, "one_half": 0, "merged": 1}
    assert frames[1] == {"frame": "frame 2", "state": "failed", "reason": "No points on the q > 0 half",
                         "halves": "positive", "points": 0, "prepared": ""}


def test_the_real_fit_takes_a_cut_of_any_length_on_each_of_the_three_halves() -> None:
    """More than 1000 points on each half (a Lambda 9M cut): every choice gives a solution; a half that is not
    there is the reason of that frame, not another half fitted in its place."""
    from src.gimap.features.fitting.bootstrap import create_quick_fit

    from src.gimap.features.analyze.application import HALVES_WORDS

    quick_fit = create_quick_fit()
    curve = _signed_cut(1260, scale=0.6)
    assert np.count_nonzero(curve.x > 0) == np.count_nonzero(curve.x < 0) == 1260
    radius = {}
    for halves in FIT_HALVES:
        rows = fit_model_curve(curve, quick_fit, "sphere", halves=halves)
        row = model_row("frame 1", rows, halves=halves)
        assert row["halves"] == halves and 8 <= row["points"] <= 1000, row
        entry = model_fit_frame("frame 1", rows, halves=halves)
        assert entry["state"] == "ok" and "neighbouring points merged" in entry["prepared"] and entry["merged"] == 2
        # the fitted curve of the frame: the points that were fitted (one curve on |q|) and the model at them
        comments, header, body = model_fit_curve("frame 1", rows, halves=halves)
        assert header == ["q (1/nm)", "I", "sigma", "fit"] and len(body) == row["points"]
        q_fitted = np.array([line[0] for line in body])
        assert q_fitted.min() > 0 and np.all(np.diff(q_fitted) > 0) and q_fitted.max() == pytest.approx(2.0, rel=0.01)
        assert f"the points fitted ({HALVES_WORDS[halves]}, {row['points']} points)" in comments[0]
        assert comments[1] == f"points: {entry['prepared']}"
        radius[halves] = row["c1 sphere R"]
        assert radius[halves] == pytest.approx(4.0, abs=0.3), (halves, row)
    one_sided = _signed_cut(1260, positive=False)
    with pytest.raises(ValueError, match="No points on the q > 0 half"):
        fit_model_curve(one_sided, quick_fit, "sphere", halves="positive")
    rows = fit_model_curve(one_sided, quick_fit, "sphere", halves="mean")  # the mean of a cut with one half is that half
    assert model_row("frame 1", rows, halves="mean")["halves"] == "negative"


def _with_intensity(curve: Curve, intensity) -> Curve:
    return Curve(curve.key, curve.title, curve.x, np.asarray(intensity, dtype=float), curve.sigma, curve.pixels, curve.x_label)


def test_a_frame_without_counts_is_not_fitted_and_the_reason_names_the_frame_not_the_algorithm() -> None:
    """A closed shutter: the cut has its points and every one of them is zero. The fit's own refusal — “at least
    one positive intensity is required” — is the wording of an algorithm; the batch says what is wrong with the
    frame, in the table, the record and on the page alike, and does not hand such a cut to the fit."""
    from src.gimap.features.analyze.application import NO_INTENSITY
    from src.gimap.features.fitting.bootstrap import create_quick_fit

    assert NO_INTENSITY == {"mean": "The horizontal cut has no intensity above zero",
                            "positive": "No intensity above zero on the q > 0 half",
                            "negative": "No intensity above zero on the q < 0 half"}
    curve = _signed_cut()
    blank, fitter = _with_intensity(curve, np.zeros(curve.x.size)), _Fitter()
    for halves in (*FIT_HALVES, "both_abs"):
        with pytest.raises(ValueError) as refused:
            fit_model_curve(blank, fitter, "sphere", halves=halves)
        # the frame is blank, not only the half that was chosen: the other half would not help, and it is not offered
        assert str(refused.value) == NO_INTENSITY["mean"], halves
    assert fitter.calls == []
    row = model_row("frame 3", [], error=NO_INTENSITY["mean"], halves="negative")
    assert row == {"frame": "frame 3", "halves": "negative", "points": 0,
                   "note": "The horizontal cut has no intensity above zero"}

    # counts on one half only: that half and the mean are fitted; the half without counts is the reason, and the
    # fit is not asked. A half that is not there at all stays the fit's own reason (no points are not no counts).
    one_lit = _with_intensity(curve, np.where(curve.x > 0, curve.intensity, 0.0))
    for halves in ("positive", "mean"):
        assert fit_model_curve(one_lit, fitter, "sphere", halves=halves)
    with pytest.raises(ValueError) as refused:
        fit_model_curve(one_lit, fitter, "sphere", halves="negative")
    assert str(refused.value) == NO_INTENSITY["negative"]
    assert [call["halves"] for call in fitter.calls] == ["positive", "mean"]
    below = _with_intensity(curve, np.where(curve.x > 0, curve.intensity, -0.5))  # under a subtracted background
    with pytest.raises(ValueError) as refused:
        fit_model_curve(below, fitter, "sphere", halves="negative")
    assert str(refused.value) == NO_INTENSITY["negative"] and len(fitter.calls) == 2
    quick_fit = create_quick_fit()
    with pytest.raises(ValueError, match="No points on the q > 0 half"):
        fit_model_curve(_signed_cut(positive=False), quick_fit, "sphere", halves="positive")

    # what the real fit says when it is asked all the same: the sentences the page has to know (``SERIES_ZH``)
    twice = np.sort(np.concatenate([curve.x, curve.x[curve.x > 0][:3]]))
    doubled = Curve("fit_input", curve.title, twice, np.interp(twice, curve.x, curve.intensity),
                    np.interp(twice, curve.x, curve.sigma), np.full(twice.size, 6.0), curve.x_label)
    for cut, halves, sentence in (
        (blank, "mean", "at least one positive intensity is required"),
        (blank, "negative", "at least one positive intensity is required"),
        (doubled, "positive", "duplicate q values; combine replicates with their uncertainties first"),
    ):
        with pytest.raises(ValueError) as refused:
            quick_fit(cut.x, cut.intensity, cut.sigma, components=(1,), halves=halves)
        assert str(refused.value) == sentence, (halves, str(refused.value))


def test_the_readme_and_the_words_of_the_table_name_the_halves() -> None:
    from src.gimap.features.analyze.application import HALVES_WORDS

    assert HALVES_WORDS == {"mean": "mean of both halves", "positive": "q > 0 half", "negative": "q < 0 half"}
    choices = BatchChoices(tables=False, fit_input=True, fit="model", fit_halves="negative")
    text = readme_text(choices, batch="run", frame="f", frames=3, curves=[], halves="negative")
    line = next(line for line in text.splitlines() if line.startswith("Halves"))
    assert "a fit takes the q < 0 half" in line and "both halves" in line and "signed" in line and line.isascii()
    assert "run_model_fits.csv" in line  # where every frame's halves and points are
    assert "Halves" not in readme_text(choices, batch="run", frame="f", frames=3, curves=[])  # GIWAXS: no halves

    def halves_line(**chosen) -> str:
        text = readme_text(BatchChoices(tables=False, fit_halves="positive", **chosen), batch="run", frame="f",
                           frames=3, curves=[], halves="positive")
        return next((line for line in text.splitlines() if line.startswith("Halves")), "")

    # only the model fitted: no fit-input file is written, so the README does not describe any
    only_model = halves_line(fit="model")
    assert "a fit takes the q > 0 half" in only_model and "run_model_fits.csv" in only_model
    assert "fit-input" not in only_model and "observation" not in only_model and only_model.isascii()
    # only the fit inputs written: they say the halves; there is no table of fits to point at
    only_inputs = halves_line(fit_input=True)
    assert "a fit takes the q > 0 half" in only_inputs and "fit-input files hold both halves, with signed qy" in only_inputs
    assert '"halves" of their "# observation:" line' in only_inputs and "model_fits" not in only_inputs
    assert halves_line(per_frame=True) == ""  # neither: the choice changed nothing, nothing is said


# -- the dialog ---------------------------------------------------------------------------------------------------


def _dialog(*, gisaxs: bool = True, for_fitting: bool = False, choices=None, halves=None, try_fit=None,
            frame_stem: str = "frame_0001"):
    from src.gimap.features.analyze.domain import SeriesCorrection
    from src.gimap.features.analyze.presentation.batch_dialog import BatchExportDialog

    _app()
    return BatchExportDialog(
        frames=3, files=3, stem="run", settings_text="GISAXS · profile “P03”", choices=choices or BatchChoices(),
        destination=Path("."), subfolder=True, series=SeriesCorrection(),
        curve_options=[("horizontal", "Horizontal cut I(qy)"), ("vertical", "Vertical cut I(qz)")],
        gisaxs=gisaxs, for_fitting=for_fitting, model_available=True, try_fit=try_fit, frame_stem=frame_stem,
        halves=halves,
    )


def _close(*dialogs) -> None:
    from PyQt5.QtWidgets import QDialog

    for dialog in dialogs:
        dialog.done(QDialog.Rejected)
        dialog.deleteLater()


def test_the_dialog_has_one_combo_with_three_halves_for_gisaxs_and_none_for_giwaxs() -> None:
    from src.gimap.features.analyze.presentation.batch_texts import (
        HALVES_FITTED,
        HALVES_KEPT,
        HALVES_NOTES,
        HALVES_UNUSED,
        MODEL_LINES,
    )

    dialog = _dialog()
    combo = dialog.halves_combo
    assert [combo.itemData(index) for index in range(combo.count())] == ["mean", "positive", "negative"]
    assert [combo.itemText(index) for index in range(combo.count())] == ["Mean of both halves", "q > 0 half", "q < 0 half"]
    assert not dialog.halves_group.isHidden() and not combo.isHidden() and dialog.sent_label.isHidden()
    assert dialog.halves() == "mean" == dialog.choices().fit_halves
    # tables only: the choice would change nothing, and the dialog says when it does
    assert not combo.isEnabled() and dialog.halves_note.text() == HALVES_UNUSED
    dialog.output_checks["fit_input"].setChecked(True)
    assert combo.isEnabled() and dialog.halves_note.text() == HALVES_NOTES["mean"]
    assert "mean of both halves" in HALVES_NOTES["mean"] and "keeps both halves, with signed qy" in HALVES_NOTES["mean"]
    dialog.output_checks["fit_input"].setChecked(False)
    dialog.fit_model_radio.setChecked(True)
    assert combo.isEnabled() and dialog.fit_targets_label.text() == MODEL_LINES["mean"]
    for index, halves in enumerate(("mean", "positive", "negative")):
        combo.setCurrentIndex(index)
        assert dialog.choices().fit_halves == halves and dialog.choices().fit == "model"
        assert dialog.fit_targets_label.text() == MODEL_LINES[halves]
        assert "chosen half" not in dialog.fit_targets_label.text()  # it never was a choice before
        # only the model is fitted: no file is written for Fitting, and the note does not speak of one
        assert dialog.halves_note.text() == HALVES_FITTED[halves] and "file" not in dialog.halves_note.text()
        dialog.output_checks["fit_input"].setChecked(True)
        assert dialog.halves_note.text() == HALVES_NOTES[halves] == f"{HALVES_FITTED[halves]} {HALVES_KEPT}"
        dialog.output_checks["fit_input"].setChecked(False)
    assert "q < 0 half" in MODEL_LINES["negative"] and "q > 0 half" in HALVES_NOTES["positive"]

    # where it starts: the halves of the Cuts step (the page gives its choice as it is). “Both halves on |qy|” there
    # is for looking and says nothing about a fit: then the halves of the last batch, like every other choice.
    stored = BatchChoices(fit_halves="positive")
    from_page = _dialog(choices=stored, halves="negative")
    looking = _dialog(choices=stored, halves="both_abs")
    from_choices = _dialog(choices=stored)
    assert (from_page.halves(), looking.halves(), from_choices.halves()) == ("negative", "positive", "positive")
    assert _dialog_halves(BatchChoices(), "both_abs") == "mean" == _dialog_halves(stored, "mean")

    # an I(q) has no halves: the stored choice is neither shown nor changed, whatever the Cuts step of GISAXS holds
    giwaxs = _dialog(gisaxs=False, choices=stored, halves="negative")
    assert giwaxs.halves_group.isHidden() and giwaxs.choices().fit_halves == "positive"
    _close(dialog, from_page, looking, from_choices, giwaxs)


def _dialog_halves(choices: BatchChoices, halves) -> str:
    dialog = _dialog(choices=choices, halves=halves)
    try:
        return dialog.halves()
    finally:
        _close(dialog)


def test_send_series_is_about_the_halves_and_says_what_is_written() -> None:
    from src.gimap.features.analyze.presentation.batch_texts import HALVES_NOTES

    dialog = _dialog(for_fitting=True, halves="negative")
    assert all(group.isHidden() for group in (dialog.data_group, dialog.pictures_group, dialog.frames_group, dialog.fit_group))
    assert not dialog.halves_group.isHidden() and not dialog.sent_label.isHidden()
    said = dialog.sent_label.text()
    assert "fit_input/frame_0001_fit_input.dat" in said and "Fitting ▸ In-situ series" in said and "Å⁻¹" in said
    assert dialog.halves_combo.isEnabled() and dialog.halves() == "negative"
    assert dialog.halves_note.text() == HALVES_NOTES["negative"]
    assert dialog.choices() == BatchChoices(curves=(), tables=False, fit_input=True, fit_halves="negative")
    dialog.halves_combo.setCurrentIndex(dialog.halves_combo.findData("positive"))
    assert dialog.choices().fit_halves == "positive" and dialog.halves_note.text() == HALVES_NOTES["positive"]
    dialog.show()
    _app().processEvents()
    assert dialog.height() < 640  # one group: no mostly empty window
    giwaxs = _dialog(gisaxs=False, for_fitting=True)  # an I(q): what is written is said, there are no halves
    assert not giwaxs.halves_group.isHidden() and not giwaxs.sent_label.isHidden() and giwaxs.halves_combo.isHidden()
    _close(dialog, giwaxs)


def test_try_on_this_frame_fits_the_halves_of_the_dialog_and_says_what_it_fitted() -> None:
    from src.gimap.features.analyze.presentation.bindings.batch_export import fit_summary

    asked = []
    dialog = _dialog(try_fit=lambda choices, show: asked.append(choices))
    dialog.fit_model_radio.setChecked(True)
    dialog.halves_combo.setCurrentIndex(dialog.halves_combo.findData("negative"))
    dialog.fit_try_button.click()
    assert asked[0].fit == "model" and asked[0].fit_halves == "negative"
    rows = _Fitter()(None, None, None, halves="negative")
    text, level = fit_summary(rows, asked[0])
    assert level == "ok" and text.splitlines()[0].startswith("Best: sphere, R = 5")
    assert text.splitlines()[1] == "Fitted: q < 0 half, 59 points"
    rows[0]["preparation"].update(points=630, merged=2, halves="mean")
    assert fit_summary(rows, asked[0])[0].splitlines()[1] == (
        "Fitted: Mean of both halves, 630 points (2 neighbouring points merged into one)")
    _close(dialog)


def test_a_long_frame_name_is_shortened_in_its_middle_and_does_not_widen_the_dialog() -> None:
    """Beamline names: before, every file name ran out of the dialog and every wrapped line with it."""
    stem = "jg_gisaxs_4nm_old_3ml_insitu_ds03_00001_00005_and_a_remark_of_the_night_shift"
    dialog = _dialog(frame_stem=stem)
    dialog.fit_model_radio.setChecked(True)
    dialog.fit_curves_check.setChecked(True)
    dialog.show()
    for _ in range(4):
        _app().processEvents()
    assert dialog.options_scroll.horizontalScrollBar().maximum() == 0
    label = dialog.output_files["fit_input"]
    assert "…" in label.text() and label.text().startswith("→ fit_input/jg_") and label.text().endswith("_fit_input.dat")
    assert label.toolTip() == f"→ fit_input/{stem}_fit_input.dat"
    assert dialog.fit_curves_file.text().endswith("_model_fit.csv") and dialog.fit_curves_file.toolTip().endswith(f"{stem}_model_fit.csv")
    note = dialog.halves_note
    assert note.width() <= dialog.options_scroll.viewport().width()  # the wrapped lines end inside the dialog
    short = _dialog()
    short.show()
    _app().processEvents()
    assert short.output_files["fit_input"].text() == "→ fit_input/frame_0001_fit_input.dat"
    assert short.output_files["fit_input"].toolTip() == ""
    _close(dialog, short)


# -- the page: the halves of the Cuts step, a batch on real frames ---------------------------------------------------


@pytest.fixture
def cbf_page(tmp_path: Path):
    """The three P03 frames of the repository (copies) with the test profile, a GISAXS series."""
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository

    _app()
    folder = tmp_path / "run"
    folder.mkdir()
    for path in CBFS:
        shutil.copy2(path, folder / path.name)
    sent = []
    page = AnalyzePage(create_analyze_view_model(_context(InMemoryInstrumentProfileRepository([_pilatus_profile()]))),
                       send_series_to_fitting=sent.append)
    page.sent = sent
    page.resize(1400, 900)
    page.show()
    page.add_paths([str(folder)])
    _settle(page, lambda: page.view_model.state.analysis is not None and page.view_model.state.analysis.reduction is not None)
    yield page
    page.tasks.wait(30)
    page.dispose()
    page.close()


def _observations(folder: Path) -> list[dict]:
    from src.gimap.features.fitting.infrastructure.adapters.local_files import read_fit_input_extras

    return [read_fit_input_extras(path)[1] for path in sorted(Path(folder).glob("*_fit_input.dat"))]


@requires_data
def test_the_dialog_starts_with_the_halves_of_the_cuts_step_and_hands_them_on(cbf_page, tmp_path: Path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QDialog

    from src.gimap.features.analyze.presentation.batch_dialog import BatchExportDialog

    page = cbf_page
    seen = []
    monkeypatch.setattr(BatchExportDialog, "exec_", lambda dialog: seen.append(
        (dialog.halves(), not dialog.halves_group.isHidden(), not dialog.halves_combo.isHidden())) or QDialog.Rejected)
    assert page.view_model.fit_side == "both_abs"  # the factory setting of the Cuts step: both halves, for looking
    page.batch_export_dialog()
    for side in ("negative", "positive", "mean"):
        page.view_model.set_fit_side(side)
        page.batch_export_dialog()
    assert seen == [("mean", True, True), ("negative", True, True), ("positive", True, True), ("mean", True, True)]

    # Send Series to Fitting: the halves chosen in the dialog are in every fit input; the files keep both halves
    out = tmp_path / "sent"

    def accept(dialog):
        assert dialog.halves() == "negative"  # from the Cuts step
        dialog.halves_combo.setCurrentIndex(dialog.halves_combo.findData("positive"))
        dialog.destination_edit.setText(str(out))
        return QDialog.Accepted

    kept = BatchChoices(per_frame=True, text_format="dat")  # the choices of an earlier batch
    page.view_model.remember_batch(kept, tmp_path / "kept", True)
    page.view_model.set_fit_side("negative")
    monkeypatch.setattr(BatchExportDialog, "exec_", accept)
    page.send_series_dialog()
    _settle(page, lambda: not page.batch_running() and bool(page.sent))
    assert page.sent == [out / "run_fitting" / "fit_input"]  # a folder of its own beside any Batch Export of the data
    # Send Series has one choice, the halves: it is remembered; what the last Batch Export chose stays as it was
    assert page.view_model.batch_preferences() == (replace(kept, fit_halves="positive"), str(tmp_path / "kept"), True)
    observations = _observations(page.sent[0])
    assert len(observations) == 3 and all(record["halves"] == "positive" for record in observations)
    table = np.loadtxt(next(page.sent[0].glob("*_fit_input.dat")))
    assert table[:, 0].min() < 0 < table[:, 0].max()  # whatever is fitted: both halves, signed
    assert page.view_model.fit_side == "negative"  # the dialog's choice is of this export, the Cuts step keeps its own
    record = json.loads((out / "run_fitting" / "run_batch.json").read_text(encoding="utf-8"))
    assert record["choices"]["fit_halves"] == "positive" and "model_fit" not in record
    readme = (out / "run_fitting" / "README.txt").read_text(encoding="utf-8")
    assert "Halves of the horizontal cut: a fit takes the q > 0 half" in readme

    # every listed frame as Export writes it: the halves of the Cuts step
    page.apply_to_all(tmp_path / "all")
    _settle(page, lambda: not page.batch_running() and (tmp_path / "all" / "README.txt").exists())
    assert [record["halves"] for record in _observations(tmp_path / "all" / "fit_input")] == ["negative"] * 3

    # The halves of the last batch are where the dialog starts again when the Cuts step shows both halves on |qy|
    # (its factory setting, for looking): a second Export without any change fits what the first one fitted.
    opened = []
    look = lambda dialog: opened.append(dialog.halves()) or QDialog.Rejected  # noqa: E731
    monkeypatch.setattr(BatchExportDialog, "exec_", look)
    page.view_model.set_fit_side("both_abs")
    page.batch_export_dialog()
    page.send_series_dialog()
    assert opened == ["positive", "positive"]  # as Send Series left them

    def export_negative(dialog):
        dialog.output_checks["fit_input"].setChecked(True)  # the curve for Fitting is saved: the halves are a choice
        dialog.halves_combo.setCurrentIndex(dialog.halves_combo.findData("negative"))
        dialog.destination_edit.setText(str(tmp_path / "first"))
        return QDialog.Accepted

    monkeypatch.setattr(BatchExportDialog, "exec_", export_negative)
    page.batch_export_dialog()
    _settle(page, lambda: not page.batch_running() and (tmp_path / "first" / "run" / "README.txt").exists())
    assert page.view_model.batch_preferences()[0] == replace(kept, fit_input=True, fit_halves="negative")
    opened.clear()
    monkeypatch.setattr(BatchExportDialog, "exec_", look)
    page.batch_export_dialog()
    page.send_series_dialog()
    page.view_model.set_fit_side("mean")  # a choice a fit can take: the Cuts step says which, as before
    page.batch_export_dialog()
    page.view_model.set_fit_side("both_abs")
    page.batch_export_dialog()
    assert opened == ["negative", "negative", "mean", "negative"]
    assert page.view_model.fit_side == "both_abs"  # the dialog never writes into the Cuts step


@requires_data
def test_try_on_this_frame_of_the_page_fits_the_halves_of_the_dialog(cbf_page) -> None:
    """The button of the dialog ends in the view model's ``try_fit``. It has to hand the dialog's halves to the
    fit: a trial on the mean before a batch on one half would check another fit than the one that runs."""
    from src.gimap.shared.signed_cut import MissingHalf

    page = cbf_page
    fitter = _Fitter(fail={4: MissingHalf(MISSING_HALF["positive"])})
    page.set_model_fitter(fitter)
    analysis = page.view_model.state.analysis
    for halves in FIT_HALVES:
        rows = page.view_model.try_fit(analysis, BatchChoices(fit="model", fit_model="sphere", fit_halves=halves))
        assert rows[0]["preparation"]["halves"] == halves
    assert [call["halves"] for call in fitter.calls] == ["mean", "positive", "negative"]
    assert all(call["q"].min() < 0 < call["q"].max() and call["components"] == (1,) for call in fitter.calls)
    assert page.view_model.try_fit(analysis, BatchChoices(fit="model")) and fitter.calls[3]["halves"] == "mean"

    # … and as the button asks for it, through the page: the outcome names the halves that were fitted, and a half
    # without points is said with the way out
    shown = []
    page.try_batch_fit(BatchChoices(fit="model", fit_halves="positive"), lambda text, level: shown.append((level, text)))
    _settle(page, lambda: len(shown) == 2)
    page.try_batch_fit(BatchChoices(fit="model", fit_halves="negative"), lambda text, level: shown.append((level, text)))
    _settle(page, lambda: len(shown) == 4)
    assert [call["halves"] for call in fitter.calls[4:]] == ["positive", "negative"]
    assert [level for level, _text in shown] == ["info", "warning", "info", "ok"]
    assert shown[1][1] == ("No points on the q > 0 half in this frame: choose the mean of both halves or the other half "
                           "(Halves of the cut, above).")
    assert shown[3][1].splitlines()[1] == "Fitted: q < 0 half, 59 points"


def _batch(page, target: Path, choices: BatchChoices) -> dict:
    page.run_batch(target, choices, stem="run")
    _settle(page, lambda: not page.batch_running() and (target / "README.txt").exists(), timeout_s=120.0)
    return json.loads((target / "run_batch.json").read_text(encoding="utf-8"))


@requires_data
def test_a_batch_whose_model_fits_failed_does_not_end_as_a_success(cbf_page, tmp_path: Path, monkeypatch) -> None:
    from src.gimap.app.presentation.i18n import apply_language
    from src.gimap.features.analyze.presentation.bindings import batch_finish, batch_run, workspace
    from src.gimap.shared.signed_cut import MissingHalf

    page = cbf_page
    toasts = []
    for module in (batch_finish, batch_run, workspace):
        monkeypatch.setattr(module, "show_toast", lambda parent, text, **options: toasts.append((options.get("level"), text, options)))
    missing = MissingHalf(MISSING_HALF["positive"])
    choices = BatchChoices(tables=False, fit_input=True, fit="model", fit_model="sphere", fit_halves="positive",
                           fit_curves=True)

    # one frame of three has no points on the chosen half: it is not fitted, the others are, and the end says so
    fitter = _Fitter(fail={1: missing})
    page.set_model_fitter(fitter)
    target = tmp_path / "some"
    record = _batch(page, target, choices)
    assert [call["halves"] for call in fitter.calls] == ["positive"] * 3
    assert all(call["q"].min() < 0 < call["q"].max() for call in fitter.calls)  # the whole cut: the fit prepares it
    status = page.status_text()
    assert page.status_level() == "warning" and "Exported 3/3 frames" in status
    assert status.endswith("model fit: 2 of 3 frames fitted — No points on the q > 0 half")
    assert [toast[0] for toast in toasts] == ["warning"] and toasts[0][2]["action"][0] == "Open Folder"
    assert toasts[0][1] == "Batch Export done; model fit: 2 of 3 frames fitted — No points on the q > 0 half"
    assert "model fit: 2 of 3 frames fitted — No points on the q > 0 half" in page.batch_panel.now_label.text()
    lines = (target / "run_model_fits.csv").read_text(encoding="utf-8").splitlines()
    assert any("halves of the cut: q > 0 half (positive)" in line for line in lines if line.startswith("#"))
    rows = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    assert list(rows[0])[:4] == ["frame", "frame number", "halves", "points"] and len(rows) == 3
    assert [row["frame number"] for row in rows] == ["1", "2", "3"]  # the x of the trend plot
    assert [(row["halves"], row["points"]) for row in rows] == [("positive", "59"), ("positive", "0"), ("positive", "59")]
    assert rows[1]["note"] == "No points on the q > 0 half" and rows[1]["model"] == "" and rows[2]["model"] == "sphere"
    assert record["choices"]["fit_halves"] == "positive" and record["failed"] == []  # every frame was exported
    fits = record["model_fit"]
    assert (fits["halves"], fits["frames"], fits["fitted"]) == ("positive", 3, 2)
    assert [frame["state"] for frame in fits["per_frame"]] == ["ok", "failed", "ok"]
    assert fits["per_frame"][0]["points"] == 59 and fits["per_frame"][0]["prepared"] == "the positive of the cut"
    assert fits["per_frame"][1]["reason"] == "No points on the q > 0 half"
    assert record["fit_failed"] == [{"frame": record["frames"][1], "why": "No points on the q > 0 half"}]
    assert all(observation["halves"] == "positive" for observation in _observations(target / "fit_input"))
    assert "a fit takes the q > 0 half" in (target / "README.txt").read_text(encoding="utf-8")
    assert len(list((target / "fits").glob("*_model_fit.csv"))) == 2  # the fitted curve of each frame that was fitted

    # every frame fails, for two reasons: no success, no “Exported” toast, each reason with its number of frames
    toasts.clear()
    page.set_model_fitter(_Fitter(fail={0: missing, 1: ValueError(BAD_SIGMA), 2: missing}))
    record = _batch(page, tmp_path / "none", choices)
    status = page.status_text()
    assert page.status_level() == "warning" and [toast[0] for toast in toasts] == ["warning"]
    assert "model fit: 0 of 3 frames fitted — 2 frame(s): No points on the q > 0 half; 1 frame(s): Every supplied sigma" in status
    assert record["model_fit"]["fitted"] == 0 and len(record["fit_failed"]) == 3
    assert not (tmp_path / "none" / "run_model_fits.png").exists()  # no value: no picture of values
    # the panel's line about the latest fit is of the last frame, and follows the interface language like the rest
    last = record["frames"][-1]
    assert page.batch_panel.fit_label.text() == f"Latest fit ({last}): No points on the q > 0 half"
    try:
        apply_language("zh", [page])
        page.refresh_language()
        assert page.batch_panel.fit_label.text() == f"最新拟合（{last}）：q > 0 一半没有数据点"
        assert page.batch_panel.now_label.text().endswith("1 帧：给出的每个 σ 都必须是有限的正数")
    finally:
        apply_language("en", [page])
        page.refresh_language()
    assert page.batch_panel.fit_label.text() == f"Latest fit ({last}): No points on the q > 0 half"

    # every frame fitted: a success, and it says how many were fitted
    toasts.clear()
    page.set_model_fitter(_Fitter())
    record = _batch(page, tmp_path / "all", choices)
    assert page.status_level() == "ok" and page.status_text().endswith("model fit: 3 of 3 frames fitted")
    assert [toast[0] for toast in toasts] == ["ok"] and record["fit_failed"] == []

    # stopped before its end, with a frame that was not fitted: the notice — the loudest message of the end — does
    # not say that the batch is done
    toasts.clear()
    page.set_model_fitter(_Fitter(fail={0: missing}))
    target = tmp_path / "stopped"
    page.run_batch(target, replace(choices, speed="gentle"), stem="run")
    page.cancel_batch()  # the first frame is being reduced and finishes; the two not started are dropped
    _settle(page, lambda: not page.batch_running() and (target / "README.txt").exists(), timeout_s=120.0)
    assert page.batch_panel.title_label.text() == "Batch Export stopped" and page.status_level() == "warning"
    assert [toast[:2] for toast in toasts] == [  # … and says how far it got
        ("warning", "Batch Export stopped: 1 of 3 frames exported; model fit: 0 of 1 frames fitted — No points on the q > 0 half")]


@requires_data
def test_the_single_send_to_fitting_writes_the_halves_of_the_cuts_step(cbf_page) -> None:
    """Send to Fitting of the frame on screen, and the hand-over of a watched folder: the fit input says the halves
    of the Cuts step, so a folder of such files opens in Fitting ▸ In-situ series on them. “Both halves on |qy|” is
    for looking: nothing is written and Fitting decides. The file keeps both halves, signed."""
    from src.gimap.features.fitting.infrastructure.adapters.local_files import read_fit_input_extras

    page = cbf_page
    for side, written in (("negative", "negative"), ("mean", "mean"), ("positive", "positive"), ("both_abs", None)):
        page.view_model.set_fit_side(side)
        path = page.view_model.write_fit_input()
        assert read_fit_input_extras(path)[1].get("halves") == written, side
        table = np.loadtxt(path)
        assert table[:, 0].min() < 0 < table[:, 0].max()
    # Export of the frame (Ctrl+E) writes the same file: it says the same halves
    page.view_model.set_fit_side("negative")
    exported = next(path for path in page.view_model.export() if path.name.endswith("_fit_input.dat"))
    assert read_fit_input_extras(exported)[1].get("halves") == "negative"
    # … and so does the export of every frame a watched folder hands over
    page.view_model.set_fit_side("positive")
    request = page.view_model.batch_requests()[1]
    watched = page.view_model.analyze_and_export(request, page.view_model.default_export_dir() / "watched")
    handed = next(path for path in watched if path.name.endswith("_fit_input.dat"))
    assert read_fit_input_extras(handed)[1].get("halves") == "positive"


# -- the words ------------------------------------------------------------------------------------------------------


def test_every_new_sentence_is_said_in_chinese_and_the_reasons_are_recognised() -> None:
    from src.gimap.app.presentation.i18n import ZH, apply_language
    from src.gimap.features.analyze.application import NO_CURVE, NO_FITTER, NO_INTENSITY, NO_SOLUTION
    from src.gimap.features.analyze.presentation import batch_texts
    from src.gimap.features.analyze.presentation.views.batch_export_view import HALVES
    from src.gimap.features.fitting.application.single_fit import TOO_FEW_FOR_QUICK_FIT

    assert set(batch_texts.QUICK_FIT_REASONS) == set(TOO_FEW_FOR_QUICK_FIT.values())  # written out: kept alike
    assert dict(HALVES) == batch_texts.HALVES_TEXTS
    dialog = _dialog()
    said = [
        *batch_texts.HALVES_TEXTS.values(), *batch_texts.HALVES_NOTES.values(), *batch_texts.HALVES_FITTED.values(),
        batch_texts.HALVES_UNUSED,
        *batch_texts.MODEL_LINES.values(), batch_texts.SENT_TO_FITTING, *batch_texts.QUICK_FIT_REASONS,
        dialog.halves_group.title(), dialog.halves_caption.text(), dialog.halves_combo.toolTip(),
        "Fitted: {halves}, {points} points", "Fitted: {halves}, {points} points ({merged} neighbouring points merged into one)",
        "Fitted: {points} points", "model fit: {fitted} of {frames} frames fitted",
        "model fit: {fitted} of {frames} frames fitted — {why}", "{n} frame(s): {reason}",
        "{n} frame(s) on one half alone", "{first}; {second}", "Batch Export done", "Batch Export stopped",
        batch_texts.TRY_OTHER_HALVES,
        *MISSING_HALF.values(), *NO_INTENSITY.values(), BAD_SIGMA, NO_CURVE, NO_FITTER, NO_SOLUTION,
        # what the quick fit itself refuses a curve with (``prepare_sides`` of Fitting, its “positive: ” taken off):
        # a batch does not let a frame without counts get there, but a reason shown on the page is never English
        "at least one positive intensity is required", "duplicate q values; combine replicates with their uncertainties first",
        "q and intensity must have equal lengths and contain finite values", "No nonzero q points in the selected side",
        "The horizontal cut has no measured points.",
    ]
    # a trial whose chosen half has no points, or no counts, says the way out; any other reason is said as it is
    assert batch_texts.try_failed_text(MISSING_HALF["positive"]) == (
        "No points on the q > 0 half in this frame: choose the mean of both halves or the other half (Halves of the cut, above).")
    assert batch_texts.try_failed_text(NO_INTENSITY["negative"]) == (
        "No intensity above zero on the q < 0 half in this frame: choose the mean of both halves or the other half "
        "(Halves of the cut, above).")
    assert batch_texts.try_failed_text(NO_INTENSITY["mean"]) == NO_INTENSITY["mean"]  # no other half would help
    assert batch_texts.try_failed_text(BAD_SIGMA) == BAD_SIGMA
    _close(dialog)
    assert [text for text in said if text not in ZH] == []
    # under the mean a frame whose cut has one half is fitted on that half: no failure, but said at the end
    one_half = SimpleNamespace(choices=BatchChoices(fit_halves="mean"), fit_frames=[
        {"frame": "a", "state": "ok", "halves": "mean"}, {"frame": "b", "state": "ok", "halves": "negative"}])
    assert batch_texts.fit_end_text(one_half) == "model fit: 2 of 2 frames fitted — 1 frame(s) on one half alone"
    assert batch_texts.fits_failed(one_half) == 0
    chosen = SimpleNamespace(choices=BatchChoices(fit_halves="negative"), fit_frames=one_half.fit_frames[1:])
    assert batch_texts.fit_end_text(chosen) == "model fit: 1 of 1 frames fitted"  # the half that was asked for
    run = SimpleNamespace(fit_frames=[
        {"frame": "a", "state": "ok"}, {"frame": "b", "state": "failed", "reason": MISSING_HALF["negative"]},
        {"frame": "c", "state": "failed", "reason": "Only 3 points on the q > 0 half: this fit needs at least 8"},
    ])
    assert batch_texts.fits_failed(run) == 2 and batch_texts.fit_end_text(SimpleNamespace(fit_frames=[])) == ""
    try:
        apply_language("zh")
        assert batch_texts.fit_reason_text("Only 3 points on the q > 0 half: this fit needs at least 8") == (
            "q > 0 一半只有 3 个数据点：这种拟合至少需要 8 个")
        assert batch_texts.fit_reason_text(MISSING_HALF["positive"]) == "q > 0 一半没有数据点"
        assert batch_texts.try_failed_text(MISSING_HALF["negative"]).startswith("这一帧 q < 0 一半没有数据点：请选择")
        assert batch_texts.fit_reason_text(BAD_SIGMA) == "给出的每个 σ 都必须是有限的正数"
        assert batch_texts.fit_reason_text("sphere, χ² 1.1") == "sphere, χ² 1.1"  # no reason: as it came
        # a frame without counts: the batch's own reason and, were the fit asked all the same, the fit's
        blank = SimpleNamespace(fit_frames=[{"frame": "c", "state": "failed", "reason": NO_INTENSITY["mean"]}])
        assert batch_texts.fit_end_text(blank) == "模型拟合：1 帧中拟合了 0 帧 —— 水平切线上没有大于零的强度"
        assert batch_texts.fit_reason_text(NO_INTENSITY["positive"]) == "q > 0 一半没有大于零的强度"
        assert batch_texts.try_failed_text(NO_INTENSITY["positive"]).startswith("这一帧 q > 0 一半没有大于零的强度：请选择")
        assert batch_texts.fit_reason_text("at least one positive intensity is required") == "至少要有一个大于零的强度"
        assert batch_texts.halves_text("negative") == "q < 0 一半"
        end = batch_texts.fit_end_text(run)
        assert end.startswith("模型拟合：3 帧中拟合了 1 帧") and "q < 0 一半没有数据点" in end and "只有 3 个数据点" in end
        assert "；" in end and "; " not in end  # the clauses are joined in the interface language too
        assert batch_texts.fit_toast_text(run).startswith("批量导出完成；模型拟合：")
        assert batch_texts.fit_toast_text(run, stopped=True).startswith("批量导出已停止；模型拟合：")
        assert batch_texts.with_fits("已导出", run).startswith("已导出；模型拟合：")
    finally:
        apply_language("en")
    assert batch_texts.fit_end_text(run).startswith("model fit: 1 of 3 frames fitted — 1 frame(s): ")
    assert batch_texts.fit_toast_text(run).startswith("Batch Export done; model fit: 1 of 3 frames fitted — ")
    # a batch that was stopped is not “done”: the notice says which
    assert batch_texts.fit_toast_text(run, stopped=True).startswith("Batch Export stopped; model fit: 1 of 3 frames fitted — ")
