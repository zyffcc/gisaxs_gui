"""The halves of Fitting ▸ In-situ series: its own three-way choice in the Curves step, where its value
comes from (the person, the fit inputs Analyze wrote, Single analysis, the mean), what changing it does, a
frame without the chosen half, and the project."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication, QMessageBox

from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language
from src.gimap.features.fitting.application.single_fit import Curve, prepare_curve
from tests.test_fit_series import RADII, _answer, _cut, _model, _pages, _wait

FROM_FILES, FROM_SINGLE = "from the fit inputs written by Analyze", "as in Single analysis"
ANALYZE_HEADER = "# GIMaP Analyze fit input (q in 1/A)"


def _fit_input(path, cut, halves=None) -> None:
    """A fit input as Analyze writes it: both halves, signed, in Å⁻¹; the halves it was sent with in the
    ``# observation:`` line (``fit_input_observation(analysis, halves=…)``)."""
    observation = {"file_format": "cbf", "source": "native_detector_columns", "summed_frames": 1}
    if halves is not None:
        observation["halves"] = halves
    lines = [ANALYZE_HEADER, "# source: /synthetic/frame.cbf", "# curve: Horizontal cut I(qy)",
             "# observation: " + json.dumps(observation), "# columns: qy (1/A)  I  sigma  pixels"]
    lines += [f"{q:.8g} {intensity:.8g} {sigma:.8g} 5" for q, intensity, sigma in zip(*cut)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cuts(tmp_path, halves=None, *, name="gimap_analysis", lost=None, negative: float = 1.3, radii=RADII):
    """A folder of cuts through the beam (``lost``: ``{frame: the one half it holds}``)."""
    folder = tmp_path / name / "fit_input"
    folder.mkdir(parents=True)
    for index, radius in enumerate(radii):
        _fit_input(folder / f"cut_{index + 1:05d}_fit_input.dat",
                   _cut(radius, index, only=(lost or {}).get(index), negative=negative), halves)
    return folder


def _stages(series, limit: float = 60.0) -> None:
    """Until the search for stages (which reads every listed curve) is over."""
    end = time.monotonic() + limit
    QApplication.processEvents()
    while (series._stages_timer.isActive() or series.stage_tasks.is_busy()) and time.monotonic() < end:
        series.stage_tasks.wait(0.05)
        QApplication.processEvents()
        time.sleep(0.01)
    QApplication.processEvents()


def _choose(series, side: str) -> None:
    """As the person does: picks an entry of the box."""
    combo = series.side_combo
    combo.setCurrentIndex(combo.findData(side))
    combo.activated.emit(combo.currentIndex())


def _shown(series) -> tuple:
    name, x, y = series.frame_plot.figure_state()["curves"][0]
    return np.asarray(x), np.asarray(y)


def _close(*pages) -> None:
    for page in pages:
        page.dispose()


def test_the_series_has_its_own_choice_of_three_halves_in_the_curves_step(tmp_path) -> None:
    from src.gimap.features.fitting.application.single_fit import SERIES_SIDES

    single, series = _pages()
    combo = series.side_combo
    assert [combo.itemData(index) for index in range(combo.count())] == ["mean", "positive", "negative"]
    assert [combo.itemText(index) for index in range(combo.count())] == [text for _key, text in SERIES_SIDES]
    assert "Both halves on |q|" not in [combo.itemText(index) for index in range(combo.count())]  # never both together
    assert series.step_pages["curves"].isAncestorOf(combo) and series.side_label.text() == "Halves of the cut"
    assert "mean of both halves, or one half" in combo.toolTip()
    assert not combo.isEnabled() and series.halves_note.isHidden()  # no curves yet: nothing to choose for
    folder = _cuts(tmp_path)
    assert series.open_series(folder)
    assert series.series_side() == "mean" and series.halves_source() == "default"  # nothing says otherwise
    assert combo.isEnabled() and combo.currentData() == "mean" and series.halves_note.isHidden()
    assert "Mean of both halves; every point" in series.model_summary.text()
    _close(series, single)


def test_the_halves_come_from_the_fit_inputs_then_single_analysis_then_the_mean(tmp_path) -> None:
    """D3: the person's choice for the series → the halves the first listed fit input was sent with →
    Single analysis' halves unless they are both on |q| → the mean."""
    single, series = _pages()
    sent = _cuts(tmp_path, "negative", name="sent")  # Analyze ▸ Send Series to Fitting with “q < 0 half”
    plain = _cuts(tmp_path, name="plain")  # curves that say nothing of the halves
    single.open_curve(plain / "cut_00001_fit_input.dat", "positive")  # Single analysis shows a cut on its q > 0 half
    single.set_model(_model(5.2))
    assert single.session.side == "positive" and not single.side_combo.isHidden()

    assert series.open_series(sent)
    assert series.series_side() == "negative" and series.halves_source() == "files"  # before Single analysis'
    assert series.side_combo.currentData() == "negative" and series.side_combo.isEnabled()
    assert series.halves_note.text() == FROM_FILES and not series.halves_note.isHidden()
    assert FROM_FILES in series.side_combo.toolTip()
    assert f"q < 0 half ({FROM_FILES}); every point" in series.model_summary.text()  # the Start step says it
    x, y = _shown(series)  # the frame shown is the half that will be fitted
    q, intensity, sigma = np.loadtxt(sent / "cut_00001_fit_input.dat", unpack=True, usecols=(0, 1, 2))
    expected = prepare_curve(Curve.from_arrays(q, intensity, sigma, name="c"), "negative")
    assert np.allclose(x, expected.q) and np.allclose(y, expected.intensity)

    assert series.open_series(plain)
    assert series.series_side() == "positive" and series.halves_source() == "single"
    assert series.halves_note.text() == FROM_SINGLE and f"q > 0 half ({FROM_SINGLE})" in series.model_summary.text()
    single.set_side("negative")  # it follows Single analysis while nothing else says
    series.refresh()
    assert series.series_side() == "negative" and series.side_combo.currentData() == "negative"
    single.set_side("both")  # both halves on |q| is not what a series fits: every frame on the mean
    series.refresh()
    assert series.series_side() == "mean" and series.halves_source() == "default"
    assert series.halves_note.text().startswith("Single analysis fits both halves on |q|")

    # The person's own choice, here: before everything else, and Single analysis keeps its own.
    assert series.open_series(sent)
    _choose(series, "positive")
    assert series.series_side() == "positive" and series.halves_source() == "chosen" and series.halves_note.isHidden()
    assert single.session.side == "both"
    series.pattern_edit.setText("cut_0000[1-4]_fit_input.dat")  # another pattern, the same series: the choice stays
    series._list_curves()
    assert len(series.paths) == 4 and series.series_side() == "positive"
    # A folder opened anew is another series: it is fitted as it was sent, not as the one before was chosen.
    assert series.open_series(sent, "*_fit_input.dat")
    assert series.series_side() == "negative" and series.halves_source() == "files"
    _close(series, single)


def test_the_run_fits_the_halves_of_the_series_and_the_record_says_where_they_came_from(tmp_path) -> None:
    single, series = _pages()
    folder = _cuts(tmp_path, "negative")
    single.open_curve(folder / "cut_00001_fit_input.dat", "both_abs")  # Analyze's display default in Single analysis
    single.set_model(_model(5.2))
    assert single.session.side == "both" and series.open_series(folder)
    series.start()
    assert not series.side_combo.isEnabled()  # not while the series runs
    _wait(series)
    assert series.side_combo.isEnabled() and series._settings.side == "negative"
    assert [frame.halves_used for frame in series.fits.values()] == ["negative"] * 5 and all(f.ok for f in series.fits.values())
    scale = [series.fits[index].result.model.get((0, "Int")).value for index in range(5)]
    assert scale == pytest.approx([2600.0] * 5, rel=0.1)  # the q < 0 half is 1.3 times as intense
    summary = series.results_summary.text()
    assert f"q < 0 half ({FROM_FILES}); every point · 200 points per frame" in summary
    assert any(f"q < 0 half ({FROM_FILES}); every point" in line for line in series.log_view.toPlainText().splitlines())
    assert series.save_table(str(tmp_path / "series.csv"))
    record = json.loads((tmp_path / "series.json").read_text(encoding="utf-8"))
    assert record["settings"]["side"] == "negative" and record["settings"]["side_from"] == "files"
    assert {item["halves_used"] for item in record["per_frame"]} == {"negative"}
    _close(series, single)


def test_changing_the_halves_draws_the_frame_again_finds_the_stages_again_and_marks_a_run(tmp_path) -> None:
    from src.gimap.features.fitting.presentation.single.series_results import STALE

    single, series = _pages()
    folder = _cuts(tmp_path)
    single.open_curve(folder / "cut_00001_fit_input.dat")
    single.set_model(_model(5.2))
    assert series.open_series(folder)
    _stages(series)
    mean = _shown(series)[1]
    key = series._stages_key
    _choose(series, "negative")
    negative = _shown(series)[1]
    assert np.median(negative / mean) == pytest.approx(1.3 / 1.15, rel=0.05)  # the selected frame: the q < 0 half now
    assert series._stages_key != key and series._stages_timer.isActive()  # the stages: found again on these points
    assert single.session.side == "mean"  # Single analysis is not changed from here
    _stages(series)
    series.start()
    _wait(series)
    assert series.step_rail.state("results") == "ok" and series._settings.side == "negative"
    _choose(series, "mean")
    assert series.step_rail.state("results") == "warn" and series.step_rail.detail("results") == STALE == \
        "Changed after the fit: Start again"
    assert STALE in series.results_summary.text()
    assert np.allclose(_shown(series)[1], prepare_curve(series._curves[series._selected], "negative").intensity)  # still the run's
    _choose(series, "negative")
    assert series.step_rail.state("results") == "ok" and STALE not in series.results_summary.text()
    single.set_range((0.5, 1.5))  # so does another fitting range in Single analysis
    series.refresh()
    assert series.step_rail.state("results") == "warn" and series.step_rail.detail("results") == STALE
    _close(series, single)


def test_curves_with_one_side_have_no_halves_to_choose(tmp_path) -> None:
    from tests.test_fit_series import _series

    single, series = _pages()
    folder = _series(tmp_path)  # five curves, q > 0 only
    cut = tmp_path / "cut_fit_input.dat"
    np.savetxt(cut, np.column_stack(_cut(RADII[0], 0, negative=1.3)))
    single.open_curve(cut, "negative")  # Single analysis is on the q < 0 half of a cut
    single.set_model(_model(5.2))
    assert series.open_series(folder)
    assert not series.curves_signed() and series.series_side() == "mean" and series.halves_source() == "one-sided"
    assert not series.side_combo.isEnabled() and series.side_combo.currentData() == "mean"
    assert series.halves_note.text() == "These curves have q on one side only: there are no halves to choose."
    assert not series.halves_note.isHidden() and "one side only" in series.side_combo.toolTip()
    assert series.set_series_side("negative") and series.series_side() == "mean"  # nothing to choose from
    assert series.frame_note.text() == "200 points"  # no word about halves for such a curve
    _stages(series)
    assert not series.side_combo.isEnabled() and series._skipped == {}
    _close(series, single)


def test_before_start_the_page_says_the_halves_that_start_fits(tmp_path, monkeypatch) -> None:
    """The first listed curve has lost a half, or cannot be read: a few more of the listed curves are looked
    at when they are listed — not only at Start — so the box, its note and the Start summary say what the
    run uses (with two curves no stage search ever reads the other one)."""
    _answer(monkeypatch, QMessageBox.Discard)
    single, series = _pages()
    single.set_model(_model(5.2))
    lost = _cuts(tmp_path, "negative", name="lost", lost={0: "positive"}, radii=RADII[:2])  # frame 1 without its q < 0 half
    assert series.open_series(lost)
    assert series.curves_signed() and series.series_side() == "negative" and series.halves_source() == "files"
    assert series.side_combo.isEnabled() and series.side_combo.currentData() == "negative"
    assert series.halves_note.text() == FROM_FILES and not series.halves_note.isHidden()
    assert f"q < 0 half ({FROM_FILES}); every point" in series.model_summary.text()
    assert series.frame_note.text() == "Frame 1 cannot be fitted: No points on the q < 0 half"  # the frame shown
    series.start()
    _wait(series)
    assert series._settings.side == "negative" and series._side_from == "files"
    assert [series.fits[index].ok for index in range(2)] == [False, True]
    assert series.side_combo.currentData() == "negative"  # what the page said is what was fitted

    corrupt = _cuts(tmp_path, "positive", name="corrupt", radii=RADII[:2])
    (corrupt / "cut_00001_fit_input.dat").write_text("not a curve\n", encoding="utf-8")
    assert series.open_series(corrupt)
    assert series.series_side() == "positive" and series.halves_source() == "files"
    assert series.side_combo.isEnabled() and series.halves_note.text() == FROM_FILES
    assert f"q > 0 half ({FROM_FILES}); every point" in series.model_summary.text()
    series.start()
    _wait(series)
    assert series._settings.side == "positive" and [series.fits[index].ok for index in range(2)] == [False, True]
    assert series.fits[1].halves_used == "positive"

    one_sided = tmp_path / "iq"  # curves with one side, the first unreadable: still no halves to choose
    one_sided.mkdir()
    for index, radius in enumerate(RADII[:3]):
        np.savetxt(one_sided / f"iq_{index + 1:05d}_fit_input.dat", np.column_stack(_cut(radius, index, only="positive")))
    (one_sided / "iq_00001_fit_input.dat").write_text("not a curve\n", encoding="utf-8")
    assert series.open_series(one_sided)
    assert not series.curves_signed() and series.halves_source() == "one-sided" and not series.side_combo.isEnabled()
    _close(series, single)


def test_the_search_of_a_folder_listed_before_is_not_taken_for_one_with_as_many_curves(tmp_path) -> None:
    """A search reads every curve in the background and may end after another folder is listed: as many
    files are not the same curves (their stages, and whether they are cuts with two halves)."""
    from tests.test_fit_series import _series

    single, series = _pages()
    cuts = _cuts(tmp_path, "negative", name="cuts")  # five cuts through the beam
    assert series.open_series(cuts)
    _stages(series)
    listed, stages = series._listed(), series._series_stages
    curves = [series._preview_curve(index) for index in listed]
    assert stages is not None and all(curve.signed and curve.path == series.paths[index] for index, curve in zip(listed, curves))
    plain = _series(tmp_path)  # five curves with one side
    assert series.open_series(plain) and series._listed() == listed
    assert not series.curves_signed() and series._series_stages is None
    read = [(index, np.array([-0.1, 0.1]), {"halves": "negative"}) for index in listed]
    series._stages_found(listed, series._stage_settings(), stages, "", curves, {}, read)  # the search of the cuts ends now
    assert series._series_stages is None and series.stage_of_frame(0) is None
    assert not series.curves_signed() and series.series_side() == "mean" and not series.side_combo.isEnabled()
    _stages(series)  # the search of the curves that are listed is taken
    assert series._series_stages is not None and series._series_stages is not stages
    _close(series, single)


@pytest.mark.parametrize("language", ["en", "zh"])
def test_a_frame_without_the_chosen_half_fails_with_its_sentence_and_the_run_goes_on(tmp_path, language) -> None:
    from src.gimap.app.presentation.i18n import ZH

    sentence = "No points on the q > 0 half"
    said = sentence if language == "en" else ZH[sentence]
    single, series = _pages()
    folder = _cuts(tmp_path, "positive", lost={2: "negative", 3: "positive"})  # the third frame has lost its q > 0 half
    single.open_curve(folder / "cut_00001_fit_input.dat")
    single.set_model(_model(5.2))
    try:
        apply_language(language, [series, single])
        series.refresh_language()
        assert series.open_series(folder) and series.series_side() == "positive"
        _stages(series)
        # Before the run: the frame says in the list that it cannot be used, the search goes on without it.
        item = series.frame_list.item(2)
        assert item.text().startswith("✗") and item.text().endswith(said) and said in item.toolTip()
        assert series._skipped == {2: sentence} and series.frame_list.item(1).text().startswith("·")
        assert ("cannot be used: frames 3" if language == "en" else "3") in series.stages_label.text()
        series.frame_list.setCurrentRow(2)
        assert series.frame_plot.curve_count() == 0  # the other half is not drawn in its place
        assert said in series.frame_plot.empty_overlay.label.text() and said in series.frame_note.text()
        if language == "en":
            assert series.frame_note.text() == "Frame 3 cannot be fitted: No points on the q > 0 half"
            series.frame_list.setCurrentRow(3)  # the fourth holds that half alone: fitted, and the half is named
            assert series.frame_note.text() == "q > 0 half · 200 points"
        series.start()
        _wait(series)
        assert [series.fits[index].ok for index in range(5)] == [True, True, False, True, True]  # the run went on
        assert series.fits[2].error == sentence and series.fits[2].state == "failed"  # the record keeps the English
        item = series.frame_list.item(2)
        assert item.text().startswith("✗") and item.text().endswith(said)
        cell = series.results_table.item(2, 1)
        assert cell.text() == f"✗ {said}" and series.results_table.columnSpan(2, 1) > 1
        assert [series.fits[index].result.model.get((0, "R")).value for index in (0, 1, 3, 4)] == pytest.approx(
            [RADII[index] for index in (0, 1, 3, 4)], rel=0.05)
        log = series.log_view.toPlainText()
        assert said in log and "cut_00003_fit_input.dat" in log
        # On the mean a frame with one half is that half — fitted, and named where the frame is shown.
        if language == "en":
            _choose(series, "mean")
            assert series.series_side() == "mean"
    finally:
        apply_language(DEFAULT_LANGUAGE, [series, single])
        _close(series, single)


def test_under_the_mean_a_frame_with_one_half_is_fitted_on_it_and_says_so(tmp_path) -> None:
    single, series = _pages()
    folder = _cuts(tmp_path, lost={2: "negative"})
    single.open_curve(folder / "cut_00001_fit_input.dat")
    single.set_model(_model(5.2))
    assert series.open_series(folder) and series.series_side() == "mean"
    _stages(series)
    assert series._skipped == {}
    series.frame_list.setCurrentRow(2)
    assert series.frame_note.text() == "q < 0 half only · 200 points"
    series.frame_list.setCurrentRow(1)
    assert series.frame_note.text() == "Mean of both halves · 200 points"
    series.start()
    _wait(series)
    assert [frame.halves_used for frame in series.fits.values()] == ["mean", "mean", "negative", "mean", "mean"]
    assert series.frame_list.item(2).text().startswith("✓") and series.frame_list.item(2).text().endswith("q < 0 half only")
    assert series.results_table.item(2, series.results_table.columnCount() - 1).text() == "q < 0"
    assert series.results_table.item(1, series.results_table.columnCount() - 1).text() == "mean"
    _close(series, single)


def test_the_frame_shown_says_what_is_fitted_and_shows_the_points_that_are_not(tmp_path) -> None:
    single, series = _pages()
    folder = _cuts(tmp_path)
    single.open_curve(folder / "cut_00001_fit_input.dat")
    single.set_model(_model(5.2))
    points = single.session.all_points()
    left_out = {float(f"{points.q[index]:.9g}") for index in (60, 61, 62)}
    single.session.excluded = set(left_out)
    single.set_range((0.5, 1.5))
    assert series.open_series(folder)
    series.frame_list.setCurrentRow(3)
    state = series.frame_plot.figure_state()
    names = [name for name, _x, _y in state["curves"]]
    assert names == ["measured", "outside the range", "left out"]  # the fitted points first, the others to be seen
    fitted, outside, crossed = (np.asarray(x) for _name, x, _y in state["curves"])
    assert fitted.min() >= 0.5 and fitted.max() <= 1.5 and crossed.size == 3 and np.allclose(sorted(crossed), sorted(left_out))
    assert (outside < 0.5).any() and (outside > 1.5).any() and fitted.size + outside.size + crossed.size == 200
    inside = int(((points.q >= 0.5) & (points.q <= 1.5)).sum())
    assert series.frame_note.text() == f"Mean of both halves · {inside - 3} points · |q| 0.5–1.5 nm⁻¹ · 3 left out"
    assert series.frame_plot.x_window.isVisible() and not series.frame_plot.x_window.movable  # the range, to look at
    assert tuple(series.frame_plot._window.interval) == (0.5, 1.5)
    single.set_range(None)
    series.refresh()
    assert not series.frame_plot.x_window.isVisible() and "|q|" not in series.frame_note.text()
    _close(series, single)


def test_a_project_keeps_the_halves_of_the_series(tmp_path) -> None:
    single, series = _pages()
    folder = _cuts(tmp_path)
    assert series.open_series(folder)
    _choose(series, "negative")
    state = series.project_state()
    assert state["side"] == "negative" and state["side_from"] == "chosen"
    assert json.loads(json.dumps(state)) == state
    other_single, other = _pages()  # another window: Single analysis holds no curve
    assert other.apply_project_state(state) == []
    assert other.series_side() == "negative" and other.halves_source() == "project"
    assert other.side_combo.currentData() == "negative" and other.halves_note.text() == "as saved in the project"
    assert other_single.session.side == "mean"
    # A project of before this choice existed has no halves: the usual order decides.
    older = {key: value for key, value in state.items() if not key.startswith("side")}
    sent = _cuts(tmp_path, "positive", name="sent")
    older["folder"] = str(sent)
    assert other.apply_project_state(older) == [] and other.series_side() == "positive" and other.halves_source() == "files"
    # A project without a series: the page as it starts, without the halves of the one before.
    _choose(other, "negative")
    other.close_series()
    assert other.series_side() == "mean" and other.halves_source() == "default" and other.paths == []
    _close(series, single, other, other_single)


def test_a_second_send_of_the_series_is_fitted_as_it_was_sent(tmp_path, monkeypatch) -> None:
    """Analyze ▸ Send Series to Fitting again, with other halves, into the same folder."""
    asked = _answer(monkeypatch, QMessageBox.Discard)
    single, series = _pages()
    folder = _cuts(tmp_path, "positive")
    assert series.open_series(folder) and series.series_side() == "positive"
    _choose(series, "mean")
    for index, radius in enumerate(RADII):  # written again, sent with the q < 0 half this time
        _fit_input(folder / f"cut_{index + 1:05d}_fit_input.dat", _cut(radius, index, negative=1.3), "negative")
    assert series.open_series(folder, "*_fit_input.dat")
    assert series.series_side() == "negative" and series.halves_source() == "files" and asked == []
    _close(series, single)
