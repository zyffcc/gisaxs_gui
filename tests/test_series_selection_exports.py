"""The series Save menu should tell the same story in CSV, JSON and the readable report."""

import csv
from dataclasses import replace
import json

from PyQt5.QtWidgets import QFileDialog

from src.gimap.features.fitting.application.fit_state import settings_from_dict, settings_to_dict
from src.gimap.features.fitting.application.series_fit import FrameFit, SeriesSettings, fit_frame, series_record, series_table
from src.gimap.features.fitting.application.series_report import series_report
from src.gimap.features.fitting.application.series_selection import PredictAttempt, SolutionChoice
from tests.test_fit_series import _model, _pages
from tests.test_series_chain import _curve


def test_save_menu_exports_the_path_probability_and_raw_rank(tmp_path, monkeypatch):
    single, series = _pages()
    try:
        series.folder = str(tmp_path)
        model = series._start_model = _model(4)
        settings = series._settings = SeriesSettings(method="predict", smooth=True, smooth_scale=0.2)
        fit = fit_frame(0, "sample.dat", _curve(4), model, SeriesSettings())
        selection = SolutionChoice(2, 0.4, 0.6, "HMM", smoothed_probability=0.85,
                                   probabilities=(0.6, 0.4), smoothed_probabilities=(0.15, 0.85),
                                   intervals={"R_1": (4.0, 4.2, 3.9, 4.4, 3.7, 4.6, 0.25)})
        series.fits = {0: replace(fit, start_from="predict", selection=selection, predict_choice="HMM"),
                       1: FrameFit(1, "missing.dat", error="no data")}
        table = tmp_path / "run.csv"
        assert series.save_table(table) == table
        with table.open(encoding="utf-8", newline="") as stream:
            fitted, failed = csv.DictReader(stream)
        assert [fitted[k] for k in ("predict_rank", "predict_probability", "predict_max_probability",
                                    "predict_rule", "predict_smoothed_probability")] == ["2", "0.4", "0.6", "HMM", "0.85"]
        assert failed["predict_rank"] == failed["predict_probability"] == ""
        record = json.loads(table.with_suffix(".json").read_text(encoding="utf-8"))
        assert record["series_policy"]["selection"]["scale"] == settings.smooth_scale
        assert record["per_frame"][0]["selected_solution"]["smoothed_probabilities"] == [0.15, 0.85]
        assert record["per_frame"][0]["selected_solution"]["intervals"]["R_1"][1:4] == [4.2, 3.9, 4.4]
        report = tmp_path / "report.md"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(report), ""))
        series.save_report_action.trigger()
        text = report.read_text(encoding="utf-8")
        assert "s = 0.2" in text and "| HMM | 2 | 0.4 | 0.6 | 0.85 |" in text
        assert "missing.dat" in text and "no data" in text and "within the selected solution" in text
    finally:
        series.dispose()
        single.dispose()


def test_chain_report_and_csv_keep_discarded_restart_and_failed_backward_attempt():
    model, settings = _model(4), SeriesSettings(start="chain", backward=True)
    choice = SolutionChoice(2, 0.12, 0.88, "closest", reference_frame=3)
    attempts = (PredictAttempt("forward", "a new stage begins", choice, False),
                PredictAttempt("backward", "the fit did not converge", None, False, error="no solutions"))
    frame = replace(fit_frame(3, "next.dat", _curve(4), model, settings), predict_attempts=attempts,
                    start_from="previous", fit_pass="forward")
    header, rows = series_table([frame], model, details=True)
    row = dict(zip(header, rows[0]))
    assert row["predict_rank"] == 2 and row["predict_fit_kept"] is False and row["predict_reference_frame"] == 3
    assert json.loads(row["predict_attempts"])[1]["error"] == "no solutions"
    record = series_record(model, settings, "sample", "*.dat", [frame])
    policy = record["series_policy"]
    assert policy["selection"]["min_relative_probability"] == 0.1
    assert policy["chain_breaks"]["chi2_ratio_above"] == 3 and policy["chain_breaks"]["chi2_reduced_above"] == 2
    assert "selected_solution" not in record["per_frame"][0]
    text = series_report(record)
    assert "χ²ᵣ exceeds both 3" in text and "and 2." in text and "p ≥ 0.1 p_max" in text
    assert "forward | Predict restart discarded | a new stage begins | closest to frame 3 | 2 | 0.12 | 0.88" in text
    assert "backward | Predict restart discarded | the fit did not converge · no solutions" in text


def test_resaving_an_older_project_preserves_the_known_rule_without_assigning_new_probabilities():
    old = settings_from_dict({"method": "predict", "follow": True})
    frame = replace(fit_frame(0, "early.dat", _curve(4), _model(4), SeriesSettings()),
                    start_from="predict", predict_choice="the most probable solution")
    record = series_record(_model(4), settings_from_dict(settings_to_dict(old)), "sample", "*.dat", [frame])
    assert record["series_policy"]["selection"] == {"rule": "unrecorded"}
    assert "selected_solution" not in record["per_frame"][0]
    text = series_report(record)
    assert "predates the record" in text and "the most probable solution" in text
