"""Continuity acts on per-frame Predict solutions. The intervals continue to describe one mode."""

from dataclasses import replace
from itertools import product
import json
import math
import time

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from src.gimap.features.fitting.application.fit_state import (
    frame_from_dict, frame_to_dict, settings_from_dict, settings_to_dict,
)
from src.gimap.features.fitting.application.series_chain import chained_fit, keep_pass_attempts
from src.gimap.features.fitting.application.series_fit import FrameFit, SeriesSettings, fit_frame
from src.gimap.features.fitting.application.series_selection import FramePredictor
from src.gimap.features.fitting.application.single_predict import Predicted
from src.gimap.features.fitting.domain.series_continuity import composition_penalty, smooth_modes, solution_distance
from tests.test_fit_series import _model, _open_series, _pages, _series, _wait
from tests.test_series_chain import _curve


def mode(radius, probability, spread=0.15):
    model = _model(radius).with_values({(0, "sigma_R"): spread})
    return Predicted(model, 1.0, "Sphere", probability,
                     {"R_1": (radius, radius + 0.2, radius, radius + 0.4, radius - 0.1, radius + 0.5, 0.2)})


def test_joint_path_and_marginals_match_direct_summation():
    # The most probable mode of each individual frame jumps. The second mode is continuous.
    frames = [[mode(r, 0.55), mode(4.0 + 0.02*i, 0.45)] for i, r in enumerate((6.0, 2.0, 8.0, 2.0))]
    path, marginals = smooth_modes(frames, 0.3)
    assert path == [1, 1, 1, 1]
    assert np.asarray(marginals).sum(axis=1) == pytest.approx(np.ones(4))
    weights = {}
    for states in product(range(2), repeat=4):
        selected = [frame[i] for frame, i in zip(frames, states)]
        weight = math.prod(s.probability for s in selected)
        for a, b in zip(selected, selected[1:]):
            d = math.log(a.model.components[0].value("R") / b.model.components[0].value("R"))
            weight *= math.exp(-0.5 * (d / 0.3)**2)
        weights[states] = weight
    assert tuple(path) == max(weights, key=weights.get)
    for t, marg in enumerate(marginals):
        for i in (0, 1):
            exact = sum(weight for states, weight in weights.items() if states[t] == i) / sum(weights.values())
            assert marg[i] == pytest.approx(exact)
    reverse, reverse_marginals = smooth_modes(frames[::-1], 0.3)
    assert reverse[::-1] == path
    assert np.asarray(reverse_marginals[::-1]) == pytest.approx(np.asarray(marginals))


def test_flat_prior_recovers_independent_evidences_including_composition_changes():
    first, second = mode(4.0, 0.7), mode(4.0, 0.8)
    second = replace(second, model=replace(second.model, components=(replace(second.model.components[0], structure=False),)))
    frames = [[first, replace(second, probability=0.3)], [second, replace(first, probability=0.2)]]
    assert composition_penalty(0.1) == pytest.approx(0.01)
    assert composition_penalty(math.inf) == 1.0
    path, marginal = smooth_modes(frames, 1e12)
    assert path == [0, 0]
    assert np.asarray(marginal) == pytest.approx(np.array([[0.7, 0.3], [0.8, 0.2]]))
    # Changing composition once has exactly the preset odds penalty at default scale.
    _, marginal = smooth_modes([[first], [second, replace(first, probability=0.2)]], 0.1)
    assert marginal[1][0] / marginal[1][1] == pytest.approx(0.01 * 0.8 / 0.2)
    with pytest.raises(ValueError):
        smooth_modes(frames, 0)


def test_closest_uses_log_relative_spread_and_inclusive_evidence_threshold():
    previous = fit_frame(5, "previous", _curve(4.0), _model(4.0), SeriesSettings())
    previous = replace(previous, result=replace(previous.result, model=_model(4.0)))
    far, near, outside = mode(4.0, 0.8, 0.5), mode(4.1, 0.2 * 0.8), mode(4.0, 0.01)
    predictor = FramePredictor(lambda _data, _start: [outside, near, far], previous)
    assert predictor(None, None) is near.model
    choice = predictor.choice
    assert choice.rule == "closest" and choice.reference_frame == 6 and choice.rank == 2
    assert (choice.probability, choice.top_probability) == (near.probability, far.probability)
    assert solution_distance(far.model, previous.result.model) == pytest.approx(abs(math.log(0.5 / 0.15)))
    assert choice.interval((0, "R")) == near.intervals["R_1"]
    assert choice.interval(("globals", "background")) == ()
    independent = FramePredictor(lambda _data, _start: [near, far])
    assert independent(None, None) is far.model
    assert independent.choice.rule == "most probable"


def test_unselected_and_failed_chain_attempts_survive_backward_pass_and_json(monkeypatch):
    from src.gimap.features.fitting.application import series_chain

    settings = SeriesSettings(start="chain")
    previous = fit_frame(0, "previous", _curve(4), _model(4.0), settings)
    # Use an explicit stage break and equal chi-squared; the previous-result fit wins the tie.
    def fit(index, path, curve, start, settings, stop, predict=None):
        if predict:
            try:
                predict(None, start)
            except RuntimeError as exc:
                return FrameFit(index, path, error=str(exc), points=10)
        return replace(previous, index=index, path=path)

    monkeypatch.setattr(series_chain, "fit_frame", fit)
    modes = [mode(7.0, 0.8), mode(4.0, 0.2)]
    def run(solutions, direction):
        return chained_fit(1, "sample", None, previous.result.model, _model(4.0), settings, previous,
                           new_stage=True, solutions=solutions, fit_pass=direction)

    forward = run(lambda _d, _m: modes, "forward")
    backward = run(lambda _d, _m: [], "backward")
    assert forward.selection is None
    a, = forward.predict_attempts
    assert not a.kept and a.choice.rank == 2 and a.choice.reference_frame == 1
    merged = keep_pass_attempts(forward, forward, backward)
    a, b = merged.predict_attempts
    assert b.choice is None and b.error and b.fit_pass == "backward"
    restored = frame_from_dict(json.loads(json.dumps(frame_to_dict(merged), allow_nan=False)))
    assert restored.predict_attempts == (a, b)
    assert frame_from_dict({"index": 0, "path": "older"}).selection is None
    assert settings_from_dict({}).smooth is False
    smooth = SeriesSettings(method="predict", smooth=True, smooth_scale=0.25)
    assert settings_from_dict(settings_to_dict(smooth)) == smooth


def test_series_page_applies_joint_choice_after_predicting_each_frame_once(tmp_path, monkeypatch):
    single, series = _pages()
    try:
        folder = _series(tmp_path, (4.0, 4.0, 4.0, 4.0))
        single.set_model(_model(4.0))
        assert _open_series(series, folder)
        series.set_series_method("predict")
        series.smooth_check.setChecked(True)
        called = []
        def predict(_data, _model):
            jump = (7.0, 2.0, 8.0, 2.0)[len(called)]
            called.append(jump)
            recipe = {"operation": "predict", "reference": {}, "frame": len(called)}
            return [replace(mode(4.0, 0.45), reproduction=recipe, solution_index=0),
                    replace(mode(jump, 0.55), reproduction=recipe, solution_index=1)]
        monkeypatch.setattr(series, "predict_solutions", lambda: predict)
        series.start()
        _wait(series, 40)
        assert not series.running and len(called) == 4 and len(series.fits) == 4
        for frame in series.fits.values():
            assert frame.fitted and frame.selection.rank == 2 and frame.selection.rule == "HMM"
            assert frame.reproduction["start_solution_index"] == 0
            assert frame.reproduction["prediction"]["frame"] == frame.index + 1
            assert sum(frame.selection.smoothed_probabilities) == pytest.approx(1.0)
            assert frame.selection.interval((0, "R"))[1] == 4.2
            jump, continuous = frame.selection.ranked_modes
            assert jump["populations"][0]["R_nm"] == called[frame.index]
            assert continuous["populations"] == [{"family": "sphere", "structure": True, "R_nm": 4.0,
                                                  "D_nm": 40.0, "sigma_R_over_R": 0.15}]
            restored = frame_from_dict(json.loads(json.dumps(frame_to_dict(frame))))
            assert restored.selection == frame.selection and restored.reproduction == frame.reproduction
    finally:
        series.dispose()
        single.dispose()


@pytest.mark.parametrize("fail", [False, True])
def test_watch_updates_saved_provisional_result_or_discloses_smoothing_failure(tmp_path, monkeypatch, fail):
    from src.gimap.features.fitting.presentation.single import series_predict

    single, series = _pages()
    try:
        single.set_model(_model(4.0))
        assert _open_series(series, _series(tmp_path, (4.0, 4.0)))
        series.set_series_method("predict")
        series.smooth_check.setChecked(True)
        series.watch_check.setChecked(True)
        jumps = iter((7., 2.))
        monkeypatch.setattr(series, "predict_solutions", lambda: lambda *_: [mode(next(jumps), .6), mode(4., .4)])
        if fail:
            def failed(*args):
                raise ValueError("synthetic error")
            monkeypatch.setattr(series_predict, "smoothed_fits", failed)

        frame_done = series._frame_done
        exported = []
        def save_provisional(frame, curve):
            frame_done(frame, curve)
            if len(series.fits) == 2:
                exported.append(series.save_table(str(tmp_path / "provisional.csv")))
        monkeypatch.setattr(series, "_frame_done", save_provisional)
        series.start()
        end = time.monotonic() + 30
        while time.monotonic() < end:
            QApplication.processEvents()
            if (exported and (series._smoothing_error or all(f.selection.rule == "HMM" for f in series.fits.values()))
                    and series._waiting_for_curves()):
                break
            time.sleep(0.01)
        assert exported and series._waiting_for_curves()
        if fail:
            assert "synthetic error" in series.status_label.text()
            assert all(f.selection.rule == "most probable" for f in series.fits.values())
        else:
            assert series._unsaved and all(f.selection.rank == 2 for f in series.fits.values())
        series.watch_check.setChecked(False)
        _wait(series)
        assert not series.running
        if fail:
            assert "synthetic error" in series.status_label.text()
    finally:
        series.dispose()
        single.dispose()
