"""The chained fit of an In-situ series (``application/series_chain``): each frame from the result of the one before,
Predict parameters only where the chain breaks (a stand-in for it here: the method itself is not touched) and never
at a value at a bound, the solution of Predict closest to the frame before among the probable ones, the better of two
fits kept, the flags of a frame, the backward pass order, the table's flag columns and NXcanSAS of the fits."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from types import SimpleNamespace

from src.gimap.features.fitting.application.series_chain import (
    BACKWARD,
    CHOSEN_TOP,
    PREDICT_FACTOR,
    backward_order,
    better,
    chain_break,
    chained_fit,
    flags_text,
    frame_flags,
    restart_solution,
    series_nxcansas,
    solution_distance,
)
from src.gimap.features.fitting.application.series_fit import (
    FLAG_COLUMNS,
    FrameFit,
    SeriesSettings,
    fit_frame,
    next_start,
    series_record,
    series_table,
)
from src.gimap.features.fitting.application.fit_state import frame_from_dict, frame_to_dict, settings_from_dict, settings_to_dict
from src.gimap.features.fitting.application.single_fit import Curve, evaluate
from tests.test_fit_series import Q_NM, _model


def _curve(radius: float, seed: int = 0, scale: float = 1.0) -> Curve:
    exact = evaluate(_model(radius), Q_NM) * scale
    sigma = np.sqrt(exact) + 0.5
    measured = exact + np.random.default_rng(seed).normal(0.0, sigma)
    return Curve.from_arrays(Q_NM, measured, sigma, name=f"r{radius}", path=f"r{radius}.dat", unit="nm",
                             unit_from="test")


def test_a_chain_holds_while_the_fit_follows_and_breaks_where_chi2_jumps_or_a_fit_fails():
    settings = SeriesSettings(start="chain")
    first = fit_frame(0, "a.dat", _curve(5.0), _model(5.2), settings)
    assert first.ok and first.result.converged
    second = fit_frame(1, "b.dat", _curve(5.3, 1), first.result.model, settings)
    assert chain_break(second, first) == ""
    assert chain_break(second, first, new_stage=True) == "a new stage begins"
    failed = FrameFit(2, "c.dat", error="the curve could not be read")
    assert chain_break(failed, second).startswith("the fit failed")
    assert first.result.chi2_reduced < 2.0
    jumped = replace(second, result=replace(second.result, chi2_reduced=8.0))  # > 3× the frame before, not poor
    assert chain_break(jumped, first).startswith("χ²ᵣ jumped from")
    assert next_start(second, _model(5.2), settings) is second.result.model  # a chain starts from the result before


def test_a_value_at_a_bound_an_undetermined_value_or_a_poor_fit_does_not_break_the_chain():
    settings = SeriesSettings(start="chain")
    first = fit_frame(0, "a.dat", _curve(5.0), _model(5.2), settings)
    second = fit_frame(1, "b.dat", _curve(5.3, 1), first.result.model, settings)
    tail = ("globals", "res_width")
    model = second.result.model.with_parameter(("globals", "res_amplitude"), value=50.0, free=True)
    at_tail_bound = replace(second, result=replace(second.result, model=model, at_bounds=(tail, (0, "D"))))
    assert any(kind == "beam tail" for kind, _detail in frame_flags(at_tail_bound))  # flagged …
    assert chain_break(at_tail_bound, first) == ""  # … but the chain holds
    loose = replace(second, result=replace(second.result, errors={**second.result.errors, (0, "R"): math.inf}))
    assert chain_break(loose, first) == ""
    poor = replace(second, result=replace(second.result, chi2_reduced=12.0))
    calm = replace(first, result=replace(first.result, chi2_reduced=11.0))
    assert chain_break(poor, calm) == ""  # poor, but no jump from the frame before
    assert chain_break(poor, first).startswith("χ²ᵣ jumped from")  # a jump still breaks it


def test_predict_restarts_from_the_probable_solution_closest_to_the_frame_before():
    settings = SeriesSettings(start="chain")
    single = _model(5.2).with_parameter(("globals", "background"), value=0.0, free=False)
    previous = fit_frame(2, "c.dat", _curve(5.0), single, settings)
    solution = lambda radius, probability: SimpleNamespace(model=_model(radius), probability=probability)  # noqa: E731
    far, near, unlikely = solution(9.0, 1.0), solution(5.6, 0.5), solution(5.1, 0.5 / PREDICT_FACTOR)
    chosen, rule = restart_solution([far, near, unlikely], previous)
    assert chosen is near and "frame 3" in rule  # the closest of those within the factor; the nearest is too unlikely
    assert solution_distance(near.model, previous.result.model) < solution_distance(far.model, previous.result.model)
    assert restart_solution([near, far], None) == (far, CHOSEN_TOP)  # nothing to follow: the most probable
    with pytest.raises(RuntimeError):
        restart_solution([], previous)
    asked = []

    def solutions(data, start):
        asked.append(len(data.q))
        return [solution(9.0, 1.0), solution(7.5, 0.0)]

    kept = chained_fit(3, "d.dat", _curve(9.0, 3), previous.result.model, single, settings, previous,
                       new_stage=True, solutions=solutions)  # (a new stage: the chain breaks)
    assert asked and kept.chain and kept.start_from == "predict" and kept.start_frame == 0
    assert kept.predict_choice.startswith("the closest to frame 3")


def test_where_the_chain_breaks_predict_is_asked_and_the_better_fit_kept():
    settings = SeriesSettings(start="chain")
    single = _model(5.2).with_parameter(("globals", "background"), value=0.0, free=False)  # a value the chain keeps
    previous = fit_frame(0, "a.dat", _curve(5.0), single, settings)
    asked = []

    def predict(data, start):  # a stand-in for Predict parameters: near the truth, the background free
        asked.append(len(data.q))
        return _model(9.0)

    # A frame of a new stage (R 9 nm after 5 nm): refined from the frame before, its background stays wrong.
    far = _curve(9.0, 3)
    kept = chained_fit(1, "b.dat", far, previous.result.model, single, settings, previous, new_stage=True,
                       predict=predict)
    assert asked and kept.chain == "a new stage begins"  # the chain broke …
    assert kept.start_from == "predict" and kept.result.model.get((0, "R")).value == pytest.approx(9.0, rel=0.05)
    asked.clear()
    near = chained_fit(1, "b.dat", _curve(5.2, 4), previous.result.model, single, settings, previous, predict=predict)
    assert not asked and near.chain == "" and near.start_from == "previous" and near.fit_pass == "forward"
    assert near.start_frame == 1  # continued from frame 1


def test_better_prefers_a_converged_fit_then_the_lower_chi2():
    settings = SeriesSettings()
    good = fit_frame(0, "a.dat", _curve(5.0), _model(5.0), settings)
    worse = replace(good, result=replace(good.result, chi2_reduced=good.result.chi2_reduced * 4))
    unconverged = replace(good, result=replace(good.result, converged=False, chi2_reduced=0.1))
    failed = FrameFit(0, "a.dat", error="no")
    assert better(worse, good) is good and better(good, worse) is good
    assert better(unconverged, worse) is worse and better(failed, unconverged) is unconverged
    assert better(good, None) is good


def test_the_flags_name_failures_bounds_breaks_and_the_backward_pass():
    settings = SeriesSettings()
    frame = fit_frame(0, "a.dat", _curve(5.0), _model(5.0), settings)
    assert frame_flags(FrameFit(1, "b.dat", error="no points")) == [("failed", "no points")]
    at_bound = replace(frame, result=replace(frame.result, at_bounds=((0, "R"),)), chain="χ²ᵣ jumped from 1 to 9",
                       start_from="predict", fit_pass=BACKWARD)
    kinds = [kind for kind, _detail in frame_flags(at_bound)]
    assert kinds[:1] == ["at a bound"] and "chain broke" in kinds and kinds[-1] == "backward"
    assert "R" in flags_text(at_bound) and "Predict parameters" in flags_text(at_bound)
    assert backward_order([4, 0, 2, 1, 3]) == [3, 2, 1, 0] and backward_order([7]) == []


def test_the_saved_table_and_record_carry_the_flags_and_the_frames_round_trip(tmp_path):
    settings = SeriesSettings(start="chain", backward=True)
    frames = [replace(fit_frame(index, f"run_{index + 1}.dat", _curve(5.0 + 0.2 * index, index), _model(5.2), settings),
                      start_from="previous", fit_pass="forward") for index in range(3)]
    frames.append(FrameFit(3, "run_4.dat", error="the curve could not be read"))
    header, rows = series_table(frames, _model(5.2), details=True, flags=True)
    count = len(FLAG_COLUMNS)
    assert header[-count:] == list(FLAG_COLUMNS) and all(len(row) == len(header) for row in rows)
    assert rows[3][-count].startswith("failed") and rows[0][-count + 1:] == ["previous", "", "", "", "forward"]
    plain, _rows = series_table(frames, _model(5.2), details=True)
    assert plain == header[:-count]  # without ``flags`` the table is as before
    record = series_record(_model(5.2), settings, "folder", "*.dat", frames)
    assert record["settings"]["start"] == "chain" and record["settings"]["backward_pass"] is True
    assert record["per_frame"][0]["start_from"] == "previous" and record["per_frame"][0]["fit_pass"] == "forward"
    again = frame_from_dict(frame_to_dict(replace(frames[1], start_frame=1, predict_choice="the most probable")))
    assert (again.start_from, again.fit_pass, again.start_frame, again.predict_choice) == (
        "previous", "forward", 1, "the most probable")
    assert settings_from_dict(settings_to_dict(settings)).backward is True


def test_the_fits_of_a_series_as_nxcansas(tmp_path):
    from src.gimap.shared.nxcansas import read_nxcansas, write_nxcansas

    settings = SeriesSettings()
    curves = {index: _curve(5.0 + 0.3 * index, index) for index in range(2)}
    frames = [fit_frame(index, f"run_{index + 1}.dat", curve, _model(5.0), settings) for index, curve in curves.items()]
    frames.append(FrameFit(2, "run_3.dat", error="no"))
    sas = series_nxcansas(frames, curves, settings, lambda frame: f"frame {frame.index + 1}",
                          lambda frame: {"frame": frame.index + 1})
    assert [item.title for item in sas] == ["frame 1: the points fitted", "frame 1: the model",
                                            "frame 2: the points fitted", "frame 2: the model"]
    path = write_nxcansas(tmp_path / "fits.h5", sas)
    back = read_nxcansas(path)
    assert len(back) == 4 and back[0].sigma is not None and back[1].sigma is None
    assert [item.role for item in back] == ["data", "model", "data", "model"]  # opened again: the measured curves
    assert math.isclose(float(back[0].q[0]), float(Q_NM[0]), rel_tol=1e-9)
