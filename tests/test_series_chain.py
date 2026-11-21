"""The chained fit of an In-situ series (``application/series_chain``): each frame from the result of the one before,
Predict parameters only where the chain breaks (a stand-in for it here: the method itself is not touched), the better
of two fits kept, the flags of a frame, the backward pass order, the table's flag columns and NXcanSAS of the fits."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from src.gimap.features.fitting.application.series_chain import (
    BACKWARD,
    backward_order,
    better,
    chain_break,
    chained_fit,
    flags_text,
    frame_flags,
    series_nxcansas,
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


def test_where_the_chain_breaks_predict_is_asked_and_the_better_fit_kept():
    settings = SeriesSettings(start="chain")
    single = _model(5.2)
    previous = fit_frame(0, "a.dat", _curve(5.0), single, settings)
    asked = []

    def predict(data, start):  # a stand-in for Predict parameters: the particles of ``start``, near the truth
        asked.append(len(data.q))
        return _model(9.0)

    # A frame far from the one before (R 9 nm after 5 nm): refined from 5 nm it lands in a poor fit.
    far = _curve(9.0, 3)
    kept = chained_fit(1, "b.dat", far, previous.result.model, single, settings, previous, predict=predict)
    assert asked and kept.chain  # the chain broke …
    assert kept.start_from == "predict" and kept.result.model.get((0, "R")).value == pytest.approx(9.0, rel=0.05)
    asked.clear()
    near = chained_fit(1, "b.dat", _curve(5.2, 4), previous.result.model, single, settings, previous, predict=predict)
    assert not asked and near.chain == "" and near.start_from == "previous" and near.fit_pass == "forward"


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
    assert header[-4:] == list(FLAG_COLUMNS) and all(len(row) == len(header) for row in rows)
    assert rows[3][-4].startswith("failed") and rows[0][-3:] == ["previous", "", "forward"]
    plain, _rows = series_table(frames, _model(5.2), details=True)
    assert plain == header[:-4]  # without ``flags`` the table is as before
    record = series_record(_model(5.2), settings, "folder", "*.dat", frames)
    assert record["settings"]["start"] == "chain" and record["settings"]["backward_pass"] is True
    assert record["per_frame"][0]["start_from"] == "previous" and record["per_frame"][0]["fit_pass"] == "forward"
    again = frame_from_dict(frame_to_dict(frames[1]))
    assert (again.start_from, again.fit_pass) == ("previous", "forward")
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
    assert math.isclose(float(back[0].q[0]), float(Q_NM[0]), rel_tol=1e-9)
