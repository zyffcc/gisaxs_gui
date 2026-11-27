"""An In-situ series of a run that switches the incidence angle fits the frames of the model's angle only.

Analyze writes the αi of a GISAXS cut into the ``# observation:`` line of its fit input; a frame of another angle
is listed as ``other angle`` with the reason, is not fitted and does not break or feed the chain. Synthetic curves.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.gimap.features.fitting.application.series_fit import FrameFit, series_counts
from src.gimap.features.fitting.application.series_incidence import (
    OTHER_ANGLE,
    frame_incidence,
    model_incidence,
    other_angle,
    other_angle_values,
)
from tests.test_fit_series import Q_NM, RADII, _model, _open_series, _pages, _wait
from src.gimap.features.fitting.application.single_fit import evaluate

ANGLES = (0.4, 0.1, 0.4, 0.25, 0.4)


def _at(model, alpha: float):
    return model.with_values({("globals", "alpha_i"): alpha})


def _write(path: Path, radius: float, seed: int, alpha) -> None:
    exact = evaluate(_at(_model(radius), 0.4), Q_NM)  # the curves of the model the series fits
    sigma = np.sqrt(exact) + 0.5
    measured = exact + np.random.default_rng(seed).normal(0.0, sigma)
    observation = {"source": "native_detector_columns", **({} if alpha is None else {"alpha_i_deg": alpha})}
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# GIMaP Analyze fit input (q in 1/A)\n# observation: " + json.dumps(observation) + "\n")
        np.savetxt(handle, np.column_stack([Q_NM / 10.0, measured, sigma]))


def test_the_rule_compares_the_frames_angle_with_the_models():
    model = _at(_model(5.0), 0.4)
    assert frame_incidence({"alpha_i_deg": 0.1}) == pytest.approx(0.1) and frame_incidence({}) is None
    assert model_incidence(model) == pytest.approx(0.4) and model_incidence(_model(5.0)) is None  # αi 0: none
    assert other_angle(model, {"alpha_i_deg": 0.402}) == ""  # one motor position
    assert other_angle(model, {}) == "" and other_angle(_model(5.0), {"alpha_i_deg": 0.1}) == ""
    reason = other_angle(model, {"alpha_i_deg": 0.1})
    assert reason == OTHER_ANGLE.format(frame="0.1", model="0.4")
    assert other_angle_values(reason) == {"frame": "0.1", "model": "0.4"} and other_angle_values("other") is None
    frame = FrameFit(1, "run_00002.dat", other_angle=reason)
    assert frame.state == "other angle" and frame.reason == reason and not frame.ok
    frames = [frame, FrameFit(2, "run_00003.dat", error="could not be read")]
    assert series_counts(frames)["other_angle"] == 1 and series_counts(frames)["failed"] == 1


def test_the_observation_of_a_gisaxs_cut_names_its_angle():
    from src.gimap.features.analyze.application import fit_input_observation
    from src.gimap.features.analyze.domain import Corrections

    frame = SimpleNamespace(kind="gisaxs", corrections=Corrections(gap_guard_px=3), path=Path("frame.edf"), metadata={},
                            intensity_scale=None, frame_total=1, geometry=SimpleNamespace(incidence_deg=0.25))
    assert fit_input_observation(frame)["alpha_i_deg"] == pytest.approx(0.25)
    frame.geometry = SimpleNamespace(incidence_deg=0.0)  # transmission: no angle to name
    assert "alpha_i_deg" not in fit_input_observation(frame)


def test_a_series_fits_the_frames_of_the_models_angle_and_lists_the_others(tmp_path):
    single, series = _pages()
    folder = tmp_path / "gimap_analysis"
    folder.mkdir()
    for index, (radius, alpha) in enumerate(zip(RADII, ANGLES)):
        _write(folder / f"run_{index + 1:05d}_fit_input.dat", radius, index, alpha)
    try:
        single.open_curve(folder / "run_00001_fit_input.dat")
        single.set_model(_at(_model(5.2), 0.4))
        assert _open_series(series, folder)
        series.start()
        _wait(series)
        states = [series.fits[index].state for index in range(5)]
        assert states == ["ok", "other angle", "ok", "other angle", "ok"]
        assert series.fits[1].other_angle == OTHER_ANGLE.format(frame="0.1", model="0.4")
        assert series.frame_list.item(1).text().startswith("–")
        assert [series.fits[index].result.model.get((0, "R")).value for index in (0, 2, 4)] == pytest.approx(
            [RADII[index] for index in (0, 2, 4)], rel=0.05)
        assert series.save_table(str(tmp_path / "series.csv"))
        record = json.loads((tmp_path / "series.json").read_text(encoding="utf-8"))
        assert record["other_angle"] == [2, 4] and record["failed"] == []
        assert record["counts"]["other_angle"] == 2 and record["counts"]["ok"] == 3
    finally:
        series.dispose()
        single.dispose()
