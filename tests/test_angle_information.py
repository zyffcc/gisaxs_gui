"""Which incidence angle tells most: Fisher information of a fitted cut at other angles (fitting/domain/angle_information.py).
Synthetic only: a sphere population on Si, some of it buried in a cellulose-like film."""

from __future__ import annotations

import numpy as np
import pytest

from src.gimap.features.fitting.domain.angle_information import (
    Burial, FittedCut, burial_factor, expected_counts, penetration_nm, score_angles, separation,
)
from src.gimap.features.fitting.domain.cut_model import Geometry, Population, Settings, evaluate

SI = dict(delta=4.88e-6, beta=7.4e-8)  # Si at 10 keV
FILM = (3.0e-6, 2.0e-9)  # a cellulose film at 10 keV: critical angle 0.140°


def _cut(*, burial=None, radius=5.0, scale=1.0) -> FittedCut:
    geometry = Geometry(mode="dwba", wavelength_nm=0.12398, alpha_i_deg=0.4, alpha_f_deg=None, **SI)
    population = Population("sphere", radius=radius, radius_spread=0.1, distance=40.0, distance_spread=0.3)
    q = np.linspace(0.05, 1.5, 300)
    k = 200.0 / evaluate([population], Settings(geometry=geometry), q).max() * scale
    return FittedCut((population,), Settings(geometry=geometry, k=k, background=0.5), q, np.full(q.size, 5.0),
                     exposure_s=1.0, burial=burial)


def test_the_wave_reaches_a_few_nm_into_the_film_below_its_critical_angle_and_microns_above() -> None:
    k = 2 * np.pi / 0.12398
    assert penetration_nm(0.10, *FILM, k) == pytest.approx(5.7, abs=0.2)
    assert penetration_nm(0.20, *FILM, k) > 5000
    deep, shallow = Burial(*FILM, depth_nm=50.0), Burial(*FILM, depth_nm=2.0)
    assert burial_factor(deep, 0.10, 0.18, k) < 0.2 * burial_factor(shallow, 0.10, 0.18, k)  # buried deep: dark
    assert burial_factor(deep, 0.40, 0.18, k) == pytest.approx(burial_factor(shallow, 0.40, 0.18, k), rel=0.02)


def test_the_counts_follow_the_exposure_and_the_model_floor_limits_what_more_photons_give() -> None:
    cut = _cut()
    assert np.allclose(expected_counts(cut, 0.4, 2.0), 2 * expected_counts(cut, 0.4, 1.0))
    def bound(exposure, floor):
        return score_angles([cut], [0.4], exposure, floor=floor)[0].bounds["0:radius"]
    assert bound(100.0, 0.0) == pytest.approx(bound(1.0, 0.0) / 10, rel=1e-3)  # photons alone: 1/√t
    assert bound(100.0, 0.05) > 2 * bound(100.0, 0.0)  # a 5 % model limits the error …
    assert bound(1000.0, 0.05) > 0.6 * bound(100.0, 0.05)  # … and ten times the photons help little


def test_below_the_film_s_critical_angle_an_exposure_tells_the_depth_of_buried_particles() -> None:
    cut = _cut(burial=Burial(*FILM, fraction=0.3, depth_nm=50.0))
    scores = score_angles([cut], [0.10, 0.12, 0.15, 0.25, 0.6], 1.0, made=[(0.4, 10.0)])
    best = max(scores, key=lambda score: score.gain)
    assert best.alpha_i_deg < 0.14  # below the film's critical angle
    depth = {score.alpha_i_deg: score.bounds["burial:depth_nm"] for score in scores}
    assert depth[best.alpha_i_deg] < 0.2 * min(depth[0.15], depth[0.25], depth[0.6])
    only_reference = score_angles([cut], [0.4], 10.0)[0].bounds["burial:depth_nm"]
    assert only_reference > 2.0  # the reference angle alone leaves the depth open (above 200 %)


def test_candidate_solutions_are_told_apart_where_their_counts_differ() -> None:
    cut = _cut()
    assert separation([cut, cut], 0.4, 1.0) == pytest.approx(0.0)
    other = _cut(radius=5.6, scale=0.8)
    assert separation([cut, other], 0.4, 1.0) > 25.0
    assert np.isnan(separation([cut], 0.4, 1.0))


def test_a_fit_of_fitting_is_scored_like_the_cut_model_it_stands_for() -> None:
    from dataclasses import replace

    from src.gimap.features.fitting.domain.angle_information import ModelCut
    from src.gimap.features.fitting.domain.fit_model import FitModel, new_component
    from src.gimap.features.fitting.domain.substrates import Substrate

    model = FitModel(components=(new_component("sphere", radius=5.0, distance=40.0),), substrate=Substrate(delta=SI["delta"], beta=SI["beta"]))
    model = model.with_values({("globals", "alpha_i"): 0.4, ("globals", "alpha_f"): 0.18, ("globals", "wavelength"): 0.12398,
                               ("globals", "background"): 0.5})
    q = np.linspace(0.05, 1.5, 300)
    cut = ModelCut(model, q, np.full(q.size, 5.0), exposure_s=1.0)
    assert cut.exit_deg == pytest.approx(np.degrees(np.sqrt(2 * SI["delta"])))  # the Yoneda row at every angle
    assert {name for _index, name in cut.wanted()} >= {"R", "D"}
    scores = score_angles([cut], [0.1, 0.4, 0.6], 1.0)
    assert scores[0].counts > scores[1].counts > scores[2].counts  # more light below the critical angle
    buried = replace(cut, burial=Burial(*FILM, fraction=0.3, depth_nm=50.0))
    after = score_angles([buried], [0.12, 0.25], 1.0, made=[(0.4, 10.0)])
    assert after[0].bounds["burial:depth_nm"] < 0.3 * after[1].bounds["burial:depth_nm"]


# -- Fitting ▸ Rank Incidence Angles… -----------------------------------------------------------------------------------


def _fitted_session(tmp_path):
    from types import SimpleNamespace

    from src.gimap.features.fitting.domain.fit_engine import FitData
    from src.gimap.features.fitting.domain.fit_model import FitModel, new_component
    from src.gimap.features.fitting.domain.fit_model import evaluate as model_curve
    from src.gimap.features.fitting.domain.substrates import Substrate

    model = FitModel(components=(new_component("sphere", radius=5.0, distance=40.0),),
                     substrate=Substrate(delta=SI["delta"], beta=SI["beta"]))
    model = model.with_values({("globals", "alpha_i"): 0.4, ("globals", "alpha_f"): 0.18,
                               ("globals", "wavelength"): 0.12398, ("globals", "background"): 0.5})
    q = np.linspace(0.05, 1.5, 200)
    model = model.with_values({(0, "Int"): 200.0 / model_curve(model, q).max()})
    intensity = model_curve(model, q)
    data = FitData(q, intensity, np.sqrt(intensity / 5.0))  # five pixels a point
    curve = SimpleNamespace(path=str(tmp_path / "frame_00007_horizontal.dat"))
    return SimpleNamespace(curve=curve, model=model, data=lambda: data), data


def test_the_ranking_uses_the_fit_and_the_points_and_records_the_choice(tmp_path) -> None:
    import json

    from src.gimap.features.fitting.application.angle_choice import (
        AngleRequest, append_decision, decision_record, pixels_of, rank_angles, with_burial,
    )

    session, data = _fitted_session(tmp_path)
    assert np.allclose(pixels_of(data), 5.0)
    request = with_burial(AngleRequest((0.1, 0.12, 0.25, 0.6), made=((0.4, 10.0),)), delta=FILM[0], beta=FILM[1],
                          fraction=0.3, depth_nm=50.0)
    rows = rank_angles(session.model, data, request)
    assert [row["alpha_i_deg"] for row in rows] == [0.1, 0.12, 0.25, 0.6]
    best = min(rows, key=lambda row: row["rank"])
    assert best["alpha_i_deg"] < 0.14 and "burial:depth_nm" in best["bounds"]
    path = append_decision(tmp_path, decision_record(session.model, session.curve.path, request, rows, 0.12, "test"))
    append_decision(tmp_path, decision_record(session.model, session.curve.path, request, rows, 0.6))
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [line["chosen_deg"] for line in lines] == [0.12, 0.6] and lines[0]["note"] == "test"
    assert lines[0]["parameters"]["0:R"] == 5.0 and len(lines[0]["candidates"]) == 4


def test_the_window_ranks_the_candidates_and_writes_the_choice_next_to_the_curve(tmp_path) -> None:
    from types import SimpleNamespace

    from src.gimap.features.fitting.presentation.single.angle_rank_dialog import AngleRankDialog
    from tests.test_analyze_workspace import _app

    _app()
    session, _data = _fitted_session(tmp_path)
    dialog = AngleRankDialog(SimpleNamespace(session=session), None)
    try:
        dialog.angle_list_edit.setText("0.1, 0.12, 0.25")
        dialog.angle_made_edit.setText("0.4:10")
        dialog.angle_burial_check.setChecked(True)
        assert dialog.rank()
        table = dialog.angle_table
        assert table.rowCount() == 3 and table.horizontalHeaderItem(0).text() == "Rank"
        assert "Best next: αi" in dialog.angle_status.text() and dialog.angle_record_button.isEnabled()
        path = dialog.record()
        assert path is not None and path.parent == tmp_path and path.name == "angle_decisions.jsonl"
        dialog.angle_list_edit.setText("0.1, 95")
        assert not dialog.rank() and "separated by commas" in dialog.angle_status.text()
    finally:
        dialog.dispose()
