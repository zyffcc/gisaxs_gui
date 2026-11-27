"""While Predict runs, the page draws what it does: the network's first guess, then the fit just refined and the best
so far (``posterior.prediction.predict``'s ``preview``, ``posterior_job.StageReport.preview``,
``presentation/single/predict_progress``). The preview changes none of the numbers."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.gimap.app.jobs.models import JobProgress
from src.gimap.app.presentation.i18n.zh import ZH
from src.gimap.features.fitting.domain import fit_model as fm
from src.gimap.features.fitting.domain.cut_model import model as cut
from src.gimap.features.fitting.domain.posterior import encoding as E
from src.gimap.features.fitting.domain.posterior import refine
from src.gimap.features.fitting.domain.posterior.flow import Network
from src.gimap.features.fitting.infrastructure.adapters.posterior_job import StageReport
from src.gimap.features.fitting.presentation.single import predict_progress as P

DATA = Path(__file__).resolve().parent / "data" / "ml1d"
SPHERE = E.composition_index([("sphere", True)])


def _curve(seed=3):
    q = np.linspace(0.02, 2.0, 300)
    population = cut.Population("sphere", 4.0, 0.15, distance=14.0, distance_spread=0.2)
    truth = 1e4 * cut.unit_terms([population], cut.Settings(), q)[0] + 2.0
    counts = np.random.default_rng(seed).poisson(truth).astype(float)
    return q, counts, np.sqrt(np.maximum(counts, 1.0))


def test_the_preview_shows_each_fit_and_changes_nothing():
    network = Network.load(DATA / "flow_fixture.npz")
    q, counts, sigma = _curve()
    seen = []
    options = dict(compositions=[SPHERE], samples=150, local_samples=120, seed=1)
    alone = refine.predict(network, q, counts, sigma, **options)
    shown = refine.predict(network, q, counts, sigma, preview=lambda latest, best: seen.append((latest, best)),
                           **options)
    assert [(s.name, s.probability, s.chi2_reduced) for s in alone] == [
        (s.name, s.probability, s.chi2_reduced) for s in shown]
    first, *fits = seen
    assert first[0][1] is None and first[1] is None  # the network's first guess, before any fit
    assert fits and all(isinstance(latest[0], fm.FitModel) and np.isfinite(latest[1]) for latest, _best in fits)
    placed = [latest[1] for latest, _best in fits]
    assert all(best[1] in placed[:n + 1] for n, (_latest, best) in enumerate(fits))  # the best is a fit placed so far


def test_the_job_sends_the_fits_as_json():
    reports = []
    stages = StageReport(lambda *args: reports.append(args), {"fitted": 3}, 0.0)
    stages.progress("refinement", 1, 4)
    model = fm.FitModel((fm.new_component("sphere", radius=3.0),))
    stages.preview((model, 2.5), (model, 1.25))
    details = reports[-1][3]
    assert details["stage"] == "refinement" and (details["done"], details["total"]) == (1, 4)
    json.dumps(details)  # the job sends it through a queue
    assert details["preview"]["latest"]["chi2"] == 2.5 and details["preview"]["best"]["chi2"] == 1.25
    assert fm.model_from_dict(details["preview"]["best"]["model"]).components[0].value("R") == 3.0
    stages.preview((model, None), None)
    assert reports[-1][3]["preview"] == {"latest": {"model": fm.model_to_dict(model), "chi2": None}, "best": None}


def test_preview_curves_are_kept_by_model_and_the_best_is_drawn_once():
    grid = P.preview_grid(np.linspace(-1.5, 1.5, 101))
    assert grid.size == P.PREVIEW_POINTS and grid.min() > 0
    sphere = fm.model_to_dict(fm.FitModel((fm.new_component("sphere", radius=3.0),)))
    other = fm.model_to_dict(fm.FitModel((fm.new_component("sphere", radius=5.0),)))
    cache = {}
    same = P.preview_curves({"latest": {"model": sphere, "chi2": 2.0}, "best": {"model": sphere, "chi2": 2.0}},
                            grid, cache)
    assert set(same) == {"best"} and len(cache) == 1
    both = P.preview_curves({"latest": {"model": other, "chi2": 3.0}, "best": {"model": sphere, "chi2": 2.0}},
                            grid, cache)
    assert set(both) == {"best", "latest"} and len(cache) == 2
    rows = P.preview_series(both, grid)
    assert [row[3] for row in rows] == [P.LATEST_COLOR, P.BEST_COLOR]
    assert "χ²ᵣ 2" in rows[1][0] and "χ²ᵣ 3" in rows[0][0]
    guess = P.preview_series(P.preview_curves({"latest": {"model": other, "chi2": None}, "best": None}, grid, cache),
                             grid)
    assert len(guess) == 1 and "first guess" in guess[0][0]
    for text in (P.BEST_SO_FAR, P.JUST_REFINED, P.FIRST_GUESS):
        assert ZH[text]


def test_the_page_draws_the_fits_while_predict_runs_and_removes_them(tmp_path, monkeypatch):
    from tests.test_fit_page import _page
    from tests.test_fit_page_predict import _sphere_file

    page = _page()
    try:
        assert page.open_curve(_sphere_file(tmp_path)), page.status_label.text()
        page._predict_grid = P.preview_grid(page.session.data(predict=True).q)
        page._running, page._predict_job_id = "predict", "job-1"
        page._predict_started(None)
        model = fm.model_to_dict(fm.FitModel((fm.new_component("sphere", radius=4.0),)))
        progress = JobProgress(job_id="job-1", completed=0.3, total=1.0, message="",
                               details={"stage": "refinement", "done": 1, "total": 2,
                                        "preview": {"latest": {"model": model, "chi2": 1.5}, "best": None}})
        page._predict_report(progress)
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()
        assert page._predict_preview is not None and "latest" in page._predict_preview
        assert any("just refined" in name for name, _x, _y in page.plot._curves)
        other = JobProgress(job_id="another", completed=0.4, total=1.0, message="",
                            details={"stage": "refinement", "done": 1, "total": 2,
                                     "preview": {"latest": None, "best": {"model": model, "chi2": 1.0}}})
        page._predict_report(other)
        QApplication.processEvents()
        assert "best" not in page._predict_preview  # a report of another job is not drawn
        page._finish()
        assert page._predict_preview is None
        assert not any("just refined" in name for name, _x, _y in page.plot._curves)
    finally:
        page._running = None
        page.dispose()
