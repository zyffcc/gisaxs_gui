"""A stopped fitting workflow keeps usable values without reporting a completed fit.

Cancellation is placed at deterministic boundaries, including after a worker has queued its result.
All model values and curves below are synthetic; no real data or optional prediction runtime is used.
"""

from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from src.gimap.features.fitting.application import single_fit
from src.gimap.features.fitting.domain import fit_edges
from src.gimap.features.fitting.domain.fit_engine import FitData, FitResult
from src.gimap.features.fitting.domain.fit_model import FitModel, new_component


def _completed_result(*, at_edge=True, model=None, points=100):
    if model is None:
        model = FitModel((new_component("sphere", radius=6.0, distance=30.0),))
        model = model.with_parameter((0, "R"), lower=1.0, upper=6.0)
    return FitResult(
        model=model, method="global", weighting="sigma", points=points, free=4,
        chi2_reduced=1.25, log_rmse=0.08, evaluations=32, seconds=0.2,
        converged=True, stopped=False, message="Converged", errors={(0, "R"): 0.1},
        at_bounds=((0, "R"),) if at_edge else (),
    )


def _assert_stopped_with_values(result, usable):
    assert result.stopped, "The overall task was stopped, even if an earlier fit had converged"
    assert not result.converged
    assert result.model == usable.model
    for field in ("method", "weighting", "points", "free", "chi2_reduced", "log_rmse",
                  "evaluations", "seconds", "errors", "correlated", "at_bounds"):
        assert getattr(result, field) == getattr(usable, field), field


@pytest.mark.parametrize("resolve", [False, True])
def test_stop_after_initial_fit_is_kept_by_the_whole_use_case(monkeypatch, resolve):
    stopped, calls = Event(), []
    usable = _completed_result()
    q = np.linspace(0.1, 1.0, 100)
    data = FitData.prepare(q, np.full(100, 10.0), np.ones(100))

    def completed_then_stopped(model, fitted_data, **options):
        calls.append((model, fitted_data))
        assert options["stop"] == stopped.is_set
        stopped.set()  # optimization finished; the whole request is still finishing
        return usable

    monkeypatch.setattr(single_fit, "fit", completed_then_stopped)
    result, actions, _remaining = single_fit.fit_resolving(
        usable.model, data, stop=stopped.is_set, resolve=resolve,
    )
    assert len(calls) == 1, "No later edge refit may start after Stop"
    assert actions == []
    _assert_stopped_with_values(result, usable)


@pytest.mark.parametrize("stage", [fit_edges.resolve, fit_edges.settle], ids=["resolve", "settle"])
def test_stop_before_an_edge_stage_keeps_values_without_starting_a_trial(stage):
    stopped = Event()
    stopped.set()
    usable = _completed_result()

    def unwanted_trial(_model):
        pytest.fail("A cancelled edge stage must not start another fit")

    result, actions, *_remaining = stage(unwanted_trial, usable, cancelled=stopped.is_set)
    assert actions == []
    _assert_stopped_with_values(result, usable)


@pytest.mark.parametrize("stage", [fit_edges.resolve, fit_edges.settle], ids=["widen", "settle"])
def test_stopped_edge_trial_keeps_the_prior_fit_and_does_not_claim_it_was_worse(stage):
    stopped, calls = Event(), []
    usable = _completed_result()

    def interrupted_trial(model):
        calls.append(model)
        stopped.set()
        # The partial trial even has lower chi2: cancellation must not be called a worse fit.
        return replace(usable, model=model, chi2_reduced=0.8, stopped=True, converged=False,
                       message="Stopped during edge refinement")

    result, actions, *_remaining = stage(interrupted_trial, usable, cancelled=stopped.is_set)
    assert len(calls) == 1, "Neither another trial nor another edge stage may run after Stop"
    assert actions == [], "An interrupted trial establishes neither a worse fit nor a settled bound"
    _assert_stopped_with_values(result, usable)


def test_stop_after_worker_completion_before_gui_delivery_is_reported_as_stopped(tmp_path, monkeypatch):
    from src.gimap.features.fitting.presentation.single import run
    from tests.test_fit_page import _curve_file, _page, _wait

    page = _page()
    try:
        assert page.open_curve(_curve_file(tmp_path))
        usable = _completed_result(at_edge=False, model=page.session.model, points=240)
        worker_finished = Event()

        def completed(*_args, **_kwargs):
            worker_finished.set()
            return usable, [], []

        monkeypatch.setattr(run, "fit_resolving", completed)
        page.run_fit("global")
        # Wait for the worker without processing Qt events: its result is now queued, not displayed.
        assert page.tasks._pool.waitForDone(5000)
        assert worker_finished.is_set() and page.fit_running()
        assert page.session.result is None
        page.stop_fit()
        assert page.fit_progress_text.text() == "Stopping … the best values so far are kept."
        _wait(page)
        _assert_stopped_with_values(page.session.result, usable)
        assert page.status_label.text().startswith("Stopped: the best values so far are kept (")
        assert page.step_rail.state("fit") == "warn"
        assert not page.fit_running()
    finally:
        page.dispose()
