"""Predict parameters on the single-curve page: the job, the solutions with their probability, the posterior lines
(with the stored random network of tests/data/ml1d, set through GIMAP_ML1D_NETWORK: the job's least squares finds the
values whatever it proposed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.fitting.application.single_fit import FitModel, new_component
from src.gimap.features.fitting.application.single_predict import compositions, cut_geometry, predicted
from src.gimap.features.fitting.domain.cut_model import model as cut
from src.gimap.features.fitting.domain.posterior import encoding as E
from tests.test_fit_page import _page, _wait

NETWORK = Path(__file__).resolve().parent / "data" / "ml1d" / "flow_fixture"


def _sphere_file(tmp_path):
    q = np.linspace(0.02, 2.0, 300)
    population = cut.Population("sphere", 4.0, 0.15, distance=14.0, distance_spread=0.2)
    y = np.random.default_rng(0).poisson(1e4 * cut.unit_terms([population], cut.Settings(), q)[0] + 2.0).astype(float)
    path = tmp_path / "spheres.dat"
    np.savetxt(path, np.column_stack([q / 10, y, np.sqrt(np.maximum(y, 1))]))
    return path


def test_compositions_and_geometry_of_a_model():
    model = FitModel((new_component("vertical_cylinder", radius=6.0), new_component("sphere", radius=2.0)))
    (index,) = compositions(model, "model")["compositions"]
    assert E.composition_types(index) == (("sphere", True), ("vertical_cylinder", True))  # ordered by radius
    assert len(compositions(model, "spheres_cylinders")["allowed"]) == 4 + 4 ** 2  # 4 types (S or not), K ≤ 2
    assert compositions(FitModel(), "any")["allowed"] == list(range(len(E.COMPOSITIONS)))
    with pytest.raises(ValueError):
        compositions(FitModel(), "model")
    assert cut_geometry(model) == (0.0, {})
    tilted = model.with_parameter(("globals", "alpha_i"), value=0.4).with_parameter(("globals", "alpha_f"), value=0.15)
    tilted = tilted.with_parameter(("globals", "wavelength"), value=0.1)
    qz, grazing = cut_geometry(tilted)
    k = 2 * np.pi / 0.1
    assert qz == pytest.approx(k * (np.sin(np.radians(0.4)) + np.sin(np.radians(0.15))))
    assert grazing["alpha_i"] == 0.4 and grazing["refl_amp"] == 0.0


def test_predict_on_the_page(tmp_path, monkeypatch):
    monkeypatch.setenv("GIMAP_ML1D_NETWORK", str(NETWORK))
    page = _page()
    try:
        assert page.open_curve(_sphere_file(tmp_path)), page.status_label.text()
        page.set_model(FitModel((new_component("sphere", radius=3.0),)))
        page.set_method("predict")
        assert not page.predict_row.isHidden() and page.fit1d_row.isHidden()
        page.predict_particles.setCurrentIndex(page.predict_particles.findData("model"))
        page.run_fit()
        _wait(page)
        solutions = page.session.solutions
        assert solutions, page.status_label.text()
        assert all(s.probability is not None and s.source.startswith("Predict parameters") for s in solutions)
        assert sum(s.probability for s in solutions) <= 1.0 + 1e-9
        best = page.session.model.components[0]
        assert best.family == "sphere" and best.value("R") == pytest.approx(4.0, rel=0.05)
        assert best.value("D") == pytest.approx(14.0, rel=0.05)
        assert not page.solutions_table.isColumnHidden(5)
        assert "Posterior of this solution" in page.warnings_label.text()
        assert page.method() == "local"  # next: Refine under Model's own limits
    finally:
        page.dispose()


def test_a_missing_network_is_said(tmp_path, monkeypatch):
    monkeypatch.setenv("GIMAP_ML1D_NETWORK", str(tmp_path / "nowhere"))
    page = _page()
    try:
        assert page.open_curve(_sphere_file(tmp_path))
        page.set_method("predict")
        page.run_fit()
        assert not page.fit_running() and "proposal network, which is missing" in page.status_label.text()
    finally:
        page.dispose()


def test_decoding_a_job_result():
    value = {"solutions": [{"composition": 0, "name": "sphere (S)", "probability": 0.7, "chi2": 1.1,
                            "model": __import__("src.gimap.features.fitting.domain.fit_model",
                                                fromlist=["model_to_dict"]).model_to_dict(
                                FitModel((new_component("sphere"),))),
                            "values": {"R_1": [5.0, 5.1, 4.9, 5.3, 4.7, 5.5, 0.1]}, "flags": ["low_ess"], "ess": 12}]}
    (one,) = predicted(value)
    assert one.probability == 0.7 and one.intervals["R_1"][2] == 4.9 and one.flags == ("low_ess",)
