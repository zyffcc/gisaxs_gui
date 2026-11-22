"""The inputs and the report of a known-substrate fit through automatic GISAXS analysis."""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from src.gimap.features.analyze.presentation.automation import header_values
from src.gimap.features.assistant.application.gisaxs_tools import GisaxsToolsMixin
from src.gimap.features.assistant.domain.notes import substrate_from_notes
from src.gimap.shared.detector_io.nexus_metadata import extract_nexus_metadata
from tests.test_guided_gisaxs import _row


def test_explicit_substrate_in_session_notes():
    assert substrate_from_notes("energy 10 keV\nsubstrate = SiO2\n") == "SiO2"
    assert substrate_from_notes("基底：Au") == "Au"
    assert substrate_from_notes("Ag on Si") is None
    assert substrate_from_notes("substrate = Si with oxide") is None
    assert substrate_from_notes("substrate = Si\nsubstrate = Au") is None


@pytest.mark.parametrize("optical", [False, True])
def test_nexus_substrate_of_current_entry_reaches_fitter(tmp_path, optical):
    path = tmp_path / "public_model.nxs"
    with h5py.File(path, "w") as file:
        file["entry/sample/substrate/material"] = "Au"
        if optical:
            file["scan/sample/substrate/delta"] = 3.5e-6
            file["scan/sample/substrate/beta"] = 1e-8
        else:
            file["scan/sample/substrate"] = "Si"
        header = header_values(extract_nexus_metadata(file, entry="/scan"))
    status = dict(header=header, geometry={"wavelength_A": 1.18, "incidence_deg": .3},
                  gisaxs={"horizontal_source": "yoneda", "yoneda_alpha_f_deg": .15})
    tools = GisaxsToolsMixin()
    tools.results = SimpleNamespace(status=status, gisaxs={})
    tools.cancelled = lambda: False
    tools._ok = lambda data, summary: data
    q = np.linspace(.02, .2, 80)
    tools._fit_input = lambda side: ("positive", q, 1 / q**2, 1 / q, "positive half")

    def fitter(x, y, sigma, **kwargs):
        assert kwargs["grazing"] == pytest.approx(dict(wavelength=.118, alpha_i=.3, alpha_f=.15))
        expected = {"material": "custom", "delta": 3.5e-6, "beta": 1e-8} if optical else {"material": "Si"}
        assert kwargs["substrate"] == expected
        return [dict(_row(x, y, sigma, 1, "sphere", 4., 14., 1.), scattering="four-term DWBA on test substrate")]

    tools.fitter = fitter
    tools._tool_fit_horizontal_cut()
    assert tools.results.gisaxs["fit"]["substrate_level"] == 4
    assert tools.results.gisaxs["fit"]["sphere_model"] == "four-term DWBA on test substrate"


def test_session_override_reaches_standard_pipeline_and_report():
    from src.gimap.app.headless_assistant import open_headless_session
    from src.gimap.features.assistant.application import PipelineOptions
    from tests.test_guided_gisaxs import DATA, GALAXI, _geometry

    layer = dict(thickness_nm=50., delta=1.7e-6, beta=1e-9)
    substrate = dict(material="Si", layers=[layer])

    def fitter(q, intensity, sigma, *, substrate, grazing, **kwargs):
        assert substrate["layers"] == [layer]
        assert grazing["wavelength"] == pytest.approx(GALAXI["wavelength_angstrom"] / 10)
        return [dict(_row(q, intensity, sigma, 1, "sphere", 4., 14., 1.), scattering="four-term DWBA on film on Si")]

    session = open_headless_session(DATA / GALAXI["frame"], saved_profiles=False)
    try:
        session.catalog.fitter = fitter
        report = session.run_pipeline(PipelineOptions(technique="gisaxs", geometry=_geometry(),
                                                      incidence_deg=GALAXI["incidence_deg"], substrate=substrate))
    finally:
        session.close()
    fit = report["gisaxs"]["fit"]
    assert fit["substrate_level"] == 4 and fit["substrate"] == substrate
    assert any(item["why"] == "four-term DWBA on film on Si" for item in report["decisions"])


def test_automatic_fit_handoff_and_replay_keep_the_layer():
    from src.gimap.features.assistant.application.gisaxs_tools import _native, _solution
    from src.gimap.features.assistant.presentation.guided_details import candidate_row
    from src.gimap.features.fitting.application.single_fit import model_from_solution
    from src.gimap.features.fitting.bootstrap import create_curve_fit
    from src.gimap.features.fitting.domain.cut_model import population_terms
    from src.gimap.features.fitting.domain.substrates import Layer, Substrate

    q = np.linspace(.02, 2., 150)
    base = Substrate("Si", layers=(Layer(92., 1.6e-6, 2.1e-9),))
    geometry = base.geometry(.105, .4, .15)
    y = 1e5 * population_terms("sphere", q, radius=4.5, radius_spread=.15, distance=16.,
                              distance_spread=.19, geometry=geometry)[0] + 3.
    rows = create_curve_fit()(q / 10., y, np.sqrt(y), components=(1,), substrate=base.to_dict(),
                              grazing={"wavelength": .105, "alpha_i": .4, "alpha_f": .15})
    row = rows[0]
    assert row["diagnostics"]["substrate_level"] == 4 and row["scattering"].startswith("four-term DWBA on")
    assert row["reproduction"]["options"]["substrate"] == base.to_dict()
    fit = {"native": [_native(row)], "solutions": [_solution(row)]}
    model, deviation = model_from_solution(candidate_row(fit, 0))
    assert model.substrate == base and deviation < 1e-10
    assert model.components[0].value("R") == pytest.approx(4.5, rel=.01)
