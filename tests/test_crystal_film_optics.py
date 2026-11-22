"""CIF optical constants, exit fields, and averaging on the measured detector bins."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.gimap.features.analyze.application.indexing import phase_from_cif
from src.gimap.features.analyze.domain.crystal.film_optics import film_optics, exit_transmission, ring_transmission
from src.gimap.shared.geometry import DetectorGeometry
from src.gimap.shared.geometry.q_mapping import refracted_qz


CIFS = Path(__file__).parent / "data" / "external" / "giwaxs_p08_mapi"


@pytest.mark.parametrize("name,populations,rho,delta,beta", [
    ("MAPbI3_reference", {"Pb": 4, "I": 12, "C": 4, "N": 4, "H": 24},
     1040/(8.874**2*12.671), 2.18e-6, 1.27e-7),
    ("PbI2_reference", {"Pb": 1, "I": 2}, 188/(4.557**2*6.979*np.sin(np.pi/3)), 3.13e-6, 2.23e-7),
])
def test_cif_compositions_and_optical_constants_at_18_kev(name, populations, rho, delta, beta):
    phase = phase_from_cif(CIFS / (name + ".cif")).phase()
    record = film_optics(phase, 0.688801)
    assert record["available"]
    assert record["atoms_per_cell"] == populations
    assert record["electron_density_e_per_a3"] == pytest.approx(rho)
    assert record["delta"] == pytest.approx(delta, rel=.004)
    assert record["beta"] == pytest.approx(beta, rel=.004)


def test_unsupported_sum_formula_uses_expanded_sites_including_occupancy():
    phase = phase_from_cif(CIFS / "MAPbI3_reference.cif").phase()
    molecule = film_optics(phase, .688801)
    assert molecule["composition_source"] == "CIF sum formula × Z"
    assert molecule["site_atoms_per_cell"] == {"Pb": 4., "I": 12.}
    partial = replace(phase, formula_sum="(CH3NH3)PbI3")
    sites = film_optics(partial, .688801)
    assert sites["composition_source"] == "symmetry-expanded CIF atom sites"
    assert sites["composition_note"] and sites["atoms_per_cell"] == {"Pb": 4., "I": 12.}
    dilute = film_optics(replace(partial, atoms=tuple(replace(a, occupancy=.5) for a in partial.atoms)), .688801)
    assert dilute["delta"] == pytest.approx(sites["delta"]/2)
    assert dilute["beta"] == pytest.approx(sites["beta"]/2)


def test_unavailable_optics_distinguish_unknown_absorption_from_zero_absorption():
    phase = phase_from_cif(CIFS / "PbI2_reference.cif").phase()
    empty = film_optics(replace(phase, atoms=()), 1.)
    assert not empty["available"] and empty["delta"] is None and empty["beta"] is None
    for missing in (film_optics(phase, 5.), film_optics(replace(phase, atoms=(replace(phase.atoms[0], element="Xe"),)), 1.)):
        assert not missing["available"] and missing["delta"] > 0 and missing["beta"] is None
        assert "neutral-atom Z" in missing["source"]


def test_exit_field_limits_and_yoneda_peak():
    sine_c = .002
    sine = np.array([0., .1*sine_c, sine_c, .05])
    field = exit_transmission(sine, sine_c**2/2, 0.)
    assert field[0] == 0
    assert field[1] == pytest.approx(.04)
    assert field[2] == pytest.approx(4.)
    assert field[3] == pytest.approx(1., abs=.001)
    assert 1 < exit_transmission(sine_c, sine_c**2/2, 2e-7) < field[2]


@pytest.mark.parametrize("map_critical_deg", [0., .05])
def test_folded_field_uses_measured_rays_on_both_q_maps(map_critical_deg):
    geometry = DetectorGeometry(100e-6, 100e-6, .5, 100., 200., 1., .10, critical_angle_deg=map_critical_deg)
    optics = {"delta": 2e-6, "beta": 1e-7}
    s = np.array([.0015, .002, .003, .005])
    qz = refracted_qz(s, np.sin(geometry.incidence_rad), geometry.critical_rad, geometry.wavevector_inv_angstrom)
    chi = np.array([-87.9, -87.2, 87.2, 87.9])
    profile = SimpleNamespace(x=np.array([-87.5, 87.5]), sigma=np.array([4., 2.]))
    actual = ring_transmission(optics, geometry, .5, chi, qz, profile)
    # Equal pixel weights in each signed-bin mean; inverse-variance weights between signs.
    expected = exit_transmission(s, **optics)
    assert actual[87] == pytest.approx((expected[:2].mean() + 4*expected[2:].mean()) / 5)
    assert np.all(np.isfinite(actual))
