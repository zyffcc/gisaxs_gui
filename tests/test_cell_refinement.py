"""Known-cell recovery, absolute errors, degeneracies, the transmitted branch and fibre strains."""

from dataclasses import replace
import json

import numpy as np
import pytest

from src.gimap.features.analyze.domain.crystal.assignment import Candidate, assign
from src.gimap.features.analyze.domain.crystal.cell import UnitCell
from src.gimap.features.analyze.domain.crystal.cell_observations import corrected
from src.gimap.features.analyze.domain.crystal.fibre import TRANSMITTED, observed_qz
from src.gimap.features.analyze.domain.crystal.refinement import (
    CellPeak, Refraction, refine_cell, refine_fibre_strain, select_peaks, system_for_cell,
)
from src.gimap.features.analyze.domain.crystal.reflections import ReflectionFamily
from tools.benchmarks.cell_coverage import coverage


def test_noisy_cells_recover_all_free_parameters_and_one_sigma_coverage():
    summary = coverage(64)
    fractions = []
    for case in summary.values():
        for param in case["parameters"].values():
            assert abs(param["truth"] - param["mean"]) < .55 * param["mean_sigma"]
            assert .42 < param["coverage"] < .94
            fractions.append(param["coverage"])
    assert .65 < np.mean(fractions) < .82


def test_singular_and_exactly_determined_designs_and_absolute_errors():
    cell = UnitCell(5., 5., 7.)
    hkls = [(0, 0, i) for i in range(1, 6)]
    peaks = [CellPeak(h, q, .001) for h, q in zip(hkls, cell.q_of(hkls))]
    assert "independently" in refine_cell(cell, "tetragonal", peaks)["reason"]
    assert "Fewer" in refine_cell(cell, "tetragonal", peaks[:1])["reason"]
    basal = CellPeak((1, 0, 0), 2 * np.pi / cell.a, .002)
    result = refine_cell(cell, "tetragonal", [basal, peaks[0]])
    assert result["ok"] and result["dof"] == 0 and result["chi2_red"] is None
    assert not result["zero_shift_fitted"] and result["zero_shift_error"] is None
    assert result["errors"]["a"] == pytest.approx(cell.a * basal.sigma / basal.q)
    assert result["errors"]["c"] == pytest.approx(cell.c * peaks[0].sigma / peaks[0].q)
    relvar = (2 * basal.sigma / basal.q)**2 + (peaks[0].sigma / peaks[0].q)**2
    assert result["errors"]["V"] == pytest.approx(cell.volume * np.sqrt(relvar))
    # Four measurements but only two distinct q centres. Cell is determined, shift is not.
    assert not refine_cell(cell, "tetragonal", [basal, peaks[0]] * 2)["zero_shift_fitted"]
    typed = replace(basal, q=np.float32(basal.q), sigma=np.float32(basal.sigma), source_index=np.int64(2))
    bad = replace(typed, q=np.nan)
    assert json.loads(json.dumps(refine_cell(cell, "tetragonal", [typed, peaks[0], bad]),
                                 allow_nan=False))["excluded"][0]["q"] is None


def test_chi_square_inflates_covariance_and_strain_against_fixed_cif():
    cell = UnitCell(4., 4., 4.)
    hkls = [(i, 0, 0) for i in range(1, 5)]
    measured = UnitCell(4.04, 4.04, 4.04).q_of(hkls) + .002
    measured += np.array([1, -3, 3, -1]) * .001
    fits = [refine_cell(cell, "cubic", [CellPeak(h, q, sigma) for h, q in zip(hkls, measured)])
            for sigma in (.001, .010)]
    fit, large_sigma = fits
    json.dumps(fit, allow_nan=False)
    assert fit["chi2_red"] == pytest.approx(10.)
    assert fit["strain"]["a"] == pytest.approx(.01)
    assert fit["strain_errors"]["a"] == fit["errors"]["a"] / 4
    assert fit["zero_shift"] == pytest.approx(.002)
    assert fit["errors"]["a"] / large_sigma["errors"]["a"] == pytest.approx(np.sqrt(10) / 10)
    for record in fit["residuals"]:
        assert record["q_corrected"] - record["q_calc"] == pytest.approx(record["residual"])


def test_selection_uses_summed_family_intensity_roundtrip_and_keeps_measured_coordinates():
    weak = ReflectionFamily((0, 0, 2), 2., ((0, 0, 2), (0, 0, -2)))
    strong = ReflectionFamily((1, 1, 0), 9., ((1, 1, 0), (1, -1, 0)))
    candidates = [Candidate("p", "002", 1., components=(weak,)),
                  Candidate("p", "110", 1.01, components=(strong,))]
    item = assign(1.0001, .0003, .03, candidates)
    assert item.match.label == "002"
    records = [dict(item.record(), q_parallel=.8, qz=.6)]
    for inputs in ([item], records):
        [peak], excluded = select_peaks("p", inputs, with_atoms=True)
        assert not excluded and peak.hkl == strong.representative and peak.equivalents == strong.members
        assert not select_peaks("p", inputs)[0]
    assert select_peaks("p", records, with_atoms=True)[0][0].qz == .6
    assert "tentative" in select_peaks("p", [dict(records[0], tentative=True)], with_atoms=True)[1][0]["reason"]
    joined = Candidate("p", "002, 110", 1., components=(weak, strong))
    assert select_peaks("p", [assign(1., .001, .03, [joined]).record()], with_atoms=True)[0][0].hkl == (1, 1, 0)
    extra = Candidate("other", "010", 1.002, components=(weak,))
    unresolved = assign(1., .001, .03, [joined, extra]).record()
    for name in ("p", "other"):
        assert select_peaks(name, [unresolved], with_atoms=True)[1][0]["reason"].startswith("Reflections from")


def test_radiating_transmitted_spot_refraction_and_propagation_near_horizon():
    optics = Refraction(5., .18, .15)
    true_z, true_par = .012, .9
    z_map = float(observed_qz(true_z, optics.k, optics.incidence_deg, optics.critical_deg, 0., TRANSMITTED))
    peak = CellPeak((1, 0, 0), np.hypot(z_map, true_par), 1.e-4, true_par, z_map)
    q, sigma, direction = corrected(peak, optics)
    assert q == pytest.approx(np.hypot(true_z, true_par))
    assert direction[1] == pytest.approx(true_z / q)
    dq = peak.sigma * .001
    derivative = (corrected(replace(peak, q=peak.q+dq), optics)[0]
                  - corrected(replace(peak, q=peak.q-dq), optics)[0]) / (2*dq)
    assert sigma == pytest.approx(derivative * peak.sigma, rel=1e-7)
    no_angle = replace(peak, q_parallel=None, qz=None)
    result = refine_cell(UnitCell(5, 5, 5), "cubic", [no_angle], refraction=optics)
    assert not result["ok"] and "direction" in result["excluded"][0]["reason"]
    assert corrected(no_angle, replace(optics, map_critical_deg=.15))[:2] == (peak.q, peak.sigma)
    dark = replace(peak, qz=.002)
    assert "propagating" in refine_cell(UnitCell(5, 5, 5), "cubic", [dark], refraction=optics)["excluded"][0]["reason"]


def test_directional_strain_selects_equivalent_pole_and_requires_both_directions():
    ref = UnitCell(5., 5., 5.)
    ep, ez = .012, -.007
    axis, peaks = (0, 0, 1), []
    for h in ((1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 1), (2, 0, 2)):
        vector = 2 * np.pi * np.asarray(h) / ref.a
        par, z = np.hypot(*vector[:2]) / (1 + ep), vector[2] / (1 + ez)
        family = ((1, 0, 0), (0, 1, 0), (0, 0, 1)) if sum(h) == 1 else (h,)
        peaks.append(CellPeak(family[0], np.hypot(par, z), .0005, par, z, equivalents=family))
    result = refine_fibre_strain(ref, axis, peaks)
    assert result["ok"] and result["axis_hkl"] == list(axis)
    assert result["values"] == pytest.approx({"in_plane": ep, "out_of_plane": ez}, abs=1e-9)
    assert result["errors"]["in_plane"] > 0 and result["errors"]["out_of_plane"] > 0
    assert result["residuals"][1]["hkl"] == [0, 0, 1]
    assert not refine_fibre_strain(ref, axis, [peaks[0]] * 3)["ok"]
    parallel_only = [replace(p, q_parallel=p.q, qz=0.) for p in peaks]
    assert not refine_fibre_strain(ref, axis, parallel_only)["ok"]
    assert not refine_fibre_strain(ref, axis, [replace(p, qz=None) for p in peaks])["ok"]
    # The sectors each measure a radius over an aperture, not an invented 2D centre.
    basal = replace(peaks[0], q_parallel=None, qz=None, chi_range_deg=(80., 90.))
    up = replace(peaks[1], q_parallel=None, qz=None, chi_range_deg=(0., 10.))
    sector_fit = refine_fibre_strain(ref, axis, [basal, up])
    assert sector_fit["values"] == pytest.approx(result["values"], abs=1e-9)
    assert sector_fit["residuals"][0]["chi_range_deg"] == (80., 90.)
    assert sector_fit["residuals"][0]["q_parallel"] is None
    for unavailable in ((0., 90.), (50., 55.)):
        rejected = refine_fibre_strain(ref, axis, [replace(up, chi_range_deg=unavailable)])
        assert "sector" in rejected["excluded"][0]["reason"]


def test_system_inference_respects_declared_symmetry_of_an_accidentally_cubic_metric():
    cell = UnitCell(5, 5, 5)
    assert system_for_cell(cell) == "cubic"
    assert system_for_cell(cell, number=1) == "triclinic"
    assert system_for_cell(cell, symbol="P 4") == "tetragonal"
    assert system_for_cell(cell, operations=("x,y,z", "-x,-y,-z")) == "triclinic"
    assert system_for_cell(cell, operations=("x,y,z", "z,x,y", "y,z,x")) == "trigonal"
    peak = CellPeak((1, 0, 0), 2*np.pi/5, .001, equivalents=((1, 0, 0), (0, 1, 0)))
    assert "coincident" in refine_cell(cell, "orthorhombic", [peak])["excluded"][0]["reason"]
