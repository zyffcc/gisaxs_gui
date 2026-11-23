"""Check the normal indexing and automatic-report paths, including directional sector peaks."""

from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from src.gimap.features.analyze.application.cell_refinement import frame_cell_refinement
from src.gimap.features.analyze.application.indexing import IndexingSettings, PhaseSpec, assign_peaks
from src.gimap.features.analyze.domain.crystal.fibre import TRANSMITTED, observed_qz
from src.gimap.features.analyze.domain.models import Curve, Reduction
from src.gimap.shared.geometry import DetectorGeometry
from tools.benchmarks.cell_refinement import public_case


def test_all_positions_are_reassigned_with_refined_metric_and_offset():
    from src.gimap.features.analyze.application.cell_assignments import fit_assignments

    spec = PhaseSpec("Test phase", (6., 6., 8.4, 90., 90., 90.))
    def position(h, k, l):
        return 2*np.pi*np.sqrt((h*h+k*k)/6.018**2 + l*l/8.485**2) + .0015

    hkls = [(0, 0, 1), (1, 0, 0), (1, 0, 1), (1, 1, 1), (2, 1, 0), (1, 0, 2)]
    rows = [{"q": position(*h), "q_err": .0003, "fwhm": .024} for h in hkls]
    rows.append({"q": .5*(position(1,1,0) + position(0,0,2)), "q_err": .0003, "fwhm": .024})
    initial = assign_peaks(rows, [spec])
    assert not initial["assignments"][-2]["assigned"]
    assert initial["assignments"][-1]["hkl"] == "110" and not initial["assignments"][-1]["ambiguous"]
    (fit,), _, _ = fit_assignments([spec], initial)
    assert fit["ok"], fit["reason"]
    assert [row["hkl"] for row in fit["residuals"]] == [list(h) for h in hkls]
    excluded, = fit["excluded"]
    assert {r["hkl"] for r in excluded["candidates"]} == {"110", "002"}
    assert fit["metric_rank"] == 2 and fit["zero_shift"] == pytest.approx(.0015)
    assert fit["zero_shift_error"] > 0
    assert fit["assignment_sensitivity"]["range_A"]["a"] == pytest.approx([6.018, 6.018])
    assert fit["assignment_sensitivity"]["range_A"]["c"] == pytest.approx([8.485, 8.485])


def test_sector_radii_determine_directional_strain_only_with_a_resolved_axis():
    spec = PhaseSpec("Film", (6, 6, 6, 90, 90, 90), "F")
    geometry = DetectorGeometry(.0001, .0001, .2, 128., 128., 1., .2)
    # In a [001]-textured cubic film, {200} has one pole in each aperture.
    # Directional peak centres use independent intensity measurements, while the radial peak is integrated.
    qref = 4*np.pi/6.
    x = np.linspace(1.92, 2.25, 600)
    ip, op = .0015, -.002
    curves = []
    for key, centre, limits in (("radial", qref, (0., 90.)), ("in_plane", qref/(1+ip), (80., 90.)),
                                ("out_of_plane", qref/(1+op), (0., 10.))):
        y = 12. + 500*np.exp(-.5*((x-centre)/.008)**2)
        curves.append(Curve(key, key, x, y, np.ones_like(x), np.full_like(x, 50), "q (Å⁻¹)",
                            region={"chi_deg": limits}))
    analysis = SimpleNamespace(kind="giwaxs", geometry=geometry, reduction=Reduction("giwaxs", tuple(curves), {}))
    assigned = assign_peaks([{"q": qref, "q_err": 2e-4, "fwhm": .02}], [spec], geometry=geometry)
    orientation = {"phase": spec.name, "axes": [[0, 0, 1]], "determined": True}
    [record] = frame_cell_refinement(analysis, [spec], assignment=assigned, orientations=[orientation])
    assert record["values"]["a"] == pytest.approx(6.)
    fibre = record["fibre_strain"]
    assert fibre["ok"], fibre["reason"]
    assert fibre["values"] == pytest.approx({"in_plane": ip, "out_of_plane": op}, abs=1e-5)
    assert all(r["qz"] is None and r["chi_range_deg"] for r in fibre["residuals"])
    orientation["determined"] = False
    [record] = frame_cell_refinement(analysis, [spec], assignment=assigned, orientations=[orientation])
    assert "fibre_strain" not in record


def test_xeuss_three_documented_orders_and_original_cif_reference(tmp_path):
    run = public_case("giwaxs_xeuss_pbte", tmp_path)
    fit = run["rock_salt_hypothesis"]
    assert fit["reference"]["a"] == 6.462
    assert [r["hkl"] for r in fit["residuals"]] == [[1, 1, 1], [2, 0, 0], [2, 2, 0]]
    assert 6.8 < fit["values"]["a"] < 7.1
    assert fit["errors"]["a"] > 0 and fit["strain"]["a"] == pytest.approx(fit["values"]["a"]/6.462-1.)


def test_refraction_reassigns_measured_spots_without_replacing_them_with_radial_peaks():
    spec = PhaseSpec("Cubic film", (5., 5., 5., 90., 90., 90.))
    geometry = DetectorGeometry(.0001, .0001, .2, 40., 40., 1., .2)
    analysis = SimpleNamespace(kind="giwaxs", geometry=geometry, reduction=Reduction("giwaxs", (), {}))
    rows = []
    for indices, cosine in (((1, 0, 0), .015), ((1, 1, 0), .7), ((1, 1, 1), .98)):
        q = float(spec.reference_cell.q_of(indices)[0])
        par = q * np.sqrt(1 - cosine**2)
        z = float(observed_qz(q*cosine, geometry.wavevector_inv_angstrom, .2, .18, 0., TRANSMITTED))
        rows.append({"q": np.hypot(par, z), "q_parallel": par, "qz": z, "q_err": .0003, "fwhm": .04})
    assignment = assign_peaks(rows, [spec], geometry=geometry)
    fit, = frame_cell_refinement(analysis, [spec], IndexingSettings(refraction=True, critical_angle_deg=.18),
                                 assignment=assignment)
    assert fit["ok"] and len(fit["residuals"]) == 3
    assert fit["values"]["a"] == pytest.approx(5.)
    assert [r["before_refraction"]["qz"] for r in fit["residuals"]] == [r["qz"] for r in rows]
    assert fit["q_axis"]["critical_angle_deg"] == .18
    compare = fit["refraction_sensitivity"]
    assert compare["available"] and compare["source"] == "indexing or map setting"
    assert compare["off"]["ok"] and compare["on"]["ok"] and compare["on"]["values"] == fit["values"]
    assert abs(compare["on_minus_off_A"]["a"]) > 1e-5


def test_overlay_refraction_rebins_full_ring_once_on_the_transmitted_map(tmp_path):
    from src.gimap.features.analyze.application import AnalysisRequest, AnalyzeFrame
    from src.gimap.features.analyze.infrastructure.adapters import DetectorIoFrameSource
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
    from tools.benchmarks.cell_series import make_stack

    path, _, _, _, profile, phase, _, _ = make_stack(tmp_path)
    analyze = AnalyzeFrame(DetectorIoFrameSource(), InMemoryInstrumentProfileRepository([profile]))
    request = AnalysisRequest(path, mode="giwaxs", with_map=False)
    raw = analyze(request)
    refracted = analyze(replace(request, corrections=replace(request.corrections, critical_angle_deg=.1)))
    setting = IndexingSettings(refraction=True, critical_angle_deg=.1)
    overlay, = frame_cell_refinement(raw, [phase], setting)
    direct, = frame_cell_refinement(refracted, [phase], setting)
    assert overlay["ok"] and direct["ok"] and overlay["q_axis"]["rebinned_for_refraction"]
    assert not direct["q_axis"]["rebinned_for_refraction"]
    assert overlay["values"] == pytest.approx(direct["values"], rel=1e-8)
