"""GIWAXS indexing: cells, CIF files, scattering factors, reflections, fibre spots with refraction, the assignment of
rings and the pole figure (``analyze/domain/crystal``, ``analyze/application/indexing.py`` and ``pole_figure.py``).

- LaB₆ powder rings fall on its standard d-spacings (and on GIMaP's calibrant lines), with their multiplicities;
- MAPbI₃ (I4/mcm) from a CIF: the extinctions of the space group, the positions of the strong rings, the fibre
  spots of a (110) texture at the interplanar angles, and the rings of the public P08 frame assigned to it and
  to PbI₂;
- refraction: the transmitted channel falls on the hand example of Physics and Methods 2.6, the reflected one
  2 k_z,i above on a corrected map; the detector position of a spot gives back its q;
- the pole figure fit recovers the FWHM, the isotropic fraction and f of a synthetic I(χ) within its errors,
  with a missing wedge; the Maier–Saupe fit recovers m.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.gimap.features.analyze.application.indexing import (
    IndexingSettings,
    PhaseSpec,
    assign_peaks,
    detector_positions,
    phase_from_cif,
    phase_overlay,
    table_rows,
)
from src.gimap.features.analyze.application.pole_figure import pole_figure, pole_figure_csv
from src.gimap.features.analyze.domain.crystal import (
    COEFFICIENTS,
    Candidate,
    Phase,
    UnitCell,
    assign,
    centring_operations,
    f0,
    fibre_spots,
    fit_pole_figure,
    fold_profile,
    observed_qz,
    parse_cif,
    parse_operation,
    powder_rings,
    reflections,
    systematically_absent,
)
from src.gimap.features.analyze.domain.crystal import pole_models
from src.gimap.features.analyze.domain.crystal.scattering import ATOMIC_NUMBER
from src.gimap.shared.geometry import DetectorGeometry
from src.gimap.shared.geometry.q_inverse import q_of_positions

DATA = Path(__file__).parent / "data" / "external"
MANIFEST = json.loads((DATA / "manifest.json").read_text(encoding="utf-8")) if (DATA / "manifest.json").exists() else {}

CRYSTAL_DATA = Path(__file__).parent / "data" / "crystal"
MAPI_CIF = (CRYSTAL_DATA / "MAPbI3_tetragonal.cif").read_text(encoding="utf-8")
PBI2_CIF = (CRYSTAL_DATA / "PbI2.cif").read_text(encoding="utf-8")
LAB6_CIF = "\n".join([
    "data_LaB6", "_cell_length_a 4.1569", "_cell_length_b 4.1569", "_cell_length_c 4.1569",
    "_symmetry_space_group_name_H-M 'P 1'",
    "loop_", "_atom_site_label", "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z",
    "La1 0 0 0",
    *[f"B{index} {x} {y} {z}" for index, (x, y, z) in enumerate((
        (0.1993, 0.5, 0.5), (0.8007, 0.5, 0.5), (0.5, 0.1993, 0.5), (0.5, 0.8007, 0.5),
        (0.5, 0.5, 0.1993), (0.5, 0.5, 0.8007)), 1)],
    "",
])


def _phase(text: str) -> Phase:
    structure = parse_cif(text)
    from src.gimap.features.analyze.domain.crystal import parse_operations

    return Phase(structure.name, structure.cell, parse_operations(structure.operations), structure.atoms)


# -- scattering factors, operations, cells -------------------------------------------------

def test_every_scattering_factor_row_sums_to_its_atomic_number() -> None:
    for element, row in COEFFICIENTS.items():
        assert abs(sum(row[0:8:2]) + row[8] - ATOMIC_NUMBER[element]) < 0.06, element
        values = f0(element, np.linspace(0.0, 12.0, 50))
        assert np.all(np.diff(values) < 0), element  # falls with q
    assert f0("C", 0.5 * 4 * math.pi) == pytest.approx(1.69, abs=0.01)  # sin θ/λ = 0.5 Å⁻¹ (ITC Table 6.1.1.1)
    assert f0("Po", 0.0) == pytest.approx(84.0, abs=0.1)  # not tabulated: Bi scaled by Z


def test_another_phase_in_the_extracted_band_excludes_an_otherwise_unambiguous_assignment():
    from src.gimap.features.analyze.application.crystal_orientation import frame_crystal_orientation

    phases = [PhaseSpec("A", (6., 6., 6., 90., 90., 90.)),
              PhaseSpec("B", (5.2, 5.2, 5.2, 90., 90., 90.))]
    q = 2 * math.pi / 6.
    peak = {"q": q, "fwhm": .01, "q_err": .0001, "q_window": [q - .02, 1.24]}
    found = assign_peaks([peak], phases)
    assigned, = found["assignments"]
    assert assigned["phase"] == "A" and assigned["overlapping_phases"] == ["B"]
    assert not assigned["alternatives"]  # the other phase is outside the radial tolerance
    frame = SimpleNamespace(kind="giwaxs", geometry=object(), reduction=object())
    a, b = frame_crystal_orientation(frame, phases, assignment=found)
    assert not a["ok"] and a["excluded_rings"][0]["overlapping_phases"] == ["B"]
    assert not b["ok"] and not b["rings"]


def test_symmetry_operations_are_read_in_the_usual_notations() -> None:
    operation = parse_operation("-x+1/2, y, -Z")
    assert operation.rotation.tolist() == [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
    assert operation.translation.tolist() == [0.5, 0.0, 0.0]
    assert parse_operation("x-y,x,1/2+z").rotation.tolist() == [[1, -1, 0], [1, 0, 0], [0, 0, 1]]
    assert parse_operation("'y+0.25,-x,z'").translation.tolist() == [0.25, 0.0, 0.0]
    for bad in ("x,y", "2x,y,z", "x,y,w"):
        with pytest.raises(ValueError):
            parse_operation(bad)
    body_centred = centring_operations("I")
    assert systematically_absent([(1, 0, 0), (1, 1, 0), (1, 1, 1)], body_centred).tolist() == [True, False, True]


def test_lab6_rings_fall_on_its_standard_d_spacings_and_on_the_calibrant_lines() -> None:
    from src.gimap.features.calibration.domain.standards import LAB6_Q

    lab6 = Phase("LaB6", UnitCell(4.1569, 4.1569, 4.1569), centring_operations("P"))
    rings = powder_rings(reflections(lab6, 5.0), lab6.operations)
    standard_d = (4.1569, 2.9394, 2.4000, 2.0785, 1.8590, 1.6971, 1.4697, 1.3856, 1.3145)
    assert [ring.d for ring in rings[:9]] == pytest.approx(standard_d, abs=1e-4)
    assert [ring.multiplicity for ring in rings[:6]] == [6, 12, 8, 6, 24, 24]
    assert [ring.label for ring in rings[:6]] == ["100", "110", "111", "200", "210", "211"]
    assert rings[7].label == "221, 300" or rings[7].label == "300, 221"  # N = 9: two families on one ring
    np.testing.assert_allclose([ring.q for ring in rings[:len(LAB6_Q)]], LAB6_Q[:len(rings)], atol=1e-6)
    # From a CIF with every atom listed (P1): the same rings, intensities from La and B.
    from_cif = _phase(LAB6_CIF)
    assert from_cif.cell.a == pytest.approx(4.1569) and len(from_cif.atoms) == 7
    found = reflections(from_cif, 4.0)
    assert found.with_atoms and np.all(found.f2 > 0)
    assert [ring.d for ring in powder_rings(found)[:6]] == pytest.approx(standard_d[:6], abs=1e-4)


def test_mapbi3_from_a_cif_has_the_extinctions_of_i4mcm_and_its_strong_rings() -> None:
    structure = parse_cif(MAPI_CIF)
    assert structure.cell.as_tuple() == pytest.approx((8.874, 8.874, 12.671, 90, 90, 90))
    assert len(structure.operations) == 32 and structure.space_group == "I 4/m c m"
    assert [atom.element for atom in structure.atoms] == ["Pb", "I", "I"]
    assert structure.atoms[0].b_iso == pytest.approx(8 * math.pi ** 2 * 0.02)
    phase = _phase(MAPI_CIF)
    absent = systematically_absent([(1, 1, 1), (1, 0, 1), (1, 0, 0), (0, 0, 1), (1, 1, 3), (3, 0, 1)], phase.operations)
    assert absent.all()
    present = [(1, 1, 0), (0, 0, 2), (1, 1, 2), (2, 0, 0), (2, 1, 1), (2, 2, 0), (0, 0, 4), (3, 1, 0), (1, 1, 4)]
    assert not systematically_absent(present, phase.operations).any()
    rings = {ring.label: ring for ring in powder_rings(reflections(phase, 2.5), phase.operations)}
    assert rings["110"].q == pytest.approx(2 * math.pi * math.sqrt(2) / 8.874, abs=1e-9)
    assert rings["002"].q == pytest.approx(2 * math.pi * 2 / 12.671, abs=1e-9)
    assert rings["110"].multiplicity == 4 and rings["002"].multiplicity == 2
    strongest = sorted(rings.values(), key=lambda ring: -ring.intensity)[:4]
    assert {ring.label for ring in strongest} <= {"220", "004", "310", "114", "222", "110"}


def test_fibre_spots_of_a_110_texture_lie_at_the_interplanar_angles() -> None:
    phase = _phase(MAPI_CIF)
    normal = phase.cell.plane_normal((1, 1, 0))
    spots = {spot.label: spot for spot in fibre_spots(reflections(phase, 2.1), normal, phase.operations)}
    assert spots["110"].q_parallel == pytest.approx(0.0, abs=1e-9) and spots["110"].chi_deg == pytest.approx(0.0)
    assert spots["002"].chi_deg == pytest.approx(90.0)  # c lies in the substrate plane
    expected = math.degrees(math.acos(float(phase.cell.plane_normal((1, 1, 2)) @ normal)))
    assert spots["112"].chi_deg == pytest.approx(expected, abs=1e-6)
    for spot in spots.values():
        assert spot.q_parallel ** 2 + spot.qz ** 2 == pytest.approx(spot.q ** 2) and spot.qz >= 0


# -- refraction and the detector -----------------------------------------------------------

def test_the_transmitted_channel_reproduces_the_hand_example_and_the_reflected_one_lies_above() -> None:
    k, incidence, critical = 2 * math.pi, 0.2, 0.15  # λ = 1 Å (Physics and Methods 2.6)
    true_qz = 0.122923
    kinematic = observed_qz(true_qz, k, incidence, critical, 0.0, "T")
    assert kinematic == pytest.approx(2 * math.pi * (math.sin(math.radians(1.0)) + math.sin(math.radians(0.2))), abs=2e-6)
    assert observed_qz(true_qz, k, incidence, critical, critical, "T") == pytest.approx(true_qz, abs=1e-9)
    kzi = k * math.sqrt(math.sin(math.radians(incidence)) ** 2 - math.sin(math.radians(critical)) ** 2)
    assert observed_qz(true_qz, k, incidence, critical, critical, "R") == pytest.approx(true_qz + 2 * kzi, abs=1e-9)
    assert math.isnan(observed_qz(0.5 * kzi, k, incidence, critical, 0.0, "T"))  # scattered into the substrate


def test_a_spot_on_the_detector_gives_back_its_q() -> None:
    geometry = DetectorGeometry(172e-6, 172e-6, 0.3, 300.5, 900.25, 1.0, 0.15, critical_angle_deg=0.1)
    q_parallel, qz = np.array([1.2, -0.8, 0.4]), np.array([0.3, 1.1, 0.05])
    x, y = detector_positions(geometry, q_parallel, qz)
    q_y, q_z = q_of_positions(geometry, x, y)
    np.testing.assert_allclose(q_z, qz, atol=1e-9)
    k = geometry.wavevector_inv_angstrom
    from src.gimap.shared.geometry.q_inverse import exit_sine_of_qz

    sin_f = exit_sine_of_qz(geometry, q_z)
    cos_f, cos_i = np.sqrt(1 - sin_f ** 2), math.cos(geometry.incidence_rad)
    q_x = k * (cos_f * np.sqrt(1 - (q_y / (k * cos_f)) ** 2) - cos_i)
    np.testing.assert_allclose(np.copysign(np.hypot(q_x, q_y), q_y), q_parallel, atol=1e-9)


def test_the_overlay_of_a_fibre_phase_has_both_channels_and_a_table() -> None:
    geometry = DetectorGeometry(172e-6, 172e-6, 0.25, 600.0, 1100.0, 1.0, 0.2)
    spec = PhaseSpec("film", (8.874, 8.874, 12.671, 90, 90, 90), "I", orientation="plane", indices=(1, 1, 0))
    plain = phase_overlay(spec, IndexingSettings(q_max=2.1), geometry)
    shifted = phase_overlay(spec, IndexingSettings(q_max=2.1, refraction=True, critical_angle_deg=0.15, reflected=True),
                            geometry)
    assert {marker.channel for marker in plain.markers} == {"T"}
    assert {marker.channel for marker in shifted.markers} == {"T", "R"}
    on_rod = [marker for marker in plain.markers if abs(marker.q_parallel) < 1e-9]
    assert on_rod and not any(marker.seen for marker in on_rod)  # the missing wedge
    rows = table_rows(shifted)
    assert rows[0]["intensity"] <= 100.0 and max(row["intensity"] for row in rows) == pytest.approx(100.0)
    assert all(row["qz_R"] is None or row["qz_R"] > row["qz_T"] for row in rows if row["qz_T"] is not None)


# -- assignment ------------------------------------------------------------------------------

def test_a_ring_is_assigned_to_the_nearest_reflection_within_its_window() -> None:
    candidates = [Candidate("A", "110", 1.000), Candidate("B", "001", 1.004), Candidate("A", "200", 1.414)]
    near = assign(1.0012, 2e-4, 0.010, candidates)
    assert near.match.label == "110" and near.delta_q == pytest.approx(0.0012)
    assert near.tolerance == pytest.approx(0.005) and [item.phase for item in near.alternatives] == ["B"]
    assert near.delta_sigma == pytest.approx(6.0) and near.relative_d == pytest.approx(1.000 / 1.0012 - 1)
    record = near.record()
    assert record["ambiguous"] and record["alternatives"][0]["relative_d"] == pytest.approx(1.004 / 1.0012 - 1)
    assert not assign(1.2, 1e-4, 0.01, candidates).assigned
    sharp = assign(1.4146, 4e-4, 0.001, candidates)  # 3σ = 0.0012 is the larger window
    assert sharp.assigned and sharp.tolerance == pytest.approx(0.0012)
    assert not sharp.record()["ambiguous"]
    coincident = Candidate("cubic", "300, 221", 3, families=("300", "221"))
    assert assign(3.0, 0.001, 0.01, [coincident]).record()["ambiguous"]


def test_report_names_the_radial_axis_and_every_unassigned_peak():
    from src.gimap.features.assistant.application.pipeline_assignment import assignment_lines

    geometry = DetectorGeometry(172e-6, 172e-6, 0.25, 600.0, 1100.0, 1.0, 0.2)
    spec = PhaseSpec("cubic", (5, 5, 5, 90, 90, 90), orientation="powder")
    peaks = [{"q": q, "q_err": 0.0003, "fwhm": 0.02} for q in (1.258, 1.32)]
    for critical, basis in ((0.0, "kinematic |q|"), (0.1, "transmitted-channel |q|")):
        assignment = assign_peaks(peaks, [spec], geometry=geometry.with_critical_angle(critical))
        assert assignment["unassigned_q"] == [1.32]
        expected = geometry.wavevector_inv_angstrom * math.sin(geometry.incidence_rad)
        assert assignment["q_axis"]["k_sin_alpha_i_inv_A"] == pytest.approx(expected)
        report = "\n".join(assignment_lines({"phase_assignment": assignment}))
        assert basis in report and "transmitted and reflected channels" in report
        assert "Δd/d = " in report and "Δq = " in report and "1.3200 Å⁻¹ unassigned" in report


@pytest.mark.skipif(not MANIFEST, reason="public test data not downloaded (tests/data/external)")
def test_the_rings_of_the_public_p08_frame_are_mapbi3_and_pbi2(tmp_path: Path) -> None:
    from src.gimap.features.analyze.domain import GiwaxsSettings
    from src.gimap.features.analyze.domain.giwaxs import reduce_giwaxs
    from src.gimap.features.assistant.domain import Profile, find_peaks, parse_poni, reliable
    from src.gimap.shared.detector_io import load_detector_image

    info = MANIFEST["giwaxs_p08_mapi"]
    poni = parse_poni((DATA / info["poni"]).read_text(encoding="utf-8"))
    image = load_detector_image(DATA / info["frame"])
    geometry = DetectorGeometry(poni.pixel_size_x_m, poni.pixel_size_y_m, poni.distance_mm / 1e3,
                                poni.beam_center_x_px, poni.beam_center_y_px, poni.wavelength_angstrom,
                                info["incidence_deg"])
    reduction = reduce_giwaxs(image.data, np.isfinite(image.data) & ~image.mask, geometry,
                              GiwaxsSettings(chi_q_window=(0.982, 1.012)))
    radial = reduction.curve("radial")
    search = find_peaks(Profile.of(radial.x, radial.intensity, radial.sigma, radial.pixels))
    (tmp_path / "mapi.cif").write_text(MAPI_CIF, encoding="utf-8")
    (tmp_path / "pbi2.cif").write_text(PBI2_CIF, encoding="utf-8")
    phases = [phase_from_cif(tmp_path / "mapi.cif"), phase_from_cif(tmp_path / "pbi2.cif")]
    peaks = [{"q": peak.q, "q_err": peak.q_err, "fwhm": peak.fwhm} for peak in search.peaks]
    found = {round(item["q"], 3): item for item in assign_peaks(peaks, phases)["assignments"]}
    strongest = sorted((peak for peak in search.peaks if reliable(peak)), key=lambda peak: -peak.area)[:3]
    for peak in strongest:  # 0.997, 1.990 and 2.224 Å⁻¹: MAPbI₃ 110, 004 (220), 114 (310)
        item = found[round(peak.q, 3)]
        assert item["phase"] == "MAPbI3 (tetragonal)" and item["hkl"] in {"110", "004", "114", "002", "220", "310"}
        assert abs(item["delta_q"]) <= item["tolerance"] < 0.012
    pbi2 = min(found.values(), key=lambda item: abs(item["q"] - 0.899))
    assert pbi2["phase"] == "PbI2" and pbi2["hkl"] == "001"
    # The ring at 0.997 Å⁻¹: a fit with a small oriented fraction along the normal, its f within [−½, 1].
    result = pole_figure(SimpleNamespace(reduction=reduction, geometry=geometry, corrections=SimpleNamespace(lorentz=False)),
                         chi0_deg=None, chi_max=80.0)
    assert result.fit.ok and 0.0 < result.fit.values["oriented_fraction"] < 0.2
    assert result.fit.values["chi0_deg"] == 0.0 and -0.5 <= result.fit.values["herman"] <= 1.0
    assert "pole figure" in pole_figure_csv(result, {"source": "P08 public frame"}).splitlines()[2]


# -- pole figure -----------------------------------------------------------------------------

def _synthetic(model: str, fraction: float, width: float, chi0: float, *, seed: int = 3):
    chi = np.arange(-89.5, 90.0, 1.0)
    rho = pole_models.pole_density(np.radians(np.abs(chi)), model, fraction, width, chi0)
    background, scale = 40.0, 5000.0
    raw = background + scale * rho
    sigma = np.sqrt(raw)
    raw = raw + np.random.default_rng(seed).normal(0.0, sigma)
    raw[np.abs(chi) < 4.0] = np.nan  # the missing wedge
    return chi, raw, sigma, background


def test_the_pole_figure_fit_recovers_width_isotropic_fraction_and_f() -> None:
    chi, intensity, sigma, background = _synthetic("gaussian", 0.35, 12.0, 0.0)
    profile = fold_profile(chi, intensity, sigma, background)
    assert not profile.measured[:4].any() and profile.measured[4:].all()
    fit = fit_pole_figure(profile, "gaussian", chi0_deg=0.0, chi_max=90.0)
    values, errors = fit.values, fit.errors
    assert fit.ok and fit.chi2_red < 2.0
    assert abs(values["fwhm_deg"] - 12.0) < 3 * errors["fwhm_deg"] + 0.05
    assert abs(values["fraction_iso"] - 0.35) < 3 * errors["fraction_iso"] + 0.003
    true_f = pole_models.hermans("gaussian", 0.35, 12.0, 0.0)
    assert abs(values["herman"] - true_f) < 3 * errors["herman"] + 0.003
    assert values["oriented_fraction"] == pytest.approx(1.0 - values["fraction_iso"])
    assert profile.herman_measured() < true_f - 0.03  # the measured bins alone miss the wedge


def test_a_tilted_texture_and_the_maier_saupe_form_are_recovered() -> None:
    chi, intensity, sigma, background = _synthetic("gaussian", 0.2, 15.0, 40.0)
    fit = fit_pole_figure(fold_profile(chi, intensity, sigma, background), "gaussian", chi0_deg=None, chi_max=90.0)
    assert abs(fit.values["chi0_deg"] - 40.0) < 3 * fit.errors["chi0_deg"] + 0.05
    assert abs(fit.values["fwhm_deg"] - 15.0) < 3 * fit.errors["fwhm_deg"] + 0.05
    chi, intensity, sigma, background = _synthetic("maier_saupe", 0.2, 30.0, 0.0, seed=5)
    fit = fit_pole_figure(fold_profile(chi, intensity, sigma, background), "maier_saupe", chi0_deg=0.0, chi_max=90.0)
    assert abs(fit.values["width"] - 30.0) < 3 * fit.errors["width"] + 0.1
    assert fit.values["fwhm_deg"] == pytest.approx(pole_models.fwhm_deg("maier_saupe", fit.values["width"]))


def test_a_flat_profile_is_refused_with_its_reason() -> None:
    chi = np.arange(-89.5, 90.0, 1.0)
    fit = fit_pole_figure(fold_profile(chi, np.full(chi.shape, 10.0), np.ones(chi.shape), 10.0))
    assert not fit.ok and "3 σ" in fit.message
