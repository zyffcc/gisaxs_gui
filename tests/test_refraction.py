"""Refraction of q_z in a film (optional): q_z = k(Re√(sin²αf − sin²αc) + Re√(sin²αi − sin²αc)), q∥ unchanged.

Hand case: λ = 1 Å (k = 2π Å⁻¹), αi = 0.2°, αc = 0.15°. For a ray at αf = 1.0° the film's q_z is
2π (√(sin²1° − sin²0.15°) + √(sin²0.2° − sin²0.15°)) = 2π (0.0172549 + 0.0023088) = 0.122923 Å⁻¹, against the
kinematic 2π (sin 1° + sin 0.2°) = 0.131583 Å⁻¹.
"""

from __future__ import annotations

import math

from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.analyze.application import AnalysisRequest, AnalyzeFrame, q_at_position
from src.gimap.features.analyze.application.settings_file import AnalyzeSettings, settings_from_record, settings_record
from src.gimap.features.analyze.domain.corrections import Corrections
from src.gimap.features.analyze.domain.gisaxs import gisaxs_maps, vertical_cut
from src.gimap.features.analyze.domain.giwaxs import giwaxs_maps
from src.gimap.features.analyze.infrastructure.adapters import DetectorIoFrameSource
from src.gimap.features.assistant.application.giwaxs_decisions import refraction_decision
from src.gimap.integrations.state.instrument_profiles import InMemoryInstrumentProfileRepository
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile, grazing_q_region, refracted_qz

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = DetectorGeometry(172e-6, 172e-6, 0.3, 500.0, 900.0, 1.0, 0.2)


def test_the_film_q_z_is_the_hand_computed_one_and_the_rest_of_q_does_not_change() -> None:
    k = 2 * math.pi
    sin_f = math.sin(math.radians(1.0))
    assert refracted_qz(np.array([sin_f]), math.sin(math.radians(0.2)), math.radians(0.15), k)[0] == pytest.approx(
        0.122923, abs=2e-6)
    kinematic = grazing_q_region(GEOMETRY, (0, 900), (0, 1000))
    film = grazing_q_region(GEOMETRY.with_critical_angle(0.15), (0, 900), (0, 1000))
    assert np.array_equal(kinematic.qx, film.qx) and np.array_equal(kinematic.qy, film.qy)
    above = film.sin_exit >= math.sin(math.radians(0.15))  # below αc the film has no q_z of its own
    assert np.all(film.qz[above] <= kinematic.qz[above] + 1e-12)
    point = q_at_position(GEOMETRY.with_critical_angle(0.15), 500.0, GEOMETRY.row_for_exit_angle(1.0))
    assert point["alpha_f_deg"] == pytest.approx(1.0, abs=1e-9)
    assert point["qz"] == pytest.approx(0.122923, abs=2e-6)


def test_no_critical_angle_is_the_kinematic_q_as_before() -> None:
    q = grazing_q_region(GEOMETRY, (0, 900), (0, 1000))
    k, incidence = GEOMETRY.wavevector_inv_angstrom, GEOMETRY.incidence_rad
    assert np.allclose(q.qz, k * (q.sin_exit + math.sin(incidence)), rtol=0, atol=1e-15)
    with pytest.raises(ValueError):
        DetectorGeometry(172e-6, 172e-6, 0.3, 500.0, 900.0, 1.0, 0.2, critical_angle_deg=-0.1)


def test_pixels_below_the_critical_angle_are_left_out_of_the_maps_and_the_vertical_cut() -> None:
    film = GEOMETRY.with_critical_angle(0.15)
    shape = (900, 1000)
    row_c = film.row_for_exit_angle(0.15)
    for maps in (giwaxs_maps(shape, film), gisaxs_maps(shape, film)):
        assert not maps.above_horizon[int(math.ceil(row_c)) + 1:].any()
        assert maps.above_horizon[: int(math.floor(row_c)) - 1, 500].all()
    assert giwaxs_maps(shape, film).above_horizon.sum() < giwaxs_maps(shape, GEOMETRY).above_horizon.sum()
    image = np.ones(shape)
    cut = vertical_cut(image, np.ones(shape, bool), film, center_column=500.0, half_width_px=2.0)
    assert cut.region["rows"][1] == int(math.floor(row_c))
    assert cut.x.min() >= refracted_qz(np.array([math.sin(math.radians(0.15))]), math.sin(film.incidence_rad),
                                       film.critical_rad, film.wavevector_inv_angstrom)[0] - 1e-12


def test_analyze_applies_the_critical_angle_of_the_corrections_and_the_settings_keep_it() -> None:
    folder = ROOT / "tests" / "data" / "external" / "giwaxs_p08_mapi"
    frame = folder / "S121_MAI_A2_00841.tif"
    source = DetectorIoFrameSource()
    shape = source.load(frame, 0).data.shape
    geometry = DetectorGeometry(200e-6, 200e-6, 0.8118, 545.46, 1826.95, 0.6888, 0.075)
    use = AnalyzeFrame(source, InMemoryInstrumentProfileRepository([InstrumentProfile("p08", geometry, None, shape)]))
    plain = use(AnalysisRequest(path=frame, mode="giwaxs", profile_name="p08", with_map=False))
    film = use(AnalysisRequest(path=frame, mode="giwaxs", profile_name="p08", with_map=False,
                               corrections=Corrections(critical_angle_deg=0.2)))
    assert plain.geometry.critical_angle_deg == 0.0 and film.geometry.critical_angle_deg == 0.2
    peaks = []
    for analysis in (plain, film):  # the strongest point of the out-of-plane sector: refraction lowers its q
        curve = analysis.reduction.curve("out_of_plane")
        peaks.append(float(curve.x[int(np.argmax(curve.intensity))]))
    assert peaks[1] < peaks[0]
    record = settings_record(AnalyzeSettings(corrections=Corrections(critical_angle_deg=0.2)))
    assert settings_from_record(record).corrections.critical_angle_deg == 0.2


def test_the_report_says_what_q_is() -> None:
    geometry = {"wavelength_A": 1.0, "incidence_deg": 0.2}
    assert refraction_decision(geometry)["decision"] == "kinematic (no refraction correction)"
    film = refraction_decision({**geometry, "critical_angle_deg": 0.15})
    assert film["decision"] == "q_z corrected for refraction in the film (αc = 0.15°)"
