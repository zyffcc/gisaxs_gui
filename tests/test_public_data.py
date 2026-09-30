"""GIMaP on public detector frames (tests/data/external, see its README): reading, .poni files,
calibration, GISAXS and GIWAXS reduction against what the sources document."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.analyze.domain.gisaxs import reduce_gisaxs
from src.gimap.features.analyze.domain.giwaxs import reduce_giwaxs
from src.gimap.features.analyze.domain.symmetry import symmetric_center_x
from src.gimap.features.analyze.domain.validity import valid_pixels
from src.gimap.features.assistant.domain import parse_poni
from src.gimap.shared.detector_io import load_detector_image
from src.gimap.shared.geometry import DetectorGeometry

DATA = Path(__file__).parent / "data" / "external"
MANIFEST = json.loads((DATA / "manifest.json").read_text(encoding="utf-8")) if (DATA / "manifest.json").exists() else {}
pytestmark = pytest.mark.skipif(not MANIFEST, reason="public test data not downloaded (tests/data/external)")


def _poni(key: str):
    return parse_poni((DATA / MANIFEST[key]["poni"]).read_text(encoding="utf-8"))


def test_every_public_frame_opens_with_its_gaps_marked() -> None:
    galaxi = load_detector_image(DATA / MANIFEST["gisaxs_galaxi"]["frame"])
    gaps = ~valid_pixels(galaxi.data, galaxi.mask)  # Pilatus module gaps are −1: Analyze's validity rule
    assert galaxi.data.shape == (1043, 981) and 5000 < int(gaps.sum()) < 10000 and galaxi.metadata["stored_dtype"] == "int32"
    agbh = load_detector_image(DATA / MANIFEST["saxs_agbh_pyfai"]["frame"])
    assert agbh.data.shape == (1043, 981) and agbh.metadata["format"] == "edf" and agbh.pixel_size_x_m == 172e-6
    pbte = load_detector_image(DATA / MANIFEST["giwaxs_xeuss_pbte"]["frame"])
    assert pbte.data.shape == (619, 487) and pbte.mask.any()  # the header's Dummy −1.5 ± 0.6
    assert pbte.metadata["header_beam_xy_px"] == (8.3, 590.39) and pbte.metadata["header_distance_m"] == 0.37
    p08 = load_detector_image(DATA / MANIFEST["giwaxs_p08_mapi"]["frame"])
    assert p08.data.shape == (2048, 2048) and (p08.data < 0).sum() > 1000
    assert not p08.mask[p08.data < 0].any() and p08.metadata["stored_dtype"] == "float32"  # negatives are data


def test_poni_files_put_the_direct_beam_where_the_rings_are_centred() -> None:
    agbh = _poni("saxs_agbh_pyfai")
    assert (agbh.beam_center_x_px, agbh.beam_center_y_px) == pytest.approx(MANIFEST["saxs_agbh_pyfai"]["beam_center_px"], abs=0.2)
    image = load_detector_image(DATA / MANIFEST["saxs_agbh_pyfai"]["frame"])
    valid = image.data >= 0
    rows, columns = np.indices(image.data.shape)
    radius = np.round(np.hypot(columns + 0.5 - agbh.beam_center_x_px, rows + 0.5 - agbh.beam_center_y_px)).astype(int)
    sums, counts = np.bincount(radius[valid], image.data[valid]), np.bincount(radius[valid])
    profile = np.where(counts > 20, sums / np.maximum(counts, 1), np.nan)
    distances = []
    for order in range(1, 6):  # AgBh d001 = 58.38 Å, λ = 1 Å as the .poni states
        two_theta = 2 * math.asin(order * 1.0 / (2 * 58.38))
        guess = 1600.0 * math.tan(two_theta) / 0.172
        low, high = int(guess * 0.9), int(guess * 1.1)
        peak = low + int(np.nanargmax(profile[low:high]))
        weights = np.nan_to_num(profile[peak - 3:peak + 4] - np.nanmin(profile[low:high]))
        radius_px = float((weights * np.arange(peak - 3, peak + 4)).sum() / weights.sum())
        distances.append(radius_px * 0.172 / math.tan(two_theta))
    assert max(distances) - min(distances) < 0.002 * np.mean(distances)  # concentric: the centre is right
    p08 = _poni("giwaxs_p08_mapi")
    assert (p08.beam_center_x_px, p08.beam_center_y_px) == pytest.approx((545.5, 1826.9), abs=0.2)
    pbte = _poni("giwaxs_xeuss_pbte")
    assert pbte.tilt_deg > 5 and pbte.notes  # a flat-detector geometry is only approximate here: said so
    assert (pbte.beam_center_x_px, pbte.beam_center_y_px) == pytest.approx((8.3, 590.39), abs=2.0)  # the EDF header


def test_the_agbh_standard_calibrates_to_its_rings() -> None:
    from src.gimap.features.calibration.bootstrap import create_headless_calibration

    result = create_headless_calibration().calibrate(
        str(DATA / MANIFEST["saxs_agbh_pyfai"]["frame"]), standard="agbh", energy_kev=12.398419843, pixel_size_m=172e-6,
    )
    expected = MANIFEST["saxs_agbh_pyfai"]
    assert result["distance_mm"] == pytest.approx(expected["ring_distance_mm_at_stated_wavelength"], rel=0.01)
    assert np.hypot(*(np.array(result["beam_center_px"]) - expected["beam_center_px"])) < 5.0
    # Known limit (2026-09-29): on these partial arcs the fit centre sits ~3 px off, 0.45 % in distance.


def test_galaxi_gisaxs_yoneda_band_and_symmetry_axis() -> None:
    info = MANIFEST["gisaxs_galaxi"]
    image = load_detector_image(DATA / info["frame"])
    valid = valid_pixels(image.data, image.mask)
    geometry = DetectorGeometry(
        info["pixel_size_m"], info["pixel_size_m"], info["distance_m"], *info["beam_center_px"],
        info["wavelength_angstrom"], info["incidence_deg"],
    )
    reduction = reduce_gisaxs(image.data, valid, geometry)
    yoneda = reduction.markers["yoneda"]
    assert yoneda is not None and yoneda.row == pytest.approx(info["yoneda_row"], abs=4)
    assert 0.12 < yoneda.alpha_f_deg < 0.25  # αc of a Si-based sample at 1.34 Å
    low, high = reduction.markers["horizontal_band"]
    symmetry = symmetric_center_x(image.data, valid, (int(low), int(high) + 1), info["beam_center_px"][0])
    assert symmetry.x_px == pytest.approx(info["beam_center_px"][0], abs=1.0)  # BornAgain's documented centre
    horizontal = next(curve for curve in reduction.curves if curve.key == "horizontal")
    assert np.nanmin(horizontal.x) < -0.2 and np.nanmax(horizontal.x) > 0.15


def test_p08_giwaxs_rings_of_mapbi3() -> None:
    from scipy.signal import find_peaks

    info = MANIFEST["giwaxs_p08_mapi"]
    poni = _poni("giwaxs_p08_mapi")
    image = load_detector_image(DATA / info["frame"])
    valid = np.isfinite(image.data) & ~image.mask
    geometry = DetectorGeometry(
        poni.pixel_size_x_m, poni.pixel_size_y_m, poni.distance_mm / 1e3, poni.beam_center_x_px, poni.beam_center_y_px,
        poni.wavelength_angstrom, info["incidence_deg"],
    )
    radial = reduce_giwaxs(image.data, valid, geometry).curves[0]
    x, y = np.asarray(radial.x), np.nan_to_num(np.asarray(radial.intensity))
    # Bins match the pixel spacing (≈ 0.0022 Å⁻¹ here), so the sharp PbI₂ (001) line at 0.899 spans 2–3 bins:
    # its prominence is ≈ 4.8 % of the strongest line.
    found = [float(x[index]) for index in find_peaks(y, prominence=y.max() * 0.04)[0]]
    for q in info["peaks_q"]:
        assert min(abs(peak - q) for peak in found) < 0.01, (q, found)
