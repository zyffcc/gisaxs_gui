"""Displayed pixel positions and exported .poni rotations for each pyFAI detector orientation."""
import json
from dataclasses import replace

import numpy as np
import pytest

from src.gimap.features.analyze.application.geometry_files import SiblingPoni
from src.gimap.features.analyze.domain.gisaxs import locate_yoneda, vertical_cut
from src.gimap.features.analyze.domain.gisaxs_cuts import row_band_q
from src.gimap.features.analyze.domain.intensity import intensity_factor
from src.gimap.features.analyze.domain.preprocess import mirror_fill
from src.gimap.features.analyze.domain.sources import curve_source_mask
from src.gimap.features.assistant.application.calibration_match import shape_mismatch
from src.gimap.features.assistant.domain import parse_poni
from src.gimap.shared.geometry import InstrumentProfile, grazing_q_map
from src.gimap.shared.geometry.poni import format_poni, read_poni
from src.gimap.shared.geometry.q_inverse import positions_of_q
from src.gimap.shared.geometry.q_mapping import geometry_rays

SHAPE = (31, 47)


def poni_text(orientation=3, *, rotated=True, shape=SHAPE, detector="Detector"):
    config = {"pixel1": 0.0002, "pixel2": 0.00013}
    if shape:
        config["max_shape"] = shape
    if orientation is not None:
        config["orientation"] = orientation
    return "\n".join([
        "poni_version: 2.1", f"Detector: {detector}", f"Detector_config: {json.dumps(config)}",
        "Distance: 0.18", "Poni1: 0.00291", "Poni2: 0.00354", "Wavelength: 1.2e-10",
        f"Rot1: {0.014 if rotated else 0}", f"Rot2: {-0.009 if rotated else 0}", f"Rot3: {0.36 if rotated else 0}",
    ])


@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("orientation", [1, 2, 3, 4])
def test_all_q_components_are_the_native_arrays_with_the_stated_axes_reversed(orientation, rotated):
    text = poni_text(orientation, rotated=rotated)
    stated = read_poni(text)
    geometry = stated.geometry(incidence_deg=0.34)
    native = read_poni(poni_text(None, rotated=rotated)).geometry(incidence_deg=0.34)
    row_step = -1 if orientation in (1, 2) else 1
    col_step = -1 if orientation in (1, 4) else 1
    fields = grazing_q_map(SHAPE, geometry)
    reference = grazing_q_map(SHAPE, native)
    for name in ("qx", "qy", "qz", "sin_exit"):
        expected = getattr(reference, name)[::row_step, ::col_step]
        np.testing.assert_allclose(getattr(fields, name), expected, rtol=0, atol=1e-14)
    x, y = positions_of_q(geometry, fields.qy, fields.qz)
    rows, cols = np.indices(SHAPE)
    np.testing.assert_allclose(x, cols + 0.5, atol=1e-11)
    np.testing.assert_allclose(y, rows + 0.5, atol=1e-11)
    restored = read_poni(format_poni(geometry, shape=SHAPE)).geometry(incidence_deg=0.34)
    assert restored.orientation == orientation
    np.testing.assert_allclose(grazing_q_map(SHAPE, restored).qz, fields.qz, rtol=0, atol=1e-14)
    # The automatic analysis builds the initial profile from a beam centre plus a detector_tilt record.
    legacy = parse_poni(text)
    flat = stated.tilt.flat_geometry().with_beam_center(legacy.beam_center_x_px, legacy.beam_center_y_px)
    saved = InstrumentProfile("synthetic", replace(flat, orientation=3), dropped_tilt=legacy.tilt)
    reused = InstrumentProfile.from_dict(json.loads(json.dumps(saved.to_dict()))).geometry
    assert reused.orientation == orientation and reused.rot3_rad == geometry.rot3_rad


@pytest.mark.parametrize("orientation", [1, 2, 3, 4])
def test_signed_rotated_ray_at_every_pixel_against_pyfai(orientation, tmp_path):
    pyfai = pytest.importorskip("pyFAI")
    path = tmp_path / "tilted.poni"
    text = poni_text(orientation)
    path.write_text(text, encoding="utf-8")
    native = pyfai.load(str(path))
    rows, columns = np.indices(SHAPE, dtype=float)
    beam, down, right = native.calc_pos_zyx(d1=rows, d2=columns)
    rays = geometry_rays(read_poni(text).geometry(), (0, SHAPE[0]), (0, SHAPE[1]))
    np.testing.assert_allclose(rays, [beam, right, -down], atol=2e-15, rtol=0)


def test_flip_dimensions_from_named_detector_or_frame(tmp_path):
    generic = poni_text(4, shape=None)
    with pytest.raises(ValueError, match="detector dimensions"):
        read_poni(generic)
    named = read_poni(poni_text(4, shape=None, detector="Pilatus100k"))
    assert named.shape == (195, 487)
    (tmp_path / "test.poni").write_text(generic, encoding="utf-8")
    find = SiblingPoni()
    for shape in (SHAPE, (31, 93)):
        profile = find(tmp_path / "test.edf", shape)
        assert profile.geometry == read_poni(poni_text(4, shape=shape)).geometry()
    with pytest.raises(ValueError, match="1, 2, 3 or 4"):
        read_poni(poni_text(5))
    legacy = poni_text(None) + "\nOrientation: 2"
    assert read_poni(legacy).geometry().orientation == 2
    info = {"detector": "Pilatus100k", "detector_tilt": named.tilt.to_dict()}
    assert shape_mismatch("test.poni", info, {"shape": (487, 195)})


@pytest.mark.parametrize("orientation", [1, 2, 4])
def test_reversed_image_axes_preserve_the_yoneda_angle_and_intensity_corrections(orientation):
    native = read_poni(poni_text(3, rotated=False)).geometry(incidence_deg=0.34)
    oriented = read_poni(poni_text(orientation, rotated=False)).geometry(incidence_deg=0.34)
    row_step = -1 if orientation in (1, 2) else 1
    col_step = -1 if orientation in (1, 4) else 1
    rows, _ = np.indices(SHAPE)
    image = 5 + 150 * np.exp(-0.5 * ((rows - 6) / 1.4) ** 2)
    valid = np.ones(SHAPE, bool)
    first = locate_yoneda(image, valid, native, exclusion_px=2)
    second = locate_yoneda(image[::row_step, ::col_step], valid, oriented, exclusion_px=2)
    assert first.row == 6.5
    assert second.row == (SHAPE[0] - first.row if row_step < 0 else first.row)
    assert second.alpha_f_deg == pytest.approx(first.alpha_f_deg, abs=1e-12)
    lower, upper = row_band_q(oriented, second.row - 1, second.row + 1)["qz_edges"]
    assert lower < upper
    corrections = dict(solid_angle=True, polarization=0.9, film_thickness_nm=20, attenuation_length_um=1.4)
    expected = intensity_factor(SHAPE, native, **corrections)[::row_step, ::col_step]
    np.testing.assert_allclose(intensity_factor(SHAPE, oriented, **corrections), expected, rtol=1e-7)

    image = image + np.indices(SHAPE)[1]  # asymmetric counts identify the source of each mirrored value
    valid[2:4, 25] = False
    flip = np.s_[::row_step, ::col_step]
    filled = mirror_fill(image, valid, native.beam_center_x_px, geometry=native)
    flipped_fill = mirror_fill(image[flip], valid[flip], oriented.beam_center_x_px, geometry=oriented)
    np.testing.assert_allclose(flipped_fill[0], filled[0][flip], rtol=2e-7)  # float32 interpolation weights
    for flags, other in zip(filled[1:], flipped_fill[1:]):
        np.testing.assert_array_equal(other, flags[flip])
    native_cut = vertical_cut(image, valid, native, center_column=26.5, half_width_px=2)
    col = SHAPE[1] - 26.5 if col_step < 0 else 26.5
    flipped_cut = vertical_cut(image[flip], valid[flip], oriented, center_column=col, half_width_px=2)
    np.testing.assert_allclose(flipped_cut.x, native_cut.x, atol=1e-14)
    np.testing.assert_allclose(flipped_cut.intensity, native_cut.intensity)
    np.testing.assert_array_equal(curve_source_mask(flipped_cut, valid[flip]), curve_source_mask(native_cut, valid)[flip])
