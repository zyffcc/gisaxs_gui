"""pyFAI .poni files in and out, where a frame's geometry comes from, the transmission technique and the readout.

* ``shared/geometry/poni``: reading and writing are each other's inverse (the q of every pixel to 1e-12), and the
  assistant's reader gives the numbers it gave before.
* ``analyze/application/geometry_resolution``: a profile picked for frames of another size (another detector) is
  never used silently; the .poni beside the frame is used and named; Use Anyway; a profile saved from that .poni
  wins over the file.
* ``analyze/application/technique``: a calibrant frame (complete rings, no αi) is reduced in transmission, I(χ)
  round the whole ring; an αi above 0 keeps grazing incidence.
* ``analyze/application/readout``, ``batch_geometry``, ``frame_import.other_detector``.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.analyze.application import (
    AnalysisRequest,
    AnalyzeFrame,
    batch_geometry_lines,
    detector_point,
    map_point,
    poni_problem,
    q_at_position,
    read_poni_file,
    sibling_poni,
    write_poni_file,
)
from src.gimap.features.analyze.application.frame_import import ImportSource, other_detector
from src.gimap.features.analyze.application.geometry_resolution import ResolveGeometry
from src.gimap.features.analyze.infrastructure.adapters import DetectorIoFrameSource
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile, best_profile, transmission_map
from src.gimap.shared.geometry.poni import format_poni, read_poni, tilt_from_geometry

EXTERNAL = Path(__file__).resolve().parent / "data" / "external"
PONIS = sorted(EXTERNAL.glob("*/*.poni"))
AGBH = EXTERNAL / "saxs_agbh_pyfai" / "Pilatus1M.edf"
GALAXI = EXTERNAL / "gisaxs_galaxi" / "galaxi_data.tif"
requires_public = pytest.mark.skipif(not AGBH.exists() or not GALAXI.exists(), reason="public test data")


class Store:
    def __init__(self, profiles=()):
        self.profiles = list(profiles)

    def load_all(self):
        return list(self.profiles)

    def find(self, name):
        return next((profile for profile in self.profiles if profile.name == name), None)

    def save(self, profile):
        self.profiles = [p for p in self.profiles if p.name != profile.name] + [profile]

    def match(self, *, detector_name, shape):
        return best_profile(self.profiles, detector_name=detector_name, shape=shape)


# -- .poni in and out ------------------------------------------------------------------------------


@pytest.mark.parametrize("path", PONIS, ids=[path.parent.name for path in PONIS])
def test_a_poni_written_from_its_geometry_reads_back_the_same(path: Path, tmp_path: Path) -> None:
    first = read_poni(path.read_text(encoding="utf-8"))
    geometry = first.geometry(wavelength_angstrom=1.0)
    written = write_poni_file(geometry, tmp_path / "copy.poni", shape=(300, 250))
    again = read_poni_file(written)
    for name in ("distance_m", "poni1_m", "poni2_m", "rot1_rad", "rot2_rad", "rot3_rad", "pixel1_m", "pixel2_m"):
        assert getattr(again.tilt, name) == pytest.approx(getattr(first.tilt, name), rel=1e-12, abs=1e-15)
    assert again.shape == (300, 250) and "poni_version: 2.1" in written.read_text(encoding="utf-8")
    shape = (64, 48)
    q = transmission_map(shape, geometry).q
    assert np.max(np.abs(transmission_map(shape, again.geometry()).q - q)) <= 1e-12


def test_the_assistant_reads_a_poni_as_before() -> None:
    from src.gimap.features.assistant.domain import parse_poni

    for path in PONIS:
        old = parse_poni(path.read_text(encoding="utf-8"))
        new = read_poni(path.read_text(encoding="utf-8"))
        geometry = new.geometry(wavelength_angstrom=1.0)
        assert old.distance_mm == pytest.approx(geometry.distance_m * 1e3, rel=1e-15)
        assert (old.beam_center_x_px, old.beam_center_y_px) == (geometry.beam_center_x_px, geometry.beam_center_y_px)
        assert (old.tilt is not None) == new.tilted


def test_tilt_and_poni_are_each_others_inverse() -> None:
    geometry = DetectorGeometry(75e-6, 75e-6, 0.31, 812.4, 1003.7, 0.95, rot1_rad=0.04, rot2_rad=-0.02, rot3_rad=0.01)
    tilt = tilt_from_geometry(geometry)
    assert tilt.direct_beam_px() == pytest.approx((812.4, 1003.7), abs=1e-9)
    assert tilt.beam_distance_m == pytest.approx(0.31, rel=1e-14)
    text = format_poni(geometry, shape=(2167, 2070), detector="Detector")
    assert '"max_shape": [2167, 2070]' in text and "Wavelength: 9.5e-11" in text


def test_a_poni_of_another_detector_is_said_and_one_without_wavelength_needs_one() -> None:
    poni = read_poni(format_poni(DetectorGeometry(172e-6, 172e-6, 1.0, 10, 10, 1.0), shape=(1043, 981)))
    assert poni_problem(poni, (2048, 2048)) == "The .poni file is for 1043×981 frames; this frame is 2048×2048."
    assert poni_problem(poni, (1043, 981), (75e-6, 75e-6)).startswith("The .poni file has 172 µm pixels")
    assert poni_problem(poni, (1043, 981), (172e-6, None)) is None
    no_wavelength = read_poni("Distance: 0.1\nPoni1: 0.01\nPoni2: 0.02\nDetector: Pilatus1M\n")
    assert no_wavelength.wavelength_angstrom is None
    with pytest.raises(ValueError, match="no wavelength"):
        no_wavelength.geometry()


def test_the_poni_beside_a_frame(tmp_path: Path) -> None:
    frame = tmp_path / "run7_00012.cbf"
    frame.write_bytes(b"")
    assert sibling_poni(frame) is None
    (tmp_path / "LaB6.poni").write_text("x", encoding="utf-8")
    assert sibling_poni(frame).name == "LaB6.poni"  # the only one in the folder
    (tmp_path / "other.poni").write_text("x", encoding="utf-8")
    assert sibling_poni(frame) is None  # two unrelated files: none is taken
    (tmp_path / "run7.poni").write_text("x", encoding="utf-8")
    assert sibling_poni(frame).name == "run7.poni"  # named like the frame


# -- where the geometry comes from ---------------------------------------------------------------------


def _giwaxs_profile() -> InstrumentProfile:
    return InstrumentProfile("Perkin 2048", DetectorGeometry(200e-6, 200e-6, 0.8, 545.5, 1826.9, 0.6888), "Perkin",
                             (2048, 2048))


@requires_public
def test_a_profile_of_another_detector_is_not_used_and_the_poni_beside_the_frame_is(tmp_path: Path) -> None:
    frame = tmp_path / GALAXI.name
    shutil.copy2(GALAXI, frame)
    analyze = AnalyzeFrame(DetectorIoFrameSource(), Store([_giwaxs_profile()]))
    request = AnalysisRequest(frame, profile_name="Perkin 2048", incidence_deg=0.463)
    alone = analyze(request)
    assert alone.geometry is None and alone.resolution.refused_profile == "Perkin 2048"
    assert any("is from another detector, so it is not used" in message for message in alone.messages)
    shutil.copy2(EXTERNAL / "gisaxs_galaxi" / "galaxi_bornagain.poni", tmp_path / "galaxi_bornagain.poni")
    beside = analyze(request)
    assert beside.resolution.how == "file" and beside.resolution.profile.name == "galaxi_bornagain.poni"
    assert beside.geometry.distance_m == pytest.approx(1.73) and beside.kind == "gisaxs"
    assert str(tmp_path / "galaxi_bornagain.poni") in beside.resolution.profile.source
    forced = analyze(AnalysisRequest(frame, profile_name="Perkin 2048", incidence_deg=0.463, forced_shapes=((1043, 981),)))
    assert forced.resolution.how == "chosen" and forced.resolution.profile.name == "Perkin 2048"  # Use Anyway


def test_a_profile_saved_from_the_poni_beside_the_frames_wins_over_the_file(tmp_path: Path) -> None:
    frame = tmp_path / "f.tif"
    poni = tmp_path / "set.poni"
    poni.write_text(format_poni(DetectorGeometry(172e-6, 172e-6, 1.0, 10, 10, 1.0)), encoding="utf-8")
    from src.gimap.features.analyze.application import SiblingPoni

    stale = InstrumentProfile("lab", DetectorGeometry(172e-6, 172e-6, 2.0, 5, 5, 1.5), None, (100, 90), source="calibration")
    resolve = ResolveGeometry(Store([stale]), SiblingPoni())
    assert resolve(detector_name=None, shape=(100, 90), path=frame).how == "file"  # another experiment's profile
    saved = InstrumentProfile("lab", stale.geometry.with_beam_center(11, 12), None, (100, 90),
                              source=f"beam centre set in Analyze · from .poni beside the frame: {poni}")
    resolution = ResolveGeometry(Store([saved]), SiblingPoni())(detector_name=None, shape=(100, 90), path=frame)
    assert resolution.how == "matched" and resolution.geometry.beam_center_x_px == 11


# -- the transmission technique ----------------------------------------------------------------------


@requires_public
def test_a_calibrant_frame_is_reduced_in_transmission_round_the_whole_ring() -> None:
    analyze = AnalyzeFrame(DetectorIoFrameSource(), Store())
    analysis = analyze(AnalysisRequest(AGBH))  # its .poni lies beside it
    assert analysis.kind == "transmission" and analysis.detected_kind == "transmission"
    assert any(message.startswith("Taken as SAXS/WAXS in transmission") for message in analysis.messages)
    keys = [curve.key for curve in analysis.reduction.curves]
    assert keys[:2] == ["radial", "azimuthal"] and "horizontal" not in keys  # no Yoneda cut through the rings
    chi = analysis.reduction.curve("azimuthal")
    assert chi.x.min() < -170 and chi.x.max() > 170  # the rings below the beam count too
    assert "horizon_row" not in analysis.reduction.markers
    cake = analyze.cake(analysis)
    assert cake.chi_range == (-180.0, 180.0)
    grazing = analyze(AnalysisRequest(AGBH, incidence_deg=0.2))  # an incidence angle: grazing incidence
    assert grazing.kind == "gisaxs"
    pinned = analyze(AnalysisRequest(AGBH, mode="giwaxs"))
    assert pinned.kind == "giwaxs" and pinned.detected_kind == "transmission"


def test_a_file_named_after_a_standard_without_incidence_is_transmission(tmp_path: Path) -> None:
    from src.gimap.features.analyze.application.technique import BY_NAME, TechniqueGuess

    geometry = DetectorGeometry(172e-6, 172e-6, 1.0, 50, 50, 1.0)
    data, valid = np.ones((100, 100), np.float32), np.ones((100, 100), bool)
    guess = TechniqueGuess()
    assert guess.detect(tmp_path / "AgBH_001.tif", (0, ()), data, valid, geometry, None) == ("transmission", BY_NAME)
    assert guess.detect(tmp_path / "AgBH_001.tif", (0, ()), data, valid, geometry, 0.2)[0] == "gisaxs"
    assert guess.detect(tmp_path / "film_001.tif", (0, ()), data, valid, geometry, None) == ("gisaxs", None)


# -- the readout --------------------------------------------------------------------------------------


def test_the_readout_gives_every_coordinate_of_a_point() -> None:
    geometry = DetectorGeometry(172e-6, 172e-6, 0.3, 200.0, 400.0, 1.0, 0.2, rot1_rad=0.01)
    point = detector_point(geometry, 250.0, 150.0, "giwaxs")
    exact = q_at_position(geometry, 250.0, 150.0)
    assert point["q"] == pytest.approx(exact["q"]) and point["qz"] == pytest.approx(exact["qz"])
    assert point["alpha_f_deg"] == pytest.approx(exact["alpha_f_deg"])
    assert point["d_angstrom"] == pytest.approx(2 * math.pi / exact["q"])
    assert point["chi_deg"] == pytest.approx(math.degrees(math.atan2(exact["q_parallel"], exact["qz"])))
    flat = DetectorGeometry(172e-6, 172e-6, 0.3, 200.0, 400.0, 1.0)
    side = detector_point(flat, 300.0, 400.0, "transmission")  # right of the beam: χ = 0, 2θ = atan(r/D)
    assert side["chi_deg"] == pytest.approx(0.0, abs=1e-9) and side["alpha_f_deg"] is None
    assert side["two_theta_deg"] == pytest.approx(math.degrees(math.atan(100 * 172e-6 / 0.3)))
    up = detector_point(flat, 200.0, 300.0, "transmission")
    assert up["chi_deg"] == pytest.approx(90.0)
    on_map = map_point(flat, 0.0, 0.1, "giwaxs")
    assert on_map["chi_deg"] == pytest.approx(0.0) and on_map["d_angstrom"] == pytest.approx(2 * math.pi / 0.1)


# -- Batch Export and frames of another detector ----------------------------------------------------------


def test_batch_export_says_where_each_folder_gets_its_geometry(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir(), second.mkdir()
    files = [first / "x1.tif", first / "x2.tif", second / "y1.tif"]
    (second / "cal.poni").write_text("x", encoding="utf-8")
    lines, possible = batch_geometry_lines(files, files[0], "profile “P”", have_profiles=True)
    assert lines[0] == ("{n} frame(s) in {folder}: {source}, as the frame on screen.",
                        {"n": 2, "folder": "a", "source": "profile “P”"})
    assert lines[1][0].endswith("geometry from {poni} beside them.") and lines[1][1]["poni"] == "cal.poni"
    (second / "cal.poni").unlink()
    _lines, possible = batch_geometry_lines(files, files[0], None, have_profiles=False)
    assert not possible  # nothing can give these frames a geometry: Batch Export is off


def test_frames_of_another_detector_are_not_added_to_the_list() -> None:
    sources = [ImportSource(Path("g/f1.tif"), 1, (1043, 981), None), ImportSource(Path("g/f2.tif"), 1)]
    text = other_detector(sources, (2048, 2048), "Perkin")
    assert text.startswith("These frames are 1043×981 (detector not named); the frames listed are 2048×2048 (Perkin)")
    assert other_detector(sources, (1043, 981)) is None and other_detector(sources, None) is None


def test_a_folder_that_changes_all_the_time_is_not_listed_again_for_every_frame(tmp_path: Path) -> None:
    from src.gimap.features.analyze.application import geometry_files

    frame = tmp_path / "run7_00012.cbf"
    frame.write_bytes(b"")
    (tmp_path / "LaB6.poni").write_text("x", encoding="utf-8")
    assert sibling_poni(frame).name == "LaB6.poni"
    key = str(tmp_path).casefold()
    when, changed, found, _took = geometry_files._PONI_LISTINGS[key]
    geometry_files._PONI_LISTINGS[key] = (when, changed, found, 60.0)  # as if listing the folder took a minute
    (tmp_path / "run7.poni").write_text("x", encoding="utf-8")  # the folder changed meanwhile
    assert sibling_poni(frame).name == "LaB6.poni"  # the listing before is still taken
    geometry_files._PONI_LISTINGS[key] = (when - 601.0, changed, found, 60.0)  # ten listings' time later
    assert sibling_poni(frame).name == "run7.poni"
