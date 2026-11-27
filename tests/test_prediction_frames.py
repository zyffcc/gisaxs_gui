"""2D Prediction reads CBF, EDF and TIFF frames through GIMaP's readers, and tells a module that asks about them.

A P03 CBF comes out exactly as the fabio reader gave it (the shipped modules see the same input); on a Pilatus EDF
the module gaps stored as 0 are given as −1, the code a CBF stores there. A module whose preprocessing takes
``context`` is given the frame's facts and, with a matching instrument profile, its geometry and q; the plain part
goes into the first step's record. All frames here are synthetic or public (``TestSAXSdata/``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.prediction.application import LoadedPredictionImage, PreparePredictionInput
from src.gimap.features.prediction.domain import FRAME_SUFFIXES, PredictionModule, PreprocessSpec, extract_frame_index
from src.gimap.features.prediction.infrastructure import (
    DetectorPredictionImageRepository,
    LocalPredictionFileCatalog,
    ModuleEntryPreprocessor,
    ProfileFrameContexts,
)
from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
from src.gimap.shared.detector_io.module_gaps import layout_gaps
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from tests.test_als733_frames import _side
from tests.test_normalisation import _save_edf

ROOT = Path(__file__).resolve().parents[1]
P03_CBF = sorted((ROOT / "TestSAXSdata").glob("*.cbf"))

PLAIN = '''
import numpy as np


def run(image, config, *, module_folder="", return_steps=False):
    out = np.asarray(image, dtype=np.float32)
    return (out, [{"step": "plain", "image": out}]) if return_steps else out
'''

ASKING = '''
import numpy as np

SEEN = []


def run(image, config, *, module_folder="", return_steps=False, context=None):
    SEEN.append(context)
    maps = context["q_map"]() if context and context.get("geometry") else None
    out = np.asarray(maps["qz"] if maps else image, dtype=np.float32)
    return (out, [{"step": "q grid", "image": out}]) if return_steps else out
'''


def _module(tmp_path: Path, source: str, name: str, **fields) -> PredictionModule:
    folder = tmp_path / name
    folder.mkdir()
    (folder / "pre.py").write_text(source, encoding="utf-8")
    return PredictionModule(id=name, name=name, folder=folder, preprocess=PreprocessSpec(entry="pre:run"), **fields)


def _edf_2m(tmp_path: Path, name: str, alpha: float | None = 0.25) -> tuple[Path, np.ndarray]:
    _detector, gaps = layout_gaps((1679, 1475))
    counts = np.random.default_rng(7).poisson(2.0, (1679, 1475)).astype(np.int32)
    counts[gaps] = 0
    path = _save_edf(tmp_path / name, counts, {"count_time": "0.1"})
    if alpha is not None:
        _side(path, alpha)
    return path, gaps


@pytest.mark.skipif(not P03_CBF, reason="TestSAXSdata/ CBF frames are not in this checkout")
def test_a_p03_cbf_comes_out_exactly_as_before():
    import fabio

    stack = tuple(P03_CBF[:2])
    loaded = DetectorPredictionImageRepository().load(stack)
    before = sum(np.asarray(fabio.open(str(path)).data, dtype=np.float32) for path in stack)
    assert loaded.image.dtype == np.float32
    np.testing.assert_array_equal(loaded.image, before)
    assert loaded.facts["format"] == "cbf" and loaded.facts["shape"] == list(before.shape)


def test_the_zero_gaps_of_a_pilatus_edf_are_given_as_minus_one(tmp_path):
    path, gaps = _edf_2m(tmp_path, "spray_00003.edf")
    loaded = DetectorPredictionImageRepository().load((path,))
    assert (loaded.image[gaps] == -1).all()
    inside = loaded.image[~gaps]
    assert (inside >= 0).all() and (inside == 0).any()  # a zero count inside a module stays a measurement
    assert loaded.facts["incidence_deg"] == pytest.approx(0.25)
    assert "Sample Alpha Stage" in loaded.facts["incidence_source"]
    assert loaded.facts["module_gaps"]["pixels"] == int(gaps.sum())


def test_frames_of_one_format_make_a_stack_and_a_numbered_folder(tmp_path):
    for index in (1, 2, 3):
        _edf_2m(tmp_path, f"spray_{index:05d}.edf", alpha=None)
    (tmp_path / "spray_00001.txt").write_text("not a frame", encoding="ascii")
    catalog = LocalPredictionFileCatalog()
    stack = catalog.stack_paths(tmp_path / "spray_00002.edf", 2)
    assert [path.name for path in stack] == ["spray_00002.edf", "spray_00003.edf"]
    numbered = catalog.numbered_files(tmp_path, FRAME_SUFFIXES)
    assert [(item.path.name, item.index) for item in numbered] == [
        ("spray_00001.edf", 1), ("spray_00002.edf", 2), ("spray_00003.edf", 3)]
    assert extract_frame_index("run_00042.CBF") == 42 and extract_frame_index("notes.txt") is None


def test_a_module_that_asks_is_given_the_frame_and_its_geometry(tmp_path):
    path, _gaps = _edf_2m(tmp_path, "spray_00004.edf", alpha=0.12)
    loaded = DetectorPredictionImageRepository().load((path,))
    profile = InstrumentProfile(
        "ALS 7.3.3 (made up)", DetectorGeometry(172e-6, 172e-6, 3.5, 700.0, 1600.0, 1.2398, 0.4), None, (1679, 1475))
    prepare = PreparePredictionInput(
        ModuleEntryPreprocessor(), ProfileFrameContexts(InMemoryInstrumentProfileRepository([profile])))

    asking = _module(tmp_path, ASKING, "asking")
    prepared = prepare.execute(loaded.image, asking, loaded)
    record = prepared.steps[0]["frame_context"]
    assert record["geometry"]["incidence_deg"] == pytest.approx(0.12)  # the frame's αi, not the profile's 0.4
    assert record["geometry"]["beam_center_x_px"] == pytest.approx(700.0)
    assert "ALS 7.3.3 (made up)" in record["geometry_source"] and "Sample Alpha Stage" in record["geometry_source"]
    assert "q_map" not in record and record["frame"]["format"] == "edf"
    assert prepared.values.shape[1:3] == (1679, 1475)  # q_z of every pixel, row 0 at the top
    qz = prepared.values[0, :, :, 0]
    assert qz[0, 700] > qz[1678, 700]  # q_z grows upward from the bottom of the detector

    plain = _module(tmp_path, PLAIN, "plain")
    unchanged = prepare.execute(loaded.image, plain, loaded)
    assert "frame_context" not in unchanged.steps[0]
    np.testing.assert_array_equal(unchanged.values[0, :, :, 0], loaded.image)

    # Another image than the loaded one: the module is told nothing (a context never describes another frame).
    other = loaded.image.copy()
    seen = prepare.execute(other, asking, loaded)
    assert "frame_context" not in seen.steps[0]


def test_a_module_that_needs_the_geometry_says_what_is_missing(tmp_path):
    path, _gaps = _edf_2m(tmp_path, "spray_00005.edf")
    loaded = DetectorPredictionImageRepository().load((path,))
    prepare = PreparePredictionInput(ModuleEntryPreprocessor(), ProfileFrameContexts(InMemoryInstrumentProfileRepository()))
    needs = _module(tmp_path, ASKING, "needs", needs_geometry=True)
    with pytest.raises(ValueError, match="needs the geometry.*no instrument profile.*1475 × 1679"):
        prepare.execute(loaded.image, needs, loaded)


def test_a_loaded_image_without_facts_is_still_accepted():
    loaded = LoadedPredictionImage(np.ones((4, 4)), (Path("frame.cbf"),))
    assert loaded.facts == {}


COUNTING = '''
import numpy as np

CALLS = []
VERSION = VERSION_VALUE


def run(image, config, *, module_folder="", return_steps=False):
    CALLS.append(VERSION)
    out = np.asarray(image, dtype=np.float32) + VERSION
    return (out, [{"step": "count", "image": out, "calls": list(CALLS)}]) if return_steps else out
'''


def test_a_module_is_imported_once_and_again_when_its_file_changes(tmp_path):
    module = _module(tmp_path, COUNTING.replace("VERSION_VALUE", "0"), "counting")
    preprocessor = ModuleEntryPreprocessor()
    image = np.zeros((4, 5), np.float32)
    preprocessor.preprocess(image, module)
    second = preprocessor.preprocess(image, module)
    assert second.steps[0]["calls"] == [0, 0]  # what the module keeps stays from frame to frame
    (module.folder / "pre.py").write_text(COUNTING.replace("VERSION_VALUE", "1") + "\n# edited\n", encoding="utf-8")
    edited = preprocessor.preprocess(image, module)
    assert edited.steps[0]["calls"] == [1] and float(edited.values.max()) == 1.0  # the edited file, imported anew


def test_the_q_maps_of_a_geometry_are_computed_once_and_are_read_only():
    from dataclasses import replace

    geometry = DetectorGeometry(172e-6, 172e-6, 3.5, 20.0, 60.0, 1.2398, 0.4)
    contexts = ProfileFrameContexts()
    first, again = contexts.q_maps((64, 48), geometry), contexts.q_maps((64, 48), geometry)
    assert first["qz"] is again["qz"] and not first["qz"].flags.writeable
    with pytest.raises(ValueError):
        first["qz"][0, 0] = 1.0
    assert contexts.q_maps((64, 48), replace(geometry, incidence_deg=0.1))["qz"] is not first["qz"]
