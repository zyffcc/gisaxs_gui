"""One way to name and number a frame in everything Analyze exports.

A frame of a source that holds several is counted from 1 where a person reads it — headers, column heads,
picture titles, records — and by its 0-based index in file names (``_frame0003``, as scripts and Fitting's
listings know them; the README says so). A set of module files is named by its scan, without ``_m01``, and
its record lists every module file. The curve for Fitting is q in Å⁻¹ whatever axis Analyze shows, says its
frame, and is never seen half written. Every file is synthetic.
"""

from __future__ import annotations

import gc
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.gimap.features.analyze.application import (
    AnalysisRequest,
    AnalyzeFrame,
    BatchChoices,
    ExportAnalysis,
    FrameJob,
    FrameTools,
    batch_stem,
    export_stem,
    fit_input_curve,
    fit_input_observation,
    frame_name,
    frame_name_of,
    frame_stem,
    output_names,
    readme_text,
    run_frame_job,
)
from src.gimap.features.analyze.application.export_record import analysis_metadata, source_files
from src.gimap.features.analyze.application.frame_preprocessing import FrameSum
from src.gimap.features.analyze.domain import GiwaxsSettings, sum_frames
from src.gimap.features.analyze.infrastructure.adapters import CsvCurveWriter, DetectorIoFrameSource
from src.gimap.features.analyze.infrastructure.adapters.csv_export import _frame_lines
from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
from src.gimap.shared.detector_io import forget_layout
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from tests.test_analyze_nexus_sources import _module, _stack
from tests.test_assistant_calibration import SHAPE, giwaxs_frame, save_tiff
from tests.test_giwaxs_workspace import GEOMETRY


@pytest.fixture(autouse=True)
def _nothing_remembered():
    forget_layout()
    yield
    forget_layout()


def _tools(profile: InstrumentProfile, figures=None) -> FrameTools:
    source = DetectorIoFrameSource()
    return FrameTools(AnalyzeFrame(source, InMemoryInstrumentProfileRepository([profile])), ExportAnalysis(CsvCurveWriter()),
                      figures)


def _module_set(folder: Path, frames: int = 4, modules: int = 3) -> tuple[Path, InstrumentProfile]:
    """A scan written as one file per detector module, each with every frame, and a geometry that fits it."""
    paths = [_module(folder / f"scan_00007_m{number:02d}.nxs", number, frames) for number in range(1, modules + 1)]
    shape = DetectorIoFrameSource().load(paths[0], 0).data.shape
    geometry = DetectorGeometry(75e-6, 75e-6, 0.02, shape[1] / 2, shape[0] - 2.0, 1.0, 0.2)
    return paths[0], InstrumentProfile("set", geometry, None, shape)


class _Figures:
    """Stands in for the figure writer: what each picture is called and titled."""

    def __init__(self):
        self.pictures = []

    def write_frame(self, data, valid, path, *, title="", **_display):
        self.pictures.append((Path(path).name, title))
        return Path(path)


# -- names ----------------------------------------------------------------------------------------------------------


def test_a_frame_is_named_by_its_source_its_0_based_index_and_its_sum() -> None:
    module_set = {"layout": "module_set"}
    assert frame_stem("d/scan_00007_m01.nxs", module_set, frame_index=3, frame_count=12) == "scan_00007_frame0003"
    assert frame_stem("d/scan_00007_m01.nxs", {"layout": "stack"}, frame_index=3, frame_count=12) == "scan_00007_m01_frame0003"
    assert frame_stem("d/run.nxs", None, frame_index=0, frame_count=5, frame_total=3) == "run_frame0000_sum3"
    assert frame_stem("d/run_0001.tif") == "run_0001"  # one frame in its file: no index
    assert frame_stem("d/run_0001.tif", frame_total=4) == "run_0001_sum4"
    # … and what people read counts from 1: the same frame is “frame 4 of 12”
    assert frame_name_of("d/scan_00007_m01.nxs", module_set, frame_index=3, frame_count=12) == "scan_00007 frame 4 of 12"
    assert frame_name_of("d/run.nxs", None, frame_index=0, frame_count=5, summed=2) == "run.nxs frame 1 of 5 (+2 summed)"
    assert frame_name_of("d/run_0001.tif") == "run_0001.tif"
    assert batch_stem([Path("d/scan_00007_m01.nxs")], module_set) == "scan_00007"  # the folder of a batch too
    assert batch_stem([Path("d/scan_00007_m01.nxs")]) == "scan_00007_m01" and batch_stem([Path("d/a.tif"), Path("d/b.tif")]) == "d"


def test_a_set_of_module_files_is_exported_under_its_scan_name_and_every_module_is_recorded(tmp_path: Path) -> None:
    first, profile = _module_set(tmp_path / "raw")
    figures = _Figures()
    tools = _tools(profile, figures)
    out = tmp_path / "out"
    choices = BatchChoices(tables=True, per_frame=True, fit_input=True, q_map=True, frames=True, frame_format="npy",
                           detector_image=True, q_map_image=True)
    job = FrameJob(index=0, request=AnalysisRequest(path=first, frame_index=2, mode="giwaxs", profile_name="set"),
                   choices=choices, destination=out, number=3)
    outcome = run_frame_job(tools, job)
    names = sorted(path.relative_to(out).as_posix() for path in outcome.written if path.parent != out / "images")
    assert outcome.stem == "scan_00007_frame0002" and outcome.label == "scan_00007 frame 3 of 4" and outcome.number == 3
    assert all("_m01" not in name and Path(name).name.startswith("scan_00007_frame0002") for name in names), names
    assert {"curves/scan_00007_frame0002_radial.csv", "curves/scan_00007_frame0002_analysis.json",
            "fit_input/scan_00007_frame0002_fit_input.dat", "maps/scan_00007_frame0002_qmap.csv",
            "frames/scan_00007_frame0002.npy"} <= set(names)
    # the pictures are named like the files and titled with the frame as people count it
    assert figures.pictures == [("scan_00007_frame0002_detector.png", "scan_00007 frame 3 of 4"),
                                ("scan_00007_frame0002_qmap.png", "scan_00007 frame 3 of 4")]
    record = json.loads((out / "curves" / "scan_00007_frame0002_analysis.json").read_text(encoding="utf-8"))
    assert record["source_file"].endswith("scan_00007_m01.nxs") and record["frame_index"] == 2 and record["frame_count"] == 4
    assert record["frame"] == "frame 3 of 4"
    assert [Path(name).name for name in record["source_files"]] == [f"scan_00007_m{n:02d}.nxs" for n in (1, 2, 3)]
    assert record["source_layout"]["kind"] == "module_set" and record["source_layout"]["dataset"].endswith("detector/data")
    assert record["source_layout"]["transformations"]  # how the image was turned is part of where it comes from
    for name in ("curves/scan_00007_frame0002_radial.csv", "fit_input/scan_00007_frame0002_fit_input.dat",
                 "maps/scan_00007_frame0002_qmap.csv"):
        lines = (out / name).read_text(encoding="utf-8").splitlines()
        assert "# frame: 3 of 4 (index 2)" in lines, name
        assert ("# files: this frame is read from 3 files: scan_00007_m01.nxs, scan_00007_m02.nxs, scan_00007_m03.nxs"
                in lines), name
    observation = json.loads(next(line for line in (out / "fit_input" / "scan_00007_frame0002_fit_input.dat").read_text(
        encoding="utf-8").splitlines() if line.startswith("# observation:")).split(":", 1)[1])
    assert observation["frame_index"] == 2 and observation["frame_count"] == 4  # for a series in Fitting


def test_the_header_says_the_frame_from_1_with_its_index_and_what_was_summed() -> None:
    assert _frame_lines({"frame_index": 3, "frame_count": 12}) == ["# frame: 4 of 12 (index 3)\n"]
    assert _frame_lines({"frame_index": 0}) == ["# frame: 1 (index 0)\n"]
    assert _frame_lines({"source_file": "a.dat"}) == []  # a record that names no frame: no line made up
    same = _frame_lines({"source_file": "d/run.nxs", "frame_index": 3, "frame_count": 12, "summed_frames": [
        {"file": "d/run.nxs", "frame_index": 4}, {"file": "d/run.nxs", "frame_index": 5}]})
    assert same == ["# frame: 4 of 12 (index 3)\n",
                    "# sum of 3 frames: 4 to 6 of 12 (index 3 to 5); intensities are sums over these frames\n"]
    files = _frame_lines({"source_file": "d/a_001.tif", "frame_index": 0, "frame_count": 1, "summed_frames": [
        {"file": "d/a_002.tif", "frame_index": 0}]})
    assert files[1] == "# sum of 2 frames: a_001.tif, a_002.tif; intensities are sums over these frames\n"
    many = _frame_lines({"frame_index": 0, "source_files": [f"d/s_m{n:02d}.nxs" for n in range(1, 31)]})
    assert many[1].startswith("# files: this frame is read from 30 files: s_m01.nxs,") and "(18 more)" in many[1]
    assert many[1].rstrip().endswith("s_m30.nxs")


def test_a_frame_of_one_file_and_a_master_say_their_files(tmp_path: Path) -> None:
    assert source_files("d/run.tif", {}) == [] and source_files("d/run.nxs", {"layout": "stack"}) == []
    assert source_files("d/s_master.h5", {"data_files": ["d/s_data_000001.h5"]}) == ["d/s_master.h5", "d/s_data_000001.h5"]
    assert source_files("d/s_m01.nxs", {"module_files": ["d/s_m01.nxs", "d/s_m02.nxs"]}) == ["d/s_m01.nxs", "d/s_m02.nxs"]
    # a multi-frame file that is no module set keeps its whole name, and its frames their index
    stack = _stack(tmp_path / "scan_m01.nxs", 5)
    geometry = DetectorGeometry(75e-6, 75e-6, 0.02, 18.0, 22.0, 1.0, 0.2)
    shape = DetectorIoFrameSource().load(stack, 0).data.shape
    tools = _tools(InstrumentProfile("one", geometry, None, shape))
    analysis = tools.analyze_frame(AnalysisRequest(path=stack, frame_index=4, mode="giwaxs", profile_name="one"))
    assert export_stem(analysis) == "scan_m01_frame0004" and frame_name(analysis) == "scan_m01.nxs frame 5 of 5"
    assert "source_files" not in analysis_metadata(analysis)


def test_the_readme_says_how_frames_are_named_and_counted() -> None:
    text = readme_text(BatchChoices(per_frame=True), batch="run", frame="f", frames=3, curves=[])
    assert '"_frameNNNN" is the index of the frame in a source that holds several, counted from 0' in text
    assert '"frame 4 of 12" is _frame0003' in text and 'without "_mNN"' in text


# -- the curve for Fitting ------------------------------------------------------------------------------------------


def _giwaxs(tmp_path: Path, x_axis: str):
    path = save_tiff(tmp_path / "film_0001.tif", giwaxs_frame(seed=1))
    tools = _tools(InstrumentProfile("synthetic", GEOMETRY, None, SHAPE))
    request = AnalysisRequest(path=path, mode="giwaxs", profile_name="synthetic", giwaxs=GiwaxsSettings(x_axis=x_axis))
    return tools, tools.analyze_frame(request)


def test_a_fit_input_is_q_in_inverse_angstrom_also_when_analyze_shows_two_theta(tmp_path: Path) -> None:
    tools, in_q = _giwaxs(tmp_path, "q")
    _tools_again, in_angle = _giwaxs(tmp_path, "two_theta")
    assert in_angle.reduction.curve("radial").x_label.startswith("2θ")  # what the plot shows
    wanted, curve = fit_input_curve(in_q), fit_input_curve(in_angle)
    assert curve.x_label == "q (Å⁻¹)" and curve.title == wanted.title == "I(q)"
    np.testing.assert_allclose(curve.x, wanted.x, rtol=1e-12)
    np.testing.assert_array_equal(curve.intensity, wanted.intensity)
    written = tools.export.fit_input(in_angle, tmp_path / "out")
    lines = written.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# GIMaP Analyze fit input (q in 1/A)" and "# columns: q (1/A)  I  sigma  pixels" in lines
    np.testing.assert_allclose(np.loadtxt(written)[:, 0], wanted.x, rtol=1e-7)  # 8 digits in the file
    # the whole export of the frame (Export, the folder watch) writes the same curve
    exported = tools.export(in_angle, tmp_path / "all")
    fit_input = next(path for path in exported if path.name.endswith("_fit_input.dat"))
    np.testing.assert_allclose(np.loadtxt(fit_input)[:, 0], wanted.x, rtol=1e-7)


def test_a_fit_input_is_complete_or_not_there(tmp_path: Path, monkeypatch) -> None:
    """It is written under a name no listing of ``*_fit_input.dat`` matches and renamed when complete: a folder
    that Fitting watches never shows half a curve, and a write that fails leaves nothing behind."""
    from src.gimap.features.analyze.domain import Curve
    from src.gimap.features.analyze.infrastructure.adapters import csv_export

    x = np.linspace(0.01, 0.2, 50)
    curve = Curve("fit_input", "I(q)", x, 1.0 / x, np.full(50, 0.1), np.full(50, 7), "q (Å⁻¹)")
    seen = []
    replace = csv_export.os.replace

    def watching(source, target):
        seen.append((sorted(path.name for path in tmp_path.glob("*_fit_input.dat")), Path(source).name))
        return replace(source, target)

    monkeypatch.setattr(csv_export.os, "replace", watching)
    path = CsvCurveWriter().write_xy(curve, tmp_path / "f_fit_input.dat", {"source_file": "f.tif", "frame_index": 0})
    assert seen == [([], ".f_fit_input.dat.part")]  # while it was written, no fit input was there
    assert sorted(item.name for item in tmp_path.iterdir()) == ["f_fit_input.dat"] and np.loadtxt(path).shape == (50, 4)

    class Broken:  # a curve whose pixels cannot be read: the write ends in the middle of the file
        title, x_label = "I(q)", "q"

        def __init__(self, values):
            self.x = self.intensity = self.sigma = values

        @property
        def pixels(self):
            raise OSError("disk full")

    with pytest.raises(OSError):
        CsvCurveWriter().write_xy(Broken(x), tmp_path / "g_fit_input.dat", {"source_file": "g.tif"})
    assert sorted(item.name for item in tmp_path.iterdir()) == ["f_fit_input.dat"]  # neither half a file nor its draft


def test_the_observation_names_the_frame_only_of_a_source_that_holds_several() -> None:
    from src.gimap.features.analyze.domain import Corrections

    frame = dict(kind="giwaxs", corrections=Corrections(), path=Path("run.nxs"), metadata={}, intensity_scale=None, frame_total=1)
    alone = fit_input_observation(SimpleNamespace(**frame, frame_index=0, frame_count=1))
    assert "frame_index" not in alone and "frame_count" not in alone
    fourth = fit_input_observation(SimpleNamespace(**frame, frame_index=3, frame_count=12))
    assert fourth["frame_index"] == 3 and fourth["frame_count"] == 12


def test_the_fitted_curve_of_a_peak_is_named_after_its_region_and_the_model_after_itself() -> None:
    peaks = BatchChoices(fit="peaks", fit_curves=True)
    assert output_names(peaks, batch="run", frame="f", curve="radial", fit_key="region1")["fit_curves"] == "fits/f_region1_fit.csv"
    model = BatchChoices(fit="model", fit_curves=True, text_format="txt")
    assert output_names(model, batch="run", frame="f", curve="horizontal")["fit_curves"] == "fits/f_model_fit.txt"


# -- summing --------------------------------------------------------------------------------------------------------


def test_frames_summed_one_by_one_give_what_the_list_gave() -> None:
    rng = np.random.default_rng(3)
    frames = []
    for _ in range(4):
        data = rng.poisson(5.0, size=(6, 8)).astype(np.float32)
        valid = rng.random((6, 8)) > 0.1
        frames.append((data, valid))
    total = FrameSum()
    for data, valid in frames:
        total.add(data, valid)
    summed, valid = total.result()
    expected, expected_valid = sum_frames(frames)
    np.testing.assert_array_equal(valid, expected_valid)
    np.testing.assert_array_equal(summed[valid], expected[expected_valid])
    assert np.isnan(summed[~valid]).all() and summed.dtype == np.float32
    with pytest.raises(ValueError, match="Cannot sum a 3×3 frame with 6×8 frames"):
        total.add(np.zeros((3, 3)), np.ones((3, 3), dtype=bool))
    with pytest.raises(ValueError, match="No frames to sum"):
        FrameSum().result()


def test_a_sum_of_many_frames_keeps_one_frame_in_memory_at_a_time() -> None:
    """Sum 40 of a 15 Mpx detector held every frame at once (3 GB): now the total and the frame being added."""
    alive: list = []

    class Frames:
        most = 0

        def load(self, path, frame_index=0):
            gc.collect()
            Frames.most = max(Frames.most, sum(1 for ref in alive if ref() is not None))
            data = np.full((8, 8), float(frame_index + 1), dtype=np.float32)
            image = SimpleNamespace(data=data, mask=None, metadata={}, detector_name="d")
            alive.append(weakref.ref(data))
            return image

        def frame_count(self, path):
            return 30

    analyze = AnalyzeFrame(Frames(), None)
    request = AnalysisRequest(path=Path("run.nxs"), summed_frames=tuple((Path("run.nxs"), index) for index in range(1, 30)))
    data, valid, _common = analyze._load_frames(request)
    assert float(data[0, 0]) == sum(range(1, 31)) and valid.all()
    assert Frames.most <= 2  # the first frame (its metadata are kept) and the one before: never the whole series
