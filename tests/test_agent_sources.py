"""What the files given to ``tools/gimap_agent.py auto`` stand for.

A set of detector-module files is one stitched series however many of its files are given, and is analysed
once; a source with a part missing is said so; files that look like one series stored as one frame per file
are named as such and, with ``--series``, analysed as one series. Synthetic files only; nothing is written
next to them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest
from PIL import Image

from src.gimap.app.headless_sources import batch_summary, given_sources, series_block, series_hints
from src.gimap.features.assistant.application import PipelineOptions
from src.gimap.shared.detector_io import forget_layout

DET = "/entry/instrument/detector"
H, W = 40, 60


@pytest.fixture(autouse=True)
def _nothing_remembered():
    forget_layout()
    yield
    forget_layout()


def _module(path: Path, module: int, frames: int = 4) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        data = np.full((frames, H, W), 10 * module, dtype=np.int32) + np.arange(frames, dtype=np.int32)[:, None, None]
        handle.create_dataset(f"{DET}/data", data=data, chunks=(1, H, W))
        handle.create_dataset(f"{DET}/translation/distance", data=np.asarray([0.0, (module - 1) * (H + 4), 0.0], dtype=np.float32))
    return path


def _module_set(folder: Path, modules=(1, 2, 3), stem: str = "scan_00007", frames: int = 4) -> list[Path]:
    return [_module(folder / f"{stem}_m{module:02d}.nxs", module, frames) for module in modules]


def _tiffs(folder: Path, count: int = 4, stem: str = "film") -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for number in range(1, count + 1):
        path = folder / f"{stem}_{number:04d}.tif"
        Image.fromarray(np.full((H, W), 100 * number, dtype=np.uint16)).save(path)
        paths.append(path)
    return paths


def _snapshot(folder: Path) -> list[str]:
    return sorted(str(path.relative_to(folder)) for path in folder.rglob("*"))


# -- which sources the arguments are ---------------------------------------------------------------------


def test_the_files_of_one_module_set_are_one_source_whatever_was_given(tmp_path: Path) -> None:
    modules = _module_set(tmp_path / "data")
    tiff = _tiffs(tmp_path / "data", count=1)[0]
    given = [str(modules[2]), str(tiff), str(modules[0]), str(modules[1]), str(tmp_path / "missing.cbf")]
    sources = given_sources(given)
    assert [(Path(source.path).name, len(source.given)) for source in sources] == [
        ("scan_00007_m01.nxs", 3), ("film_0001.tif", 1), ("missing.cbf", 1),
    ]
    assert (sources[0].stem, sources[0].frames) == ("scan_00007", 4) and sources[0].layout.kind == "module_set"
    assert sources[1].path == str(tiff) and sources[2].path == str(tmp_path / "missing.cbf")  # as they were given
    assert (sources[1].stem, sources[1].frames) == ("film_0001", 1)
    # One module file that is not the first: the set, named by its first file.
    [one] = given_sources([str(modules[1])])
    assert Path(one.path).name == "scan_00007_m01.nxs" and one.given == (str(modules[1]),)


def test_files_numbered_alike_with_one_frame_each_are_named_as_a_possible_series(tmp_path: Path) -> None:
    films = _tiffs(tmp_path / "films", count=5)
    other = _tiffs(tmp_path / "films", count=2, stem="dark")
    hints = series_hints(given_sources([str(path) for path in [*films, *other]]))
    assert len(hints) == 1 and [Path(source.path).name for source in hints[0]] == [path.name for path in films]
    assert series_hints(given_sources([str(path) for path in films[:2]])) == []  # two files: not worth a remark
    # A multi-frame source is a series by itself, never part of one made of files.
    stack = _module_set(tmp_path / "set")
    assert series_hints(given_sources([str(path) for path in stack])) == []


def test_the_block_of_a_file_series_is_chosen_like_the_frames_of_a_file(tmp_path: Path) -> None:
    films = _tiffs(tmp_path / "films", count=12)
    shuffled = [str(films[index]) for index in (10, 0, 11, 3, 1, 2, 4, 5, 6, 7, 8, 9)]
    ordered, first, count = series_block(given_sources(shuffled), None, None)
    assert [Path(source.path).name for source in ordered] == [path.name for path in films]  # by their numbers
    assert (first, count) == (11, 2)  # the last fifth of twelve
    assert series_block(given_sources(shuffled), 1, 3)[1:] == (1, 3)
    assert series_block(given_sources(shuffled), -1, 30)[1:] == (1, 12)  # never past the ends
    assert series_block(given_sources(shuffled), 12, 5)[1:] == (8, 5)
    with pytest.raises(ValueError, match="one frame each"):
        series_block(given_sources([str(films[0]), *(str(path) for path in _module_set(tmp_path / "set"))]), None, None)
    with pytest.raises(ValueError, match="at least two files"):
        series_block(given_sources([str(films[0])]), None, None)
    with pytest.raises(ValueError, match="one format"):
        mixed = _tiffs(tmp_path / "mixed", count=2)
        (tmp_path / "mixed" / "film_0003.cbf").touch()
        series_block(given_sources([*(str(path) for path in mixed), str(tmp_path / "mixed" / "film_0003.cbf")]), None, None)


# -- the analysis ------------------------------------------------------------------------------------------


def test_a_module_set_given_as_all_its_files_is_analysed_once_and_says_so(tmp_path: Path) -> None:
    from src.gimap.app.headless_assistant import analyse_frames

    modules = _module_set(tmp_path / "data")
    before = _snapshot(tmp_path / "data")
    said: list[str] = []
    reports = analyse_frames(
        [str(path) for path in reversed(modules)], PipelineOptions(), tmp_path / "out", progress=said.append, saved_profiles=False,
    )
    assert len(reports) == 1
    [report] = reports
    assert Path(report["frame"]).name == "scan_00007_m01.nxs"
    assert report["source"]["summary"] == "3 module files · 4 frames · stitched"
    assert [Path(path).name for path in report["source"]["files"]] == [path.name for path in modules]
    source = next(item for item in report["decisions"] if item["what"] == "source")
    assert source["decision"] == "3 module files · 4 frames · stitched"
    assert "3 of its files were given: they are one source, analysed once" in source["why"]
    assert any("3 files given for one source" in line and "analysed once" in line for line in said)
    # Named by the scan, without the module number; nothing next to the data.
    assert (tmp_path / "out" / "scan_00007" / "report.md").is_file() and not (tmp_path / "out" / "scan_00007_m01").exists()
    assert "**source**: 3 module files · 4 frames · stitched" in (tmp_path / "out" / "scan_00007" / "report.md").read_text(encoding="utf-8")
    summary = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert "1 frame(s)" in summary and "3 files were given for scan_00007_m01.nxs" in summary
    assert _snapshot(tmp_path / "data") == before
    records = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert len(records) == 1 and records[0]["source"]["layout"] == "module_set"


def test_a_set_with_a_module_file_missing_is_an_open_point(tmp_path: Path) -> None:
    from src.gimap.app.headless_assistant import analyse_frames
    from tools import gimap_agent

    modules = _module_set(tmp_path / "data", modules=(1, 3, 4))  # m02 is not there
    [report] = analyse_frames([str(modules[0])], PipelineOptions(), tmp_path / "out", saved_profiles=False)
    item = next(item for item in report["needs_attention"] if item["item"] == "source")
    assert "m02" in item["why"] and "missing" in item["why"] and item["option"] is None
    assert report["source"]["complete"] is False
    assert gimap_agent.exit_code([report]) == 2


def test_files_of_a_possible_series_are_analysed_one_by_one_and_the_summary_says_how_to_join_them(tmp_path: Path) -> None:
    from src.gimap.app.headless_assistant import analyse_frames

    films = _tiffs(tmp_path / "films", count=3)
    reports = analyse_frames([str(path) for path in films], PipelineOptions(), tmp_path / "out", saved_profiles=False)
    assert [Path(report["frame"]).name for report in reports] == [path.name for path in films]
    summary = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert "film_0001.tif … film_0003.tif (3 files) look like one series stored as one frame per file" in summary
    assert "each was analysed on its own" in summary and "`--series`" in summary
    assert batch_summary(reports) == summary  # what the command prints


def test_with_series_the_files_are_one_series_and_its_last_files_are_summed(tmp_path: Path) -> None:
    from src.gimap.app.headless_assistant import analyse_frames

    films = _tiffs(tmp_path / "films", count=12)
    said: list[str] = []
    reports = analyse_frames(
        [str(path) for path in reversed(films)], PipelineOptions(), tmp_path / "out", progress=said.append,
        saved_profiles=False, series=True,
    )
    assert len(reports) == 1
    [report] = reports
    assert report["frames"] == {"total": 12, "first": 11, "summed": 2}  # the last fifth, as inside a file
    assert Path(report["frame"]).name == "film_0011.tif"
    assert [Path(path).name for path in report["series"]["summed_files"]] == ["film_0011.tif", "film_0012.tif"]
    assert len(report["series"]["files"]) == 12
    frames = next(item for item in report["decisions"] if item["what"] == "frames")
    assert frames["decision"] == "files 11–12 summed of 12 (film_0011.tif … film_0012.tif)"
    assert frames["why"].startswith("a series of 12 frames (an in-situ run or a scan): the last frames show the final state")
    assert (tmp_path / "out" / "film_series" / "report.md").is_file()
    assert any("12 files as one series" in line for line in said)
    text = (tmp_path / "out" / "film_series" / "report.md").read_text(encoding="utf-8")
    assert "series of 12 frames; analysed from frame 11, 2 summed" in text

    # A block that was asked for.
    [asked] = analyse_frames(
        [str(path) for path in films], PipelineOptions(frame=1, sum_frames=3), tmp_path / "start", saved_profiles=False, series=True,
    )
    assert asked["frames"] == {"total": 12, "first": 1, "summed": 3}
    assert next(item for item in asked["decisions"] if item["what"] == "frames")["why"] == "as asked"


def test_series_refuses_files_that_are_not_a_series_of_single_frames(tmp_path: Path) -> None:
    from src.gimap.app.headless_assistant import analyse_frames

    modules = _module_set(tmp_path / "data")
    film = _tiffs(tmp_path / "data", count=1)[0]
    [report] = analyse_frames(
        [str(film), *(str(path) for path in modules)], PipelineOptions(), tmp_path / "out", saved_profiles=False, series=True,
    )
    assert not report["ok"] and "one frame each" in report["error"] and "--frame" in report["error"]


# -- the command -------------------------------------------------------------------------------------------


def test_the_command_passes_series_on_and_its_help_says_what_a_module_set_is(monkeypatch, tmp_path: Path) -> None:
    from src.gimap.app import headless_assistant
    from tools import gimap_agent

    seen = {}

    def analyse_frames(frames, options, out, **kwargs):
        seen.update(kwargs, frames=list(frames))
        return [{"ok": True, "frame": "a.tif", "procedure": "giwaxs", "needs_attention": [], "decisions": [], "steps": []}]

    monkeypatch.setattr(headless_assistant, "analyse_frames", analyse_frames)
    assert gimap_agent.main(["auto", "a.tif", "b.tif", "--quiet", "--out", str(tmp_path), "--series"]) == 0
    assert seen["series"] is True and seen["frames"] == ["a.tif", "b.tif"]
    assert gimap_agent.main(["auto", "a.tif", "--quiet", "--out", str(tmp_path)]) == 0 and seen["series"] is False
    auto_help = next(
        action for action in gimap_agent.parser()._subparsers._group_actions[0].choices["auto"]._actions if action.dest == "frames"
    ).help
    assert "module files" in auto_help and "one stitched series" in auto_help and "analysed once" in auto_help
    text = gimap_agent.parser()._subparsers._group_actions[0].choices["auto"].format_help()
    assert "--series" in text and "one frame per file" in text and "the last fifth" in text


def test_a_session_opened_on_one_file_says_what_its_source_is_too(tmp_path: Path) -> None:
    """``open_frame`` + ``run_standard_pipeline`` (MCP, ``call``) and ``HeadlessSession.run_pipeline``: the same
    source decision and the same open point for a missing part as ``auto``, once each."""
    from src.gimap.app.headless_assistant import open_headless_session

    modules = _module_set(tmp_path / "data", modules=(1, 3, 4))  # m02 is not there
    session = open_headless_session(modules[1], saved_profiles=False, out_dir=tmp_path / "out")
    try:
        outcome = session.call("run_standard_pipeline")
        compact = json.loads(outcome.content)
        sources = [item for item in compact["decisions"] if item["what"] == "source"]
        assert len(sources) == 1 and sources[0]["decision"] == "3 module files · 4 frames · stitched"
        assert "scan_00007_m03.nxs was given: it is a file of this source" in sources[0]["why"]
        missing = [item for item in compact["needs_attention"] if item["item"] == "source"]
        assert len(missing) == 1 and "m02" in missing[0]["why"]
        assert f"{len(compact['needs_attention'])} open question(s)" in outcome.summary
        assert session.results.pipeline["source"]["complete"] is False

        report = session.run_pipeline(PipelineOptions())
        assert len([item for item in report["decisions"] if item["what"] == "source"]) == 1
        assert len([item for item in report["needs_attention"] if item["item"] == "source"]) == 1
    finally:
        session.close()
    assert not (tmp_path / "data" / "gimap_analysis").exists()


# -- a file named more than once, a folder reached through a link ---------------------------------------------


def test_a_file_named_twice_is_one_source_whatever_its_format(tmp_path: Path, monkeypatch) -> None:
    from src.gimap.app.headless_sources import given_note

    films = _tiffs(tmp_path / "films", count=2)
    broken = tmp_path / "films" / "broken.h5"
    broken.write_bytes(b"not an HDF5 file")
    given = [str(films[0]), str(films[1]), str(films[0]), str(broken), str(broken), str(tmp_path / "missing.cbf"), str(tmp_path / "missing.cbf")]
    sources = given_sources(given)
    assert [(Path(source.path).name, len(source.given), len(source.files)) for source in sources] == [
        ("film_0001.tif", 2, 1), ("film_0002.tif", 1, 1), ("broken.h5", 2, 1), ("missing.cbf", 2, 1),
    ]
    assert all(source.layout is None for source in sources)  # no NeXus layout: nothing may ask for its summary
    assert [given_note(source) for source in sources] == [
        "given 2 times: analysed once", "", "given 2 times: analysed once", "given 2 times: analysed once",
    ]
    # The same file under two spellings is the same file.
    monkeypatch.chdir(tmp_path)
    [one] = given_sources([str(films[0]), os.path.join("films", "film_0001.tif")])
    assert one.path == str(films[0]) and len(one.given) == 2 and len(one.files) == 1


def test_a_batch_with_a_file_named_twice_is_analysed_and_says_so(tmp_path: Path) -> None:
    """It ended in a traceback before any file was analysed (also for overlapping globs)."""
    from src.gimap.app.headless_assistant import analyse_frames

    films = _tiffs(tmp_path / "films", count=2)
    broken = tmp_path / "broken.h5"
    broken.write_bytes(b"not an HDF5 file")
    said: list[str] = []
    given = [str(films[0]), str(films[1]), str(films[0]), str(broken), str(broken), str(tmp_path / "missing.cbf"), str(tmp_path / "missing.cbf")]
    reports = analyse_frames(given, PipelineOptions(), tmp_path / "out", progress=said.append, saved_profiles=False)
    assert [Path(report["frame"]).name for report in reports] == ["film_0001.tif", "film_0002.tif", "broken.h5", "missing.cbf"]
    assert [line.strip() for line in said].count("given 2 times: analysed once") == 3
    assert reports[0]["given"] == [str(films[0]), str(films[0])] and "given" not in reports[1]
    assert reports[2].get("failed") and reports[2]["given"] == [str(broken), str(broken)]  # unreadable: said by its report
    assert reports[3]["ok"] is False and "No such file" in reports[3]["error"]
    summary = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert "film_0001.tif was given 2 times: it was analysed once." in summary
    assert "broken.h5 was given 2 times: it was analysed once." in summary
    assert "missing.cbf was given 2 times: it was analysed once." in summary
    assert "film_0002.tif was given" not in summary
    assert (tmp_path / "out" / "film_0001" / "report.md").is_file() and (tmp_path / "out" / "film_0002" / "report.md").is_file()


def test_a_stack_named_twice_and_a_set_with_a_file_named_twice_count_their_files(tmp_path: Path) -> None:
    from src.gimap.app.headless_sources import describe_source, given_note, source_notes

    modules = _module_set(tmp_path / "data")
    [source] = given_sources([*(str(path) for path in modules), str(modules[0])])  # data/*.nxs data/*m01.nxs
    assert (len(source.given), len(source.files)) == (4, 3)
    assert given_note(source) == "3 files given for one source (3 module files · 4 frames · stitched): analysed once"
    report: dict = {"frame": source.path}
    describe_source(report, source)
    assert "3 of its files were given: they are one source, analysed once" in report["decisions"][0]["why"]
    assert source_notes([report])[0].startswith("3 files were given for scan_00007_m01.nxs")

    stack = tmp_path / "stack.nxs"
    with h5py.File(stack, "w") as handle:
        handle.create_dataset("/entry/data/data", data=np.zeros((4, H, W), dtype=np.uint16))
    [twice] = given_sources([str(stack), str(stack)])
    assert twice.layout is not None and given_note(twice) == "given 2 times: analysed once"
    report = {"frame": twice.path}
    describe_source(report, twice)
    assert "it was given 2 times: analysed once" in report["decisions"][0]["why"]
    assert source_notes([report]) == ["stack.nxs was given 2 times: it was analysed once.", ""]


def test_a_module_set_in_a_folder_reached_through_a_link_is_one_source(tmp_path: Path) -> None:
    """``current_link/*.nxs``: the reader names the other modules' first file by its real path."""
    modules = _module_set(tmp_path / "real" / "raw")
    link = tmp_path / "current_link"
    try:
        link.symlink_to(tmp_path / "real" / "raw", target_is_directory=True)
    except OSError:
        pytest.skip("this system does not let the test make a link")
    through = [str(link / path.name) for path in modules]
    for given in (through, list(reversed(through)), [through[1]], [str(modules[2]), *through]):
        sources = given_sources(given)
        assert len(sources) == 1, given
        [source] = sources
        assert Path(source.path).name == "scan_00007_m01.nxs" and source.given == tuple(given)
        assert len(source.files) == len(set(Path(text).name for text in given))
        if all(text.startswith(str(link)) for text in given):  # named in the folder as the person named it
            assert source.path == str(link / "scan_00007_m01.nxs")
    # Another file through the link, and the same file by its real path.
    tiff = _tiffs(tmp_path / "real" / "raw", count=1)[0]
    [one] = given_sources([str(link / tiff.name), str(tiff)])
    assert one.path == str(link / tiff.name) and len(one.files) == 1


def test_a_set_given_through_a_link_is_analysed_once(tmp_path: Path) -> None:
    from src.gimap.app.headless_assistant import analyse_frames

    modules = _module_set(tmp_path / "real" / "raw")
    link = tmp_path / "current_link"
    try:
        link.symlink_to(tmp_path / "real" / "raw", target_is_directory=True)
    except OSError:
        pytest.skip("this system does not let the test make a link")
    reports = analyse_frames(
        [str(link / path.name) for path in modules], PipelineOptions(), tmp_path / "out", saved_profiles=False,
    )
    assert len(reports) == 1 and reports[0]["source"]["summary"] == "3 module files · 4 frames · stitched"
    summary = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert "1 frame(s)" in summary and "3 files were given for scan_00007_m01.nxs" in summary
