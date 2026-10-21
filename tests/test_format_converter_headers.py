"""The Format Converter writes every output format with the metadata kept (the default).

A CBF or EDF header holds ASCII only, and the record of a frame does not: the reader's summary of a NeXus
source has a middle dot ("1 file · 4 frames"), and a folder may be named in any script. The record is
written as ASCII JSON there, so it reads back exactly; a write that fails leaves no empty file behind.
This is the shared writer's doing (``shared/detector_io/writing.py``), so Analyze's frame export has it too;
a CBF file whose own name is outside ASCII is written as well. Synthetic files only.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.gimap.features.format_converter.domain.models import ConversionOptions, ConversionRequest
from src.gimap.features.format_converter.infrastructure.adapters import (
    LocalConversionExecutor,
    LocalSourceRepository,
)
from src.gimap.features.format_converter.infrastructure.adapters import local_files
from src.gimap.shared.detector_io import forget_layout, load_detector_image, write_frame

H, W = 20, 30
SUFFIX = {"CBF": ".cbf", "EDF": ".edf", "TIFF": ".tif", "NumPy": ".npy", "HDF5": ".h5"}


@pytest.fixture(autouse=True)
def _nothing_remembered():
    forget_layout()
    yield
    forget_layout()


def _stack(path: Path, frames: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        data = np.arange(frames, dtype=np.uint16)[:, None, None] + np.zeros((frames, H, W), dtype=np.uint16) + 7
        handle.create_dataset("/entry/data/data", data=data)
    return path


def _convert(sources, destination: Path, output_format: str, **options):
    request = ConversionRequest(
        sources=tuple(sources),
        options=ConversionOptions(output_format=output_format, destination=str(destination), **options),
    )
    return LocalConversionExecutor(LocalSourceRepository()).execute(request)


def _header_record(path: Path) -> dict:
    """The record in the text header of a CBF file (the lines before the binary section)."""
    head = path.read_bytes().split(b"--CIF-BINARY-FORMAT-SECTION--")[0]
    assert head.isascii()  # that is all a CBF header holds
    line = next(line for line in head.decode("ascii").splitlines() if line.startswith("# GIMaP_metadata "))
    return json.loads(line[len("# GIMaP_metadata "):])


def test_metadata_is_kept_by_default() -> None:
    assert ConversionOptions("CBF", "somewhere").preserve_metadata is True


@pytest.mark.parametrize("output_format", ["CBF", "TIFF", "NumPy", "HDF5"])
def test_a_nexus_stack_is_written_in_every_format_with_its_metadata(tmp_path: Path, output_format: str) -> None:
    source = LocalSourceRepository().inspect_source(_stack(tmp_path / "stack.nxs"))
    report = _convert([source], tmp_path / "out", output_format)
    assert report.failed == [] and len(report.succeeded) == 3
    written = sorted((tmp_path / "out").glob(f"*{SUFFIX[output_format]}"))
    assert [path.name for path in written] == [f"stack_{frame:06d}{SUFFIX[output_format]}" for frame in (1, 2, 3)]
    assert all(path.stat().st_size > 0 for path in written)
    for frame, path in enumerate(written):
        expected = np.asarray(load_detector_image(tmp_path / "stack.nxs", frame_idx=frame).data)
        back = np.load(path) if output_format == "NumPy" else np.asarray(load_detector_image(path).data)
        np.testing.assert_array_equal(back, expected)


def test_the_cbf_header_holds_the_record_of_the_frame_exactly(tmp_path: Path) -> None:
    source = LocalSourceRepository().inspect_source(_stack(tmp_path / "stack.nxs"))
    assert not _convert([source], tmp_path / "out", "CBF").failed
    record = _header_record(tmp_path / "out" / "stack_000002.cbf")
    assert record["layout_summary"] == "1 file · 3 frames"  # the middle dot, back as it was
    assert (record["frame_number"], record["frame_index"], record["dataset_path"]) == (2, 1, "/entry/data/data")
    assert record["source"] == str(tmp_path / "stack.nxs")
    assert b"1 file \\u00b7 3 frames" in (tmp_path / "out" / "stack_000002.cbf").read_bytes()  # as its JSON escape
    # The same record is in the side file, readable as it is.
    side = json.loads((tmp_path / "out" / "conversion_metadata.json").read_text(encoding="utf-8"))
    assert side[1]["layout_summary"] == record["layout_summary"]


def test_a_folder_named_outside_ascii_is_kept_exactly_in_the_cbf_header(tmp_path: Path) -> None:
    """The baseline failed here as well: the source path is part of the record of every frame."""
    folder = tmp_path / "数据 样品"
    source = LocalSourceRepository().inspect_source(_stack(folder / "scan.nxs", 2))
    report = _convert([source], folder / "out", "CBF")
    assert report.failed == [] and len(report.succeeded) == 2
    written = folder / "out" / "scan_000001.cbf"
    np.testing.assert_array_equal(
        np.asarray(load_detector_image(written).data), np.asarray(load_detector_image(folder / "scan.nxs").data)
    )
    record = _header_record(written)
    assert record["source"] == str(folder / "scan.nxs") and record["output"] == str(written)


def test_an_edf_header_gets_the_record_in_ascii_too(tmp_path: Path) -> None:
    """EDF is not offered as an output yet; its writer is there, and fabio drops every character outside
    ASCII from an EDF header without a word (a path would come back damaged)."""
    import fabio

    written = tmp_path / "frame.edf"
    LocalConversionExecutor._write_frame(
        written, np.ones((H, W), dtype=np.int32), {"source": "/数据/样品/a.cbf"}, ConversionOptions("EDF", str(tmp_path))
    )
    text = fabio.open(str(written)).header["GIMaP_metadata"]  # fabio returns the value without its outer braces
    assert text.isascii() and json.loads("{" + text + "}") == {"source": "/数据/样品/a.cbf"}


def test_the_pixels_of_an_ascii_header_file_are_those_of_the_shared_writer(tmp_path: Path) -> None:
    """Float data with holes and integer data: the frame is stored exactly as ``write_frame`` stores it."""
    holes = np.linspace(-3.0, 9.0, H * W).reshape(H, W)
    holes[2, 3], holes[4, 5] = np.nan, np.inf
    for label, data in (("float", holes), ("int", (np.arange(H * W).reshape(H, W) % 251).astype(np.int32))):
        for fmt, suffix in (("cbf", ".cbf"), ("edf", ".edf")):
            plain, kept = tmp_path / f"{label}_plain{suffix}", tmp_path / f"{label}_kept{suffix}"
            write_frame(plain, data, fmt, None)
            LocalConversionExecutor._write_frame(
                kept, data, {"layout_summary": "1 file · 3 frames"}, ConversionOptions(fmt.upper(), str(tmp_path))
            )
            import fabio

            first, second = fabio.open(str(plain)).data, fabio.open(str(kept)).data
            assert first.dtype == second.dtype
            np.testing.assert_array_equal(first, second)


def test_a_write_that_fails_leaves_no_empty_file_and_the_other_frames_are_written(tmp_path: Path, monkeypatch) -> None:
    source = LocalSourceRepository().inspect_source(_stack(tmp_path / "stack.nxs"))
    real = local_files.write_frame

    def failing(path, data, fmt, metadata=None):
        if Path(path).name.endswith("000002.tif"):
            Path(path).write_bytes(b"")  # what a writer leaves when it fails after opening its file
            raise OSError("disk says no")
        return real(path, data, fmt, metadata)

    monkeypatch.setattr(local_files, "write_frame", failing)
    report = _convert([source], tmp_path / "out", "TIFF")
    assert [item["frame"] for item in report.failed] == [2] and report.failed[0]["error"] == "disk says no"
    assert sorted(path.name for path in (tmp_path / "out").glob("*.tif")) == ["stack_000001.tif", "stack_000003.tif"]


def test_a_failed_write_does_not_remove_a_file_that_was_there_and_is_still_whole(tmp_path: Path, monkeypatch) -> None:
    """Overwriting is the person's choice; a writer that fails before touching the file leaves it alone."""
    before = tmp_path / "kept.tif"
    before.write_bytes(b"an older frame")

    def failing(path, data, fmt, metadata=None):
        raise ValueError("refused before opening")

    monkeypatch.setattr(local_files, "write_frame", failing)
    with pytest.raises(ValueError, match="refused before opening"):
        LocalConversionExecutor._write_frame(before, np.zeros((4, 4)), {}, ConversionOptions("TIFF", str(tmp_path)))
    assert before.read_bytes() == b"an older frame"


# -- the shared writer (Analyze's frame export uses it too) ----------------------------------------------------------

RECORD = {"source_file": "/数据/样品 A/scan_00007_m01.nxs", "layout_summary": "11 module files · 240 frames · stitched",
          "frame": 4, "note": "αi = 0.4°"}


@pytest.mark.parametrize("fmt", ["cbf", "edf", "tiff", "hdf5", "npy"])
def test_the_shared_writer_takes_a_record_outside_ascii_in_every_format(tmp_path: Path, fmt: str) -> None:
    data = (np.arange(H * W).reshape(H, W) % 97).astype(np.int32)
    written = write_frame(tmp_path / f"frame{local_files_suffix(fmt)}", data, fmt, RECORD)
    assert written.stat().st_size > 0 and sorted(path.name for path in tmp_path.iterdir()) == [written.name]
    back = np.load(written) if fmt == "npy" else np.asarray(load_detector_image(written).data)
    np.testing.assert_array_equal(back, data)
    if fmt == "cbf":
        assert _header_record(written) == RECORD  # exactly, from ASCII text
    elif fmt == "edf":
        import fabio

        text = fabio.open(str(written)).header["GIMaP_metadata"]  # fabio returns the value without its outer braces
        assert text.isascii() and json.loads("{" + text + "}") == RECORD
    elif fmt == "hdf5":
        with h5py.File(written, "r") as handle:
            text = handle["/entry/data/data"].attrs["metadata_json"]
        assert json.loads(text) == RECORD and "样品" in text  # an HDF5 attribute holds any text: as it always was


def local_files_suffix(fmt: str) -> str:
    from src.gimap.shared.detector_io import frame_suffix

    return frame_suffix(fmt)


@pytest.mark.parametrize("name", ["样品_0001.cbf", "Probe_größer_0002.cbf", "плёнка 3.cbf"])
def test_a_cbf_file_named_outside_ascii_is_written_and_reads_back(tmp_path: Path, name: str) -> None:
    """fabio writes the stem of the name into the text of the file, as ASCII: such a file could not be written."""
    data = (np.arange(H * W).reshape(H, W) % 251).astype(np.int32)
    written = write_frame(tmp_path / "数据" / name, data, "cbf", RECORD)
    assert written.name == name and [path.name for path in written.parent.iterdir()] == [name]  # no other file is left
    np.testing.assert_array_equal(np.asarray(load_detector_image(written).data), data)
    assert _header_record(written) == RECORD
    # Written again over itself: the same file, still alone in its folder.
    write_frame(written, data + 1, "cbf", None)
    np.testing.assert_array_equal(np.asarray(load_detector_image(written).data), data + 1)
    assert [path.name for path in written.parent.iterdir()] == [name]


@pytest.mark.parametrize("name", ["frame_0001.cbf", "样品_0001.cbf"])
def test_a_shared_write_that_fails_leaves_no_file_behind(tmp_path: Path, monkeypatch, name: str) -> None:
    from fabio.cbfimage import CbfImage

    def failing(self, fname):
        Path(fname).write_bytes(b"")  # what a writer leaves when it fails after opening its file
        raise OSError("disk says no")

    monkeypatch.setattr(CbfImage, "write", failing)
    with pytest.raises(OSError, match="disk says no"):
        write_frame(tmp_path / name, np.zeros((H, W), dtype=np.int32), "cbf", RECORD)
    assert list(tmp_path.iterdir()) == []
    older = tmp_path / name
    older.write_bytes(b"an older frame")
    with pytest.raises(OSError, match="disk says no"):
        write_frame(older, np.zeros((H, W), dtype=np.int32), "cbf", RECORD)
    if name.isascii():  # the failed write emptied it: nothing is left that looks like a frame
        assert list(tmp_path.iterdir()) == []
    else:  # written under another name first: the older frame is whole
        assert [path.name for path in tmp_path.iterdir()] == [name] and older.read_bytes() == b"an older frame"


def test_the_converter_writes_cbf_through_the_shared_writer(tmp_path: Path, monkeypatch) -> None:
    """No writer of its own beside the shared one: what the shared writer stores is what the converter stores."""
    calls = []
    real = local_files.write_frame
    monkeypatch.setattr(local_files, "write_frame", lambda *args: (calls.append(args[2]), real(*args))[1])
    source = LocalSourceRepository().inspect_source(_stack(tmp_path / "数据" / "stack.nxs", 2))
    assert not _convert([source], tmp_path / "out", "CBF").failed
    assert calls == ["cbf", "cbf"]
    assert not hasattr(local_files, "write_with_ascii_header") and not hasattr(local_files, "ASCII_HEADER_FORMATS")
