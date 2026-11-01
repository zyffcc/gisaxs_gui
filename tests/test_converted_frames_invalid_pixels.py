"""Converted frames keep their pixels without data: Format Converter and Batch Export ▸ Converted frames.

The readers hand every frame over as 32-bit float. A Pilatus CBF keeps its gap codes (−1, −2) as values and
the reader masks them; the converter wrote that float array without the mask, so the file read back as a float
frame whose gap codes were *data* (about 2·10⁵ pixels of the CBF below). Now a frame is written in the
type it was stored in (``shared.detector_io.output_frame``) and reads back as its source did: the same pixels
without data (Analyze's rule — masked, not finite or, in integer counts, negative), the same values elsewhere,
and still counts. HDF5 files also carry a NeXus ``pixel_mask``.

The real frame is the Pilatus CBF in the repository's ``TestSAXSdata``; every other case is synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from src.gimap.features.analyze.application.batch_export import BatchChoices, export_frame
from src.gimap.features.analyze.application.frame_preprocessing import floating
from src.gimap.features.analyze.domain import valid_pixels
from src.gimap.features.format_converter.domain.models import ConversionOptions, ConversionRequest
from src.gimap.features.format_converter.infrastructure.adapters import LocalConversionExecutor, LocalSourceRepository
from src.gimap.shared.detector_io import (
    cached_layout,
    forget_layout,
    invalid_pixels,
    load_detector_image,
    output_frame,
    write_frame,
)

ROOT = Path(__file__).resolve().parents[1]
CBF = ROOT / "TestSAXSdata" / "jg_gisaxs_4nm_old_3ml_insitu_ds03_00001_00005.cbf"
needs_cbf = pytest.mark.skipif(not CBF.is_file(), reason="TestSAXSdata is not in this checkout")
SUFFIX = {"TIFF": ".tif", "HDF5": ".h5", "NumPy": ".npy", "CBF": ".cbf"}
H, W = 20, 30


@pytest.fixture(autouse=True)
def _nothing_remembered():
    forget_layout()
    yield
    forget_layout()


def _as_analyze_reads(path: Path):
    """``(values, valid pixels, floating)`` of a file as Analyze reads it (a ``.npy`` as its array says)."""
    if path.suffix == ".npy":
        values = np.load(path)
        metadata = {"stored_dtype": values.dtype.name}
        return values.astype(np.float32), valid_pixels(values, None, negatives_valid=floating(metadata)), False
    image = load_detector_image(path)
    data = np.asarray(image.data)
    return data, valid_pixels(data, image.mask, negatives_valid=floating(image.metadata)), floating(image.metadata)


def _convert(source_path: Path, destination: Path, output_format: str, **options):
    repository = LocalSourceRepository()
    source = repository.inspect_source(source_path)
    request = ConversionRequest(sources=(source,), options=ConversionOptions(output_format, str(destination), **options))
    report = LocalConversionExecutor(repository).execute(request)
    assert report.failed == [], report.failed
    return [Path(item["output"]) for item in report.succeeded]


# -- the Pilatus CBF of TestSAXSdata --------------------------------------------------------------------------------


@needs_cbf
def test_the_cbf_reader_says_the_frame_is_stored_as_integer_counts() -> None:
    image = load_detector_image(CBF)
    assert image.metadata["stored_dtype"] == "int32" and not floating(image.metadata)
    assert int(image.mask.sum()) == int(np.count_nonzero(image.data < 0)) > 100_000  # the gap codes, masked


@needs_cbf
@pytest.mark.parametrize(("output_format", "container"),
                         [("TIFF", False), ("HDF5", False), ("HDF5", True), ("NumPy", False), ("CBF", False)])
def test_a_pilatus_cbf_converted_to_every_format_reads_back_with_the_same_pixels_without_data(
    tmp_path: Path, output_format: str, container: bool
) -> None:
    before, valid_before, kind_before = _as_analyze_reads(CBF)
    without_data = int(np.count_nonzero(~valid_before))
    assert without_data > 100_000  # module gaps and bad pixels of a Pilatus 1M

    [written] = _convert(CBF, tmp_path / "out", output_format, container=container)
    after, valid_after, kind_after = _as_analyze_reads(written)

    assert int(np.count_nonzero(~valid_after)) == without_data  # the masked pixel count before == after
    np.testing.assert_array_equal(valid_after, valid_before)  # … and the same pixels
    np.testing.assert_array_equal(after[valid_after], before[valid_before])  # every value with data as stored
    assert kind_after is kind_before is False  # still integer counts: Poisson errors, dead-pixel test
    if output_format == "HDF5":  # the reader itself masks them (a NeXus pixel_mask): XRR sums none of them
        assert int(load_detector_image(written).mask.sum()) == without_data
    record = json.loads((tmp_path / "out" / "conversion_metadata.json").read_text(encoding="utf-8"))[0]
    values = record["pixel_values"]
    assert (values["stored_dtype"], values["written_dtype"], values["pixels_without_data"]) == (
        "int32", "int32", without_data)
    assert values["pixels_without_data_written_as"] == "their stored codes (-2, -1)" and record["dtype"] == "int32"
    assert ("pixel_mask" in values) is (output_format == "HDF5")


@needs_cbf
def test_the_stored_gap_codes_are_written_as_they_are(tmp_path: Path) -> None:
    import fabio

    [written] = _convert(CBF, tmp_path / "out", "NumPy")
    np.testing.assert_array_equal(np.load(written), np.asarray(fabio.open(str(CBF)).data))  # the file's own values


@needs_cbf
def test_a_single_hdf5_output_offers_no_mask_as_an_image(tmp_path: Path) -> None:
    [written] = _convert(CBF, tmp_path / "out", "HDF5")
    layout = cached_layout(written)
    assert (layout.kind, layout.dataset, layout.alternatives) == ("single_frame", "/entry/data/data", ())
    with h5py.File(written, "r") as handle:
        mask = handle["/entry/instrument/detector/pixel_mask"][()]
        assert handle["/entry/instrument/detector"].attrs["NX_class"] == "NXdetector"
    assert mask.dtype == np.uint32 and set(np.unique(mask)) == {0, 1}  # bit 0, "gap": no data


@needs_cbf
@pytest.mark.parametrize("data_mode", ["scale_uint16", "clip_uint16"])
def test_the_uint16_choices_range_over_the_pixels_with_data_and_mark_the_others_as_no_data(
    tmp_path: Path, data_mode: str
) -> None:
    before, valid_before, _kind = _as_analyze_reads(CBF)
    [written] = _convert(CBF, tmp_path / "out", "HDF5", data_mode=data_mode, preserve_values=False)
    with h5py.File(written, "r") as handle:
        values = handle["/entry/data/data"][()]
    assert values.dtype == np.uint16 and (values[~valid_before] == 65535).all()
    if data_mode == "scale_uint16":  # the gap codes no longer stretch the range: the pixels with data span it
        assert (int(values[valid_before].min()), int(values[valid_before].max())) == (0, 65534)
        expected = valid_before
    else:  # a count cut at the top is no measurement either: 65535 reads as "no data" (NeXus)
        expected = valid_before & (before < 65535)
    image = load_detector_image(written)
    np.testing.assert_array_equal(~image.mask, expected)
    record = json.loads((tmp_path / "out" / "conversion_metadata.json").read_text(encoding="utf-8"))[0]
    assert record["pixel_values"]["pixels_without_data_written_as"] == "65535 (the largest uint16: no data)"
    assert record["pixel_values"]["written_dtype"] == "uint16"


@needs_cbf
@pytest.mark.parametrize("fmt", ["tiff", "edf", "npy", "hdf5"])
def test_batch_export_converted_frames_keep_the_gap_pixels_and_the_values_of_hot_pixels(
    tmp_path: Path, fmt: str
) -> None:
    image = load_detector_image(CBF)
    data = np.asarray(image.data)
    detector_valid = valid_pixels(data, image.mask)
    hot = (100, 200)
    assert detector_valid[hot]
    usable = detector_valid.copy()
    usable[hot] = False  # left out by the analysis (a hot pixel found, a drawn mask): a setting, not the detector
    analysis = SimpleNamespace(
        path=CBF, frame_index=0, frame_count=1, frame_total=1, summed_frames=(), metadata=dict(image.metadata),
        raw_data=None, data=data, valid=usable, raw_valid=detector_valid,
    )
    [written] = export_frame(None, analysis, tmp_path / "batch",
                             BatchChoices(tables=False, frames=True, frame_format=fmt))
    assert written.parent.name == "frames"
    after, valid_after, kind_after = _as_analyze_reads(written)
    np.testing.assert_array_equal(valid_after, detector_valid)
    np.testing.assert_array_equal(after[valid_after], data[detector_valid])
    assert after[hot] == data[hot] and kind_after is False


# -- synthetic frames ------------------------------------------------------------------------------------------------


def _unsigned_stack(path: Path) -> Path:
    """Two uint32 frames of a NeXus detector: its "no data" value in row 0, a dead pixel in its pixel_mask."""
    data = np.arange(2 * H * W, dtype=np.uint32).reshape(2, H, W) + 5
    data[:, 0, :] = np.iinfo(np.uint32).max
    mask = np.zeros((H, W), dtype=np.uint32)
    mask[4, 7] = 2  # bit 1: dead
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/entry/instrument/detector/data", data=data)
        handle["/entry/instrument/detector"].attrs["NX_class"] = "NXdetector"
        handle.create_dataset("/entry/instrument/detector/pixel_mask", data=mask)
    return path


@pytest.mark.parametrize(("output_format", "written_dtype"),
                         [("TIFF", "int32"), ("CBF", "int32"), ("NumPy", "int32"), ("HDF5", "uint32")])
def test_an_unsigned_nexus_frame_marks_its_pixels_without_data_in_every_format(
    tmp_path: Path, output_format: str, written_dtype: str
) -> None:
    source = _unsigned_stack(tmp_path / "stack.nxs")
    before = load_detector_image(source, frame_idx=1)
    valid_before = valid_pixels(before.data, before.mask, negatives_valid=floating(before.metadata))
    assert int(np.count_nonzero(~valid_before)) == W + 1

    written = _convert(source, tmp_path / "out", output_format)[1]
    after, valid_after, kind_after = _as_analyze_reads(written)
    np.testing.assert_array_equal(valid_after, valid_before)
    np.testing.assert_array_equal(after[valid_after], before.data[valid_before])
    assert kind_after is False
    record = json.loads((tmp_path / "out" / "conversion_metadata.json").read_text(encoding="utf-8"))[1]
    assert (record["pixel_values"]["stored_dtype"], record["pixel_values"]["written_dtype"]) == ("uint32", written_dtype)
    # TIFF, CBF and NumPy have no mask: −1 in the smallest signed type that holds every value. HDF5 keeps the type,
    # with the largest value of it ("no data", as NeXus writes it) and the pixel_mask.
    expected = "4294967295 (the largest uint32: no data)" if output_format == "HDF5" else "-1"
    assert record["pixel_values"]["pixels_without_data_written_as"] == expected


def test_a_floating_point_frame_keeps_its_negative_values_as_data(tmp_path: Path) -> None:
    """A dark-subtracted frame: negative values are data; a pixel the reader masks becomes NaN."""
    values = np.linspace(-50.0, 400.0, H * W, dtype=np.float32).reshape(H, W)
    values[3, 3] = np.nan
    mask = np.zeros((H, W), dtype=bool)
    mask[5, 6] = True
    for fmt in ("tiff", "edf", "npy", "hdf5"):
        frame = output_frame(values, fmt, invalid=invalid_pixels(values, mask, "float32"), stored_dtype="float32")
        written = write_frame(tmp_path / f"frame.{fmt}", frame.data, fmt, None, invalid=frame.invalid)
        after, valid_after, kind_after = _as_analyze_reads(written)
        if fmt != "npy":
            assert kind_after is True
            assert int(np.count_nonzero(~valid_after)) == 2 and not valid_after[5, 6] and not valid_after[3, 3]
        np.testing.assert_array_equal(after[~frame.invalid], values[~frame.invalid])
        assert frame.record["pixels_without_data_written_as"] == "NaN" and frame.data.dtype == np.float32


def test_how_an_integer_frame_is_written() -> None:
    gaps = np.zeros((4, 4), dtype=bool)
    gaps[0, 0] = True
    counts = np.full((4, 4), 7.0, dtype=np.float32)
    counts[0, 0] = np.nan  # a pixel the reader set aside: no stored code left
    signed = output_frame(counts, "tiff", invalid=gaps, stored_dtype="int32")
    assert signed.data.dtype == np.int32 and signed.data[0, 0] == -1 and signed.record["pixels_without_data"] == 1
    big = counts.copy()
    big[1, 1] = 70_000.0  # an unsigned frame with values beyond int16: int32
    assert output_frame(big, "cbf", invalid=gaps, stored_dtype="uint16").data.dtype == np.int32
    huge = counts.copy()
    huge[1, 1] = 2.0**33  # beyond int32 (a sum of many frames): int64
    assert output_frame(huge, "tiff", invalid=gaps, stored_dtype="int32").data.dtype == np.int64
    scaled = counts.copy()
    scaled[1, 1] = 7.5  # no longer whole numbers (scaled or corrected): float, NaN on the pixels without data
    as_float = output_frame(scaled, "tiff", invalid=gaps, stored_dtype="int32")
    assert as_float.data.dtype == np.float32 and np.isnan(as_float.data[0, 0]) and as_float.data[1, 1] == 7.5
    in_cbf = output_frame(scaled, "cbf", invalid=gaps, stored_dtype="int32")  # CBF holds integers
    assert in_cbf.data.dtype == np.int32 and in_cbf.data[0, 0] == -1
    clean = output_frame(counts[1:, 1:], "hdf5", stored_dtype="uint16")  # nothing to mark: the stored type
    assert clean.data.dtype == np.uint16 and clean.record["pixels_without_data_written_as"] == "none"
    saturated = counts[1:, 1:].copy()
    saturated[0, 0] = 65535.0  # in HDF5 the largest uint16 reads as "no data": a signed type instead
    assert output_frame(saturated, "hdf5", stored_dtype="uint16").data.dtype == np.int32


def _container(path: Path, frames: list[tuple[np.ndarray, np.ndarray]]) -> Path:
    options = ConversionOptions(output_format="HDF5", destination=str(path.parent), container=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["NX_class"] = "NXroot"
        handle.create_group("entry").attrs["NX_class"] = "NXentry"
        for index, (frame, invalid) in enumerate(frames, start=1):
            LocalConversionExecutor._write_container_frame(handle, frame, {}, index, options, invalid)
    return path


def test_the_container_marks_each_frames_own_pixels_without_data(tmp_path: Path) -> None:
    frames = []
    for index in range(3):
        frame = np.full((H, W), 10 + index, dtype=np.int32)
        invalid = np.zeros((H, W), dtype=bool)
        if index:  # the first frame has none: its row of the mask is empty
            invalid[index, :] = True
            frame[invalid] = -1
        frames.append((frame, invalid))
    path = _container(tmp_path / "converted_images.h5", frames)
    layout = cached_layout(path)
    assert (layout.frames, layout.alternatives) == (3, ())
    for index, (_frame, invalid) in enumerate(frames):
        np.testing.assert_array_equal(load_detector_image(path, frame_idx=index).mask, invalid)


def test_a_container_of_frames_of_several_sizes_has_no_mask_whose_rows_are_not_theirs(tmp_path: Path) -> None:
    small = np.full((H, W), 3, dtype=np.int32)
    gaps = np.zeros((H, W), dtype=bool)
    gaps[0, 0] = True
    small[gaps] = -1
    other = np.full((H + 4, W), 4, dtype=np.int32)
    path = _container(tmp_path / "converted_images.h5", [(small, gaps), (other, np.zeros(other.shape, dtype=bool))])
    with h5py.File(path, "r") as handle:
        assert "pixel_mask" not in handle["/entry"].get("instrument/detector", {})
    image = load_detector_image(path, frame_idx=0)  # the value still says it: −1 in integer counts
    assert not valid_pixels(image.data, image.mask, negatives_valid=floating(image.metadata))[0, 0]


def test_a_numbered_frame_of_a_container_read_on_its_own_gets_its_own_row_of_the_mask(tmp_path: Path) -> None:
    frames = []
    for index in range(3):
        frame = np.full((H, W), 10 + index, dtype=np.int32)
        invalid = np.zeros((H, W), dtype=bool)
        invalid[:, index] = True  # each frame its own gaps
        frame[invalid] = -1
        frames.append((frame, invalid))
    path = _container(tmp_path / "converted_images.h5", frames)
    with h5py.File(path, "r") as handle:
        names = sorted(handle["/entry/data"])
    assert len(names) == 3
    for index, (_frame, invalid) in enumerate(frames):
        np.testing.assert_array_equal(load_detector_image(path, frame_idx=index).mask, invalid)  # the series
        alone = load_detector_image(path, frame_idx=0, dataset_path=f"/entry/data/{names[index]}")
        np.testing.assert_array_equal(alone.mask, invalid)  # read on its own: its row, not row 0


def test_a_big_endian_floating_point_frame_keeps_its_negative_values_as_data(tmp_path: Path) -> None:
    from src.gimap.shared.detector_io.output_values import floating as floating_dtype

    values = np.linspace(-20.0, 300.0, H * W).reshape(H, W).astype(">f4")  # invented, dark-subtracted
    path = tmp_path / "big_endian.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/entry/instrument/detector/data", data=values)
        handle["/entry/instrument/detector"].attrs["NX_class"] = "NXdetector"
    image = load_detector_image(path)
    assert image.metadata["stored_dtype"] == ">f4" and floating(image.metadata)
    negative = values < 0
    assert negative.sum() > 10 and not invalid_pixels(image.data, image.mask, image.metadata["stored_dtype"])[negative].any()
    assert output_frame(image.data, "tiff", stored_dtype=">f4").record["pixels_without_data"] == 0
    for counts in (">i4", ">u2", "int32", "uint16", ""):  # integer counts (and an unknown type) stay counts
        assert not floating_dtype(counts), counts
    assert floating_dtype("float64") and floating_dtype("<f8")


def test_a_floating_point_frame_in_cbf_says_what_it_loses(tmp_path: Path) -> None:
    """CBF holds int32: a fractional part is cut and a value of −1 or less reads back as a gap code."""
    tile = np.array([[1.5, -0.7, 3.2, 10.0], [np.nan, -5.0, 6.0, 2.0]], dtype=np.float32)  # invented values
    values = np.tile(tile, (4, 4))
    frame = output_frame(values, "cbf", stored_dtype="float32")
    record = frame.record
    assert frame.data.dtype == np.int32
    assert record["pixels_without_data"] == 16  # the NaN pixels
    assert record["negative_data_read_as_gap_codes"] == 16  # the −5 pixels
    assert record["values_truncated"] == 48  # 1.5, −0.7 and 3.2
    assert record["pixels_read_back_without_data"] == 32
    written = write_frame(tmp_path / "frame.cbf", frame.data, "cbf", None, invalid=frame.invalid)
    assert int(load_detector_image(written).mask.sum()) == record["pixels_read_back_without_data"]
    chosen = output_frame(values, "cbf", stored_dtype="float32", chosen_dtype="float32")  # the type chosen
    assert chosen.record["negative_data_read_as_gap_codes"] == 16
    whole = output_frame(np.full((4, 4), 7.0, dtype=np.float32), "cbf", stored_dtype="float32").record
    assert whole["pixels_read_back_without_data"] == 0 and "values_truncated" not in whole
    assert "pixels_read_back_without_data" not in output_frame(values, "tiff", stored_dtype="float32").record
