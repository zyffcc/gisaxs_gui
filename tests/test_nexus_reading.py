"""Frames of NeXus / HDF5 sources through ``load_detector_image`` (``shared/detector_io/nexus_reading.py``).

The real P03 Lambda 9M module files (``TestSAXSdata/lmbdp03``) pin the stitched frame; every other
file is synthetic: which frame is read, how it is oriented, which pixels hold no data.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.gimap.shared.detector_io import (
    HDF5_SUFFIXES,
    cached_layout,
    detect_nxs_frame_count,
    forget_layout,
    inspect_nexus,
    load_detector_image,
    read_nexus_frame,
    write_frame,
)

ROOT = Path(__file__).resolve().parents[1]
LAMBDA = ROOT / "TestSAXSdata" / "lmbdp03"
DET = "/entry/instrument/detector"
H, W = 20, 30


@pytest.fixture(autouse=True)
def _nothing_remembered():
    forget_layout()
    yield
    forget_layout()


def _marked(frames: int, base: int = 100, dtype=np.int32, rows: int = H, columns: int = W) -> np.ndarray:
    """Frames that tell their index, their rows and their columns apart."""
    grid = np.arange(rows * columns, dtype=np.int64).reshape(rows, columns)
    return np.stack([base * (k + 1) + grid for k in range(frames)]).astype(dtype)


def _write(path: Path, data: np.ndarray, dataset: str = f"{DET}/data", **more) -> Path:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(dataset, data=data)
        for name, value in more.items():
            handle.create_dataset(name, data=value)
    return path


def _module(path: Path, module: int, frames: int, det: str = DET) -> Path:
    data = np.stack([np.full((H, W), 1000 * (k + 1) + module, dtype=np.int32) for k in range(frames)])
    mask = np.zeros((1, H, W), dtype=np.uint32)
    mask[0, :2, :] = 1  # two gap rows per module
    return _write(path, data, f"{det}/data", **{
        f"{det}/pixel_mask": mask,
        f"{det}/translation/distance": np.array([0.0, (module - 1) * (H + 4), 0.0], dtype=np.float32),
    })


# -- the real P03 Lambda 9M frame ----------------------------------------------------------------------------------------


@pytest.mark.skipif(not LAMBDA.is_dir(), reason="TestSAXSdata/lmbdp03 is not in this checkout")
def test_the_real_lambda_module_set_reads_to_the_frame_it_always_gave() -> None:
    files = sorted(LAMBDA.glob("*_m*.nxs"))
    assert len(files) == 11
    image = load_detector_image(files[4])  # any module file: the stitched detector
    data, mask = image.data, image.mask
    assert data.shape == mask.shape == (4727, 3142) and data.dtype == np.float32 and mask.dtype == bool
    assert int((~mask).sum()) == 8_824_977 and int(np.isnan(data).sum()) == 6_027_257
    assert np.array_equal(mask, ~np.isfinite(data))
    assert float(np.nansum(data, dtype=np.float64)) == 16_605_284.0 and float(np.nanmax(data)) == 15110.0
    assert np.unravel_index(np.nanargmax(data), data.shape) == (2569, 2715)
    assert (data[1850, 2656], data[100, 1400], data[361, 1548]) == (1938.0, 1.0, 2159.0)
    assert mask[0, 0] and mask[4726, 3141] and mask[2000, 600]  # no module there
    # Module 8 sits at translation (0, 1965): stored pixel [row 300, column 700] of that file.
    with h5py.File(files[7], "r") as handle:
        stored = handle[f"{DET}/data"][0]
    assert data[4727 - 1 - 700, 1965 + 300] == stored[300, 700] == 4
    metadata = image.metadata
    assert metadata["layout"] == "module_set" and metadata["layout_summary"] == "11 module files · 1 frame · stitched"
    assert [Path(name).name for name in metadata["module_files"]] == [path.name for path in files]
    assert metadata["transformations"] == ["stitched using P03 translations", "canvas transpose", "vertical flip"]
    assert (metadata["orientation"], metadata["stored_dtype"], metadata["sentinel_pixels"]) == ("p03_lambda", "int32", 0)
    assert (metadata["frame_index"], metadata["frame_count"], metadata["layout_warnings"]) == (0, 1, [])
    assert (image.detector_name, image.energy_kev) == ("Lambda", 12.0)
    assert image.pixel_size_x_m == pytest.approx(55e-6) and image.source_path == files[4].resolve()
    with pytest.raises(IndexError, match="frame 2 of 1"):
        load_detector_image(files[0], frame_idx=1)


# -- which frame ---------------------------------------------------------------------------------------------------------------


def test_every_frame_of_a_module_set_is_stitched_from_the_same_index(tmp_path: Path) -> None:
    for module, frames in ((1, 5), (2, 3), (3, 5)):
        _module(tmp_path / f"scan_m{module:02d}.nxs", module, frames)
    for frame in range(3):
        image = load_detector_image(tmp_path / "scan_m03.nxs", frame_idx=frame)
        assert image.data.shape == (W, 3 * H + 8)  # three modules and two gaps of 4, transposed
        values = sorted(set(np.unique(image.data[np.isfinite(image.data)]).tolist()))
        assert values == [1000 * (frame + 1) + module for module in (1, 2, 3)]
        assert image.metadata["frame_index"] == frame and image.metadata["frame_count"] == 3
        assert len(image.metadata["module_files"]) == 3
        assert image.metadata["layout_warnings"] == [
            "Module files hold different numbers of frames (m02: 3; the others 5): frames after frame 3 are left out"]
        # Per module two masked rows, and the gaps between the modules: all invalid and NaN.
        assert int(image.mask.sum()) == 3 * 2 * W + 2 * 4 * W and np.array_equal(image.mask, np.isnan(image.data))
    for beyond in (3, 4, 99, -1):  # frames 4 and 5 exist in two modules only: never the last frame instead
        with pytest.raises(IndexError, match=f"frame {beyond + 1} of 3"):
            load_detector_image(tmp_path / "scan_m01.nxs", frame_idx=beyond)


def test_a_stack_gives_each_frame_and_refuses_one_beyond_the_end(tmp_path: Path) -> None:
    stored = _marked(6)
    path = _write(tmp_path / "insitu.nxs", stored)
    for frame in (0, 3, 5):
        image = load_detector_image(path, frame_idx=frame)
        np.testing.assert_array_equal(image.data, stored[frame].astype(np.float32))
        assert not image.mask.any() and image.metadata["frame_index"] == frame and image.metadata["frame_count"] == 6
    for beyond in (6, 99):
        with pytest.raises(IndexError, match=f"frame {beyond + 1} of 6"):
            load_detector_image(path, frame_idx=beyond)
    with pytest.raises(IndexError):
        read_nexus_frame(cached_layout(path), -1)
    single = _write(tmp_path / "one.nxs", stored[2])  # a plain 2-D frame
    np.testing.assert_array_equal(load_detector_image(single).data, stored[2])
    with pytest.raises(IndexError, match="frame 2 of 1"):
        load_detector_image(single, frame_idx=1)


def test_a_grid_of_frames_is_read_in_storage_order_one_frame_at_a_time(tmp_path: Path) -> None:
    stored = _marked(12).reshape(3, 4, H, W)
    path = _write(tmp_path / "mesh.nxs", stored)
    assert detect_nxs_frame_count(path) == 12
    for frame in (0, 5, 11):
        image = load_detector_image(path, frame_idx=frame)
        np.testing.assert_array_equal(image.data, stored[np.unravel_index(frame, (3, 4))])
    assert image.metadata["layout_warnings"] == ["A 3×4 grid of frames is read in storage order as 12 frames"]
    with pytest.raises(IndexError, match="frame 13 of 12"):
        load_detector_image(path, frame_idx=12)


def _eiger(folder: Path, suffix: str, virtual: bool) -> tuple[Path, np.ndarray]:
    stored = _marked(8, dtype=np.uint32)
    stored[:, 8:10, :] = np.iinfo(np.uint32).max  # the module gap, as the detector writes it
    master = folder / f"run7_master{suffix}"
    with h5py.File(master, "w") as handle:
        handle.attrs["default"] = "entry"
        entry = handle.create_group("entry")
        entry.attrs["NX_class"], entry.attrs["default"] = "NXentry", "data"
        data = entry.create_group("data")
        data.attrs["NX_class"], data.attrs["signal"] = "NXdata", ("data" if virtual else "data_000001")
        detector = entry.create_group("instrument/detector")
        detector.attrs["NX_class"] = "NXdetector"
        detector.create_dataset("description", data="Dectris EIGER2 X 4M")
        distance = detector.create_dataset("detector_distance", data=3200.0)
        distance.attrs["units"] = "mm"
        pixel_mask = np.zeros((H, W), dtype=np.uint32)
        pixel_mask[0, 0] = 2  # one dead pixel, in the Dectris place for the mask
        detector.create_dataset("detectorSpecific/pixel_mask", data=pixel_mask)
        detector.create_dataset("detectorSpecific/flatfield", data=np.ones((H, W), dtype=np.float32))
        layout, start = h5py.VirtualLayout(shape=(8, H, W), dtype=np.uint32), 0
        for number, count in enumerate((3, 3, 2), start=1):
            name = f"run7_data_{number:06d}.h5"
            with h5py.File(folder / name, "w") as chunk:
                chunk.create_dataset("/entry/data/data", data=stored[start:start + count], chunks=(1, H, W))
            data[f"data_{number:06d}"] = h5py.ExternalLink(name, "/entry/data/data")
            layout[start:start + count] = h5py.VirtualSource(name, "/entry/data/data", shape=(count, H, W))
            start += count
        if virtual:
            data.create_virtual_dataset("data", layout, fillvalue=0)
    return master, stored


@pytest.mark.parametrize("suffix", [".h5", ".nxs"])
@pytest.mark.parametrize("virtual", [False, True])
def test_an_eiger_master_reads_every_frame_of_its_data_files(tmp_path: Path, suffix: str, virtual: bool) -> None:
    master, stored = _eiger(tmp_path, suffix, virtual)
    assert detect_nxs_frame_count(master) == 8
    for frame in (0, 2, 3, 5, 6, 7):  # across the chunk borders 3 | 3 | 2
        image = load_detector_image(master, frame_idx=frame)
        expected = stored[frame].astype(np.float32)
        gap = np.zeros((H, W), dtype=bool)
        gap[8:10, :] = True
        gap[0, 0] = True
        np.testing.assert_array_equal(image.mask, gap)
        np.testing.assert_array_equal(image.data[~gap], expected[~gap])  # as stored: row 0 on top
        assert np.isnan(image.data[gap]).all()
        assert image.metadata["sentinel_pixels"] == 2 * W and image.metadata["stored_dtype"] == "uint32"
        assert image.metadata["layout"] == "chunked_master" and image.metadata["transformations"] == []
        assert [Path(name).name for name in image.metadata["data_files"]] == [
            "run7_data_000001.h5", "run7_data_000002.h5", "run7_data_000003.h5"]
    assert image.metadata["dataset_path"] == ("/entry/data/data" if virtual else "/entry/data/data_000003")
    assert image.metadata["format"] == ("nxs" if suffix == ".nxs" else "hdf5")
    assert image.detector_name == "Dectris EIGER2 X 4M" and image.distance_m == pytest.approx(3.2)
    with pytest.raises(IndexError, match="frame 9 of 8"):
        load_detector_image(master, frame_idx=8)


# -- orientation ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(24, 36), (36, 24), (32, 32)])
def test_a_plain_image_is_read_as_stored_and_a_lambda_file_gets_the_p03_view(tmp_path: Path, shape) -> None:
    stored = _marked(2, rows=shape[0], columns=shape[1])
    plain = load_detector_image(_write(tmp_path / "plain.nxs", stored), frame_idx=1)
    np.testing.assert_array_equal(plain.data, stored[1])  # row 0 on top, like the CBF or TIFF of the detector
    assert plain.metadata["orientation"] == "as_stored" and plain.metadata["transformations"] == []
    expected = np.flipud(stored[1].T)  # the same for a square module: nothing depends on shape == (y, x)
    evidence = {
        "position": {f"{DET}/translation/distance": np.zeros(3, dtype=np.float32)},
        "layout": {f"{DET}/layout": f"{shape[1]}X{shape[0]}"},
        "name": {f"{DET}/description": "Lambda"},
        "local_name": {f"{DET}/local_name": "haspp03LAMBDA750k"},
    }
    for label, more in evidence.items():
        image = load_detector_image(_write(tmp_path / f"lambda_{label}.nxs", stored, **more), frame_idx=1)
        np.testing.assert_array_equal(image.data, expected, err_msg=label)
        assert image.metadata["orientation"] == "p03_lambda"
        assert image.metadata["transformations"] == ["canvas transpose", "vertical flip"]


def test_a_module_the_layout_says_is_stored_turned_is_turned_back(tmp_path: Path) -> None:
    stored = _marked(1, rows=24, columns=36)
    turned = _write(tmp_path / "turned.nxs", stored, **{f"{DET}/layout": "24X36"})  # 24 columns × 36 rows
    image = load_detector_image(turned)
    np.testing.assert_array_equal(image.data, np.flipud(stored[0]))  # module transpose, canvas transpose, flip
    assert image.metadata["transformations"] == ["module transpose", "canvas transpose", "vertical flip"]


def test_a_single_module_is_placed_where_its_translation_says(tmp_path: Path) -> None:
    stored = _marked(1)
    path = _write(tmp_path / "module.nxs", stored, **{f"{DET}/translation/distance": [5.0, 3.0, 0.0]})
    image = load_detector_image(path)
    canvas = np.full((H + 3, W + 5), np.nan, dtype=np.float32)
    canvas[3:, 5:] = stored[0]
    np.testing.assert_array_equal(image.data, np.flipud(canvas.T))
    assert image.metadata["transformations"] == ["placed at its P03 translation", "canvas transpose", "vertical flip"]
    assert int(image.mask.sum()) == canvas.size - H * W


@pytest.mark.parametrize("shape", [(24, 36), (32, 32), (8, 12)])
@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_gimaps_own_hdf5_frame_export_reads_back_unchanged(tmp_path: Path, shape, dtype) -> None:
    frame = (np.arange(shape[0] * shape[1]).reshape(shape) - 7).astype(dtype)
    path = write_frame(tmp_path / "exported.h5", frame, "hdf5", {"source_file": "x.cbf"})
    image = load_detector_image(path)
    np.testing.assert_array_equal(image.data, frame.astype(np.float32))
    assert not image.mask.any() and image.metadata["stored_dtype"] == str(np.dtype(dtype))
    assert image.metadata["dataset_path"] == "/entry/data/data" and image.metadata["layout"] == "single_frame"


@pytest.mark.parametrize("suffix", HDF5_SUFFIXES)
def test_every_hdf5_suffix_is_read(tmp_path: Path, suffix: str) -> None:
    stored = _marked(3)
    image = load_detector_image(_write(tmp_path / f"stack{suffix}", stored), frame_idx=2)
    np.testing.assert_array_equal(image.data, stored[2])
    with pytest.raises(ValueError, match="Unsupported detector image"):
        load_detector_image(_write(tmp_path / "stack.xyz", stored))


def test_the_header_beam_centre_follows_the_orientation(tmp_path: Path) -> None:
    stored = _marked(1)
    center = {f"{DET}/beam_center_x": [12.0], f"{DET}/beam_center_y": [7.0]}
    plain = load_detector_image(_write(tmp_path / "plain.nxs", stored, **center))
    assert plain.metadata["header_beam_center"] == (12.0, 7.0)  # (column, row) of the stored image
    turned = load_detector_image(_write(tmp_path / "lambda.nxs", stored, **center, **{f"{DET}/description": "Lambda"}))
    assert turned.metadata["header_beam_center"] == (7.0, W - 12.0)


# -- pixels without data -------------------------------------------------------------------------------------------------------


def test_the_mask_and_the_module_positions_are_found_next_to_the_data(tmp_path: Path) -> None:
    """The same three modules at ``/entry/instrument/detector`` and under another name read the same."""
    images = []
    for label, det in (("default", DET), ("renamed", "/scan/instrument/lambda")):
        folder = tmp_path / label
        folder.mkdir()
        for module in (1, 2, 3):
            path = _module(folder / f"run_m{module:02d}.nxs", module, 2, det)
            with h5py.File(path, "a") as handle:
                size = handle.create_dataset(f"{det}/x_pixel_size", data=[55.0])
                size.attrs["units"] = "micrometer"
                energy = handle.create_dataset(f"{det}/collection/beam_energy", data=[12000.0])
                energy.attrs["units"] = "eV"
        images.append(load_detector_image(folder / "run_m01.nxs", frame_idx=1))
    default, renamed = images
    assert renamed.metadata["dataset_path"] == "/scan/instrument/lambda/data"
    np.testing.assert_array_equal(default.data, renamed.data)
    np.testing.assert_array_equal(default.mask, renamed.mask)
    assert renamed.data.shape == (W, 3 * H + 8) and int(renamed.mask.sum()) == 3 * 2 * W + 2 * 4 * W
    assert renamed.pixel_size_x_m == pytest.approx(55e-6) and renamed.energy_kev == pytest.approx(12.0)


@pytest.mark.parametrize("dtype", [np.uint32, np.uint16, np.uint8])
def test_the_no_data_value_of_unsigned_pixels_is_masked_and_counted(tmp_path: Path, dtype) -> None:
    stored = np.full((3, H, W), 10, dtype=dtype)
    stored[:, 8:12, :] = np.iinfo(dtype).max  # 4 gap rows
    stored[1, 0, 0] = np.iinfo(dtype).max     # and one saturated pixel in frame 2 only
    path = _write(tmp_path / "eiger.nxs", stored, **{f"{DET}/saturation_value": np.array(5, dtype=dtype)})
    image = load_detector_image(path, frame_idx=0)
    assert int(image.mask.sum()) == 4 * W == image.metadata["sentinel_pixels"]
    assert float(image.data[~image.mask].sum()) == 10.0 * (H * W - 4 * W)  # the saturation value is not applied here
    assert np.isnan(image.data[image.mask]).all() and image.metadata["stored_dtype"] == str(np.dtype(dtype))
    assert load_detector_image(path, frame_idx=1).metadata["sentinel_pixels"] == 4 * W + 1


def test_signed_and_float_pixels_keep_their_values(tmp_path: Path) -> None:
    signed = np.full((1, H, W), 10, dtype=np.int32)
    signed[0, 8:12, :] = -1                       # a Pilatus-style gap code: Analyze's validity rule, not the reader's
    signed[0, 0, 0] = np.iinfo(np.int32).max      # not a sentinel in signed data
    image = load_detector_image(_write(tmp_path / "signed.nxs", signed))
    np.testing.assert_array_equal(image.data, signed[0].astype(np.float32))
    assert not image.mask.any() and image.metadata["sentinel_pixels"] == 0 and image.metadata["stored_dtype"] == "int32"


def test_a_mask_of_another_shape_is_not_applied(tmp_path: Path) -> None:
    stored = _marked(2)
    path = _write(tmp_path / "odd.nxs", stored, **{f"{DET}/pixel_mask": np.ones((H + 1, W), dtype=np.uint32)})
    assert not load_detector_image(path).mask.any()
    per_frame = np.zeros((2, H, W), dtype=np.uint32)
    per_frame[1, 3, 4] = 1
    path = _write(tmp_path / "per_frame.nxs", stored, **{f"{DET}/pixel_mask": per_frame})
    assert not load_detector_image(path, frame_idx=0).mask.any()
    assert load_detector_image(path, frame_idx=1).mask[3, 4] and int(load_detector_image(path, frame_idx=1).mask.sum()) == 1


def test_the_callers_dataset_is_read_and_a_file_without_an_image_says_so(tmp_path: Path) -> None:
    path = tmp_path / "two.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/entry/saxs/data", data=_marked(2, base=100))
        handle.create_dataset("/entry/waxs/data", data=_marked(3, base=500))
    automatic = load_detector_image(path)
    assert automatic.metadata["dataset_path"] == "/entry/saxs/data"
    assert automatic.metadata["layout_warnings"] == [
        "2 image datasets in this file: /entry/saxs/data is shown, not /entry/waxs/data "
        "(Format Converter and Geometry Calibration let you choose the dataset)"]
    chosen = load_detector_image(path, frame_idx=2, dataset_path="/entry/waxs/data")
    np.testing.assert_array_equal(chosen.data, _marked(3, base=500)[2])
    assert chosen.metadata["dataset_path"] == "/entry/waxs/data" and chosen.metadata["layout_warnings"] == []
    assert inspect_nexus(path).alternatives == ("/entry/waxs/data",)
    curve = _write(tmp_path / "curve.h5", np.arange(100.0), "/entry/result/intensity")
    with pytest.raises(ValueError, match="curve.h5 holds no detector image"):
        load_detector_image(curve)


# -- compressed data -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("codec", ["bitshuffle_lz4", "lz4"])
def test_bitshuffle_and_lz4_data_read_in_a_session_that_only_imports_the_reader(tmp_path: Path, codec: str) -> None:
    hdf5plugin = pytest.importorskip("hdf5plugin")
    stored = _marked(4, dtype=np.uint32)
    path = tmp_path / "eiger_compressed.nxs"
    filter_ = hdf5plugin.Bitshuffle(nelems=0, cname="lz4") if codec == "bitshuffle_lz4" else hdf5plugin.LZ4()
    with h5py.File(path, "w") as handle:
        handle.create_dataset(f"{DET}/data", data=stored, chunks=(1, H, W), **filter_)
    script = (
        "import sys\n"
        "from src.gimap.shared.detector_io import load_detector_image\n"
        "assert 'fabio' not in sys.modules, 'fabio was imported: it would register the filters itself'\n"
        "image = load_detector_image(sys.argv[1], frame_idx=2)\n"
        "print(image.data.shape, float(image.data.sum()))\n"
    )
    done = subprocess.run([sys.executable, "-c", script, str(path)], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.split() == [f"({H},", f"{W})", str(float(stored[2].sum()))]
