from pathlib import Path

import h5py
import numpy as np
import pytest

from utils.format_converter import (
    ConversionEngine,
    ConversionOptions,
    build_jobs,
    compact_frame_summary,
    inspect_source,
    parse_custom_frames,
)


def _write_nxs(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        data = np.arange(3 * 64 * 48, dtype=np.uint16).reshape(3, 64, 48)
        handle.create_dataset("/entry/instrument/detector/data", data=data)
        handle.create_dataset("/entry/instrument/detector/translation/distance", data=[0, 0])
        handle.create_dataset("/entry/instrument/detector/x_pixel_size", data=172e-6)
        handle.create_dataset("/entry/instrument/detector/y_pixel_size", data=172e-6)


def test_custom_frame_parser_and_summary():
    assert parse_custom_frames("1, 5, 8–10", 12) == [0, 4, 7, 8, 9]
    assert compact_frame_summary([0, 1, 2], 3) == "1–3"
    with pytest.raises(ValueError):
        parse_custom_frames("1, 13", 12)


def test_nxs_inspection_uses_shared_dataset_selection(tmp_path):
    source_path = tmp_path / "scan_001.nxs"
    _write_nxs(source_path)
    source = inspect_source(source_path)
    assert source.file_type == "NXS"
    assert source.dataset_path == "/entry/instrument/detector/data"
    assert source.dataset_shape == (3, 64, 48)
    assert source.selected_frames == [0, 1, 2]


def test_nxs_to_numpy_conversion_and_metadata_report(tmp_path):
    source_path = tmp_path / "scan_001.nxs"
    output_path = tmp_path / "converted"
    _write_nxs(source_path)
    source = inspect_source(source_path)
    source.selected_frames = [0, 2]
    options = ConversionOptions(output_format="NumPy", destination=str(output_path))
    report = ConversionEngine(options).run([source])
    assert len(report.succeeded) == 2
    assert not report.failed
    assert (output_path / "scan_001_000001.npy").is_file()
    assert (output_path / "scan_001_000003.npy").is_file()
    assert (output_path / "conversion_metadata.json").is_file()
    assert Path(report.report_path).is_file()


def test_duplicate_names_require_suffix_or_fail(tmp_path):
    first_path = tmp_path / "a" / "sample.tif"
    second_path = tmp_path / "b" / "sample.tif"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    from fabio.tifimage import TifImage

    TifImage(data=np.ones((32, 32), dtype=np.uint16)).write(str(first_path))
    TifImage(data=np.ones((32, 32), dtype=np.uint16)).write(str(second_path))
    sources = [inspect_source(first_path), inspect_source(second_path)]
    options = ConversionOptions(output_format="NumPy", destination=str(tmp_path / "out"), add_suffix=False)
    with pytest.raises(FileExistsError):
        build_jobs(sources, options)

