from pathlib import Path

import h5py
import numpy as np
import pytest

from src.gimap.app import AppContext
from src.gimap.features.format_converter.bootstrap import create_format_converter_view_model
from src.gimap.features.format_converter.application import ConvertFile
from src.gimap.features.format_converter.application import NormalizePath
from src.gimap.features.format_converter.domain import (
    ConversionOptions,
    ConversionRequest,
    InputSource,
    select_frame_indices,
    visible_output_formats,
)
from src.gimap.features.format_converter.infrastructure.adapters import (
    LocalConversionExecutor,
    LocalSourceRepository,
)
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)


def _write_nxs(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        data = np.arange(2 * 8 * 6, dtype=np.uint16).reshape(2, 8, 6)
        handle.create_dataset("/entry/instrument/detector/data", data=data)


def test_convert_file_succeeds_without_qapplication(tmp_path: Path) -> None:
    source_path = tmp_path / "scan.nxs"
    destination = tmp_path / "converted"
    _write_nxs(source_path)
    repository = LocalSourceRepository()
    source = repository.inspect_source(source_path)
    source.selected_frames = [1]
    request = ConversionRequest(
        sources=(source,),
        options=ConversionOptions(output_format="NumPy", destination=str(destination)),
    )

    result = ConvertFile(LocalConversionExecutor(repository))(request)

    assert not result.failed
    assert len(result.succeeded) == 1
    raw_frame = np.arange(2 * 8 * 6, dtype=np.uint16).reshape(2, 8, 6)[1]
    expected = np.flip(raw_frame.T, axis=0).astype(np.float32)
    assert np.array_equal(
        np.load(destination / "scan_000002.npy"),
        expected,
    )


def test_convert_file_rejects_invalid_input(tmp_path: Path) -> None:
    request = ConversionRequest(
        sources=(),
        options=ConversionOptions(output_format="NumPy", destination=str(tmp_path)),
    )

    with pytest.raises(ValueError, match="Select at least one"):
        ConvertFile(LocalConversionExecutor())(request)


def test_convert_file_reports_source_file_error(tmp_path: Path) -> None:
    missing_source = InputSource(
        path=str(tmp_path / "missing.tif"),
        file_type="TIFF",
    )
    request = ConversionRequest(
        sources=(missing_source,),
        options=ConversionOptions(
            output_format="NumPy",
            destination=str(tmp_path / "converted"),
        ),
    )

    result = ConvertFile(LocalConversionExecutor())(request)

    assert not result.succeeded
    assert len(result.failed) == 1
    assert result.failed[0]["source"] == missing_source.path
    assert Path(result.report_path).is_file()


def test_format_converter_runs_with_in_memory_app_context(tmp_path: Path) -> None:
    source_path = tmp_path / "standalone.nxs"
    _write_nxs(source_path)
    context = AppContext(
        settings=InMemorySettingsRepository(),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
    )
    view_model = create_format_converter_view_model(context)
    assert view_model.add_paths([str(source_path)]).added == 1
    destination = tmp_path / "output"

    result = view_model.convert(
        ConversionOptions(output_format="NumPy", destination=str(destination))
    )

    assert len(result.succeeded) == 2
    assert view_model.state.destination == str(destination)


def test_normalize_path_use_case_delegates_to_file_repository(tmp_path: Path) -> None:
    repository = LocalSourceRepository()

    normalized = NormalizePath(repository)(tmp_path)

    assert normalized == str(tmp_path)


def test_frame_selection_rules_preserve_all_legacy_modes() -> None:
    assert select_frame_indices(6, "All") == [0, 1, 2, 3, 4, 5]
    assert select_frame_indices(6, "Current frame", current_frame=4) == [3]
    assert select_frame_indices(6, "Frame range", range_start=2, range_end=4) == [1, 2, 3]
    assert select_frame_indices(12, "Custom", custom_frames="1, 5, 8–10") == [
        0,
        4,
        7,
        8,
        9,
    ]
    assert select_frame_indices(6, "Every Nth frame", nth_frame=2) == [0, 2, 4]
    with pytest.raises(ValueError, match="frame range is empty"):
        select_frame_indices(6, "Frame range", range_start=5, range_end=4)


def test_output_format_visibility_preserves_no_op_hiding_rules() -> None:
    assert not visible_output_formats(["TIFF"])["TIFF"]
    assert not visible_output_formats(["CBF"])["CBF"]
    assert not visible_output_formats(["NXS"])["HDF5"]
    assert visible_output_formats(["NXS"], container=True)["HDF5"]
    assert all(visible_output_formats(["NXS", "TIFF"]).values())
