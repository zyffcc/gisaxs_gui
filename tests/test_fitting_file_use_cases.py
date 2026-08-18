from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from src.gimap.features.fitting.application import (
    ExportFitResult,
    ExportFitResultRequest,
    LoadCurve,
    LoadCurveRequest,
    LoadScatteringFile,
    LoadScatteringFileRequest,
    InspectScatteringSequence,
    ManageRemoteFileCache,
    ManageFittingParameterFiles,
    SaveFittingLog,
    CheckFittingDependency,
)
from src.gimap.features.fitting.infrastructure.adapters import (
    LocalCurveRepository,
    LocalFitResultRepository,
    LocalScatteringFileRepository,
    LocalRemoteFileCacheAdapter,
    LocalFittingParameterFileRepository,
    LocalFittingLogRepository,
    ImportlibFittingDependencyAvailabilityAdapter,
)


def test_load_curve_succeeds_without_qapplication(tmp_path):
    source = tmp_path / "curve.dat"
    source.write_text("# q I err\n0.1 10 1\n0.2 20 2\n", encoding="utf-8")

    outcome = LoadCurve(LocalCurveRepository()).execute(
        LoadCurveRequest(source, q_source_unit="angstrom")
    )

    assert outcome.succeeded
    np.testing.assert_allclose(outcome.value.q, [0.1, 0.2])
    np.testing.assert_allclose(outcome.value.intensity, [10.0, 20.0])
    np.testing.assert_allclose(outcome.value.error, [1.0, 2.0])
    assert outcome.value.source_path == str(source.resolve())


def test_load_curve_returns_structured_invalid_and_missing_errors(tmp_path):
    unsupported = tmp_path / "curve.csv"
    unsupported.write_text("0.1,10\n0.2,20\n", encoding="utf-8")
    use_case = LoadCurve(LocalCurveRepository())

    invalid = use_case.execute(LoadCurveRequest(unsupported))
    missing = use_case.execute(LoadCurveRequest(tmp_path / "missing.dat"))

    assert invalid.error.code == "unsupported_format"
    assert invalid.error.path == str(unsupported)
    assert missing.error.code == "not_found"


def test_load_tiff_stack_preserves_natural_order_and_sum(tmp_path):
    paths = [tmp_path / name for name in ("frame1.tif", "frame2.tif", "frame10.tif")]
    for value, path in enumerate(paths, start=1):
        Image.fromarray(np.full((4, 6), value, dtype=np.uint16)).save(path)

    outcome = LoadScatteringFile(LocalScatteringFileRepository()).execute(
        LoadScatteringFileRequest(paths[1], stack_count=99)
    )

    assert outcome.succeeded
    np.testing.assert_allclose(outcome.value.image, np.full((4, 6), 5.0))
    assert [path.name for path in outcome.value.source_files] == ["frame2.tif", "frame10.tif"]
    assert outcome.value.metadata["stack_count"] == 2


def test_load_scattering_file_returns_structured_format_error(tmp_path):
    source = tmp_path / "image.png"
    source.write_bytes(b"not an image")

    outcome = LoadScatteringFile(LocalScatteringFileRepository()).execute(
        LoadScatteringFileRequest(source)
    )

    assert outcome.error.code == "unsupported_format"
    assert outcome.error.details["operation"] == "read"


def test_inspect_scattering_sequence_uses_repository_without_qapplication(tmp_path):
    paths = [tmp_path / f"scan_m0{index}.nxs" for index in (1, 2)]
    for path in paths:
        with h5py.File(path, "w") as handle:
            handle.create_dataset(
                "/entry/instrument/detector/data",
                data=np.zeros((3, 4, 6), dtype=np.float32),
            )

    info = InspectScatteringSequence(LocalScatteringFileRepository()).execute(paths[1])

    assert info.logical_path == paths[0].resolve()
    assert info.series_paths == tuple(path.resolve() for path in paths)
    assert info.frame_count == 3
    assert info.uses_internal_frames


def test_export_fit_result_preserves_legacy_txt_and_csv_format(tmp_path):
    use_case = ExportFitResult(LocalFitResultRepository())
    common = dict(
        q=np.array([0.1, 0.2]),
        intensity=np.array([10.0, 20.0]),
        header_lines=("# GIMaP Export", "# Data Type: Fitting Data"),
        x_column_name="q (nm^-1)",
        y_column_name="Intensity (a.u.)",
    )

    text_path = tmp_path / "fit.txt"
    csv_path = tmp_path / "fit.csv"
    text_result = use_case.execute(ExportFitResultRequest(path=text_path, **common))
    csv_result = use_case.execute(ExportFitResultRequest(path=csv_path, **common))

    assert text_result.succeeded and csv_result.succeeded
    assert text_path.read_text(encoding="utf-8") == (
        "# GIMaP Export\n"
        "# Data Type: Fitting Data\n"
        "q (nm^-1)\tIntensity (a.u.)\n"
        "1.000000e-01\t1.000000e+01\n"
        "2.000000e-01\t2.000000e+01\n"
    )
    assert csv_path.read_text(encoding="utf-8").splitlines()[-1] == (
        "2.000000e-01,2.000000e+01"
    )


def test_export_fit_result_returns_structured_file_error(tmp_path):
    missing_parent = tmp_path / "missing" / "fit.txt"
    outcome = ExportFitResult(LocalFitResultRepository()).execute(
        ExportFitResultRequest(
            path=missing_parent,
            q=np.array([1.0]),
            intensity=np.array([2.0]),
        )
    )

    assert not outcome.succeeded
    assert outcome.error.code in {"not_found", "write_failed"}
    assert Path(outcome.error.path) == missing_parent


def test_remote_cache_use_case_preserves_copy_reuse_and_clear_contract(tmp_path):
    project_root = tmp_path / "project"
    source_dir = tmp_path / "OneDrive" / "beamtime"
    source_dir.mkdir(parents=True)
    source = source_dir / "frame.cbf"
    source.write_bytes(b"detector-data")
    cache = ManageRemoteFileCache(LocalRemoteFileCacheAdapter(project_root))
    progress = []

    copied = cache.prepare(
        str(source),
        cache.default_directory(),
        3.0,
        on_progress=lambda *values: progress.append(values),
    )
    reused = cache.prepare(
        str(source),
        cache.default_directory(),
        3.0,
        on_progress=lambda *values: progress.append(values),
    )
    unrelated = copied.parent / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")

    assert cache.is_remote(str(source))
    assert copied == reused
    assert copied.read_bytes() == source.read_bytes()
    assert progress[-1][0] == 100
    assert cache.clear(cache.default_directory()) == 1
    assert unrelated.is_file()


def test_parameter_file_use_case_preserves_json_and_copy_contract(tmp_path):
    files = ManageFittingParameterFiles(LocalFittingParameterFileRepository())
    snapshot = tmp_path / "nested" / "fitting.json"
    values = {"schema_version": 1, "fitting": {"points_num": 50}}

    files.save_snapshot(snapshot, values)
    assert files.load_snapshot(snapshot) == values
    assert snapshot.read_text(encoding="utf-8").startswith("{\n    \"schema_version\"")

    exported = tmp_path / "exported.json"
    files.export_model_parameters(snapshot, exported)
    assert exported.read_bytes() == snapshot.read_bytes()


def test_fitting_log_use_case_preserves_plain_text(tmp_path):
    target = tmp_path / "logs" / "fitting.log"

    saved = SaveFittingLog(LocalFittingLogRepository()).execute(
        target,
        "first line\nsecond line",
    )

    assert saved == target
    assert target.read_text(encoding="utf-8") == "first line\nsecond line"


def test_optional_dependency_query_does_not_import_runtime():
    availability = CheckFittingDependency(
        ImportlibFittingDependencyAvailabilityAdapter()
    )

    assert availability.execute("numpy") is True
    assert availability.execute("definitely_missing_gimap_runtime") is False
