from pathlib import Path

import numpy as np
from PIL import Image

from src.gimap.features.fitting.application import (
    ExportFitResult,
    ExportFitResultRequest,
    LoadCurve,
    LoadCurveRequest,
    LoadScatteringFile,
    LoadScatteringFileRequest,
)
from src.gimap.features.fitting.infrastructure.adapters import (
    LocalCurveRepository,
    LocalFitResultRepository,
    LocalScatteringFileRepository,
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
