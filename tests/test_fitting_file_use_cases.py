from pathlib import Path

import numpy as np

from src.gimap.features.fitting.application import (
    ExportCurveFigure,
    ExportCurveFigureRequest,
    ExportFitResult,
    ExportFitResultRequest,
    FigureSeries,
    DiscoverInSituFrames,
    DiscoverInSituFramesRequest,
    InSituSourceFrame,
    LoadCurve,
    LoadCurveRequest,
    ManageFittingParameterFiles,
    SaveFittingLog,
    CheckFittingDependency,
)
from src.gimap.features.fitting.infrastructure.adapters import (
    MatplotlibCurveFigureWriter,
    LocalCurveRepository,
    LocalFitResultRepository,
    LocalInSituFrameRepository,
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


def test_curve_figure_is_a_column_wide_600_dpi_raster_or_a_vector(tmp_path):
    from PIL import Image

    q = np.linspace(0.01, 0.3, 40)
    request = dict(
        series=(
            FigureSeries("Data", q, 100 * np.exp(-q / 0.05), "#1f4e9c", "scatter"),
            FigureSeries("Model", q, 100 * np.exp(-q / 0.05), "#c92a2a", "line"),
        ),
        x_label="q (nm⁻¹)",
        y_label="Intensity (a.u.)",
        log_y=True,
    )
    use_case = ExportCurveFigure(MatplotlibCurveFigureWriter())

    raster = use_case.execute(ExportCurveFigureRequest(path=tmp_path / "plot.png", **request))
    vector = use_case.execute(ExportCurveFigureRequest(path=tmp_path / "plot.pdf", **request))
    empty = use_case.execute(
        ExportCurveFigureRequest(path=tmp_path / "empty.png", **{**request, "series": ()})
    )

    assert raster.succeeded and vector.succeeded
    with Image.open(raster.value) as image:
        assert abs(image.width - round(8.5 / 2.54 * 600)) <= 2  # one column at 600 dpi
    assert vector.value.read_bytes().startswith(b"%PDF")
    assert not empty.succeeded and "Nothing is plotted" in empty.error.message


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


def test_discover_insitu_curves_in_natural_order_and_restore_tokens(tmp_path):
    nested = tmp_path / "run_2"
    nested.mkdir()
    curve = "0.1 1\n0.2 2\n"
    for name in ("s_00010_fit_input.dat", "s_00002_fit_input.dat", "s_00001_horizontal.csv"):
        (tmp_path / name).write_text(curve, encoding="ascii")
    (nested / "s_00003_fit_input.dat").write_text(curve, encoding="ascii")
    use_case = DiscoverInSituFrames(LocalInSituFrameRepository())

    direct = use_case.execute(DiscoverInSituFramesRequest(tmp_path))
    recursive = use_case.execute(DiscoverInSituFramesRequest(tmp_path, recursive=True))

    assert [frame.path.name for frame in direct] == ["s_00002_fit_input.dat", "s_00010_fit_input.dat"]
    # Natural order of the path below the folder: "run_2/…" sorts before "s_…".
    assert [frame.path.name for frame in recursive] == [
        "s_00003_fit_input.dat",
        "s_00002_fit_input.dat",
        "s_00010_fit_input.dat",
    ]
    restored = InSituSourceFrame.from_token(direct[-1].token)
    assert restored.path == direct[-1].path and restored.display_name == "s_00010_fit_input.dat"
