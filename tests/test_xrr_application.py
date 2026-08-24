"""Use-case tests for memory-bounded XRR series extraction."""

from pathlib import Path

import numpy as np

from src.gimap.features.xrr.application import (
    ExportXrrCurve,
    ExportXrrCurveRequest,
    ExtractXrrSeries,
    InspectXrrSeries,
    RunXrrExtraction,
    SpecularGeometry,
    XrrDetectorFrame,
    XrrExtractionRequest,
    XrrExtractionResult,
    XrrExtractionSettings,
    XrrFrameRef,
    XrrSeriesSpec,
)


class FakeSeriesRepository:
    def __init__(self):
        self.refs = tuple(
            XrrFrameRef(Path(f"frame_{index}.cbf"), 0, index) for index in range(3)
        )
        self.loaded = []

    def discover(self, spec):
        return self.refs

    def load_frame(self, ref):
        self.loaded.append(ref.sequence_index)
        image = np.full((9, 9), ref.sequence_index + 1, dtype=np.float32)
        return XrrDetectorFrame(image, None, {"energy_kev": 12.0})


def _request():
    return XrrExtractionRequest(
        series=XrrSeriesSpec(Path("series"), source_kind="cbf", theta_step_deg=0.1),
        geometry=SpecularGeometry(
            distance_m=0.001,
            energy_kev=12.0,
            pixel_size_x_m=1.0,
            pixel_size_y_m=1.0,
            beam_center_x_px=4.0,
            beam_center_y_px=4.0,
        ),
        extraction=XrrExtractionSettings(radius_px=0, aggregation="sum"),
    )


def test_inspection_loads_only_first_frame():
    repository = FakeSeriesRepository()
    result = InspectXrrSeries(repository).execute(_request().series)
    assert result.frame_count == 3
    assert repository.loaded == [0]


def test_extraction_loads_in_sequence_and_emits_one_transient_frame_at_a_time():
    repository = FakeSeriesRepository()
    progress = []
    result = ExtractXrrSeries(repository).execute(_request(), on_progress=progress.append)

    assert repository.loaded == [0, 1, 2]
    assert [point.theta_deg for point in result.points] == [0.0, 0.1, 0.2]
    assert [point.intensity for point in result.points] == [1.0, 2.0, 3.0]
    assert [item.completed for item in progress] == [1, 2, 3]
    assert all(item.preview.shape == (9, 9) for item in progress)


def test_nxs_angle_mode_requires_angles_from_repository():
    repository = FakeSeriesRepository()
    request = _request()
    request = XrrExtractionRequest(
        series=XrrSeriesSpec(
            Path("series.nxs"),
            source_kind="nxs",
            angle_mode="nxs_dataset",
            angle_dataset_path="/entry/sample/omega",
        ),
        geometry=request.geometry,
        extraction=request.extraction,
    )
    with pytest.raises(ValueError, match="did not provide"):
        ExtractXrrSeries(repository).execute(request)


def test_run_and_export_use_cases_delegate_to_application_ports(tmp_path):
    expected = XrrExtractionResult(())

    class Runner:
        def run(self, request, *, on_progress=None):
            self.request = request
            self.on_progress = on_progress
            return expected

        def cancel(self):
            return True

    class Exporter:
        def export(self, path, result):
            self.call = (path, result)

    runner = Runner()
    workflow = RunXrrExtraction(runner)
    assert workflow.execute(_request()) is expected
    assert runner.request == _request()
    assert workflow.cancel() is True

    exporter = Exporter()
    export = ExportXrrCurve(exporter)
    point_result = ExtractXrrSeries(FakeSeriesRepository()).execute(_request())
    destination = tmp_path / "curve.csv"
    export.execute(ExportXrrCurveRequest(destination, point_result))
    assert exporter.call == (destination, point_result)


import pytest
