"""Process-isolated XRR extraction smoke test."""

import h5py
import numpy as np

from src.gimap.features.xrr.application import (
    SpecularGeometry,
    XrrExtractionRequest,
    XrrExtractionSettings,
    XrrSeriesSpec,
)
from src.gimap.features.xrr.infrastructure import JobRunnerXrrExtractionAdapter
from src.gimap.integrations.jobs import LocalProcessJobRunner


def test_two_frame_nxs_xrr_runs_in_worker_process(tmp_path):
    path = tmp_path / "scan.nxs"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "/entry/instrument/detector/data",
            data=np.stack(
                (
                    np.full((32, 32), 3, dtype=np.uint16),
                    np.full((32, 32), 7, dtype=np.uint16),
                )
            ),
        )
    request = XrrExtractionRequest(
        series=XrrSeriesSpec(path, source_kind="nxs", theta_start_deg=0.0),
        geometry=SpecularGeometry(
            distance_m=1.0,
            energy_kev=12.0,
            pixel_size_x_m=100e-6,
            pixel_size_y_m=100e-6,
            beam_center_x_px=15.0,
            beam_center_y_px=15.0,
        ),
        extraction=XrrExtractionSettings(radius_px=0, aggregation="sum"),
    )
    runner = LocalProcessJobRunner()
    progress = []
    try:
        result = JobRunnerXrrExtractionAdapter(runner).run(
            request, on_progress=progress.append
        )
    finally:
        runner.shutdown()

    assert [point.intensity for point in result.points] == [3.0, 7.0]
    assert [item.completed for item in progress] == [1, 2]
    assert progress[-1].preview_shape == (32, 32)
