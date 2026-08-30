from pathlib import Path

import numpy as np

from src.gimap.app.jobs import JobProgress, JobResult
from src.gimap.features.waxs.application import WaxsBatchRequest, WaxsBatchSource
from src.gimap.features.waxs.infrastructure import (
    JobRunnerWaxsBatchAdapter,
    LocalWaxsExportAdapter,
    LocalWaxsFileCatalog,
    LocalWaxsPathAdapter,
)
from src.gimap.features.waxs.infrastructure.batch_serialization import (
    request_from_payload,
    request_to_payload,
)


def _request(tmp_path):
    return WaxsBatchRequest(
        folder=tmp_path,
        pattern="*.tif",
        output_folder=tmp_path / "out",
        export_images=False,
        export_curves=True,
        export_background_subtracted=False,
        display={
            "log_scale": False,
            "colormap": "viridis",
            "auto_scale": True,
            "vmin": 0.0,
            "vmax": 10.0,
            "mask_min": -1e12,
            "mask_max": 1e12,
        },
        geometry={},
        integration={},
        mask_min=-1e12,
        mask_max=1e12,
    )


def test_local_catalog_filters_supported_extensions_and_sorts(tmp_path):
    for name in ("b.tif", "a.nxs", "skip.txt"):
        (tmp_path / name).write_bytes(b"x")

    paths = LocalWaxsFileCatalog().discover(tmp_path, "*")

    assert [path.name for path in paths] == ["a.nxs", "b.tif"]


def test_local_path_adapter_normalizes_and_inspects_directories(tmp_path):
    adapter = LocalWaxsPathAdapter()

    assert adapter.normalize(tmp_path) == str(tmp_path)
    assert adapter.is_directory(tmp_path)
    assert Path(adapter.current_directory()).is_dir()


def test_local_export_adapter_writes_curve_matrix_and_png(tmp_path):
    exporter = LocalWaxsExportAdapter()
    x = np.array([0.0, 1.0])
    y = np.array([2.0, 3.0])
    curve = tmp_path / "curve.csv"
    matrix = tmp_path / "matrix.csv"
    image = tmp_path / "image.png"
    q_image = tmp_path / "q_image.png"
    curve_image = tmp_path / "curve.png"

    exporter.export_curve(curve, x, y)
    exporter.export_matrix(matrix, (x, y), ("x", "scan"))
    exporter.export_image(
        image,
        np.arange(16, dtype=float).reshape(4, 4) + 1,
        {
            "log_scale": True,
            "colormap": "viridis",
            "auto_scale": True,
            "mask_min": 1.0,
            "mask_max": 16.0,
        },
    )
    exporter.export_image(
        q_image,
        np.arange(16, dtype=float).reshape(4, 4) + 1,
        {
            "log_scale": False,
            "colormap": "magma",
            "auto_scale": True,
            "mask_min": 1.0,
            "mask_max": 16.0,
            "coordinate_mode": "q",
            "geometry": {
                "incidence": 0.2,
                "center_x": 2.0,
                "center_y": 2.0,
                "distance": 1000.0,
                "pixel_x": 100.0,
                "pixel_y": 100.0,
                "wavelength": 1.0,
            },
            "q_range": {
                "qr_min": -0.01,
                "qr_max": 0.01,
                "qz_min": 0.0,
                "qz_max": 0.05,
            },
        },
    )
    exporter.export_curve_image(
        curve_image, x, y, {"x_label": "q (Å⁻¹)"}
    )

    assert curve.read_text(encoding="utf-8").splitlines()[0] == "x,intensity"
    assert matrix.read_text(encoding="utf-8").splitlines()[0] == "x,scan"
    assert image.is_file() and image.stat().st_size > 0
    assert q_image.is_file() and q_image.stat().st_size > 0
    assert curve_image.is_file() and curve_image.stat().st_size > 0


def test_job_runner_batch_adapter_uses_serializable_request_and_progress(tmp_path):
    class Runner:
        def __init__(self):
            self.request = None

        def run(self, request, on_progress=None):
            self.request = request
            on_progress(
                JobProgress(
                    request.job_id,
                    1,
                    1,
                    "Processed scan",
                    {"name": "scan", "status": "succeeded"},
                )
            )
            return JobResult(
                request.job_id,
                "succeeded",
                {
                    "items": [
                        {
                            "path": str(tmp_path / "scan.tif"),
                            "frame_index": 0,
                            "name": "scan",
                            "status": "succeeded",
                            "error_message": None,
                        }
                    ],
                    "cancelled": False,
                },
            )

        def cancel(self, job_id):
            return True

    runner = Runner()
    progress = []
    result = JobRunnerWaxsBatchAdapter(runner).run(
        _request(tmp_path), on_progress=progress.append
    )

    assert runner.request.handler.endswith(":process_waxs_batch_job")
    assert runner.request.payload["folder"] == str(tmp_path)
    assert result.items[0].name == "scan"
    assert progress[0].status == "succeeded"


def test_job_runner_batch_adapter_remembers_cancel_before_worker_start(tmp_path):
    class Runner:
        def run(self, request, on_progress=None):
            raise AssertionError("cancelled job must not start")

        def cancel(self, job_id):
            return True

    adapter = JobRunnerWaxsBatchAdapter(Runner())

    assert adapter.cancel() is True
    result = adapter.run(_request(tmp_path))

    assert result.cancelled is True


def test_job_runner_batch_adapter_remembers_pause_before_worker_start(tmp_path):
    class Runner:
        control_value = None

        def run(self, request, on_progress=None):
            self.control_value = Path(request.payload["_control_file"]).read_text(
                encoding="utf-8"
            )
            return JobResult(
                request.job_id,
                "succeeded",
                {"items": [], "cancelled": False},
            )

        def cancel(self, job_id):
            return True

    runner = Runner()
    adapter = JobRunnerWaxsBatchAdapter(runner)

    assert adapter.set_paused(True) is True
    result = adapter.run(_request(tmp_path))

    assert result.cancelled is False
    assert runner.control_value == "paused"


def test_batch_serialization_preserves_multiple_sources_and_export_options(tmp_path):
    request = _request(tmp_path)
    request = WaxsBatchRequest(
        **{
            **request.__dict__,
            "sources": (
                WaxsBatchSource(tmp_path / "a", "*.nxs", "sample_a"),
                WaxsBatchSource(tmp_path / "b", "*.tif", "sample_b"),
            ),
            "export_q_images": True,
            "export_curve_images": True,
            "q_range": {
                "qr_min": -1.0,
                "qr_max": 1.0,
                "qz_min": 0.0,
                "qz_max": 2.0,
            },
        }
    )

    restored = request_from_payload(request_to_payload(request))

    assert restored.sources == request.sources
    assert restored.export_q_images is True
    assert restored.export_curve_images is True
    assert restored.q_range == request.q_range
