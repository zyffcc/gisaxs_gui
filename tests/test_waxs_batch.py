from pathlib import Path

import numpy as np

from src.gimap.features.waxs.application import (
    ProcessWaxsBatch,
    RunWaxsBatch,
    WaxsBatchRequest,
    WaxsBatchResult,
    WaxsBatchSource,
)


class _Images:
    def frame_count(self, path):
        return 2 if path.name == "multi.nxs" else 1

    def load_frame(self, path, frame_index):
        if path.name == "bad.tif":
            raise OSError("damaged TIFF")
        return np.full((5, 5), frame_index + 1.0, dtype=np.float32)


class _Catalog:
    def __init__(self, root):
        self.root = root

    def discover(self, folder, pattern):
        assert folder == self.root
        assert pattern == "*"
        return (folder / "multi.nxs", folder / "bad.tif")


class _Exporter:
    def __init__(self):
        self.images = []
        self.curves = []
        self.matrices = []
        self.curve_images = []

    def export_image(self, path, image, display):
        self.images.append((path, image.copy(), display))

    def export_curve(self, path, x, y):
        self.curves.append((path, x.copy(), y.copy()))

    def export_curve_image(self, path, x, y, display):
        self.curve_images.append((path, x.copy(), y.copy(), display))

    def export_matrix(self, path, columns, headers):
        self.matrices.append((path, columns, headers))


def _request(tmp_path):
    return WaxsBatchRequest(
        folder=tmp_path,
        pattern="*",
        output_folder=tmp_path / "out",
        export_images=True,
        export_curves=True,
        export_background_subtracted=True,
        display={"log_scale": False},
        geometry={
            "incidence": 0.2,
            "center_x": 2.0,
            "center_y": 2.0,
            "distance": 1000.0,
            "pixel_x": 100.0,
            "pixel_y": 100.0,
            "wavelength": 1.0,
            "qr_min": -121.0,
            "qr_max": -121.0,
            "qz_min": -121.0,
            "qz_max": -121.0,
        },
        integration={"mode": "radial", "bins": 5, "x_axis": "pixel"},
        mask_min=-1e12,
        mask_max=1e12,
    )


def test_batch_expands_frames_keeps_names_and_continues_file_error(tmp_path):
    exporter = _Exporter()
    progress = []
    result = ProcessWaxsBatch(
        _Images(), _Catalog(tmp_path), exporter
    ).execute(_request(tmp_path), on_progress=progress.append)

    assert [item.name for item in result.items] == [
        "multi_f0001",
        "multi_f0002",
        "bad",
    ]
    assert [item.status for item in result.items] == [
        "succeeded",
        "succeeded",
        "failed",
    ]
    assert result.failed_count == 1
    assert result.items[-1].error_message == "damaged TIFF"
    assert [item.completed for item in progress] == [1, 2, 3]
    assert len(exporter.images) == 2
    assert len(exporter.curves) == 4
    np.testing.assert_allclose(exporter.curves[1][2], 0.0)
    np.testing.assert_allclose(exporter.curves[3][2], 1.0)
    assert [entry[0].name for entry in exporter.matrices] == [
        "output.csv",
        "output_subbg.csv",
    ]


def test_batch_cancellation_stops_between_frames(tmp_path):
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls > 1

    result = ProcessWaxsBatch(_Images(), _Catalog(tmp_path), _Exporter()).execute(
        _request(tmp_path), is_cancelled=cancelled
    )

    assert result.cancelled is True
    assert len(result.items) == 1


def test_run_batch_use_case_uses_runner_port(tmp_path):
    class Runner:
        def __init__(self):
            self.request = None

        def run(self, request, *, on_progress=None):
            self.request = request
            return WaxsBatchResult(())

        def cancel(self):
            return True

        def set_paused(self, paused):
            return paused

    runner = Runner()
    use_case = RunWaxsBatch(runner)

    result = use_case.execute(_request(tmp_path))

    assert result.items == ()
    assert runner.request.folder == tmp_path
    assert use_case.cancel() is True
    assert use_case.set_paused(True) is True


def test_batch_processes_multiple_sources_into_named_subfolders(tmp_path):
    first = tmp_path / "run_a"
    second = tmp_path / "run_b"

    class Catalog:
        def discover(self, folder, pattern):
            assert pattern == "*.nxs"
            return (folder / "scan.nxs",)

    class Images:
        def frame_count(self, path):
            return 1

        def load_frame(self, path, frame_index):
            return np.ones((5, 5), dtype=np.float32)

    request = _request(tmp_path)
    request = WaxsBatchRequest(
        **{
            **request.__dict__,
            "sources": (
                WaxsBatchSource(first, "*.nxs", "sample_a"),
                WaxsBatchSource(second, "*.nxs", "sample_b"),
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
    exporter = _Exporter()

    result = ProcessWaxsBatch(Images(), Catalog(), exporter).execute(request)

    assert len(result.items) == 2
    image_paths = {str(entry[0].relative_to(tmp_path / "out")) for entry in exporter.images}
    assert image_paths == {
        str(Path("sample_a/2D_pixel/scan.png")),
        str(Path("sample_a/2D_q/scan.png")),
        str(Path("sample_b/2D_pixel/scan.png")),
        str(Path("sample_b/2D_q/scan.png")),
    }
    assert len(exporter.curve_images) == 2
    assert exporter.images[1][2]["coordinate_mode"] == "q"
    assert exporter.images[1][2]["q_range"] == request.q_range
