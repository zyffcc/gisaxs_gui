from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.waxs.application import (
    ProcessWaxsBatch,
    PreviewWaxsBatchFrame,
    RunWaxsBatch,
    WaxsBatchRequest,
    WaxsBatchResult,
    WaxsBatchSource,
    WaxsCurve,
    WaxsPreprocessedFrame,
    WaxsBatchPreviewRequest,
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


def test_batch_normalization_supports_group_first_and_per_frame_modes(tmp_path):
    class Catalog:
        def discover(self, folder, pattern):
            return (folder / "multi.nxs",)

    class Images:
        def frame_count(self, path):
            return 2

        def load_frame(self, path, frame_index):
            return np.full((3, 3), frame_index + 1.0, dtype=np.float32)

    class Preprocess:
        def execute(self, request):
            image = request.image
            peak = 2.0 * float(np.mean(image))
            factor = request.normalization_factor or (
                request.normalization_target_intensity / peak
            )
            curve = WaxsCurve(
                np.array([1.9, 2.0, 2.1]),
                np.array([1.0, peak, 1.0]) * factor,
            )
            return WaxsPreprocessedFrame(
                image * factor, curve, dict(request.geometry), factor
            )

    def run(mode):
        request = _request(tmp_path)
        request = WaxsBatchRequest(
            **{
                **request.__dict__,
                "export_background_subtracted": False,
                "normalization_enabled": True,
                "normalization_target_q": 2.0,
                "normalization_half_width": 0.05,
                "normalization_mode": mode,
            }
        )
        exporter = _Exporter()
        ProcessWaxsBatch(
            Images(), Catalog(), exporter, preprocess_frame=Preprocess()
        ).execute(request)
        return exporter

    group = run("source_first")
    per_frame = run("per_frame")

    np.testing.assert_allclose([entry[2][1] for entry in group.curves], [1.0, 2.0])
    np.testing.assert_allclose(
        [entry[2][1] for entry in per_frame.curves], [1.0, 1.0]
    )
    np.testing.assert_allclose(
        [np.mean(entry[1]) for entry in per_frame.images], [0.5, 0.5]
    )


def test_batch_calibration_updates_sdd_before_q_export(tmp_path):
    class Catalog:
        def discover(self, folder, pattern):
            return (folder / "scan.tif",)

    class Images:
        def frame_count(self, path):
            return 1

        def load_frame(self, path, frame_index):
            return np.ones((3, 3), dtype=np.float32)

    class Preprocess:
        def execute(self, request):
            geometry = {**request.geometry, "distance": 1000.0}
            return WaxsPreprocessedFrame(
                request.image,
                WaxsCurve(np.array([1.99, 2.0, 2.01]), np.array([1.0, 5.0, 1.0])),
                geometry,
                None,
            )

    request = _request(tmp_path)
    request = WaxsBatchRequest(
        **{
            **request.__dict__,
            "geometry": {**request.geometry, "distance": 900.0},
            "export_q_images": True,
            "calibration_enabled": True,
            "calibration_target_q": 2.0,
            "calibration_half_width": 0.3,
        }
    )
    exporter = _Exporter()

    result = ProcessWaxsBatch(
        Images(), Catalog(), exporter, preprocess_frame=Preprocess()
    ).execute(request)

    assert result.failed_count == 0
    q_exports = [entry for entry in exporter.images if entry[2]["coordinate_mode"] == "q"]
    assert q_exports[0][2]["geometry"]["distance"] == pytest.approx(1000.0)


def test_batch_preview_uses_selected_group_item_and_first_frame_factor(tmp_path):
    source = WaxsBatchSource(tmp_path, "*.nxs", "group")

    class Catalog:
        def discover(self, folder, pattern):
            return (folder / "scan.nxs",)

    class Images:
        def frame_count(self, path):
            return 3

        def load_frame(self, path, frame_index):
            return np.full((2, 2), frame_index + 1.0)

    class Preprocess:
        def __init__(self):
            self.factors = []

        def execute(self, request):
            self.factors.append(request.normalization_factor)
            factor = request.normalization_factor or 0.5
            curve = WaxsCurve(np.array([2.0]), np.array([1.0]))
            return WaxsPreprocessedFrame(
                request.image * factor,
                curve,
                dict(request.geometry),
                factor,
            )

    preprocessing = Preprocess()
    use_case = PreviewWaxsBatchFrame(
        Images(), Catalog(), preprocess_frame=preprocessing
    )
    batch = _request(tmp_path)
    batch = WaxsBatchRequest(
        **{
            **batch.__dict__,
            "normalization_enabled": True,
            "normalization_mode": "source_first",
        }
    )

    result = use_case.execute(WaxsBatchPreviewRequest(source, 1, batch))

    assert result.item_index == 1
    assert result.item_count == 3
    assert result.frame_index == 1
    assert preprocessing.factors == [None, 0.5]
    np.testing.assert_allclose(result.frame.image, 1.0)
