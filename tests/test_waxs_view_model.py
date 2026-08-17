from pathlib import Path

import numpy as np

from src.gimap.features.waxs.application import (
    IntegrateWaxsImageRequest,
    LoadedWaxsImage,
    WaxsBatchItem,
    WaxsBatchProgress,
    WaxsBatchResult,
    WaxsCurve,
)
from src.gimap.features.waxs.presentation import WaxsViewModel


class _UseCase:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.requests = []

    def execute(self, request, **kwargs):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.value


class _BatchUseCase(_UseCase):
    def __init__(self, value):
        super().__init__(value)
        self.cancelled = False
        self.paused = None

    def execute(self, request, *, on_progress=None):
        self.requests.append(request)
        if on_progress:
            on_progress(WaxsBatchProgress(1, 2, "scan", "succeeded"))
        return self.value

    def cancel(self):
        self.cancelled = True
        return True

    def set_paused(self, paused):
        self.paused = paused
        return True


def _view_model(*, loader=None, integrator=None, batch=None, curve=None, image=None):
    return WaxsViewModel(
        load_image=loader or _UseCase(),
        integrate_image=integrator or _UseCase(),
        run_batch=batch or _BatchUseCase(WaxsBatchResult(())),
        export_curve=curve or _UseCase(),
        export_image=image or _UseCase(),
        compute_q_maps=_UseCase((np.zeros((2, 2)), np.ones((2, 2)))),
        cut_image=_UseCase(),
        prepare_display=_UseCase(),
        estimate_display_limits=_UseCase((0.0, 1.0)),
    )


def test_view_model_loads_image_and_records_structured_error(tmp_path):
    loaded = LoadedWaxsImage(
        tmp_path / "scan.tif",
        0,
        1,
        np.ones((3, 3), dtype=np.float32),
    )
    view_model = _view_model(loader=_UseCase(loaded))

    assert view_model.load_image(loaded.path) is loaded
    assert view_model.state.image_status == "ready"
    assert view_model.state.current_image is loaded

    failing = _view_model(loader=_UseCase(error=OSError("damaged TIFF")))
    assert failing.load_image(tmp_path / "bad.tif") is None
    assert failing.state.image_status == "error"
    assert failing.state.error_message == "damaged TIFF"


def test_view_model_integrates_and_exports_without_qapplication(tmp_path):
    curve = WaxsCurve(np.array([0.0, 1.0]), np.array([2.0, 3.0]))
    integrator = _UseCase(curve)
    curve_export = _UseCase(tmp_path / "curve.csv")
    image_export = _UseCase(tmp_path / "image.png")
    view_model = _view_model(
        integrator=integrator,
        curve=curve_export,
        image=image_export,
    )
    request = IntegrateWaxsImageRequest(
        np.ones((2, 2)), {}, {}, -1e12, 1e12
    )

    assert view_model.integrate(request) is curve
    assert view_model.state.current_curve is curve
    assert view_model.export_curve(tmp_path / "curve.csv") == tmp_path / "curve.csv"
    assert view_model.export_image(
        tmp_path / "image.png", np.ones((2, 2)), {"log_scale": False}
    ) == tmp_path / "image.png"
    np.testing.assert_array_equal(curve_export.requests[0].intensity, curve.intensity)


def test_view_model_batch_progress_cancel_pause_and_result(tmp_path):
    result = WaxsBatchResult(
        (WaxsBatchItem(tmp_path / "scan.tif", 0, "scan", "succeeded"),)
    )
    batch = _BatchUseCase(result)
    view_model = _view_model(batch=batch)
    progress = []

    assert view_model.run_batch(object(), on_progress=progress.append) is result
    assert view_model.state.batch_status == "ready"
    assert view_model.state.progress == 1.0
    assert progress[0].name == "scan"
    assert view_model.cancel_batch() is True
    assert batch.cancelled is True
    assert view_model.set_batch_paused(True) is True
    assert batch.paused is True


def test_view_model_batch_error_does_not_escape_to_presentation():
    batch = _BatchUseCase(WaxsBatchResult(()))
    batch.error = RuntimeError("worker crashed")

    def fail(request, *, on_progress=None):
        raise batch.error

    batch.execute = fail
    view_model = _view_model(batch=batch)

    assert view_model.run_batch(object()) is None
    assert view_model.state.batch_status == "error"
    assert view_model.state.error_message == "worker crashed"
