import numpy as np
from PIL import Image

from src.gimap.features.waxs.application import RunWaxsBatch, WaxsBatchRequest
from src.gimap.features.waxs.infrastructure import JobRunnerWaxsBatchAdapter
from src.gimap.integrations.jobs import LocalProcessJobRunner


def test_two_file_waxs_batch_runs_in_worker_process(tmp_path):
    Image.fromarray(np.arange(25, dtype=np.uint16).reshape(5, 5) + 1).save(
        tmp_path / "one.tif"
    )
    Image.fromarray(np.arange(25, dtype=np.uint16).reshape(5, 5) + 2).save(
        tmp_path / "two.tif"
    )
    request = WaxsBatchRequest(
        folder=tmp_path,
        pattern="*.tif",
        output_folder=tmp_path / "out",
        export_images=False,
        export_curves=True,
        export_background_subtracted=True,
        display={},
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
        timeout_seconds=60,
    )
    runner = LocalProcessJobRunner()
    try:
        result = RunWaxsBatch(JobRunnerWaxsBatchAdapter(runner)).execute(request)
    finally:
        runner.shutdown()

    assert [item.status for item in result.items] == ["succeeded", "succeeded"]
    assert (tmp_path / "out/1D/one.csv").is_file()
    assert (tmp_path / "out/1D/two_subbg.csv").is_file()
    assert (tmp_path / "out/1D/output.csv").is_file()
