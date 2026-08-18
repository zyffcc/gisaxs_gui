from pathlib import Path

import numpy as np

from src.gimap.features.waxs.application import (
    ComputeWaxsQMaps,
    CutWaxsImage,
    EstimateWaxsDisplayLimits,
    ExportWaxsCurve,
    ExportWaxsCurveRequest,
    ExportWaxsImage,
    ExportWaxsImageRequest,
    GetWaxsWorkingDirectory,
    NormalizeWaxsPath,
    PrepareWaxsDisplay,
    ValidateWaxsDirectory,
    WaxsCutImageRequest,
    WaxsDisplayLimitsRequest,
    WaxsDisplayRequest,
    WaxsQMapRequest,
)


GEOMETRY = {
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
}


class _Exporter:
    def __init__(self):
        self.curve = None
        self.image = None

    def export_curve(self, path, x, y):
        self.curve = (path, x, y)

    def export_image(self, path, image, display):
        self.image = (path, image, display)

    def export_matrix(self, path, columns, headers):
        raise AssertionError("matrix export is not part of these use cases")


class _Paths:
    def __init__(self, root: Path):
        self.root = root
        self.normalized = None
        self.checked = None

    def normalize(self, path):
        self.normalized = path
        return str(self.root / Path(path).name)

    def current_directory(self):
        return str(self.root)

    def is_directory(self, path):
        self.checked = path
        return Path(path) == self.root


def test_geometry_application_use_cases_preserve_domain_results():
    image = np.arange(16, dtype=float).reshape(4, 4) + 1.0

    qr, qz = ComputeWaxsQMaps().execute(WaxsQMapRequest(image.shape, GEOMETRY))
    cut = CutWaxsImage().execute(WaxsCutImageRequest(image, GEOMETRY))

    assert qr.shape == image.shape
    assert qz.shape == image.shape
    np.testing.assert_array_equal(cut.image, image)
    assert cut.extent is not None


def test_display_application_use_cases_apply_mask_log_and_limits():
    image = np.array([[0.0, 1.0], [10.0, 100.0]])

    display = PrepareWaxsDisplay().execute(
        WaxsDisplayRequest(image, True, 1.0, 100.0)
    )
    limits = EstimateWaxsDisplayLimits().execute(
        WaxsDisplayLimitsRequest(image, True, 1.0, 100.0, stride_hint=1)
    )

    assert np.isnan(display[0, 0])
    np.testing.assert_allclose(display[0, 1], 0.0)
    assert limits is not None
    assert limits[0] <= limits[1]


def test_export_application_use_cases_only_use_export_port(tmp_path):
    exporter = _Exporter()
    x = np.array([0.0, 1.0])
    y = np.array([2.0, 3.0])
    image = np.ones((2, 2))

    curve_path = ExportWaxsCurve(exporter).execute(
        ExportWaxsCurveRequest(tmp_path / "curve.csv", x, y)
    )
    image_path = ExportWaxsImage(exporter).execute(
        ExportWaxsImageRequest(
            tmp_path / "image.png", image, {"log_scale": False}
        )
    )

    assert curve_path == Path(tmp_path / "curve.csv")
    assert image_path == Path(tmp_path / "image.png")
    assert exporter.curve[0] == curve_path
    assert exporter.image[0] == image_path


def test_workspace_path_use_cases_only_use_path_port(tmp_path):
    paths = _Paths(tmp_path)

    assert NormalizeWaxsPath(paths)("scan.nxs") == str(tmp_path / "scan.nxs")
    assert GetWaxsWorkingDirectory(paths)() == str(tmp_path)
    assert ValidateWaxsDirectory(paths)(tmp_path)
    assert paths.normalized == "scan.nxs"
    assert paths.checked == tmp_path
