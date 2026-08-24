"""Detector-series discovery and JobRunner serialization tests."""

from pathlib import Path

import h5py
import numpy as np

from src.gimap.features.xrr.application import XrrSeriesSpec
from src.gimap.features.xrr.infrastructure.adapters import LocalXrrSeriesRepository
from src.gimap.features.xrr.infrastructure.serialization import decode_preview, encode_preview


def test_cbf_series_uses_natural_filename_order(tmp_path):
    for name in ("scan_10.cbf", "scan_2.cbf", "scan_1.cbf", "ignore.txt"):
        (tmp_path / name).touch()
    refs = LocalXrrSeriesRepository().discover(
        XrrSeriesSpec(tmp_path, source_kind="cbf", pattern="scan_*.cbf")
    )
    assert [ref.path.name for ref in refs] == ["scan_1.cbf", "scan_2.cbf", "scan_10.cbf"]


def test_nxs_module_group_becomes_one_frame_sequence_with_motor_angles(tmp_path):
    for module in (1, 2):
        path = tmp_path / f"scan_m{module}.nxs"
        with h5py.File(path, "w") as handle:
            handle.create_dataset(
                "/entry/instrument/detector/data",
                data=np.zeros((3, 32, 32), dtype=np.uint16),
            )
            angle = handle.create_dataset(
                "/entry/sample/transformations/omega",
                data=np.radians([0.1, 0.2, 0.3]),
            )
            angle.attrs["units"] = "rad"
    refs = LocalXrrSeriesRepository().discover(
        XrrSeriesSpec(
            tmp_path / "scan_m2.nxs",
            source_kind="nxs",
            angle_mode="nxs_dataset",
            angle_dataset_path="/entry/sample/transformations/omega",
        )
    )
    assert len(refs) == 3
    assert all(ref.path.name == "scan_m1.nxs" for ref in refs)
    assert [ref.theta_deg for ref in refs] == pytest.approx([0.1, 0.2, 0.3])


def test_preview_transport_is_downsampled_and_preserves_full_shape():
    image = np.arange(800 * 600, dtype=np.float32).reshape(800, 600)
    preview, full_shape = decode_preview(encode_preview(image, max_side=200))
    assert max(preview.shape) <= 200
    assert full_shape == image.shape
    assert preview.dtype == np.float32


import pytest
