"""Analyze → Fitting hand-over keeps the V5 counting contract of the old detector path.

Cut & Fitting used to build "native CBF columns" itself: the mean counts per
pixel of each detector column in the cut band, a Poisson σ and the pixel
count, plus a working tolerance on σ.  Analyze now produces those points;
these tests pin the equivalence and the file format that carries them.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.gimap.features.analyze.domain import guard_invalid, native_profile, valid_pixels
from src.gimap.features.analyze.domain.models import Curve
from src.gimap.features.analyze.infrastructure.adapters import CsvCurveWriter
from src.gimap.features.fitting.application import LoadCurveRequest
from src.gimap.features.fitting.infrastructure.adapters.local_files import LocalCurveRepository
from src.gimap.features.fitting.presentation.bindings.workflow_v5_binding import (
    WorkflowV5BindingMixin,
)


def _reference_column_observations(image, region):
    """Frozen copy of Fitting's former ``column_observations`` (q omitted)."""
    r0, r1, x0, x1 = region
    data = np.asarray(image, float)[r0 : r1 + 1, x0 : x1 + 1]
    valid = np.isfinite(data)
    count = valid.sum(axis=0)
    total = np.where(valid, data, 0).sum(axis=0)
    means = total / np.maximum(count, 1)
    sigma = np.sqrt(np.maximum(total, 1)) / np.maximum(count, 1)
    keep = count > 0
    return means[keep], sigma[keep], count[keep]


def _reference_preprocessing(raw, margin):
    """Frozen copy of the former CBF invalid-pixel guard (negative codes → NaN)."""
    from scipy.ndimage import maximum_filter

    invalid = ~np.isfinite(raw) | (raw < 0)
    invalid = maximum_filter(invalid, size=2 * margin + 1, mode="constant", cval=0)
    image = np.array(raw, dtype=float, copy=True)
    image[invalid] = np.nan
    return image


def test_native_columns_equal_the_former_fitting_observations() -> None:
    rng = np.random.default_rng(7)
    raw = rng.poisson(4.0, size=(60, 300)).astype(np.float32)
    raw[:, 120:127] = -1  # module gap
    raw[30, 40] = -2  # dead pixel
    rows = (20, 26)
    means, sigma, count = _reference_column_observations(
        _reference_preprocessing(raw, 3), (rows[0], rows[1] - 1, 0, raw.shape[1] - 1)
    )
    valid = guard_invalid(valid_pixels(raw), 3)
    columns = np.tile(np.arange(raw.shape[1], dtype=float), (rows[1] - rows[0], 1))
    x, mean, sigma_new, pixels = native_profile(
        raw[rows[0] : rows[1]], valid[rows[0] : rows[1]], columns, axis=0
    )
    np.testing.assert_array_equal(mean, means)
    np.testing.assert_array_equal(sigma_new, sigma)
    np.testing.assert_array_equal(pixels, count)
    assert not np.any((x >= 117) & (x <= 129))  # the gap and its guard stay empty


def _fit_input(tmp_path: Path, observation) -> Path:
    curve = Curve(
        "fit_input",
        "Horizontal cut I(qy)",
        np.linspace(-0.05, 0.05, 20),
        np.full(20, 8.0),
        np.full(20, 0.9),
        np.full(20, 5, dtype=np.int64),
        "qy (Å⁻¹)",
    )
    metadata = {"source_file": "frame.cbf"}
    if observation is not None:
        metadata["observation"] = observation
    return CsvCurveWriter().write_xy(curve, tmp_path / "frame_fit_input.dat", metadata)


OBSERVATION = {
    "source": "native_detector_columns",
    "file_format": "cbf",
    "intensity_unit": "counts_per_pixel",
    "counting_model_valid": True,
    "threshold_enabled": False,
    "gap_guard_px": 3,
    "summed_frames": 1,
}


def test_fit_input_carries_pixels_and_observation_into_fitting(tmp_path: Path) -> None:
    path = _fit_input(tmp_path, OBSERVATION)
    header = path.read_text("ascii").splitlines()
    assert header[0].startswith("# GIMaP Analyze fit input")
    assert any(line.startswith("# observation:") for line in header)
    np.testing.assert_allclose(np.loadtxt(path)[:, 3], 5)

    curve = LocalCurveRepository().load(LoadCurveRequest(path=path, q_source_unit="angstrom"))
    np.testing.assert_array_equal(curve.pixels, np.full(20, 5.0))
    assert curve.observation == OBSERVATION
    np.testing.assert_allclose(curve.error, 0.9)

    # A plain three-column file has neither.
    plain = tmp_path / "plain.dat"
    plain.write_text("0.1 1 0.1\n0.2 2 0.1\n0.3 3 0.1\n", encoding="ascii")
    other = LocalCurveRepository().load(LoadCurveRequest(path=plain, q_source_unit="angstrom"))
    assert other.pixels is None and other.observation == {}


def test_v5_gets_the_native_cbf_contract_and_the_working_tolerance(tmp_path: Path) -> None:
    curve = LocalCurveRepository().load(
        LoadCurveRequest(path=_fit_input(tmp_path, OBSERVATION), q_source_unit="angstrom")
    )
    data = {
        "q": curve.q,
        "I": curve.intensity,
        "err": curve.error,
        "pixels": curve.pixels,
        "observation": curve.observation,
        "q_source_unit": "angstrom",
    }
    owner = SimpleNamespace()
    options = {"relative_noise": 0.1, "absolute_noise": 0.0}
    native = WorkflowV5BindingMixin._native_curve_input(owner, data, options)
    metadata = owner._workflow_observation_metadata
    assert metadata["source"] == "native_cbf_columns"
    assert metadata["valid_pixel_counts"] == [5.0] * 20
    assert metadata["intensity_unit"] == "counts_per_pixel"
    assert metadata["stack_count"] == 1 and not metadata["threshold_enabled"]
    np.testing.assert_allclose(native["err"], np.hypot(0.9, 0.8))

    # Summed or background-subtracted curves are flagged, so the learned
    # branch falls back to numerical fitting exactly as before.
    summed = {**data, "observation": {**OBSERVATION, "summed_frames": 4, "counting_model_valid": False}}
    WorkflowV5BindingMixin._native_curve_input(owner, summed, options)
    assert owner._workflow_observation_metadata["stack_count"] == 4
    assert owner._workflow_observation_metadata["counting_model_valid"] is False

    # GIWAXS radial bins and foreign files keep the plain 1D path.
    radial = {**data, "observation": {**OBSERVATION, "source": "radial_bins"}}
    assert WorkflowV5BindingMixin._native_curve_input(SimpleNamespace(), radial, options) is None


def test_v5_batch_reader_accepts_the_pixels_column(tmp_path: Path) -> None:
    from src.gimap.features.fitting.infrastructure.adapters.workflow_v5 import read_curve

    q, intensity, sigma = read_curve(_fit_input(tmp_path, OBSERVATION))
    assert q.size == 20 and sigma == pytest.approx(np.full(20, 0.9))
