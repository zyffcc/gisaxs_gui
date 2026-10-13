"""Phase-1 gate: the shared q mapping reproduces every old q map pixel by pixel.

The references in ``geometry_legacy_reference.py`` are frozen copies of the
implementations the features used before ``src.gimap.shared.geometry``.  The
only permitted difference is floating-point reassociation, bounded here by
1e-12 of the largest |q| in the map (about 10⁴ times below one ulp of a q value
printed with 8 significant digits).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.gimap.features.trainset.domain.geometry import q_vectors
from tests.geometry_legacy_reference import trainset_q_vectors as frozen_trainset

MAP_TOLERANCE = 1e-12

# (rows, columns): Pilatus 2M (TestSAXSdata CBF), one Lambda module (TestSAXSdata NXS),
# and small odd shapes that exercise edges and centre handling.
SHAPES = [(1679, 1475), (516, 1556), (17, 23), (2, 3), (64, 1)]


def _random_cases(count: int, seed: int):
    rng = np.random.default_rng(seed)
    for index in range(count):
        rows, columns = SHAPES[index % len(SHAPES)]
        if rows * columns > 5000:
            rows, columns = int(rng.integers(8, 120)), int(rng.integers(8, 120))
        center_x = float(rng.uniform(-20, columns + 20))
        center_y = float(rng.uniform(-20, rows + 20))
        if index % 3 == 0:  # users often type integer centres
            center_x, center_y = float(round(center_x)), float(round(center_y))
        yield {
            "shape": (rows, columns),
            "pixel_x_um": float(rng.uniform(55.0, 200.0)),
            "pixel_y_um": float(rng.uniform(55.0, 200.0)),
            "center_x": center_x,
            "center_y": center_y,
            "distance_mm": float(rng.uniform(80.0, 8000.0)),
            "incidence_deg": float(rng.uniform(0.0, 1.5)) if index % 5 else 0.0,
            "wavelength_nm": float(rng.uniform(0.06, 0.16)),
        }


def _assert_same_map(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
    worst = float(np.max(np.abs(actual - expected)))
    assert worst <= MAP_TOLERANCE * scale, f"max |Δq| = {worst:.3e} (scale {scale:.3e})"
    # Pixels exactly on an axis keep their sign convention (values within the
    # tolerance of zero may carry either sign).
    settled = np.abs(expected) > MAP_TOLERANCE * scale
    np.testing.assert_array_equal(np.sign(actual[settled]), np.sign(expected[settled]))
    np.testing.assert_array_equal(actual == 0.0, expected == 0.0)


@pytest.mark.parametrize("case", list(_random_cases(60, seed=7)))
def test_trainset_q_vectors_match_frozen_implementation(case) -> None:
    rows, columns = case["shape"]
    config = {
        "detector": {
            "pixels_x": columns,
            "pixels_y": rows,
            "pixel_size_x_mm": case["pixel_x_um"] / 1000.0,
            "pixel_size_y_mm": case["pixel_y_um"] / 1000.0,
            "distance_mm": case["distance_mm"],
            "beam_center_x_px": case["center_x"],
            "beam_center_y_px": case["center_y"],
        },
        "beam": {
            "grazing_angle_deg": case["incidence_deg"],
            "wavelength_nm": case["wavelength_nm"],
        },
    }
    actual, expected = q_vectors(config), frozen_trainset(config)
    assert set(actual) == set(expected)
    for key in expected:
        _assert_same_map(actual[key], expected[key])
