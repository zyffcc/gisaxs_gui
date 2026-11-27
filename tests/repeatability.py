"""Whether a computation repeated on this computer gives the same bits.

On 64-bit x86 it does, in one process and on any number of threads. With NumPy's arm64 build (Apple silicon, macOS
CI) the size averages of spheres over two or four paths, whose amplitudes are complex, can differ by a few units in
the last place between two calls with the same values: up to 3e-14 relative, another set of rows on every run. There
"the same" allows ``LAST_DIGITS``. Predict on a cut with several solutions of nearly the same probability (GALAXI)
turns such differences into another set of solutions, so a replay of its export can give another fit there."""

from __future__ import annotations

import platform

import numpy as np

SAME_BITS = platform.machine().lower() not in ("arm64", "aarch64")
LAST_DIGITS = 1e-12
"""Relative difference allowed where the bits can change, 30 times the largest seen."""


def assert_same(actual, expected) -> None:
    """The same bits; on arm64 the same to ``LAST_DIGITS``."""
    if SAME_BITS:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=LAST_DIGITS, atol=0, equal_nan=True)


def assert_replayed(report) -> None:
    """A table replay (``reproduce_record``) gives the recorded columns: bit for bit; on arm64 every column of every
    table to ``LAST_DIGITS``. A fit whose residuals are at the rounding level differs there by 100 % in χ² and the
    errors (macOS CI), so the data of such a test need residuals above it."""
    if SAME_BITS or report["status"] == "identical":
        assert report["status"] == "identical", report
        return
    assert_columns_replayed(report["tables"])


def assert_columns_replayed(tables) -> None:
    """Every column of every replayed table (a replay report's ``tables``) the same to ``LAST_DIGITS``."""
    for name, table in tables.items():
        assert table["file"] and table["columns"], name
        for column, result in table["columns"].items():
            difference = result["max_relative_difference"]
            assert result["status"] == "identical" or (
                isinstance(difference, float) and difference <= LAST_DIGITS), (name, column, result)
