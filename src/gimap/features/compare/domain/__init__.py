"""Compare several series of curves: pure NumPy / SciPy."""

from .comparison import (
    COMPARE_ERRORS,
    END_FRAMES,
    Comparison,
    SeriesData,
    SeriesResult,
    common_grid,
    compare,
    on_grid,
)

__all__ = ["COMPARE_ERRORS", "END_FRAMES", "Comparison", "SeriesData", "SeriesResult", "common_grid", "compare", "on_grid"]
