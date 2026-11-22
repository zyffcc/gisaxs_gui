"""The χ² profile of one parameter (Physics and Methods, section 9.16).

The parameter is fixed at each value of a grid and the other free values are fitted again (Refine, from the
neighbouring grid point's result, outwards from the best fit). Δχ² is the rise of Σr² over the best fit in units of
s² = Σr²/(N − p) of the best fit, the same scale as the covariance errors; the 68 % interval is where Δχ² ≤ 1
(linear interpolation between grid points; open on a side where the grid ends first). With χ²ᵣ = 1 this is the usual
Δχ² = 1 rule; a curve whose points are weighted relatively (no σ) gets the interval its scatter allows, as its errors do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .fit_engine import FitData, FitStopped, fit, residuals
from .fit_model import FitModel

POINTS = 15
SPAN = 3.0
"""The default grid: ±``SPAN`` errors around the best value (its range when the error is not known)."""


@dataclass(frozen=True)
class Profile:
    path: tuple
    values: tuple
    totals: tuple
    """Σr² at each value (NaN where the fit failed)."""
    best_value: float
    best_total: float
    scale: float
    """s² = Σr²/(N − p) of the best fit."""
    stopped: bool = False

    @property
    def delta(self) -> np.ndarray:
        return (np.asarray(self.totals, float) - self.best_total) / self.scale

    def interval(self) -> tuple[Optional[float], Optional[float]]:
        """(low, high) where Δχ² crosses 1 on each side of the best value; None on a side where it does not."""
        values, delta = np.asarray(self.values, float), self.delta
        order = np.argsort(values)
        values, delta = values[order], delta[order]
        below = values <= self.best_value
        low = _crossing(values[below][::-1], delta[below][::-1], self.best_value)
        high = _crossing(values[~below], delta[~below], self.best_value)
        return low, high


def _crossing(values: np.ndarray, delta: np.ndarray, start: float) -> Optional[float]:
    """The first value, walking away from the best one, where Δχ² reaches 1."""
    previous_value, previous_delta = start, 0.0
    for value, rise in zip(values, delta):
        if not math.isfinite(rise):
            continue
        if rise >= 1.0:
            if rise == previous_delta:
                return float(value)
            return float(previous_value + (1.0 - previous_delta) * (value - previous_value) / (rise - previous_delta))
        previous_value, previous_delta = value, rise
    return None


def grid(model: FitModel, path: tuple, error: Optional[float], points: int = POINTS) -> np.ndarray:
    """``points`` values: ±``SPAN`` errors around the value, inside the parameter's range; its range when the error
    is not known (a log grid when the range spans a decade and more)."""
    parameter = model.get(path)
    low, high = parameter.lower, parameter.upper
    if error is not None and math.isfinite(error) and error > 0:
        low, high = max(low, parameter.value - SPAN * error), min(high, parameter.value + SPAN * error)
    if not (math.isfinite(low) and math.isfinite(high) and high > low):
        raise ValueError("Give this parameter a finite range to draw its profile")
    if low > 0 and high >= 10 * low:
        return np.geomspace(low, high, points)
    return np.linspace(low, high, points)


def profile(model: FitModel, data: FitData, path: tuple, values, *, progress: Optional[Callable[[float, str], None]] = None,
            stop: Optional[Callable[[], bool]] = None) -> Profile:
    """The profile of ``path`` over ``values`` for the fitted ``model`` (its values are the best fit)."""
    best = residuals(model, data)
    free = sum(1 for _path, p in model.parameters() if p.free and not p.tie)
    best_total = float(best @ best)
    scale = best_total / max(data.q.size - free, 1)
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError("The best fit has no scatter to measure the profile against")
    centre = float(model.get(path).value)
    values = np.asarray(sorted(float(v) for v in values), float)
    totals = np.full(values.shape, np.nan)
    by_distance = list(np.argsort(np.abs(values - centre), kind="stable"))  # outwards from the best fit, each side
    upwards = [i for i in by_distance if values[i] >= centre]
    downwards = [i for i in by_distance if values[i] < centre]
    done, stopped = 0, False
    for side in (upwards, downwards):
        start = model
        for index in side:
            if stop is not None and stop():
                stopped = True
                break
            trial = start.with_parameter(path, value=float(values[index]), free=False)
            try:
                result = fit(trial, data, method="local", stop=stop)
            except FitStopped:
                stopped = True
                break
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                continue
            if result.stopped:
                stopped = True
                break
            residual = residuals(result.model, data)
            totals[index] = float(residual @ residual) if np.all(np.isfinite(residual)) else math.nan
            start = result.model.with_parameter(path, free=model.get(path).free)
            done += 1
            if progress is not None:
                progress(done / len(values), f"{done} of {len(values)} values")
        if stopped:
            break
    return Profile(tuple(path), tuple(values.tolist()), tuple(totals.tolist()), centre, best_total, scale, stopped)


__all__ = ["POINTS", "SPAN", "Profile", "grid", "profile"]
