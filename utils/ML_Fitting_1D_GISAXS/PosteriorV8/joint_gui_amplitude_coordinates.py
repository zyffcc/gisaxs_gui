"""Feasible direct coordinates for independent GUI amplitude parameters.

The GUI exposes one shared positive ``k`` and independent ``Int_i`` and
``int_Res`` values.  Mapping that product box directly to effective
coefficients, ``a_i=k*Int_i``, makes every exact-forward objective call legal
without inventing a unit-sum simplex.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .gui_amplitude_constraints import CoefficientPolytope, GuiAmplitudeConstraint


JOINT_GUI_AMPLITUDE_COORDINATES_VERSION = (
    "posterior_v8_joint_independent_gui_amplitude_coordinates_v2"
)
_UNIT_TOLERANCE = 1.0e-12


def _unit_value(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < -_UNIT_TOLERANCE or result > 1.0 + _UNIT_TOLERANCE:
        raise ValueError(f"{name} must be inside the unit interval")
    return float(np.clip(result, 0.0, 1.0))


def _linear_decode(unit: float, low: float, high: float) -> float:
    if high == low:
        return low
    return low + unit * (high - low)


def _linear_encode(value: float, low: float, high: float, name: str) -> float:
    if value < low - _UNIT_TOLERANCE or value > high + _UNIT_TOLERANCE:
        raise ValueError(f"{name} is outside its GUI interval")
    if high == low:
        return 0.5
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


@dataclass(frozen=True)
class FeasibleGuiAmplitudeCoordinates:
    """Unit-cube coordinates for one independent GUI amplitude box.

    ``k=0`` is excluded because a positive effective particle coefficient
    cannot be reconstructed from it.  If the requested interval starts at
    zero, the smallest positive normal float represents that open boundary.
    """

    constraint: GuiAmplitudeConstraint
    polytope: CoefficientPolytope
    coordinate_names: tuple[str, ...]
    effective_k_lower: float
    version: str = JOINT_GUI_AMPLITUDE_COORDINATES_VERSION

    @classmethod
    def build(cls, constraint: GuiAmplitudeConstraint) -> FeasibleGuiAmplitudeCoordinates:
        if not isinstance(constraint, GuiAmplitudeConstraint):
            raise TypeError("constraint must be a GuiAmplitudeConstraint")
        polytope = constraint.coefficient_polytope()
        k_lower = max(constraint.k.low, np.finfo(np.float64).tiny)
        if k_lower > constraint.k.high:
            raise ValueError("GUI k interval has no numerically representable positive value")
        names = ("BG", "k")
        names += tuple(f"Int_{index}" for index in range(1, constraint.particle_count + 1))
        if constraint.resolution_present:
            names += ("int_Res",)
        return cls(
            constraint=constraint,
            polytope=polytope,
            coordinate_names=names,
            effective_k_lower=float(k_lower),
        )

    @property
    def size(self) -> int:
        return len(self.coordinate_names)

    @property
    def lower(self) -> np.ndarray:
        return np.zeros(self.size, dtype=np.float64)

    @property
    def upper(self) -> np.ndarray:
        return np.ones(self.size, dtype=np.float64)

    @property
    def varying_mask(self) -> np.ndarray:
        intervals = [self.constraint.background]
        intervals.append(
            type(self.constraint.k)(self.effective_k_lower, self.constraint.k.high)
        )
        intervals.extend(self.constraint.component_intensities)
        if self.constraint.resolution_present:
            assert self.constraint.int_res is not None
            intervals.append(self.constraint.int_res)
        return np.asarray(
            [value.high > value.low for value in intervals],
            dtype=np.bool_,
        )

    def decode_with_k(self, coordinates: Sequence[float]) -> tuple[np.ndarray, float]:
        values = np.asarray(coordinates, dtype=np.float64)
        if values.shape != (self.size,) or not np.all(np.isfinite(values)):
            raise ValueError("GUI amplitude coordinates have the wrong shape or are non-finite")
        unit = np.asarray(
            [_unit_value(value, name) for value, name in zip(values, self.coordinate_names)],
            dtype=np.float64,
        )
        cursor = 0
        background = _linear_decode(
            unit[cursor], self.constraint.background.low, self.constraint.background.high
        )
        cursor += 1
        k_value = _linear_decode(unit[cursor], self.effective_k_lower, self.constraint.k.high)
        cursor += 1
        intensities = []
        for interval in self.constraint.component_intensities:
            intensities.append(_linear_decode(unit[cursor], interval.low, interval.high))
            cursor += 1
        coefficients = [background, *(k_value * value for value in intensities)]
        if self.constraint.resolution_present:
            assert self.constraint.int_res is not None
            int_res = _linear_decode(
                unit[cursor], self.constraint.int_res.low, self.constraint.int_res.high
            )
            coefficients.append(k_value * int_res)
        result = np.asarray(coefficients, dtype=np.float64)
        if not self.polytope.contains_with_k(result, k_value):
            raise RuntimeError("direct GUI parameterization escaped its requested ranges")
        return result, float(k_value)

    def decode(self, coordinates: Sequence[float]) -> np.ndarray:
        coefficients, _ = self.decode_with_k(coordinates)
        return coefficients

    def encode(
        self,
        coefficients: Sequence[float],
        *,
        k: float | None = None,
    ) -> np.ndarray:
        values = np.asarray(coefficients, dtype=np.float64)
        if not self.polytope.contains(values):
            raise ValueError("amplitudes are outside the requested GUI amplitude polytope")
        count = self.constraint.particle_count
        k_value = self.polytope.select_k(values, preferred=k)
        intensities = values[1 : count + 1] / k_value
        encoded = [
            _linear_encode(
                float(values[0]),
                self.constraint.background.low,
                self.constraint.background.high,
                "BG",
            ),
            _linear_encode(
                k_value,
                self.effective_k_lower,
                self.constraint.k.high,
                "k",
            ),
        ]
        encoded.extend(
            _linear_encode(float(value), interval.low, interval.high, f"Int_{index}")
            for index, (value, interval) in enumerate(
                zip(intensities, self.constraint.component_intensities), 1
            )
        )
        if self.constraint.resolution_present:
            assert self.constraint.int_res is not None
            encoded.append(
                _linear_encode(
                    float(values[-1] / k_value),
                    self.constraint.int_res.low,
                    self.constraint.int_res.high,
                    "int_Res",
                )
            )
        result = np.asarray(encoded, dtype=np.float64)
        # Round-trip validation prevents a merely axis-feasible seed from
        # entering the exact objective under floating-point edge cases.
        decoded, decoded_k = self.decode_with_k(result)
        if not np.allclose(decoded, values, rtol=2.0e-12, atol=1.0e-12):
            raise RuntimeError("GUI amplitude coordinate round trip is inconsistent")
        if not np.isclose(decoded_k, k_value, rtol=2.0e-12, atol=1.0e-12):
            raise RuntimeError("GUI k coordinate round trip is inconsistent")
        return result


__all__ = [
    "JOINT_GUI_AMPLITUDE_COORDINATES_VERSION",
    "FeasibleGuiAmplitudeCoordinates",
]
