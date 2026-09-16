"""Named scrambled-Sobol coordinates bound to a V5 guarded split plan."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Integral
from typing import Sequence

import numpy as np
import scipy
from scipy.stats import qmc

from .split_design_v5 import V5SplitPlan


V5_SOBOL_DESIGN_SCHEMA = "gisaxs.posterior_v8.sobol_design/v1"
V5_SOBOL_DESIGN_VERSION = "scipy_scrambled_sobol_named_coordinates_bits52_v1"
_UINT32_MAX = (1 << 32) - 1
V5_SOBOL_INLINE_COORDINATE_CONTRACT_SCHEMA = (
    "gisaxs.posterior_v8.inline_named_coordinate_contract/v1"
)


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _strict_json_object(encoded: str) -> dict[str, object]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate Sobol-design JSON field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(encoded, object_pairs_hook=reject_duplicates)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid Sobol-design JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Sobol-design JSON must contain one object")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a lowercase SHA-256") from exc
    if len(raw) != 32 or value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _inline_coordinate_contract_sha256(names: tuple[str, ...]) -> str:
    return sha256(
        _canonical_json(
            {
                "schema": V5_SOBOL_INLINE_COORDINATE_CONTRACT_SCHEMA,
                "coordinate_names": list(names),
            }
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, kw_only=True)
class V5SobolDesign:
    coordinate_names: tuple[str, ...]
    scramble_seed: int
    bits: int = 52
    coordinate_contract_sha256: str | None = None

    def __post_init__(self) -> None:
        names = tuple(self.coordinate_names)
        if not names or any(not isinstance(value, str) or not value.strip() for value in names):
            raise ValueError("coordinate_names must contain non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("Sobol coordinate names must be unique")
        seed = _integer(self.scramble_seed, "scramble_seed")
        if seed > _UINT32_MAX:
            raise ValueError("scramble_seed must fit in uint32")
        bits = _integer(self.bits, "bits", minimum=1)
        if bits != 52:
            raise ValueError("V5 Sobol design freezes bits=52")
        contract_hash = (
            _inline_coordinate_contract_sha256(names)
            if self.coordinate_contract_sha256 is None
            else _sha256(self.coordinate_contract_sha256, "coordinate_contract_sha256")
        )
        object.__setattr__(self, "coordinate_names", names)
        object.__setattr__(self, "scramble_seed", seed)
        object.__setattr__(self, "bits", bits)
        object.__setattr__(self, "coordinate_contract_sha256", contract_hash)

    def payload(self) -> dict[str, object]:
        return {
            "schema": V5_SOBOL_DESIGN_SCHEMA,
            "version": V5_SOBOL_DESIGN_VERSION,
            "engine": "scipy.stats.qmc.Sobol",
            "scipy_version": scipy.__version__,
            "scramble": True,
            "bits": self.bits,
            "scramble_seed": self.scramble_seed,
            "coordinate_names": list(self.coordinate_names),
            "coordinate_contract_sha256": self.coordinate_contract_sha256,
        }

    @property
    def sha256(self) -> str:
        return sha256(_canonical_json(self.payload()).encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        payload = self.payload()
        payload["design_sha256"] = self.sha256
        return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, encoded: str) -> "V5SobolDesign":
        payload = _strict_json_object(encoded)
        expected_fields = {
            "schema",
            "version",
            "engine",
            "scipy_version",
            "scramble",
            "bits",
            "scramble_seed",
            "coordinate_names",
            "coordinate_contract_sha256",
            "design_sha256",
        }
        if set(payload) != expected_fields:
            raise ValueError("V5 Sobol-design fields are incomplete or unsupported")
        if payload.get("schema") != V5_SOBOL_DESIGN_SCHEMA:
            raise ValueError("unsupported V5 Sobol-design schema")
        names = payload.get("coordinate_names")
        if not isinstance(names, list):
            raise ValueError("coordinate_names must be a JSON array")
        try:
            replay = cls(
                coordinate_names=tuple(names),
                scramble_seed=payload["scramble_seed"],
                bits=payload["bits"],
                coordinate_contract_sha256=payload["coordinate_contract_sha256"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid V5 Sobol-design payload") from exc
        supplied_hash = payload.pop("design_sha256")
        if payload != replay.payload() or supplied_hash != replay.sha256:
            raise ValueError("V5 Sobol-design payload/hash does not reproduce")
        return replay


@dataclass(frozen=True)
class V5DesignPoint:
    sobol_index: int
    assigned_split: str
    ood_label: str | None
    clean_group_id: str
    unit_coordinates: tuple[float, ...]


def v5_clean_group_id(
    plan: V5SplitPlan,
    design: V5SobolDesign,
    sobol_index: int,
) -> str:
    """Return the immutable parent identity before views are generated."""

    if not isinstance(plan, V5SplitPlan) or not isinstance(design, V5SobolDesign):
        raise TypeError("plan/design have invalid types")
    index = _integer(sobol_index, "sobol_index")
    plan.split_for_index(index)
    return sha256(
        b"\0".join(
            (
                V5_SOBOL_DESIGN_VERSION.encode("ascii"),
                plan.sha256.encode("ascii"),
                design.sha256.encode("ascii"),
                str(index).encode("ascii"),
            )
        )
    ).hexdigest()


def materialize_v5_design_points(
    plan: V5SplitPlan,
    design: V5SobolDesign,
) -> tuple[V5DesignPoint, ...]:
    """Materialize assigned points from one reproducible scrambled Sobol prefix."""

    return materialize_v5_design_points_for_indices(plan, design, plan.assigned_indices())


def _random_access_sobol_points(
    design: V5SobolDesign,
    indices: tuple[int, ...],
) -> np.ndarray:
    """Evaluate scrambled Sobol Gray codes without allocating their prefix.

    SciPy's ``fast_forward`` is broken for ``bits=52`` in supported releases
    because its compiled path expects uint32 direction numbers.  The frozen
    design already binds the exact SciPy version, direction-number engine,
    scrambling seed, and bit depth.  Evaluating the documented Sobol Gray-code
    XOR directly from that engine's frozen direction table is byte-identical
    to ``random_base2`` and supports arbitrary uint52 indices.
    """

    engine = qmc.Sobol(
        d=len(design.coordinate_names),
        scramble=True,
        bits=design.bits,
        seed=design.scramble_seed,
    )
    directions = np.asarray(engine._sv)  # noqa: SLF001 - version-bound audited engine state
    shift = np.asarray(engine._shift)  # noqa: SLF001 - version-bound audited engine state
    scale = float(engine._scale)  # noqa: SLF001 - version-bound audited engine state
    expected = (len(design.coordinate_names), design.bits)
    if directions.shape != expected or directions.dtype != np.uint64:
        raise RuntimeError("SciPy Sobol direction-table contract changed")
    if shift.shape != (expected[0],) or shift.dtype != np.uint64:
        raise RuntimeError("SciPy Sobol digital-shift contract changed")
    integer_indices = np.asarray(indices, dtype=np.uint64)
    gray = integer_indices ^ (integer_indices >> np.uint64(1))
    quasi = np.broadcast_to(shift, (len(indices), shift.size)).copy()
    for bit in range(design.bits):
        selected = (gray & (np.uint64(1) << np.uint64(bit))) != 0
        quasi[selected] ^= directions[:, bit]
    return quasi.astype(np.float64) * scale


def materialize_v5_unit_coordinates_for_indices(
    design: V5SobolDesign,
    indices: Sequence[int],
) -> tuple[tuple[float, ...], ...]:
    """Materialize raw design coordinates without assigning a global split plan."""

    if not isinstance(design, V5SobolDesign):
        raise TypeError("design must be a V5SobolDesign")
    if isinstance(indices, (str, bytes)):
        raise TypeError("indices must be a sequence of integers")
    try:
        requested = tuple(_integer(value, "sobol_index") for value in indices)
    except TypeError as exc:
        raise TypeError("indices must be a sequence of integers") from exc
    if not requested:
        raise ValueError("indices cannot be empty")
    if len(set(requested)) != len(requested):
        raise ValueError("indices must be unique")
    if max(requested) >= 1 << design.bits:
        raise ValueError("requested index exceeds the frozen Sobol period")
    coordinates = _random_access_sobol_points(design, requested)
    return tuple(
        tuple(float(value) for value in np.asarray(row, dtype="<f8"))
        for row in coordinates
    )


def materialize_v5_design_points_for_indices(
    plan: V5SplitPlan,
    design: V5SobolDesign,
    indices: Sequence[int],
) -> tuple[V5DesignPoint, ...]:
    """Materialize selected assigned indices without allocating the full prefix.

    Output follows the caller's index order.  Coordinates are byte-identical to
    full-prefix materialization by the exact frozen SciPy engine, including on
    SciPy releases whose ``fast_forward`` cannot handle the required 52 bits.
    """

    if not isinstance(plan, V5SplitPlan) or not isinstance(design, V5SobolDesign):
        raise TypeError("plan/design have invalid types")
    if isinstance(indices, (str, bytes)):
        raise TypeError("indices must be a sequence of integers")
    try:
        requested = tuple(_integer(value, "sobol_index") for value in indices)
    except TypeError as exc:
        raise TypeError("indices must be a sequence of integers") from exc
    if not requested:
        raise ValueError("indices cannot be empty")
    if len(set(requested)) != len(requested):
        raise ValueError("indices must be unique")
    for index in requested:
        plan.split_for_index(index)
    if max(requested) >= 1 << design.bits:
        raise ValueError("requested index exceeds the frozen Sobol period")

    coordinates = materialize_v5_unit_coordinates_for_indices(design, requested)
    return tuple(
        V5DesignPoint(
            index,
            plan.split_for_index(index),
            plan.ood_label_for_index(index),
            v5_clean_group_id(plan, design, index),
            row,
        )
        for index, row in zip(requested, coordinates)
    )


__all__ = [
    "V5_SOBOL_DESIGN_SCHEMA",
    "V5_SOBOL_DESIGN_VERSION",
    "V5_SOBOL_INLINE_COORDINATE_CONTRACT_SCHEMA",
    "V5DesignPoint",
    "V5SobolDesign",
    "materialize_v5_design_points",
    "materialize_v5_design_points_for_indices",
    "materialize_v5_unit_coordinates_for_indices",
    "v5_clean_group_id",
]
