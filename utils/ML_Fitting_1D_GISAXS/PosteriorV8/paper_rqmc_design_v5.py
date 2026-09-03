"""Formal independent-scramble RQMC design for V5 paper holdouts.

Every ``(split, replicate)`` cell owns a fresh scrambled Sobol engine with a
globally unique, predeclared seed.  Independent scramble *replicate means* are
the randomization units.  Points inside one scramble are a dependent QMC net:
they are never described as iid and must never be individually bootstrapped.

This module freezes design metadata and can materialize a complete power-of-two
Sobol prefix.  It does not alter the legacy V5 split contract and makes no
population claim about a single realized scramble.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np
import scipy
from scipy.stats import qmc


V5_PAPER_RQMC_DESIGN_SCHEMA = "gisaxs.posterior_v8.paper_rqmc_replicate_design/v3"
V5_PAPER_RQMC_DESIGN_VERSION = (
    "independent_test_reference_ood_scrambles_minimum8_target16_"
    "authoritative_coordinate_contract_v3"
)
V5_PAPER_RQMC_IDENTITY_SCHEMA = "gisaxs.posterior_v8.paper_rqmc_identity/v3"
V5_PAPER_RQMC_SPLITS = ("test", "reference", "ood")
V5_PAPER_RQMC_ENGINEERING_MIN_INDEPENDENT_REPLICATES = 4
V5_PAPER_RQMC_MIN_INDEPENDENT_REPLICATES = 8
V5_PAPER_RQMC_TARGET_INDEPENDENT_REPLICATES = 16
V5_PAPER_RQMC_RANDOMIZATION_UNIT = "independent_scramble_replicate_mean"
V5_PAPER_RQMC_INFERENCE_RULE = "student_t_over_paired_independent_scramble_replicate_means"
V5_PAPER_RQMC_CONFORMAL_CALIBRATION_INCLUDED = False
V5_COMPATIBILITY_CALIBRATION_SAMPLING_REQUIREMENT = (
    "separate_iid_exchangeable_calibration_contract_required_for_finite_sample_"
    "split_conformal_coverage"
)

_ENGINE = "scipy.stats.qmc.Sobol"
_BITS = 52
_UINT32_MAX = (1 << 32) - 1


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _strict_json_object(encoded: str) -> dict[str, object]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate RQMC-design JSON field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(encoded, object_pairs_hook=reject_duplicates)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid RQMC-design JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("RQMC-design JSON must contain one object")
    return value


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a lowercase SHA-256") from exc
    if len(raw) != 32:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _identity(kind: str, payload: Mapping[str, object]) -> str:
    return sha256(
        _canonical_json(
            {
                "schema": V5_PAPER_RQMC_IDENTITY_SCHEMA,
                "kind": kind,
                **dict(payload),
            }
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, kw_only=True)
class V5PaperRQMCCoordinateContractIdentity:
    """Authoritative external coordinate transform identity, never names-only."""

    schema: str
    version: str
    sha256: str
    dimension: int
    coordinate_names: tuple[str, ...]

    def __post_init__(self) -> None:
        schema = _text(self.schema, "coordinate contract schema")
        version = _text(self.version, "coordinate contract version")
        digest = _digest(self.sha256, "coordinate contract sha256")
        dimension = _integer(self.dimension, "coordinate contract dimension", minimum=1)
        names = tuple(_text(value, "coordinate_name") for value in self.coordinate_names)
        if len(names) != dimension or len(names) != len(set(names)):
            raise ValueError(
                "coordinate names must be unique and match the authoritative dimension"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "coordinate_names", names)

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "sha256": self.sha256,
            "dimension": self.dimension,
            "coordinate_names": list(self.coordinate_names),
        }

    @property
    def identity_sha256(self) -> str:
        return _identity("authoritative_coordinate_contract", self.payload())


@dataclass(frozen=True, kw_only=True)
class V5PaperRQMCSplitSpec:
    split_name: str
    points_per_replicate: int
    scramble_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        name = _text(self.split_name, "split_name")
        if name not in V5_PAPER_RQMC_SPLITS:
            raise ValueError("split_name is not a formal paper RQMC split")
        count = _integer(self.points_per_replicate, "points_per_replicate", minimum=2)
        if count & (count - 1):
            raise ValueError("points_per_replicate must be a power of two")
        seeds = tuple(_integer(value, "scramble_seed") for value in self.scramble_seeds)
        if len(seeds) < V5_PAPER_RQMC_MIN_INDEPENDENT_REPLICATES:
            raise ValueError(
                "formal RQMC splits require at least eight independent scramble replicates"
            )
        if any(value > _UINT32_MAX for value in seeds):
            raise ValueError("scramble_seed must fit in uint32")
        if len(seeds) != len(set(seeds)):
            raise ValueError("scramble seeds must be unique within a split")
        object.__setattr__(self, "split_name", name)
        object.__setattr__(self, "points_per_replicate", count)
        object.__setattr__(self, "scramble_seeds", seeds)


@dataclass(frozen=True, kw_only=True)
class V5PaperRQMCReplicateBlock:
    split_name: str
    split_id: str
    replicate_index: int
    replicate_id: str
    scramble_seed: int
    scramble_seed_id: str
    sobol_start: int
    sobol_stop: int
    block_id: str

    @property
    def count(self) -> int:
        return self.sobol_stop - self.sobol_start


def _blocks_for_spec(
    spec: V5PaperRQMCSplitSpec,
    *,
    coordinate_contract: V5PaperRQMCCoordinateContractIdentity,
) -> tuple[V5PaperRQMCReplicateBlock, ...]:
    split_id = _identity(
        "formal_split",
        {
            "design_version": V5_PAPER_RQMC_DESIGN_VERSION,
            "split_name": spec.split_name,
            "coordinate_contract": coordinate_contract.payload(),
            "coordinate_contract_identity_sha256": coordinate_contract.identity_sha256,
            "bits": _BITS,
            "points_per_replicate": spec.points_per_replicate,
            "scramble_seeds": list(spec.scramble_seeds),
        },
    )
    blocks = []
    for replicate_index, seed in enumerate(spec.scramble_seeds):
        seed_id = _identity(
            "scramble_seed",
            {
                "split_id": split_id,
                "replicate_index": replicate_index,
                "scramble_seed": seed,
            },
        )
        replicate_id = _identity(
            "independent_scramble_replicate",
            {
                "split_id": split_id,
                "replicate_index": replicate_index,
                "scramble_seed_id": seed_id,
            },
        )
        block_id = _identity(
            "sobol_prefix_block",
            {
                "replicate_id": replicate_id,
                "sobol_start": 0,
                "sobol_stop": spec.points_per_replicate,
            },
        )
        blocks.append(
            V5PaperRQMCReplicateBlock(
                split_name=spec.split_name,
                split_id=split_id,
                replicate_index=replicate_index,
                replicate_id=replicate_id,
                scramble_seed=seed,
                scramble_seed_id=seed_id,
                sobol_start=0,
                sobol_stop=spec.points_per_replicate,
                block_id=block_id,
            )
        )
    return tuple(blocks)


def _design_payload(
    coordinate_contract: V5PaperRQMCCoordinateContractIdentity,
    specs: tuple[V5PaperRQMCSplitSpec, ...],
    blocks: tuple[V5PaperRQMCReplicateBlock, ...],
) -> dict[str, object]:
    by_split = {name: [] for name in V5_PAPER_RQMC_SPLITS}
    for block in blocks:
        by_split[block.split_name].append(asdict(block))
    return {
        "schema": V5_PAPER_RQMC_DESIGN_SCHEMA,
        "version": V5_PAPER_RQMC_DESIGN_VERSION,
        "engine": _ENGINE,
        "scipy_version": scipy.__version__,
        "scramble": True,
        "bits": _BITS,
        "coordinate_contract": coordinate_contract.payload(),
        "coordinate_contract_identity_sha256": coordinate_contract.identity_sha256,
        "randomization_unit": V5_PAPER_RQMC_RANDOMIZATION_UNIT,
        "engineering_minimum_independent_replicates": (
            V5_PAPER_RQMC_ENGINEERING_MIN_INDEPENDENT_REPLICATES
        ),
        "engineering_minimum_allows_formal_inference": False,
        "formal_minimum_independent_replicates": (V5_PAPER_RQMC_MIN_INDEPENDENT_REPLICATES),
        "paper_target_independent_replicates": (V5_PAPER_RQMC_TARGET_INDEPENDENT_REPLICATES),
        "within_scramble_points_are_iid": False,
        "point_bootstrap_allowed": False,
        "finite_sample_conformal_calibration_included": (
            V5_PAPER_RQMC_CONFORMAL_CALIBRATION_INCLUDED
        ),
        "compatibility_calibration_sampling_requirement": (
            V5_COMPATIBILITY_CALIBRATION_SAMPLING_REQUIREMENT
        ),
        "intended_inference": V5_PAPER_RQMC_INFERENCE_RULE,
        "splits": [
            {
                "split_name": spec.split_name,
                "split_id": by_split[spec.split_name][0]["split_id"],
                "points_per_replicate": spec.points_per_replicate,
                "replicate_count": len(spec.scramble_seeds),
                "blocks": by_split[spec.split_name],
            }
            for spec in specs
        ],
    }


@dataclass(frozen=True)
class V5PaperRQMCDesign:
    coordinate_contract: V5PaperRQMCCoordinateContractIdentity
    split_specs: tuple[V5PaperRQMCSplitSpec, ...]
    blocks: tuple[V5PaperRQMCReplicateBlock, ...]
    canonical_json: str
    sha256: str

    @classmethod
    def create(
        cls,
        *,
        coordinate_contract: V5PaperRQMCCoordinateContractIdentity,
        split_specs: Sequence[V5PaperRQMCSplitSpec],
    ) -> "V5PaperRQMCDesign":
        if not isinstance(coordinate_contract, V5PaperRQMCCoordinateContractIdentity):
            raise TypeError("coordinate_contract must be an authoritative typed identity")
        if isinstance(split_specs, (str, bytes)):
            raise TypeError("split_specs must be a sequence")
        specs = tuple(split_specs)
        if not all(isinstance(value, V5PaperRQMCSplitSpec) for value in specs):
            raise TypeError("split_specs contain an invalid value")
        if {value.split_name for value in specs} != set(V5_PAPER_RQMC_SPLITS) or len(specs) != len(
            V5_PAPER_RQMC_SPLITS
        ):
            raise ValueError("split_specs must contain each formal paper split exactly once")
        order = {name: index for index, name in enumerate(V5_PAPER_RQMC_SPLITS)}
        specs = tuple(sorted(specs, key=lambda value: order[value.split_name]))
        all_seeds = tuple(seed for spec in specs for seed in spec.scramble_seeds)
        if len(all_seeds) != len(set(all_seeds)):
            raise ValueError(
                "every formal split/replicate must use a globally unique scramble seed"
            )
        blocks = tuple(
            block
            for spec in specs
            for block in _blocks_for_spec(spec, coordinate_contract=coordinate_contract)
        )
        canonical = _canonical_json(_design_payload(coordinate_contract, specs, blocks))
        return cls(
            coordinate_contract,
            specs,
            blocks,
            canonical,
            sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def __post_init__(self) -> None:
        coordinate_contract = self.coordinate_contract
        if not isinstance(coordinate_contract, V5PaperRQMCCoordinateContractIdentity):
            raise TypeError("coordinate_contract must be an authoritative typed identity")
        specs = tuple(self.split_specs)
        if not all(isinstance(value, V5PaperRQMCSplitSpec) for value in specs):
            raise TypeError("split_specs contain an invalid value")
        if tuple(value.split_name for value in specs) != V5_PAPER_RQMC_SPLITS:
            raise ValueError("RQMC design splits are not canonical")
        all_seeds = tuple(seed for spec in specs for seed in spec.scramble_seeds)
        if len(all_seeds) != len(set(all_seeds)):
            raise ValueError(
                "every formal split/replicate must use a globally unique scramble seed"
            )
        blocks = tuple(
            block
            for spec in specs
            for block in _blocks_for_spec(spec, coordinate_contract=coordinate_contract)
        )
        canonical = _canonical_json(_design_payload(coordinate_contract, specs, blocks))
        digest = sha256(canonical.encode("utf-8")).hexdigest()
        if self.blocks != blocks or self.canonical_json != canonical or self.sha256 != digest:
            raise ValueError("RQMC design does not reproduce its identities or audit hash")

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        return self.coordinate_contract.coordinate_names

    @property
    def coordinate_contract_sha256(self) -> str:
        return self.coordinate_contract.sha256

    def blocks_for_split(self, split_name: str) -> tuple[V5PaperRQMCReplicateBlock, ...]:
        name = _text(split_name, "split_name")
        values = tuple(value for value in self.blocks if value.split_name == name)
        if not values:
            raise ValueError("split_name is absent from the formal RQMC design")
        return values

    def block(self, split_name: str, replicate_index: int) -> V5PaperRQMCReplicateBlock:
        index = _integer(replicate_index, "replicate_index")
        matches = tuple(
            value for value in self.blocks_for_split(split_name) if value.replicate_index == index
        )
        if len(matches) != 1:
            raise ValueError("replicate_index is absent from the selected split")
        return matches[0]

    def to_json(self) -> str:
        payload = json.loads(self.canonical_json)
        payload["design_sha256"] = self.sha256
        return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, encoded: str) -> "V5PaperRQMCDesign":
        payload = _strict_json_object(encoded)
        expected = {
            "schema",
            "version",
            "engine",
            "scipy_version",
            "scramble",
            "bits",
            "coordinate_contract",
            "coordinate_contract_identity_sha256",
            "randomization_unit",
            "engineering_minimum_independent_replicates",
            "engineering_minimum_allows_formal_inference",
            "formal_minimum_independent_replicates",
            "paper_target_independent_replicates",
            "within_scramble_points_are_iid",
            "point_bootstrap_allowed",
            "finite_sample_conformal_calibration_included",
            "compatibility_calibration_sampling_requirement",
            "intended_inference",
            "splits",
            "design_sha256",
        }
        if (
            set(payload) != expected
            or payload.get("schema") != V5_PAPER_RQMC_DESIGN_SCHEMA
            or payload.get("version") != V5_PAPER_RQMC_DESIGN_VERSION
        ):
            raise ValueError("unsupported or incomplete formal RQMC-design fields")
        if payload.get("within_scramble_points_are_iid") is not False:
            raise ValueError("within-scramble Sobol points are not iid")
        if (
            payload.get("engineering_minimum_independent_replicates")
            != V5_PAPER_RQMC_ENGINEERING_MIN_INDEPENDENT_REPLICATES
            or payload.get("engineering_minimum_allows_formal_inference") is not False
            or payload.get("formal_minimum_independent_replicates")
            != V5_PAPER_RQMC_MIN_INDEPENDENT_REPLICATES
            or payload.get("paper_target_independent_replicates")
            != V5_PAPER_RQMC_TARGET_INDEPENDENT_REPLICATES
        ):
            raise ValueError("unsupported RQMC replicate-count claim")
        if payload.get("point_bootstrap_allowed") is not False:
            raise ValueError("individual Sobol-point bootstrap is forbidden")
        if payload.get("finite_sample_conformal_calibration_included") is not False:
            raise ValueError("finite-sample conformal calibration requires a separate iid contract")
        if (
            payload.get("compatibility_calibration_sampling_requirement")
            != V5_COMPATIBILITY_CALIBRATION_SAMPLING_REQUIREMENT
        ):
            raise ValueError("unsupported compatibility-calibration sampling claim")
        if (
            payload.get("randomization_unit") != V5_PAPER_RQMC_RANDOMIZATION_UNIT
            or payload.get("intended_inference") != V5_PAPER_RQMC_INFERENCE_RULE
        ):
            raise ValueError("unsupported RQMC inferential-unit claim")
        raw_contract = payload.get("coordinate_contract")
        raw_splits = payload.get("splits")
        if not isinstance(raw_contract, Mapping) or not isinstance(raw_splits, list):
            raise ValueError("RQMC coordinate contract/splits have invalid types")
        specs = []
        try:
            coordinate_contract = V5PaperRQMCCoordinateContractIdentity(
                schema=raw_contract["schema"],
                version=raw_contract["version"],
                sha256=raw_contract["sha256"],
                dimension=raw_contract["dimension"],
                coordinate_names=tuple(raw_contract["coordinate_names"]),
            )
            for raw_split in raw_splits:
                if not isinstance(raw_split, Mapping):
                    raise TypeError("split must be an object")
                raw_blocks = raw_split.get("blocks")
                if not isinstance(raw_blocks, list):
                    raise TypeError("blocks must be an array")
                ordered = sorted(raw_blocks, key=lambda value: value["replicate_index"])
                specs.append(
                    V5PaperRQMCSplitSpec(
                        split_name=raw_split["split_name"],
                        points_per_replicate=raw_split["points_per_replicate"],
                        scramble_seeds=tuple(value["scramble_seed"] for value in ordered),
                    )
                )
            replay = cls.create(
                coordinate_contract=coordinate_contract,
                split_specs=tuple(specs),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid formal RQMC-design payload") from exc
        supplied_hash = payload.pop("design_sha256")
        if payload != json.loads(replay.canonical_json) or supplied_hash != replay.sha256:
            raise ValueError("formal RQMC-design payload/hash does not reproduce")
        return replay


def v5_recipe_coordinate_contract_identity() -> V5PaperRQMCCoordinateContractIdentity:
    """Read the live owner constants for the authoritative V5 recipe map."""

    from .sobol_recipe_coordinates_v5 import (  # local import avoids a generic-design cycle
        V5_SOBOL_RECIPE_COORDINATE_NAMES,
        V5_SOBOL_RECIPE_COORDINATE_SCHEMA,
        V5_SOBOL_RECIPE_COORDINATE_SHA256,
        V5_SOBOL_RECIPE_COORDINATE_VERSION,
        V5_SOBOL_RECIPE_DIM,
    )

    return V5PaperRQMCCoordinateContractIdentity(
        schema=V5_SOBOL_RECIPE_COORDINATE_SCHEMA,
        version=V5_SOBOL_RECIPE_COORDINATE_VERSION,
        sha256=V5_SOBOL_RECIPE_COORDINATE_SHA256,
        dimension=V5_SOBOL_RECIPE_DIM,
        coordinate_names=V5_SOBOL_RECIPE_COORDINATE_NAMES,
    )


def create_v5_recipe_paper_rqmc_design(
    *, split_specs: Sequence[V5PaperRQMCSplitSpec]
) -> V5PaperRQMCDesign:
    """Create a paper design bound to the current authoritative V5 recipe contract."""

    return V5PaperRQMCDesign.create(
        coordinate_contract=v5_recipe_coordinate_contract_identity(),
        split_specs=split_specs,
    )


@dataclass(frozen=True)
class V5PaperRQMCPoint:
    split_name: str
    split_id: str
    replicate_index: int
    replicate_id: str
    block_id: str
    scramble_seed_id: str
    sobol_index: int
    point_id: str
    unit_coordinates: tuple[float, ...]


def materialize_v5_paper_rqmc_block(
    design: V5PaperRQMCDesign,
    block: V5PaperRQMCReplicateBlock,
) -> tuple[V5PaperRQMCPoint, ...]:
    """Materialize one complete net; its points are not inferential replicates."""

    if not isinstance(design, V5PaperRQMCDesign) or not isinstance(
        block, V5PaperRQMCReplicateBlock
    ):
        raise TypeError("design/block have invalid types")
    if design.block(block.split_name, block.replicate_index) != block:
        raise ValueError("block is not bound to this formal RQMC design")
    exponent = block.count.bit_length() - 1
    engine = qmc.Sobol(
        d=len(design.coordinate_names),
        scramble=True,
        bits=_BITS,
        seed=block.scramble_seed,
    )
    coordinates = engine.random_base2(exponent)
    return tuple(
        V5PaperRQMCPoint(
            split_name=block.split_name,
            split_id=block.split_id,
            replicate_index=block.replicate_index,
            replicate_id=block.replicate_id,
            block_id=block.block_id,
            scramble_seed_id=block.scramble_seed_id,
            sobol_index=index,
            point_id=_identity(
                "sobol_design_point",
                {"block_id": block.block_id, "sobol_index": index},
            ),
            unit_coordinates=tuple(float(value) for value in np.asarray(row, dtype="<f8")),
        )
        for index, row in enumerate(coordinates)
    )


__all__ = [
    "V5_COMPATIBILITY_CALIBRATION_SAMPLING_REQUIREMENT",
    "V5_PAPER_RQMC_CONFORMAL_CALIBRATION_INCLUDED",
    "V5_PAPER_RQMC_DESIGN_SCHEMA",
    "V5_PAPER_RQMC_DESIGN_VERSION",
    "V5_PAPER_RQMC_ENGINEERING_MIN_INDEPENDENT_REPLICATES",
    "V5_PAPER_RQMC_IDENTITY_SCHEMA",
    "V5_PAPER_RQMC_INFERENCE_RULE",
    "V5_PAPER_RQMC_MIN_INDEPENDENT_REPLICATES",
    "V5_PAPER_RQMC_RANDOMIZATION_UNIT",
    "V5_PAPER_RQMC_SPLITS",
    "V5_PAPER_RQMC_TARGET_INDEPENDENT_REPLICATES",
    "V5PaperRQMCCoordinateContractIdentity",
    "V5PaperRQMCDesign",
    "V5PaperRQMCPoint",
    "V5PaperRQMCReplicateBlock",
    "V5PaperRQMCSplitSpec",
    "create_v5_recipe_paper_rqmc_design",
    "materialize_v5_paper_rqmc_block",
    "v5_recipe_coordinate_contract_identity",
]
