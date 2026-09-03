"""Deterministic low-discrepancy seeds for Posterior V8 branch refinement."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np
from scipy.stats import qmc

from .branch_codec import (
    BRANCH_CODEC_VERSION,
    BranchCoordinates,
    ProfiledBranchCodec,
    ResolutionBounds,
    UNIT_CUBE_DIMENSIONS,
)
from .contract import GuiComponentBounds, LatentComponentParameters
from .profiled_forward import ResolutionShape


PROPOSAL_SAMPLING_VERSION = "posterior_v8_scrambled_sobol_branch_v1"


@dataclass(frozen=True)
class ProfiledBranchSeed:
    """One proposal plus arguments accepted directly by ``refine_profiled_branch``."""

    topology_id: int
    topology: tuple[str, ...]
    d_present: tuple[bool, ...]
    component_bounds: tuple[GuiComponentBounds, ...]
    seed_components: tuple[LatentComponentParameters, ...]
    resolution_bounds: ResolutionBounds | None
    resolution_seed: ResolutionShape | None
    unit_cube: tuple[float, ...]
    active_mask: tuple[bool, ...]
    sequence_index: int
    generator_seed: int
    codec_version: str = BRANCH_CODEC_VERSION
    generator_version: str = PROPOSAL_SAMPLING_VERSION

    def __post_init__(self) -> None:
        coordinates = BranchCoordinates(self.unit_cube, self.active_mask)
        object.__setattr__(self, "unit_cube", coordinates.unit_cube)
        object.__setattr__(self, "active_mask", coordinates.active_mask)
        if self.codec_version != BRANCH_CODEC_VERSION:
            raise ValueError("unsupported branch codec version")
        if self.generator_version != PROPOSAL_SAMPLING_VERSION:
            raise ValueError("unsupported proposal sampling version")

    @property
    def coordinates(self) -> BranchCoordinates:
        return BranchCoordinates(self.unit_cube, self.active_mask)

    def refinement_kwargs(self) -> dict[str, object]:
        return {
            "component_bounds": self.component_bounds,
            "seed_components": self.seed_components,
            "resolution_bounds": self.resolution_bounds,
            "resolution_seed": self.resolution_seed,
        }


def _integer(value: int, name: str, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < int(positive):
        qualifier = "strictly positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _sobol_sampler(
    *,
    seed: int,
    topology_id: int,
    d_present: tuple[bool, ...],
    resolution_present: bool,
) -> qmc.Sobol:
    branch_bits = sum(int(present) << index for index, present in enumerate(d_present))
    derived_seed = int(
        np.random.SeedSequence(
            [seed, topology_id, branch_bits, int(resolution_present), 0x5638]
        ).generate_state(1, dtype=np.uint32)[0]
    )
    return qmc.Sobol(d=UNIT_CUBE_DIMENSIONS, scramble=True, seed=derived_seed)


def _sobol_points(
    count: int,
    *,
    seed: int,
    topology_id: int,
    d_present: tuple[bool, ...],
    resolution_present: bool,
) -> np.ndarray:
    sampler = _sobol_sampler(
        seed=seed,
        topology_id=topology_id,
        d_present=d_present,
        resolution_present=resolution_present,
    )
    return sampler.random_base2((count - 1).bit_length())[:count]


def _sobol_point_at(
    sequence_index: int,
    *,
    seed: int,
    topology_id: int,
    d_present: tuple[bool, ...],
    resolution_present: bool,
) -> np.ndarray:
    sampler = _sobol_sampler(
        seed=seed,
        topology_id=topology_id,
        d_present=d_present,
        resolution_present=resolution_present,
    )
    if sequence_index:
        sampler.fast_forward(sequence_index)
    return sampler.random(1)[0]


def _seed_from_point(
    codec: ProfiledBranchCodec,
    point: Sequence[float],
    *,
    sequence_index: int,
    seed: int,
) -> ProfiledBranchSeed:
    coordinates = codec.canonical_coordinates(point)
    components, resolution = codec.decode(coordinates)
    # Fixed ranges have a unique inverse independent of arbitrary Sobol bits.
    coordinates = codec.encode(components, resolution)
    return ProfiledBranchSeed(
        topology_id=codec.topology_id,
        topology=codec.topology,
        d_present=codec.d_present,
        component_bounds=codec.component_bounds,
        seed_components=components,
        resolution_bounds=codec.resolution_bounds,
        resolution_seed=resolution,
        unit_cube=coordinates.unit_cube,
        active_mask=coordinates.active_mask,
        sequence_index=sequence_index,
        generator_seed=seed,
    )


def generate_profiled_branch_seed_at_index(
    codec: ProfiledBranchCodec,
    *,
    seed: int,
    sequence_index: int,
) -> ProfiledBranchSeed:
    """Materialize exactly one indexed point from a branch's stable Sobol stream."""

    if not isinstance(codec, ProfiledBranchCodec):
        raise TypeError("codec must be a ProfiledBranchCodec")
    seed = _integer(seed, "seed")
    sequence_index = _integer(sequence_index, "sequence_index")
    point = _sobol_point_at(
        sequence_index,
        seed=seed,
        topology_id=codec.topology_id,
        d_present=codec.d_present,
        resolution_present=codec.resolution_bounds is not None,
    )
    return _seed_from_point(
        codec,
        point,
        sequence_index=sequence_index,
        seed=seed,
    )


def generate_profiled_branch_seeds(
    topology: Sequence[str],
    component_bounds: Sequence[GuiComponentBounds],
    d_present: bool | Sequence[bool],
    *,
    resolution_bounds: ResolutionBounds | None = None,
    seed: int,
    count: int,
) -> tuple[ProfiledBranchSeed, ...]:
    """Generate deterministic 26D Sobol seeds for one fixed discrete branch.

    The shared :class:`ProfiledBranchCodec` performs all physical decoding.
    Inactive coordinates use a canonical value of 0.5 and are accompanied by
    the fixed-width mask stored on every returned seed.
    """

    seed = _integer(seed, "seed")
    count = _integer(count, "count", positive=True)
    codec = ProfiledBranchCodec.build(
        topology,
        component_bounds,
        d_present,
        resolution_bounds=resolution_bounds,
    )
    points = _sobol_points(
        count,
        seed=seed,
        topology_id=codec.topology_id,
        d_present=codec.d_present,
        resolution_present=resolution_bounds is not None,
    )

    return tuple(
        _seed_from_point(
            codec,
            point,
            sequence_index=sequence_index,
            seed=seed,
        )
        for sequence_index, point in enumerate(points)
    )


__all__ = [
    "PROPOSAL_SAMPLING_VERSION",
    "ProfiledBranchSeed",
    "generate_profiled_branch_seed_at_index",
    "generate_profiled_branch_seeds",
]
