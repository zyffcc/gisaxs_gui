"""Versioned Sobol index blocks for the V5 paper dataset.

The design assigns a clean physical recipe before any observation view is
created.  Main splits occupy contiguous, mutually exclusive Sobol-index
blocks separated by unused guard bands.  The OOD block is further partitioned
into four contiguous, explicitly labelled one-factor challenge blocks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np


V5_SPLIT_PLAN_SCHEMA = "gisaxs.posterior_v8.sobol_split_plan/v1"
V5_SPLIT_PLAN_VERSION = "posterior_v8_clean_recipe_contiguous_guarded_sobol_blocks_v1"

MAIN_SPLITS = (
    "train",
    "tuning_validation",
    "calibration",
    "test",
    "reference",
    "ood",
)
OOD_LABELS = (
    "topology_holdout",
    "range_width_holdout",
    "weak_component_holdout",
    "acquisition_policy_holdout",
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
                raise ValueError(f"duplicate JSON field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(encoded, object_pairs_hook=reject_duplicates)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid split-design JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("split-design JSON must contain one object")
    return value


@dataclass(frozen=True, kw_only=True)
class V5SplitCounts:
    train: int
    tuning_validation: int
    calibration: int
    test: int
    reference: int
    ood_topology: int
    ood_range_width: int
    ood_weak_component: int
    ood_acquisition_policy: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            object.__setattr__(self, name, _integer(value, name, minimum=1))

    @property
    def ood(self) -> int:
        return (
            self.ood_topology
            + self.ood_range_width
            + self.ood_weak_component
            + self.ood_acquisition_policy
        )

    def main_counts(self) -> dict[str, int]:
        return {
            "train": self.train,
            "tuning_validation": self.tuning_validation,
            "calibration": self.calibration,
            "test": self.test,
            "reference": self.reference,
            "ood": self.ood,
        }

    def ood_counts(self) -> dict[str, int]:
        return {
            "topology_holdout": self.ood_topology,
            "range_width_holdout": self.ood_range_width,
            "weak_component_holdout": self.ood_weak_component,
            "acquisition_policy_holdout": self.ood_acquisition_policy,
        }


@dataclass(frozen=True, order=True)
class V5IndexBlock:
    name: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("block name must be non-empty")
        start = _integer(self.start, "start")
        stop = _integer(self.stop, "stop", minimum=1)
        if stop <= start:
            raise ValueError("block stop must be greater than start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)

    @property
    def count(self) -> int:
        return self.stop - self.start

    def contains(self, index: int) -> bool:
        value = _integer(index, "index")
        return self.start <= value < self.stop


def _expected_blocks(
    counts: V5SplitCounts,
    *,
    start_index: int,
    guard_band: int,
) -> tuple[
    tuple[V5IndexBlock, ...],
    tuple[V5IndexBlock, ...],
    tuple[V5IndexBlock, ...],
]:
    cursor = start_index
    blocks = []
    guards = []
    for offset, (name, count) in enumerate(counts.main_counts().items()):
        blocks.append(V5IndexBlock(name, cursor, cursor + count))
        cursor += count
        if offset != len(MAIN_SPLITS) - 1:
            guards.append(V5IndexBlock(f"guard_after_{name}", cursor, cursor + guard_band))
            cursor += guard_band
    ood_parent = blocks[-1]
    ood_cursor = ood_parent.start
    ood_blocks = []
    for label, count in counts.ood_counts().items():
        ood_blocks.append(V5IndexBlock(label, ood_cursor, ood_cursor + count))
        ood_cursor += count
    if ood_cursor != ood_parent.stop:  # pragma: no cover - construction invariant
        raise RuntimeError("OOD subblocks do not partition the OOD block")
    return tuple(blocks), tuple(guards), tuple(ood_blocks)


def _plan_payload(
    counts: V5SplitCounts,
    *,
    start_index: int,
    guard_band: int,
    blocks: Sequence[V5IndexBlock],
    guard_blocks: Sequence[V5IndexBlock],
    ood_blocks: Sequence[V5IndexBlock],
) -> dict[str, object]:
    return {
        "schema": V5_SPLIT_PLAN_SCHEMA,
        "version": V5_SPLIT_PLAN_VERSION,
        "start_index": start_index,
        "guard_band": guard_band,
        "counts": asdict(counts),
        "blocks": [asdict(value) for value in blocks],
        "guard_blocks": [asdict(value) for value in guard_blocks],
        "ood_blocks": [asdict(value) for value in ood_blocks],
        "statistical_unit": "independent_clean_physical_recipe",
        "view_assignment": "all_views_follow_clean_parent_group",
    }


@dataclass(frozen=True)
class V5SplitPlan:
    counts: V5SplitCounts
    start_index: int
    guard_band: int
    blocks: tuple[V5IndexBlock, ...]
    guard_blocks: tuple[V5IndexBlock, ...]
    ood_blocks: tuple[V5IndexBlock, ...]
    canonical_json: str
    sha256: str

    @classmethod
    def create(
        cls,
        counts: V5SplitCounts,
        *,
        start_index: int = 0,
        guard_band: int = 1024,
    ) -> "V5SplitPlan":
        if not isinstance(counts, V5SplitCounts):
            raise TypeError("counts must be V5SplitCounts")
        start = _integer(start_index, "start_index")
        guard = _integer(guard_band, "guard_band", minimum=1)
        blocks, guards, ood_blocks = _expected_blocks(counts, start_index=start, guard_band=guard)
        core = _plan_payload(
            counts,
            start_index=start,
            guard_band=guard,
            blocks=blocks,
            guard_blocks=guards,
            ood_blocks=ood_blocks,
        )
        canonical = _canonical_json(core)
        return cls(
            counts,
            start,
            guard,
            blocks,
            guards,
            ood_blocks,
            canonical,
            sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.counts, V5SplitCounts):
            raise TypeError("counts must be V5SplitCounts")
        start = _integer(self.start_index, "start_index")
        guard = _integer(self.guard_band, "guard_band", minimum=1)
        expected = _expected_blocks(self.counts, start_index=start, guard_band=guard)
        if (self.blocks, self.guard_blocks, self.ood_blocks) != expected:
            raise ValueError("split plan blocks do not reproduce")
        canonical = _canonical_json(
            _plan_payload(
                self.counts,
                start_index=start,
                guard_band=guard,
                blocks=self.blocks,
                guard_blocks=self.guard_blocks,
                ood_blocks=self.ood_blocks,
            )
        )
        digest = sha256(canonical.encode("utf-8")).hexdigest()
        if self.canonical_json != canonical or self.sha256 != digest:
            raise ValueError("split plan does not reproduce its audit hash")

    @property
    def prefix_stop(self) -> int:
        return self.blocks[-1].stop

    @property
    def assigned_count(self) -> int:
        return sum(value.count for value in self.blocks)

    def split_for_index(self, index: int) -> str:
        for block in self.blocks:
            if block.contains(index):
                return block.name
        raise ValueError("index is outside assigned blocks or lies in a guard band")

    def ood_label_for_index(self, index: int) -> str | None:
        split = self.split_for_index(index)
        if split != "ood":
            return None
        for block in self.ood_blocks:
            if block.contains(index):
                return block.name
        raise RuntimeError("OOD index is not covered by an OOD subblock")  # pragma: no cover

    def assigned_indices(self) -> tuple[int, ...]:
        return tuple(index for block in self.blocks for index in range(block.start, block.stop))

    def to_json(self) -> str:
        payload = json.loads(self.canonical_json)
        payload["plan_sha256"] = self.sha256
        return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, encoded: str) -> "V5SplitPlan":
        payload = _strict_json_object(encoded)
        if payload.get("schema") != V5_SPLIT_PLAN_SCHEMA:
            raise ValueError("unsupported V5 split-plan schema")
        expected_fields = {
            "schema",
            "version",
            "start_index",
            "guard_band",
            "counts",
            "blocks",
            "guard_blocks",
            "ood_blocks",
            "statistical_unit",
            "view_assignment",
            "plan_sha256",
        }
        if set(payload) != expected_fields:
            raise ValueError("V5 split-plan fields are incomplete or unsupported")
        try:
            raw_counts = payload["counts"]
            if not isinstance(raw_counts, Mapping):
                raise TypeError("counts must be an object")
            counts = V5SplitCounts(**raw_counts)
            replay = cls.create(
                counts,
                start_index=payload["start_index"],
                guard_band=payload["guard_band"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid V5 split-plan payload") from exc
        supplied_hash = payload.pop("plan_sha256")
        if payload != json.loads(replay.canonical_json) or supplied_hash != replay.sha256:
            raise ValueError("V5 split-plan payload/hash does not reproduce")
        return replay


__all__ = [
    "MAIN_SPLITS",
    "OOD_LABELS",
    "V5_SPLIT_PLAN_SCHEMA",
    "V5_SPLIT_PLAN_VERSION",
    "V5IndexBlock",
    "V5SplitCounts",
    "V5SplitPlan",
]
