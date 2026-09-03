"""Build the single-view compatibility calibration from the reserved split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .compatibility_calibration import (
    COMPATIBILITY_STRATUM_FIELDS,
    COMPATIBILITY_STRATUM_VERSION,
    DESIGN_STRATUM_UNIVERSE_FIELDS,
    DESIGN_STRATUM_UNIVERSE_SEMANTICS,
    DESIGN_STRATUM_UNIVERSE_SHA256,
    DESIGN_STRATUM_UNIVERSE_VERSION,
    CompatibilityCalibrationArtifact,
    CompatibilityCalibrationSample,
    CompatibilityStratum,
    fit_compatibility_calibration,
    write_compatibility_calibration_atomic,
)
from .dataset import (
    SPLIT_CODE,
    NumpyShard,
    PilotDatasetConfig,
    calibration_stratum_values,
    load_shard,
    reconstruct_recipe,
)
from .evaluation import natural_log_rmse
from .observation_v5 import v5_acquisition_policy_id
from .simulation import (
    sample_observation_view,
    simulate_recipe,
)
from .uncertainty_provenance_v5 import V5UncertaintyProvenance


CALIBRATION_RUNNER_VERSION = "posterior_v8_reserved_split_single_view_calibration_runner_v5"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_collection(
    shard_paths: Sequence[str | Path],
) -> tuple[tuple[NumpyShard, ...], str]:
    paths = tuple(sorted((Path(value) for value in shard_paths), key=lambda value: value.name))
    if not paths:
        raise ValueError("at least one dataset shard is required")
    if len({value.name for value in paths}) != len(paths):
        raise ValueError("dataset shard basenames must be unique")

    shards = tuple(load_shard(path) for path in paths)
    fingerprint = None
    entries = []
    seen_recipe_views: set[tuple[int, int]] = set()
    for shard in shards:
        current = (
            json.dumps(shard.metadata["config"], sort_keys=True),
            shard.metadata["source_sha256_aggregate"],
            shard.metadata["dataset_schema_version"],
        )
        if fingerprint is None:
            fingerprint = current
        elif fingerprint != current:
            raise ValueError("shards do not share one dataset config/source contract")
        metadata_path = shard.path.with_suffix(".json")
        entries.append(
            {
                "npz_file": shard.path.name,
                "npz_sha256": shard.metadata["npz_sha256"],
                "metadata_file": metadata_path.name,
                "metadata_sha256": _sha256_file(metadata_path),
                "start_index": shard.metadata["shard"]["start_index"],
                "stop_index_exclusive": shard.metadata["shard"]["stop_index_exclusive"],
            }
        )
        for recipe_index, view_index in zip(
            shard.arrays["recipe_index"], shard.arrays["view_index"]
        ):
            key = (int(recipe_index), int(view_index))
            if key in seen_recipe_views:
                raise ValueError("duplicate recipe/view identity across calibration shards")
            seen_recipe_views.add(key)
    return shards, _sha256_json({"runner_version": CALIBRATION_RUNNER_VERSION, "shards": entries})


def calibration_samples_from_shards(
    shards: Sequence[NumpyShard],
) -> tuple[tuple[CompatibilityCalibrationSample, ...], str]:
    """Reconstruct exact truth scores from only the reserved calibration split."""

    values = tuple(shards)
    if not values or not all(isinstance(value, NumpyShard) for value in values):
        raise TypeError("shards must contain NumpyShard values")
    samples = []
    split_identities = []
    seen_recipe_strata: set[tuple[str, CompatibilityStratum]] = set()
    for shard in values:
        config = PilotDatasetConfig(**shard.metadata["config"])
        selected_rows = np.flatnonzero(shard.arrays["assigned_split"] == SPLIT_CODE["calibration"])
        for row_value in selected_rows:
            row = int(row_value)
            recipe_index = int(shard.arrays["recipe_index"][row])
            view_index = int(shard.arrays["view_index"][row])
            group_id = f"{config.master_seed}:{recipe_index}"
            stratum = CompatibilityStratum(**calibration_stratum_values(shard.arrays, row))
            recipe_stratum_key = (group_id, stratum)
            if recipe_stratum_key in seen_recipe_strata:
                raise ValueError(
                    "duplicate clean-recipe/acquisition-stratum observation is forbidden "
                    "for the primary single-view calibration estimand"
                )
            seen_recipe_strata.add(recipe_stratum_key)
            recipe = reconstruct_recipe(
                recipe_seed=int(shard.arrays["recipe_seed"][row]),
                topology_id=int(shard.arrays["topology_id"][row]),
                config=config,
            )
            view = sample_observation_view(
                recipe.seed, view_index, max_points=config.max_raw_points
            )
            simulated = simulate_recipe(view.simulation_recipe(recipe))
            selected = (
                view.selection_mask(simulated.q)
                & (simulated.q >= view.preprocess_q_range[0])
                & (simulated.q <= view.preprocess_q_range[1])
            )
            effective_valid_point_count = int(np.count_nonzero(selected))
            if effective_valid_point_count < 16:  # pragma: no cover - view contract
                raise RuntimeError("calibration observation has too few selected points")
            if (
                simulated.sigma is None
                or not np.all(np.isfinite(simulated.sigma[selected]))
                or np.any(simulated.sigma[selected] <= 0.0)
            ):  # pragma: no cover - SimulatedCurve contract
                raise ValueError("measurement sigma is required for calibration")
            sigma_log = simulated.sigma[selected] / simulated.intensity[selected]
            score = natural_log_rmse(
                simulated.clean_intensity[selected],
                simulated.intensity[selected],
                sigma_log=sigma_log,
            )
            sample_id = f"{config.master_seed}:{recipe_index}:{view_index}"
            acquisition_policy_id = v5_acquisition_policy_id(
                view,
                V5UncertaintyProvenance("simulated_sigma"),
            )
            samples.append(
                CompatibilityCalibrationSample(
                    sample_id=sample_id,
                    independent_group_id=group_id,
                    stratum=stratum,
                    score=score,
                    effective_valid_point_count=effective_valid_point_count,
                    acquisition_policy_id=acquisition_policy_id,
                    measurement_sigma_available=True,
                )
            )
            split_identities.append(
                {
                    "sample_id": sample_id,
                    "independent_group_id": group_id,
                    "physical_cell_id": shard.arrays["physical_cell_id"][row].decode("ascii"),
                    "stratum": {
                        "point_count": stratum.point_count,
                        "noise_id": stratum.noise_id,
                        "q_window_id": stratum.q_window_id,
                    },
                    "effective_valid_point_count": effective_valid_point_count,
                    "acquisition_policy_id": acquisition_policy_id,
                    "measurement_sigma_available": True,
                }
            )
    if not samples:
        raise ValueError("dataset collection contains no reserved calibration rows")
    samples.sort(key=lambda value: value.sample_id)
    split_identities.sort(key=lambda value: value["sample_id"])
    return tuple(samples), _sha256_json({"split": "calibration", "samples": split_identities})


def build_dataset_calibration(
    shard_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    target_coverage: float = 0.95,
    minimum_samples_per_stratum: int = 20,
) -> CompatibilityCalibrationArtifact:
    """Fit and atomically publish one immutable calibration artifact."""

    shards, dataset_digest = _load_collection(shard_paths)
    samples, split_digest = calibration_samples_from_shards(shards)
    artifact = fit_compatibility_calibration(
        samples,
        dataset_manifest_sha256=dataset_digest,
        calibration_split_sha256=split_digest,
        target_coverage=target_coverage,
        minimum_samples_per_stratum=minimum_samples_per_stratum,
    )
    write_compatibility_calibration_atomic(output_path, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-coverage", type=float, default=0.95)
    parser.add_argument("--minimum-samples-per-stratum", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    artifact = build_dataset_calibration(
        args.shard,
        args.output,
        target_coverage=args.target_coverage,
        minimum_samples_per_stratum=args.minimum_samples_per_stratum,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "input_sha256": artifact.input_sha256,
                "artifact_sha256": artifact.sha256,
                "calibration_split_id": artifact.calibration_split_id,
                "calibration_split_sha256": artifact.calibration_split_sha256,
                "calibration_groups": artifact.input_summary.independent_group_count,
                "recipe_stratum_observations": (artifact.input_summary.recipe_stratum_count),
                "strata": artifact.input_summary.stratum_count,
                "preregistered_design_strata": (artifact.preregistered_design_stratum_count),
                "design_stratum_universe_version": DESIGN_STRATUM_UNIVERSE_VERSION,
                "design_stratum_universe_fields": list(DESIGN_STRATUM_UNIVERSE_FIELDS),
                "design_stratum_universe_semantics": DESIGN_STRATUM_UNIVERSE_SEMANTICS,
                "design_stratum_universe_sha256": DESIGN_STRATUM_UNIVERSE_SHA256,
                "effective_valid_point_count_range": [
                    artifact.input_summary.effective_valid_point_count_min,
                    artifact.input_summary.effective_valid_point_count_max,
                ],
                "acquisition_policy_count": (artifact.input_summary.acquisition_policy_count),
                "observation_estimand": artifact.observation_estimand,
                "compatibility_stratum_version": COMPATIBILITY_STRATUM_VERSION,
                "compatibility_stratum_fields": list(COMPATIBILITY_STRATUM_FIELDS),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a module on Slurm
    raise SystemExit(main())


__all__ = [
    "CALIBRATION_RUNNER_VERSION",
    "build_dataset_calibration",
    "calibration_samples_from_shards",
    "main",
]
