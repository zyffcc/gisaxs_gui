"""Command-line entry point for one Posterior V8 Phase-2 candidate shard.

Submit one invocation per Slurm array element.  ``--start-index`` and
``--recipe-count`` describe a deterministic half-open clean-recipe range.
Physical cells assign all resulting views to one versioned 75/10/5/10 split;
reruns never overwrite.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence
import zipfile

import numpy as np

from .branch_catalog import BRANCH_CATALOG_VERSION
from .branch_codec import BRANCH_CODEC_VERSION
from .contract import CODEC_VERSION, CONTRACT_VERSION, FORWARD_MODEL_VERSION

from .dataset import (
    ARRAY_ORDER,
    DATASET_GENERATOR_VERSION,
    DATASET_SCHEMA_VERSION,
    PHYSICAL_CELL_VERSION,
    PILOT_LIMITATIONS,
    PILOT_PHASE,
    RANGE_GENERATOR_VERSION,
    SEED_SCHEME_VERSION,
    SPLIT_POLICY_VERSION,
    TOPOLOGY_SCHEDULES,
    PilotDatasetConfig,
    ShardSpec,
    _array_schema,
    _canonical_json,
    _file_sha256,
    _observation_policy,
    _range_policy,
    _split_policy,
    generate_shard_arrays,
)
from .preprocessing import DEFAULT_CONTRACT, PREPROCESSING_VERSION
from .simulation import (
    OBSERVATION_STRATUM_VERSION,
    OBSERVATION_VIEW_VERSION,
    SIMULATION_VERSION,
)


@dataclass(frozen=True)
class ShardArtifact:
    npz_path: Path
    metadata_path: Path
    sample_count: int
    recipe_count: int
    npz_sha256: str


def _repository_root(module_file: Path | None = None) -> Path:
    """Locate a source snapshot root without requiring a ``.git`` directory."""

    start = Path(__file__) if module_file is None else Path(module_file)
    for candidate in start.resolve().parents:
        if (candidate / "src").is_dir() and (candidate / "utils").is_dir():
            return candidate
    raise RuntimeError("could not locate a source root containing both src/ and utils/")


def _source_hashes() -> dict[str, str]:
    root = _repository_root()
    relative_paths = (
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/dataset.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/build_dataset.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/contract.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/branch_catalog.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/branch_codec.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/compatibility_calibration.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/preprocessing.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/simulation.py",
        "src/gimap/features/fitting/domain/scattering_model.py",
        "src/gimap/features/fitting/domain/physical_constraints.py",
    )
    return {name: _file_sha256(root / name) for name in relative_paths}


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in ARRAY_ORDER:
            payload = BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                payload.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return output.getvalue()


def _publish_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _metadata(
    config: PilotDatasetConfig,
    spec: ShardSpec,
    arrays: Mapping[str, np.ndarray],
    file_name: str,
    npz_digest: str,
) -> dict[str, object]:
    sources = _source_hashes()
    topology_values, topology_counts = np.unique(arrays["topology_id"], return_counts=True)
    pattern_values, pattern_counts = np.unique(arrays["branch_pattern_id"], return_counts=True)
    split_values, split_counts = np.unique(arrays["assigned_split"], return_counts=True)
    range_values, range_counts = np.unique(arrays["range_regime"], return_counts=True)
    grid_values, grid_counts = np.unique(arrays["grid_kind"], return_counts=True)
    categorical_histograms = {}
    for field in ("q_window_id", "noise_id", "mask_id", "crop_id"):
        keys, counts = np.unique(arrays[field], return_counts=True)
        categorical_histograms[field] = {
            str(int(key)): int(value) for key, value in zip(keys, counts)
        }
    return {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "dataset_generator_version": DATASET_GENERATOR_VERSION,
        "phase": PILOT_PHASE,
        "versions": {
            "contract": CONTRACT_VERSION,
            "parameter_codec": CODEC_VERSION,
            "forward_model": FORWARD_MODEL_VERSION,
            "branch_catalog": BRANCH_CATALOG_VERSION,
            "branch_codec": BRANCH_CODEC_VERSION,
            "simulation": SIMULATION_VERSION,
            "observation_view": OBSERVATION_VIEW_VERSION,
            "observation_stratum": OBSERVATION_STRATUM_VERSION,
            "preprocessing": PREPROCESSING_VERSION,
            "seed_scheme": SEED_SCHEME_VERSION,
            "physical_cell": PHYSICAL_CELL_VERSION,
            "split_policy": SPLIT_POLICY_VERSION,
            "range_generator": RANGE_GENERATOR_VERSION,
        },
        "config": asdict(config),
        "preprocessing_contract": asdict(DEFAULT_CONTRACT),
        "observation_generation": _observation_policy(config),
        "split_assignment": _split_policy(),
        "range_generation": _range_policy(),
        "pilot_limitations": PILOT_LIMITATIONS,
        "shard": {
            "shard_index": spec.shard_index,
            "start_index": spec.start_index,
            "stop_index_exclusive": spec.stop_index,
            "recipe_count": spec.recipe_count,
            "row_count": arrays["topology_id"].shape[0],
            "file_name": file_name,
        },
        "array_schema": _array_schema(arrays),
        "recipe_seed_range": {
            "minimum": int(np.min(arrays["recipe_seed"])),
            "maximum": int(np.max(arrays["recipe_seed"])),
        },
        "topology_histogram": {
            str(int(key)): int(value) for key, value in zip(topology_values, topology_counts)
        },
        "branch_pattern_histogram": {
            str(int(key)): int(value) for key, value in zip(pattern_values, pattern_counts)
        },
        "split_histogram": {
            str(int(key)): int(value) for key, value in zip(split_values, split_counts)
        },
        "range_regime_histogram": {
            str(int(key)): int(value) for key, value in zip(range_values, range_counts)
        },
        "grid_kind_histogram": {
            str(int(key)): int(value) for key, value in zip(grid_values, grid_counts)
        },
        "q_window_histogram": categorical_histograms["q_window_id"],
        "noise_histogram": categorical_histograms["noise_id"],
        "mask_histogram": categorical_histograms["mask_id"],
        "crop_histogram": categorical_histograms["crop_id"],
        "physical_cell_count": int(np.unique(arrays["physical_cell_id"]).size),
        "source_sha256": sources,
        "source_sha256_aggregate": sha256(_canonical_json(sources)).hexdigest(),
        "npz_sha256": npz_digest,
    }


def build_shard(
    output_dir: str | os.PathLike[str],
    config: PilotDatasetConfig,
    spec: ShardSpec,
) -> ShardArtifact:
    """Build and atomically publish one immutable shard plus JSON sidecar."""

    if not isinstance(config, PilotDatasetConfig) or not isinstance(spec, ShardSpec):
        raise TypeError("config and spec must be PilotDatasetConfig and ShardSpec")
    output = Path(output_dir)
    stem = (
        f"mixed-{config.topology_schedule}-v{config.views_per_recipe}-"
        f"{spec.shard_index:05d}-{spec.start_index:012d}-{spec.stop_index:012d}"
    )
    npz_path = output / f"{stem}.npz"
    metadata_path = output / f"{stem}.json"
    if npz_path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite existing shard {stem!r}")
    arrays = generate_shard_arrays(config, spec)
    npz_payload = _deterministic_npz_bytes(arrays)
    npz_digest = sha256(npz_payload).hexdigest()
    metadata = _metadata(config, spec, arrays, npz_path.name, npz_digest)
    _publish_exclusive(npz_path, npz_payload)
    try:
        _publish_exclusive(metadata_path, _canonical_json(metadata))
    except Exception:
        npz_path.unlink(missing_ok=True)
        raise
    return ShardArtifact(
        npz_path,
        metadata_path,
        arrays["topology_id"].shape[0],
        spec.recipe_count,
        npz_digest,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--start-index", required=True, type=int)
    parser.add_argument("--recipe-count", required=True, type=int)
    parser.add_argument("--master-seed", type=int, default=20260902)
    parser.add_argument("--topology-schedule", choices=tuple(TOPOLOGY_SCHEDULES), default="all34")
    parser.add_argument("--views-per-recipe", type=int, default=2)
    parser.add_argument("--max-raw-points", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = PilotDatasetConfig(
        master_seed=arguments.master_seed,
        topology_schedule=arguments.topology_schedule,
        views_per_recipe=arguments.views_per_recipe,
        max_raw_points=arguments.max_raw_points,
    )
    spec = ShardSpec(
        shard_index=arguments.shard_index,
        start_index=arguments.start_index,
        recipe_count=arguments.recipe_count,
    )
    artifact = build_shard(arguments.output_dir, config, spec)
    print(
        json.dumps(
            {
                "npz_path": str(artifact.npz_path),
                "metadata_path": str(artifact.metadata_path),
                "sample_count": artifact.sample_count,
                "recipe_count": artifact.recipe_count,
                "npz_sha256": artifact.npz_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a module by Slurm
    raise SystemExit(main())


__all__ = ["ShardArtifact", "build_parser", "build_shard", "main"]
