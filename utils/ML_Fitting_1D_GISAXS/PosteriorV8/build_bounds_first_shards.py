"""Build or audit immutable Posterior V8 bounds-first solution shards."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Mapping, Sequence
import zipfile

import numpy as np

from .bounds_first_contract import (
    BOUNDS_EMBEDDING_DIM,
    BOUNDS_EMBEDDING_VERSION,
    BOUNDS_FIRST_SCHEMA_VERSION,
    LOCAL_TARGET_SEMANTICS,
)
from .bounds_first_dataset import (
    BOUNDS_GENERATOR_VERSION,
    LOCAL_TARGET_OPEN_EPSILON,
    LOCAL_TARGET_SAMPLING_VERSION,
)
from .bounds_first_schedule import (
    BOUNDS_BRANCH_SCHEDULE_VERSION,
    SCHEDULE_SEMANTICS_SHA256,
    TOPOLOGY_SCHEDULES,
    schedule_policy,
)
from .bounds_first_shards import (
    ARRAY_ORDER,
    PILOT_LIMITATIONS,
    SEED_SCHEME_VERSION,
    SHARD_GENERATOR_VERSION,
    SHARD_SCHEMA_VERSION,
    SOLUTION_ONLY_PHASE,
    SPLIT_NAMES,
    SPLIT_POLICY_VERSION,
    BoundsFirstShardConfig,
    BoundsFirstShardSpec,
    array_schema,
    generate_shard_arrays,
    split_policy,
    validate_shard_semantics,
)
from .branch_catalog import BRANCH_CATALOG_VERSION
from .branch_codec import BRANCH_CODEC_VERSION
from .canonical_branch_catalog import CANONICAL_BRANCH_CATALOG_VERSION
from .canonical_component_slots import CANONICAL_COMPONENT_SLOTS_VERSION
from .contract import CODEC_VERSION, CONTRACT_VERSION, FORWARD_MODEL_VERSION
from .preprocessing import DEFAULT_CONTRACT, PREPROCESSING_VERSION
from .simulation import (
    OBSERVATION_CROP_PROFILES,
    OBSERVATION_GRID_KINDS,
    OBSERVATION_KEEP_PROBABILITIES,
    OBSERVATION_NOISE_PROFILES,
    OBSERVATION_Q_WINDOWS,
    OBSERVATION_STRATUM_VERSION,
    OBSERVATION_VIEW_VERSION,
    SIMULATION_VERSION,
)


DEFAULT_MAXWELL_OUTPUT_ROOT = Path(
    "/data/dust/user/zhaiyufe/MaxwellRuns/GISAXS_POSTERIOR_V8_20260902/"
    "datasets/posterior_v8_bounds_first_v4"
)
MERGE_AUDIT_SCHEMA_VERSION = "gisaxs.posterior_v8.bounds_first_merge_audit/v1"
_SOURCE_RELATIVE_PATHS = (
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/bounds_first_contract.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/bounds_first_dataset.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/bounds_first_schedule.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/bounds_first_shards.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/build_bounds_first_shards.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/branch_catalog.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/branch_codec.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/canonical_branch_catalog.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/canonical_component_slots.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/contract.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/preprocessing.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/profiled_forward.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/simulation.py",
    "src/gimap/features/fitting/domain/scattering_model.py",
    "src/gimap/features/fitting/domain/physical_constraints.py",
)


@dataclass(frozen=True)
class BoundsFirstShardArtifact:
    npz_path: Path
    metadata_path: Path
    recipe_count: int
    row_count: int
    npz_sha256: str


@dataclass(frozen=True)
class BoundsFirstNumpyShard:
    path: Path
    metadata: Mapping[str, object]
    arrays: Mapping[str, np.ndarray]

    @property
    def recipe_count(self) -> int:
        return int(self.metadata["shard"]["recipe_count"])

    @property
    def row_count(self) -> int:
        return int(self.arrays["global_recipe_index"].size)


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_root(module_file: Path | None = None) -> Path:
    start = Path(__file__) if module_file is None else Path(module_file)
    for candidate in start.resolve().parents:
        if (candidate / "src").is_dir() and (candidate / "utils").is_dir():
            return candidate
    raise RuntimeError("could not locate source root containing src/ and utils/")


def source_hashes() -> dict[str, str]:
    root = _repository_root()
    return {name: file_sha256(root / name) for name in _SOURCE_RELATIVE_PATHS}


def _versions() -> dict[str, str]:
    return {
        "contract": CONTRACT_VERSION,
        "parameter_codec": CODEC_VERSION,
        "forward_model": FORWARD_MODEL_VERSION,
        "branch_catalog": BRANCH_CATALOG_VERSION,
        "canonical_branch_catalog": CANONICAL_BRANCH_CATALOG_VERSION,
        "canonical_component_slots": CANONICAL_COMPONENT_SLOTS_VERSION,
        "branch_codec": BRANCH_CODEC_VERSION,
        "bounds_first_contract": BOUNDS_FIRST_SCHEMA_VERSION,
        "bounds_embedding": BOUNDS_EMBEDDING_VERSION,
        "bounds_generator": BOUNDS_GENERATOR_VERSION,
        "local_target_semantics": LOCAL_TARGET_SEMANTICS,
        "local_target_sampling": LOCAL_TARGET_SAMPLING_VERSION,
        "simulation": SIMULATION_VERSION,
        "observation_view": OBSERVATION_VIEW_VERSION,
        "observation_stratum": OBSERVATION_STRATUM_VERSION,
        "preprocessing": PREPROCESSING_VERSION,
        "seed_scheme": SEED_SCHEME_VERSION,
        "split_policy": SPLIT_POLICY_VERSION,
        "bounds_branch_schedule": BOUNDS_BRANCH_SCHEDULE_VERSION,
        "bounds_branch_schedule_sha256": SCHEDULE_SEMANTICS_SHA256,
    }


def _observation_policy(config: BoundsFirstShardConfig) -> dict[str, object]:
    return {
        "views_per_clean_recipe": config.views_per_recipe,
        "max_raw_points": config.max_raw_points,
        "grid_kinds": list(OBSERVATION_GRID_KINDS),
        "q_windows": [list(value) for value in OBSERVATION_Q_WINDOWS],
        "noise_profiles": [list(value) for value in OBSERVATION_NOISE_PROFILES],
        "point_keep_probabilities": list(OBSERVATION_KEEP_PROBABILITIES),
        "crop_profiles": [list(value) for value in OBSERVATION_CROP_PROFILES],
        "policy_inputs": ["recipe_seed", "view_index", "max_raw_points"],
        "unknown_physical_parameter_access": False,
        "all_views_share_bounds_and_clean_physics": True,
    }


def _bounds_policy() -> dict[str, object]:
    return {
        "generation_order": [
            "sample_gui_physical_bounds",
            "build_user_bounds_profiled_branch_codec",
            "sample_local_unit_truth",
            "decode_and_store_gui_physical_truth",
            "sample_observation_views",
        ],
        "target_space": "bounds_specific_local_unit_coordinate",
        "target_sampling": {
            "version": LOCAL_TARGET_SAMPLING_VERSION,
            "distribution": "uniform_on_open_local_unit_interval",
            "epsilon": LOCAL_TARGET_OPEN_EPSILON,
            "closed_boundary_atoms_in_training": False,
            "closed_boundary_optimizer_stress_set_required": True,
        },
        "global_reference_space": "full_domain_branch_codec_unit_coordinate",
        "bounds_embedding_dimension": BOUNDS_EMBEDDING_DIM,
        "bounds_embedding_semantics": "normalized_gui_physical_low_high_presence",
        "simple_global_26d_box_is_gui_bounds": False,
        "truth_conditioned_bounds_generation": False,
        "task_population": "in_domain_solution_only",
        "negative_or_ood_truth_fabrication_allowed": False,
        "branch_bounds_schedule": schedule_policy(),
        "branch_bounds_schedule_sha256": SCHEDULE_SEMANTICS_SHA256,
    }


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
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


def publish_exclusive(path: Path, payload: bytes) -> None:
    """Atomically hard-link one synced temp file; an existing target always wins."""

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


def _histogram(values: np.ndarray) -> dict[str, int]:
    keys, counts = np.unique(values, return_counts=True)
    return {str(int(key)): int(value) for key, value in zip(keys, counts)}


def _identity_payload(
    config: BoundsFirstShardConfig, source_aggregate: str
) -> dict[str, object]:
    return {
        "dataset_schema_version": SHARD_SCHEMA_VERSION,
        "dataset_generator_version": SHARD_GENERATOR_VERSION,
        "phase": SOLUTION_ONLY_PHASE,
        "versions": _versions(),
        "config": asdict(config),
        "source_sha256_aggregate": source_aggregate,
        "pilot_limitations": list(PILOT_LIMITATIONS),
    }


def _metadata(config, spec, arrays, records, file_name, npz_digest):
    sources = source_hashes()
    source_aggregate = sha256(canonical_json_bytes(sources)).hexdigest()
    identity = _identity_payload(config, source_aggregate)
    return {
        **identity,
        "dataset_identity_sha256": sha256(canonical_json_bytes(identity)).hexdigest(),
        "preprocessing_contract": asdict(DEFAULT_CONTRACT),
        "bounds_generation": _bounds_policy(),
        "observation_generation": _observation_policy(config),
        "split_assignment": split_policy(),
        "execution": {
            name.lower(): os.environ.get(name)
            for name in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")
        },
        "shard": {
            "shard_index": spec.shard_index,
            "start_recipe_index": spec.start_recipe_index,
            "stop_recipe_index_exclusive": spec.stop_recipe_index,
            "recipe_count": spec.recipe_count,
            "row_count": arrays["global_recipe_index"].size,
            "file_name": file_name,
            "json_is_commit_marker": True,
        },
        "array_schema": array_schema(arrays),
        "recipe_records": list(records),
        "topology_histogram": _histogram(arrays["topology_id"]),
        "branch_pattern_histogram": _histogram(arrays["branch_pattern_id"]),
        "row_split_histogram": _histogram(arrays["assigned_split"]),
        "recipe_split_histogram": {
            name: sum(record["assigned_split"] == name for record in records)
            for name in SPLIT_NAMES
        },
        "range_regime_histogram": _histogram(arrays["range_regime"]),
        "bound_placement_histogram": _histogram(arrays["bound_placement"]),
        "source_sha256": sources,
        "npz_sha256": npz_digest,
    }


def build_shard(output_dir, config, spec) -> BoundsFirstShardArtifact:
    if not isinstance(config, BoundsFirstShardConfig) or not isinstance(
        spec, BoundsFirstShardSpec
    ):
        raise TypeError("config and spec must be V4 bounds-first shard contracts")
    output = Path(output_dir)
    stem = (
        f"bounds-first-v4-{config.topology_schedule}-views{config.views_per_recipe}-"
        f"{spec.shard_index:05d}-{spec.start_recipe_index:012d}-"
        f"{spec.stop_recipe_index:012d}"
    )
    npz_path, metadata_path = output / f"{stem}.npz", output / f"{stem}.json"
    if npz_path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite existing shard {stem!r}")
    arrays, records = generate_shard_arrays(config, spec)
    npz_payload = _deterministic_npz_bytes(arrays)
    npz_digest = sha256(npz_payload).hexdigest()
    metadata = _metadata(config, spec, arrays, records, npz_path.name, npz_digest)
    publish_exclusive(npz_path, npz_payload)
    try:
        publish_exclusive(metadata_path, canonical_json_bytes(metadata))
    except Exception:
        npz_path.unlink(missing_ok=True)
        raise
    return BoundsFirstShardArtifact(
        npz_path,
        metadata_path,
        spec.recipe_count,
        arrays["global_recipe_index"].size,
        npz_digest,
    )


def _validate_metadata(path, metadata, arrays) -> tuple[BoundsFirstShardConfig, BoundsFirstShardSpec]:
    if metadata.get("dataset_schema_version") != SHARD_SCHEMA_VERSION:
        raise ValueError("unsupported V4 bounds-first shard schema")
    if metadata.get("dataset_generator_version") != SHARD_GENERATOR_VERSION:
        raise ValueError("unsupported V4 bounds-first shard generator")
    if metadata.get("phase") != SOLUTION_ONLY_PHASE:
        raise ValueError("shard is not the solution-only V4 phase")
    if tuple(metadata.get("pilot_limitations", ())) != PILOT_LIMITATIONS:
        raise ValueError("V4 pilot limitation provenance is inconsistent")
    if metadata.get("versions") != _versions():
        raise ValueError("V4 scientific version contract mismatch")
    if metadata.get("preprocessing_contract") != asdict(DEFAULT_CONTRACT):
        raise ValueError("preprocessing contract mismatch")
    if metadata.get("bounds_generation") != _bounds_policy():
        raise ValueError("bounds-first generation policy mismatch")
    if metadata.get("split_assignment") != split_policy():
        raise ValueError("recipe-grouped split policy mismatch")
    execution = metadata.get("execution")
    execution_keys = {
        "slurm_job_id", "slurm_array_job_id", "slurm_array_task_id"
    }
    if not isinstance(execution, dict) or set(execution) != execution_keys or not all(
        value is None or isinstance(value, str) for value in execution.values()
    ):
        raise ValueError("Slurm execution provenance is inconsistent")
    sources = metadata.get("source_sha256")
    if not isinstance(sources, dict) or tuple(sorted(sources)) != tuple(
        sorted(_SOURCE_RELATIVE_PATHS)
    ):
        raise ValueError("source SHA-256 provenance file set is inconsistent")
    if not all(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        for value in sources.values()
    ):
        raise ValueError("source SHA-256 provenance contains an invalid digest")
    source_aggregate = sha256(canonical_json_bytes(sources)).hexdigest()
    if metadata.get("source_sha256_aggregate") != source_aggregate:
        raise ValueError("source SHA-256 aggregate is inconsistent")
    try:
        config = BoundsFirstShardConfig(**metadata["config"])
        shard = metadata["shard"]
        spec = BoundsFirstShardSpec(
            shard["shard_index"],
            shard["start_recipe_index"],
            shard["recipe_count"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid V4 shard config/range metadata") from exc
    identity = _identity_payload(config, source_aggregate)
    if metadata.get("dataset_identity_sha256") != sha256(
        canonical_json_bytes(identity)
    ).hexdigest():
        raise ValueError("dataset identity digest is inconsistent")
    expected_rows = spec.recipe_count * config.views_per_recipe
    if (
        shard.get("stop_recipe_index_exclusive") != spec.stop_recipe_index
        or shard.get("row_count") != expected_rows
        or shard.get("file_name") != path.name
        or shard.get("json_is_commit_marker") is not True
    ):
        raise ValueError("shard file/range provenance is inconsistent")
    if metadata.get("observation_generation") != _observation_policy(config):
        raise ValueError("observation-generation policy mismatch")
    if metadata.get("array_schema") != array_schema(arrays):
        raise ValueError("array schema metadata mismatch")
    records = metadata.get("recipe_records")
    if not isinstance(records, list):
        raise ValueError("recipe physical provenance is missing")
    validate_shard_semantics(config, spec, arrays, records)
    for field, key in (
        ("topology_id", "topology_histogram"),
        ("branch_pattern_id", "branch_pattern_histogram"),
        ("assigned_split", "row_split_histogram"),
        ("range_regime", "range_regime_histogram"),
        ("bound_placement", "bound_placement_histogram"),
    ):
        if metadata.get(key) != _histogram(arrays[field]):
            raise ValueError(f"{key} is inconsistent")
    recipe_split_histogram = {
        name: sum(record["assigned_split"] == name for record in records)
        for name in SPLIT_NAMES
    }
    if metadata.get("recipe_split_histogram") != recipe_split_histogram:
        raise ValueError("recipe split histogram is inconsistent")
    return config, spec


def load_shard(path: str | os.PathLike[str]) -> BoundsFirstNumpyShard:
    npz_path = Path(path)
    if npz_path.suffix != ".npz" or not npz_path.is_file():
        raise ValueError("V4 shard must be an existing .npz file")
    metadata_path = npz_path.with_suffix(".json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("could not read V4 shard commit metadata") from exc
    if file_sha256(npz_path) != metadata.get("npz_sha256"):
        raise ValueError("V4 shard NPZ checksum mismatch")
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError("could not read V4 NPZ arrays") from exc
    _validate_metadata(npz_path, metadata, arrays)
    for value in arrays.values():
        value.setflags(write=False)
    return BoundsFirstNumpyShard(
        npz_path,
        MappingProxyType(metadata),
        MappingProxyType(arrays),
    )


def audit_shards(shard_paths: Sequence[str | os.PathLike[str]]) -> dict[str, object]:
    paths = tuple(sorted(Path(value).resolve() for value in shard_paths))
    if not paths:
        raise ValueError("at least one V4 shard is required")
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate shard paths are not allowed")
    dataset_identity = None
    shard_indices, recipe_indices, view_keys = set(), set(), set()
    shard_records, split_counts = [], {name: 0 for name in SPLIT_NAMES}
    first_metadata = None
    for path in paths:
        shard = load_shard(path)
        metadata = shard.metadata
        identity = metadata["dataset_identity_sha256"]
        if dataset_identity is None:
            dataset_identity, first_metadata = identity, metadata
        elif identity != dataset_identity:
            raise ValueError("shards have a config/source/schema mismatch")
        shard_index = int(metadata["shard"]["shard_index"])
        if shard_index in shard_indices:
            raise ValueError("duplicate shard_index across V4 shards")
        shard_indices.add(shard_index)
        shard_recipe_indices = set(
            int(value) for value in np.unique(shard.arrays["global_recipe_index"])
        )
        overlap = recipe_indices.intersection(shard_recipe_indices)
        if overlap:
            raise ValueError("overlapping global_recipe_index across V4 shards")
        recipe_indices.update(shard_recipe_indices)
        for recipe_index, view_index in zip(
            shard.arrays["global_recipe_index"], shard.arrays["view_index"]
        ):
            key = (int(recipe_index), int(view_index))
            if key in view_keys:
                raise ValueError("overlapping recipe/view row across V4 shards")
            view_keys.add(key)
        for record in metadata["recipe_records"]:
            split_counts[record["assigned_split"]] += 1
        metadata_path = path.with_suffix(".json")
        shard_records.append(
            {
                "path": str(path),
                "shard_index": shard_index,
                "start_recipe_index": metadata["shard"]["start_recipe_index"],
                "stop_recipe_index_exclusive": metadata["shard"][
                    "stop_recipe_index_exclusive"
                ],
                "recipe_count": shard.recipe_count,
                "row_count": shard.row_count,
                "npz_sha256": metadata["npz_sha256"],
                "metadata_sha256": file_sha256(metadata_path),
                "execution": metadata["execution"],
            }
        )
    shard_records.sort(key=lambda item: (item["start_recipe_index"], item["shard_index"]))
    gaps = []
    cursor = shard_records[0]["start_recipe_index"]
    for record in shard_records:
        if record["start_recipe_index"] > cursor:
            gaps.append([cursor, record["start_recipe_index"]])
        cursor = max(cursor, record["stop_recipe_index_exclusive"])
    assert first_metadata is not None
    payload = {
        "audit_schema_version": MERGE_AUDIT_SCHEMA_VERSION,
        "dataset_identity_sha256": dataset_identity,
        "dataset_schema_version": SHARD_SCHEMA_VERSION,
        "phase": SOLUTION_ONLY_PHASE,
        "merge_eligible": True,
        "recipe_coverage_contiguous": not gaps,
        "recipe_index_gaps": gaps,
        "recipe_count": len(recipe_indices),
        "row_count": len(view_keys),
        "recipe_split_counts": split_counts,
        "config": first_metadata["config"],
        "versions": first_metadata["versions"],
        "source_sha256": first_metadata["source_sha256"],
        "source_sha256_aggregate": first_metadata["source_sha256_aggregate"],
        "split_assignment": first_metadata["split_assignment"],
        "pilot_limitations": first_metadata["pilot_limitations"],
        "shards": shard_records,
    }
    return {
        **payload,
        "audit_content_sha256": sha256(canonical_json_bytes(payload)).hexdigest(),
    }


def write_merge_audit(path, shard_paths) -> dict[str, object]:
    payload = audit_shards(shard_paths)
    publish_exclusive(Path(path), canonical_json_bytes(payload))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build one immutable array shard")
    build.add_argument("--output-dir", type=Path, default=DEFAULT_MAXWELL_OUTPUT_ROOT)
    build.add_argument("--shard-index", required=True, type=int)
    build.add_argument("--start-recipe-index", required=True, type=int)
    build.add_argument("--recipe-count", required=True, type=int)
    build.add_argument("--master-seed", type=int, default=20260903)
    build.add_argument(
        "--topology-schedule", choices=tuple(TOPOLOGY_SCHEDULES), default="all34"
    )
    build.add_argument("--views-per-recipe", type=int, default=3)
    build.add_argument("--max-raw-points", type=int, default=512)
    audit = commands.add_parser("audit", help="validate compatible non-overlapping shards")
    audit.add_argument("--shards", nargs="+", required=True, type=Path)
    audit.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MAXWELL_OUTPUT_ROOT / "bounds-first-v4-merge-audit.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        config = BoundsFirstShardConfig(
            args.master_seed,
            args.topology_schedule,
            args.views_per_recipe,
            args.max_raw_points,
        )
        artifact = build_shard(
            args.output_dir,
            config,
            BoundsFirstShardSpec(
                args.shard_index, args.start_recipe_index, args.recipe_count
            ),
        )
        output = asdict(artifact)
        output["npz_path"], output["metadata_path"] = (
            str(artifact.npz_path),
            str(artifact.metadata_path),
        )
    else:
        payload = write_merge_audit(args.output, args.shards)
        output = {
            "audit_path": str(args.output),
            "recipe_count": payload["recipe_count"],
            "row_count": payload["row_count"],
            "audit_content_sha256": payload["audit_content_sha256"],
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
