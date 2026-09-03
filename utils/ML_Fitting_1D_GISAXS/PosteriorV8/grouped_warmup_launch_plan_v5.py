"""Fail-closed, write-free planning for the Maxwell V5.1 warmup launcher."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

from .sobol_design_v5 import V5SobolDesign
from .sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_NAMES,
    V5_SOBOL_RECIPE_COORDINATE_SHA256,
)
from .split_design_v5 import V5SplitPlan


V5_WARMUP_LAUNCH_SCHEMA = "gisaxs.posterior_v8.maxwell_grouped_warmup_launch/v1"
V5_WARMUP_LAUNCH_VERSION = "posterior_v8_v5_1_exclusive_three_job_warmup_launch_v1"
MAXWELL_DUST_ROOT = Path("/data/dust/user/zhaiyufe")
LAUNCH_MANIFEST_FILENAME = "launch-manifest.json"
POSTERIOR_ROOT_RELATIVE = Path("utils/ML_Fitting_1D_GISAXS/PosteriorV8")
_FITTING_DOMAIN_RELATIVE = Path("src/gimap/features/fitting/domain")
DATASET_WRAPPER_RELATIVE = (
    POSTERIOR_ROOT_RELATIVE / "slurm/v5_grouped_warmup_dataset_cpu.sbatch"
)
TRAIN_WRAPPER_RELATIVE = POSTERIOR_ROOT_RELATIVE / "slurm/v5_grouped_train_gpu4.sbatch"
_REQUIRED_SOURCE_FILES = (
    POSTERIOR_ROOT_RELATIVE / "build_formal_sobol_grouped_shard_v5.py",
    POSTERIOR_ROOT_RELATIVE / "train_grouped_v5.py",
    POSTERIOR_ROOT_RELATIVE / "grouped_warmup_launch_plan_v5.py",
    POSTERIOR_ROOT_RELATIVE / "launch_grouped_warmup_v5.py",
    _FITTING_DOMAIN_RELATIVE / "physical_constraints.py",
    _FITTING_DOMAIN_RELATIVE / "scattering_model.py",
    DATASET_WRAPPER_RELATIVE,
    TRAIN_WRAPPER_RELATIVE,
)
_SOURCE_SUFFIXES = frozenset({".py", ".sbatch"})


@dataclass(frozen=True)
class V5WarmupLaunchConfig:
    source_root: Path
    run_root: Path
    split_plan: Path
    sobol_design: Path
    train_recipes: int
    validation_recipes: int
    recipes_per_shard: int
    warmup_epochs: int = 10
    seed: int = 20260903


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def safe_export_value(value: str, name: str, *, forbid_colon: bool = False) -> str:
    if any(token in value for token in (",", "\n", "\r", "\0")) or (
        forbid_colon and ":" in value
    ):
        raise ValueError(f"{name} contains a character unsupported by the Slurm export contract")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _under_root(path: Path, root: Path, name: str, *, must_exist: bool) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=must_exist)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{name} must be under {resolved_root}") from exc
    if resolved == resolved_root:
        raise ValueError(f"{name} must not be the Maxwell dust root itself")
    return resolved


def _source_fingerprint(source_root: Path) -> dict[str, object]:
    if not source_root.is_absolute():
        raise ValueError("source_root must be an absolute path")
    root = source_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("source_root must be an existing directory")
    bundle_roots = tuple(
        root / relative for relative in (POSTERIOR_ROOT_RELATIVE, _FITTING_DOMAIN_RELATIVE)
    )
    if any(not bundle.is_dir() for bundle in bundle_roots):
        raise FileNotFoundError("missing PosteriorV8 or authoritative fitting-domain source bundle")
    for relative in _REQUIRED_SOURCE_FILES:
        target = root / relative
        if not target.is_file():
            raise FileNotFoundError(f"missing required source contract: {target}")

    selected = []
    for bundle in bundle_roots:
        for path in bundle.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"source bundle must not contain symlinks: {path}")
            if (
                path.is_file()
                and path.suffix in _SOURCE_SUFFIXES
                and "__pycache__" not in path.parts
            ):
                selected.append(path)
    selected.sort(key=lambda item: item.relative_to(root).as_posix())
    if not selected:
        raise ValueError("PosteriorV8 source bundle contains no auditable source files")
    tree_digest = sha256()
    for path in selected:
        relative_bytes = path.relative_to(root).as_posix().encode("utf-8")
        tree_digest.update(len(relative_bytes).to_bytes(4, "big"))
        tree_digest.update(relative_bytes)
        tree_digest.update(bytes.fromhex(_file_sha256(path)))
    return {
        "source_root": str(root),
        "bundle_relative_paths": [
            POSTERIOR_ROOT_RELATIVE.as_posix(),
            _FITTING_DOMAIN_RELATIVE.as_posix(),
        ],
        "bundle_sha256": tree_digest.hexdigest(),
        "bundle_file_count": len(selected),
        "hash_algorithm": "sha256(length_prefixed_relative_path || file_sha256)",
        "required_file_sha256": {
            relative.as_posix(): _file_sha256(root / relative)
            for relative in _REQUIRED_SOURCE_FILES
        },
    }


def _contract_fingerprints(
    split_plan_path: Path,
    sobol_design_path: Path,
) -> tuple[V5SplitPlan, dict[str, object]]:
    plan_text = split_plan_path.read_text(encoding="utf-8")
    design_text = sobol_design_path.read_text(encoding="utf-8")
    plan = V5SplitPlan.from_json(plan_text)
    design = V5SobolDesign.from_json(design_text)
    if (
        design.coordinate_names != V5_SOBOL_RECIPE_COORDINATE_NAMES
        or design.coordinate_contract_sha256 != V5_SOBOL_RECIPE_COORDINATE_SHA256
    ):
        raise ValueError("Sobol design is not bound to the frozen direct-recipe coordinates")
    return plan, {
        "split_plan": {
            "path": str(split_plan_path),
            "file_sha256": sha256(plan_text.encode()).hexdigest(),
            "contract_sha256": plan.sha256,
        },
        "sobol_design": {
            "path": str(sobol_design_path),
            "file_sha256": sha256(design_text.encode()).hexdigest(),
            "contract_sha256": design.sha256,
            "coordinate_contract_sha256": design.coordinate_contract_sha256,
            "scipy_version": design.payload()["scipy_version"],
        },
    }


def replay_v5_warmup_fingerprints(
    config: V5WarmupLaunchConfig,
    allowed_root: Path,
) -> dict[str, object]:
    plan_path = _under_root(config.split_plan, allowed_root, "split_plan", must_exist=True)
    design_path = _under_root(config.sobol_design, allowed_root, "sobol_design", must_exist=True)
    _, contracts = _contract_fingerprints(plan_path, design_path)
    return {"source": _source_fingerprint(config.source_root), "contracts": contracts}


def _shard_windows(split: str, total: int, size: int, directory: Path) -> list[dict[str, object]]:
    result = []
    for index in range(math.ceil(total / size)):
        start = index * size
        result.append(
            {
                "array_task_id": index,
                "split_offset": start,
                "recipe_count": min(size, total - start),
                "output": str(directory / f"{split}-shard-{index:06d}.gvd5"),
            }
        )
    if sum(int(item["recipe_count"]) for item in result) != total:
        raise RuntimeError("internal shard coverage mismatch")
    return result


def build_v5_warmup_launch_plan(
    config: V5WarmupLaunchConfig,
    *,
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> dict[str, object]:
    """Validate immutable inputs and return a write-free submission plan."""

    if not isinstance(config, V5WarmupLaunchConfig):
        raise TypeError("config must be V5WarmupLaunchConfig")
    train_count = _positive_integer(config.train_recipes, "train_recipes")
    validation_count = _positive_integer(config.validation_recipes, "validation_recipes")
    shard_size = _positive_integer(config.recipes_per_shard, "recipes_per_shard")
    _positive_integer(config.warmup_epochs, "warmup_epochs")
    _positive_integer(config.seed, "seed")

    source = _source_fingerprint(config.source_root)
    run_root = _under_root(config.run_root, allowed_root, "run_root", must_exist=False)
    plan_path = _under_root(config.split_plan, allowed_root, "split_plan", must_exist=True)
    design_path = _under_root(config.sobol_design, allowed_root, "sobol_design", must_exist=True)
    if run_root.exists():
        raise FileExistsError(f"refusing to reuse existing run root: {run_root}")
    resolved_source_root = Path(str(source["source_root"]))
    if run_root == resolved_source_root or resolved_source_root in run_root.parents:
        raise ValueError("run_root must not be inside the immutable versioned source root")
    if not plan_path.is_file() or not design_path.is_file():
        raise ValueError("split_plan and sobol_design must be regular files")
    plan, contracts = _contract_fingerprints(plan_path, design_path)
    if train_count > plan.counts.train:
        raise ValueError("train_recipes exceeds the frozen train split")
    if validation_count > plan.counts.tuning_validation:
        raise ValueError("validation_recipes exceeds the frozen tuning-validation split")

    layout = {
        "logs": str(run_root / "logs"),
        "train_data": str(run_root / "data/train"),
        "validation_data": str(run_root / "data/tuning_validation"),
        "model": str(run_root / "models/warmup"),
        "manifest": str(run_root / LAUNCH_MANIFEST_FILENAME),
    }
    train_windows = _shard_windows(
        "train", train_count, shard_size, Path(layout["train_data"])
    )
    validation_windows = _shard_windows(
        "tuning_validation", validation_count, shard_size, Path(layout["validation_data"])
    )
    outputs = [
        *(Path(str(item["output"])) for item in train_windows),
        *(Path(str(item["output"])) for item in validation_windows),
        Path(layout["model"]),
        Path(layout["manifest"]),
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError("one or more planned output paths already exist")
    exported_paths = {
        **layout,
        "source_root": str(source["source_root"]),
        "split_plan": str(plan_path),
        "design": str(design_path),
    }
    for name, path in exported_paths.items():
        safe_export_value(path, name, forbid_colon=True)

    core: dict[str, object] = {
        "schema": V5_WARMUP_LAUNCH_SCHEMA,
        "version": V5_WARMUP_LAUNCH_VERSION,
        "run_root": str(run_root),
        "layout": layout,
        "source": source,
        "contracts": contracts,
        "configuration": {
            "train_recipes": train_count,
            "validation_recipes": validation_count,
            "recipes_per_shard": shard_size,
            "warmup_epochs": config.warmup_epochs,
            "full_epochs": 0,
            "seed": config.seed,
            "observation_view_indices": [0, 1, 2],
            "generating_candidate_only": False,
            "source_topology_candidate_catalog": "all_feasible_wire_branches",
        },
        "arrays": {
            "train": {
                "array_spec": f"0-{len(train_windows) - 1}",
                "task_count": len(train_windows),
                "windows": train_windows,
            },
            "tuning_validation": {
                "array_spec": f"0-{len(validation_windows) - 1}",
                "task_count": len(validation_windows),
                "windows": validation_windows,
            },
        },
        "dependency": "GPU warmup uses afterok on both complete dataset arrays",
        "login_node_work": "hash/contract/path validation, directory reservation, sbatch only",
        "overwrite_policy": "exclusive_new_run_root_and_artifacts",
    }
    core["plan_sha256"] = sha256(canonical_json(core).encode()).hexdigest()
    return core


__all__ = [
    "DATASET_WRAPPER_RELATIVE",
    "LAUNCH_MANIFEST_FILENAME",
    "MAXWELL_DUST_ROOT",
    "TRAIN_WRAPPER_RELATIVE",
    "V5WarmupLaunchConfig",
    "build_v5_warmup_launch_plan",
    "canonical_json",
    "replay_v5_warmup_fingerprints",
    "safe_export_value",
]
