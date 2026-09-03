"""Build an engineering-only checked K=1 memorization dataset on Slurm."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from numbers import Integral
import os
from pathlib import Path
import socket
from typing import Mapping, Sequence

import numpy as np

from .branch_catalog import BRANCH_PATTERN_COUNT, VALID_BRANCH_PATTERN_MASK
from .build_grouped_dataset_v5 import build_tiny_v5_grouped_dataset
from .contract import SHAPES, topology_id_for
from .grouped_artifact_v5 import canonical_json
from .grouped_dataset_v5 import write_v5_grouped_dataset


V5_K1_DATASET_BUILDER_SCHEMA = "gisaxs.posterior_v8.k1_memorization_dataset_builder/v1"
V5_K1_DATASET_BUILDER_VERSION = "posterior_v8_v5_2_engineering_k1_known_truth_dataset_v1"
V5_K1_DATASET_BUILDER_ROLE = "engineering_memorization_wiring_only_not_model_acceptance"
MAXWELL_DUST_ROOT = Path("/data/dust/user/zhaiyufe")


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _view_indices(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            selected = tuple(int(item.strip()) for item in value.split(",") if item.strip())
        except ValueError as exc:
            raise ValueError("view_indices must be comma-separated integers") from exc
    else:
        selected = tuple(value)
    result = tuple(_integer(item, "view_index") for item in selected)
    if not result or len(result) != len(set(result)):
        raise ValueError("view_indices must be non-empty and unique")
    return result


def _under_root(path: Path, root: Path, name: str) -> Path:
    resolved_root = root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{name} must be below {resolved_root}")
    return resolved


def _execution_guard(hostname: str, environment: Mapping[str, str]) -> str:
    if hostname.split(".", 1)[0].startswith("max-wgs"):
        raise RuntimeError("K1 dataset generation must run on a Slurm worker, not max-wgs")
    job_id = environment.get("SLURM_JOB_ID", "")
    if not job_id.isdigit() or int(job_id) < 1:
        raise RuntimeError("K1 dataset generation requires a valid SLURM_JOB_ID")
    return job_id


def build_v5_k1_memorization_dataset(
    output: str | os.PathLike[str],
    *,
    recipe_count: int,
    topology: str,
    base_seed: int,
    view_indices: str | Sequence[int],
    pattern_id: int,
    dry_run: bool = False,
    allowed_root: Path = MAXWELL_DUST_ROOT,
    hostname: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Plan or exclusively publish one deterministic K=1 checked artifact."""

    count = _integer(recipe_count, "recipe_count", minimum=1)
    seed = _integer(base_seed, "base_seed")
    if seed > np.iinfo(np.uint64).max:
        raise ValueError("base_seed must fit in uint64")
    if topology not in SHAPES:
        raise ValueError(f"topology must be one of {SHAPES}")
    pattern = _integer(pattern_id, "pattern_id")
    if pattern >= BRANCH_PATTERN_COUNT:
        raise ValueError(f"pattern_id must be below {BRANCH_PATTERN_COUNT}")
    topology_id = topology_id_for((topology,))
    if not VALID_BRANCH_PATTERN_MASK[topology_id][pattern]:
        raise ValueError("pattern_id is not valid for the selected K=1 topology")
    views = _view_indices(view_indices)
    if type(dry_run) is not bool:
        raise TypeError("dry_run must be a bool")

    target = _under_root(Path(output), allowed_root, "output")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing checked dataset: {target}")
    if not target.parent.is_dir():
        raise FileNotFoundError(f"output parent directory does not exist: {target.parent}")
    plan_core = {
        "schema_version": V5_K1_DATASET_BUILDER_SCHEMA,
        "version": V5_K1_DATASET_BUILDER_VERSION,
        "scientific_role": V5_K1_DATASET_BUILDER_ROLE,
        "model_acceptance_evidence": False,
        "recipe_count": count,
        "topology": topology,
        "topology_id": topology_id,
        "base_seed": seed,
        "view_indices": list(views),
        "pattern_id": pattern,
        "generating_candidate_only": True,
        "output": str(target),
    }
    plan = {
        **plan_core,
        "plan_sha256": sha256(canonical_json(plan_core).encode()).hexdigest(),
    }
    if dry_run:
        return {
            "status": "checked_dry_run",
            "writes_performed": False,
            "plan": plan,
        }

    host = socket.gethostname() if hostname is None else hostname
    selected_environment = os.environ if environment is None else environment
    job_id = _execution_guard(host, selected_environment)
    dataset = build_tiny_v5_grouped_dataset(
        recipe_count=count,
        topology=(topology,),
        base_seed=seed,
        view_indices=views,
        split_id="train",
        pattern_id=pattern,
        generating_only=True,
    )
    receipt = write_v5_grouped_dataset(dataset, target)
    return {
        "status": "published",
        "writes_performed": True,
        "plan": plan,
        "dataset": {
            "dataset_id": dataset.manifest["dataset_id"],
            "manifest_sha256": dataset.manifest["manifest_sha256"],
            "artifact_sha256": receipt.artifact_sha256,
            "byte_count": receipt.byte_count,
            "source_sha256": dict(dataset.manifest["source_sha256"]),
        },
        "execution": {"hostname": host, "slurm_job_id": job_id},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-count", required=True, type=int)
    parser.add_argument("--topology", required=True, choices=SHAPES)
    parser.add_argument("--base-seed", required=True, type=int)
    parser.add_argument("--view-indices", required=True)
    parser.add_argument("--pattern-id", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_v5_k1_memorization_dataset(
        args.output,
        recipe_count=args.recipe_count,
        topology=args.topology,
        base_seed=args.base_seed,
        view_indices=args.view_indices,
        pattern_id=args.pattern_id,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MAXWELL_DUST_ROOT",
    "V5_K1_DATASET_BUILDER_ROLE",
    "V5_K1_DATASET_BUILDER_SCHEMA",
    "V5_K1_DATASET_BUILDER_VERSION",
    "build_v5_k1_memorization_dataset",
    "main",
]
