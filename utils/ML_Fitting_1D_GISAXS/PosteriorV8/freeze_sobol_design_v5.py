"""Exclusively freeze one V5 split plan and direct-recipe Sobol design.

This command is intentionally lightweight so the version-bound Sobol design
can be created on a Maxwell login node inside the exact runtime later used by
the Slurm dataset builders.  It never materializes curves or Sobol points.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from typing import Sequence

import numpy as np
import scipy

from .sobol_recipe_coordinates_v5 import v5_sobol_recipe_design
from .split_design_v5 import V5SplitCounts, V5SplitPlan


V5_DESIGN_FREEZE_RECEIPT_SCHEMA = "gisaxs.posterior_v8.sobol_design_freeze_receipt/v1"
V5_DESIGN_FREEZE_RECEIPT_VERSION = "posterior_v8_exclusive_runtime_bound_design_freeze_v1"
SPLIT_PLAN_FILENAME = "split-plan.json"
SOBOL_DESIGN_FILENAME = "sobol-design.json"
FREEZE_RECEIPT_FILENAME = "freeze-receipt.json"


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace")
    return value


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _exclusive_write(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def freeze_v5_sobol_design(
    output_directory: str | os.PathLike[str],
    *,
    experiment_id: str,
    counts: V5SplitCounts,
    scramble_seed: int,
    start_index: int = 0,
    guard_band: int = 1024,
) -> dict[str, object]:
    """Create a new immutable directory containing plan, design, and receipt.

    The directory must not already exist.  The returned receipt binds the
    exact serialized bytes as well as the SciPy/NumPy/Python runtime.  Official
    Maxwell data builders load and reproduce both JSON contracts independently.
    """

    experiment = _identifier(experiment_id, "experiment_id")
    if not isinstance(counts, V5SplitCounts):
        raise TypeError("counts must be V5SplitCounts")
    plan = V5SplitPlan.create(
        counts,
        start_index=start_index,
        guard_band=guard_band,
    )
    design = v5_sobol_recipe_design(scramble_seed=scramble_seed)
    plan_text = plan.to_json()
    design_text = design.to_json()

    target = Path(output_directory)
    target.mkdir(parents=True, exist_ok=False)
    plan_path = target / SPLIT_PLAN_FILENAME
    design_path = target / SOBOL_DESIGN_FILENAME
    receipt_path = target / FREEZE_RECEIPT_FILENAME
    _exclusive_write(plan_path, plan_text)
    _exclusive_write(design_path, design_text)

    receipt_core: dict[str, object] = {
        "schema": V5_DESIGN_FREEZE_RECEIPT_SCHEMA,
        "version": V5_DESIGN_FREEZE_RECEIPT_VERSION,
        "experiment_id": experiment,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "split_counts": asdict(counts),
        "start_index": plan.start_index,
        "guard_band": plan.guard_band,
        "assigned_recipe_count": plan.assigned_count,
        "sobol_scramble_seed": design.scramble_seed,
        "split_plan_filename": SPLIT_PLAN_FILENAME,
        "split_plan_contract_sha256": plan.sha256,
        "split_plan_file_sha256": _sha256_bytes(plan_text.encode("utf-8")),
        "sobol_design_filename": SOBOL_DESIGN_FILENAME,
        "sobol_design_contract_sha256": design.sha256,
        "sobol_design_file_sha256": _sha256_bytes(design_text.encode("utf-8")),
        "coordinate_contract_sha256": design.coordinate_contract_sha256,
        "heavy_compute_performed": False,
        "overwrite_policy": "exclusive_new_directory_and_files",
    }
    canonical = json.dumps(receipt_core, sort_keys=True, separators=(",", ":"), allow_nan=False)
    receipt = {
        **receipt_core,
        "receipt_sha256": _sha256_bytes(canonical.encode("utf-8")),
    }
    _exclusive_write(
        receipt_path,
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--scramble-seed", required=True, type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--guard-band", type=int, default=1024)
    parser.add_argument("--train", required=True, type=int)
    parser.add_argument("--tuning-validation", required=True, type=int)
    parser.add_argument("--calibration", required=True, type=int)
    parser.add_argument("--test", required=True, type=int)
    parser.add_argument("--reference", required=True, type=int)
    parser.add_argument("--ood-topology", required=True, type=int)
    parser.add_argument("--ood-range-width", required=True, type=int)
    parser.add_argument("--ood-weak-component", required=True, type=int)
    parser.add_argument("--ood-acquisition-policy", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = freeze_v5_sobol_design(
        args.output_directory,
        experiment_id=args.experiment_id,
        counts=V5SplitCounts(
            train=args.train,
            tuning_validation=args.tuning_validation,
            calibration=args.calibration,
            test=args.test,
            reference=args.reference,
            ood_topology=args.ood_topology,
            ood_range_width=args.ood_range_width,
            ood_weak_component=args.ood_weak_component,
            ood_acquisition_policy=args.ood_acquisition_policy,
        ),
        scramble_seed=args.scramble_seed,
        start_index=args.start_index,
        guard_band=args.guard_band,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())


__all__ = [
    "FREEZE_RECEIPT_FILENAME",
    "SOBOL_DESIGN_FILENAME",
    "SPLIT_PLAN_FILENAME",
    "V5_DESIGN_FREEZE_RECEIPT_SCHEMA",
    "V5_DESIGN_FREEZE_RECEIPT_VERSION",
    "freeze_v5_sobol_design",
    "main",
]
