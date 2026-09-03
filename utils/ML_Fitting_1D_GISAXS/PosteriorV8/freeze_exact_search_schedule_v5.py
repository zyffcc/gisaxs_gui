"""Freeze the V5.1 local-Sobol exact-search schedule in the active runtime.

The schedule binds SciPy's exact version and the generated point bytes.  Run
this lightweight command with Maxwell's designated ``tf`` environment.  The
default is a write-free preview; ``--write`` publishes one checked artifact
exclusively and never replaces an earlier schedule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from typing import Sequence

import scipy

from .exact_search_schedule_v5 import (
    V5FrozenLocalSobolSchedule,
    write_v5_frozen_local_sobol_schedule,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-id", required=True)
    parser.add_argument("--point-count", required=True, type=int)
    parser.add_argument("--base-seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Publish the checked artifact; without this flag nothing is written.",
    )
    return parser


def freeze_v5_exact_search_schedule(
    *,
    schedule_id: str,
    point_count: int,
    base_seed: int,
    output: Path,
    write: bool = False,
) -> dict[str, object]:
    """Generate the exact runtime-bound schedule and optionally publish it."""

    schedule = V5FrozenLocalSobolSchedule.generate(
        schedule_id=schedule_id,
        point_count=point_count,
        base_seed=base_seed,
    )
    schedule.verify_runtime_replay()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing schedule: {output}")
    result: dict[str, object] = {
        "status": "written" if write else "dry_run",
        "writes_performed": bool(write),
        "output": str(output),
        "runtime": {
            "python": platform.python_version(),
            "scipy": scipy.__version__,
        },
        "schedule": schedule.audit_payload(),
        "schedule_sha256": schedule.sha256,
        "instruction": (
            "freeze once in the exact Maxwell tf/SciPy runtime, then pass this checked "
            "artifact unchanged to both train and tuning-validation arrays"
        ),
    }
    if write:
        if not output.parent.is_dir():
            raise FileNotFoundError(
                f"schedule parent directory does not exist: {output.parent}"
            )
        receipt = write_v5_frozen_local_sobol_schedule(schedule, output)
        result["artifact_sha256"] = receipt.artifact_sha256
        result["manifest_sha256"] = receipt.manifest_sha256
        result["byte_count"] = receipt.byte_count
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = freeze_v5_exact_search_schedule(
        schedule_id=args.schedule_id,
        point_count=args.point_count,
        base_seed=args.base_seed,
        output=args.output,
        write=args.write,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["freeze_v5_exact_search_schedule", "main"]
