"""Immutable inputs and replay fingerprints for V5 frozen-search launch."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .calibrated_search_threshold_v5 import (
    V5CheckedCompatibilityCalibration,
    inspect_v5_compatibility_calibration,
)
from .exact_search_executor_v5 import build_v5_frozen_exact_search_protocol
from .exact_search_schedule_v5 import (
    V5FrozenExactOptimizerSchedule,
    read_v5_frozen_local_sobol_schedule,
)
from .frozen_search_pipeline_contract_v5 import V5SelectedTopologySearchSchedule
from .search_supervision_contract_v5 import (
    V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
)
from .sobol_design_v5 import V5SobolDesign
from .sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_NAMES,
    V5_SOBOL_RECIPE_COORDINATE_SHA256,
)
from .split_design_v5 import V5SplitPlan


MAXWELL_DUST_ROOT = Path("/data/dust/user/zhaiyufe")
LAUNCH_MANIFEST_FILENAME = "frozen-search-launch-manifest.json"
POSTERIOR_ROOT_RELATIVE = Path("utils/ML_Fitting_1D_GISAXS/PosteriorV8")
FROZEN_SEARCH_WRAPPER_RELATIVE = (
    POSTERIOR_ROOT_RELATIVE / "slurm/v5_frozen_search_labels_cpu.sbatch"
)
_DOMAIN_RELATIVE = Path("src/gimap/features/fitting/domain")
_REQUIRED_SOURCE_FILES = (
    POSTERIOR_ROOT_RELATIVE / "calibrated_search_threshold_v5.py",
    POSTERIOR_ROOT_RELATIVE / "exact_search_executor_v5.py",
    POSTERIOR_ROOT_RELATIVE / "exact_search_schedule_v5.py",
    POSTERIOR_ROOT_RELATIVE / "freeze_exact_search_schedule_v5.py",
    POSTERIOR_ROOT_RELATIVE / "frozen_search_launch_contracts_v5.py",
    POSTERIOR_ROOT_RELATIVE / "frozen_search_pipeline_contract_v5.py",
    POSTERIOR_ROOT_RELATIVE / "frozen_search_pipeline_v5.py",
    POSTERIOR_ROOT_RELATIVE / "frozen_search_workload_v5.py",
    POSTERIOR_ROOT_RELATIVE / "run_frozen_search_pipeline_v5.py",
    POSTERIOR_ROOT_RELATIVE / "search_supervision_sidecar_v5.py",
    POSTERIOR_ROOT_RELATIVE / "sobol_universal_query_design_v5.py",
    FROZEN_SEARCH_WRAPPER_RELATIVE,
    _DOMAIN_RELATIVE / "physical_constraints.py",
    _DOMAIN_RELATIVE / "scattering_model.py",
)


@dataclass(frozen=True, kw_only=True)
class V5FrozenSearchLaunchConfig:
    source_root: Path
    run_root: Path
    split_plan: Path
    sobol_design: Path
    local_sobol_schedule: Path
    train_start: int
    train_recipes: int
    validation_start: int
    validation_recipes: int
    recipes_per_shard: int
    view_indices: tuple[int, ...]
    topology_schedule_id: str
    selected_topology_ids: tuple[int, ...]
    optimizer_schedule_id: str
    direct_scout_seed_count: int
    per_seed_forward_evaluation_limit: int
    protocol_id: str
    standardized_threshold_name: str | None
    standardized_threshold_value: float | None
    raw_threshold_name: str | None
    raw_threshold_value: float | None
    threshold_source_id: str | None
    compatibility_calibration: Path | None
    pilot_throughput_source_id: str
    pilot_effective_seconds_per_exact_forward_call: float
    runtime_safety_factor: float = 10.0
    ftol: float = 1.0e-8
    xtol: float = 1.0e-8
    gtol: float = 1.0e-8


def under_v5_launch_root(
    path: Path, allowed_root: Path, name: str, *, must_exist: bool
) -> Path:
    root = allowed_root.resolve()
    selected = path.resolve(strict=must_exist)
    if selected == root or root not in selected.parents:
        raise ValueError(f"{name} must be below {root}")
    return selected


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_v5_frozen_search_source(source_root: Path) -> dict[str, object]:
    """Hash the complete worker source bundle without executing physics."""

    root = source_root.resolve(strict=True)
    for relative in _REQUIRED_SOURCE_FILES:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"required immutable source is missing: {path}")
    selected = []
    for relative_root in (POSTERIOR_ROOT_RELATIVE, _DOMAIN_RELATIVE):
        for path in (root / relative_root).rglob("*"):
            if path.is_symlink():
                raise ValueError(f"source bundle must not contain symlinks: {path}")
            if path.is_file() and path.suffix in {".py", ".sbatch"}:
                if "__pycache__" not in path.parts:
                    selected.append(path)
    selected.sort(key=lambda value: value.relative_to(root).as_posix())
    digest = sha256()
    for path in selected:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_file_sha256(path)))
    return {
        "source_root": str(root),
        "bundle_sha256": digest.hexdigest(),
        "bundle_file_count": len(selected),
        "required_file_sha256": {
            str(value): _file_sha256(root / value) for value in _REQUIRED_SOURCE_FILES
        },
    }


def load_v5_frozen_search_launch_contracts(config: V5FrozenSearchLaunchConfig):
    plan_text = config.split_plan.read_text(encoding="utf-8")
    design_text = config.sobol_design.read_text(encoding="utf-8")
    plan = V5SplitPlan.from_json(plan_text)
    design = V5SobolDesign.from_json(design_text)
    if (
        design.coordinate_names != V5_SOBOL_RECIPE_COORDINATE_NAMES
        or design.coordinate_contract_sha256 != V5_SOBOL_RECIPE_COORDINATE_SHA256
    ):
        raise ValueError("Sobol design is not bound to the direct V5 recipe contract")
    seed_schedule, seed_receipt = read_v5_frozen_local_sobol_schedule(
        config.local_sobol_schedule
    )
    seed_schedule.verify_runtime_replay()
    optimizer = V5FrozenExactOptimizerSchedule(
        schedule_id=config.optimizer_schedule_id,
        direct_scout_seed_count=config.direct_scout_seed_count,
        per_seed_forward_evaluation_limit=config.per_seed_forward_evaluation_limit,
        ftol=config.ftol,
        xtol=config.xtol,
        gtol=config.gtol,
    )
    calibration = (
        None
        if config.compatibility_calibration is None
        else inspect_v5_compatibility_calibration(config.compatibility_calibration)
    )
    protocol_kwargs = {
        "protocol_id": config.protocol_id,
        "seed_schedule": seed_schedule,
        "optimizer_schedule": optimizer,
    }
    if calibration is None:
        protocol = build_v5_frozen_exact_search_protocol(
            **protocol_kwargs,
            standardized_threshold_name=config.standardized_threshold_name,
            standardized_threshold_value=config.standardized_threshold_value,
            raw_threshold_name=config.raw_threshold_name,
            raw_threshold_value=config.raw_threshold_value,
            threshold_source_id=config.threshold_source_id,
        )
    else:
        protocol = build_v5_frozen_exact_search_protocol(
            **protocol_kwargs,
            protocol_tier=V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
            calibration_identity=calibration.identity,
        )
    topology = V5SelectedTopologySearchSchedule(
        schedule_id=config.topology_schedule_id,
        selected_topology_ids=config.selected_topology_ids,
    )
    return (
        plan,
        design,
        seed_schedule,
        seed_receipt,
        optimizer,
        protocol,
        topology,
        calibration,
        plan_text,
        design_text,
    )


def v5_frozen_search_contract_payload(
    config: V5FrozenSearchLaunchConfig,
    *,
    plan: V5SplitPlan,
    design: V5SobolDesign,
    seed_schedule,
    seed_receipt,
    optimizer: V5FrozenExactOptimizerSchedule,
    protocol,
    topology: V5SelectedTopologySearchSchedule,
    calibration: V5CheckedCompatibilityCalibration | None,
    plan_text: str,
    design_text: str,
) -> dict[str, object]:
    payload = {
        "split_plan": {
            "path": str(config.split_plan.resolve()),
            "file_sha256": sha256(plan_text.encode()).hexdigest(),
            "contract_sha256": plan.sha256,
        },
        "sobol_design": {
            "path": str(config.sobol_design.resolve()),
            "file_sha256": sha256(design_text.encode()).hexdigest(),
            "contract_sha256": design.sha256,
            "scipy_version": design.payload()["scipy_version"],
        },
        "local_sobol_schedule": {
            "path": str(config.local_sobol_schedule.resolve()),
            "artifact_sha256": seed_receipt.artifact_sha256,
            "manifest_sha256": seed_receipt.manifest_sha256,
            "schedule_sha256": seed_schedule.sha256,
        },
        "optimizer_schedule": optimizer.audit_payload(),
        "optimizer_schedule_sha256": optimizer.sha256,
        "protocol": protocol.audit_payload(),
        "protocol_sha256": protocol.sha256,
        "topology_schedule": topology.audit_payload(),
        "topology_schedule_sha256": topology.sha256,
    }
    payload["compatibility_calibration"] = (
        None
        if calibration is None
        else {
            "path": str(config.compatibility_calibration.resolve()),
            "identity": calibration.identity.audit_payload(),
            "identity_sha256": calibration.identity.sha256,
        }
    )
    return payload


def replay_v5_frozen_search_launch_fingerprints(
    config: V5FrozenSearchLaunchConfig,
    *,
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> dict[str, object]:
    """Recompute mutation-sensitive identities immediately before each sbatch."""

    under_v5_launch_root(
        config.source_root, allowed_root, "source_root", must_exist=True
    )
    for name in ("split_plan", "sobol_design", "local_sobol_schedule"):
        under_v5_launch_root(
            getattr(config, name), allowed_root, name, must_exist=True
        )
    if config.compatibility_calibration is not None:
        under_v5_launch_root(
            config.compatibility_calibration,
            allowed_root,
            "compatibility_calibration",
            must_exist=True,
        )
    source = fingerprint_v5_frozen_search_source(config.source_root)
    values = load_v5_frozen_search_launch_contracts(config)
    contracts = v5_frozen_search_contract_payload(
        config,
        plan=values[0],
        design=values[1],
        seed_schedule=values[2],
        seed_receipt=values[3],
        optimizer=values[4],
        protocol=values[5],
        topology=values[6],
        calibration=values[7],
        plan_text=values[8],
        design_text=values[9],
    )
    return {"source": source, "contracts": contracts}


__all__ = [
    "FROZEN_SEARCH_WRAPPER_RELATIVE",
    "LAUNCH_MANIFEST_FILENAME",
    "MAXWELL_DUST_ROOT",
    "V5FrozenSearchLaunchConfig",
    "fingerprint_v5_frozen_search_source",
    "load_v5_frozen_search_launch_contracts",
    "replay_v5_frozen_search_launch_fingerprints",
    "under_v5_launch_root",
    "v5_frozen_search_contract_payload",
]
