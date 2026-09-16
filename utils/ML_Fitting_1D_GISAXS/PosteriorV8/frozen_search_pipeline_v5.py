"""Replay one immutable V5.1 cross-topology search shard.
Heavy work is Slurm-only; outputs publish in evidence-to-completion order.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
from typing import Callable, Mapping, Sequence

import numpy as np

from .build_formal_sobol_grouped_shard_v5 import (
    V5_FORMAL_SOBOL_GROUPED_BUILDER_SCHEMA,
    V5_FORMAL_SOBOL_GROUPED_BUILDER_VERSION,
)
from .build_grouped_dataset_v5 import (
    V5GroupedRecipeSpec,
    build_v5_grouped_solution_dataset,
)
from .calibrated_search_threshold_v5 import (
    V5CheckedCompatibilityCalibration,
    bind_v5_calibrated_observation_threshold,
)
from .exact_search_executor_v5 import V5FrozenExactSearchExecutor
from .exact_search_schedule_v5 import (
    V5FrozenExactOptimizerSchedule,
    V5FrozenLocalSobolSchedule,
)
from .frozen_search_pipeline_contract_v5 import (
    V5FrozenSearchShardPlan,
    V5SelectedTopologySearchSchedule,
    V5_FROZEN_SEARCH_PIPELINE_SCHEMA,
    V5_FROZEN_SEARCH_PIPELINE_VERSION,
    V5_SEARCH_PIPELINE_ALLOWED_SPLITS,
    V5_SEARCH_PIPELINE_FORMAL_SCOPE,
    V5_SEARCH_PIPELINE_FORMAL_SIDECAR_PREFIX,
    V5_SEARCH_PIPELINE_PILOT_SIDECAR_PREFIX,
    V5_SEARCH_PIPELINE_SCOPE,
    V5_SEARCH_PIPELINE_TRAINING_SCOPE,
    V5_TOPOLOGY_SEARCH_SCHEDULE_SCHEMA,
    V5_TOPOLOGY_SEARCH_SCHEDULE_VERSION,
    plan_v5_frozen_search_shard,
)
from .formal_production_search_runtime_v5 import (
    V5FormalProductionExecutableShard,
)
from .formal_production_search_plan_v5 import (
    V5FormalProductionSearchAuthorization,
)
from .grouped_artifact_v5 import V5ArtifactReceipt, canonical_json
from .grouped_dataset_v5 import (
    V5GroupedDataset,
    observation_array,
    read_v5_grouped_dataset,
    write_v5_grouped_dataset,
)
from .k1_balanced_full_search_runtime_v5 import (
    V5K1BalancedFullSearchExecutableTask,
    V5K1BalancedFullSearchQueryDesign,
)
from .observation_v5 import build_v5_observation_data_views
from .search_evidence_receipt_v5 import (
    V5_SEARCH_PIPELINE_TRAINING_SIDECAR_PREFIX,
    V5_SEARCH_LABEL_PURPOSE_FORMAL_CONTRACT_SMOKE,
    V5_SEARCH_LABEL_PURPOSE_PILOT,
    V5_SEARCH_LABEL_PURPOSE_TRAINING,
    audit_and_write_v5_search_evidence_receipt,
    build_v5_search_label_binding,
    evidence_receipt_path_for_sidecar,
)
from .search_supervision_contract_v5 import (
    V5ExactSearchObservation,
    V5FrozenExactSearchProtocol,
    V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
    V5UniversalSearchSpec,
)
from .search_supervision_sidecar_v5 import (
    collect_v5_search_supervision_sidecar,
    write_v5_search_supervision_sidecar,
)
from .sobol_recipe_v5 import materialize_v5_sobol_clean_recipe
from .sobol_universal_query_design_v5 import (
    V5SobolUniversalTopologyQueryDesign,
)
from .universal_query_v5 import build_v5_universal_candidate_context


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    if not path.parent.is_dir():
        raise FileNotFoundError(f"JSON parent directory does not exist: {path.parent}")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _expected_parent_dataset(
    shard_plan: (
        V5FrozenSearchShardPlan
        | V5FormalProductionExecutableShard
        | V5K1BalancedFullSearchExecutableTask
    ),
) -> tuple[V5GroupedDataset, tuple[object, ...]]:
    if isinstance(shard_plan, V5K1BalancedFullSearchExecutableTask):
        return shard_plan.parent_dataset, shard_plan.recipes
    formal = isinstance(shard_plan, V5FormalProductionExecutableShard)
    grouped = None if formal else shard_plan.grouped_plan
    split_plan = shard_plan.split_plan if formal else grouped.plan
    sobol_design = shard_plan.sobol_design if formal else grouped.design
    recipes = tuple(
        materialize_v5_sobol_clean_recipe(point, sobol_design)
        for point in shard_plan.points
    )
    specs = tuple(
        V5GroupedRecipeSpec.from_design_point(
            recipe,
            point,
            split_plan_sha256=split_plan.sha256,
            sobol_design_sha256=sobol_design.sha256,
            view_indices=(
                shard_plan.view_indices_for_recipe(index)
                if formal
                else grouped.view_indices
            ),
        )
        for index, (recipe, point) in enumerate(zip(recipes, shard_plan.points))
    )
    identity = {
        "builder_schema": V5_FORMAL_SOBOL_GROUPED_BUILDER_SCHEMA,
        "builder_version": V5_FORMAL_SOBOL_GROUPED_BUILDER_VERSION,
        "selection_sha256": (
            shard_plan.sha256
            if formal
            else grouped.selection["selection_sha256"]
        ),
    }
    dataset_id = (
        "formal-sobol-v5-"
        + sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    )
    dataset = build_v5_grouped_solution_dataset(
        specs,
        dataset_id=dataset_id,
        generating_only=False,
        shard_selection=None if formal else grouped.selection,
    )
    return dataset, recipes


def _same_dataset(expected: V5GroupedDataset, actual: V5GroupedDataset) -> bool:
    if dict(expected.manifest) != dict(actual.manifest):
        return False
    if set(expected.arrays) != set(actual.arrays):
        return False
    return all(
        expected.arrays[name].dtype == actual.arrays[name].dtype
        and expected.arrays[name].shape == actual.arrays[name].shape
        and np.array_equal(expected.arrays[name], actual.arrays[name])
        for name in expected.arrays
    )


def materialize_or_verify_v5_frozen_search_parent(
    shard_plan: (
        V5FrozenSearchShardPlan
        | V5FormalProductionExecutableShard
        | V5K1BalancedFullSearchExecutableTask
    ),
    parent_path: str | os.PathLike[str],
) -> tuple[V5GroupedDataset, V5ArtifactReceipt, tuple[object, ...], bool]:
    """Create a deterministic parent, or byte-strictly validate an existing one."""

    expected, recipes = _expected_parent_dataset(shard_plan)
    target = Path(parent_path)
    if target.exists():
        actual, receipt = read_v5_grouped_dataset(target)
        if not _same_dataset(expected, actual):
            raise ValueError("existing grouped parent does not reproduce the frozen shard")
        return actual, receipt, recipes, True
    if not target.parent.is_dir():
        raise FileNotFoundError(f"grouped parent directory does not exist: {target.parent}")
    receipt = write_v5_grouped_dataset(expected, target)
    return expected, receipt, recipes, False


def _catalog_artifact_id(
    design: V5SobolUniversalTopologyQueryDesign | V5K1BalancedFullSearchQueryDesign,
) -> str:
    if isinstance(design, V5K1BalancedFullSearchQueryDesign):
        return f"v5-k1-forced-universal-query-set/{design.sha256}"
    return f"v5-sobol-universal-topology-query-design/{design.sha256}"


def _publish_or_verify_query_catalogs(
    shard_plan: (
        V5FrozenSearchShardPlan
        | V5FormalProductionExecutableShard
        | V5K1BalancedFullSearchExecutableTask
    ),
    directory: Path,
) -> tuple[dict[str, tuple[str, str]], dict[str, object]]:
    directory.mkdir(parents=False, exist_ok=True)
    bindings: dict[str, tuple[str, str]] = {}
    rows = []
    forced_balanced = isinstance(shard_plan, V5K1BalancedFullSearchExecutableTask)
    for design in shard_plan.query_designs:
        path = directory / f"{design.clean_group_id}.json"
        encoded = design.to_json()
        if path.exists():
            if path.read_text(encoding="utf-8") != encoded:
                raise ValueError("existing topology-query catalog does not reproduce")
        else:
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        artifact_id = _catalog_artifact_id(design)
        bindings[design.clean_group_id] = (artifact_id, design.sha256)
        rows.append(
            {
                "sobol_index": design.sobol_index,
                "clean_group_id": design.clean_group_id,
                "relative_path": path.name,
                "artifact_id": artifact_id,
                "artifact_sha256": design.sha256,
                "file_sha256": _sha256_file(path),
            }
        )
    core = {
        "schema": (
            "gisaxs.posterior_v8.k1_forced_universal_query_catalog_index/v1"
            if forced_balanced
            else "gisaxs.posterior_v8.sobol_universal_query_catalog_index/v1"
        ),
        "pipeline_plan_sha256": shard_plan.sha256,
        "topology_schedule_sha256": shard_plan.topology_schedule.sha256,
        "catalogs": rows,
    }
    index = {**core, "index_sha256": sha256(canonical_json(core).encode()).hexdigest()}
    index_path = directory / "catalog-index.json"
    if index_path.exists():
        observed = json.loads(index_path.read_text(encoding="utf-8"))
        if observed != index:
            raise ValueError("existing topology-query catalog index does not reproduce")
    else:
        _write_json_exclusive(index_path, index)
    return bindings, index


def _search_specs(
    shard_plan: (
        V5FrozenSearchShardPlan
        | V5FormalProductionExecutableShard
        | V5K1BalancedFullSearchExecutableTask
    ),
    parent: V5GroupedDataset,
    recipes: Sequence[object],
    catalog_bindings: Mapping[str, tuple[str, str]],
    calibration: V5CheckedCompatibilityCalibration | None,
) -> tuple[V5UniversalSearchSpec, ...]:
    result = []
    observation_index = 0
    observation_ids = parent.arrays[observation_array("observation_id")]
    formal = isinstance(
        shard_plan,
        (V5FormalProductionExecutableShard, V5K1BalancedFullSearchExecutableTask),
    )
    for recipe_index, (recipe, point, query_design) in enumerate(zip(
        recipes, shard_plan.points, shard_plan.query_designs
    )
    ):
        views = build_v5_observation_data_views(
            recipe,
            (
                shard_plan.view_indices_for_recipe(recipe_index)
                if formal
                else shard_plan.grouped_plan.view_indices
            ),
            split_id=(
                shard_plan.target_split
                if formal
                else shard_plan.grouped_plan.target_split
            ),
        )
        artifact_id, artifact_sha = catalog_bindings[point.clean_group_id]
        for view in views:
            calibrated_threshold = (
                None
                if calibration is None
                else bind_v5_calibrated_observation_threshold(calibration, view)
            )
            observation_id = str(observation_ids[observation_index])
            exact = V5ExactSearchObservation.from_observation_view(
                view, curve_id=observation_id
            )
            context = build_v5_universal_candidate_context(
                view.preprocessed,
                view.uncertainty,
                query_design.topology_queries,
                allowed_topology_ids=query_design.selected_topology_ids,
            )
            result.append(
                V5UniversalSearchSpec(
                    parent_observation_index=observation_index,
                    context=context,
                    exact_observation=exact,
                    query_catalog_artifact_id=artifact_id,
                    query_catalog_artifact_sha256=artifact_sha,
                    calibrated_threshold=calibrated_threshold,
                )
            )
            observation_index += 1
    if observation_index != parent.observation_count:
        raise RuntimeError("replayed views do not cover the grouped parent")
    return tuple(result)


@dataclass(frozen=True, kw_only=True)
class V5FrozenSearchExecution:
    seed_schedule: V5FrozenLocalSobolSchedule
    optimizer_schedule: V5FrozenExactOptimizerSchedule
    protocol: V5FrozenExactSearchProtocol
    launch_source_bundle_sha256: str
    launch_plan_sha256: str
    calibration: V5CheckedCompatibilityCalibration | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.seed_schedule, V5FrozenLocalSobolSchedule):
            raise TypeError("seed_schedule must be V5FrozenLocalSobolSchedule")
        if not isinstance(self.optimizer_schedule, V5FrozenExactOptimizerSchedule):
            raise TypeError("optimizer_schedule must be V5FrozenExactOptimizerSchedule")
        if not isinstance(self.protocol, V5FrozenExactSearchProtocol):
            raise TypeError("protocol must be V5FrozenExactSearchProtocol")
        for value, name in (
            (self.launch_source_bundle_sha256, "launch_source_bundle_sha256"),
            (self.launch_plan_sha256, "launch_plan_sha256"),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        formal = (
            self.protocol.protocol_tier
            == V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED
        )
        if formal:
            if not isinstance(self.calibration, V5CheckedCompatibilityCalibration):
                raise ValueError(
                    "paper/full protocol requires its checked calibration artifact"
                )
            if self.calibration.identity != self.protocol.calibration_identity:
                raise ValueError("protocol escaped its checked calibration identity")
        elif self.calibration is not None:
            raise ValueError("engineering-pilot protocol cannot bind formal calibration")
        expected = (
            (
                self.protocol.exact_forward_call_budget,
                self.seed_schedule.point_count,
                "exact-forward budget",
            ),
            (
                self.protocol.seed_schedule_id,
                self.seed_schedule.schedule_id,
                "seed schedule ID",
            ),
            (
                self.protocol.seed_schedule_sha256,
                self.seed_schedule.sha256,
                "seed schedule SHA-256",
            ),
            (
                self.protocol.optimizer_schedule_id,
                self.optimizer_schedule.schedule_id,
                "optimizer schedule ID",
            ),
            (
                self.protocol.optimizer_schedule_sha256,
                self.optimizer_schedule.sha256,
                "optimizer schedule SHA-256",
            ),
            (
                self.protocol.termination_policy_id,
                self.optimizer_schedule.termination_policy_id,
                "termination policy",
            ),
        )
        for actual, wanted, label in expected:
            if actual != wanted:
                raise ValueError(f"{label} does not bind the execution contracts")


def _assert_execution_host(*, allow_local_smoke: bool) -> None:
    host = socket.gethostname().split(".", 1)[0]
    if host.startswith("max-wgs"):
        raise RuntimeError("frozen exact search is forbidden on the Maxwell login node")
    if not allow_local_smoke and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("frozen exact search must run inside a Slurm worker")


class V5FrozenSearchPipelineError(RuntimeError):
    def __init__(self, message: str, *, failure_audit_path: Path) -> None:
        super().__init__(message)
        self.failure_audit_path = failure_audit_path


SourceBundleFingerprint = Callable[[], str]


def _verify_source_bundle_before_publication(
    verifier: SourceBundleFingerprint | None,
    *,
    expected_sha256: str,
) -> str | None:
    if verifier is None:
        return None
    observed = verifier()
    if observed != expected_sha256:
        raise RuntimeError(
            "worker source bundle changed during exact search; refusing publication"
        )
    return observed


def execute_v5_frozen_search_shard(
    shard_plan: (
        V5FrozenSearchShardPlan
        | V5FormalProductionExecutableShard
        | V5K1BalancedFullSearchExecutableTask
    ),
    execution: V5FrozenSearchExecution,
    output_root: str | os.PathLike[str],
    *,
    allow_local_smoke: bool = False,
    source_bundle_fingerprint: SourceBundleFingerprint | None = None,
    formal_production_authorization: (
        V5FormalProductionSearchAuthorization | None
    ) = None,
) -> dict[str, object]:
    """Execute all branches and publish sidecar last; preserve failures separately."""

    if not isinstance(
        shard_plan,
        (
            V5FrozenSearchShardPlan,
            V5FormalProductionExecutableShard,
            V5K1BalancedFullSearchExecutableTask,
        ),
    ):
        raise TypeError("shard_plan has an invalid type")
    if not isinstance(execution, V5FrozenSearchExecution):
        raise TypeError("execution has an invalid type")
    if not allow_local_smoke and source_bundle_fingerprint is None:
        raise ValueError("Slurm execution requires a pre-publication source verifier")
    if formal_production_authorization is not None and source_bundle_fingerprint is None:
        raise ValueError("formal TRAINING execution requires a source-bundle verifier")
    _assert_execution_host(allow_local_smoke=allow_local_smoke)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    parent_path = root / "grouped-parent.gvd5"
    catalog_directory = root / "query-catalogs"
    executor_directory = root / "executor-evidence"
    sidecar_path = root / "search-supervision.gvd5"
    evidence_receipt_path = evidence_receipt_path_for_sidecar(sidecar_path)
    completion_path = root / "completion.json"
    failure_path = root / "failure.json"
    for path in (sidecar_path, evidence_receipt_path, completion_path, failure_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing pipeline output: {path}")
    if executor_directory.exists():
        raise FileExistsError(
            f"refusing to reuse exact-search evidence directory: {executor_directory}"
        )

    started = datetime.now(timezone.utc).isoformat()
    stage = "parent"
    parent_receipt = None
    try:
        parent, parent_receipt, recipes, parent_reused = (
            materialize_or_verify_v5_frozen_search_parent(shard_plan, parent_path)
        )
        stage = "query_catalogs"
        bindings, catalog_index = _publish_or_verify_query_catalogs(
            shard_plan, catalog_directory
        )
        stage = "search"
        executor_directory.mkdir(exist_ok=False)
        specs = _search_specs(
            shard_plan, parent, recipes, bindings, execution.calibration
        )
        formal = (
            execution.protocol.protocol_tier
            == V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED
        )
        if formal_production_authorization is not None:
            if not formal:
                raise ValueError("engineering search cannot carry a formal authorization")
            label_purpose = V5_SEARCH_LABEL_PURPOSE_TRAINING
            sidecar_prefix = V5_SEARCH_PIPELINE_TRAINING_SIDECAR_PREFIX
        else:
            label_purpose = (
                V5_SEARCH_LABEL_PURPOSE_FORMAL_CONTRACT_SMOKE
                if formal
                else V5_SEARCH_LABEL_PURPOSE_PILOT
            )
            sidecar_prefix = (
                V5_SEARCH_PIPELINE_FORMAL_SIDECAR_PREFIX
                if formal
                else V5_SEARCH_PIPELINE_PILOT_SIDECAR_PREFIX
            )
        label_binding = build_v5_search_label_binding(
            protocol=execution.protocol,
            seed_schedule_sha256=execution.seed_schedule.sha256,
            optimizer_schedule_sha256=execution.optimizer_schedule.sha256,
            launch_source_bundle_sha256=execution.launch_source_bundle_sha256,
            launch_plan_sha256=execution.launch_plan_sha256,
            shard_plan_sha256=shard_plan.sha256,
            label_purpose=label_purpose,
            formal_production_authorization=formal_production_authorization,
        )
        runner = V5FrozenExactSearchExecutor(
            output_directory=executor_directory,
            seed_schedule=execution.seed_schedule,
            optimizer_schedule=execution.optimizer_schedule,
        )
        sidecar = collect_v5_search_supervision_sidecar(
            parent_path,
            specs,
            sidecar_id=(
                f"{sidecar_prefix}{label_binding['label_binding_sha256']}"
            ),
            protocol=execution.protocol,
            runner=runner,
        )
        stage = "pre_sidecar_source_fingerprint"
        _verify_source_bundle_before_publication(
            source_bundle_fingerprint,
            expected_sha256=execution.launch_source_bundle_sha256,
        )
        stage = "sidecar_publish"
        sidecar_receipt = write_v5_search_supervision_sidecar(sidecar, sidecar_path)
        stage = "pre_receipt_source_fingerprint"
        _verify_source_bundle_before_publication(
            source_bundle_fingerprint,
            expected_sha256=execution.launch_source_bundle_sha256,
        )
        stage = "task_bound_evidence_receipt"
        evidence_receipt = audit_and_write_v5_search_evidence_receipt(
            parent_dataset_path=parent_path,
            sidecar_path=sidecar_path,
            specs=specs,
            protocol=execution.protocol,
            seed_schedule=execution.seed_schedule,
            optimizer_schedule=execution.optimizer_schedule,
            executor_directory=executor_directory,
            output_path=evidence_receipt_path,
            launch_source_bundle_sha256=execution.launch_source_bundle_sha256,
            launch_plan_sha256=execution.launch_plan_sha256,
            shard_plan_sha256=shard_plan.sha256,
            label_purpose=label_purpose,
            pre_publish_guard=lambda: _verify_source_bundle_before_publication(
                source_bundle_fingerprint,
                expected_sha256=execution.launch_source_bundle_sha256,
            ),
            formal_production_authorization=formal_production_authorization,
        )
        completion_core = {
            "schema": V5_FROZEN_SEARCH_PIPELINE_SCHEMA,
            "version": V5_FROZEN_SEARCH_PIPELINE_VERSION,
            "status": "complete",
            "started_at_utc": started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline_plan_sha256": shard_plan.sha256,
            "launch_plan_sha256": execution.launch_plan_sha256,
            "scientific_scope": (
                V5_SEARCH_PIPELINE_TRAINING_SCOPE
                if formal_production_authorization is not None
                else V5_SEARCH_PIPELINE_FORMAL_SCOPE
                if formal
                else V5_SEARCH_PIPELINE_SCOPE
            ),
            "label_purpose": label_purpose,
            "full_training_label_claimed": evidence_receipt.full_training_eligible,
            "authorized_consumer_role": (
                None
                if formal_production_authorization is None
                else formal_production_authorization.consumer_role
            ),
            "label_binding": label_binding,
            "parent_reused_after_strict_replay": parent_reused,
            "parent": {
                "path": str(parent_path),
                "artifact_sha256": parent_receipt.artifact_sha256,
                "manifest_sha256": parent_receipt.manifest_sha256,
            },
            "query_catalog_index": catalog_index,
            "seed_schedule_sha256": execution.seed_schedule.sha256,
            "optimizer_schedule_sha256": execution.optimizer_schedule.sha256,
            "protocol_sha256": execution.protocol.sha256,
            "sidecar": {
                "path": str(sidecar_path),
                "artifact_sha256": sidecar_receipt.artifact_sha256,
                "manifest_sha256": sidecar_receipt.manifest_sha256,
                "queries": sidecar.query_count,
                "branches": sidecar.branch_count,
                "outcomes": dict(sidecar.manifest["counts"]),
            },
            "task_bound_evidence_receipt": {
                "path": str(evidence_receipt.path),
                "file_sha256": evidence_receipt.file_sha256,
                "receipt_sha256": evidence_receipt.manifest["receipt_sha256"],
                "full_training_eligible": evidence_receipt.full_training_eligible,
            },
            "heavy_compute_performed_on_login_node": False,
            "publication_order": (
                "executor_evidence_then_atomic_sidecar_then_task_bound_evidence_"
                "receipt_then_completion"
            ),
        }
        completion = {
            **completion_core,
            "completion_sha256": sha256(
                canonical_json(completion_core).encode("utf-8")
            ).hexdigest(),
        }
        _write_json_exclusive(completion_path, completion)
        return completion
    except (Exception, KeyboardInterrupt) as exc:
        failure_core = {
            "schema": V5_FROZEN_SEARCH_PIPELINE_SCHEMA,
            "version": V5_FROZEN_SEARCH_PIPELINE_VERSION,
            "status": "failed",
            "started_at_utc": started,
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "failed_stage": stage,
            "exception_type": type(exc).__name__,
            "message": str(exc)[:2000],
            "pipeline_plan_sha256": shard_plan.sha256,
            "launch_plan_sha256": execution.launch_plan_sha256,
            "protocol_sha256": execution.protocol.sha256,
            "parent_artifact_sha256": (
                None if parent_receipt is None else parent_receipt.artifact_sha256
            ),
            "sidecar_published": sidecar_path.exists(),
            "task_bound_evidence_receipt_published": evidence_receipt_path.exists(),
            "completion_published": completion_path.exists(),
            "partial_executor_evidence_retained": executor_directory.exists(),
            "outputs_are_never_overwritten": True,
        }
        failure = {
            **failure_core,
            "failure_sha256": sha256(
                canonical_json(failure_core).encode("utf-8")
            ).hexdigest(),
        }
        if not failure_path.exists():
            _write_json_exclusive(failure_path, failure)
        raise V5FrozenSearchPipelineError(
            f"frozen search failed during {stage}; audit preserved at {failure_path}",
            failure_audit_path=failure_path,
        ) from exc


__all__ = [
    "V5FrozenSearchExecution",
    "V5FrozenSearchPipelineError",
    "V5FrozenSearchShardPlan",
    "V5SelectedTopologySearchSchedule",
    "V5_FROZEN_SEARCH_PIPELINE_SCHEMA",
    "V5_FROZEN_SEARCH_PIPELINE_VERSION",
    "V5_SEARCH_PIPELINE_ALLOWED_SPLITS",
    "V5_SEARCH_PIPELINE_FORMAL_SCOPE",
    "V5_SEARCH_PIPELINE_FORMAL_SIDECAR_PREFIX",
    "V5_SEARCH_PIPELINE_PILOT_SIDECAR_PREFIX",
    "V5_SEARCH_PIPELINE_SCOPE",
    "V5_SEARCH_PIPELINE_TRAINING_SCOPE",
    "V5_TOPOLOGY_SEARCH_SCHEDULE_SCHEMA",
    "V5_TOPOLOGY_SEARCH_SCHEDULE_VERSION",
    "execute_v5_frozen_search_shard",
    "materialize_or_verify_v5_frozen_search_parent",
    "plan_v5_frozen_search_shard",
]
