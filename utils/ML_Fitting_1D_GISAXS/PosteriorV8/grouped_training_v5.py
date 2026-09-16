"""Reproducible V5.2 two-stage recipe-macro training artifacts.

Warmup reads known positives from every grouped parent shard.  Full training
reads only cryptographically paired frozen-search sidecars and requires both
development splits to contain completed positives and negatives with no
unverified rows.  Sidecar-labeled parent subsets may be smaller than warmup.
Every completed full epoch is retained as an immutable checkpoint candidate;
paper checkpoint selection remains the responsibility of an external
exact-budget evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Callable, Mapping, Sequence
from uuid import uuid4

import numpy as np
import tensorflow as tf

from .grouped_artifact_v5 import canonical_json
from .grouped_training_data_v5 import (
    V5_GROUPED_TRAINING_SEMANTICS,
    V5GroupedTrainingConfig,
)
from .grouped_training_inventory_v5 import (
    resolved_artifact_paths,
    training_input_manifest_audit,
)
from .grouped_training_shards_v5 import (
    V5GroupedTrainingAudit,
    inspect_v5_grouped_training,
)
from .job_local_input_capability_v5 import (
    V5JobLocalInputCapability,
    _job_local_capability_post_validation_payload,
    _validate_job_local_input_capability,
)
from .model_v5 import (
    build_branch_conditioned_proposal_model,
    validate_model_v5_graph_contract,
)
from .model_v5_contract import model_v5_contract_payload
from .training_objective_v5 import (
    V5CandidateObjectiveConfig,
    compute_v5_candidate_training_objective,
)


V5_GROUPED_TRAINER_SCHEMA = "gisaxs.posterior_v8.grouped_training_run/v4"
V5_GROUPED_TRAINER_VERSION = (
    "v5_2_policy_bound_center_aligned_mass_coverage_checkpoint_inventory_v4"
)
RUN_PLAN_FILE = "run_plan.json"
HISTORY_FILE = "history.json"
RESULT_MANIFEST_FILE = "result_manifest.json"
FAILURE_FILE = "failure.json"
BEST_MODEL_FILE = "best.keras"
LAST_MODEL_FILE = "last.keras"
FULL_CHECKPOINT_DIRECTORY = "checkpoints"
FULL_CHECKPOINT_SELECTION_STATUS = "pending_external_exact_budget_auc_and_ttfc_selection"
BEST_MODEL_ROLE = "validation_objective_convenience_not_paper_selected"
_COUNT_METRICS = {
    "candidate_count",
    "verified_count",
    "compatible_found_count",
    "no_compatible_found_within_budget_count",
    "unverified_count",
    "local_target_count",
    "local_mdn_contributing_count",
    "local_coverage_contributing_count",
    "pairwise_ranking_pair_count",
    "pairwise_ranking_recipe_count",
    "search_yield_recipe_count",
    "local_mdn_recipe_count",
    "local_coverage_recipe_count",
}


@dataclass(frozen=True)
class V5GroupedTrainingResult:
    output_dir: Path
    best_model_path: Path
    last_model_path: Path
    history_path: Path
    manifest_path: Path
    best_epoch: int
    best_validation_loss: float
    full_checkpoint_paths: tuple[Path, ...]
    checkpoint_selection_status: str
    paper_model_eligible: bool


def _source_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    paths = {
        Path(module.__file__).resolve()
        for module in tuple(sys.modules.values())
        if getattr(module, "__file__", None)
        and Path(module.__file__).resolve().is_relative_to(root)
        and Path(module.__file__).suffix == ".py"
    }
    paths.add(root / "slurm" / "v5_grouped_train_gpu4.sbatch")
    paths = sorted(paths)
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("the V5.2 training source bundle is incomplete")
    return {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest() for path in paths}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_weights_sha256(model: tf.keras.Model) -> str:
    """Hash numerical state independently of nondeterministic archive metadata."""

    digest = sha256()
    for variable in model.weights:
        value = np.ascontiguousarray(variable.numpy())
        digest.update(getattr(variable, "path", variable.name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object], *, exclusive: bool = False) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive and path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_model(
    model: tf.keras.Model,
    path: Path,
    *,
    exclusive: bool = False,
) -> str:
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.keras")
    try:
        model.save(temporary)
        if exclusive:
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise FileExistsError(f"refusing to overwrite {path}") from None
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _file_sha256(path)


def _full_checkpoint_relative_path(*, epoch: int, phase_epoch: int) -> Path:
    return Path(FULL_CHECKPOINT_DIRECTORY) / (
        f"full_epoch_{phase_epoch:04d}_global_epoch_{epoch:04d}.keras"
    )


def _checkpoint_selection_payload() -> dict[str, object]:
    return {
        "status": FULL_CHECKPOINT_SELECTION_STATUS,
        "selector_owner": "external_paper_exact_budget_evaluator",
        "paper_selected_checkpoint": None,
        "best_model_role": BEST_MODEL_ROLE,
        "validation_objective_alone_selects_paper_checkpoint": False,
    }


def _paper_model_status(
    config: V5GroupedTrainingConfig,
    audit: V5GroupedTrainingAudit,
    *,
    full_checkpoint_count: int,
) -> dict[str, object]:
    full_training_completed = bool(
        config.full_epochs > 0 and full_checkpoint_count == config.full_epochs
    )
    expansion = audit.sidecar_expansion_safety
    expansion_allows_paper_claim = bool(
        expansion is not None and expansion.get("paper_claim_allowed") is True
    )
    candidate_inventory_eligible = bool(
        full_training_completed
        and audit.full_stage_permitted
        and expansion_allows_paper_claim
        and config.pairwise_ranking_weight > 0.0
        and config.local_coverage_weight > 0.0
        and config.operational_top_l_alignment_weight > 0.0
    )
    reasons = []
    if config.full_epochs == 0:
        reasons.append("full_epochs_is_zero_warmup_only_engineering_run")
    elif not full_training_completed:
        reasons.append("full_epoch_checkpoint_inventory_incomplete")
    if config.full_epochs > 0 and not expansion_allows_paper_claim:
        reasons.append("sidecar_expansion_gate_does_not_allow_paper_claim")
    if config.full_epochs > 0 and not audit.full_stage_permitted:
        reasons.append("full_search_supervision_stage_not_permitted")
    if config.full_epochs > 0 and config.pairwise_ranking_weight == 0.0:
        reasons.append("full_pairwise_ranking_objective_disabled")
    if config.full_epochs > 0 and config.local_coverage_weight == 0.0:
        reasons.append("full_mass_aware_coverage_objective_disabled")
    if (
        config.full_epochs > 0
        and config.operational_top_l_alignment_weight == 0.0
    ):
        reasons.append("full_operational_top_l_alignment_objective_disabled")
    reasons.append("paper_checkpoint_selection_pending")
    return {
        "run_scope": (
            "engineering_warmup_only"
            if config.full_epochs == 0
            else "full_search_supervision_checkpoint_candidates"
        ),
        "full_training_requested": config.full_epochs > 0,
        "full_training_completed": full_training_completed,
        "full_checkpoint_count": full_checkpoint_count,
        "expected_full_checkpoint_count": config.full_epochs,
        "sidecar_expansion_paper_claim_allowed": expansion_allows_paper_claim,
        "paper_checkpoint_candidates_eligible_for_external_selection": (
            candidate_inventory_eligible
        ),
        "operational_ranking_eligible": candidate_inventory_eligible,
        "warmup_only_ranking_eligible": False,
        "checkpoint_selection_status": FULL_CHECKPOINT_SELECTION_STATUS,
        "paper_model_eligible": False,
        "paper_model_ineligibility_reasons": reasons,
    }


def _runtime_payload(strategy: tf.distribute.Strategy) -> dict[str, object]:
    slurm_names = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_CLUSTER_NAME",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_NODELIST",
        "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS",
        "SLURM_GPUS_ON_NODE",
        "CUDA_VISIBLE_DEVICES",
    )
    physical_gpus = tf.config.list_physical_devices("GPU")
    return {
        "utc_started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "strategy": type(strategy).__name__,
        "replicas": int(strategy.num_replicas_in_sync),
        "logical_gpus": [value.name for value in tf.config.list_logical_devices("GPU")],
        "physical_gpus": [
            {
                "name": value.name,
                "details": {
                    str(name): str(detail)
                    for name, detail in tf.config.experimental.get_device_details(value).items()
                },
            }
            for value in physical_gpus
        ],
        "tensorflow_build": {
            str(name): str(value) for name, value in tf.sysconfig.get_build_info().items()
        },
        "slurm": {name: os.environ[name] for name in slurm_names if name in os.environ},
    }


def _require_dust_paths_under_slurm(
    datasets: Sequence[Path],
    output: Path,
    job_local_input_capability: V5JobLocalInputCapability | None = None,
) -> None:
    if "SLURM_JOB_ID" not in os.environ:
        if job_local_input_capability is not None:
            raise RuntimeError(
                "job-local input capability is valid only inside its Slurm allocation"
            )
        return
    dust = Path("/data/dust/user/zhaiyufe")
    try:
        output.relative_to(dust)
    except ValueError as exc:
        raise ValueError(f"Maxwell Slurm output must be under {dust}/") from exc
    if job_local_input_capability is not None:
        _validate_job_local_input_capability(
            datasets,
            job_local_input_capability,
            phase="pre_training",
        )
        return
    for path in datasets:
        try:
            path.relative_to(dust)
        except ValueError as exc:
            raise ValueError(f"Maxwell Slurm dataset must be under {dust}/") from exc


def _distributed_step_factory(strategy, model, optimizer, objective, clip_norm):
    replicas = tf.cast(strategy.num_replicas_in_sync, tf.float32)
    mixed = isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer)

    def replica_step(batch):
        inputs, labels = batch
        with tf.GradientTape() as tape:
            metrics = compute_v5_candidate_training_objective(
                model(inputs, training=True), labels, objective
            )
            gradient_loss = metrics["loss"] / replicas
            tape_loss = optimizer.get_scaled_loss(gradient_loss) if mixed else gradient_loss
        gradients = tape.gradient(tape_loss, model.trainable_variables)
        if mixed:
            gradients = optimizer.get_unscaled_gradients(gradients)
        connected = [
            (gradient, variable)
            for gradient, variable in zip(gradients, model.trainable_variables)
            if gradient is not None
        ]
        if not connected:
            raise RuntimeError("V5.2 objective is disconnected from all trainable variables")
        gradient_values = [value for value, _ in connected]
        if not mixed:
            checks = [
                tf.debugging.assert_all_finite(
                    value.values if isinstance(value, tf.IndexedSlices) else value,
                    "V5.2 proposal gradient contains NaN/Inf",
                )
                for value in gradient_values
            ]
        else:
            checks = []
        before = tf.identity(optimizer.iterations)
        with tf.control_dependencies(checks):
            gradient_values, _ = tf.clip_by_global_norm(gradient_values, clip_norm)
            optimizer.apply_gradients(
                (gradient, variable) for gradient, (_, variable) in zip(gradient_values, connected)
            )
        result = {name: tf.identity(value) for name, value in metrics.items()}
        result["optimizer_update_applied"] = tf.cast(optimizer.iterations > before, tf.float32)
        return result

    @tf.function(reduce_retracing=True)
    def distributed_step(batch):
        values = strategy.run(replica_step, args=(batch,))
        result = {}
        for name, value in values.items():
            operation = (
                tf.distribute.ReduceOp.SUM
                if name in _COUNT_METRICS
                else tf.distribute.ReduceOp.MEAN
            )
            result[name] = strategy.reduce(operation, value, axis=None)
        return result

    return distributed_step


def _metric_values(metrics: Mapping[str, tf.Tensor]) -> dict[str, float]:
    return {name: float(value.numpy()) for name, value in metrics.items()}


def _average_metric_records(
    records: Sequence[tuple[Mapping[str, float], int]],
) -> dict[str, float]:
    total_recipes = sum(recipe_count for _, recipe_count in records)
    result = {}
    for name in records[0][0]:
        if name in _COUNT_METRICS:
            result[name] = float(sum(values[name] for values, _ in records))
        else:
            result[name] = float(
                sum(values[name] * recipes for values, recipes in records) / total_recipes
            )
    return result


def _validation_epoch(adapter, model, recipes, phase, objective, recipe_batch_size):
    records = []
    for start in range(0, len(recipes), recipe_batch_size):
        selected = recipes[start : start + recipe_batch_size]
        inputs, labels = adapter.tensor_batch(selected, phase=phase)
        metrics = compute_v5_candidate_training_objective(
            model(inputs, training=False), labels, objective
        )
        records.append((_metric_values(metrics), len(selected)))
    return _average_metric_records(records)


def _train_epoch(
    *,
    adapter,
    strategy,
    step_function,
    recipes,
    phase,
    epoch_seed,
    audit,
):
    order = np.array(recipes, copy=True)
    np.random.default_rng(epoch_seed).shuffle(order)
    selected_steps = (
        audit.selected_steps_per_epoch
        if phase == "warmup"
        else audit.full_selected_steps_per_epoch
    )
    used = selected_steps * audit.global_recipes_per_step
    order = order[:used]
    records = []
    for step in range(selected_steps):
        selected = order[
            step * audit.global_recipes_per_step : (step + 1) * audit.global_recipes_per_step
        ]
        chunks = np.split(selected, audit.replicas)
        local = [adapter.tensor_batch(chunk, phase=phase) for chunk in chunks]

        def value_fn(context):
            return local[int(context.replica_id_in_sync_group)]

        distributed = strategy.experimental_distribute_values_from_function(value_fn)
        metrics = _metric_values(step_function(distributed))
        records.append((metrics, audit.global_recipes_per_step))
    return _average_metric_records(records), int(used)


def train_v5_grouped_model(
    train_dataset_paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    validation_dataset_paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
    config: V5GroupedTrainingConfig = V5GroupedTrainingConfig(),
    *,
    train_sidecar_paths: (
        str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None
    ) = None,
    validation_sidecar_paths: (
        str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None
    ) = None,
    strategy: tf.distribute.Strategy | None = None,
    model_builder: Callable[..., tf.keras.Model] = build_branch_conditioned_proposal_model,
    job_local_input_capability: V5JobLocalInputCapability | None = None,
) -> V5GroupedTrainingResult:
    """Train in an exclusive run directory; existing outputs are never reused."""

    if not isinstance(config, V5GroupedTrainingConfig):
        raise TypeError("config must be V5GroupedTrainingConfig")
    if socket.gethostname().split(".", 1)[0].startswith(("max-wgs", "max-fs-display")):
        raise RuntimeError("training is forbidden on the Maxwell login node; submit through Slurm")
    train_sources = resolved_artifact_paths(train_dataset_paths)
    validation_sources = resolved_artifact_paths(validation_dataset_paths)
    train_sidecar_sources = resolved_artifact_paths(train_sidecar_paths)
    validation_sidecar_sources = resolved_artifact_paths(validation_sidecar_paths)
    sources_all = (
        *train_sources,
        *validation_sources,
        *train_sidecar_sources,
        *validation_sidecar_sources,
    )
    output = Path(output_dir).resolve()
    _require_dust_paths_under_slurm(
        sources_all, output, job_local_input_capability
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing training output: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"training output parent does not exist: {output.parent}")
    if config.deterministic_ops:
        tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(config.seed)
    distribution = tf.distribute.MirroredStrategy() if strategy is None else strategy
    adapter, audit = inspect_v5_grouped_training(
        train_sources,
        validation_sources,
        config=config,
        train_sidecar_paths=train_sidecar_sources or None,
        validation_sidecar_paths=validation_sidecar_sources or None,
        replicas=int(distribution.num_replicas_in_sync),
    )
    runtime = _runtime_payload(distribution)
    sources = _source_sha256()
    config_payload = config.audit_payload()
    phase_objectives = {
        "warmup": V5CandidateObjectiveConfig(
            search_yield_weight=0.0,
            pairwise_ranking_weight=0.0,
            local_mdn_weight=config.local_mdn_weight,
            local_coverage_weight=config.local_coverage_weight,
            operational_top_l_alignment_weight=(
                config.operational_top_l_alignment_weight
            ),
            logistic_epsilon=config.logistic_epsilon,
            local_coverage_temperature=config.local_coverage_temperature,
            operational_hit_rms_threshold=config.operational_hit_rms_threshold,
            operational_duplicate_rms_threshold=(
                config.operational_duplicate_rms_threshold
            ),
            proposal_execution_policy_sha256=(
                config.proposal_execution_policy_sha256
            ),
        ),
        "full": V5CandidateObjectiveConfig(
            search_yield_weight=config.search_yield_weight,
            pairwise_ranking_weight=config.pairwise_ranking_weight,
            local_mdn_weight=config.local_mdn_weight,
            local_coverage_weight=config.local_coverage_weight,
            operational_top_l_alignment_weight=(
                config.operational_top_l_alignment_weight
            ),
            logistic_epsilon=config.logistic_epsilon,
            local_coverage_temperature=config.local_coverage_temperature,
            operational_hit_rms_threshold=config.operational_hit_rms_threshold,
            operational_duplicate_rms_threshold=(
                config.operational_duplicate_rms_threshold
            ),
            proposal_execution_policy_sha256=(
                config.proposal_execution_policy_sha256
            ),
        ),
    }
    plan_core = {
        "schema": V5_GROUPED_TRAINER_SCHEMA,
        "version": V5_GROUPED_TRAINER_VERSION,
        "config": config_payload,
        "config_sha256": sha256(canonical_json(config_payload).encode("utf-8")).hexdigest(),
        "input_manifests": training_input_manifest_audit(adapter.shards),
        "training_audit": audit.audit_payload(),
        "model_contract": model_v5_contract_payload(),
        "objectives": {name: value.audit_payload() for name, value in phase_objectives.items()},
        "checkpoint_contract": {
            "full_epoch_checkpoint_directory": FULL_CHECKPOINT_DIRECTORY,
            "full_epoch_checkpoint_retention": "every_completed_full_epoch_exclusive",
            "checkpoint_selection": _checkpoint_selection_payload(),
        },
        "source_sha256": sources,
        "runtime": runtime,
        "job_local_input_capability": (
            None
            if job_local_input_capability is None
            else job_local_input_capability.audit_payload()
        ),
    }
    plan = {
        **plan_core,
        "plan_sha256": sha256(canonical_json(plan_core).encode("utf-8")).hexdigest(),
    }
    output.mkdir(exist_ok=False)
    _atomic_json(output / RUN_PLAN_FILE, plan, exclusive=True)
    history: dict[str, object] = {
        "schema": V5_GROUPED_TRAINER_SCHEMA,
        "version": V5_GROUPED_TRAINER_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "status": "running",
        "epochs": [],
        "full_checkpoints": [],
        "best_epoch": None,
        "best_validation_loss": None,
        "checkpoint_selection": _checkpoint_selection_payload(),
        "paper_model_status": _paper_model_status(
            config,
            audit,
            full_checkpoint_count=0,
        ),
    }
    _atomic_json(output / HISTORY_FILE, history, exclusive=True)
    previous_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy(
        "mixed_float16" if config.mixed_precision else "float32"
    )
    try:
        with distribution.scope():
            model = model_builder(
                max_points=audit.max_points,
                width=config.width,
                encoder_blocks=config.encoder_blocks,
                mixture_components=config.mixture_components,
            )
            validate_model_v5_graph_contract(model)
            base_optimizer = tf.keras.optimizers.Adam(config.learning_rate)
            optimizer = (
                tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
                if config.mixed_precision
                else base_optimizer
            )
            if hasattr(optimizer, "build"):
                optimizer.build(model.trainable_variables)
        steps = {
            name: _distributed_step_factory(
                distribution,
                model,
                optimizer,
                objective,
                config.gradient_clip_norm,
            )
            for name, objective in phase_objectives.items()
        }
        phases = (("warmup", config.warmup_epochs), ("full", config.full_epochs))
        epoch_number = 0
        best_phase = "full" if config.full_epochs else "warmup"
        best_loss = np.inf
        for phase_index, (phase, epoch_count) in enumerate(phases):
            train_recipes = (
                adapter.train_recipes
                if phase == "warmup"
                else adapter.full_train_recipes
            )
            validation_recipes = (
                adapter.validation_recipes
                if phase == "warmup"
                else adapter.full_validation_recipes
            )
            for phase_epoch in range(epoch_count):
                epoch_number += 1
                started = time.monotonic()
                train_metrics, used_recipes = _train_epoch(
                    adapter=adapter,
                    strategy=distribution,
                    step_function=steps[phase],
                    recipes=train_recipes,
                    phase=phase,
                    epoch_seed=config.seed + phase_index * 1_000_003 + phase_epoch,
                    audit=audit,
                )
                validation_metrics = _validation_epoch(
                    adapter,
                    model,
                    validation_recipes,
                    phase,
                    phase_objectives[phase],
                    config.validation_recipes_per_batch,
                )
                last_hash = _atomic_save_model(model, output / LAST_MODEL_FILE)
                last_weights_hash = _model_weights_sha256(model)
                checkpoint_record = None
                if phase == "full":
                    checkpoint_directory = output / FULL_CHECKPOINT_DIRECTORY
                    if not checkpoint_directory.exists():
                        checkpoint_directory.mkdir(exist_ok=False)
                    checkpoint_relative = _full_checkpoint_relative_path(
                        epoch=epoch_number,
                        phase_epoch=phase_epoch + 1,
                    )
                    checkpoint_path = output / checkpoint_relative
                    checkpoint_record = {
                        "epoch": epoch_number,
                        "phase_epoch": phase_epoch + 1,
                        "relative_path": checkpoint_relative.as_posix(),
                        "file_sha256": _atomic_save_model(
                            model,
                            checkpoint_path,
                            exclusive=True,
                        ),
                        "weights_sha256": last_weights_hash,
                    }
                    history["full_checkpoints"].append(checkpoint_record)
                if phase == best_phase and validation_metrics["loss"] < best_loss:
                    best_loss = validation_metrics["loss"]
                    best_hash = _atomic_save_model(model, output / BEST_MODEL_FILE)
                    history["best_epoch"] = epoch_number
                    history["best_validation_loss"] = best_loss
                    history["best_model_sha256"] = best_hash
                    history["best_model_weights_sha256"] = last_weights_hash
                record = {
                    "epoch": epoch_number,
                    "phase": phase,
                    "data_source": dict(audit.phase_data_sources[phase]),
                    "phase_epoch": phase_epoch + 1,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "train_recipes_used": used_recipes,
                    "train_recipes_not_used": int(train_recipes.size) - used_recipes,
                    "validation_recipes_used": int(validation_recipes.size),
                    "optimizer_iterations": int(optimizer.iterations.numpy()),
                    "last_model_sha256": last_hash,
                    "last_model_weights_sha256": last_weights_hash,
                    "seconds": time.monotonic() - started,
                }
                if checkpoint_record is not None:
                    record["full_checkpoint"] = checkpoint_record
                history["epochs"].append(record)
                history["paper_model_status"] = _paper_model_status(
                    config,
                    audit,
                    full_checkpoint_count=len(history["full_checkpoints"]),
                )
                _atomic_json(output / HISTORY_FILE, history)
        history["status"] = "complete"
        history["utc_completed"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _atomic_json(output / HISTORY_FILE, history)
        best_path, last_path = output / BEST_MODEL_FILE, output / LAST_MODEL_FILE
        loaded_best = validate_model_v5_graph_contract(
            tf.keras.models.load_model(best_path, compile=False)
        )
        loaded_last = validate_model_v5_graph_contract(
            tf.keras.models.load_model(last_path, compile=False)
        )
        if _model_weights_sha256(loaded_best) != history["best_model_weights_sha256"]:
            raise RuntimeError("serialized best-model weights changed during round trip")
        if _model_weights_sha256(loaded_last) != history["epochs"][-1]["last_model_weights_sha256"]:
            raise RuntimeError("serialized last-model weights changed during round trip")
        for artifact in adapter.input_artifact_audit():
            if _file_sha256(Path(str(artifact["path"]))) != artifact["artifact_sha256"]:
                raise RuntimeError("an input parent/sidecar artifact changed during training")
        job_local_post_validation = None
        if job_local_input_capability is not None:
            _validate_job_local_input_capability(
                sources_all,
                job_local_input_capability,
                phase="post_training",
            )
            job_local_post_validation = (
                _job_local_capability_post_validation_payload(
                    job_local_input_capability
                )
            )
        for checkpoint in history["full_checkpoints"]:
            checkpoint_path = output / str(checkpoint["relative_path"])
            if _file_sha256(checkpoint_path) != checkpoint["file_sha256"]:
                raise RuntimeError("a retained full-epoch checkpoint changed during training")
        history_hash = _file_sha256(output / HISTORY_FILE)
        paper_model_status = _paper_model_status(
            config,
            audit,
            full_checkpoint_count=len(history["full_checkpoints"]),
        )
        result_core = {
            "schema": V5_GROUPED_TRAINER_SCHEMA,
            "version": V5_GROUPED_TRAINER_VERSION,
            "plan_sha256": plan["plan_sha256"],
            "input_artifacts": [dict(value) for value in adapter.input_artifact_audit()],
            "training_audit": audit.audit_payload(),
            "phase_data_sources": {
                phase: dict(roles)
                for phase, roles in audit.phase_data_sources.items()
            },
            "best_epoch": history["best_epoch"],
            "best_validation_loss": history["best_validation_loss"],
            "best_model_sha256": _file_sha256(best_path),
            "best_model_weights_sha256": _model_weights_sha256(loaded_best),
            "best_model_role": BEST_MODEL_ROLE,
            "last_model_sha256": _file_sha256(last_path),
            "last_model_weights_sha256": _model_weights_sha256(loaded_last),
            "full_checkpoints": [dict(value) for value in history["full_checkpoints"]],
            "checkpoint_selection": _checkpoint_selection_payload(),
            "paper_model_status": paper_model_status,
            "history_sha256": history_hash,
            "source_sha256": sources,
            "slurm": runtime["slurm"],
            "job_local_input_capability": job_local_post_validation,
            "status": "complete",
        }
        result = {
            **result_core,
            "result_sha256": sha256(canonical_json(result_core).encode("utf-8")).hexdigest(),
        }
        _atomic_json(output / RESULT_MANIFEST_FILE, result, exclusive=True)
        return V5GroupedTrainingResult(
            output_dir=output,
            best_model_path=best_path,
            last_model_path=last_path,
            history_path=output / HISTORY_FILE,
            manifest_path=output / RESULT_MANIFEST_FILE,
            best_epoch=int(history["best_epoch"]),
            best_validation_loss=float(history["best_validation_loss"]),
            full_checkpoint_paths=tuple(
                output / str(value["relative_path"])
                for value in history["full_checkpoints"]
            ),
            checkpoint_selection_status=FULL_CHECKPOINT_SELECTION_STATUS,
            paper_model_eligible=False,
        )
    except Exception as exc:
        failure = {
            "schema": V5_GROUPED_TRAINER_SCHEMA,
            "version": V5_GROUPED_TRAINER_VERSION,
            "plan_sha256": plan["plan_sha256"],
            "type": type(exc).__name__,
            "message": str(exc),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            _atomic_json(
                output / FAILURE_FILE, failure, exclusive=not (output / FAILURE_FILE).exists()
            )
        except Exception:
            pass
        raise
    finally:
        tf.keras.mixed_precision.set_global_policy(previous_policy)


__all__ = [
    "BEST_MODEL_FILE",
    "BEST_MODEL_ROLE",
    "FULL_CHECKPOINT_DIRECTORY",
    "FULL_CHECKPOINT_SELECTION_STATUS",
    "HISTORY_FILE",
    "LAST_MODEL_FILE",
    "RESULT_MANIFEST_FILE",
    "RUN_PLAN_FILE",
    "V5_GROUPED_TRAINER_SCHEMA",
    "V5_GROUPED_TRAINER_VERSION",
    "V5_GROUPED_TRAINING_SEMANTICS",
    "V5GroupedTrainingAudit",
    "V5GroupedTrainingConfig",
    "V5GroupedTrainingResult",
    "V5JobLocalInputCapability",
    "inspect_v5_grouped_training",
    "train_v5_grouped_model",
]
