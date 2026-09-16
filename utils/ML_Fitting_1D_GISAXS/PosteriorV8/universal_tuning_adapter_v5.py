"""Checkpoint-backed exact-budget traces; no gradient or acceptance authority."""

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType

from .evaluation import EvaluationThresholds, ObservedCurve
from .calibrated_search_threshold_v5 import (
    bind_v5_calibrated_observation_threshold, read_v5_checked_compatibility_calibration,
)
from .search_supervision_contract_v5 import V5ExactSearchObservation, observed_curve_sha256
from .grouped_artifact_v5 import canonical_json
from .k1_phase_b_contract_v5 import model_weights_sha256
from .k1_staging_files_v5 import read_only_identity
from .paper_budget_evaluator_v5 import V5FrozenReferenceSet
from .paper_checkpoint_selector_v5 import build_v5_checkpoint_evaluation_method_binding
from .tuning_checkpoint_model_v5 import load_v5_verified_tuning_checkpoint
from .tuning_checkpoint_runtime_v5 import V5RetainedFullCheckpoint, V5TuningExactTraceRunResult
from .tuning_search_recorder_v5 import V5TuningSearchRecorder
from .universal_inference_contract_v5 import (
    V5UniversalInferenceBudget, V5_UNIVERSAL_INFERENCE_VERSION,
)
from .universal_inference_v5 import run_v5_universal_one_click_inference
from .universal_query_v5 import V5UniversalCandidateContext, validate_v5_universal_curve_alignment


V5_UNIVERSAL_TUNING_PROTOCOL_ID = "checkpoint_universal_full_budget_snapshot_v1"
V5_CALIBRATED_UNIVERSAL_TUNING_PROTOCOL_ID = "checkpoint_universal_calibrated_query_snapshot_v2"


class V5UniversalCheckpointTraceRunner:
    """Adapt real sealed checkpoints to the common tuning runtime callback.

    The caller owns worker placement, sealed query materialization, calibration,
    cohort isolation and production authorization. This adapter does not grant
    any of those claims. Reference parameter values never enter inference.
    """

    def __init__(self, *, queries, thresholds, budget, source_summary_artifact_sha256,
                 calibration=None, observation_views=None, calibration_path=None):
        if type(thresholds) is not EvaluationThresholds or type(budget) is not V5UniversalInferenceBudget:
            raise TypeError("expected typed thresholds and universal budget")
        maximum = budget.forward_evaluation_limit
        if maximum < 1 or budget.fallback_attempt_limit < maximum or budget.sobol_seeds_per_branch < maximum:
            raise ValueError("full-budget protocol requires an explicit sufficient Sobol horizon")
        if (not isinstance(source_summary_artifact_sha256, str)
                or len(source_summary_artifact_sha256) != 64
                or any(c not in "0123456789abcdef" for c in source_summary_artifact_sha256)):
            raise ValueError("expected explicit source summary SHA-256")
        resolved = {}
        for query_id, pair in queries.items():
            if not isinstance(query_id, str) or not query_id or query_id != query_id.strip():
                raise ValueError("query IDs must be non-empty stripped strings")
            context, curve = pair
            if type(context) is not V5UniversalCandidateContext or type(curve) is not ObservedCurve:
                raise TypeError("query entries require a typed context and observed curve")
            validate_v5_universal_curve_alignment(context, curve)
            resolved[query_id] = (context, curve)
        if not resolved:
            raise ValueError("at least one materialized query is required")
        self._queries = MappingProxyType(resolved)
        self._thresholds = thresholds
        self._budget = budget
        self._source_sha = source_summary_artifact_sha256
        if (calibration is None) != (observation_views is None):
            raise ValueError("calibration and observation views must be supplied together")
        self._calibration = calibration
        self._calibration_path = None if calibration_path is None else Path(calibration_path)
        if self._calibration_path is not None and calibration is None:
            raise ValueError("sealed calibration requires a checked calibration identity")
        self._calibration_file_identity = self._read_calibration_file()
        self._views = None if observation_views is None else MappingProxyType(dict(observation_views))
        if self._views is not None and set(self._views) != set(self._queries):
            raise ValueError("calibrated observation IDs must exactly cover the queries")
        self._calibration_json = self._calibration_binding()
        # Each verified candidate needs a real call. B+1 makes the compatible
        # target unreachable within B without changing GUI stopping semantics.
        self._protocol_json = canonical_json({
            "protocol_id": self.protocol_id,
            "inference_version": V5_UNIVERSAL_INFERENCE_VERSION,
            "budget": budget.audit_payload(), "thresholds": asdict(thresholds),
            "target_parameter_mode_count": maximum + 1,
            "retrieval": "none", "reference_guided_search": False,
            "representative_selection_policy": "budget_snapshot",
            "short_run": "fail_closed_without_padding",
            **({"calibration_file_guard": "0400_single_link_pre_post_v1"}
               if self._calibration_path is not None else {}),
            **({"calibrated_queries": json.loads(self._calibration_json)}
               if self._calibration_json is not None else {}),
        })

    @property
    def protocol_id(self):
        return (V5_UNIVERSAL_TUNING_PROTOCOL_ID if self._views is None
                else V5_CALIBRATED_UNIVERSAL_TUNING_PROTOCOL_ID)

    def _calibration_binding(self):
        if self._read_calibration_file() != self._calibration_file_identity:
            raise RuntimeError("sealed calibration file identity changed during tuning")
        if self._views is None:
            return None
        payload = {}
        for query_id, view in self._views.items():
            context, curve = self._queries[query_id]
            exact = V5ExactSearchObservation.from_observation_view(view, curve_id=curve.curve_id)
            if observed_curve_sha256(curve) != exact.curve_sha256:
                raise ValueError("calibrated observation differs from actual tuning curve")
            validate_v5_universal_curve_alignment(context, exact.observed_curve)
            threshold = bind_v5_calibrated_observation_threshold(self._calibration, view)
            payload[query_id] = {
                "context_sha256": context.audit_sha256,
                "curve_sha256": exact.curve_sha256,
                "observation_sha256": exact.parent_observation_audit_sha256,
                "threshold": threshold.audit_payload(),
            }
        return canonical_json(payload)

    def _read_calibration_file(self):
        if self._calibration_path is None:
            return None
        before = read_only_identity(self._calibration_path, "tuning calibration")
        if before["mode_octal"] != "0400":
            raise ValueError("tuning calibration must have mode 0400")
        checked = read_v5_checked_compatibility_calibration(
            self._calibration_path,
            expected_artifact_sha256=self._calibration.identity.artifact_sha256,
            expected_file_sha256=self._calibration.identity.file_sha256,
        )
        if (checked.identity != self._calibration.identity
                or read_only_identity(self._calibration_path, "tuning calibration") != before):
            raise RuntimeError("sealed calibration changed while being loaded")
        return before

    @property
    def protocol_sha256(self):
        return sha256(self._protocol_json.encode()).hexdigest()

    @property
    def protocol_json(self):
        return self._protocol_json

    def __call__(self, checkpoint, reference, binding, seed, exact_budget):
        if self._calibration_binding() != self._calibration_json:
            raise RuntimeError("calibrated query binding changed before tuning")
        if type(checkpoint) is not V5RetainedFullCheckpoint or type(reference) is not V5FrozenReferenceSet:
            raise TypeError("expected a retained checkpoint and frozen reference envelope")
        if type(exact_budget) is not int or exact_budget != self._budget.forward_evaluation_limit:
            raise ValueError("requested budget differs from the frozen adapter protocol")
        expected = build_v5_checkpoint_evaluation_method_binding(
            checkpoint_epoch=checkpoint.full_epoch,
            checkpoint_artifact_sha256=checkpoint.checkpoint_artifact_sha256,
            checkpoint_weights_sha256=checkpoint.checkpoint_weights_sha256,
            training_result_sha256=checkpoint.training_result_sha256,
            source_summary_artifact_sha256=self._source_sha,
            method_id=binding["method_id"],
            base_method_protocol_id=self.protocol_id,
            base_method_protocol_sha256=self.protocol_sha256,
            inference_seed=seed, representative_selection_policy="budget_snapshot",
        )
        if dict(binding) != expected:
            raise ValueError("checkpoint/source/seed/protocol binding differs from actual execution")
        context, curve = self._queries[reference.query_id]
        if context.audit_sha256 != reference.query_context_sha256:
            raise ValueError("reference envelope and actual query context differ")
        validate_v5_universal_curve_alignment(context, curve)
        before = read_only_identity(checkpoint.checkpoint_path, "tuning checkpoint")
        model = load_v5_verified_tuning_checkpoint(checkpoint)
        recorder = V5TuningSearchRecorder(
            query_context_sha256=context.audit_sha256,
            source_artifact_sha256=checkpoint.checkpoint_artifact_sha256,
            exact_forward_budget=exact_budget,
        )
        thresholds = self._thresholds
        if self._views is not None:
            calibrated = bind_v5_calibrated_observation_threshold(
                self._calibration, self._views[reference.query_id],
            )
            thresholds = replace(thresholds, standardized_exact_log_rmse_max=calibrated.threshold_value)
        result = run_v5_universal_one_click_inference(
            context, curve, model=model, thresholds=thresholds,
            target_parameter_mode_count=exact_budget + 1, budget=self._budget,
            seed=seed, reference_modes=(), retrieval_seeds=(),
            exact_forward_call_observer=recorder.observe_call,
            evaluated_candidate_observer=recorder.observe_candidate,
        )
        if (model_weights_sha256(model) != checkpoint.checkpoint_weights_sha256
                or read_only_identity(checkpoint.checkpoint_path, "tuning checkpoint") != before):
            raise RuntimeError("checkpoint file or loaded weights changed during tuning inference")
        validate_v5_universal_curve_alignment(context, curve)
        if self._calibration_binding() != self._calibration_json:
            raise RuntimeError("calibrated query binding changed during tuning")
        calls, emissions, snapshots, elapsed = recorder.finish()
        if (result.context_audit_sha256 != context.audit_sha256
                or result.forward_evaluations_used != len(calls)
                or result.termination_reason != "exact_forward_budget_exhausted"
                or sum(row.exact_forward_calls for row in result.attempts) != len(calls)):
            raise RuntimeError("actual inference result disagrees with the complete call ledger")
        trace_id = sha256(canonical_json({
            "binding": expected, "query_id": reference.query_id,
            "query_context_sha256": context.audit_sha256,
        }).encode()).hexdigest()
        return V5TuningExactTraceRunResult(
            trace_id=trace_id,
            checkpoint_artifact_sha256_used=checkpoint.checkpoint_artifact_sha256,
            query_id=reference.query_id,
            method_protocol_sha256_used=expected["bound_method_protocol_sha256"],
            inference_seed_used=seed, exact_forward_call_budget_used=exact_budget,
            exact_forward_calls=calls, candidate_emissions=emissions,
            representative_snapshots=snapshots, completion_elapsed_seconds=elapsed,
        )
