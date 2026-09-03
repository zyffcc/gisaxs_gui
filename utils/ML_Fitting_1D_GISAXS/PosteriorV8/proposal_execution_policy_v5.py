"""Shared, immutable proposal execution policy for V5 training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json


V5_PROPOSAL_EXECUTION_POLICY_SCHEMA = (
    "gisaxs.posterior_v8.proposal_execution_policy/v1"
)
V5_PROPOSAL_EXECUTION_POLICY_VERSION = (
    "mixture12_branch_top4_global_prefix_round_robin_soft_rank_alignment_v1"
)
V5_PROPOSAL_BRANCH_RANKING = (
    "descending_search_yield_logit_then_ascending_global_branch_key_v1"
)
V5_PROPOSAL_WITHIN_BRANCH_RANKING = (
    "descending_mixture_log_weight_then_ascending_mixture_index_v1"
)
V5_PROPOSAL_GLOBAL_ATTEMPT_PREFIX = (
    "first_n_neural_attempts_are_a_prefix_of_one_global_ranked_branch_"
    "round_robin_sequence_v1"
)
V5_PROPOSAL_BRANCH_ROUND_ROBIN = (
    "one_seed_per_ranked_branch_per_round_before_any_branch_advances_v1"
)
V5_PROPOSAL_MEDIAN_DRAW_ORDER = (
    "all_retained_mixture_medians_before_stochastic_draws_within_branch_v1"
)


@dataclass(frozen=True, kw_only=True)
class V5ProposalExecutionPolicy:
    """Scientific identity shared by the training objective and one-click runtime.

    The fields are deliberately not tunable in V5. Changing any value requires
    a new version and therefore a new policy digest.
    """

    mixture_component_count: int = 12
    per_branch_top_l: int = 4
    include_mixture_medians: bool = True
    branch_ranking: str = V5_PROPOSAL_BRANCH_RANKING
    within_branch_ranking: str = V5_PROPOSAL_WITHIN_BRANCH_RANKING
    global_attempt_prefix: str = V5_PROPOSAL_GLOBAL_ATTEMPT_PREFIX
    branch_round_robin: str = V5_PROPOSAL_BRANCH_ROUND_ROBIN
    median_draw_order: str = V5_PROPOSAL_MEDIAN_DRAW_ORDER
    soft_rank_logit_temperature: float = 1.0
    soft_rank_cutoff_temperature: float = 0.25

    def __post_init__(self) -> None:
        if type(self.mixture_component_count) is not int:
            raise TypeError("mixture_component_count must be an integer")
        if type(self.per_branch_top_l) is not int:
            raise TypeError("per_branch_top_l must be an integer")
        if type(self.include_mixture_medians) is not bool:
            raise TypeError("include_mixture_medians must be a bool")
        if type(self.soft_rank_logit_temperature) is not float:
            raise TypeError("soft_rank_logit_temperature must be a float")
        if type(self.soft_rank_cutoff_temperature) is not float:
            raise TypeError("soft_rank_cutoff_temperature must be a float")
        actual = (
            self.mixture_component_count,
            self.per_branch_top_l,
            self.include_mixture_medians,
            self.branch_ranking,
            self.within_branch_ranking,
            self.global_attempt_prefix,
            self.branch_round_robin,
            self.median_draw_order,
            self.soft_rank_logit_temperature,
            self.soft_rank_cutoff_temperature,
        )
        expected = (
            12,
            4,
            True,
            V5_PROPOSAL_BRANCH_RANKING,
            V5_PROPOSAL_WITHIN_BRANCH_RANKING,
            V5_PROPOSAL_GLOBAL_ATTEMPT_PREFIX,
            V5_PROPOSAL_BRANCH_ROUND_ROBIN,
            V5_PROPOSAL_MEDIAN_DRAW_ORDER,
            1.0,
            0.25,
        )
        if actual != expected:
            raise ValueError(
                "V5 proposal execution policy is frozen; create a new version "
                "instead of overriding a policy field"
            )

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": V5_PROPOSAL_EXECUTION_POLICY_SCHEMA,
            "version": V5_PROPOSAL_EXECUTION_POLICY_VERSION,
            "policy": asdict(self),
        }


V5_PROPOSAL_EXECUTION_POLICY = V5ProposalExecutionPolicy()


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


V5_PROPOSAL_EXECUTION_POLICY_SHA256 = sha256(
    _canonical_json(V5_PROPOSAL_EXECUTION_POLICY.audit_payload()).encode("utf-8")
).hexdigest()


def validate_v5_proposal_execution_policy_sha256(value: str) -> str:
    """Return the canonical digest or fail closed on a training/runtime mismatch."""

    if not isinstance(value, str) or value != V5_PROPOSAL_EXECUTION_POLICY_SHA256:
        raise ValueError(
            "proposal_execution_policy_sha256 does not match the frozen V5 policy"
        )
    return value


__all__ = [
    "V5_PROPOSAL_BRANCH_RANKING",
    "V5_PROPOSAL_BRANCH_ROUND_ROBIN",
    "V5_PROPOSAL_EXECUTION_POLICY",
    "V5_PROPOSAL_EXECUTION_POLICY_SCHEMA",
    "V5_PROPOSAL_EXECUTION_POLICY_SHA256",
    "V5_PROPOSAL_EXECUTION_POLICY_VERSION",
    "V5_PROPOSAL_GLOBAL_ATTEMPT_PREFIX",
    "V5_PROPOSAL_MEDIAN_DRAW_ORDER",
    "V5_PROPOSAL_WITHIN_BRANCH_RANKING",
    "V5ProposalExecutionPolicy",
    "validate_v5_proposal_execution_policy_sha256",
]
