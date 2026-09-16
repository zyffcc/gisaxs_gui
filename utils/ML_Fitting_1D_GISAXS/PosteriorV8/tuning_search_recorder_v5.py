"""Capture live universal-search events without reconstructing earlier outputs.

This recorder does not load a model, authorize training, or spend missing calls.
Its caller must run and bind the actual frozen model/search protocol.
"""

from __future__ import annotations

from numbers import Integral
import re
from time import monotonic

from .paper_budget_evaluator_v5 import (
    V5CandidateEmission, V5ExactForwardCall,
    V5_EXACT_COMPATIBLE, V5_EXACT_INCOMPATIBLE,
)
from .paper_representative_history_v5 import V5RepresentativeSnapshot
from .paper_representative_payload_v5 import (
    V5PaperParameterRepresentativePayload, V5_EMITTED_REPRESENTATIVE_ROLE,
)


class V5TuningSearchRecorder:
    """One query's call/candidate callbacks; terminal extraction is single-use."""

    def __init__(self, *, query_context_sha256, source_artifact_sha256,
                 exact_forward_budget, clock=monotonic):
        for digest in (query_context_sha256, source_artifact_sha256):
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("recorder requires explicit SHA-256 identities")
        if isinstance(exact_forward_budget, bool) or not isinstance(exact_forward_budget, Integral):
            raise TypeError("exact_forward_budget must be an integer")
        if exact_forward_budget < 1 or not callable(clock):
            raise ValueError("recorder requires a positive budget and callable clock")
        self._query_sha = query_context_sha256
        self._source_sha = source_artifact_sha256
        self._budget = int(exact_forward_budget)
        self._clock = clock
        self._start = clock()
        self._elapsed = 0.0
        self._closed = False
        self._calls = []
        self._emissions = []
        self._snapshots = []

    def _time(self):
        if self._closed:
            raise RuntimeError("recorder is already closed")
        elapsed = self._clock() - self._start
        if not self._elapsed <= elapsed < float("inf"):
            raise ValueError("recorder clock must remain finite and monotonic")
        self._elapsed = elapsed
        return elapsed

    def observe_call(self, index, phase):
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("call index must be an integer")
        if index != len(self._calls) + 1 or index > self._budget:
            raise ValueError("calls must be contiguous and within the frozen budget")
        self._calls.append(V5ExactForwardCall(
            exact_call_index=index, elapsed_seconds=self._time(),
        ))

    def observe_candidate(self, index, candidate, provenance, report):
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("candidate call index must be an integer")
        if not self._calls or index != len(self._calls):
            raise ValueError("candidate must follow its actual recorded call")
        if self._snapshots and index <= self._snapshots[-1].available_after_call:
            raise ValueError("candidate snapshots must advance the call ledger")
        expected = tuple(row.candidate_id for row in self._emissions) + (candidate.candidate_id,)
        if (tuple(row.candidate_id for row in report.candidates) != expected
                or len(set(expected)) != len(expected)
                or provenance.candidate_id != candidate.candidate_id
                or provenance.topology_id != candidate.topology_id
                or provenance.proposal_rank != candidate.proposal_rank
                or candidate.proposal_rank != len(expected)):
            raise ValueError("candidate/report/provenance identity mismatch")
        assessment = report.candidates[-1]
        if any(getattr(assessment, name) != getattr(candidate, name) for name in (
            "proposal_rank", "topology_id", "components", "resolution", "linear_solution",
            "bounds_pass", "physics_pass",
        )):
            raise ValueError("assessment physical parameter payload differs from candidate")
        if type(assessment.accepted) is not bool or (
            assessment.accepted and not (
                assessment.exact_valid and assessment.bounds_pass and assessment.physics_pass
            )
        ):
            raise ValueError("assessment compatibility contradicts its scientific gates")
        elapsed = self._time()
        emission = V5CandidateEmission(
            available_after_call=index, output_rank=candidate.proposal_rank,
            candidate_id=candidate.candidate_id, elapsed_seconds=elapsed,
            compatibility_status=(V5_EXACT_COMPATIBLE if report.candidates[-1].accepted
                                  else V5_EXACT_INCOMPATIBLE),
            payload=V5PaperParameterRepresentativePayload(
                representative_id=candidate.candidate_id,
                role=V5_EMITTED_REPRESENTATIVE_ROLE, parameter=candidate,
                global_branch_key=provenance.global_branch_key,
                query_context_sha256=self._query_sha,
                source_artifact_sha256=self._source_sha,
            ),
        )
        visible = tuple(mode.representative_candidate_id for mode in report.parameter_modes)
        compatible = {row.candidate_id for row in (*self._emissions, emission)
                      if row.compatibility_status == V5_EXACT_COMPATIBLE}
        if not set(visible) <= compatible:
            raise ValueError("report exposes an unknown or incompatible representative")
        snapshot = V5RepresentativeSnapshot(
            available_after_call=index, elapsed_seconds=elapsed, representative_ids=visible,
        )
        self._emissions.append(emission)
        self._snapshots.append(snapshot)

    def finish(self):
        """Return immutable event tuples only after the actual full budget."""
        elapsed = self._time()
        if len(self._calls) != self._budget:
            raise ValueError("search ended before its full exact-forward budget; no padding allowed")
        if not self._snapshots:
            self._snapshots.append(V5RepresentativeSnapshot(
                available_after_call=self._budget, elapsed_seconds=elapsed,
                representative_ids=(),
            ))
        self._closed = True
        return tuple(self._calls), tuple(self._emissions), tuple(self._snapshots), elapsed
