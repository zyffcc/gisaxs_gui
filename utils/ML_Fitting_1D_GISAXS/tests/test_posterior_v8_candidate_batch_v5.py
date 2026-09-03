from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_query_v5 import V5AmplitudeQuery
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_query_sampling_v5 import (
    full_range_v5_query,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.candidate_batch_v5 import (
    build_v5_candidate_context_batch,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import ClosedInterval, SPHERE
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.model_v5_contract import MODEL_V5_INPUT_KEYS
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.preprocessing import preprocess_curve
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.uncertainty_provenance_v5 import (
    V5UncertaintyProvenance,
)


def _curve(scale: float = 1.0):
    q = np.geomspace(1.0e-3, 1.0, 64)
    intensity = scale * (50.0 + 2.0e4 / (1.0 + (q / 0.04) ** 4))
    sigma = 0.02 * intensity
    return preprocess_curve(q, intensity, sigma)


def _amplitude_query(query):
    return V5AmplitudeQuery.create(
        background=ClosedInterval(0.0, 1.0e8),
        k=ClosedInterval(1.0e-2, 1.0e8),
        component_intensities=tuple(ClosedInterval(0.0, 1.0) for _ in query.topology),
        resolution_presence_policy=query.resolution_presence_policy,
        int_res=(
            None
            if query.resolution_presence_policy == "absent"
            else ClosedInterval(0.0, 1.0e8)
        ),
    )


def test_candidate_batch_enumerates_every_feasible_branch_in_stable_order():
    query = full_range_v5_query((SPHERE,), query_seed=17)
    amplitude_query = _amplitude_query(query)
    provenance = V5UncertaintyProvenance("simulated_sigma")
    batch = build_v5_candidate_context_batch(_curve(), provenance, query, amplitude_query)

    assert batch.pattern_ids == query.feasible_wire_pattern_ids
    assert batch.branch_count == len(query.feasible_wire_pattern_ids)
    assert tuple(batch.model_inputs) == MODEL_V5_INPUT_KEYS
    assert all(value.shape[0] == batch.branch_count for value in batch.model_inputs.values())
    assert np.all(batch.model_inputs["uncertainty_provenance"] == (0.0, 1.0, 0.0))
    assert batch.model_inputs["amplitude_bounds_embedding"].shape[1] == 21
    assert len(batch.amplitude_constraints) == batch.branch_count
    assert not any(value.flags.writeable for value in batch.model_inputs.values())
    assert batch.for_model()["branch_pattern_id"].reshape(-1).tolist() == list(
        query.feasible_wire_pattern_ids
    )
    assert '"all_feasible_branches_enumerated":true' in batch.audit_json


def test_candidate_batch_supports_explicit_subset_without_reordering():
    query = full_range_v5_query((SPHERE,), query_seed=18)
    amplitude_query = _amplitude_query(query)
    selected = tuple(reversed(query.feasible_wire_pattern_ids[-2:]))
    batch = build_v5_candidate_context_batch(
        _curve(),
        V5UncertaintyProvenance("encoder_proxy_missing_sigma", 0.015),
        query,
        amplitude_query,
        pattern_ids=selected,
    )

    assert batch.pattern_ids == selected
    assert np.all(batch.model_inputs["uncertainty_provenance"] == (0.0, 0.0, 1.0))
    assert '"all_feasible_branches_enumerated":false' in batch.audit_json


def test_candidate_batch_quotients_only_equal_complete_slot_contracts():
    query = full_range_v5_query((SPHERE, SPHERE), query_seed=180)
    provenance = V5UncertaintyProvenance("simulated_sigma")
    equal_amplitude = _amplitude_query(query)
    unequal_amplitude = V5AmplitudeQuery.create(
        background=ClosedInterval(0.0, 1.0e8),
        k=ClosedInterval(1.0e-2, 1.0e8),
        component_intensities=(
            ClosedInterval(0.0, 0.4),
            ClosedInterval(0.6, 1.0),
        ),
        resolution_presence_policy=query.resolution_presence_policy,
        int_res=ClosedInterval(0.0, 1.0e8),
    )

    equal = build_v5_candidate_context_batch(
        _curve(), provenance, query, equal_amplitude
    )
    unequal = build_v5_candidate_context_batch(
        _curve(), provenance, query, unequal_amplitude
    )

    assert query.feasible_wire_pattern_ids == (0, 1, 2, 3, 16, 17, 18, 19)
    assert equal.pattern_ids == (0, 2, 3, 16, 18, 19)
    assert unequal.pattern_ids == query.feasible_wire_pattern_ids
    with pytest.raises(ValueError, match="not feasible"):
        build_v5_candidate_context_batch(
            _curve(), provenance, query, equal_amplitude, pattern_ids=(1,)
        )


def test_candidate_batch_fails_closed_on_duplicate_or_infeasible_patterns():
    query = full_range_v5_query((SPHERE,), query_seed=19)
    amplitude_query = _amplitude_query(query)
    provenance = V5UncertaintyProvenance("measured_sigma")
    pattern = query.feasible_wire_pattern_ids[0]
    with pytest.raises(ValueError, match="duplicates"):
        build_v5_candidate_context_batch(
            _curve(), provenance, query, amplitude_query, pattern_ids=(pattern, pattern)
        )
    with pytest.raises(ValueError, match="not feasible"):
        build_v5_candidate_context_batch(
            _curve(), provenance, query, amplitude_query, pattern_ids=(31,)
        )


def test_candidate_batch_digest_detects_post_construction_tampering():
    query = full_range_v5_query((SPHERE,), query_seed=20)
    amplitude_query = _amplitude_query(query)
    batch = build_v5_candidate_context_batch(
        _curve(), V5UncertaintyProvenance("simulated_sigma"), query, amplitude_query
    )
    replacement = dict(batch.model_inputs)
    changed = np.array(replacement["global_features"], copy=True)
    changed[0, 0] += 0.1
    changed.setflags(write=False)
    replacement["global_features"] = changed
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(batch, model_inputs=replacement)


def test_candidate_model_inputs_are_equivariant_to_joint_curve_and_amplitude_scaling():
    query = full_range_v5_query((SPHERE,), query_seed=21)
    amplitude_query = _amplitude_query(query)
    provenance = V5UncertaintyProvenance("simulated_sigma")
    first = build_v5_candidate_context_batch(_curve(), provenance, query, amplitude_query)
    scaled = build_v5_candidate_context_batch(
        _curve(37.0),
        provenance,
        query,
        amplitude_query.rescaled_intensity(37.0),
    )

    for key in MODEL_V5_INPUT_KEYS:
        np.testing.assert_allclose(first.model_inputs[key], scaled.model_inputs[key], atol=1e-7)
    assert first.query_sha256 != scaled.query_sha256
