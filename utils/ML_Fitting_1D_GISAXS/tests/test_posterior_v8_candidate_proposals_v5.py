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
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.candidate_proposals_v5 import (
    V5BatchedProposalOutput,
    sample_v5_local_proposals,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import ClosedInterval, SPHERE
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.preprocessing import preprocess_curve
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.uncertainty_provenance_v5 import (
    V5UncertaintyProvenance,
)


def _batch():
    q = np.geomspace(1.0e-3, 1.0, 64)
    intensity = 20.0 + 5.0e3 / (1.0 + (q / 0.03) ** 4)
    curve = preprocess_curve(q, intensity, 0.02 * intensity)
    query = full_range_v5_query((SPHERE,), query_seed=83)
    amplitude_query = V5AmplitudeQuery.create(
        background=ClosedInterval(0.0, 1.0e8),
        k=ClosedInterval(1.0e-2, 1.0e8),
        component_intensities=(ClosedInterval(1.0, 1.0),),
        resolution_presence_policy=query.resolution_presence_policy,
        int_res=ClosedInterval(0.0, 1.0e8),
    )
    return build_v5_candidate_context_batch(
        curve,
        V5UncertaintyProvenance("simulated_sigma"),
        query,
        amplitude_query,
        pattern_ids=query.feasible_wire_pattern_ids[:2],
    )


def _two_sphere_batch(component_intensities):
    q = np.geomspace(1.0e-3, 1.0, 64)
    intensity = 20.0 + 5.0e3 / (1.0 + (q / 0.03) ** 4)
    curve = preprocess_curve(q, intensity, 0.02 * intensity)
    query = full_range_v5_query((SPHERE, SPHERE), query_seed=84)
    amplitude_query = V5AmplitudeQuery.create(
        background=ClosedInterval(0.0, 1.0e8),
        k=ClosedInterval(1.0e-2, 1.0e8),
        component_intensities=component_intensities,
        resolution_presence_policy=query.resolution_presence_policy,
        int_res=ClosedInterval(0.0, 1.0e8),
    )
    return build_v5_candidate_context_batch(
        curve,
        V5UncertaintyProvenance("simulated_sigma"),
        query,
        amplitude_query,
        pattern_ids=(0,),
    )


def _outputs(batch, mixtures=3):
    loc = np.zeros((batch.branch_count, mixtures, 26), dtype=np.float32)
    loc[:, 1, :] = 1.0
    return {
        "proposal_search_yield_logit": np.asarray([[0.2], [1.4]], dtype=np.float32),
        "mixture_logits": np.asarray([[0.0, 2.0, -1.0], [3.0, 0.0, -2.0]], np.float32),
        "mixture_loc": loc,
        "mixture_logscale": np.full_like(loc, -2.0),
    }


def test_v5_sampler_emits_deterministic_medians_and_draws_within_user_bounds():
    batch = _batch()
    first = sample_v5_local_proposals(
        batch,
        _outputs(batch),
        mixture_limit=2,
        stochastic_draws_per_mixture=2,
        seed=91,
    )
    replay = sample_v5_local_proposals(
        batch,
        _outputs(batch),
        mixture_limit=2,
        stochastic_draws_per_mixture=2,
        seed=91,
    )

    assert first == replay
    assert len(first) == batch.branch_count * 2 * 3
    assert first[0].pattern_id == batch.branch_conditions[1].pattern_id
    assert first[0].mixture_index == 0
    for proposal in first:
        codec = batch.query.codec_for(proposal.pattern_id)
        encoded = codec.encode(proposal.latent_components, proposal.resolution)
        assert np.allclose(encoded.unit_cube, proposal.local_unit, rtol=0.0, atol=1e-12)
        assert np.all(np.asarray(proposal.local_unit) >= 0.0)
        assert np.all(np.asarray(proposal.local_unit) <= 1.0)
        varying = np.asarray(
            batch.branch_conditions[proposal.branch_batch_index].varying_dimension_mask
        )
        assert np.all(np.asarray(proposal.local_unit)[~varying] == 0.5)


def test_v5_sampler_keeps_branch_and_mixture_scores_separate():
    batch = _batch()
    values = sample_v5_local_proposals(
        batch,
        _outputs(batch),
        mixture_limit=1,
        stochastic_draws_per_mixture=0,
        seed=2,
    )
    assert len(values) == 2
    assert values[0].search_yield_logit > values[1].search_yield_logit
    assert values[0].ranking_key[0] < values[1].ranking_key[0]


def test_v5_output_parser_fails_closed_on_shape_or_nonfinite_values():
    batch = _batch()
    outputs = _outputs(batch)
    outputs["mixture_loc"] = outputs["mixture_loc"][:, :, :-1]
    with pytest.raises(ValueError, match="mixture_loc"):
        V5BatchedProposalOutput.from_mapping(outputs, branch_count=batch.branch_count)

    outputs = _outputs(batch)
    outputs["proposal_search_yield_logit"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        sample_v5_local_proposals(
            batch,
            outputs,
            mixture_limit=1,
            stochastic_draws_per_mixture=0,
            seed=2,
        )


def test_v5_sampler_requires_at_least_one_seed_per_selected_mixture():
    batch = _batch()
    with pytest.raises(ValueError, match="at least one"):
        sample_v5_local_proposals(
            batch,
            _outputs(batch),
            mixture_limit=1,
            stochastic_draws_per_mixture=0,
            include_mixture_medians=False,
            seed=2,
        )


def _two_sphere_outputs(first_radius_unit: float, second_radius_unit: float):
    loc = np.zeros((1, 1, 26), dtype=np.float32)
    loc[0, 0, 0] = np.log(first_radius_unit / (1.0 - first_radius_unit))
    loc[0, 0, 6] = np.log(second_radius_unit / (1.0 - second_radius_unit))
    return {
        "proposal_search_yield_logit": np.zeros((1, 1), dtype=np.float32),
        "mixture_logits": np.zeros((1, 1), dtype=np.float32),
        "mixture_loc": loc,
        "mixture_logscale": np.full_like(loc, -2.0),
    }


def test_v5_sampler_preserves_labels_when_per_slot_int_ranges_differ():
    batch = _two_sphere_batch(
        (ClosedInterval(0.0, 0.4), ClosedInterval(0.6, 1.0))
    )
    proposals = sample_v5_local_proposals(
        batch,
        _two_sphere_outputs(0.8, 0.2),
        mixture_limit=1,
        stochastic_draws_per_mixture=0,
        seed=3,
    )

    assert len(proposals) == 1
    radii = tuple(value.log_R for value in proposals[0].latent_components)
    assert radii[0] > radii[1]
    assert proposals[0].local_unit[0] > proposals[0].local_unit[6]
    with pytest.raises(ValueError, match="unsupported V5 proposal sampler"):
        replace(proposals[0], version="stale-v1")


def test_v5_sampler_is_permutation_invariant_only_with_identical_complete_context():
    batch = _two_sphere_batch((ClosedInterval(0.0, 1.0),) * 2)
    forward = sample_v5_local_proposals(
        batch,
        _two_sphere_outputs(0.8, 0.2),
        mixture_limit=1,
        stochastic_draws_per_mixture=0,
        seed=3,
    )[0]
    reverse = sample_v5_local_proposals(
        batch,
        _two_sphere_outputs(0.2, 0.8),
        mixture_limit=1,
        stochastic_draws_per_mixture=0,
        seed=3,
    )[0]

    assert forward.latent_components == reverse.latent_components
    np.testing.assert_allclose(forward.local_unit, reverse.local_unit, rtol=0.0, atol=1e-7)
