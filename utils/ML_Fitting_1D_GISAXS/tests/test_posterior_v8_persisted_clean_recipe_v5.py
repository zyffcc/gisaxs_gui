from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.build_grouped_dataset_v5 import (
    build_tiny_v5_grouped_dataset,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_b_evaluation_v5 import (
    replay_parent_contexts,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.persisted_clean_recipe_v5 import (
    persisted_v5_clean_recipe_from_json,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.synthetic_recipe_v5 import (
    sample_v5_clean_recipe,
)


def _canonical(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _digest(encoded: str) -> str:
    return sha256(encoded.encode("utf-8")).hexdigest()


def test_persisted_recipe_decodes_without_invoking_the_seeded_sampler(monkeypatch):
    recipe = sample_v5_clean_recipe(("sphere",), recipe_seed=20260903, pattern_id=0)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("cross-CPU persisted decoding must not regenerate seeded physics")

    monkeypatch.setattr(
        "utils.ML_Fitting_1D_GISAXS.PosteriorV8.synthetic_recipe_v5.sample_v5_bounds_query",
        forbidden,
    )
    replay = persisted_v5_clean_recipe_from_json(recipe.canonical_json, recipe.sha256)

    assert replay.canonical_json == recipe.canonical_json
    assert replay.sha256 == recipe.sha256
    assert replay.target.local_target_unit == recipe.target.local_target_unit
    assert replay.target.truth_components == recipe.target.truth_components
    assert replay.amplitude == recipe.amplitude


def test_persisted_recipe_accepts_float64_truth_ulp_drift_at_exact_float32_wire():
    recipe = sample_v5_clean_recipe(("sphere",), recipe_seed=20260903, pattern_id=0)
    payload = json.loads(recipe.canonical_json)
    original = float(payload["truth_components"][0]["R"])
    payload["truth_components"][0]["R"] = float(np.nextafter(original, np.inf))
    encoded = _canonical(payload)

    replay = persisted_v5_clean_recipe_from_json(encoded, _digest(encoded))

    assert replay.target.truth_components[0].R != recipe.target.truth_components[0].R
    assert np.float32(replay.target.truth_components[0].R) == np.float32(
        recipe.target.truth_components[0].R
    )


def test_persisted_recipe_accepts_roundoff_only_drift_at_fixed_coupled_endpoint():
    recipe = sample_v5_clean_recipe(("sphere",), recipe_seed=91, pattern_id=0)
    codec = recipe.query.codec_for(recipe.target.pattern_id)
    assert codec.active_mask[1] and not codec.varying_mask[1]

    payload = json.loads(recipe.canonical_json)
    original = float(payload["truth_components"][0]["sigma_R"])
    payload["truth_components"][0]["sigma_R"] = float(np.nextafter(original, np.inf))
    encoded = _canonical(payload)

    replay = persisted_v5_clean_recipe_from_json(encoded, _digest(encoded))

    assert replay.target.truth_components[0].sigma_R != original
    assert np.float32(replay.target.truth_components[0].sigma_R) == np.float32(original)


def test_persisted_recipe_rejects_material_drift_at_fixed_coupled_endpoint():
    recipe = sample_v5_clean_recipe(("sphere",), recipe_seed=91, pattern_id=0)
    payload = json.loads(recipe.canonical_json)
    original = float(payload["truth_components"][0]["sigma_R"])
    payload["truth_components"][0]["sigma_R"] = original * 1.001
    encoded = _canonical(payload)

    with pytest.raises(ValueError, match="persisted clean recipe is invalid"):
        persisted_v5_clean_recipe_from_json(encoded, _digest(encoded))


def test_persisted_recipe_fails_closed_on_hash_nested_field_and_target_drift():
    recipe = sample_v5_clean_recipe(("sphere",), recipe_seed=20260903, pattern_id=0)
    with pytest.raises(ValueError, match="SHA-256"):
        persisted_v5_clean_recipe_from_json(recipe.canonical_json, "0" * 64)

    payload = json.loads(recipe.canonical_json)
    payload["query"]["component_bounds"][0]["R"]["high"] = float(
        np.nextafter(payload["query"]["component_bounds"][0]["R"]["high"], np.inf)
    )
    encoded = _canonical(payload)
    with pytest.raises(ValueError, match="persisted clean recipe is invalid"):
        persisted_v5_clean_recipe_from_json(encoded, _digest(encoded))

    payload = json.loads(recipe.canonical_json)
    active_index = recipe.query.codec_for(recipe.target.pattern_id).varying_indices[0]
    payload["local_target_unit"][active_index] += 0.1
    encoded = _canonical(payload)
    with pytest.raises(ValueError, match="persisted clean recipe is invalid"):
        persisted_v5_clean_recipe_from_json(encoded, _digest(encoded))


def test_phase_b_parent_replay_uses_persisted_recipe_contract(monkeypatch):
    dataset = build_tiny_v5_grouped_dataset(recipe_count=1)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Phase-B must not use cross-CPU seeded replay")

    monkeypatch.setattr(
        "utils.ML_Fitting_1D_GISAXS.PosteriorV8.synthetic_recipe_v5.sample_v5_clean_recipe",
        forbidden,
    )
    contexts = replay_parent_contexts(dataset)

    assert len(contexts) == 1
    assert contexts[0][0].sha256 == dataset.arrays["clean__recipe_sha256"][0]
