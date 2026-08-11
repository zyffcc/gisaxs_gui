from utils.ai_fitting_profiles import DEFAULT_PROFILE_NAME, PROFILE_DEFAULTS, profile_registry


def test_profiles_are_ordered_by_search_cost_and_balanced_is_default():
    fast = PROFILE_DEFAULTS["Fast"]
    balanced = PROFILE_DEFAULTS["Balanced"]
    exhaustive = PROFILE_DEFAULTS["Exhaustive"]
    assert DEFAULT_PROFILE_NAME == "Balanced"
    assert fast.candidate_count < balanced.candidate_count < exhaustive.candidate_count
    assert fast.refinement_count < balanced.refinement_count < exhaustive.refinement_count
    assert fast.q_stride > balanced.q_stride > exhaustive.q_stride
    assert profile_registry.get().name == "Balanced"


def test_profile_override_becomes_custom_without_mutating_default():
    balanced = profile_registry.get("Balanced")
    custom = balanced.with_updates(candidate_count=777, random_seed=9)
    assert custom.name == "Custom"
    assert custom.candidate_count == 777
    assert profile_registry.get("Balanced").candidate_count == 192
