from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from src.gimap.features.fitting.domain.physical_constraints import exclusion_size
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_first_contract import (
    BOUND_PLACEMENTS,
    BOUNDS_EMBEDDING_DIM,
    BOUNDS_FIRST_SCHEMA_VERSION,
    RANGE_REGIMES,
    BoundsFirstLabel,
    BoundsProvenance,
    local_varying_mask,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import bounds_first_dataset as dataset_module
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_first_dataset import (
    ARRAY_ORDER,
    LOCAL_TARGET_OPEN_EPSILON,
    BoundsFirstPilotConfig,
    build_compact_pilot,
    generate_compact_pilot,
    sample_solution_label,
    sample_user_bounds,
    validate_compact_pilot,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_codec import (
    HARD_CORE_SPACING_MARGIN,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.canonical_component_slots import (
    CANONICAL_COMPONENT_SLOTS_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    R_DOMAIN,
    TOPOLOGIES,
    ClosedInterval,
    GuiComponentBounds,
    full_component_bounds,
    gui_component_to_latent,
)


def _config():
    return BoundsFirstPilotConfig(
        master_seed=20260902,
        sample_count=6,
        topology_ids=(0, 1, 2, 3),
        points=64,
        noise_mode="clean",
    )


def test_bounds_are_reproducible_and_independent_of_later_local_truth_draws():
    topology = TOPOLOGIES[0]
    first = sample_user_bounds(
        topology,
        (True,),
        True,
        regime="narrow",
        placement="asymmetric_low",
        bounds_seed=771,
    )
    replay = sample_user_bounds(
        topology,
        (True,),
        True,
        regime="narrow",
        placement="asymmetric_low",
        bounds_seed=771,
    )
    target_a = sample_solution_label(first, local_target_seed=1001)
    target_b = sample_solution_label(first, local_target_seed=2002)

    assert first == replay
    assert first.sha256 == replay.sha256
    assert first.embedding == replay.embedding
    assert len(first.embedding) == BOUNDS_EMBEDDING_DIM
    assert target_a.local_target_unit != target_b.local_target_unit
    assert target_a.bounds is first and target_b.bounds is first
    assert target_a.global_reference_unit != target_a.local_target_unit

    for target in (target_a, target_b):
        component = target.truth_components[0]
        required = HARD_CORE_SPACING_MARGIN * exclusion_size(
            component.shape, {"R": component.R}
        )
        assert component.D > required
        first.local_codec().encode(
            tuple(gui_component_to_latent(item) for item in target.truth_components),
            target.truth_resolution,
        )


def test_compact_pilot_covers_range_shapes_edges_fixed_axes_and_hard_core():
    arrays, metadata = generate_compact_pilot(_config())

    assert tuple(arrays) == ARRAY_ORDER
    assert metadata["dataset_schema_version"] == BOUNDS_FIRST_SCHEMA_VERSION
    assert (
        metadata["canonical_component_slots_version"]
        == CANONICAL_COMPONENT_SLOTS_VERSION
    )
    assert set(metadata["range_codes"]) == set(RANGE_REGIMES)
    assert set(metadata["placement_codes"]) == set(BOUND_PLACEMENTS)
    assert arrays["bounds_embedding"].shape == (6, BOUNDS_EMBEDDING_DIM)
    assert np.all(arrays["truth_available"])
    assert "branch_low" not in arrays and "branch_high" not in arrays

    records = metadata["records"]
    assert {item["bounds"]["range_regime"] for item in records} == set(RANGE_REGIMES)
    assert {item["bounds"]["placement"] for item in records} == set(BOUND_PLACEMENTS)
    asymmetric_low = next(
        item for item in records if item["bounds"]["placement"] == "asymmetric_low"
    )
    asymmetric_high = next(
        item for item in records if item["bounds"]["placement"] == "asymmetric_high"
    )
    domain_log_center = np.sqrt(R_DOMAIN.low * R_DOMAIN.high)
    low_r = asymmetric_low["bounds"]["component_bounds"][0]["R"]
    high_r = asymmetric_high["bounds"]["component_bounds"][0]["R"]
    assert np.sqrt(low_r["low"] * low_r["high"]) < domain_log_center
    assert np.sqrt(high_r["low"] * high_r["high"]) > domain_log_center
    high_edge = next(
        item for item in records if item["bounds"]["placement"] == "edge_high"
    )
    assert high_edge["bounds"]["component_bounds"][0]["R"]["high"] == R_DOMAIN.high
    partial = next(
        item for item in records if item["bounds"]["placement"] == "partial_fixed"
    )
    assert (
        partial["bounds"]["component_bounds"][0]["R"]["low"]
        == partial["bounds"]["component_bounds"][0]["R"]["high"]
    )
    partial_row = partial["record_index"]
    assert not arrays["local_varying_mask"][partial_row, 0]
    assert arrays["target_local_unit"][partial_row, 0] == np.float32(0.5)

    varying = arrays["local_varying_mask"]
    assert np.all(arrays["target_local_unit"][varying] > np.float32(0.0))
    assert np.all(arrays["target_local_unit"][varying] < np.float32(1.0))
    assert metadata["local_target_open_epsilon"] == LOCAL_TARGET_OPEN_EPSILON

    hard_core_checked = 0
    for item in records:
        for component, d_present in zip(
            item["truth_components"], item["bounds"]["d_present"]
        ):
            if not d_present:
                continue
            parameters = {"R": component["R"]}
            if component["h"] is not None:
                parameters["h"] = component["h"]
            required = HARD_CORE_SPACING_MARGIN * exclusion_size(
                component["shape"], parameters
            )
            assert component["D"] > required
            hard_core_checked += 1
    assert hard_core_checked > 0
    validate_compact_pilot(_config(), arrays, metadata)


def test_edge_high_range_is_exact_under_inward_platform_exp_roundoff(monkeypatch):
    native_exp = np.exp
    encoded_high = float(np.log(R_DOMAIN.high))

    def inward_endpoint_exp(value):
        result = native_exp(value)
        if float(value) == encoded_high:
            return np.nextafter(R_DOMAIN.high, -np.inf)
        return result

    monkeypatch.setattr(dataset_module.np, "exp", inward_endpoint_exp)
    bounds = sample_user_bounds(
        TOPOLOGIES[0],
        (False,),
        False,
        regime="narrow",
        placement="edge_high",
        bounds_seed=771,
    )

    assert bounds.component_bounds[0].R.high == R_DOMAIN.high


def test_compact_pilot_rejects_missing_component_slot_provenance():
    arrays, metadata = generate_compact_pilot(_config())
    tampered = {**metadata, "canonical_component_slots_version": "legacy"}

    with pytest.raises(ValueError, match="component-slot contract"):
        validate_compact_pilot(_config(), arrays, tampered, regenerate=False)


def test_compact_pilot_canonicalizes_repeated_shape_d_flags_before_bounds_sampling():
    config = BoundsFirstPilotConfig(
        master_seed=20260902,
        sample_count=1,
        topology_ids=(3,),
        points=64,
        noise_mode="clean",
    )

    _, metadata = generate_compact_pilot(config)

    assert metadata["records"][0]["bounds"]["d_present"] == [False, True]


def test_bounds_provenance_rejects_noncanonical_repeated_shape_d_flags():
    with pytest.raises(ValueError, match="absent-before-present"):
        BoundsProvenance.create(
            bounds_seed=9,
            generation_attempt=0,
            range_regime="full",
            placement="interior",
            component_bounds=(
                full_component_bounds("sphere", d_policy="required"),
                full_component_bounds("sphere", d_policy="absent"),
            ),
            d_present=(True, False),
            resolution_bounds=None,
        )
    with pytest.raises(ValueError, match="absent-before-present"):
        sample_user_bounds(
            ("sphere", "sphere"),
            (True, False),
            False,
            regime="full",
            placement="interior",
            bounds_seed=9,
        )


def test_negative_and_ood_annotations_cannot_carry_fabricated_truth():
    bounds = sample_user_bounds(
        TOPOLOGIES[2],
        (False,),
        False,
        regime="wide",
        placement="interior",
        bounds_seed=81,
    )
    solution = sample_solution_label(bounds, local_target_seed=91)
    no_solution = BoundsFirstLabel(
        task_kind="no_solution",
        bounds=bounds,
        annotation_reason="certified lower bound exceeds tolerance",
        annotation_certificate="exact_search_certificate:sha256:abc",
    )
    ood = BoundsFirstLabel(
        task_kind="ood",
        bounds=bounds,
        annotation_reason="acquisition q domain outside training support",
        annotation_certificate="acquisition_policy_v1",
    )

    assert no_solution.local_target_unit is None
    assert no_solution.truth_components is None
    assert ood.global_reference_unit is None
    with pytest.raises(ValueError, match="must not carry inverse truth"):
        BoundsFirstLabel(
            task_kind="no_solution",
            bounds=bounds,
            local_target_unit=solution.local_target_unit,
            global_reference_unit=solution.global_reference_unit,
            truth_components=solution.truth_components,
            annotation_reason="not enough",
            annotation_certificate="certificate",
        )
    with pytest.raises(ValueError, match="require reason and certificate"):
        BoundsFirstLabel(task_kind="ood", bounds=bounds)
    bad_local = list(solution.local_target_unit)
    bad_local[0] = 1.0 - bad_local[0]
    with pytest.raises(ValueError, match="local target"):
        replace(solution, local_target_unit=tuple(bad_local))


def test_local_varying_mask_uses_effective_coupled_codec_axes():
    provenance = BoundsProvenance.create(
        bounds_seed=31,
        generation_attempt=0,
        range_regime="narrow",
        placement="partial_fixed",
        component_bounds=(
            GuiComponentBounds(
                "sphere",
                R=ClosedInterval(1.0, 2.0),
                sigma_R=ClosedInterval(1.8, 1.8),
            ),
        ),
        d_present=(False,),
        resolution_bounds=None,
    )

    mask = local_varying_mask(provenance)
    label = sample_solution_label(provenance, local_target_seed=41)
    assert not mask[0] and not mask[1]
    assert label.local_target_unit[0] == 0.5
    assert label.local_target_unit[1] == 0.5


def test_build_is_compact_replayable_and_refuses_overwrite(tmp_path):
    config = _config()
    artifact = build_compact_pilot(tmp_path / "pilot", config)
    assert artifact.sample_count == config.sample_count
    assert len(artifact.npz_sha256) == 64
    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    assert metadata["npz_sha256"] == artifact.npz_sha256
    assert metadata["generation_order"][0] == "sample_gui_physical_bounds"
    assert "truth_conditioned_or_truth_containing_range_generation" in metadata[
        "forbidden_shortcuts"
    ]
    with np.load(artifact.npz_path, allow_pickle=False) as archive:
        assert archive["target_local_unit"].shape == (config.sample_count, 26)
        assert archive["bounds_embedding"].shape == (
            config.sample_count,
            BOUNDS_EMBEDDING_DIM,
        )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_compact_pilot(tmp_path / "pilot", config)
