from __future__ import annotations

import json

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_rqmc_design_v5 import (
    V5_COMPATIBILITY_CALIBRATION_SAMPLING_REQUIREMENT,
    V5_PAPER_RQMC_CONFORMAL_CALIBRATION_INCLUDED,
    V5_PAPER_RQMC_ENGINEERING_MIN_INDEPENDENT_REPLICATES,
    V5_PAPER_RQMC_MIN_INDEPENDENT_REPLICATES,
    V5_PAPER_RQMC_RANDOMIZATION_UNIT,
    V5_PAPER_RQMC_SPLITS,
    V5_PAPER_RQMC_TARGET_INDEPENDENT_REPLICATES,
    V5PaperRQMCCoordinateContractIdentity,
    V5PaperRQMCDesign,
    V5PaperRQMCSplitSpec,
    create_v5_recipe_paper_rqmc_design,
    materialize_v5_paper_rqmc_block,
    v5_recipe_coordinate_contract_identity,
)


def _specs(*, replicates: int = 8, points: int = 8):
    return tuple(
        V5PaperRQMCSplitSpec(
            split_name=name,
            points_per_replicate=points,
            scramble_seeds=tuple(1000 * (split_index + 1) + index for index in range(replicates)),
        )
        for split_index, name in enumerate(V5_PAPER_RQMC_SPLITS)
    )


def _design() -> V5PaperRQMCDesign:
    return V5PaperRQMCDesign.create(
        coordinate_contract=_coordinate_contract(),
        split_specs=_specs(),
    )


def _coordinate_contract(
    *, version: str = "test_transform_v1", digest: str = "a" * 64
) -> V5PaperRQMCCoordinateContractIdentity:
    return V5PaperRQMCCoordinateContractIdentity(
        schema="test.authoritative.coordinates/v1",
        version=version,
        sha256=digest,
        dimension=4,
        coordinate_names=("shape", "radius", "noise", "q_window"),
    )


def test_formal_design_has_stable_independent_scramble_and_block_identities() -> None:
    first = _design()
    reordered = V5PaperRQMCDesign.create(
        coordinate_contract=first.coordinate_contract,
        split_specs=tuple(reversed(_specs())),
    )

    assert first == reordered
    assert V5PaperRQMCDesign.from_json(first.to_json()) == first
    assert len(first.blocks) == 8 * len(V5_PAPER_RQMC_SPLITS)
    assert len({value.split_id for value in first.blocks}) == len(V5_PAPER_RQMC_SPLITS)
    assert len({value.scramble_seed_id for value in first.blocks}) == len(first.blocks)
    assert len({value.replicate_id for value in first.blocks}) == len(first.blocks)
    assert len({value.block_id for value in first.blocks}) == len(first.blocks)
    assert json.loads(first.canonical_json)["randomization_unit"] == (
        V5_PAPER_RQMC_RANDOMIZATION_UNIT
    )
    assert V5_PAPER_RQMC_SPLITS == ("test", "reference", "ood")
    payload = json.loads(first.canonical_json)
    assert payload["finite_sample_conformal_calibration_included"] is False
    assert payload["compatibility_calibration_sampling_requirement"] == (
        V5_COMPATIBILITY_CALIBRATION_SAMPLING_REQUIREMENT
    )
    assert V5_PAPER_RQMC_CONFORMAL_CALIBRATION_INCLUDED is False
    assert V5_PAPER_RQMC_ENGINEERING_MIN_INDEPENDENT_REPLICATES == 4
    assert V5_PAPER_RQMC_MIN_INDEPENDENT_REPLICATES == 8
    assert V5_PAPER_RQMC_TARGET_INDEPENDENT_REPLICATES == 16
    assert payload["engineering_minimum_independent_replicates"] == 4
    assert payload["engineering_minimum_allows_formal_inference"] is False
    assert payload["formal_minimum_independent_replicates"] == 8
    assert payload["paper_target_independent_replicates"] == 16


def test_complete_power_two_blocks_replay_but_distinct_scrambles_differ() -> None:
    design = _design()
    first = materialize_v5_paper_rqmc_block(design, design.block("test", 0))
    replay = materialize_v5_paper_rqmc_block(design, design.block("test", 0))
    second = materialize_v5_paper_rqmc_block(design, design.block("test", 1))

    assert first == replay
    assert len(first) == 8
    assert len({value.point_id for value in first + second}) == 16
    assert np.all(np.asarray([value.unit_coordinates for value in first]) >= 0.0)
    assert np.all(np.asarray([value.unit_coordinates for value in first]) < 1.0)
    assert [value.unit_coordinates for value in first] != [
        value.unit_coordinates for value in second
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("within_scramble_points_are_iid", True, "not iid"),
        ("point_bootstrap_allowed", True, "bootstrap is forbidden"),
        (
            "engineering_minimum_allows_formal_inference",
            True,
            "replicate-count claim",
        ),
        ("formal_minimum_independent_replicates", 4, "replicate-count claim"),
        (
            "finite_sample_conformal_calibration_included",
            True,
            "separate iid contract",
        ),
        (
            "compatibility_calibration_sampling_requirement",
            "rqmc_points_are_exchangeable",
            "sampling claim",
        ),
        ("randomization_unit", "independent_iid_recipe", "inferential-unit claim"),
        ("intended_inference", "percentile_bootstrap_over_sobol_points", "inferential-unit claim"),
    ],
)
def test_design_rejects_iid_or_point_bootstrap_claims(field, value, message) -> None:
    payload = json.loads(_design().to_json())
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        V5PaperRQMCDesign.from_json(json.dumps(payload))


def test_design_fails_closed_on_too_few_replicates_non_power_two_or_reused_seed() -> None:
    with pytest.raises(ValueError, match="not a formal paper RQMC split"):
        V5PaperRQMCSplitSpec(
            split_name="calibration",
            points_per_replicate=8,
            scramble_seeds=(1, 2, 3, 4, 5, 6, 7, 8),
        )
    with pytest.raises(ValueError, match="at least eight"):
        V5PaperRQMCSplitSpec(
            split_name="test",
            points_per_replicate=8,
            scramble_seeds=(1, 2, 3, 4, 5, 6, 7),
        )
    with pytest.raises(ValueError, match="power of two"):
        V5PaperRQMCSplitSpec(
            split_name="test",
            points_per_replicate=7,
            scramble_seeds=(1, 2, 3, 4, 5, 6, 7, 8),
        )

    specs = list(_specs())
    specs[-1] = V5PaperRQMCSplitSpec(
        split_name="ood",
        points_per_replicate=8,
        scramble_seeds=(1000, 4001, 4002, 4003, 4004, 4005, 4006, 4007),
    )
    with pytest.raises(ValueError, match="globally unique"):
        V5PaperRQMCDesign.create(coordinate_contract=_coordinate_contract(), split_specs=specs)


def test_design_hash_fails_closed_on_identity_tampering() -> None:
    payload = json.loads(_design().to_json())
    payload["splits"][1]["blocks"][0]["block_id"] = "0" * 64

    with pytest.raises(ValueError, match="does not reproduce"):
        V5PaperRQMCDesign.from_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("version", "digest"),
    (("test_transform_v2", "a" * 64), ("test_transform_v1", "b" * 64)),
)
def test_authoritative_transform_identity_changes_design_despite_same_names(
    version, digest
) -> None:
    first = V5PaperRQMCDesign.create(
        coordinate_contract=_coordinate_contract(), split_specs=_specs()
    )
    changed = V5PaperRQMCDesign.create(
        coordinate_contract=_coordinate_contract(version=version, digest=digest),
        split_specs=_specs(),
    )

    assert changed.coordinate_names == first.coordinate_names
    assert changed.sha256 != first.sha256
    assert changed.blocks[0].split_id != first.blocks[0].split_id
    assert changed.blocks[0].block_id != first.blocks[0].block_id

    tampered = json.loads(first.to_json())
    tampered["coordinate_contract"]["version"] = "test_transform_v2"
    with pytest.raises(ValueError, match="does not reproduce"):
        V5PaperRQMCDesign.from_json(json.dumps(tampered))


def test_v5_recipe_convenience_constructor_uses_live_owner_identity() -> None:
    identity = v5_recipe_coordinate_contract_identity()
    design = create_v5_recipe_paper_rqmc_design(split_specs=_specs())

    assert design.coordinate_contract == identity
    assert design.coordinate_names == identity.coordinate_names
    assert design.coordinate_contract_sha256 == identity.sha256


def test_authoritative_coordinate_dimension_must_match_names() -> None:
    with pytest.raises(ValueError, match="authoritative dimension"):
        V5PaperRQMCCoordinateContractIdentity(
            schema="test.coordinates/v1",
            version="test_transform_v1",
            sha256="a" * 64,
            dimension=3,
            coordinate_names=("x", "y"),
        )
