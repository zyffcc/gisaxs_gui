from __future__ import annotations

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_catalog import (
    branch_pattern_id,
    decode_branch_pattern,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contextual_branch_catalog import (
    CONTEXTUAL_BRANCH_CATALOG_SCHEMA,
    CONTEXTUAL_BRANCH_CATALOG_SEMANTICS,
    CONTEXTUAL_BRANCH_CATALOG_VERSION,
    build_contextual_branch_catalog,
    contextual_canonical_branch_pattern_id,
    contextual_canonicalize_d_flags,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    ClosedInterval,
    GuiComponentBounds,
    SPHERE,
    full_component_bounds,
)

import pytest


def _overlapping_sphere_bounds(*, shifted: bool = False) -> GuiComponentBounds:
    if shifted:
        radius = ClosedInterval(20.0, 40.0)
        sigma_radius = ClosedInterval(2.0, 6.0)
    else:
        radius = ClosedInterval(10.0, 30.0)
        sigma_radius = ClosedInterval(1.0, 4.0)
    return GuiComponentBounds(
        shape=SPHERE,
        R=radius,
        sigma_R=sigma_radius,
        D=ClosedInterval(50.0, 100.0),
        sigma_D=ClosedInterval(5.0, 20.0),
        allow_D_absent=True,
    )


def _physical_branch_set(catalog) -> set[tuple[tuple[tuple[str, bool], ...], bool]]:
    result = set()
    for pattern_id in catalog.wire_pattern_ids:
        d_present, resolution_present = decode_branch_pattern(pattern_id)
        components = tuple(
            sorted(
                (repr(bounds), present)
                for bounds, present in zip(catalog.component_bounds, d_present)
            )
        )
        result.add((components, resolution_present))
    return result


def test_identical_repeated_spheres_are_quotiented_but_heterogeneous_ones_are_not():
    identical = full_component_bounds(SPHERE, d_policy="optional")
    equal_intensities = (ClosedInterval(0.0, 1.0),) * 2
    quotient = build_contextual_branch_catalog(
        (identical, identical),
        component_intensity_bounds=equal_intensities,
        resolution_presence_policy="absent",
    )
    assert quotient.d_equivalence_classes == ((0, 1),)
    assert quotient.wire_pattern_ids == (0, 2, 3)
    assert contextual_canonicalize_d_flags(
        (identical, identical),
        (True, False, False, False),
        component_intensity_bounds=equal_intensities,
    ) == (False, True, False, False)

    first = _overlapping_sphere_bounds()
    second = _overlapping_sphere_bounds(shifted=True)
    heterogeneous = build_contextual_branch_catalog(
        (first, second),
        component_intensity_bounds=equal_intensities,
        resolution_presence_policy="absent",
    )
    assert heterogeneous.d_equivalence_classes == ((0,), (1,))
    assert heterogeneous.wire_pattern_ids == (0, 1, 2, 3)
    assert 1 in heterogeneous.wire_pattern_ids
    assert 2 in heterogeneous.wire_pattern_ids
    assert contextual_canonicalize_d_flags(
        (first, second),
        (True, False, False, False),
        component_intensity_bounds=equal_intensities,
    ) == (True, False, False, False)


def test_unequal_amplitude_ranges_preserve_wire_branches_and_missing_context_fails_closed():
    bounds = full_component_bounds(SPHERE, d_policy="optional")
    unequal = (ClosedInterval(0.0, 0.4), ClosedInterval(0.6, 1.0))

    amplitude_aware = build_contextual_branch_catalog(
        (bounds, bounds),
        component_intensity_bounds=unequal,
        resolution_presence_policy="absent",
    )
    geometry_only = build_contextual_branch_catalog(
        (bounds, bounds), resolution_presence_policy="absent"
    )

    assert amplitude_aware.d_equivalence_classes == ((0,), (1,))
    assert amplitude_aware.wire_pattern_ids == (0, 1, 2, 3)
    assert geometry_only.d_equivalence_classes == ((0,), (1,))
    assert geometry_only.wire_pattern_ids == (0, 1, 2, 3)
    assert not geometry_only.audit_payload()["exchangeability_context_complete"]


def test_component_d_presence_policies_are_applied_per_slot():
    bounds = (
        full_component_bounds(SPHERE, d_policy="absent"),
        full_component_bounds(SPHERE, d_policy="required"),
        full_component_bounds(SPHERE, d_policy="optional"),
    )
    catalog = build_contextual_branch_catalog(bounds, resolution_presence_policy="absent")
    assert catalog.d_policies == ("absent", "required", "optional")
    assert catalog.wire_pattern_ids == (2, 6)

    with pytest.raises(ValueError, match="absent D policy"):
        contextual_canonicalize_d_flags(bounds, (True, True, False, False))
    with pytest.raises(ValueError, match="required D policy"):
        contextual_canonicalize_d_flags(bounds, (False, False, False, False))


def test_resolution_absent_optional_and_required_preserve_the_wire_bit():
    bounds = (full_component_bounds(SPHERE, d_policy="optional"),)
    absent = build_contextual_branch_catalog(bounds, resolution_presence_policy="absent")
    optional = build_contextual_branch_catalog(bounds, resolution_presence_policy="optional")
    required = build_contextual_branch_catalog(bounds, resolution_presence_policy="required")

    assert absent.wire_pattern_ids == (0, 1)
    assert optional.wire_pattern_ids == (0, 1, 16, 17)
    assert required.wire_pattern_ids == (16, 17)
    assert (
        contextual_canonical_branch_pattern_id(bounds, 17, resolution_presence_policy="required")
        == 17
    )
    with pytest.raises(ValueError, match="required Resolution policy"):
        contextual_canonical_branch_pattern_id(bounds, 1, resolution_presence_policy="required")


def test_masks_are_exactly_32_wire_entries_and_audit_identity_is_stable():
    bounds = (full_component_bounds(SPHERE, d_policy="absent"),)
    catalog = build_contextual_branch_catalog(bounds, resolution_presence_policy="optional")
    assert len(catalog.wire_pattern_mask) == 32
    assert (
        tuple(index for index, selected in enumerate(catalog.wire_pattern_mask) if selected)
        == catalog.wire_pattern_ids
        == (0, 16)
    )

    audit = catalog.audit_payload()
    assert audit["catalog_schema"] == CONTEXTUAL_BRANCH_CATALOG_SCHEMA
    assert audit["catalog_version"] == CONTEXTUAL_BRANCH_CATALOG_VERSION
    assert audit["semantics"] == CONTEXTUAL_BRANCH_CATALOG_SEMANTICS
    assert audit["wire_pattern_ids"] == [0, 16]


def test_slot_permutations_preserve_the_physical_branch_set():
    repeated = _overlapping_sphere_bounds()
    distinct = _overlapping_sphere_bounds(shifted=True)
    left = build_contextual_branch_catalog(
        (repeated, repeated, distinct),
        component_intensity_bounds=(
            ClosedInterval(0.0, 0.4),
            ClosedInterval(0.0, 0.4),
            ClosedInterval(0.6, 1.0),
        ),
        resolution_presence_policy="optional",
    )
    right = build_contextual_branch_catalog(
        (distinct, repeated, repeated),
        component_intensity_bounds=(
            ClosedInterval(0.6, 1.0),
            ClosedInterval(0.0, 0.4),
            ClosedInterval(0.0, 0.4),
        ),
        resolution_presence_policy="optional",
    )

    assert left.d_equivalence_classes == ((0, 1), (2,))
    assert right.d_equivalence_classes == ((0,), (1, 2))
    assert len(left.wire_pattern_ids) == len(right.wire_pattern_ids) == 12
    assert _physical_branch_set(left) == _physical_branch_set(right)


@pytest.mark.parametrize("component_count", range(1, 5))
def test_identical_optional_sphere_boundaries_k1_through_k4(component_count):
    bounds = full_component_bounds(SPHERE, d_policy="optional")
    catalog = build_contextual_branch_catalog(
        (bounds,) * component_count,
        component_intensity_bounds=(ClosedInterval(0.0, 1.0),) * component_count,
        resolution_presence_policy="optional",
    )
    assert len(catalog.wire_pattern_ids) == 2 * (component_count + 1)
    assert sum(catalog.wire_pattern_mask) == len(catalog.wire_pattern_ids)
    for pattern_id in catalog.wire_pattern_ids:
        flags, _ = decode_branch_pattern(pattern_id)
        assert not any(flags[component_count:])
        assert (
            contextual_canonical_branch_pattern_id(
                catalog.component_bounds,
                pattern_id,
                component_intensity_bounds=catalog.component_intensity_bounds,
                resolution_presence_policy="optional",
            )
            == pattern_id
        )


def test_context_and_policy_inputs_fail_closed():
    optional = full_component_bounds(SPHERE, d_policy="optional")
    with pytest.raises(ValueError, match="between one and four"):
        build_contextual_branch_catalog((), resolution_presence_policy="absent")
    with pytest.raises(TypeError, match="GuiComponentBounds"):
        build_contextual_branch_catalog((object(),), resolution_presence_policy="absent")
    with pytest.raises(ValueError, match="exactly four"):
        contextual_canonicalize_d_flags((optional,), (False,))
    with pytest.raises(TypeError, match="boolean"):
        contextual_canonicalize_d_flags((optional,), (0, False, False, False))
    with pytest.raises(ValueError, match="unused"):
        contextual_canonicalize_d_flags((optional,), (False, True, False, False))
    with pytest.raises(TypeError, match="string"):
        build_contextual_branch_catalog((optional,), resolution_presence_policy=True)
    with pytest.raises(ValueError, match="absent, optional, or required"):
        build_contextual_branch_catalog((optional,), resolution_presence_policy="sometimes")
    with pytest.raises(ValueError, match="one interval per component"):
        build_contextual_branch_catalog(
            (optional,),
            component_intensity_bounds=(),
            resolution_presence_policy="absent",
        )
    with pytest.raises(TypeError, match="ClosedInterval"):
        build_contextual_branch_catalog(
            (optional,),
            component_intensity_bounds=(object(),),
            resolution_presence_policy="absent",
        )
    with pytest.raises(ValueError, match=r"\[0, 31\]"):
        contextual_canonical_branch_pattern_id(
            (optional,), 32, resolution_presence_policy="optional"
        )


def test_canonical_pattern_id_uses_context_not_shape_alone():
    identical = full_component_bounds(SPHERE, d_policy="optional")
    one_present_in_first_slot = branch_pattern_id((True, False, False, False), False)
    assert contextual_canonical_branch_pattern_id(
        (identical, identical),
        one_present_in_first_slot,
        component_intensity_bounds=(ClosedInterval(0.0, 1.0),) * 2,
        resolution_presence_policy="absent",
    ) == branch_pattern_id((False, True, False, False), False)

    heterogeneous = (_overlapping_sphere_bounds(), _overlapping_sphere_bounds(shifted=True))
    assert (
        contextual_canonical_branch_pattern_id(
            heterogeneous,
            one_present_in_first_slot,
            resolution_presence_policy="absent",
        )
        == one_present_in_first_slot
    )
