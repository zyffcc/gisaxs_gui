from utils.ai_fitting_constraints import ConstraintSet, exclusion_size


def component(kind, **params):
    return {"type": kind, "params": params}


def test_geometry_specific_exclusion_sizes():
    assert exclusion_size("sphere", {"R": 10.0}) == 20.0
    assert exclusion_size("vertical_cylinder", {"R": 10.0}) == 20.0
    assert exclusion_size("cylinder", {"R": 3.0, "h": 8.0}) == 10.0


def test_optional_d_zero_is_valid_for_every_geometry():
    constraints = ConstraintSet.defaults()
    rows = [
        component("sphere", R=10.0, sigma_R=1.0, D=0.0, sigma_D=0.0),
        component("cylinder", R=3.0, sigma_R=0.2, h=8.0, sigma_h=0.5, D=0.0, sigma_D=0.0),
        component("vertical_cylinder", R=10.0, sigma_R=0.2, D=0.0, sigma_D=0.0),
    ]
    assert constraints.validate_components(rows) == []


def test_hard_core_boundaries_report_the_specific_formula():
    constraints = ConstraintSet.defaults()
    sphere = component("sphere", R=10.0, sigma_R=1.0, D=20.0, sigma_D=1.0)
    cylinder = component("cylinder", R=3.0, sigma_R=0.2, h=8.0, sigma_h=0.5, D=10.0, sigma_D=0.5)
    sphere_violations = constraints.validate_components([sphere])
    cylinder_violations = constraints.validate_components([cylinder])
    assert any(item.constraint_id == "hard_core_spacing" and "2R" in item.formula for item in sphere_violations)
    assert any("sqrt" in item.formula for item in cylinder_violations)


def test_vertical_distribution_is_fractional_but_sphere_width_is_absolute():
    constraints = ConstraintSet.defaults()
    valid_vertical = component("vertical_cylinder", R=2.0, sigma_R=0.9, D=0.0, sigma_D=0.0)
    invalid_vertical = component("vertical_cylinder", R=100.0, sigma_R=0.91, D=0.0, sigma_D=0.0)
    valid_sphere = component("sphere", R=10.0, sigma_R=9.0, D=0.0, sigma_D=0.0)
    assert constraints.validate_components([valid_vertical]) == []
    assert constraints.validate_components([valid_sphere]) == []
    assert any(item.constraint_id == "size_distribution" for item in constraints.validate_components([invalid_vertical]))
