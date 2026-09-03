from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PosteriorV8.contract import (
    CYLINDER,
    NUM_TOPOLOGIES,
    SPHERE,
    TOPOLOGIES,
    VERTICAL_CYLINDER,
    ClosedInterval,
    GuiComponentBounds,
    GuiComponentParameters,
    canonical_topology,
    full_component_bounds,
    gui_bounds_to_latent,
    gui_component_to_latent,
    latent_bounds_to_gui,
    latent_component_to_gui,
    topology_bounds_to_gui,
    topology_bounds_to_latent,
    topology_from_id,
    topology_id_for,
)


def test_topology_catalog_has_exactly_34_unordered_k1_to_k4_classes():
    assert NUM_TOPOLOGIES == 34
    assert len(set(TOPOLOGIES)) == 34
    assert Counter(map(len, TOPOLOGIES)) == {1: 3, 2: 6, 3: 10, 4: 15}
    for topology_id, topology in enumerate(TOPOLOGIES):
        assert topology_from_id(topology_id) == topology
        assert topology_id_for(reversed(topology)) == topology_id
    assert canonical_topology(["Vertical Cylinder", "sphere", "cylinder"]) == (
        SPHERE,
        CYLINDER,
        VERTICAL_CYLINDER,
    )


@pytest.mark.parametrize("bad_id", [-1, 34, 1.0, 1.5, True])
def test_topology_id_validation_is_strict(bad_id):
    with pytest.raises((TypeError, ValueError)):
        topology_from_id(bad_id)


@pytest.mark.parametrize(
    "component",
    [
        GuiComponentParameters(SPHERE, R=20.0, sigma_R=2.0, D=50.0, sigma_D=5.0),
        GuiComponentParameters(
            CYLINDER,
            R=12.0,
            sigma_R=1.2,
            h=80.0,
            sigma_h=16.0,
            D=60.0,
            sigma_D=6.0,
        ),
        GuiComponentParameters(VERTICAL_CYLINDER, R=30.0, sigma_R=0.15),
    ],
)
def test_gui_latent_component_roundtrip_uses_log_sizes_and_fraction_widths(component):
    latent = gui_component_to_latent(component)
    assert latent.log_R == pytest.approx(np.log(component.R))
    expected_sigma_r = (
        component.sigma_R
        if component.shape == VERTICAL_CYLINDER
        else component.sigma_R / component.R
    )
    assert latent.sigma_R_fraction == pytest.approx(expected_sigma_r)
    if component.h is not None:
        assert latent.log_h == pytest.approx(np.log(component.h))
        assert latent.sigma_h_fraction == pytest.approx(component.sigma_h / component.h)
    if component.D is not None:
        assert latent.log_D == pytest.approx(np.log(component.D))
        assert latent.sigma_D_fraction == pytest.approx(component.sigma_D / component.D)
    recovered = latent_component_to_gui(latent)
    assert recovered.shape == component.shape
    assert recovered.R == pytest.approx(component.R)
    assert recovered.sigma_R == pytest.approx(component.sigma_R)
    assert recovered.h == pytest.approx(component.h) if component.h is not None else recovered.h is None
    assert recovered.sigma_h == pytest.approx(component.sigma_h) if component.sigma_h is not None else recovered.sigma_h is None
    assert recovered.D == pytest.approx(component.D) if component.D is not None else recovered.D is None
    assert recovered.sigma_D == pytest.approx(component.sigma_D) if component.sigma_D is not None else recovered.sigma_D is None


def test_gui_zero_d_pair_is_canonicalized_to_absent():
    component = GuiComponentParameters(SPHERE, R=10.0, sigma_R=1.0, D=0.0, sigma_D=0.0)
    assert component.D is None and component.sigma_D is None
    latent = gui_component_to_latent(component)
    assert latent.log_D is None and latent.sigma_D_fraction is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shape": SPHERE, "R": 10.0, "sigma_R": 1.0, "D": 20.0},
        {"shape": CYLINDER, "R": 10.0, "sigma_R": 1.0},
        {"shape": SPHERE, "R": 10.0, "sigma_R": 20.0},
        {"shape": VERTICAL_CYLINDER, "R": 10.0, "sigma_R": 2.0},
    ],
)
def test_invalid_gui_component_parameters_fail_closed(kwargs):
    with pytest.raises(ValueError):
        GuiComponentParameters(**kwargs)


def test_physical_bounds_convert_losslessly_and_keep_exact_fraction_coupling():
    gui = GuiComponentBounds(
        SPHERE,
        R=ClosedInterval(10.0, 30.0),
        sigma_R=ClosedInterval(1.0, 6.0),
        D=ClosedInterval(40.0, 80.0),
        sigma_D=ClosedInterval(4.0, 16.0),
        allow_D_absent=True,
    )
    latent = gui_bounds_to_latent(gui)
    assert latent.d_policy == "optional"
    assert latent.log_R.low == pytest.approx(np.log(10.0))
    assert latent.log_R.high == pytest.approx(np.log(30.0))
    assert latent.sigma_R_fraction.envelope.low == pytest.approx(1.0 / 30.0)
    assert latent.sigma_R_fraction.envelope.high == pytest.approx(6.0 / 10.0)
    assert latent_bounds_to_gui(latent) == gui

    inside = gui_component_to_latent(
        GuiComponentParameters(SPHERE, R=20.0, sigma_R=3.0, D=50.0, sigma_D=5.0)
    )
    absent = gui_component_to_latent(GuiComponentParameters(SPHERE, R=20.0, sigma_R=3.0))
    outside_coupled_width = gui_component_to_latent(
        GuiComponentParameters(SPHERE, R=30.0, sigma_R=9.0, D=50.0, sigma_D=5.0)
    )
    assert latent.contains(inside)
    assert latent.contains(absent)
    assert not latent.contains(outside_coupled_width)


def test_vertical_sigma_r_bounds_are_already_fractional():
    gui = GuiComponentBounds(
        VERTICAL_CYLINDER,
        R=ClosedInterval(5.0, 25.0),
        sigma_R=ClosedInterval(0.10, 0.30),
    )
    latent = gui_bounds_to_latent(gui)
    assert latent.sigma_R_fraction.envelope == ClosedInterval(0.10, 0.30)
    assert latent_bounds_to_gui(latent) == gui


def test_topology_bounds_preserve_per_component_ranges_and_canonicalize_shapes():
    cylinder = GuiComponentBounds(
        CYLINDER,
        R=ClosedInterval(8.0, 20.0),
        sigma_R=ClosedInterval(1.0, 3.0),
        h=ClosedInterval(30.0, 100.0),
        sigma_h=ClosedInterval(3.0, 20.0),
    )
    sphere = GuiComponentBounds(
        SPHERE,
        R=ClosedInterval(10.0, 40.0),
        sigma_R=ClosedInterval(1.0, 8.0),
    )
    latent = topology_bounds_to_latent([cylinder, sphere])
    assert tuple(item.shape for item in latent) == (SPHERE, CYLINDER)
    assert topology_id_for(item.shape for item in latent) == topology_id_for([CYLINDER, SPHERE])
    assert topology_bounds_to_gui(latent) == (sphere, cylinder)


@pytest.mark.parametrize("shape", [SPHERE, CYLINDER, VERTICAL_CYLINDER])
@pytest.mark.parametrize("d_policy", ["absent", "required", "optional"])
def test_full_component_bounds_cover_the_versioned_latent_domain(shape, d_policy):
    bounds = full_component_bounds(shape, d_policy=d_policy)
    latent = gui_bounds_to_latent(bounds)
    assert latent.shape == shape
    assert latent.d_policy == d_policy
    assert latent.log_R.physical == ClosedInterval(1.0, 100.0)
    if shape == CYLINDER:
        assert latent.log_h.physical == ClosedInterval(2.0, 500.0)
    if d_policy == "absent":
        assert latent.log_D is None
    else:
        assert latent.log_D.physical == ClosedInterval(3.0, 500.0)


def test_full_component_bounds_reject_unknown_d_policy():
    with pytest.raises(ValueError, match="d_policy"):
        full_component_bounds(SPHERE, d_policy="maybe")


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "shape": SPHERE,
            "R": ClosedInterval(0.5, 10.0),
            "sigma_R": ClosedInterval(1.0, 2.0),
        },
        {
            "shape": CYLINDER,
            "R": ClosedInterval(10.0, 20.0),
            "sigma_R": ClosedInterval(1.0, 2.0),
        },
        {
            "shape": SPHERE,
            "R": ClosedInterval(10.0, 20.0),
            "sigma_R": ClosedInterval(1.0, 2.0),
            "D": ClosedInterval(30.0, 40.0),
        },
        {
            "shape": SPHERE,
            "R": ClosedInterval(10.0, 20.0),
            "sigma_R": ClosedInterval(1.0, 2.0),
            "allow_D_absent": True,
        },
    ],
)
def test_invalid_physical_bounds_fail_closed(kwargs):
    with pytest.raises(ValueError):
        GuiComponentBounds(**kwargs)
