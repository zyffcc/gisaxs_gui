"""The real-space picture of a whole scan as one growing film (``domain.real_space.grown``): the same field in every
window, grains that stay where they are and grow, merge when the distance D grows and are joined by new nuclei when it
shrinks."""

import math

from src.gimap.features.sputter.domain.real_space import RealSpace, arrangement, cut, grown, series_field


def _series():
    return [RealSpace(1.0, 4.0, 0.25, 1.6, 0.0),      # nucleation
            RealSpace(1.3, 4.0, 0.25, 2.1, 0.0),      # the same nuclei grow
            RealSpace(2.4, 8.0, 0.25, 3.6, 0.0),      # coalescence: D doubles
            None,                                     # a window without a solution
            RealSpace(2.6, 6.0, 0.25, 4.0, 0.0),      # new nuclei: D shrinks again
            RealSpace(3.0, 8.0, 0.25, 5.5, 1.5)]      # a closed film


def _centres(grains):
    return {(round(grain.x, 9), round(grain.y, 9)) for grain in grains}


def test_every_window_has_the_same_field():
    spaces = _series()
    fields = {grown(spaces, index)[0].field for index, space in enumerate(spaces) if space is not None}
    assert fields == {series_field(spaces)}
    assert grown(spaces, 3) is None


def test_grains_stay_and_grow_while_d_stays():
    spaces = _series()
    _, first = grown(spaces, 0)
    _, second = grown(spaces, 1)
    assert _centres(first) == _centres(second)
    by_place = {(round(grain.x, 9), round(grain.y, 9)): grain.radius for grain in first}
    assert all(grain.radius > by_place[(round(grain.x, 9), round(grain.y, 9))] for grain in second)


def test_grains_merge_when_d_grows_and_nuclei_join_when_it_shrinks():
    spaces = _series()
    field = series_field(spaces)
    count = lambda space: 2.0 / (math.sqrt(3.0) * space.distance ** 2) * field * field  # noqa: E731
    _, before = grown(spaces, 1)
    _, merged = grown(spaces, 2)
    _, renewed = grown(spaces, 4)
    assert len(merged) < len(before)
    assert abs(len(merged) - count(spaces[2])) <= 0.1 * count(spaces[2]) + 2      # visible grains, edges clipped
    assert len(renewed) > len(merged)
    assert _centres(merged) <= _centres(renewed)       # every grain of the window before is still there


def test_the_series_reaches_a_network_and_the_cut_shows_its_grains():
    spaces = _series()
    space, grains = grown(spaces, 5)
    assert space.continuous and arrangement(space, grains) in ("coalescing", "network")
    row = cut(grains, space.field / 2.0)
    assert row and all(0.0 < grain.radius <= max(g.radius for g in grains) for grain in row)
    assert [grain.x for grain in row] == sorted(grain.x for grain in row)


def test_the_after_deposition_report_names_stages_from_the_growing_series():
    from src.gimap.features.sputter.application.static_report import FILM, series_stages, stage

    rows = [dict(R="1.0", d_hex="4.0", height="1.6", t_b="0"), dict(R="2.4", d_hex="8.0", height="3.6", t_b="0"),
            dict(R="3.0", d_hex="8.0", height="5.5", t_b="2.0", T="1.0", rb="0.95")]
    named = series_stages(rows)
    assert len(named) == 3
    assert named[2] == stage("3.0", "8.0", "5.5", "2.0", "1.0", "0.95") == FILM   # a continuous layer keeps its own


def test_the_film_closes_with_channels_and_never_opens_up_again():
    from src.gimap.features.sputter.domain.real_space import CLOSED_COVER, drawn_coverage, height_map

    spaces = [RealSpace(1.0, 4.0, 0.25, 1.6, 0.0, cover=0.2), RealSpace(2.0, 6.0, 0.25, 3.0, 0.0, cover=0.6),
              RealSpace(2.2, 6.5, 0.25, 3.2, 0.0, cover=0.5),                 # the solution's coverage dips
              RealSpace(3.0, 8.0, 0.25, 5.0, 2.0, cover=0.7)]                 # a continuous layer
    covers = [drawn_coverage(*grown(spaces, index)) for index in range(len(spaces))]
    assert covers == sorted(covers)                                           # drawn coverage never falls
    assert abs(covers[1] - 0.6) < 0.05 and covers[3] >= CLOSED_COVER - 0.05
    space, grains = grown(spaces, 3)
    _, _, height = height_map(space, grains)
    holes = height <= 0.0
    assert space.channels and 0.0 < holes.mean() <= 1.0 - CLOSED_COVER + 0.05   # channels down to the substrate
    assert height[~holes].min() >= space.base                                 # the layer under the metal footprint


def test_the_cover_scale_is_that_of_comparing_every_grain():
    """The look-up of the nearest grains (2026-11-24, a slow page) gives the scale of the full comparison."""
    import importlib

    import numpy as np

    real_space = importlib.import_module("src.gimap.features.sputter.domain.real_space")
    rng = np.random.default_rng(7)
    for count, width, cover in ((0, 0.3, 0.5), (5, 0.3, 0.5), (40, 0.0, 0.2), (300, 0.3, 0.6), (900, 0.8, 0.9)):
        field = 40.0 + count / 10
        x, y = rng.uniform(0, field, count), rng.uniform(0, field, count)
        z = np.clip(rng.normal(size=count), -2.5, 2.5)
        axis = (np.arange(real_space.COVER_GRID) + 0.5) * field / real_space.COVER_GRID
        gx, gy = (a.ravel() for a in np.meshgrid(axis, axis))
        need = np.full(gx.size, np.inf)
        if count:
            need = (np.hypot(gx[:, None] - x[None, :], gy[:, None] - y[None, :]) * np.exp(-width * z)[None, :]).min(1)
        with np.errstate(invalid="ignore"):
            full = float(np.quantile(need, cover))
            fast = real_space._cover_scale(x, y, z, width, field, cover)
        assert (math.isinf(full) and math.isinf(fast)) or (math.isnan(full) and math.isnan(fast)) or \
            math.isclose(fast, full, rel_tol=1e-12)
