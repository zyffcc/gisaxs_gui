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
