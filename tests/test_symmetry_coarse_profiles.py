"""Global mirror-axis discovery follows scattering structure, including signed frames."""

from __future__ import annotations

import numpy as np
import pytest

from src.gimap.features.analyze.domain.symmetry_coarse import coarse_center


@pytest.mark.parametrize("width,center,sigma", [(1475, 613.3, 250.0), (3142, 1712.3, 600.0)])
@pytest.mark.parametrize("polarity,background", [(1.0, 20.0), (1.0, -1500.0), (-1.0, 0.0), (-1.0, 1500.0)])
def test_a_broad_peak_or_dip_keeps_its_axis_when_the_background_changes(width, center, sigma,
                                                                      polarity, background):
    columns = np.arange(width, dtype=np.float64)
    profile = background + polarity * 1000.0 * np.exp(-((columns - center) / sigma) ** 2)
    original = profile.copy()
    profile.setflags(write=False)

    found = coarse_center(profile)

    assert found == pytest.approx(center, abs=0.55)
    np.testing.assert_array_equal(profile, original)


@pytest.mark.parametrize("polarity,background", [(1.0, 20.0), (1.0, -1500.0), (-1.0, 0.0)])
def test_central_and_one_sided_missing_columns_do_not_create_a_false_axis(polarity, background):
    columns = np.arange(1475, dtype=np.float64)
    center = 613.3
    profile = background + polarity * 1000.0 * np.exp(-((columns - center) / 250.0) ** 2)
    profile[np.abs(columns - center) < 30] = np.nan
    profile[800:815] = np.nan
    profile[1200] = np.inf  # another unmeasured column, not a bright feature

    assert coarse_center(profile) == pytest.approx(center, abs=0.55)


def test_prominent_structure_beats_a_more_symmetric_periodic_tail():
    columns = np.arange(400, dtype=np.float64)
    center = 120.3
    offset = columns - center
    profile = 1000.0 * np.exp(-(offset / 25.0) ** 2) + 80.0 * np.cos(offset / 9.0) ** 2 + 20.0
    profile[np.abs(offset) < 30] = np.nan
    profile[250:257] = np.nan

    assert coarse_center(profile) == pytest.approx(center, abs=0.55)


@pytest.mark.parametrize("background", [20.0, -1000.0])
def test_repeated_wave_axes_are_ambiguous_even_for_signed_counts(background):
    profile = background + 80.0 * np.cos((np.arange(400) - 200.3) / 9.0) ** 2

    with pytest.raises(ValueError, match="several similarly supported"):
        coarse_center(profile)


def test_insufficient_pair_coverage_does_not_select_a_distant_wave_axis():
    offset = np.arange(400) - 320.3
    profile = 1000.0 * np.exp(-(offset / 25.0) ** 2) + 80.0 * np.cos(offset / 9.0) ** 2 + 20.0
    profile[np.abs(offset) < 30] = np.nan
    profile[250:257] = np.nan

    with pytest.raises(ValueError, match="mirror-symmetric|structured measured pixels"):
        coarse_center(profile)


def test_a_running_global_scan_can_be_cancelled():
    offset = np.arange(3142) - 1712.3
    profile = 20.0 + 1000.0 * np.exp(-(offset / 600.0) ** 2)
    checks = iter([False, False, True])

    with pytest.raises(ValueError, match="cancelled"):
        coarse_center(profile, cancelled=lambda: next(checks))
