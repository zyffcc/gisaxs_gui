"""The refined axis should align the measured central scattering in Cuts.

These frames separate the central scattering axis from weak outer structure
and from the median of rows.  Subpixel tolerances allow interpolation error;
they are independent of the optimiser's numerical stopping tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.gimap.features.analyze.domain.binning import native_profile
from src.gimap.features.analyze.domain.symmetry import SymmetryCenter, symmetric_center_x


WIDTH = 1600
AXIS_INDEX = 700.3
"""Synthetic column-index axis; canonical pixel coordinates add 0.5."""


def _central_scattering(columns, axis, amplitude=10000.0):
    return amplitude / (1.0 + ((columns - axis) / 5.0) ** 2) ** 2 + 10.0


def _displayed_profile(image, valid):
    coordinates = np.broadcast_to(np.arange(image.shape[1], dtype=float), image.shape)
    columns, intensities, _, _ = native_profile(
        image, valid, coordinates, axis=0, counts_model=False,
    )
    return columns, intensities


def _central_mismatch(columns, intensities, canonical_axis):
    """Median fractional difference over measured offsets 3–10 pixels."""
    axis = canonical_axis - 0.5
    distances = np.arange(3.0, 10.0, 0.5)
    left = np.interp(axis - distances, columns, intensities)
    right = np.interp(axis + distances, columns, intensities)
    return float(np.median(2.0 * np.abs(left - right) / (left + right)))


@pytest.mark.parametrize("tail_amplitude", [20.0, 60.0, 120.0])
def test_weak_outer_structure_does_not_sacrifice_the_low_q_overlap(tail_amplitude):
    columns = np.arange(WIDTH, dtype=float)
    offset = columns - AXIS_INDEX
    # The central scattering is exactly symmetric.  A weak outer component
    # has its own axis, as can happen with sample/background contributions.
    gate = 1.0 / (1.0 + np.exp(-(np.abs(offset) - 25.0) / 3.0))
    tail = gate * tail_amplitude * (1.5 + np.cos((offset - 0.45) / 4.0) ** 2)
    image = np.tile(_central_scattering(columns, AXIS_INDEX) + tail, (5, 1))
    valid = np.ones(image.shape, dtype=bool)

    found = symmetric_center_x(image, valid, (0, 5), AXIS_INDEX + 20.5)
    measured_columns, measured_mean = _displayed_profile(image, valid)

    # A 0.04-pixel displacement gives about 3% median mismatch over this
    # steep-core range.  The previous uniform-distance score shifted >0.2 px.
    assert found.x_px == pytest.approx(AXIS_INDEX + 0.5, abs=0.04)
    assert _central_mismatch(measured_columns, measured_mean, found.x_px) < 0.03


def test_refinement_follows_the_displayed_row_mean_when_rows_have_different_axes():
    columns = np.arange(WIDTH, dtype=float)
    bright = _central_scattering(columns, AXIS_INDEX, amplitude=10000.0)
    faint = _central_scattering(columns, AXIS_INDEX + 0.25, amplitude=100.0)
    image = np.vstack((bright, np.tile(faint, (9, 1))))
    valid = np.ones(image.shape, dtype=bool)

    found = symmetric_center_x(image, valid, (0, 10), AXIS_INDEX + 20.5)
    measured_columns, measured_mean = _displayed_profile(image, valid)

    # For a shift small relative to the core width, the intensity-weighted
    # axis of this mean is displaced by 9*100*0.25/(10000+9*100) = 0.0206 px.
    # The median instead follows the nine faint rows, displaced by 0.25 px.
    mean_axis = AXIS_INDEX + 0.5 + 9.0 * 100.0 * 0.25 / (10000.0 + 9.0 * 100.0)
    assert found.x_px == pytest.approx(mean_axis, abs=0.02)
    assert _central_mismatch(measured_columns, measured_mean, found.x_px) < 0.02


def test_one_sided_gap_edge_cannot_pull_the_axis_away_from_symmetric_scattering():
    columns = np.arange(WIDTH, dtype=float)
    profile = _central_scattering(columns, AXIS_INDEX)
    # A bright module-edge pixel sits immediately beside an unmeasured gap.
    # Its pairing changes with the trial axis; dropping difficult pairs must
    # not make a displaced centre score better than the symmetric core.
    profile[int(AXIS_INDEX + 7)] += 1000.0
    image = np.tile(profile, (5, 1))
    valid = np.ones(image.shape, dtype=bool)
    valid[:, (columns - AXIS_INDEX >= 8.0) & (columns - AXIS_INDEX <= 20.0)] = False
    poisoned = image.copy()
    poisoned[~valid] = 1e15

    found = symmetric_center_x(image, valid, (0, 5), AXIS_INDEX + 0.5)
    again = symmetric_center_x(poisoned, valid, (0, 5), AXIS_INDEX + 0.5)

    assert found.x_px == pytest.approx(AXIS_INDEX + 0.5, abs=0.02)
    assert again.x_px == pytest.approx(found.x_px, abs=1e-8)


def test_signed_float_pixels_and_masks_are_immutable_during_refinement():
    columns = np.arange(WIDTH, dtype=float)
    profile = 300.0 * np.exp(-((columns - AXIS_INDEX) / 38.0) ** 2) - 500.0
    image = np.tile(profile, (4, 1))
    image[0, 91] = np.nan
    image[1, 103] = np.inf
    valid = np.ones(image.shape, dtype=bool)
    valid[:, 900:909] = False
    original = image.copy()
    original_valid = valid.copy()
    image.setflags(write=False)
    valid.setflags(write=False)

    found = symmetric_center_x(image, valid, (0, 4), AXIS_INDEX - 19.5)

    assert found.x_px == pytest.approx(AXIS_INDEX + 0.5, abs=0.01)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(valid, original_valid)


def test_assistant_applies_and_undoes_a_small_but_resolved_centre_shift():
    from tests.assistant_fakes import call
    from tests.test_assistant_operations import catalog

    initial = (140.5, 800.25)
    refined = (initial[0] + 0.02, initial[1])
    result = SymmetryCenter(
        x_px=refined[0], initial_x_px=initial[0], loss_before=0.01,
        loss_after=0.005, paired_samples=100, search_px=4.0,
        coarse_x_px=initial[0],
    )

    class Workbench:
        centre = initial

        def status(self):
            return {
                "measurement": "gisaxs", "gisaxs": {"horizontal_rows": [10, 15]},
                "geometry": {"beam_center_px": list(self.centre)},
            }

        def symmetry_center(self):
            return result.to_dict()

        def set_beam_center(self, x, y):
            self.centre = (x, y)
            return self.status()

    workbench = Workbench()
    tools = catalog(workbench)
    outcome = tools.execute(call("refine_beam_center_symmetry"))

    assert not outcome.is_error
    assert outcome.data["changed"] is True
    assert workbench.centre == pytest.approx(refined, abs=1e-10)
    assert tools.undo_all_operations() == 1
    assert workbench.centre == initial


def test_pipeline_geometry_report_preserves_the_applied_subpixel_centre():
    import json

    from src.gimap.features.assistant.application import PipelineOptions, StandardPipeline
    from tests.assistant_fakes import call
    from tests.test_assistant_operations import catalog

    initial = (140.5, 800.25345678)
    refined_x = 140.52345678

    class Workbench:
        centre = initial

        def status(self):
            return {
                "measurement": "gisaxs", "gisaxs": {"horizontal_rows": [10, 15]},
                "geometry": {"beam_center_px": list(self.centre)},
            }

        def symmetry_center(self):
            return SymmetryCenter(
                x_px=refined_x, initial_x_px=initial[0], loss_before=0.01,
                loss_after=0.005, paired_samples=100, search_px=4.0,
                coarse_x_px=initial[0],
            ).to_dict()

        def set_beam_center(self, x, y):
            self.centre = (x, y)
            return self.status()

    workbench = Workbench()
    tools = catalog(workbench)
    outcome = tools.execute(call("refine_beam_center_symmetry"))
    assert not outcome.is_error and outcome.data["changed"] is True

    report = StandardPipeline(tools, PipelineOptions(fit=False))._report()
    exported = json.loads(json.dumps(report, allow_nan=False))

    # The saved geometry is used to reproduce the reduction.  Four significant
    # digits would erase the applied shift and also round the unchanged y.
    assert exported["geometry"]["beam_center_px"] == pytest.approx(
        workbench.centre, abs=1e-12, rel=0.0,
    )
    assert workbench.centre == (refined_x, initial[1])
