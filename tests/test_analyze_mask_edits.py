"""Synthetic detector masks. Expected sets are given as (row, column) pixel indices."""

from dataclasses import replace
import json
import math

import numpy as np
import pytest

from src.gimap.features.analyze.application import (
    AnalysisRequest, AnalyzeFrame, AnalyzeSettings, Corrections, MaskImage, MaskRule, MaskShape,
    analysis_metadata, settings_from_record, settings_record,
)
from src.gimap.features.analyze.domain.mask_raster import evaluate_mask, rasterize
from src.gimap.shared.geometry import DetectorGeometry
from src.gimap.shared.geometry.q_mapping import transmission_map
from tests.test_analyze_giwaxs_options import FakeFrames


GRID = (8, 9)
CROSS = {(3, 3), (4, 2), (4, 3), (4, 4), (5, 3)}
DISK = CROSS | {(2, 3), (3, 2), (3, 4), (4, 1), (4, 5), (5, 2), (5, 4), (6, 3)}


@pytest.mark.parametrize(("shape", "expected"), [
    (MaskShape("rectangle", ((1, 2), (3, 5))), {(r, c) for r in (2, 3, 4) for c in (1, 2)}),
    (MaskShape("polygon", ((1, 1), (4.1, 1), (1, 4.1))), {(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 1)}),
    (MaskShape("circle", ((3.5, 4.5), (5.5, 4.5))), DISK),
    (MaskShape("annulus", ((3.5, 4.5), (3.5, 3.5), (5.5, 4.5))), DISK - {(4, 3)}),
    (MaskShape("ellipse", ((1.5, 3.5), (5.5, 5.5))), CROSS | {(4, 1), (4, 5)}),
    (MaskShape("brush", ((1.5, 3.5), (4.5, 3.5)), radius=1),
     {(r, c) for r in (2, 4) for c in range(1, 5)} | {(3, c) for c in range(6)}),
    (MaskShape("brush", ((3.5, 4.5),), radius=1), CROSS),
    (MaskShape("circle", ((0.5, 0.5), (1.5, 0.5))), {(0, 0), (1, 0), (0, 1)}),
    (MaskShape("ellipse", ((-4, -4), (-2, -1))), set()),
])
def test_shapes_and_their_include_mode(shape, expected):
    assert set(map(tuple, np.argwhere(rasterize([shape], GRID)))) == expected
    full = MaskShape("rectangle", ((0, 0), (GRID[1], GRID[0])))
    restored = rasterize([full, replace(shape, mode="include")], GRID)
    assert set(map(tuple, np.argwhere(~restored))) == expected


def test_eraser_restores_pixels_under_its_whole_stroke():
    mark = MaskShape("circle", ((3.5, 4.5), (3.5, 2.5)))
    erase = MaskShape("eraser", ((2.5, 4.5), (4.5, 4.5)), radius=1)
    assert set(map(tuple, np.argwhere(rasterize([mark, erase], GRID)))) == {(2, 3), (6, 3)}


def test_q_annulus_follows_beam_centre_anisotropic_pitch_and_geometry():
    geometry = DetectorGeometry(0.001, 0.002, 0.1, 3.5, 4.5, 1, critical_angle_deg=0.1)
    # The shell spans 1.9–2.1 mm on this detector, i.e. two columns or one row from the beam.
    def q(mm):
        return 4 * math.pi * math.sin(0.5 * math.atan(mm * 0.001 / geometry.distance_m))

    ring = MaskShape("q_annulus", ((q(1.9), q(2.1)),))
    expected = {(4, 1), (4, 5), (3, 3), (5, 3)}
    analyze = AnalyzeFrame(None, None)
    for shift in (0, 1):
        shifted = replace(geometry, beam_center_x_px=geometry.beam_center_x_px + shift)
        mask = analyze._drawn_mask((ring,), GRID, shifted)
        assert set(map(tuple, np.argwhere(mask))) == {(r, c + shift) for r, c in expected}
    with pytest.raises(ValueError, match="geometry"):
        rasterize([ring], GRID)
    tilted = replace(geometry, rot1_rad=0.4, rot2_rad=-0.2)
    q_grid = transmission_map(GRID, tilted).q
    np.testing.assert_array_equal(rasterize([ring], GRID, tilted), (q_grid >= q(1.9)) & (q_grid <= q(2.1)))


@pytest.mark.parametrize(("rule", "indices"), [
    (MaskRule("above", 5), {3, 4}),
    (MaskRule("below", 5), {0, 1}),
    (MaskRule("negative"), {0}),
    (MaskRule("saturated", 7), {3, 4}),
    (MaskRule("above", 1.5, relative=True), {4}),
    (MaskRule("below", 0.5, relative=True), {0}),
])
def test_value_rules_apply_to_finite_raw_values(rule, indices):
    data = np.array([[-2, 4, 5, 7, 15, np.nan]])
    assert set(np.flatnonzero(evaluate_mask([rule], data, np.isfinite(data)))) == indices


def test_median_omits_invalid_pixels_and_earlier_exclusions():
    data = np.array([[1, 2, 3, 4, 20, 30, 50]])
    valid = np.array([[True, True, True, True, True, False, False]])
    mark = MaskShape("rectangle", ((4, 0), (7, 1)))
    mask = evaluate_mask([mark, MaskRule("above", 1, True)], data, valid)
    assert set(np.flatnonzero(mask)) == {2, 3, 4, 5, 6}
    reversed_mask = evaluate_mask([MaskRule("above", 1, True), mark], data, valid)
    assert set(np.flatnonzero(reversed_mask)) == {3, 4, 5, 6}


def test_threshold_boundaries_keep_the_precision_of_the_entered_value_and_even_median():
    data = np.array([[100.0, np.nextafter(np.float32(100.0), np.float32(101.0))]], dtype=np.float32)
    valid = np.ones_like(data, dtype=bool)
    for rule in (MaskRule("below", 100.000001), MaskRule("below", 1.0, relative=True)):
        np.testing.assert_array_equal(evaluate_mask([rule], data, valid), [[True, False]])


def test_morphology_uses_euclidean_pixel_distances_and_unmasked_exterior():
    data = np.zeros(GRID)
    valid = np.ones(GRID, dtype=bool)
    seed = MaskShape("rectangle", ((3, 4), (4, 5)))
    grown = evaluate_mask([seed, MaskRule("grow", 2)], data, valid)
    assert set(map(tuple, np.argwhere(grown))) == DISK
    shrunk = evaluate_mask([MaskRule("shrink", 2)], data, valid, initial=grown)
    assert set(map(tuple, np.argwhere(shrunk))) == {(4, 3)}
    inverted = evaluate_mask([MaskRule("invert")], data, valid, initial=grown)
    np.testing.assert_array_equal(inverted, ~grown)
    assert not evaluate_mask([MaskRule("clear")], data, valid, initial=inverted).any()
    eroded = evaluate_mask([MaskRule("shrink", 1)], data, valid, initial=valid)
    expected = np.zeros(GRID, dtype=bool)
    expected[1:-1, 1:-1] = True
    np.testing.assert_array_equal(eroded, expected)


def test_settings_preview_and_export_replay_the_same_ordered_mask(tmp_path):
    data = np.arange(np.prod(GRID), dtype=np.float32).reshape(GRID)
    data[0, 0] = -1  # signed count detector gap code
    path = tmp_path / "pattern.edf"
    analyze = AnalyzeFrame(FakeFrames({str(path): data}), None)
    req = AnalysisRequest(path, corrections=Corrections(bad_pixels=False))
    before = analyze(req)
    mask_a = np.zeros(GRID, dtype=np.int8)
    mask_a[:2] = 1
    mask_b = np.zeros(GRID, dtype=np.int8)
    mask_b[:, -1] = -2
    base, extra = (str(tmp_path / name) for name in ("stripe.npy", "column.npy"))
    np.save(base, mask_a)
    np.save(extra, mask_b)
    edits = (MaskRule("above", 60), MaskImage(extra, sources=("recipe.json",)),
             MaskShape("rectangle", ((0, 0), (2, 2)), mode="include"))
    corrections = replace(req.corrections, mask_path=base, mask_shapes=edits)
    settings = AnalyzeSettings(corrections=corrections)
    restored = settings_from_record(json.loads(json.dumps(settings_record(settings))))
    assert restored == settings
    expected = (mask_a != 0) | (mask_b != 0) | (data > 60)
    expected[:2, :2] = False
    preview = analyze.preview_mask(before, restored.corrections)
    np.testing.assert_array_equal(preview, expected)
    after = analyze(replace(req, corrections=restored.corrections), loaded=before)
    np.testing.assert_array_equal(after.drawn_mask, preview)
    np.testing.assert_array_equal(after.valid, (data >= 0) & ~expected)
    np.testing.assert_array_equal(analyze(req, loaded=after).valid, before.valid)
    meta = analysis_metadata(after)["corrections"]
    assert meta["mask_sources"] == [base, "recipe.json", extra]
    assert meta["mask_shapes"] == settings_record(settings)["corrections"]["mask_shapes"]


def test_negative_float_values_and_saturation_of_a_frame_sum(tmp_path):
    class Floats(FakeFrames):
        def load(self, path, frame_index=0):
            image = super().load(path, frame_index)
            image.metadata["stored_dtype"] = "float32"
            return image

    first, second = tmp_path / "first.edf", tmp_path / "second.edf"
    frames = Floats({str(first): np.array([[-4., 10., 5., 8.]]),
                     str(second): np.array([[1., -1., 5., 8.]])})
    analyze = AnalyzeFrame(frames, None)
    req = AnalysisRequest(first, corrections=Corrections(bad_pixels=False), summed_frames=((second, 0),))
    before = analyze(req)
    np.testing.assert_array_equal(before.data, [[-3., 9., 10., 16.]])
    assert before.valid.all()
    rules = (MaskRule("saturated", 10), MaskRule("negative"))
    corr = replace(req.corrections, mask_shapes=rules)
    after = analyze(replace(req, corrections=corr), loaded=before)
    np.testing.assert_array_equal(after.drawn_mask, [[True, False, True, True]])
    # On a single exposure the same rules compare each raw value with the same level.
    single = analyze(replace(req, corrections=corr, summed_frames=()))
    np.testing.assert_array_equal(single.drawn_mask, [[True, True, False, False]])
