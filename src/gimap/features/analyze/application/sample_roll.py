"""Measure the sample orientation using only observed detector pixels."""

from dataclasses import replace

from src.gimap.shared.geometry.sample_roll import measure_sample_roll

from ..domain.intensity import film_absorption_on
from ..domain.mask_bitmap import bitmap_record
from .use_cases import AnalyzeFrame


def measure_frame_roll(analysis, frame_source):
    data, valid = analysis.data, analysis.valid
    settings = analysis.corrections
    if analysis.kind == "giwaxs" and (settings.lorentz or settings.mirror_fill or film_absorption_on(settings)):
        # These corrections depend on the assumed surface normal. Restore the measured
        # pixels hidden or altered by them, retaining the user's flat, dark and mask.
        reducer = AnalyzeFrame(frame_source, None)
        data, valid, _ = reducer.measured_orientation_pixels(
            vars(analysis), replace(settings, film_thickness_nm=None, attenuation_length_um=None),
            analysis.geometry, analysis.raw_data, analysis.raw_valid, analysis.path.name,
            bitmap_record(analysis.drawn_mask, analysis.shape),
        )
    return measure_sample_roll(data, analysis.geometry, valid=valid, technique=analysis.kind)
