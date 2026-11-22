"""Re-run a recorded Analyze reduction with its own geometry, corrections and detector inputs."""

from dataclasses import replace

import numpy as np

from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from src.gimap.shared.reproduction import verified_path

from ..application.models import AnalysisRequest, GeometryResolution
from ..application.peak_reproduction import replay_peak_tables
from ..application.reduction_record import reduction_tables
from ..application.series import scaled
from ..application.settings_file import settings_from_record
from ..application.use_cases import AnalyzeFrame
from ..domain.mask_bitmap import read_bitmap
from ..domain.mask_edits import MaskImage
from .adapters.frames import DetectorIoFrameSource


def replay_reduction(record, record_dir):
    recipe = record["reproduction"]
    paths = {key: verified_path(ref, record_dir) for key, ref in recipe["inputs"].items()}

    def input_path(key):
        if key not in paths:
            raise ValueError(f"No input file digest recorded for {key}.")
        return paths[key]

    settings = settings_from_record(recipe["settings"])
    corrections = settings.corrections
    overrides = {name: str(input_path(value)) for name in ("dark_path", "background_path", "flat_field_path", "mask_path")
                 if (value := getattr(corrections, name))}
    edits = []
    for edit in corrections.mask_shapes:
        sources = tuple(str(input_path(path)) for path in edit.sources)
        updates = {"path": str(input_path(edit.path))} if isinstance(edit, MaskImage) else {}
        edits.append(replace(edit, sources=sources, **updates))
    request = AnalysisRequest(
        path=input_path(record["source_file"]), frame_index=record["frame_index"], mode=settings.mode,
        summed_frames=tuple((input_path(item["file"]), item["frame_index"]) for item in record["summed_frames"]),
        gisaxs=settings.gisaxs, giwaxs=settings.giwaxs, corrections=replace(corrections, mask_shapes=tuple(edits), **overrides),
        with_map=False, series_cut=recipe["series_cut"],
    )
    profile = InstrumentProfile.from_dict(record["instrument_profile"]) if record["instrument_profile"] else None
    resolution = GeometryResolution(DetectorGeometry(**recipe["geometry"]), profile, "record")
    analyze = AnalyzeFrame(DetectorIoFrameSource(), None)
    analysis = analyze(request, resolution=resolution, mask_bitmap=recipe["mask"])
    used = {ref["path"] for ref in (*analysis.frame_files.values(), *analysis.correction_files.values())}
    extra = used - {str(path.resolve()) for path in paths.values()}
    if extra:
        raise ValueError(f"The detector reader used an input outside the record: {min(extra)}")
    if not np.array_equal(~analysis.valid, read_bitmap(recipe["excluded_pixels"], analysis.shape)):
        raise ValueError("The reproduced detector validity differs from the recorded final mask.")
    if record.get("reference_peak_scale") is not None:
        analysis = scaled(analysis, record["reference_peak_scale"])
    tables = reduction_tables(analysis, recipe["curve_keys"])
    if "peak_fits" in recipe:
        tables.update(replay_peak_tables(recipe["peak_fits"], analysis, analyze.maps_of(analysis)))
    return tables
