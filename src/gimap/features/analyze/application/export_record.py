"""What an export says about its frame: the name of its files, the record beside them, the curve for Fitting.

**One way to number a frame.** A frame of a source that holds several is counted from 1 wherever a person
reads it — headers (``# frame: 4 of 12 (index 3)``), column heads, picture titles, records (``"frame"``), all
through ``ports.frame_label`` — and by its 0-based index in file names (``_frame0003``): those are identifiers
that scripts and Fitting's listings already rely on, and ``README.txt`` of a batch says so.

**One name for a source.** A set of module files (one file per detector module, each with every frame) is
named by its scan, without the module number (``ports.source_stem``: ``scan_00007_m01.nxs`` →
``scan_00007``), and the record lists every file the frame was read from (``source_files``).

**The curve for Fitting** (``fit_input_curve``) is always q in Å⁻¹: the whole horizontal cut with signed qy
for GISAXS, the I(q) for GIWAXS — also when the I(q) is shown against 2θ.
"""

from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from src.gimap.shared.file_digest import input_digests
from src.gimap.shared.geometry.q_inverse import q_of_positions
from src.gimap.shared.signed_cut import FIT_HALVES
from src.gimap.shared.version import SOFTWARE as RELEASE

from ..domain import (
    GISAXS,
    GIWAXS,
    Curve,
    describe_intensity_corrections,
    has_intensity_corrections,
    q_from_two_theta,
)
from ..domain.mask_edits import MaskImage, mask_edit_record
from .flat_detector import flat_detector_record
from .frame_preprocessing import photon_counts
from .models import FrameAnalysis
from .ports import frame_label, source_name, source_stem
from .reduction_inputs import source_files

SOFTWARE = "GIMaP Analyze 1"
"""``software`` of Analyze's records: the name of the record format, not the release (``software_version``)."""
Q_LABEL = "q (Å⁻¹)"
GEOMETRY_CONVENTION = (
    "canonical pixel-corner frame, row 0 at top, beam centre = direct beam; "
    "kinematic (vacuum) grazing-incidence q of a flat detector, tilted by pyFAI's rotations rot1-rot3 of the "
    "geometry (all 0: perpendicular to the beam); q_z inside the film when geometry.critical_angle_deg is above 0 "
    "(refraction, pixels below it left out), Å⁻¹"
)
"""``geometry_convention`` of the record: what the q values are (the detector tilt is ``geometry.rot1_rad`` …
``rot3_rad``; refraction only with ``geometry.critical_angle_deg`` above 0)."""
POISSON, PIXEL_SCATTER, SCALED_POISSON = "poisson", "pixel_scatter", "scaled_poisson"
"""Established record names for the Poisson and pixel scatter estimators."""
MERGE_RULE = (
    "a bin with fewer pixels than max(8, 10 % of the curve's median) is joined with its neighbours until the point "
    "has enough, never across a gap of more than 2.5 bin steps; the point has the pixel-weighted mean of q and of I"
)
"""How the sparse bins of an I(q) are joined (``domain/binning.merge_sparse_bins``), for the record's ``binning``."""


def _plain(value: Any) -> Any:
    """JSON-friendly copy of markers (dataclasses, tuples, numpy scalars)."""
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def frame_stem(path: str | Path, source: Any = None, *, frame_index: int = 0, frame_count: int = 1,
               frame_total: int = 1) -> str:
    """The stem of every file exported of one frame: the source's name (``ports.source_stem``), ``_frameNNNN``
    (the 0-based index) when the source holds several frames, ``_sumK`` when K frames were summed."""
    stem = source_stem(path, source)
    if int(frame_count) > 1:
        stem += f"_frame{int(frame_index):04d}"
    if int(frame_total) > 1:
        stem += f"_sum{int(frame_total)}"
    return stem


def export_stem(analysis: FrameAnalysis) -> str:
    return frame_stem(analysis.path, analysis.metadata, frame_index=analysis.frame_index,
                      frame_count=analysis.frame_count, frame_total=analysis.frame_total)


def frame_name_of(path: str | Path, source: Any = None, *, frame_index: int = 0, frame_count: int = 1,
                  summed: int = 0) -> str:
    """How a frame is called in tables, records and titles: the source, ``frame 4 of 12`` when it holds
    several, and how many further frames were summed with it."""
    name = source_name(path, source)
    if int(frame_count) > 1:
        name += " " + frame_label(frame_index, frame_count)
    return name + (f" (+{int(summed)} summed)" if summed else "")


def frame_name(analysis: FrameAnalysis) -> str:
    return frame_name_of(analysis.path, analysis.metadata, frame_index=analysis.frame_index,
                         frame_count=int(analysis.frame_count or 1), summed=len(analysis.summed_frames or ()))


def frame_cuts(metadata: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Where the automatic cuts of a frame were taken, from its record (``analysis_metadata``): the rows of the
    horizontal band and the columns of the vertical one (GISAXS), or the q window of the I(χ) ring (GIWAXS). A
    frame cut at the cut of a series also has ``series_cut`` (``{"frame": [file, index], …}``: the frame the cut
    is from, and which of its cuts it took) and the horizontal source ``series``."""
    metadata = metadata or {}
    markers = metadata.get("markers") or {}
    regions = {str(curve.get("key")): curve.get("region") or {} for curve in metadata.get("curves") or ()}
    cuts: dict[str, Any] = {}
    if metadata.get("measurement") == GISAXS:
        rows, columns = regions.get("horizontal", {}).get("rows"), regions.get("vertical", {}).get("columns")
        cuts["horizontal_rows"] = list(rows) if rows else None
        cuts["horizontal_source"] = markers.get("horizontal_source")
        cuts["vertical_columns"] = list(columns) if columns else None
        cuts["vertical_source"] = markers.get("vertical_source")
    elif metadata.get("measurement") in (GIWAXS, "transmission"):
        window = markers.get("chi_q_window")
        cuts["chi_q_window"] = [float(value) for value in window] if window else None
    if markers.get("series_cut"):
        cuts["series_cut"] = _plain(markers["series_cut"])
    return cuts


def recorded_region(analysis, curve):
    region = dict(curve.region or {})
    if analysis.kind == GISAXS and curve.key in ("horizontal", "vertical") and "rows" in region and "columns" in region:
        top, bottom = region["rows"]
        left, right = region["columns"]
        corners = ("top_left", "top_right", "bottom_right", "bottom_left")
        qy, qz = q_of_positions(analysis.geometry, [left, right, right, left], [top, top, bottom, bottom])
        region["detector_corners_qy_qz_A_inv"] = dict(zip(corners, np.column_stack((qy, qz)).tolist()))
    return _plain(region)


def analysis_metadata(analysis: FrameAnalysis, *, curve_keys=None, with_reproduction=True) -> dict[str, Any]:
    """Provenance written next to exported curves."""
    reduction = analysis.reduction
    profile = analysis.resolution.profile
    loaded = analysis.metadata or {}
    record = {
        "software": SOFTWARE,
        "software_version": RELEASE,  # the release that wrote the record (``shared.version.SOFTWARE``)
        "source_file": str(analysis.path),
        "frame_index": analysis.frame_index,
        "frame_count": analysis.frame_count,
        "frame": frame_label(analysis.frame_index, max(1, int(analysis.frame_count or 1))),
        "summed_frames": [
            {"file": str(path), "frame_index": index} for path, index in analysis.summed_frames
        ],
        "detector_name": analysis.detector_name,
        "frame_shape": list(analysis.shape),
        "measurement": analysis.kind,
        "geometry": (
            {**_plain(analysis.geometry), **_plain(flat_detector_record(getattr(analysis, "flat_detector", None)))}
            if analysis.geometry is not None else None
        ),
        "geometry_convention": GEOMETRY_CONVENTION,
        "geometry_resolution": {
            "method": analysis.resolution.how, "beam_center_source": analysis.resolution.center_source,
            "header_beam_center_px": _plain(analysis.resolution.header_center),
        },
        "instrument_profile": (
            profile.to_dict()
            if profile is not None
            else None
        ),
        "corrections": {**_plain(analysis.corrections), "flat_field": flat_field_record(analysis),
                        "reference_dark": getattr(analysis, "reference_dark", None)},
        "normalisation": None if getattr(analysis, "normalisation", None) is None else analysis.normalisation.to_record(),
        "header_values": _plain(getattr(analysis, "header_values", None) or {}),
        "intensity_corrections": (
            describe_intensity_corrections(analysis.corrections) if analysis.kind == GIWAXS else None
        ),
        "bad_pixels_left_out": None if analysis.bad_pixels is None else {
            "hot": analysis.bad_pixels.hot_count, "dead": analysis.bad_pixels.dead_count,
            "defective_line_pixels": analysis.bad_pixels.line_count,
        },
        "drawn_mask_pixels": None if analysis.drawn_mask is None else int(analysis.drawn_mask.sum()),
        "pixels_filled_from_mirror": None if analysis.filled_pixels is None else int(analysis.filled_pixels.sum()),
        "markers": _plain(reduction.markers) if reduction is not None else {},
        "curves": [
            {"key": curve.key, "title": curve.title, "x_label": curve.x_label, "y_label": curve.y_label,
             "region": recorded_region(analysis, curve)}
            for curve in (reduction.curves if reduction is not None else ())
        ],
        "messages": list(analysis.messages),
        "error_model": error_model(analysis),
        "binning": binning(analysis),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # UTC, when the record was written
    }
    if getattr(analysis, "calibrated_curve_unit", None) is not None:
        record["output_intensity_unit"] = analysis.calibrated_curve_unit
    if getattr(analysis, "reference_peak_scale", None) is not None:
        record["reference_peak_scale"] = analysis.reference_peak_scale
    files = source_files(analysis.path, loaded)
    corrections = analysis.corrections
    mask_sources = [corrections.mask_path] if corrections.mask_path else []
    for edit in corrections.mask_shapes:
        mask_sources.extend(edit.sources)
        if isinstance(edit, MaskImage):
            mask_sources.append(edit.path)
    record["corrections"]["mask_shapes"] = [mask_edit_record(edit) for edit in corrections.mask_shapes]
    record["corrections"]["mask_sources"] = list(dict.fromkeys(mask_sources))
    if files:  # a set of module files, a master with its data files: every file the frame was read from
        record["source_files"] = files
    if loaded.get("layout"):  # NeXus / HDF5: what the source is, which dataset, how the image was turned
        record["source_layout"] = {
            "kind": loaded.get("layout"), "summary": loaded.get("layout_summary"),
            "dataset": loaded.get("dataset_path"), "transformations": _plain(loaded.get("transformations") or []),
        }
    if getattr(analysis, "request", None) is not None and reduction is not None:
        inputs = {**analysis.frame_files, **analysis.correction_files, **analysis.geometry_files}
        if with_reproduction:
            from .reduction_record import reduction_record
            record["reproduction"] = reduction_record(analysis, inputs, curve_keys)
        record["input_sha256"] = [{"file": Path(ref["path"]).name, "sha256": ref["sha256"]} for ref in inputs.values()]
    else:
        references = [corrections.dark_path, corrections.background_path, corrections.flat_field_path]
        if getattr(analysis, "error_model_used", None) == "sigma" or any(
            curve["region"].get("error_model") == "sigma_frame" for curve in record["curves"]
        ):
            references.append(corrections.sigma_path)
        record["input_sha256"] = input_digests(
            [*(files or [analysis.path]), *(path for path, _ in analysis.summed_frames or ()), *mask_sources,
             *(path for path in references if path)])
    return record


def flat_field_record(analysis: FrameAnalysis) -> Optional[dict[str, Any]]:
    """The flat field the frame was divided by — file, path, SHA-256, normalisation mean, floor, pixels it leaves
    out, how it was applied — for the ``corrections`` block of the record; ``None`` without one."""
    use = getattr(analysis, "flat_field", None)
    return None if use is None else use.to_record()


def provenance_lines(record: Mapping[str, Any], written_by: str, settings: Optional[str] = None, *,
                     frame: bool = True) -> list[str]:
    """The lines above the columns of a curve or map file (without their ``#``), from the frame's record
    (``analysis_metadata``): the release, each input file by name and SHA-256, the frame and the geometry, so a
    table says where it came from also without its JSON record beside it. One writer for the command line's CSVs
    and the window's exports (``frame=False``: the file already has its own frame line)."""
    lines = [f"{RELEASE} ({written_by})" + (f"; settings and decisions in {settings}" if settings else "")]
    lines += [f"input: {item['file']} sha256 {item['sha256']}" for item in record.get("input_sha256") or ()]
    if frame:
        lines.append(f"frame: {record.get('frame')}")
    geometry = record.get("geometry")
    if geometry:
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, math.cos(geometry.get("rot1_rad") or 0.0)
                                                    * math.cos(geometry.get("rot2_rad") or 0.0)))))
        critical = float(geometry.get("critical_angle_deg") or 0.0)
        lines.append(
            f"geometry: distance {geometry['distance_m'] * 1e3:.4f} mm, direct beam ({geometry['beam_center_x_px']:.3f}, "
            f"{geometry['beam_center_y_px']:.3f}) px, wavelength {geometry['wavelength_angstrom']:.6g} A, "
            f"alpha_i {geometry['incidence_deg']:.4g} deg, detector tilt {tilt:.4g} deg" + (
                f", q_z refracted, alpha_c {critical:.4g} deg" if critical > 0.0 else ""))
    return lines + intensity_lines(record)


def intensity_lines(record: Mapping[str, Any]) -> list[str]:
    """What was subtracted from and divided into the intensities, from the record: the dark frame and the
    normalisation with each of its terms and where it came from (none of them: no line)."""
    lines = []
    corrections = record.get("corrections") or {}
    if corrections.get("dark_path"):
        lines.append(f"dark frame: {Path(str(corrections['dark_path'])).name} (index {int(corrections.get('dark_frame') or 0)}) "
                     f"x {float(corrections.get('dark_scale', 1.0)):g}, subtracted before the background")
    reference = corrections.get("reference_dark")
    if reference:
        lines.append(f"background dark scale: {reference['background_dark_scale']:g} ({reference['source']})")
    model = record.get("error_model")
    if model:
        def estimator(name):
            if name in (POISSON, SCALED_POISSON):
                return f"{name}, {corrections.get('poisson_variance', 'observed')} sample shot variance"
            if name == PIXEL_SCATTER and corrections.get("pooled_scatter"):
                return f"{name}, pooled below 10 pixels"
            return name

        ring = next(((c.get("region") or {}).get("error_model") for c in record.get("curves") or ()
                     if c.get("key") == "azimuthal"), None)
        if ring and ring != model:
            lines.append(f"radial and sector error model: {estimator(model)}")
            lines.append(f"ring orientation error model: {estimator(ring)}")
        else:
            lines.append(f"error model: {estimator(model)}")
    norm = record.get("normalisation")
    if norm:
        terms = ", ".join(f"{key} {item['value']:.6g} ({item['source']})" for key, item in norm.items()
                          if isinstance(item, Mapping) and "value" in item and "source" in item)
        unit = norm["intensity_unit"]
        if record.get("reference_peak_scale") is not None:
            unit = (f"relative after reference-peak scaling by {record['reference_peak_scale']:.8g}; "
                    f"before reference-peak scaling: {unit}")
        lines.append(f"intensity: {unit}; normalised I = A_abs*I/(t_exp*M*T*d), factor {norm['factor']:.6g}"
                     f" ({terms}); sigma scaled by the same factor")
    return lines


def error_model(analysis: FrameAnalysis) -> Optional[str]:
    """Describe the estimator used for the reduced curves, including supplied σ and propagated references."""
    if analysis.reduction is None:
        return None
    model = getattr(analysis, "error_model_used", None)
    if model == "sigma":
        return "sigma_frame"
    if model == "scatter" or (model is None and not photon_counts(
        analysis.metadata, analysis.corrections, mirror_filled=getattr(analysis, "filled_pixels", None) is not None
    )):
        return PIXEL_SCATTER
    propagated = analysis.intensity_scale is not None or getattr(analysis, "extra_variance", None) is not None
    return SCALED_POISSON if propagated else POISSON


def binning(analysis: FrameAnalysis) -> Optional[dict[str, Any]]:
    """How the curves were binned. GIWAXS: the radial bins (count, edges, width in q) and how sparse bins are
    joined (the number of points the I(q) kept), and the 1° bins of I(χ); GISAXS: one point per detector column
    or row. ``None`` without a reduction."""
    reduction = analysis.reduction
    if reduction is None:
        return None
    if analysis.kind == GISAXS:
        return {"cuts": "one point per detector column (horizontal cut) or row (vertical cut): the mean of the "
                        "band's valid pixels", "merge": "none"}
    radial = (reduction.markers or {}).get("radial_bins") or {}
    bins, edges = radial.get("bins"), radial.get("q_range")
    curve = reduction.curve("radial")
    return {
        "radial_bins": bins,
        "q_min": None if not edges else float(edges[0]), "q_max": None if not edges else float(edges[1]),
        "bin_width": float(edges[1] - edges[0]) / bins if bins and edges else None,
        "unit": Q_LABEL,
        "bins_chosen": None if not radial else (
            "auto: the q range over the q step of one pixel (99th percentile), 100 to 3000 bins" if radial.get("chosen") == "auto" else "set in Options"),
        "radial_points": None if curve is None else int(np.asarray(curve.x).size),
        "merge": "none (sum per original q bin)" if curve is not None and curve.region.get("aggregation") else MERGE_RULE,
        "chi_bin_deg": 1.0, "chi_merge": "none (the I(χ) bins are kept as they are)",
    }


def _for_fitting(curve: Curve, x=None, x_label: Optional[str] = None, title: Optional[str] = None) -> Curve:
    return Curve(
        "fit_input", curve.title if title is None else title, curve.x if x is None else x, curve.intensity,
        curve.sigma, curve.pixels, curve.x_label if x_label is None else x_label, curve.y_label, dict(curve.region),
    )


def fit_input_curve(analysis: FrameAnalysis) -> Optional[Curve]:
    """The curve handed to Fitting, always against q in Å⁻¹.

    GISAXS: the whole horizontal cut with signed qy — Fitting's q view shows
    the halves apart on |qy|, averages them or keeps one (fitting models are
    functions of |q|).  GIWAXS: the full I(q); shown against 2θ in Analyze, it
    is converted back (the same bins: q = 4π sin θ / λ).  ``None`` when there is
    nothing to fit.
    """
    reduction = analysis.reduction
    if reduction is None:
        return None
    if analysis.kind == GISAXS:
        curve = reduction.curve("horizontal")
        if curve is None or curve.is_empty:
            return None
        return _for_fitting(curve)
    curve = reduction.curve("radial")
    if curve is None or curve.is_empty:
        return None
    if curve.x_label.startswith("2θ") and analysis.geometry is not None:
        q = q_from_two_theta(np.asarray(curve.x, dtype=float), analysis.geometry.wavelength_angstrom)
        return _for_fitting(curve, q, Q_LABEL, curve.title.replace("I(2θ)", "I(q)"))
    return _for_fitting(curve)


def fit_input_observation(analysis: FrameAnalysis, halves: Optional[str] = None) -> dict[str, Any]:
    """How the fit-input points were measured, for Fitting's counting-statistics contract.

    GISAXS cuts are native detector columns (one point per column, mean counts
    per pixel, Poisson σ); background subtraction, a floating-point
    (dark-subtracted) frame and mirror filling break the counting model — σ then
    comes from the scatter of the pixels (``frame_preprocessing.photon_counts``,
    the same decision the cuts use) —, and so do a flat field and the GIWAXS
    intensity corrections (the values are no longer counts, though their σ still
    follows from the counts); a valid range acts as a threshold, and summed
    frames are recorded.

    ``halves`` — what a fit is to take of the GISAXS cut, chosen when the file was written (``mean`` /
    ``positive`` / ``negative``) — is recorded as ``"halves"``: Fitting starts with it. The file itself always
    holds both halves, signed. Anything else (``None``, the display choice ``both_abs``) and a GIWAXS I(q),
    which has no halves, leave the key out: Fitting decides.

    A frame of a source that holds several also says which: ``"frame_index"`` (0-based, as in the file name)
    and ``"frame_count"`` — what a series in Fitting needs to plot against the frame of the measurement. When the
    file says when the frame was measured, ``"timestamp"`` (as the file writes it) and ``"frame_time_s"`` (a time of
    each frame of a series) are recorded too: Fitting's trend can then be drawn against time.
    """
    corrections = analysis.corrections
    record = {
        "source": "native_detector_columns" if analysis.kind == GISAXS else "radial_bins",
        "file_format": analysis.path.suffix.lower().lstrip("."),
        "intensity_unit": ("counts_per_pixel" if getattr(analysis, "normalisation", None) is None
                           else getattr(analysis, "calibrated_curve_unit", None) or analysis.normalisation.unit),
        "counting_model_valid": (
            photon_counts(analysis.metadata, corrections, mirror_filled=getattr(analysis, "filled_pixels", None) is not None)
            and analysis.intensity_scale is None
            and not (corrections.flat_field_path or corrections.dark_path or corrections.background_path)
            and corrections.error_model not in ("scatter", "sigma")
            and corrections.counts_per_adu == 1 and corrections.offset_adu == 0
            and not (analysis.kind == GIWAXS and has_intensity_corrections(corrections))
        ),
        "threshold_enabled": corrections.minimum is not None or corrections.maximum is not None,
        "gap_guard_px": int(corrections.gap_guard_px),
        "summed_frames": analysis.frame_total,
    }
    if analysis.kind == GIWAXS and corrections.lorentz:
        radial = analysis.reduction.curve("radial") if analysis.reduction is not None else None
        if radial is not None:
            record["intensity_unit"] = radial.y_label.removeprefix("I (").removesuffix(")")
            record["aggregation"] = radial.region.get("aggregation")
    if analysis.kind == GISAXS and halves in FIT_HALVES:
        record["halves"] = halves
    count = int(getattr(analysis, "frame_count", 1) or 1)
    if count > 1:
        record["frame_index"], record["frame_count"] = int(getattr(analysis, "frame_index", 0)), count
    metadata = getattr(analysis, "metadata", None) or {}
    if metadata.get("timestamp"):
        record["timestamp"] = str(metadata["timestamp"])
    try:
        frame_time = float(metadata.get("frame_time_s"))
    except (TypeError, ValueError):
        frame_time = None
    if frame_time is not None and np.isfinite(frame_time):
        record["frame_time_s"] = frame_time
    return record


def fit_input_metadata(analysis: FrameAnalysis, metadata: Mapping[str, Any], halves: Optional[str] = None) -> dict[str, Any]:
    return {**metadata, "observation": fit_input_observation(analysis, halves)}


__all__ = [
    "GEOMETRY_CONVENTION", "MERGE_RULE", "PIXEL_SCATTER", "POISSON", "Q_LABEL", "SCALED_POISSON", "SOFTWARE",
    "analysis_metadata", "binning", "error_model", "export_stem", "fit_input_curve", "fit_input_metadata", "flat_field_record",
    "fit_input_observation", "frame_cuts", "frame_name", "frame_name_of", "frame_stem", "intensity_lines",
    "provenance_lines",
    "source_files",
]
