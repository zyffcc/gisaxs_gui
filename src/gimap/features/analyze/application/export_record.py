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
from .flat_detector import flat_detector_record
from .frame_preprocessing import photon_counts
from .models import FrameAnalysis
from .ports import frame_label, source_name, source_stem

SOFTWARE = "GIMaP Analyze 1"
"""``software`` of Analyze's records: the name of the record format, not the release (``software_version``)."""
Q_LABEL = "q (Å⁻¹)"
GEOMETRY_CONVENTION = (
    "canonical pixel-corner frame, row 0 at top, beam centre = direct beam; "
    "kinematic (vacuum) grazing-incidence q of a flat detector perpendicular to the beam, Å⁻¹"
)
"""``geometry_convention`` of the record: what the q values are (no refraction, no detector tilt; a ``.poni`` tilt
left out is in ``geometry.dropped_tilt_deg`` with what it costs, ``geometry.flat_detector_q_error_max``)."""
POISSON, PIXEL_SCATTER, SCALED_POISSON = "poisson", "pixel_scatter", "scaled_poisson"
"""``error_model``: σ of a point from the photon counts of its pixels (√Σ / n), from the scatter of its pixels
(not counts as measured: ``frame_preprocessing.photon_counts``), or from the counts scaled like the values (GIWAXS
intensity corrections on counts: a pixel's variance is value / factor)."""
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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def source_files(path: str | Path, metadata: Optional[Mapping[str, Any]]) -> list[str]:
    """Every file a frame was read from when it is not ``path`` alone: the module files of a set, or a master
    and its data files (as the reader lists them in the frame's metadata). Empty for a frame of one file."""
    metadata = metadata or {}
    modules = [str(item) for item in metadata.get("module_files") or ()]
    if modules:
        return modules
    data = [str(item) for item in metadata.get("data_files") or ()]
    return [str(path), *data] if data else []


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
    elif metadata.get("measurement") == GIWAXS:
        window = markers.get("chi_q_window")
        cuts["chi_q_window"] = [float(value) for value in window] if window else None
    if markers.get("series_cut"):
        cuts["series_cut"] = _plain(markers["series_cut"])
    return cuts


def analysis_metadata(analysis: FrameAnalysis) -> dict[str, Any]:
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
        "instrument_profile": (
            {"name": profile.name, "source": profile.source, "updated_at": profile.updated_at}
            if profile is not None
            else None
        ),
        "corrections": _plain(analysis.corrections),
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
            {"key": curve.key, "title": curve.title, "x_label": curve.x_label, "region": _plain(curve.region)}
            for curve in (reduction.curves if reduction is not None else ())
        ],
        "messages": list(analysis.messages),
        "error_model": error_model(analysis),
        "binning": binning(analysis),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # UTC, when the record was written
    }
    files = source_files(analysis.path, loaded)
    if files:  # a set of module files, a master with its data files: every file the frame was read from
        record["source_files"] = files
    if loaded.get("layout"):  # NeXus / HDF5: what the source is, which dataset, how the image was turned
        record["source_layout"] = {
            "kind": loaded.get("layout"), "summary": loaded.get("layout_summary"),
            "dataset": loaded.get("dataset_path"), "transformations": _plain(loaded.get("transformations") or []),
        }
    return record


def error_model(analysis: FrameAnalysis) -> Optional[str]:
    """How the σ of the frame's points were computed (``POISSON``, ``PIXEL_SCATTER``, ``SCALED_POISSON``): the
    decision the reduction took (``frame_preprocessing.photon_counts``, then the GIWAXS intensity corrections).
    ``None`` without a reduction."""
    if analysis.reduction is None:
        return None
    if not photon_counts(analysis.metadata, analysis.corrections,
                         mirror_filled=getattr(analysis, "filled_pixels", None) is not None):
        return PIXEL_SCATTER
    return SCALED_POISSON if getattr(analysis, "intensity_scale", None) is not None else POISSON


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
        "merge": MERGE_RULE,
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
    the same decision the cuts use) —, and so do the GIWAXS intensity
    corrections (the values are no longer counts, though their σ still follows
    from the counts); a valid range acts as a threshold, and summed frames are
    recorded.

    ``halves`` — what a fit is to take of the GISAXS cut, chosen when the file was written (``mean`` /
    ``positive`` / ``negative``) — is recorded as ``"halves"``: Fitting starts with it. The file itself always
    holds both halves, signed. Anything else (``None``, the display choice ``both_abs``) and a GIWAXS I(q),
    which has no halves, leave the key out: Fitting decides.

    A frame of a source that holds several also says which: ``"frame_index"`` (0-based, as in the file name)
    and ``"frame_count"`` — what a series in Fitting needs to plot against the frame of the measurement.
    """
    corrections = analysis.corrections
    record = {
        "source": "native_detector_columns" if analysis.kind == GISAXS else "radial_bins",
        "file_format": analysis.path.suffix.lower().lstrip("."),
        "intensity_unit": "counts_per_pixel",
        "counting_model_valid": (
            photon_counts(analysis.metadata, corrections, mirror_filled=getattr(analysis, "filled_pixels", None) is not None)
            and analysis.intensity_scale is None
            and not (analysis.kind == GIWAXS and has_intensity_corrections(corrections))
        ),
        "threshold_enabled": corrections.minimum is not None or corrections.maximum is not None,
        "gap_guard_px": int(corrections.gap_guard_px),
        "summed_frames": analysis.frame_total,
    }
    if analysis.kind == GISAXS and halves in FIT_HALVES:
        record["halves"] = halves
    count = int(getattr(analysis, "frame_count", 1) or 1)
    if count > 1:
        record["frame_index"], record["frame_count"] = int(getattr(analysis, "frame_index", 0)), count
    return record


def fit_input_metadata(analysis: FrameAnalysis, metadata: Mapping[str, Any], halves: Optional[str] = None) -> dict[str, Any]:
    return {**metadata, "observation": fit_input_observation(analysis, halves)}


__all__ = [
    "GEOMETRY_CONVENTION", "MERGE_RULE", "PIXEL_SCATTER", "POISSON", "Q_LABEL", "SCALED_POISSON", "SOFTWARE",
    "analysis_metadata", "binning", "error_model", "export_stem", "fit_input_curve", "fit_input_metadata",
    "fit_input_observation", "frame_cuts", "frame_name", "frame_name_of", "frame_stem", "source_files",
]
