"""Batch export: every listed frame reduced with the same settings and written as the person chose.

``BatchChoices`` says what to write, and in which formats; ``output_names`` says, for each choice,
the file (with its extension) it writes, so the dialog can show it and ``batch_report.readme_text`` can
list it in a ``README.txt`` next to the files. The files of one kind go into their own subfolder:

====================  ==================================================  ===========================
choice                file                                                what
====================  ==================================================  ===========================
``tables``            ``<batch>_<curve>_frames.<txt>``                    a curve of every frame, one column each
``per_frame``         ``curves/<frame>_<curve>.<txt>`` + ``_analysis.json``  x, I, σ, pixels of one frame
``fit_input``         ``fit_input/<frame>_fit_input.dat``                 what Fitting ▸ In-situ series reads
``q_map``             ``maps/<frame>_qmap.<txt>``                         intensity on the q∥–qz (qy–qz) grid
``cake``              ``maps/<frame>_cake.<txt>``                         intensity on the χ–q grid (GIWAXS)
``detector_image``    ``images/<frame>_detector.<img>``                   picture of the detector image
``q_map_image``       ``images/<frame>_qmap.<img>``                       picture of the q map
``frames``            ``frames/<frame>.<frame format>``                   the detector data in another format
fit (``batch_fit``)   ``<batch>_peak_fits.<txt>`` / ``_model_fits``       one row per frame
====================  ==================================================  ===========================

``<txt>`` is ``csv`` (comma), ``txt`` (tab) or ``dat`` (space); ``<img>`` ``png``, ``tif``, ``svg`` or
``pdf``; the frame format ``tiff``, ``edf``, ``npy`` or ``hdf5`` (``shared.detector_io.write_frame``,
values as read: modules stitched, frames summed). Curve tables: frames of one geometry share their x
exactly; frames whose points are the same up to rounding (another cut row, other masked columns, another
range) are matched point by point, nothing interpolated; only frames on really different grids are
interpolated linearly onto a common one, never across a hole or beyond their range — and the header of the
table says which of the three it was (``batch_tables.py``).

``curves`` names the curves to write (empty: every curve of the frame); ``every`` takes every n-th
listed frame (1: all of them). ``fit_halves`` is what a fit takes of the signed GISAXS cut — ``mean``,
``positive`` or ``negative``, never both halves together: the particle-model fit of the batch uses it,
and every fit input says it (the file itself keeps both halves, signed).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from src.gimap.shared.detector_io import frame_suffix, write_frame
from src.gimap.shared.signed_cut import FIT_HALVES, fit_halves

from ..domain import GISAXS, PROFILES, Curve
from .batch_fit import FIT_MODEL, FIT_NONE, FIT_PEAKS, MODEL_COMPONENTS, STARTS
from .batch_tables import common_x
from .export_record import source_files
from .models import FrameAnalysis
from .ports import CurveWriter, source_stem
from .use_cases import SOFTWARE, ExportAnalysis, export_stem

BATCH_TITLE = "GIMaP Analyze batch export"
TEXT_FORMATS = {"csv": "CSV — comma separated (.csv)", "txt": "TXT — tab separated (.txt)", "dat": "DAT — space separated (.dat)"}
IMAGE_FORMATS = {"png": "PNG (.png)", "tif": "TIFF (.tif)", "svg": "SVG, vector (.svg)", "pdf": "PDF, vector (.pdf)"}
FRAME_FORMATS = {
    "tiff": "TIFF, 32-bit (.tif)", "edf": "EDF (.edf)", "npy": "NumPy (.npy)", "hdf5": "HDF5 / NeXus (.h5)",
}
FOLDERS = {
    "per_frame": "curves", "fit_input": "fit_input", "q_map": "maps", "cake": "maps",
    "detector_image": "images", "q_map_image": "images", "frames": "frames", "fit_curves": "fits",
}
GENTLE, BALANCED, FAST = "gentle", "balanced", "fast"
SPEEDS = (GENTLE, BALANCED, FAST)
"""How many frames are reduced at once (see ``batch_job.frames_at_once``)."""

SCALE_AUTO, SCALE_SCREEN = "auto", "screen"
"""Pictures of a batch: colour limits of each frame's own, or those on screen for every frame."""

_CHOICES = {
    "speed": SPEEDS,
    "image_scale": (SCALE_AUTO, SCALE_SCREEN),
    "text_format": tuple(TEXT_FORMATS), "image_format": tuple(IMAGE_FORMATS), "frame_format": tuple(FRAME_FORMATS),
    "fit": (FIT_NONE, FIT_PEAKS, FIT_MODEL), "fit_profile": PROFILES, "fit_start": STARTS,
    "fit_model": tuple(MODEL_COMPONENTS), "fit_halves": FIT_HALVES,
}


@dataclass(frozen=True)
class BatchChoices:
    curves: tuple[str, ...] = ()
    tables: bool = True
    per_frame: bool = False
    fit_input: bool = False
    q_map: bool = False
    cake: bool = False
    detector_image: bool = False
    q_map_image: bool = False
    frames: bool = False
    every: int = 1
    text_format: str = "csv"
    image_format: str = "png"
    frame_format: str = "tiff"
    fit: str = FIT_NONE
    fit_profile: str = "pseudo_voigt"
    fit_start: str = "previous"
    fit_model: str = "auto"
    fit_halves: str = "mean"
    """What a fit takes of the signed GISAXS cut (``FIT_HALVES``): the model fit, and the fit inputs say it."""
    fit_curves: bool = False
    speed: str = BALANCED
    image_scale: str = SCALE_AUTO

    @property
    def needs_map(self) -> bool:
        """The reduction must also regrid the q map (it takes time; skipped otherwise)."""
        return self.q_map or self.q_map_image

    @property
    def writes_anything(self) -> bool:
        return any((
            self.tables, self.per_frame, self.fit_input, self.q_map, self.cake, self.detector_image,
            self.q_map_image, self.frames, self.fit != FIT_NONE,
        ))

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["curves"] = list(self.curves)
        return record

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "BatchChoices":
        """Stored choices; unknown keys are ignored, bad values fall back to the defaults."""
        if not isinstance(data, Mapping):
            return cls()
        values: dict[str, Any] = {}
        if data.get("images") is True:  # saved before the two images were separate choices
            values.update(detector_image=True, q_map_image=True)
        for item in fields(cls):
            if item.name not in data:
                continue
            value = data[item.name]
            if item.name == "curves":
                if isinstance(value, (list, tuple)):
                    values["curves"] = tuple(str(key) for key in value)
            elif item.name == "every":
                if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
                    values["every"] = value
            elif item.name == "fit_halves":  # a display choice (both, both_abs) is no choice of a fit: the mean
                values["fit_halves"] = fit_halves(value)
            elif item.name in _CHOICES:
                if value in _CHOICES[item.name]:
                    values[item.name] = value
            elif isinstance(value, bool):
                values[item.name] = value
        return cls(**values)


def output_names(choices: BatchChoices, *, batch: str, frame: str, curve: str = "region1",
                 fit_key: Optional[str] = None) -> dict[str, str]:
    """The file each choice writes (relative to the batch folder), for the dialog and the README.

    ``fit_key``: the curve key of a fitted peak (its region) for the name of its fitted curve — ``curve`` when
    not given; the particle model's is always ``<frame>_model_fit``."""
    text, image = choices.text_format, choices.image_format
    peaks_or_model = "peak_fits" if choices.fit != FIT_MODEL else "model_fits"
    fitted = "model" if choices.fit == FIT_MODEL else (fit_key or curve)
    return {
        "tables": f"{batch}_{curve}_frames.{text}",
        "per_frame": f"{FOLDERS['per_frame']}/{frame}_{curve}.{text}",
        "fit_input": f"{FOLDERS['fit_input']}/{frame}_fit_input.dat",
        "q_map": f"{FOLDERS['q_map']}/{frame}_qmap.{text}",
        "cake": f"{FOLDERS['cake']}/{frame}_cake.{text}",
        "detector_image": f"{FOLDERS['detector_image']}/{frame}_detector.{image}",
        "q_map_image": f"{FOLDERS['q_map_image']}/{frame}_qmap.{image}",
        "frames": f"{FOLDERS['frames']}/{frame}{frame_suffix(choices.frame_format)}",
        "fit": f"{batch}_{peaks_or_model}.{text}",
        "fit_plot": f"{batch}_{peaks_or_model}.png",
        "fit_curves": f"{FOLDERS['fit_curves']}/{frame}_{fitted}_fit.{text}",
        "record": f"{batch}_batch.json",
        "readme": "README.txt",
    }


def chosen_curves(analysis: FrameAnalysis, keys: Sequence[str] = ()) -> list[Curve]:
    """The frame's non-empty curves named in ``keys`` (in that order), or all of them."""
    reduction = analysis.reduction
    if reduction is None:
        return []
    if not keys:
        return [curve for curve in reduction.curves if not curve.is_empty]
    return [curve for key in keys if (curve := reduction.curve(key)) is not None and not curve.is_empty]


def batch_stem(paths: Sequence[Path], source: Any = None) -> str:
    """A name for a batch: the source for one listed source, else the folder the files are in. ``source``:
    what the one path stands for (its ``NexusLayout``, or a frame's metadata) — a set of module files is
    named by its scan, without the module number (``ports.source_stem``)."""
    paths = [Path(path) for path in paths]
    if not paths:
        return "batch"
    if len(paths) == 1:
        return source_stem(paths[0], source)
    parents = {path.parent for path in paths}
    return (paths[0].parent.name if len(parents) == 1 else "batch") or "batch"


def export_frame(
    export: ExportAnalysis,
    analysis: FrameAnalysis,
    destination: Path,
    choices: BatchChoices,
    *,
    cake=None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> list[Path]:
    """What ``choices`` asks for of one frame, except the images, the fits and the tables."""
    destination = Path(destination)
    written: list[Path] = []
    stem = export_stem(analysis)
    text = choices.text_format
    if choices.per_frame:
        keys = [curve.key for curve in chosen_curves(analysis, choices.curves)]
        if keys:
            written.extend(export.curves(
                analysis, keys, destination / FOLDERS["per_frame"], extra_metadata, text_format=text,
            ))
    if choices.fit_input:
        try:
            written.append(export.fit_input(analysis, destination / FOLDERS["fit_input"], halves=choices.fit_halves))
        except ValueError:
            pass  # a frame without a usable curve: nothing to fit
    if choices.q_map and analysis.reduction is not None and analysis.reduction.reciprocal_space_map is not None:
        written.append(export.q_map(analysis, destination / FOLDERS["q_map"] / f"{stem}_qmap.{text}"))
    if choices.cake and cake is not None and analysis.kind != GISAXS:
        written.append(export.cake(analysis, cake, destination / FOLDERS["cake"] / f"{stem}_cake.{text}"))
    if choices.frames:
        data = analysis.raw_data if analysis.raw_data is not None else analysis.data
        source = {"source_file": str(analysis.path), "frame_index": int(analysis.frame_index),
                  "frame_count": int(analysis.frame_count)}
        if analysis.summed_frames:
            source["summed_frames"] = [{"file": str(path), "frame_index": int(index)} for path, index in analysis.summed_frames]
        files = source_files(analysis.path, analysis.metadata)
        if files:
            source["source_files"] = files
        written.append(write_frame(
            destination / FOLDERS["frames"] / f"{stem}{frame_suffix(choices.frame_format)}", data,
            choices.frame_format, source,
        ))
    return written


def fit_curve_rows(analysis: FrameAnalysis, fits) -> list[tuple[str, list[str], list[list[float]]]]:
    """``(curve key, header, rows)`` of the fitted curves of a frame (q, the data fitted, the fit)."""
    out = []
    for target, fit in fits:
        if not fit.ok or fit.x.size == 0:
            continue
        out.append((target.key, ["q (1/A)", "I", "fit"],
                    [[float(a), float(b), float(c)] for a, b, c in zip(fit.x, fit.y, fit.fitted)]))
    return out


class CurveTables:
    """The chosen curves of every frame, gathered for one table per curve (``batch_tables.common_x`` puts the
    frames on one column of x: matched, never drawn across a hole; the header says how)."""

    def __init__(self) -> None:
        self._columns: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {}
        self._titles: dict[str, tuple[str, str]] = {}

    def add(self, label: str, curves: Sequence[Curve]) -> None:
        for curve in curves:
            self._columns.setdefault(curve.key, []).append(
                (str(label), np.asarray(curve.x, dtype=float), np.asarray(curve.intensity, dtype=float))
            )
            self._titles.setdefault(curve.key, (curve.title, curve.x_label))

    @property
    def keys(self) -> list[str]:
        return list(self._columns)

    def frames(self, key: str) -> int:
        return len(self._columns.get(key, ()))

    def write(self, writer: CurveWriter, destination: Path, stem: str, text_format: str = "csv",
              notes: Sequence[str] = ()) -> list[Path]:
        """One table per curve; ``notes``: further ``#`` lines of every table (frames that are missing, sums of
        different sizes)."""
        written = []
        for key in self.keys:
            columns = self._columns[key]
            title, x_label = self._titles[key]
            x, values, grid = common_x([(c_x, c_y) for _label, c_x, c_y in columns])
            comments = [
                f"{BATCH_TITLE} ({SOFTWARE}): {title}",
                f"one column per frame ({len(columns)} frames, in list order); mean intensity per pixel; blank = no data",
                grid, *notes,
            ]
            header = [x_label, *(label for label, _x, _y in columns)]
            rows = [[float(value), *(float(column[index]) for column in values)] for index, value in enumerate(x)]
            safe = "".join(character if character.isalnum() else "_" for character in key).strip("_")
            written.append(writer.write_table(Path(destination) / f"{stem}_{safe}_frames.{text_format}", comments, header, rows))
        return written


__all__ = [
    "BATCH_TITLE", "BatchChoices", "CurveTables", "FIT_HALVES", "FOLDERS", "FRAME_FORMATS", "IMAGE_FORMATS",
    "TEXT_FORMATS", "batch_stem", "chosen_curves", "export_frame", "fit_curve_rows", "fit_halves", "output_names",
]
