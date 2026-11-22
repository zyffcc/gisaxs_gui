"""Calibrate a set-up from one isotropic transmission standard or its radial curve."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.gimap.shared.file_digest import file_record, input_digests
from src.gimap.shared.geometry.instrument_profile import scale_geometry

from ..domain import GIWAXS, GiwaxsSettings, Normalisation, describe_intensity_corrections, reduce_transmission, valid_pixels
from ..domain.absolute_scale import ScaleFit, fit_standard, water_intensity
from ..domain.normalisation import mm_to_cm
from .export_record import _plain, source_files
from .detector_response import with_profile_response
from .reduction_errors import reduction_errors
from .frame_preprocessing import FramePreprocessingMixin, floating
from .standard_curves import read_standard_curve


@dataclass(frozen=True)
class StandardRequest:
    source_kind: str  # frame or curve
    source_path: Path | str
    frame_index: int = 0  # zero-based
    exposure_s: float = 1.0
    monitor: float = 1.0
    transmission: float = 1.0
    thickness_mm: float = 1.0
    source_q_unit: str = "Å⁻¹"
    reference_path: Path | str | None = None
    reference_q_unit: str = "Å⁻¹"
    temperature_c: float = 20.0
    reference_fractional_sigma: float = 0.0
    q_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class StandardCalibration:
    fit: ScaleFit
    record: dict


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise InterruptedError("Calibration cancelled.")


class CalibrateStandard(FramePreprocessingMixin):
    """Snapshot of the sample's geometry and correction choices, with its own preprocessing caches."""

    def __init__(self, frames, profile, geometry, shape, corrections, mode, bins=None):
        self.frames = frames
        self.profile_name = profile.name
        self.geometry = geometry
        self.shape = tuple(shape) if shape is not None else None
        self.corrections = replace(with_profile_response(corrections, profile), normalisation=Normalisation())
        self.mode = mode
        self.bins = bins
        self._init_preprocessing()

    def _frame_curve(self, path, index, cancelled):
        count = self.frames.frame_count(path)
        if not 0 <= index < count:
            raise ValueError("The selected frame number is outside this detector file.")
        image = self.frames.load(path, index)
        data = np.asarray(image.data, dtype=np.float32)
        if self.shape is not None and data.shape != self.shape:
            raise ValueError("The standard frame must have the same detector shape as the current set-up.")
        _check_cancelled(cancelled)
        valid = valid_pixels(data, image.mask, negatives_valid=floating(image.metadata))
        common = {"metadata": image.metadata}
        self.correct_pixels(common, data, valid, self.corrections, self.geometry, self.mode)
        _check_cancelled(cancelled)
        reduced = reduce_transmission(
            common["data"], common["valid"], self.geometry, GiwaxsSettings(bins=self.bins),
            with_map=False, counts=reduction_errors(self.corrections, common),
        )
        curve = reduced.curve("radial")
        if curve is None or curve.is_empty:
            raise ValueError("The standard frame has no usable radial intensity after corrections.")
        _check_cancelled(cancelled)
        corrections = self.corrections
        paths = [p for p in (corrections.mask_path, corrections.dark_path, corrections.background_path,
                             corrections.flat_field_path) if p]
        if common["error_model_used"] == "sigma":
            paths.append(corrections.sigma_path)
        corrections_record = _plain(corrections)
        corrections_record["reference_dark"] = common.get("reference_dark")
        corrections_record["error_model_used"] = common["error_model_used"]
        if common.get("flat_field"):
            corrections_record["flat_field"] = common["flat_field"].to_record()
        corrections_record.pop("normalisation")  # the entered terms are recorded separately
        sources = source_files(path, image.metadata) or [str(path)]
        record = {
            "frame_index": index, "frame_number": index + 1, "frame_count": count,
            "reduction": "full azimuthal mean about the direct beam in transmission",
            "geometry": _plain(self.geometry), "shape": list(data.shape),
            "detector_name": image.detector_name, "source_files": sources, "input_sha256": input_digests(sources),
            "pixel_correction_mode": self.mode, "corrections": corrections_record, "correction_files": input_digests(paths),
            "intensity_corrections": describe_intensity_corrections(corrections) if self.mode == GIWAXS else None,
            "radial_bins": _plain(reduced.markers["radial_bins"]),
        }
        return (curve.x, curve.intensity, curve.sigma), record

    def __call__(self, request: StandardRequest, *, cancelled=None) -> StandardCalibration:
        _check_cancelled(cancelled)
        path = Path(request.source_path).resolve()
        if request.source_kind == "frame":
            measured, details = self._frame_curve(path, request.frame_index, cancelled)
        elif request.source_kind == "curve":
            measured = read_standard_curve(path, request.source_q_unit)
            details = {"source_q_unit": request.source_q_unit, "source_file": file_record(path)}
        else:
            raise ValueError("Choose a detector frame or a reduced standard curve.")
        _check_cancelled(cancelled)
        reference = None
        if request.reference_path is None:
            standard = "Water"
            ref_details = {"temperature_c": request.temperature_c, "intensity_cm_inv": water_intensity(request.temperature_c),
                           "literature_doi": "10.1107/S0021889899015216"}
        else:
            ref_path = Path(request.reference_path).resolve()
            reference = read_standard_curve(ref_path, request.reference_q_unit, reference=True)
            standard = ref_path.name
            ref_details = {"file": file_record(ref_path), "q_unit": request.reference_q_unit, "intensity_unit": "cm⁻¹"}
        _check_cancelled(cancelled)
        fit = fit_standard(*measured, exposure_s=request.exposure_s, monitor=request.monitor,
                           transmission=request.transmission, thickness_mm=request.thickness_mm, reference=reference,
                           temperature_c=request.temperature_c, q_range=request.q_range,
                           reference_fractional_sigma=request.reference_fractional_sigma)
        record = {
            "standard": standard, "source_kind": request.source_kind, "source_path": str(path),
            "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "instrument_profile": self.profile_name,
            "instrument_geometry": scale_geometry(self.geometry),
            "detector_shape": details.get("shape", list(self.shape) if self.shape is not None else None),
            "factor": float(fit.factor), "sigma_factor": float(fit.sigma_factor), "intensity_unit": "cm⁻¹",
            "q_range_A_inv": [float(fit.q[0]), float(fit.q[-1])], "fitted_points": len(fit.q),
            "requested_q_range_A_inv": list(request.q_range) if request.q_range is not None else None,
            "reference": ref_details, "reference_fractional_sigma": request.reference_fractional_sigma,
            "standard_normalisation": {"exposure_s": request.exposure_s, "monitor": request.monitor,
                                       "transmission": request.transmission, "thickness_mm": request.thickness_mm,
                                       "thickness_cm": mm_to_cm(request.thickness_mm)},
            **details,
        }
        return StandardCalibration(fit, record)
