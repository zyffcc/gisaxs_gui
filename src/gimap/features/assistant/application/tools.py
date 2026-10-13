"""Run the model's tool calls on the workbench and the GIWAXS metrics.

Every result the model sees is JSON with 4 significant digits; the full
results are also kept in ``RunResults`` so the report shows the computed
numbers rather than retyped ones.
"""

from __future__ import annotations

import base64
import json
import math
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ..domain import (
    DEFAULT_BACKGROUND_WINDOW,
    DEFAULT_SHAPE_FACTOR,
    Profile,
    caveat,
    compare_sectors,
    downsample,
    find_peaks,
    ring_orientation,
    scherrer_size,
    significant,
)
from .gisaxs_tools import GisaxsToolsMixin
from .giwaxs_tools import GiwaxsToolsMixin
from .models import (
    PERMISSION_CONFIRM,
    PERMISSION_PREVIEW,
    AnalysisGoals,
    AssistantReport,
    ReportItem,
    RunResults,
    ToolCall,
    ToolInputError,
    ToolOutcome,
)
from .calibration_tools import CalibrationToolsMixin
from .operation_tools import OperationToolsMixin
from .operations import OPERATION_TOOLS
from .pipeline_progress import FrameChanged
from .ports import AnalysisWorkbench, Chooser, Confirmer, CurveFitter, FileExplorer, GeometryCalibrator, ResultStore
from .tool_specs import FINAL, WRITE, tool_specs

HALO_HALF_WINDOW = 0.04
"""Largest half width of a ring window around a broad halo, as a fraction of its q."""

_TYPES = {
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "string": (str,),
    "array": (list, tuple),
    "object": (dict,),
    "null": (type(None),),
}


def clean(value: Any, digits: int = 4) -> Any:
    """JSON-able copy: dataclasses and arrays unpacked, floats rounded, NaN → None."""
    if is_dataclass(value):
        return {item.name: clean(getattr(value, item.name), digits) for item in fields(value)}
    if isinstance(value, dict):
        return {str(key): clean(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item, digits) for item in value]
    if isinstance(value, np.ndarray):
        return [clean(item, digits) for item in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return None if not math.isfinite(value) else significant(value, digits)
    return value


def _check_type(name: str, schema: dict, value: Any) -> None:
    allowed = schema.get("type")
    if allowed is None:
        return
    kinds = allowed if isinstance(allowed, list) else [allowed]
    ok = any(
        isinstance(value, _TYPES[kind]) and not (kind in ("number", "integer") and isinstance(value, bool))
        for kind in kinds
    )
    if not ok:
        raise ToolInputError(f"'{name}' must be {' or '.join(kinds)}, not {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        raise ToolInputError(f"'{name}' must be one of {schema['enum']}")
    if isinstance(value, (list, tuple)) and "items" in schema:
        for index, item in enumerate(value):
            _check_type(f"{name}[{index}]", schema["items"], item)
    if isinstance(value, dict) and "properties" in schema:
        _check_object(schema, value, prefix=f"{name}.")


def _check_object(schema: dict, value: dict, *, prefix: str = "") -> None:
    properties = schema.get("properties", {})
    unknown = sorted(set(value) - set(properties))
    if unknown:
        raise ToolInputError(f"Unknown argument(s): {', '.join(prefix + key for key in unknown)}")
    missing = [prefix + key for key in schema.get("required", []) if key not in value]
    if missing:
        raise ToolInputError(f"Missing argument(s): {', '.join(missing)}")
    for key, item in value.items():
        if item is None:
            if "null" not in str(properties[key].get("type")):
                raise ToolInputError(f"'{prefix}{key}' must not be null")
            continue
        _check_type(prefix + key, properties[key], item)


def validate(schema: dict, arguments: Any) -> dict:
    """Required keys, no unknown keys, types and enums, also inside arrays and nested objects.

    Tool inputs may arrive truncated or malformed when they are streamed
    unbuffered, so every input is checked here before a tool runs.  Numbers are
    not range-checked here.
    """
    if not isinstance(arguments, dict):
        raise ToolInputError("The tool input must be a JSON object")
    _check_object(schema, arguments)
    return dict(arguments)


def results_payload(results: RunResults) -> dict:
    """The computed tables of a run (for export and the saved report)."""
    searches = {
        curve: {
            key: value
            for key, value in clean(search).items()
            if key not in ("baseline_x", "baseline_y")
        }
        for curve, search in results.peak_searches.items()
    }
    return {
        "status": clean(results.status),
        "peaks": searches,
        "sectors": clean(results.sector_rows),
        "rings": clean(results.rings),
        "crystallite_sizes": clean(results.sizes),
        "report": clean(results.report),
        "calibrations": clean(results.calibrations),
        "geometry_used": clean(results.geometry_used),
        "feature_requests": clean(results.feature_requests),
        "operations": [operation.record() for operation in results.operations],
        "exports": list(results.exports),
    }


class ToolCatalog(CalibrationToolsMixin, OperationToolsMixin, GisaxsToolsMixin, GiwaxsToolsMixin):
    def __init__(
        self,
        workbench: AnalysisWorkbench,
        goals: AnalysisGoals,
        results: RunResults,
        *,
        confirmer: Optional[Confirmer] = None,
        store: Optional[ResultStore] = None,
        explorer: Optional[FileExplorer] = None,
        calibrator: Optional[GeometryCalibrator] = None,
        chooser: Optional[Chooser] = None,
        fitter: Optional[CurveFitter] = None,
        cancelled: Callable[[], bool] = lambda: False,
    ):
        self.workbench = workbench
        self.fitter = fitter
        self.goals = goals
        self.results = results
        self.confirmer = confirmer
        self.store = store
        self.explorer = explorer
        self.calibrator = calibrator
        self.chooser = chooser
        self.cancelled = cancelled
        self._specs = {spec.name: spec for spec in tool_specs(allow_images=goals.allow_images)}
        self._setup_calibration()

    def definitions(self) -> list[dict]:
        return [spec.definition() for spec in self._specs.values()]

    def is_final(self, name: str) -> bool:
        spec = self._specs.get(name)
        return spec is not None and spec.kind == FINAL

    # -- execution -------------------------------------------------------------------

    def execute(self, call: ToolCall) -> ToolOutcome:
        spec = self._specs.get(call.name)
        if spec is None:
            return ToolOutcome(f"Unknown tool '{call.name}'.", "unknown tool", is_error=True)
        try:
            arguments = validate(spec.schema, call.input or {})
        except ToolInputError as exc:
            return ToolOutcome(json.dumps({"error": str(exc)}), f"invalid input: {exc}", is_error=True)
        # Preview first restores settings changes at the end, so only real writes (files, profiles) ask.
        asks = self.goals.permission == PERMISSION_CONFIRM or (
            self.goals.permission == PERMISSION_PREVIEW and spec.name not in OPERATION_TOOLS
        )
        if spec.kind == WRITE and asks:
            question = self._describe_write(spec.name, arguments)
            if self.confirmer is None or not self.confirmer.confirm("The AI wants to change something", question):
                data = {"declined": True, "message": f"The user declined, so this was not done: {question}"}
                return ToolOutcome(json.dumps(data), "declined by the user", data=data)
        try:
            if spec.name in OPERATION_TOOLS:
                return self._tracked(spec.name, arguments)
            return getattr(self, f"_tool_{spec.name}")(**arguments)
        except ToolInputError as exc:
            return ToolOutcome(json.dumps({"error": str(exc)}), str(exc), is_error=True)
        except FrameChanged as exc:  # the workbench refused to act on another file than the run's
            data = {"error": str(exc), "frame_changed": True}
            return ToolOutcome(json.dumps(data), f"failed: {exc}", is_error=True, data=data)
        except Exception as exc:  # a tool failure is reported to the model, never raised
            message = str(exc) or type(exc).__name__
            return ToolOutcome(json.dumps({"error": message}), f"failed: {message}", is_error=True)

    @staticmethod
    def _ok(data: Any, summary: str, *, final: bool = False) -> ToolOutcome:
        payload = clean(data)
        return ToolOutcome(json.dumps(payload, ensure_ascii=False), summary, final=final, data=payload)

    def _describe_write(self, name: str, arguments: dict) -> str:
        if name == "use_geometry":
            return self._describe_geometry_write(arguments)
        if name == "export_results":
            tables = " and the computed tables" if arguments.get("include_tables") else ""
            return f"Write the curves{tables} next to the data (gimap_analysis/)."
        if name == "set_valid_intensity_range":
            return (
                f"Exclude pixels outside {arguments.get('minimum')} … {arguments.get('maximum')} counts "
                "from the analysis for this session."
            )
        return f"Run {name}."

    # -- helpers ---------------------------------------------------------------------

    def _profile(self, key: str):
        curve = self.workbench.curve(key)
        if curve is None:
            available = [item.get("key") for item in (self.workbench.status().get("curves") or [])]
            raise ToolInputError(f"Curve '{key}' is not available; available: {available}")
        return curve, Profile.of(curve.x, curve.y, curve.sigma, curve.pixels)

    def _radial_search(self):
        search = self.results.peak_searches.get("radial")
        if search is None:
            _curve, profile = self._profile("radial")
            search = find_peaks(profile)
            self.results.peak_searches["radial"] = search
        return search

    @staticmethod
    def _nearest_peak(search, q: float):
        if search is None or not search.peaks:
            return None
        peak = min(search.peaks, key=lambda item: abs(item.q - q))
        return peak if abs(peak.q - q) <= max(2.0 * peak.fwhm, 3.0 * search.step) else None

    def _status_summary(self, status: dict) -> str:
        self.results.status = status
        kind = (status.get("measurement") or "no reduction").upper()
        return f"{kind} · {status.get('file') or 'no frame'}"

    # -- tools -----------------------------------------------------------------------

    def _tool_get_status(self) -> ToolOutcome:
        status = self.workbench.status()
        return self._ok(status, self._status_summary(status))

    def _tool_run_standard_pipeline(self, **arguments) -> ToolOutcome:
        from .pipeline import PipelineOptions, StandardPipeline, compact_report  # the pipeline drives this catalog

        notes = "\n".join(text for text in (self.goals.instructions, self.goals.standing_instructions) if text.strip())
        values = {key: value for key, value in arguments.items() if value is not None}
        report = StandardPipeline(self, PipelineOptions(notes=notes, ring_q=self.goals.ring_q, **values)).run()
        self.results.pipeline = report
        attention = report.get("needs_attention") or []
        summary = (
            f"baseline {'done' if report.get('ok') else 'stopped'}: {len(report.get('steps') or [])} steps"
            + (f", {len(attention)} open question(s)" if attention else "")
        )
        return self._ok(compact_report(report), summary)

    def _tool_set_measurement_mode(self, mode: str) -> ToolOutcome:
        status = self.workbench.set_mode(mode)
        return self._ok(status, f"mode {mode} → {self._status_summary(status)}")

    def _tool_set_frame(self, frame: int, sum=None) -> ToolOutcome:  # noqa: A002 - the tool's argument name
        total = max(1, int((self.results.status or self.workbench.status()).get("frames") or 1))
        count = max(1, min(int(sum or 1), total))
        number = int(frame)
        if number < 0:  # counted from the end: -1 is the last frame
            number += total + 1
        # A summed block never runs past the last frame: one that would ends there instead.
        number = min(max(1, number), total - count + 1)
        status = self.workbench.set_frame(number, count)
        self.results.status = status
        span = f"frames {number}–{number + count - 1} summed" if count > 1 else f"frame {number}"
        return self._ok(status, f"{span} of {total}")

    def _tool_set_incidence_angle(self, degrees: Optional[float]) -> ToolOutcome:
        if degrees is not None and not 0.0 < degrees < 10.0:
            raise ToolInputError("The incidence angle must be between 0 and 10 degrees (or null)")
        status = self.workbench.set_incidence(degrees)
        self.results.status = status  # the report's geometry gives the αi the run used, not the one before it
        return self._ok(status, f"αi = {'profile' if degrees is None else f'{degrees:g}°'}")

    def _tool_set_sector_widths(self, in_plane_half_width_deg: float, out_of_plane_half_width_deg: float) -> ToolOutcome:
        in_plane = min(45.0, max(0.5, float(in_plane_half_width_deg)))
        out_of_plane = min(45.0, max(0.5, float(out_of_plane_half_width_deg)))
        status = self.workbench.set_sector_widths(in_plane, out_of_plane)
        return self._ok(status, f"sectors ±{in_plane:g}° in-plane, ±{out_of_plane:g}° out-of-plane")

    def _tool_set_radial_bins(self, bins: int) -> ToolOutcome:
        value = None if int(bins) <= 0 else min(5000, max(50, int(bins)))
        status = self.workbench.set_radial_bins(value)
        return self._ok(status, f"radial bins {value or 'auto'}")

    def _tool_set_custom_sector(self, enabled: bool, chi_min_deg=None, chi_max_deg=None, q_min=None, q_max=None) -> ToolOutcome:
        if enabled and (chi_min_deg is None or chi_max_deg is None):
            raise ToolInputError("chi_min_deg and chi_max_deg are needed to add a sector")
        chi = (float(chi_min_deg), float(chi_max_deg)) if enabled else None
        status = self.workbench.set_custom_sector(chi, (q_min, q_max))
        return self._ok(status, f"custom sector χ = {chi_min_deg}…{chi_max_deg}°" if enabled else "custom sector removed")

    def _tool_set_q_box(self, enabled: bool, q_parallel_min=None, q_parallel_max=None, qz_min=None, qz_max=None) -> ToolOutcome:
        values = (q_parallel_min, q_parallel_max, qz_min, qz_max)
        if enabled and any(value is None for value in values):
            raise ToolInputError("All four box limits are needed to add a q box")
        box = ((q_parallel_min, q_parallel_max), (qz_min, qz_max)) if enabled else (None, None)
        status = self.workbench.set_q_box(*box)
        return self._ok(status, "q box set" if enabled else "q box removed")

    def _tool_get_curve(self, curve: str, max_points: int = 150, x_min=None, x_max=None) -> ToolOutcome:
        data, profile = self._profile(curve)
        span = None
        if x_min is not None or x_max is not None:
            finite = profile.measured()
            span = (
                x_min if x_min is not None else float(finite.x.min()),
                x_max if x_max is not None else float(finite.x.max()),
            )
        measured = profile.measured(span)
        reduced = downsample(measured, max(10, min(int(max_points), 400)))
        payload = {
            "curve": curve,
            "title": data.title,
            "x_label": data.x_label,
            "points_total": measured.size,
            "points_returned": reduced.size,
            "x": reduced.x,
            "y": reduced.y,
            "sigma": reduced.sigma,
        }
        return self._ok(payload, f"{curve}: {reduced.size} points")

    def _tool_find_peaks(self, curve: str, q_min=None, q_max=None, min_snr: float = 3.0, background_window=None) -> ToolOutcome:
        _data, profile = self._profile(curve)
        span = None
        if q_min is not None or q_max is not None:
            finite = profile.measured()
            span = (
                q_min if q_min is not None else float(finite.x.min()),
                q_max if q_max is not None else float(finite.x.max()),
            )
        search = find_peaks(
            profile,
            x_range=span,
            min_snr=max(1.0, float(min_snr)),
            background_window=background_window or DEFAULT_BACKGROUND_WINDOW,
        )
        self.results.peak_searches[curve] = search
        payload = {
            "curve": curve,
            "q_range": search.x_range,
            "bins": search.points,
            "bin_width": search.step,
            "background_window": search.background_window,
            "min_snr": search.min_snr,
            "peaks": [
                {
                    "index": index + 1, "q": peak.q, "q_err": peak.q_err, "d_A": peak.d, "d_err_A": peak.d_err,
                    "fwhm": peak.fwhm, "fwhm_err": peak.fwhm_err, "height": peak.height,
                    "background": peak.background, "area": peak.area, "snr": peak.snr, "flags": peak.flags,
                }
                for index, peak in enumerate(search.peaks)
            ],
            "series_hints": search.series,
        }
        if not search.peaks:  # with peaks listed it would only invite reading a spike as one
            payload["strongest_feature"] = search.strongest
        if search.reason:
            payload["reason"] = search.reason
        summary = (
            f"{len(search.peaks)} peak(s) on {curve}: " + ", ".join(f"{peak.q:.3g}" for peak in search.peaks[:6])
            if search.peaks else f"no peaks on {curve}"
        )
        return self._ok(payload, summary)

    def _tool_compare_sectors(self, q_values=None, window_fwhm: float = 1.0) -> ToolOutcome:
        search = self._radial_search()
        if q_values:
            pairs = []
            for q in q_values:
                peak = self._nearest_peak(search, float(q))
                pairs.append((float(q), peak.fwhm if peak is not None else 0.02 * float(q)))
        else:
            pairs = [(peak.q, peak.fwhm) for peak in search.peaks]
        if not pairs:
            raise ToolInputError("No peaks to compare: find_peaks on 'radial' found none; pass q_values.")
        _ip, in_plane = self._profile("in_plane")
        _oop, out_of_plane = self._profile("out_of_plane")
        rows = compare_sectors(
            pairs, in_plane, out_of_plane,
            background_window=search.background_window, window_fwhm=max(0.25, float(window_fwhm)),
            reference_background=search.background_at,
        )
        self.results.sector_rows = rows
        giwaxs = (self.results.status or self.workbench.status()).get("giwaxs") or {}
        payload = {
            "sector_half_widths_deg": {
                "in_plane": giwaxs.get("in_plane_half_width_deg"),
                "out_of_plane": giwaxs.get("out_of_plane_half_width_deg"),
            },
            "rows": [
                {
                    "q": row.q, "fwhm": row.fwhm, "in_plane": row.in_plane, "in_plane_err": row.in_plane_err,
                    "out_of_plane": row.out_of_plane, "out_of_plane_err": row.out_of_plane_err,
                    "ratio_out_in": row.ratio, "ratio_err": row.ratio_err,
                    "preference": row.preference, "note": row.note,
                }
                for row in rows
            ],
        }
        return self._ok(payload, "; ".join(f"{row.q:.3g}: {row.preference}" for row in rows)[:160])

    def _tool_ring_orientation(self, q_center: float, q_half_width=None) -> ToolOutcome:
        if not q_center > 0:
            raise ToolInputError("q_center must be positive")
        search = self._radial_search()
        peak = self._nearest_peak(search, float(q_center))
        center = float(q_center) if peak is None else float(peak.q)
        if q_half_width is not None and q_half_width > 0:
            half = float(q_half_width)
        elif peak is not None:
            half = max(0.75 * peak.fwhm, 2.0 * search.step)
            if "broad" in peak.flags:
                # A wide window reaches the detector edges at some χ and mixes q there.
                half = min(half, HALO_HALF_WINDOW * peak.q)
        else:
            half = 0.015 * center
        low, high = center - half, center + half
        self.workbench.set_chi_window(low, high)
        self.workbench.show(lower_profile="azimuthal")
        _curve, profile = self._profile("azimuthal")
        background = search.background_at(center) or 0.0
        result = ring_orientation(profile, q_window=(low, high), background=background)
        self.results.rings.append(result)
        payload = clean(result)
        if peak is None:
            payload["peak_used"] = None
            payload["analysed"] = f"no fitted peak near q = {q_center:g} Å⁻¹: the window is centred on the q asked for"
        else:
            payload["peak_used"] = {"q": peak.q, "fwhm": peak.fwhm, "flags": list(peak.flags), "caveat": caveat(peak)}
            if abs(peak.q - float(q_center)) > max(0.25 * peak.fwhm, 2.0 * search.step):
                payload["analysed"] = (
                    f"no peak at q = {q_center:g} Å⁻¹ in this frame: the nearest feature, at {peak.q:.4g} Å⁻¹, was analysed"
                )
        summary = (
            f"q = {center:.3g}: {result.texture}, f = {result.herman:.2f}"
            if result.herman is not None else f"q = {center:.3g}: {result.reason or result.texture}"
        )
        if peak is not None and caveat(peak):
            summary += f" ({caveat(peak).split(':')[0]})"
        return self._ok(payload, summary)

    def _tool_crystallite_size(
        self, q_center: float, curve: str = "radial", shape_factor: float = DEFAULT_SHAPE_FACTOR,
        instrumental_fwhm: float = 0.0,
    ) -> ToolOutcome:
        search = self.results.peak_searches.get(curve)
        if search is None:
            raise ToolInputError(f"Run find_peaks on '{curve}' first.")
        peak = self._nearest_peak(search, float(q_center))
        if peak is None:
            listed = ", ".join(f"{item.q:.4g}" for item in search.peaks) or "none"
            raise ToolInputError(f"No fitted peak near q = {q_center:g} Å⁻¹ on '{curve}' (peaks: {listed}).")
        size = scherrer_size(
            peak.q, peak.fwhm, peak.fwhm_err if math.isfinite(peak.fwhm_err) else None,
            shape_factor=float(shape_factor), instrumental_fwhm=max(0.0, float(instrumental_fwhm)),
            bin_width=search.step,
        )
        self.results.sizes.append(size)
        payload = clean(size)
        payload["size_nm"] = None if size.size is None else significant(size.size / 10.0)
        summary = (
            f"q = {peak.q:.3g}: L {'≥ ' if size.lower_bound else ''}{size.size / 10.0:.3g} nm"
            if size.size is not None else f"q = {peak.q:.3g}: {size.reason}"
        )
        return ToolOutcome(json.dumps(payload, ensure_ascii=False), summary, data=payload)

    def _tool_show_view(self, view=None, lower_plot=None) -> ToolOutcome:
        self.workbench.show(view, lower_plot)
        return self._ok({"view": view, "lower_plot": lower_plot}, f"showing {view or ''} {lower_plot or ''}".strip())

    def _tool_view_preview(self) -> ToolOutcome:
        image = self.workbench.preview_png(900)
        if not image:
            raise ToolInputError("No q map is available (no GIWAXS reduction yet).")
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(image).decode("ascii")}},
            {"type": "text", "text": "q∥–qz map of the current frame, log intensity (qualitative only)."},
        ]
        return ToolOutcome(content, "q map preview sent", data={"image_bytes": len(image)})

    def _tool_note_missing_capability(self, capability: str, reason: str) -> ToolOutcome:
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "file": (self.results.status or {}).get("path"),
            "capability": capability,
            "reason": reason,
        }
        self.results.feature_requests.append(entry)
        if self.store is not None:
            self.store.append_feature_request(entry)
        return self._ok({"recorded": True}, f"missing: {capability}")

    def _tool_set_valid_intensity_range(self, minimum, maximum) -> ToolOutcome:
        status = self.workbench.set_valid_range(minimum, maximum)
        return self._ok(status, f"valid range {minimum} … {maximum}")

    def _tool_export_results(self, include_tables: bool) -> ToolOutcome:
        paths = list(self.workbench.export_curves())
        if include_tables and self.store is not None:
            status = self.results.status or self.workbench.status()
            source = Path(status.get("path") or "frame")
            folder = str(Path(paths[0]).parent) if paths else str(source.parent / "gimap_analysis")
            paths.append(self.store.write_tables(folder, source.stem, results_payload(self.results)))
        self.results.exports.extend(paths)
        return self._ok({"written": paths}, f"wrote {len(paths)} file(s)")

    def _tool_submit_report(self, summary: str, items: list, caveats: list, suggestions: list) -> ToolOutcome:
        report = AssistantReport(
            summary=summary,
            items=tuple(
                ReportItem(
                    item=str(entry.get("item", "other")), status=str(entry.get("status", "partial")),
                    findings=str(entry.get("findings", "")), evidence=str(entry.get("evidence", "")),
                    reason=str(entry.get("reason", "")),
                )
                for entry in items
            ),
            caveats=tuple(str(text) for text in caveats),
            suggestions=tuple(str(text) for text in suggestions),
        )
        self.results.report = report
        return ToolOutcome("Report received. The run is complete.", "report submitted", final=True)


__all__ = ["ToolCatalog", "ToolInputError", "ToolOutcome", "clean", "results_payload", "validate"]
