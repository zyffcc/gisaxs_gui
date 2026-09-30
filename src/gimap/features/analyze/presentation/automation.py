"""Operate the Analyze page as a user would, for another component (the assistant).

Every setter changes the same controls the page shows (the Options panel,
the mode switch, the lower-plot choice) and the same view-model state, then
re-analyses once.  ``done(ok, message)`` is called on the GUI thread when the
new analysis is on screen or has failed.  Curves come back with q in Å⁻¹ even
when the page shows 2θ.  All methods must be called on the GUI thread.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from PyQt5.QtCore import QSignalBlocker

from src.gimap.shared.geometry import DetectorGeometry

from ..application import q_from_two_theta, refine_center_by_symmetry, regions_from_dicts
from .geometry_dialog import geometry_defaults
from .views.analyze_page_view import INCIDENCE_FROM_PROFILE

Done = Callable[[bool, str], None]


def _number(metadata: dict, key: str, scale: float = 1.0) -> Optional[float]:
    value = metadata.get(key)
    return float(value) * scale if isinstance(value, (int, float)) and np.isfinite(value) else None


def header_values(metadata: dict) -> dict:
    """What the file header says about the measurement (``header_*`` values can be stale)."""
    center = None
    if metadata.get("beam_center_x_px") is not None and metadata.get("beam_center_y_px") is not None:
        center = [float(metadata["beam_center_x_px"]), float(metadata["beam_center_y_px"])]
    else:
        pair = metadata.get("header_beam_xy_px") or metadata.get("header_beam_center_px")
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            center = [float(pair[0]), float(pair[1])]
    values = {
        "detector": metadata.get("detector_name"),
        "pixel_size_um": [_number(metadata, "pixel_size_x_m", 1e6), _number(metadata, "pixel_size_y_m", 1e6)],
        "energy_kev": _number(metadata, "energy_kev"),
        "wavelength_angstrom": _number(metadata, "wavelength_angstrom") or _number(metadata, "header_wavelength_angstrom"),
        "distance_mm": _number(metadata, "distance_m", 1e3) or _number(metadata, "header_distance_m", 1e3),
        "beam_center_px_as_written": center,
        "exposure_s": _number(metadata, "exposure_time_s"),
        "timestamp": metadata.get("timestamp"),
    }
    return {key: value for key, value in values.items() if value not in (None, [None, None])}


class AnalyzeAutomation:
    def __init__(self, page):
        self._page = page
        self._waiting: list[Done] = []
        page.analysisShown.connect(self._shown)
        page.analysisFailed.connect(self._failed)
        self._symmetry: Optional[dict] = None

    # -- re-analysis -------------------------------------------------------------------

    def reanalyse(self, done: Done) -> None:
        if self._page.view_model.request() is None:
            done(False, "No frame is open in Analyze.")
            return
        self._waiting.append(done)
        self._page.run_analysis()

    def _shown(self, _analysis) -> None:
        waiting, self._waiting = self._waiting, []
        for done in waiting:
            done(True, "")

    def _failed(self, message: str) -> None:
        waiting, self._waiting = self._waiting, []
        for done in waiting:
            done(False, message)

    # -- state -------------------------------------------------------------------------

    def _to_q(self, x: np.ndarray, x_label: str) -> tuple[np.ndarray, str]:
        analysis = self._page.view_model.state.analysis
        if x_label.startswith("2θ") and analysis is not None and analysis.geometry is not None:
            return q_from_two_theta(x, analysis.geometry.wavelength_angstrom), "q (Å⁻¹)"
        return np.asarray(x, dtype=float), x_label

    def status(self) -> dict:
        view_model = self._page.view_model
        state = view_model.state
        path = view_model.current_path
        info: dict = {
            "file": path.name if path else None,
            "path": str(path) if path else None,
            "files_listed": len(state.files),
            "mode": state.mode,
            "incidence_override": state.incidence_deg,
        }
        analysis = state.analysis
        if analysis is None or (path is not None and Path(analysis.path) != path):
            info["measurement"] = None
            info["message"] = "No frame has been analysed yet." if path else "No frame is open."
            return info
        info.update(
            frame=analysis.frame_index + 1,
            frames=analysis.frame_count,
            summed_frames=analysis.frame_total,
            detector=analysis.detector_name,
            shape=list(analysis.shape),
            measurement=analysis.kind,
        )
        info["header"] = header_values(analysis.metadata or {})
        geometry = analysis.geometry
        profile = analysis.resolution.profile
        info["geometry"] = None if geometry is None else {
            "distance_mm": geometry.distance_m * 1e3,
            "wavelength_A": geometry.wavelength_angstrom,
            "incidence_deg": geometry.incidence_deg,
            "beam_center_px": [geometry.beam_center_x_px, geometry.beam_center_y_px],
            "pixel_size_um": [geometry.pixel_size_x_m * 1e6, geometry.pixel_size_y_m * 1e6],
            "instrument_profile": profile.name if profile else None,
        }
        corrections = state.corrections
        info["corrections"] = {
            "background": Path(corrections.background_path).name if corrections.background_path else None,
            "valid_range": [corrections.minimum, corrections.maximum],
            "gap_guard_px": corrections.gap_guard_px,
        }
        reduction = analysis.reduction
        if reduction is not None and analysis.kind == "giwaxs":
            giwaxs = state.giwaxs
            window = reduction.markers.get("chi_q_window")
            info["giwaxs"] = {
                "in_plane_half_width_deg": giwaxs.in_plane_half_width_deg,
                "out_of_plane_half_width_deg": giwaxs.out_of_plane_half_width_deg,
                "radial_bins": giwaxs.bins or "auto",
                "chi_q_window": list(window) if window else None,
                "custom_sector": None if giwaxs.sector is None else {
                    "chi_deg": [giwaxs.sector.chi_min_deg, giwaxs.sector.chi_max_deg],
                    "q": [giwaxs.sector.q_min, giwaxs.sector.q_max],
                },
                "q_box": None if giwaxs.box is None else {
                    "q_parallel": list(giwaxs.box.q_parallel), "qz": list(giwaxs.box.qz),
                },
                "regions": [
                    {
                        "name": region.name, "q_min": None if region.q_range is None else region.q_range[0],
                        "q_max": None if region.q_range is None else region.q_range[1],
                        "chi_min_deg": region.chi_range[0], "chi_max_deg": region.chi_range[1],
                        "both_sides": region.both_sides,
                    }
                    for region in giwaxs.regions
                ],
            }
        if reduction is not None and analysis.kind == "gisaxs":
            markers = reduction.markers
            yoneda = markers.get("yoneda")
            low, high = markers["horizontal_band"]
            left, right = markers["vertical_band"]
            info["gisaxs"] = {
                "horizontal_rows": [float(low), float(high)],
                "horizontal_source": markers.get("horizontal_source"),
                "yoneda_alpha_f_deg": None if yoneda is None else float(yoneda.alpha_f_deg),
                "yoneda_row": None if yoneda is None else float(yoneda.row),
                "horizon_row": markers.get("horizon_row"),
                "vertical_columns": [float(left), float(right)],
                "halves": view_model.fit_side,
                "symmetry": self._symmetry,
            }
        curves = []
        for curve in (reduction.curves if reduction is not None else ()):
            measured = np.asarray(curve.pixels) > 0
            if curve.is_empty or not measured.any():
                continue
            x, x_label = self._to_q(np.asarray(curve.x)[measured], curve.x_label)
            curves.append({
                "key": curve.key, "title": curve.title, "points": int(measured.sum()),
                "x_range": [float(x.min()), float(x.max())], "x_label": x_label,
            })
        info["curves"] = curves
        info["messages"] = list(analysis.messages)
        return info

    def curve(self, key: str) -> Optional[dict]:
        analysis = self._page.view_model.state.analysis
        reduction = analysis.reduction if analysis is not None else None
        curve = reduction.curve(key) if reduction is not None else None
        if curve is None or curve.is_empty:
            return None
        x, x_label = self._to_q(curve.x, curve.x_label)
        return {
            "key": curve.key, "title": curve.title, "x": x, "y": np.asarray(curve.intensity, dtype=float),
            "sigma": np.asarray(curve.sigma, dtype=float), "pixels": np.asarray(curve.pixels, dtype=float),
            "x_label": x_label, "region": dict(curve.region),
        }

    # -- settings ----------------------------------------------------------------------

    def _open_options(self, step: str = "cuts") -> None:
        """Show the step whose control is about to change, so a watching user sees it move."""
        self._page.show_step(step)

    @staticmethod
    def _set(widget, value) -> None:
        with QSignalBlocker(widget):
            if hasattr(widget, "setChecked") and isinstance(value, bool):
                widget.setChecked(value)
            else:
                widget.setValue(value)

    def set_mode(self, mode: str, done: Done) -> None:
        page = self._page
        index = page.mode_combo.findData(mode)
        if index >= 0:
            with QSignalBlocker(page.mode_combo):
                page.mode_combo.setCurrentIndex(index)
        page.view_model.set_mode(mode)
        page._remember()
        self.reanalyse(done)

    def set_incidence(self, degrees: Optional[float], done: Done) -> None:
        page = self._page
        self._set(page.incidence_spin, INCIDENCE_FROM_PROFILE if degrees is None else float(degrees))
        page.view_model.set_incidence(degrees)
        page._remember()
        self.reanalyse(done)

    def set_sector_widths(self, in_plane: float, out_of_plane: float, done: Done) -> None:
        page = self._page
        self._open_options()
        self._set(page.in_plane_spin, float(in_plane))
        self._set(page.out_of_plane_spin, float(out_of_plane))
        page.view_model.set_sector_widths(in_plane, out_of_plane)
        self.reanalyse(done)

    def set_radial_bins(self, bins: Optional[int], done: Done) -> None:
        page = self._page
        self._open_options()
        self._set(page.bins_spin, int(bins or 0))
        page.view_model.set_radial_bins(bins)
        self.reanalyse(done)

    def set_custom_sector(self, chi, q_range, done: Done) -> None:
        page = self._page
        self._open_options()
        if chi is None:
            self._set(page.sector_check, False)
            page.sector_grid.setEnabled(False)
            page.view_model.set_sector(None)
        else:
            q_min, q_max = q_range
            self._set(page.sector_chi_min, float(chi[0]))
            self._set(page.sector_chi_max, float(chi[1]))
            self._set(page.sector_q_min, float(q_min or 0.0))
            self._set(page.sector_q_max, float(q_max or 0.0))
            self._set(page.sector_check, True)
            page.sector_grid.setEnabled(True)
            page.view_model.set_sector(tuple(chi), (q_min, q_max))
        self.reanalyse(done)

    def set_cut_regions(self, regions, done: Done) -> None:
        """Replace the cut regions (dicts: name, q_min, q_max, chi_min_deg, chi_max_deg, both_sides)."""
        page = self._page
        self._open_options()
        try:
            cut = regions_from_dicts(regions)
        except (TypeError, ValueError) as exc:
            done(False, f"Invalid region: {exc}")
            return
        page.view_model.set_regions(cut)
        self.reanalyse(done)

    def set_q_box(self, q_parallel, qz, done: Done) -> None:
        page = self._page
        self._open_options()
        if q_parallel is None or qz is None:
            self._set(page.box_check, False)
            page.box_grid.setEnabled(False)
            page.view_model.set_box(None)
        else:
            for spin, value in zip(
                (page.box_par_min, page.box_par_max, page.box_qz_min, page.box_qz_max),
                (q_parallel[0], q_parallel[1], qz[0], qz[1]),
            ):
                self._set(spin, float(value))
            self._set(page.box_check, True)
            page.box_grid.setEnabled(True)
            page.view_model.set_box(tuple(q_parallel), tuple(qz))
        self.reanalyse(done)

    def set_chi_window(self, low: float, high: float, done: Done) -> None:
        self._page.view_model.set_chi_window(float(low), float(high))
        self.reanalyse(done)

    def set_frame(self, frame_number: int, sum_count: int, done: Done) -> None:
        """Show frame ``frame_number`` (1-based) of a series, summing ``sum_count`` frames from it."""
        page = self._page
        path = page.view_model.current_path
        if path is None:
            done(False, "No frame is open in Analyze.")
            return
        count = max(1, int(page.view_model.frame_count(path)))
        number = min(max(1, int(frame_number)), count)
        self._open_options("data")
        with QSignalBlocker(page.frame_spin):
            page.frame_spin.setMaximum(max(count, page.frame_spin.maximum()))
            page.frame_spin.setValue(number)
        self._set(page.sum_spin, max(1, int(sum_count)))
        page.view_model.set_sum_count(int(sum_count))
        page.view_model.set_frame(number - 1)
        self.reanalyse(done)

    def set_valid_range(self, minimum: Optional[float], maximum: Optional[float], done: Done) -> None:
        page = self._page
        self._open_options("mask")
        self._set(page.minimum_check, minimum is not None)
        self._set(page.maximum_check, maximum is not None)
        if minimum is not None:
            self._set(page.minimum_spin, float(minimum))
        if maximum is not None:
            self._set(page.maximum_spin, float(maximum))
        page.view_model.set_valid_range(minimum, maximum)
        self.reanalyse(done)

    # -- GISAXS ------------------------------------------------------------------------

    def symmetry_center(self) -> dict:
        """The left–right symmetry axis of the horizontal cut, without changing anything."""
        analysis = self._page.view_model.state.analysis
        if analysis is None or analysis.geometry is None:
            raise ValueError("Open a frame with a geometry first.")
        result = refine_center_by_symmetry(analysis)
        return {
            "initial_x_px": float(result.initial_x_px), "x_px": float(result.x_px),
            "loss_before": float(result.loss_before), "loss_after": float(result.loss_after),
            "paired_samples": int(result.paired_samples), "search_px": float(result.search_px),
        }

    def set_beam_center(self, x_px: float, y_px: float, done: Done) -> None:
        """The beam centre for this session (canonical pixels); every frame is reduced with it."""
        page = self._page
        page.show_step("geometry")
        page.view_model.set_beam_center(float(x_px), float(y_px))
        self.reanalyse(done)

    def refine_center_symmetry(self, done: Done) -> None:
        """Move the beam-centre column to the left–right symmetry axis of the horizontal cut (session)."""
        page = self._page
        page.show_step("cuts")
        try:
            result = page.view_model.refine_center_x()
        except ValueError as exc:
            done(False, str(exc))
            return
        self._symmetry = {
            "initial_x_px": float(result.initial_x_px), "x_px": float(result.x_px),
            "loss_before": float(result.loss_before), "loss_after": float(result.loss_after),
            "paired_samples": int(result.paired_samples), "search_px": float(result.search_px),
        }
        text = f"centre x {result.initial_x_px:.2f} → {result.x_px:.2f} px"
        self.reanalyse(lambda ok, message: done(ok, text if ok else message))

    def set_halves(self, side: str, done: Done) -> None:
        """Which halves of I(qy) the curve for fitting uses: both_abs, mean, negative, positive."""
        page = self._page
        if side not in page.fit_side_actions:
            done(False, f"Unknown halves {side!r}.")
            return
        page.show_step("cuts")
        page.view_model.set_fit_side(side)
        page._sync_fit_side()
        page._sync_halves()
        done(True, f"halves: {side}")

    def set_gisaxs_cuts(
        self,
        horizontal_row: Optional[float],
        horizontal_half_height: Optional[float],
        vertical_column: Optional[float],
        vertical_half_width: Optional[float],
        done: Done,
    ) -> None:
        """Move the horizontal / vertical cut (``None`` keeps a value; all ``None`` = automatic cuts)."""
        page = self._page
        page.show_step("cuts")
        if all(value is None for value in (horizontal_row, horizontal_half_height, vertical_column, vertical_half_width)):
            page.view_model.reset_cuts()
        else:
            cuts = page.view_model.state.gisaxs
            analysis = page.view_model.state.analysis
            markers = analysis.reduction.markers if analysis is not None and analysis.reduction is not None else {}
            if horizontal_row is not None or horizontal_half_height is not None:
                row = horizontal_row if horizontal_row is not None else 0.5 * sum(markers.get("horizontal_band", (0, 0)))
                half = horizontal_half_height if horizontal_half_height is not None else cuts.horizontal_half_height_px
                page.view_model.set_horizontal_band(row - half, row + half)
            if vertical_column is not None or vertical_half_width is not None:
                column = vertical_column if vertical_column is not None else 0.5 * sum(markers.get("vertical_band", (0, 0)))
                half = vertical_half_width if vertical_half_width is not None else cuts.vertical_half_width_px
                page.view_model.set_vertical_band(column - half, column + half)
        self.reanalyse(done)

    def use_geometry(self, values: dict, name: Optional[str], source: str, done: Done) -> None:
        """Save ``values`` as the instrument profile of this frame's detector, then re-analyse."""
        page = self._page
        analysis = page.view_model.state.analysis
        if analysis is None:
            done(False, "No frame is open in Analyze.")
            return
        defaults = geometry_defaults(analysis.metadata or {}, analysis.shape)
        pixel_x = values.get("pixel_size_x_m") or defaults["pixel_x_um"] * 1e-6
        try:
            geometry = DetectorGeometry(
                pixel_size_x_m=float(pixel_x),
                pixel_size_y_m=float(values.get("pixel_size_y_m") or pixel_x),
                distance_m=float(values["distance_mm"]) * 1e-3,
                beam_center_x_px=float(values["beam_center_x_px"]),
                beam_center_y_px=float(values["beam_center_y_px"]),
                wavelength_angstrom=float(values["wavelength_angstrom"]),
                incidence_deg=float(values.get("incidence_deg") or 0.0),
            )
        except (KeyError, TypeError, ValueError) as exc:
            done(False, f"The geometry is not valid: {exc}")
            return
        profile_name = (name or "").strip() or page.view_model.suggested_profile_name()
        page.view_model.save_profile(profile_name, geometry, source=source)
        self._waiting.append(done)
        page._profile_saved(profile_name)

    # -- view and output ---------------------------------------------------------------

    def show(self, view: Optional[str] = None, lower_profile: Optional[str] = None) -> None:
        page = self._page
        if view in ("detector", "q_map"):
            index = 1 if view == "q_map" else 0
            if index == 0 or page.view_combo.isEnabled():
                with QSignalBlocker(page.view_combo):
                    page.view_combo.setCurrentIndex(index)
                page._view_chosen(index)
        if lower_profile:
            index = page.lower_choice.findData(lower_profile)
            if index >= 0:
                page._lower_profile = lower_profile
                with QSignalBlocker(page.lower_choice):
                    page.lower_choice.setCurrentIndex(index)
                page._show_lower_profile()

    def export_current(self) -> list[str]:
        written = self._page.view_model.export()
        self._page._status(f"Exported {len(written)} files to {written[0].parent}", "ok")
        return [str(path) for path in written]

    def preview_png(self, max_size: int = 900, rings: Sequence[dict] = ()) -> Optional[bytes]:
        """The q∥–qz map (log intensity) as a PNG, or ``None`` without a GIWAXS reduction.

        ``rings``: ``{"q": Å⁻¹, "shadowed": [(|χ| from, to), …], "label": str}`` drawn on the map —
        where the ring was measured, where it lies in a shadow and where it was not measured.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        analysis = self._page.view_model.state.analysis
        rsm = analysis.reduction.reciprocal_space_map if analysis is not None and analysis.reduction else None
        if rsm is None:
            return None
        image = np.asarray(rsm.image, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            shown = np.where(image > 0, np.log10(image), np.nan)
        finite = shown[np.isfinite(shown)]
        levels = np.percentile(finite, (1.0, 99.7)) if finite.size else (None, None)
        inches = max_size / 100.0
        figure = Figure(figsize=(inches, inches * 0.8), dpi=100, constrained_layout=True)
        FigureCanvasAgg(figure)
        axes = figure.add_subplot()
        (q0, q1), (z0, z1) = rsm.q_parallel_range, rsm.qz_range
        artist = axes.imshow(
            shown, extent=(q0, q1, z0, z1), origin="upper", aspect="auto",
            vmin=levels[0], vmax=levels[1], cmap="viridis",
        )
        figure.colorbar(artist, ax=axes, label="log₁₀ I")
        if rings:
            _draw_rings(axes, image, (q0, q1, z0, z1), rings)
        axes.set_xlabel("q∥ (Å⁻¹)")
        axes.set_ylabel("qz (Å⁻¹)")
        axes.set_title(analysis.path.name, fontsize=9)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png")
        return buffer.getvalue()

    def curve_png(self, key: str = "radial", markers: Sequence[dict] = (), max_size: int = 900) -> Optional[bytes]:
        """One reduced curve (log intensity, q in Å⁻¹) as a PNG, or ``None`` without it.

        ``markers``: ``{"q": Å⁻¹, "label": str, "reliable": bool}`` drawn as vertical lines.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        curve = self.curve(key)
        if curve is None:
            return None
        x, y = np.asarray(curve["x"], dtype=float), np.asarray(curve["y"], dtype=float)
        shown = np.isfinite(x) & np.isfinite(y) & (y > 0)
        if not shown.any():
            return None
        inches = max_size / 100.0
        figure = Figure(figsize=(inches, inches * 0.42), dpi=100, constrained_layout=True)
        FigureCanvasAgg(figure)
        axes = figure.add_subplot()
        axes.semilogy(x[shown], y[shown], color="#1f4e79", linewidth=1.1)
        for marker in markers:
            color, style = ("#c62828", "--") if marker.get("reliable", True) else ("#9e9e9e", ":")
            axes.axvline(float(marker["q"]), color=color, linestyle=style, linewidth=0.9)
            if marker.get("label"):
                axes.annotate(
                    str(marker["label"]), (float(marker["q"]), 0.98), xycoords=("data", "axes fraction"),
                    rotation=90, fontsize=7, va="top", ha="right", color=color,
                )
        axes.set_xlabel(curve["x_label"])
        axes.set_ylabel("I (counts / pixel)")
        axes.set_title(f"{curve['title']} — {self._page.view_model.state.analysis.path.name}", fontsize=9)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png")
        return buffer.getvalue()


RING_STATES = (
    ("measured", dict(color="white", linewidth=1.4, linestyle="-")),
    ("in a shadow", dict(color="#ff9f1c", linewidth=2.2, linestyle="-")),
    ("not measured", dict(color="#e63946", linewidth=1.2, linestyle=(0, (3, 2)))),
)


def _draw_rings(axes, image: np.ndarray, extent: tuple, rings: Sequence[dict]) -> None:
    """Each ring on the map, split by what its pixels are (χ from the surface normal, ±90° in plane)."""
    q0, q1, z0, z1 = extent
    rows, columns = image.shape
    chi = np.arange(-90.0, 90.001, 0.5)
    drawn = set()
    for ring in rings:
        q = float(ring["q"])
        x, z = q * np.sin(np.radians(chi)), q * np.cos(np.radians(chi))
        column = np.floor((x - q0) / max(q1 - q0, 1e-12) * columns).astype(int)
        row = np.floor((z1 - z) / max(z1 - z0, 1e-12) * rows).astype(int)  # row 0 is the top (largest qz)
        inside = (column >= 0) & (column < columns) & (row >= 0) & (row < rows)
        pixel = np.full(chi.shape, np.nan)
        pixel[inside] = image[row[inside], column[inside]]
        state = np.where(np.isfinite(pixel) & (pixel > 0), 0, 2)
        for low, high in ring.get("shadowed") or ():
            state[(state == 0) & (np.abs(chi) >= float(low)) & (np.abs(chi) <= float(high))] = 1
        state[~inside] = -1
        start = 0
        for index in range(1, chi.size + 1):
            if index < chi.size and state[index] == state[start]:
                continue
            if state[start] >= 0:
                name, style = RING_STATES[int(state[start])]
                segment = slice(start, min(index + 1, chi.size))
                axes.plot(x[segment], z[segment], label=None if name in drawn else name, **style)
                drawn.add(name)
            start = index
        if ring.get("label"):
            seen = np.flatnonzero(state == 0)  # label a measured part, near χ = 30° where possible
            at = int(seen[np.argmin(np.abs(chi[seen] - 30.0))]) if seen.size else int(np.argmin(np.abs(chi - 45.0)))
            axes.annotate(
                str(ring["label"]), (x[at], z[at]), color="white", fontsize=8, xytext=(4, 4),
                textcoords="offset points", bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.55, linewidth=0),
            )
    axes.set_xlim(q0, q1)
    axes.set_ylim(z0, z1)
    if drawn:
        axes.legend(loc="upper right", fontsize=7, facecolor="#303030", labelcolor="white", framealpha=0.8)


__all__ = ["AnalyzeAutomation"]
