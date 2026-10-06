"""Prediction Results coordination for Prediction."""

from __future__ import annotations


from typing import Dict, List, Optional, Tuple

import numpy as np

from PyQt5.QtCore import QSignalBlocker

from PyQt5.QtGui import QImage, QPixmap

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.theme import theme_manager

# Exported figures keep their light publication look in either theme (white figure, light grey
# axes, black text and spines, matplotlib's grey grid), as before the display was themed.
PUBLICATION_COLORS = {
    "surface": "#ffffff",
    "surface_alt": "#f0f0f0",
    "text": "#000000",
    "text_muted": "#000000",
    "divider": "#b0b0b0",
    "border": "#000000",
}


class PredictionResultsMixin:
    """Own prediction results presentation behavior."""

    def _collect_preprocess_steps(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[List[Dict[str, object]]]]:
        if image is None:
            return None, None
        typed_module = (
            self._current_module.get("_prediction_module")
            if isinstance(self._current_module, dict)
            else None
        )
        if typed_module is None:
            self._append_status_message(
                "Selected module has no typed prediction contract",
                level="ERROR",
            )
            return None, None
        prepared = self.prediction_view_model.prepare_input(image, typed_module)
        if prepared is None:
            return None, None
        return prepared.values, list(prepared.steps)

    def _display_prediction(self, outputs: Dict[str, np.ndarray]) -> None:
        if not outputs:
            return
        self.prediction_results = outputs
        self._refresh_predict_readiness()
        # 1) If only scalar outputs, print to status and return
        scal = outputs.get("scalars") if isinstance(outputs, dict) else None
        if isinstance(scal, np.ndarray):
            vals = ", ".join(f"{float(x):.4g}" for x in scal.reshape(-1))
            self._append_status_message(trf("Predicted scalars: [{values}]", values=vals), level="INFO")
            return
        panels: List[Dict[str, object]] = []
        params = outputs.get("parameters") if isinstance(outputs, dict) else None
        param_names = outputs.get("parameter_names") if isinstance(outputs, dict) else None
        if isinstance(params, np.ndarray):
            names = (
                [str(name) for name in param_names]
                if isinstance(param_names, list)
                else [f"p{i + 1}" for i in range(params.size)]
            )
            text = ", ".join(
                f"{name}={float(value):.6g}"
                for name, value in zip(names, np.asarray(params).reshape(-1))
            )
            self._append_status_message(trf("Predicted parameters: {values}", values=text), level="INFO")
            panels.append(
                {
                    "kind": "parameters",
                    "title": "Parameters",
                    "data": np.asarray(params, dtype=np.float32).reshape(-1),
                    "names": names,
                }
            )

        # Use the snapshots captured by the exact inference call. Re-running
        # preprocessing here can drift from the model input when the source or
        # module changes while prediction is completing.
        try:
            if self._current_image is not None:
                if getattr(
                    self, "_latest_preprocess_source", None
                ) is self._current_image and isinstance(
                    getattr(self, "_latest_model_input", None), np.ndarray
                ):
                    pre_img = self._latest_model_input
                    pre_steps = list(getattr(self, "_latest_preprocess_steps", ()))
                else:
                    pre_img, pre_steps = self._collect_preprocess_steps(self._current_image)
                if pre_steps:
                    display_steps = list(pre_steps)
                    if isinstance(pre_img, np.ndarray):
                        masked_value = next(
                            (
                                step.get("masked_value")
                                for step in reversed(display_steps)
                                if isinstance(step, dict) and "masked_value" in step
                            ),
                            None,
                        )
                        final_input = np.squeeze(pre_img)
                        if (
                            isinstance(final_input, np.ndarray)
                            and final_input.ndim == 3
                            and final_input.shape[-1] >= 2
                        ):
                            display_steps = [
                                {
                                    "step": "Final Input: intensity",
                                    "label": "Final Input: intensity",
                                    "image": final_input[..., 0],
                                    "masked_value": masked_value,
                                },
                                {
                                    "step": "Final Input: mask channel",
                                    "label": "Final Input: mask channel",
                                    "image": final_input[..., 1],
                                },
                            ] + display_steps
                        elif isinstance(final_input, np.ndarray) and final_input.ndim == 2:
                            display_steps = [
                                {
                                    "step": "Final Input",
                                    "label": "Final Input",
                                    "image": final_input,
                                    "masked_value": masked_value,
                                }
                            ] + display_steps
                    panels.append(
                        {
                            "kind": "steps",
                            "title": "Preprocessed",
                            "steps": display_steps,
                            "default_index": 0,
                        }
                    )
                elif isinstance(pre_img, np.ndarray):
                    pre_img2d = np.squeeze(pre_img)
                    if isinstance(pre_img2d, np.ndarray) and pre_img2d.ndim == 2:
                        panels.append(
                            {
                                "kind": "array",
                                "title": "Preprocessed",
                                "data": pre_img2d,
                                "colormap": self.current_parameters.get(
                                    "colormap", self._DEFAULT_COLORMAPS[0]
                                ),
                            }
                        )
        except Exception as exc:
            self._append_status_message(trf("Preprocessed panel failed: {error}", error=exc), level="ERROR")

        raw_axes = outputs.get("distribution_axes") if isinstance(outputs, dict) else None
        distribution_axes = raw_axes if isinstance(raw_axes, dict) else {}

        # Joint distribution panel
        hr = outputs.get("hr") if isinstance(outputs, dict) else None
        if isinstance(hr, np.ndarray) and hr.ndim == 2:
            column = distribution_axes.get("column")
            row = distribution_axes.get("row")
            column_label = (
                str(column.get("label"))
                if isinstance(column, dict) and column.get("label")
                else "h"
            )
            row_label = str(row.get("label")) if isinstance(row, dict) and row.get("label") else "R"
            panels.append(
                {
                    "kind": "hr",
                    "title": f"p({column_label}, {row_label})",
                    "data": hr,
                    "axes": distribution_axes,
                }
            )

        # Physical 1D marginals in matrix column/row order.
        for dimension, fallback_key, fallback_label in (
            ("column", "h", "h"),
            ("row", "r", "R"),
        ):
            axis = distribution_axes.get(dimension)
            axis_values = axis if isinstance(axis, dict) else {}
            key = str(axis_values.get("key") or fallback_key)
            label = str(axis_values.get("label") or fallback_label)
            unit = str(axis_values.get("unit") or "nm")
            curve = outputs.get(key) if isinstance(outputs, dict) else None
            curve_x = outputs.get(f"{key}_x") if isinstance(outputs, dict) else None
            if isinstance(curve, np.ndarray):
                panels.append(
                    {
                        "kind": "curve",
                        "title": f"p({label})",
                        "xlabel": f"{label} ({unit})" if unit else label,
                        "data": curve,
                        "x": curve_x,
                    }
                )

        if not panels:
            self._append_status_message("No plottable prediction outputs", level="WARN")
            return

        self._predict_tab_specs = panels
        tabs = self._get_or_create_predict2d_tabs()
        if tabs is not None:
            self._rebuild_predict_tabs(tabs)
            if hasattr(tabs, "setTabBarAutoHide"):
                tabs.setTabBarAutoHide(len(panels) <= 1)
            hr_index = next(
                (
                    idx
                    for idx, spec in enumerate(self._predict_tab_specs)
                    if spec.get("kind") == "hr"
                ),
                None,
            )
            target_index = (
                hr_index
                if hr_index is not None
                else (tabs.currentIndex() if tabs.currentIndex() >= 0 else 0)
            )
            if target_index is not None:
                blocker = QSignalBlocker(tabs)
                tabs.setCurrentIndex(target_index)
                del blocker
            self._render_predict_tab_by_index(target_index if target_index is not None else 0)
            # Ensure the outer tab switches to Predict-2D when results are ready
            self._set_predict_main_tab("result")
            self._refresh_predict_readiness()
        else:
            self._render_predict_tab_by_index(0)

    def _render_predict2d_into_view(
        self,
        image2d: np.ndarray,
        *,
        axes: Optional[Dict[str, object]] = None,
    ) -> None:
        try:
            self._predict_current_image = image2d
            disp, vmin, vmax = self._prepare_predict_image(image2d)
            target_pixels = self._predict_viewport_pixels()
            pix = self._render_hr_figure(
                disp,
                vmin=vmin,
                vmax=vmax,
                target_pixels=target_pixels,
                axes=axes,
            )
            if pix is None:
                pix = self._create_pixmap_from_array(
                    disp,
                    vmin,
                    vmax,
                    self.current_parameters.get("colormap", self._DEFAULT_COLORMAPS[0]),
                )
            if pix is None:
                return
            self._show_pixmap_in_predict_view(pix)
            self._append_status_message("Predict-2D image updated.")
        except Exception as exc:
            self._append_status_message(trf("Predict-2D draw failed: {error}", error=exc), level="ERROR")

    def _prepare_predict_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        disp = self._maybe_log_scale(
            np.array(image, dtype=np.float32),
            bool(self.current_parameters.get("predict_log_scale", False)),
        )
        auto = bool(self.current_parameters.get("predict_auto_scale", True))
        vmin = self.current_parameters.get("predict_vmin")
        vmax = self.current_parameters.get("predict_vmax")

        if auto or vmin is None or vmax is None:
            vmin, vmax = self._auto_scale_percentiles(disp, 0, 100)
            self.current_parameters["predict_vmin"] = vmin
            self.current_parameters["predict_vmax"] = vmax

        self._ui_updating = True
        try:
            self._set_checkbox("predict2dAutoScaleCheckBox", auto)
            self._set_double_spin("predict2dVminValue", vmin)
            self._set_double_spin("predict2dVmaxValue", vmax)
        finally:
            self._ui_updating = False
        self._persist_parameters()
        return disp, float(vmin), float(vmax)

    def _render_hr_figure(
        self,
        image: np.ndarray,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        target_pixels: Optional[Tuple[int, int]] = None,
        axes: Optional[Dict[str, object]] = None,
    ) -> Optional[QPixmap]:
        arguments = {
            "image": image,
            "vmin": vmin,
            "vmax": vmax,
            "target_pixels": target_pixels,
            "axes": axes,
        }
        return self._render_shown_figure("hr", arguments, "HR figure render error")

    def _render_shown_figure(self, kind: str, arguments: Dict[str, object], failure: str):
        """Render a figure for display (theme colours) and remember how, for the image export."""
        self._shown_figure = None
        try:
            build = self._hr_figure if kind == "hr" else self._curve_figure
            pixmap = _figure_pixmap(build(**arguments))
        except Exception as exc:
            self._append_status_message(trf("{failure}: {error}", failure=tr(failure), error=exc), level="ERROR")
            return None
        self._shown_figure = (kind, arguments)
        return pixmap

    def _publication_pixmap(self) -> Optional[QPixmap]:
        """The shown distribution or curve again, same size, in PUBLICATION_COLORS (for export)."""
        shown = getattr(self, "_shown_figure", None)
        if not shown or shown[0] != getattr(self, "_predict_current_kind", None):
            return None
        kind, arguments = shown
        try:
            build = self._hr_figure if kind == "hr" else self._curve_figure
            return _figure_pixmap(build(**arguments, colors=PUBLICATION_COLORS))
        except Exception as exc:
            self._append_status_message(trf("Export figure render error: {error}", error=exc), level="ERROR")
            return None

    def _hr_figure(
        self,
        image: np.ndarray,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        target_pixels: Optional[Tuple[int, int]] = None,
        axes: Optional[Dict[str, object]] = None,
        colors: Optional[Dict[str, str]] = None,
    ):
        """The joint distribution with its marginals, in theme colours unless ``colors`` is given."""
        from matplotlib.figure import Figure  # type: ignore

        img = np.array(image, dtype=np.float32)
        column_sum = np.sum(img, axis=0)
        row_sum = np.sum(img, axis=1)
        vmin_calc, vmax_calc = self._auto_scale_values(img)
        cmin = vmin if vmin is not None else vmin_calc
        cmax = vmax if vmax is not None else vmax_calc

        def axis_values(
            dimension: str,
            fallback_label: str,
            count: int,
        ) -> Tuple[str, str, float, float, np.ndarray]:
            raw = axes.get(dimension) if isinstance(axes, dict) else None
            values = raw if isinstance(raw, dict) else {}
            label = str(values.get("label") or fallback_label)
            unit = str(values.get("unit") or "nm")
            low = values.get("min")
            high = values.get("max")
            minimum = float(low) if isinstance(low, (int, float)) else 0.05
            maximum = float(high) if isinstance(high, (int, float)) else 15.0
            if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
                minimum, maximum = 0.05, 15.0
            bins = np.linspace(minimum, maximum, count + 1)
            centers = (bins[:-1] + bins[1:]) / 2.0
            return label, unit, minimum, maximum, centers

        row_label, row_unit, row_min, row_max, row_centers = axis_values("row", "R", img.shape[0])
        column_label, column_unit, column_min, column_max, column_centers = axis_values(
            "column", "h", img.shape[1]
        )

        dpi = 120.0
        if target_pixels:
            fig_w = max(4.0, target_pixels[0] / dpi)
            fig_h = max(4.0, target_pixels[1] / dpi)
        else:
            fig_w = fig_h = 10.0
        colors = colors or _theme_colors()
        fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=colors["surface"])
        ax = fig.subplots(2, 2, gridspec_kw={"width_ratios": [4, 1], "height_ratios": [1, 4]})
        # Room on the right for the colour bar, its ticks and its label (also at 480 px wide).
        fig.subplots_adjust(right=0.83)
        scale = max(0.6, min(2.0, fig_w / 10.0))
        title_size = 14 * scale
        tick_size = 12 * scale
        cbar_label_size = 14 * scale
        cbar_tick_size = 12 * scale

        cmap_name = self.current_parameters.get("colormap", self._DEFAULT_COLORMAPS[0])
        im = ax[1, 0].imshow(
            img,
            cmap=cmap_name,
            vmin=cmin,
            vmax=cmax,
            origin="upper",
            aspect="auto",
            extent=(column_min, column_max, row_max, row_min),
        )
        ax[1, 0].set_xlabel(f"{column_label} ({column_unit})" if column_unit else column_label)
        ax[1, 0].set_ylabel(f"{row_label} ({row_unit})" if row_unit else row_label)
        _style_axes(ax[1, 0], colors, tick_size, grid=False)

        ax[0, 0].plot(column_centers, column_sum, color="red", linewidth=2)
        ax[0, 0].set_title(f"p({column_label})", fontsize=title_size, fontweight="bold")
        ax[0, 0].set_xlim(column_min, column_max)
        _style_axes(ax[0, 0], colors, tick_size)

        ax[1, 1].plot(row_sum, row_centers, color="red", linewidth=2)
        ax[1, 1].set_title(f"p({row_label})", fontsize=title_size, fontweight="bold")
        _style_axes(ax[1, 1], colors, tick_size)
        ax[1, 1].set_ylim(row_max, row_min)

        ax[0, 1].axis("off")

        cax = fig.add_axes([0.855, 0.11, 0.025, 0.56])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Probability", fontsize=cbar_label_size, fontweight="bold", color=colors["text"])
        cbar.ax.tick_params(labelsize=cbar_tick_size, colors=colors["text_muted"])
        cbar.outline.set_edgecolor(colors["border"])
        cbar.ax.yaxis.get_offset_text().set_color(colors["text_muted"])
        return fig

    def _render_curve_figure(
        self,
        curve: np.ndarray,
        x_label: str,
        title: str,
        x: Optional[np.ndarray] = None,
        log_x: bool = False,
        log_y: bool = False,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ) -> Optional[QPixmap]:
        arguments = {
            "curve": curve,
            "x_label": x_label,
            "title": title,
            "x": x,
            "log_x": log_x,
            "log_y": log_y,
            "xlim": xlim,
            "ylim": ylim,
        }
        return self._render_shown_figure("curve", arguments, "Curve figure render error")

    def _curve_figure(
        self,
        curve: np.ndarray,
        x_label: str,
        title: str,
        x: Optional[np.ndarray] = None,
        log_x: bool = False,
        log_y: bool = False,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        colors: Optional[Dict[str, str]] = None,
    ):
        from matplotlib.figure import Figure  # type: ignore

        y = np.array(curve, dtype=np.float32)
        x_values = np.asarray(x, dtype=np.float32) if isinstance(x, np.ndarray) else None
        if x_values is None or x_values.shape != y.shape:
            x_values = np.arange(len(y), dtype=np.float32)
            if log_x:
                x_values = np.arange(1, len(y) + 1, dtype=np.float32)

        y_plot = y.copy()
        if log_y:
            y_plot = np.where(y_plot > 0, y_plot, np.nan)

        colors = colors or _theme_colors()
        fig = Figure(figsize=(8, 4), facecolor=colors["surface"])
        ax = fig.subplots()
        ax.plot(x_values, y_plot, color="red", linewidth=2)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(x_label)
        _style_axes(ax, colors, 12)

        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        if xlim:
            low, high = xlim
            if log_x and low <= 0:
                low = max(low, 1e-6)
            if log_x and high <= 0:
                high = max(high, low + 1e-6)
            ax.set_xlim(low, high)
        if ylim:
            low, high = ylim
            if log_y and low <= 0:
                low = max(low, 1e-6)
            if log_y and high <= 0:
                high = max(high, low + 1e-6)
            ax.set_ylim(low, high)
        return fig

    def _on_theme_changed(self, _mode: str = "") -> None:
        """Re-render a shown distribution or curve so the figure follows the theme (display only)."""
        try:
            specs = getattr(self, "_predict_tab_specs", None) or []
            if not specs or not self.prediction_results:
                return
            tabs = getattr(self, "_predict_tabs", None)
            index = tabs.currentIndex() if tabs is not None and tabs.currentIndex() >= 0 else 0
            if index < len(specs) and specs[index].get("kind") in ("hr", "curve"):
                self._render_predict_tab_by_index(index)
        except RuntimeError:
            pass  # the page was already closed


def _theme_colors() -> Dict[str, str]:
    manager = theme_manager()
    return {
        name: manager.color(name).name()
        for name in ("surface", "surface_alt", "text", "text_muted", "divider", "border")
    }


def _style_axes(ax, colors: Dict[str, str], tick_size: float, *, grid: bool = True) -> None:
    """Axes, ticks, labels and grid in theme colours (the data colours stay as they are)."""
    ax.set_facecolor(colors["surface_alt"])
    if grid:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, color=colors["divider"])
    ax.tick_params(axis="both", which="both", labelsize=tick_size, colors=colors["text_muted"])
    for spine in ax.spines.values():
        spine.set_edgecolor(colors["border"])
    ax.xaxis.label.set_color(colors["text"])
    ax.yaxis.label.set_color(colors["text"])
    ax.title.set_color(colors["text"])


def _figure_pixmap(fig) -> QPixmap:
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # type: ignore

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_rgba = np.asarray(canvas.buffer_rgba())
    height, width = img_rgba.shape[:2]
    image_q = QImage(img_rgba.data, width, height, img_rgba.strides[0], QImage.Format_RGBA8888)
    return QPixmap.fromImage(image_q.copy())
