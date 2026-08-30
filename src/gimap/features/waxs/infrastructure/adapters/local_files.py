"""Local WAXS file catalog and export adapter。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...domain import compute_q_maps, estimate_display_limits, prepare_display_array


SUPPORTED_EXTENSIONS = {".nxs", ".tif", ".tiff"}


class LocalWaxsFileCatalog:
    def discover(self, folder: Path, pattern: str) -> tuple[Path, ...]:
        return tuple(
            path
            for path in sorted(Path(folder).glob(pattern))
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )


class LocalWaxsExportAdapter:
    _AXIS_LABEL_SIZE = 16
    _TICK_LABEL_SIZE = 13
    _TITLE_SIZE = 15

    def export_curve(self, path: Path, x: np.ndarray, y: np.ndarray) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        values = np.column_stack([x, y])
        np.savetxt(
            target,
            values,
            delimiter=",",
            header="x,intensity",
            comments="",
            fmt="%.9g",
        )

    def export_matrix(self, path, columns, headers) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        max_length = max((len(column) for column in columns), default=0)
        padded = [
            np.pad(
                np.asarray(column, dtype=float).ravel(),
                (0, max_length - len(column)),
                constant_values=np.nan,
            )
            for column in columns
        ]
        matrix = np.column_stack(padded) if padded else np.empty((0, 0))
        np.savetxt(
            target,
            matrix,
            delimiter=",",
            header=",".join(headers),
            comments="",
            fmt="%.9g",
        )

    def export_image(self, path: Path, image: np.ndarray, display: dict) -> None:
        from matplotlib import colormaps
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        vmin = float(display.get("vmin", 0.0))
        vmax = float(display.get("vmax", 1.0))
        log_scale = bool(display.get("log_scale", False))
        mask_min = float(display.get("mask_min", -1e12))
        mask_max = float(display.get("mask_max", 1e12))
        if display.get("auto_scale", True):
            limits = estimate_display_limits(
                image,
                log_scale=log_scale,
                mask_min=mask_min,
                mask_max=mask_max,
            )
            if limits is not None:
                vmin, vmax = limits
        values = prepare_display_array(
            image,
            log_scale=log_scale,
            mask_min=mask_min,
            mask_max=mask_max,
            flip_vertical=False,
        )
        figure = Figure(figsize=(7.2, 6.2), constrained_layout=True)
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        colormap = colormaps.get_cmap(
            str(display.get("colormap", "viridis"))
        ).copy()
        # Match the interactive q-space preview: masked/no-data cells use the
        # figure background rather than looking like low-intensity measurements.
        no_data_color = str(display.get("no_data_color", "white"))
        colormap.set_bad(no_data_color)
        axis.set_facecolor(no_data_color)
        if display.get("coordinate_mode") == "q":
            geometry = dict(display["geometry"])
            qr, qz = compute_q_maps(values.shape, geometry)
            artists = []
            for branch in self._signed_q_branch_slices(qr):
                artists.append(
                    axis.pcolormesh(
                        qr[:, branch],
                        qz[:, branch],
                        values[:, branch],
                        shading="nearest",
                        cmap=colormap,
                        vmin=vmin,
                        vmax=vmax,
                        rasterized=True,
                    )
                )
            artist = artists[0]
            axis.set_xlabel(r"$q_r$ ($\AA^{-1}$)")
            axis.set_ylabel(r"$q_z$ ($\AA^{-1}$)")
            self._apply_q_range(axis, display.get("q_range"))
        else:
            artist = axis.imshow(
                values,
                origin="upper",
                cmap=colormap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )
            axis.set_xlabel("X (pixel)")
            axis.set_ylabel("Y (pixel)")
        axis.set_aspect("equal", adjustable="box")
        title = str(display.get("title", "")).strip()
        if title:
            axis.set_title(title, fontsize=self._TITLE_SIZE, pad=10)
        self._style_axis(axis)
        colorbar = figure.colorbar(artist, ax=axis, pad=0.025, fraction=0.05)
        colorbar.set_label(
            r"$\log_{10}$ Intensity (a.u.)" if log_scale else "Intensity (a.u.)",
            fontsize=self._AXIS_LABEL_SIZE,
        )
        colorbar.ax.tick_params(labelsize=self._TICK_LABEL_SIZE, width=1.1)
        figure.savefig(target, dpi=300, bbox_inches="tight", facecolor="white")

    def export_curve_image(
        self, path: Path, x: np.ndarray, y: np.ndarray, display: dict
    ) -> None:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure = Figure(figsize=(7.2, 5.2), constrained_layout=True)
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        axis.plot(
            np.asarray(x),
            np.asarray(y),
            color=str(display.get("curve_color", "#1f5aa6")),
            linewidth=1.8,
        )
        axis.set_xlabel(str(display.get("x_label", "q (Å⁻¹)")))
        axis.set_ylabel("Intensity (a.u.)")
        if display.get("curve_log_scale", False):
            axis.set_yscale("log")
        title = str(display.get("title", "")).strip()
        if title:
            axis.set_title(title, fontsize=self._TITLE_SIZE, pad=10)
        axis.grid(True, color="#d5d9df", linewidth=0.65, alpha=0.65)
        self._style_axis(axis)
        figure.savefig(target, dpi=300, bbox_inches="tight", facecolor="white")

    @classmethod
    def _style_axis(cls, axis) -> None:
        axis.xaxis.label.set_size(cls._AXIS_LABEL_SIZE)
        axis.yaxis.label.set_size(cls._AXIS_LABEL_SIZE)
        axis.tick_params(
            axis="both",
            which="major",
            labelsize=cls._TICK_LABEL_SIZE,
            width=1.1,
            length=5,
            direction="out",
        )
        for spine in axis.spines.values():
            spine.set_linewidth(1.1)

    @staticmethod
    def _signed_q_branch_slices(horizontal_q: np.ndarray) -> tuple[slice, ...]:
        columns = np.nanmedian(np.asarray(horizontal_q, dtype=float), axis=0)
        negative = np.flatnonzero(columns < 0.0)
        positive = np.flatnonzero(columns > 0.0)
        branches: list[slice] = []
        if negative.size >= 2:
            branches.append(slice(int(negative[0]), int(negative[-1]) + 1))
        if positive.size >= 2:
            branches.append(slice(int(positive[0]), int(positive[-1]) + 1))
        return tuple(branches) or (slice(0, horizontal_q.shape[1]),)

    @staticmethod
    def _apply_q_range(axis, q_range) -> None:
        if not q_range:
            return
        qr_min = q_range.get("qr_min")
        qr_max = q_range.get("qr_max")
        qz_min = q_range.get("qz_min")
        qz_max = q_range.get("qz_max")
        if qr_min is not None and qr_max is not None and qr_min < qr_max:
            axis.set_xlim(float(qr_min), float(qr_max))
        if qz_min is not None and qz_max is not None and qz_min < qz_max:
            axis.set_ylim(float(qz_min), float(qz_max))
