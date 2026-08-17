"""Local WAXS file catalog and export adapter。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...domain import estimate_display_limits, prepare_display_array


SUPPORTED_EXTENSIONS = {".nxs", ".tif", ".tiff"}


class LocalWaxsFileCatalog:
    def discover(self, folder: Path, pattern: str) -> tuple[Path, ...]:
        return tuple(
            path
            for path in sorted(Path(folder).glob(pattern))
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )


class LocalWaxsExportAdapter:
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
        figure = Figure()
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        colormap = colormaps.get_cmap(
            str(display.get("colormap", "viridis"))
        ).copy()
        colormap.set_bad(colormap(0.0))
        artist = axis.imshow(
            values,
            origin="upper",
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        figure.colorbar(artist, ax=axis)
        axis.set_xlabel("X (pixel)")
        axis.set_ylabel("Y (pixel)")
        figure.savefig(target, dpi=300, bbox_inches="tight")
