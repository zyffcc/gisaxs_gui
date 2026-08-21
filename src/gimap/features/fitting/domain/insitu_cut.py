"""Legacy-compatible pure in-situ cut calculation."""

from __future__ import annotations

import numpy as np

from .curve_transformations import interpolate_series


def _sort_filter_pairs(x_values, y_values):
    x_array = np.asarray(x_values, dtype=float).reshape(-1)
    y_array = np.asarray(y_values, dtype=float).reshape(-1)
    count = min(x_array.size, y_array.size)
    x_array, y_array = x_array[:count], y_array[:count]
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    x_array, y_array = x_array[finite], y_array[finite]
    if x_array.size == 0:
        raise RuntimeError("No finite cut data points")
    order = np.argsort(x_array, kind="mergesort")
    return x_array[order], y_array[order]


def compute_insitu_cut(payload: dict) -> dict:
    """Preserve the former worker math, including ordinary ``np.mean`` behavior."""

    data = np.asarray(payload.get("image_data"), dtype=float)
    if data.ndim != 2:
        raise RuntimeError("In-situ cut expects a 2D detector image")

    vertical = float(payload.get("vertical", 0.0))
    parallel = float(payload.get("parallel", 0.0))
    center_x = float(payload.get("center_x", 0.0))
    center_y = float(payload.get("center_y", 0.0))
    cut_type = str(
        payload.get("cut_type")
        or ("horizontal" if vertical <= parallel else "vertical")
    )
    show_q_axis = bool(payload.get("show_q_axis", False))
    horizontal_q_axis = "qr" if payload.get("horizontal_q_axis") == "qr" else "qy"
    point_count = max(10, int(payload.get("n_points", 300)))
    method = str(payload.get("method", "Linear"))

    if show_q_axis:
        qy_mesh = np.asarray(payload.get("qy_mesh"), dtype=float)
        qz_mesh = np.asarray(payload.get("qz_mesh"), dtype=float)
        if qy_mesh.shape != data.shape or qz_mesh.shape != data.shape:
            raise RuntimeError(
                "Q-space meshgrids are unavailable or not aligned with image"
            )
        qy_min = center_x - parallel / 2.0
        qy_max = center_x + parallel / 2.0
        qz_min = center_y - vertical / 2.0
        qz_max = center_y + vertical / 2.0
        region = (
            (qy_mesh >= qy_min)
            & (qy_mesh <= qy_max)
            & (qz_mesh >= qz_min)
            & (qz_mesh <= qz_max)
        )
        finite = (
            region
            & np.isfinite(data)
            & np.isfinite(qy_mesh)
            & np.isfinite(qz_mesh)
        )
        if cut_type == "horizontal":
            indices = np.where(np.any(finite, axis=0))[0]
            if indices.size == 0:
                raise RuntimeError("No valid data in the selected region")
            intensity = np.array(
                [np.mean(data[finite[:, column], column]) for column in indices],
                dtype=float,
            )
            x_line = np.array(
                [np.mean(qy_mesh[finite[:, column], column]) for column in indices],
                dtype=float,
            )
            x_label = (
                r"$q_r$ (nm$^{-1}$)"
                if horizontal_q_axis == "qr"
                else r"$q_y$ (nm$^{-1}$)"
            )
            title = "Horizontal Cut"
        else:
            indices = np.where(np.any(finite, axis=1))[0]
            if indices.size == 0:
                raise RuntimeError("No valid data in the selected region")
            intensity = np.array(
                [np.mean(data[row, finite[row, :]]) for row in indices],
                dtype=float,
            )
            x_line = np.array(
                [np.mean(qz_mesh[row, finite[row, :]]) for row in indices],
                dtype=float,
            )
            x_label = r"$q_z$ (nm$^{-1}$)"
            title = "Vertical Cut"
        x_values, y_values = _sort_filter_pairs(x_line, intensity)
        x_result = np.linspace(
            float(x_values.min()),
            float(x_values.max()),
            point_count,
        )
        y_result = interpolate_series(x_values, y_values, x_result, method)
        source = "q"
    else:
        image_height, image_width = data.shape
        x_min = max(0, int(center_x - parallel / 2.0))
        x_max = min(image_width - 1, int(center_x + parallel / 2.0))
        y_min = max(0, int(center_y - vertical / 2.0))
        y_max = min(image_height - 1, int(center_y + vertical / 2.0))
        row_min = max(0, image_height - 1 - y_max)
        row_max = min(image_height - 1, image_height - 1 - y_min)
        region = data[row_min : row_max + 1, x_min : x_max + 1]
        if region.size == 0:
            raise RuntimeError("Empty region selected")
        if cut_type == "horizontal":
            intensity = np.mean(region, axis=0)
            coordinates = np.arange(x_min, x_max + 1, dtype=float)
            title = "Horizontal Cut"
            x_label = "Pixel / qy"
        else:
            intensity = np.mean(region, axis=1)
            coordinates = np.arange(row_min, row_max + 1, dtype=float)
            title = "Vertical Cut"
            x_label = "Pixel / qz"
        x_values, y_values = _sort_filter_pairs(coordinates, intensity)
        if x_values.size < 2:
            x_result, y_result = x_values, y_values
        else:
            x_result = np.linspace(
                float(x_values.min()),
                float(x_values.max()),
                point_count,
            )
            y_result = interpolate_series(x_values, y_values, x_result, method)
        source = "pixel"

    return {
        "x_coords": x_result,
        "y_intensity": y_result,
        "x_label": x_label,
        "title": title,
        "source": source,
        "cut_type": cut_type,
        "points": int(len(x_result)),
        "method": method,
    }
