"""ROI/cut 的纯二维数组运算。"""

from __future__ import annotations

import numpy as np

from .image_transforms import finite_mean_axis
from .models import CutOrientation, CutSelection


def pixel_region_bounds(image_shape, selection: CutSelection) -> tuple[int, int, int, int]:
    """返回 legacy origin='lower' 语义下的 array bounds，均为闭区间。"""
    image_height, image_width = image_shape
    x_min = max(0, int(selection.center_x - selection.width / 2))
    x_max = min(image_width - 1, int(selection.center_x + selection.width / 2))
    y_min = max(0, int(selection.center_y - selection.height / 2))
    y_max = min(image_height - 1, int(selection.center_y + selection.height / 2))
    row_min = max(0, image_height - 1 - y_max)
    row_max = min(image_height - 1, image_height - 1 - y_min)
    return x_min, x_max, row_min, row_max


def extract_pixel_profile(image, selection: CutSelection):
    """提取普通 fitting pixel cut；无效 pixel 使用有限值均值。"""
    data = np.asarray(image, dtype=float)
    if data.ndim != 2:
        raise ValueError("Pixel cut expects a 2D image")
    x_min, x_max, row_min, row_max = pixel_region_bounds(data.shape, selection)
    region = data[row_min : row_max + 1, x_min : x_max + 1]
    if region.size == 0:
        raise ValueError("Empty region selected")
    if selection.orientation == "horizontal":
        return finite_mean_axis(region, axis=0), np.arange(x_min, x_max + 1)
    return finite_mean_axis(region, axis=1), np.arange(row_min, row_max + 1)


def extract_q_profile(image, horizontal_q_mesh, qz_mesh, selection: CutSelection):
    """从 qy/qz 或 signed-qr/qz 矩形 region 提取 fitting profile。"""
    data = np.asarray(image, dtype=float)
    qy = np.asarray(horizontal_q_mesh, dtype=float)
    qz = np.asarray(qz_mesh, dtype=float)
    if data.ndim != 2 or qy.shape != data.shape or qz.shape != data.shape:
        raise ValueError("Image and q meshgrids must have the same 2D shape")

    qy_min = selection.center_x - selection.width / 2
    qy_max = selection.center_x + selection.width / 2
    qz_min = selection.center_y - selection.height / 2
    qz_max = selection.center_y + selection.height / 2
    region = (qy >= qy_min) & (qy <= qy_max) & (qz >= qz_min) & (qz <= qz_max)
    finite = region & np.isfinite(data) & np.isfinite(qy) & np.isfinite(qz)

    if selection.orientation == "horizontal":
        indices = np.where(np.any(finite, axis=0))[0]
        if indices.size == 0:
            raise ValueError("No valid data in the selected region")
        intensity = np.asarray([np.mean(data[finite[:, col], col]) for col in indices])
        q_line = np.asarray([np.mean(qy[finite[:, col], col]) for col in indices])
    else:
        indices = np.where(np.any(finite, axis=1))[0]
        if indices.size == 0:
            raise ValueError("No valid data in the selected region")
        intensity = np.asarray([np.mean(data[row, finite[row, :]]) for row in indices])
        q_line = np.asarray([np.mean(qz[row, finite[row, :]]) for row in indices])
    return intensity, q_line, indices


def sample_q_mesh_line(
    mesh,
    pixel_coords,
    *,
    orientation: CutOrientation,
    image_shape,
):
    """沿 detector 中央行/列对 fractional pixel 坐标做 legacy 线性插值。"""
    image_height, image_width = image_shape
    values = np.asarray(mesh, dtype=float)
    coords = np.asarray(pixel_coords, dtype=float)
    if values.shape != (image_height, image_width):
        raise ValueError("Q mesh shape does not match the detector image")
    if orientation == "horizontal":
        row = int(np.clip(round(image_height / 2.0), 0, image_height - 1))
        clipped = np.clip(coords, 0.0, image_width - 1.0)
        return np.interp(clipped, np.arange(image_width, dtype=float), values[row, :])
    column = int(np.clip(round(image_width / 2.0), 0, image_width - 1))
    clipped = np.clip(coords, 0.0, image_height - 1.0)
    return np.interp(clipped, np.arange(image_height, dtype=float), values[:, column])
