"""Shared Matplotlib rendering primitive for signed fitting cut coordinates."""

from __future__ import annotations

import numpy as np


def plot_cut_data_with_log_handling(
    axes,
    x_coords,
    y_intensity,
    is_log_x,
    markersize=4,
    linewidth=1.5,
):
    """Render signed coordinates with the historical log-axis behavior."""

    try:
        x_array = np.array(x_coords)
        y_array = np.array(y_intensity)

        if is_log_x:
            positive_mask = x_array > 0
            x_positive = x_array[positive_mask]
            y_positive = y_array[positive_mask]
            negative_mask = x_array < 0
            x_negative_abs = np.abs(x_array[negative_mask])
            y_negative = y_array[negative_mask]
            zero_mask = x_array == 0
            x_zero = x_array[zero_mask]
            y_zero = y_array[zero_mask]

            if len(x_positive) > 0:
                axes.plot(
                    x_positive,
                    y_positive,
                    "bo-",
                    markersize=markersize,
                    linewidth=linewidth,
                    markerfacecolor="lightblue",
                    alpha=0.8,
                    label="Positive coordinates",
                )
            if len(x_negative_abs) > 0:
                axes.plot(
                    x_negative_abs,
                    y_negative,
                    "ro--",
                    markersize=markersize,
                    linewidth=linewidth,
                    markerfacecolor="lightcoral",
                    alpha=0.8,
                    label="Negative coordinates (|x|)",
                )
            if len(x_zero) > 0:
                min_positive = min(
                    np.min(x_positive) if len(x_positive) > 0 else 1e-6,
                    np.min(x_negative_abs) if len(x_negative_abs) > 0 else 1e-6,
                )
                axes.plot(
                    np.full_like(x_zero, min_positive * 0.1),
                    y_zero,
                    "go^",
                    markersize=markersize + 2,
                    markerfacecolor="lightgreen",
                    alpha=0.8,
                    label="Zero coordinates (approximated)",
                )
            axes.legend(loc="best", fontsize=max(8, markersize * 2))
        else:
            axes.plot(
                x_array,
                y_array,
                "bo-",
                markersize=markersize,
                linewidth=linewidth,
                markerfacecolor="lightblue",
                alpha=0.8,
            )
    except Exception as exc:
        raise RuntimeError(f"Plot data error: {exc}") from exc
