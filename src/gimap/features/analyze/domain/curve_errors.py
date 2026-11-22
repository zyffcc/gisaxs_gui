"""Pixel error inputs and the uncertainty of a reduced mean."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class PixelErrors:
    """Variances in corrected detector units, before binning.

    For Poisson errors, scale is the corrected intensity of one count. Variance
    is the sample's shot variance (by default value × scale). Extra variance is
    the independent contribution from the correction frames. For supplied σ,
    variance is the supplied variance after propagation through the corrections.
    """

    model: str = "poisson"
    scale: np.ndarray | float | None = None
    variance: np.ndarray | float | None = None
    pooled_scatter: bool = True
    poisson_variance: str = "observed"
    extra_variance: np.ndarray | float | None = None

    def subset(self, index) -> PixelErrors:
        """Slice the pixel arrays with the same window or mask as the frame."""
        def selected(value):
            return value[index] if isinstance(value, np.ndarray) and value.ndim else value

        return replace(self, scale=selected(self.scale), variance=selected(self.variance),
                       extra_variance=selected(self.extra_variance))


def pixel_errors(counts) -> PixelErrors:
    """Also accept the original domain API: a counting flag or a map of scales."""
    if isinstance(counts, PixelErrors):
        return counts
    if isinstance(counts, np.ndarray):
        return PixelErrors(scale=counts)
    return PixelErrors(model="poisson" if counts else "scatter")


def _pooled_variance(means, sample_variance, pixels):
    """Huber line through unbiased within-bin pixel variances, weighted by degrees of freedom."""
    measured = pixels >= 2
    x, y = means[measured], sample_variance[measured]
    dof = pixels[measured] - 1
    if not len(x):
        return np.zeros_like(means)
    low, high = x.min(), x.max()
    # For two or three pixels s² has a long right tail. Before robust regression
    # combine nearby intensities by degrees of freedom, so high values from that
    # tail contribute to the mean. A linear variance model is preserved by this
    # averaging. Short curves use smaller groups.
    target_dof = min(24, int(dof.sum() // 8))
    if target_dof >= 2 and np.median(dof) < target_dof:
        order = np.argsort(x, kind="stable")
        x, y, dof = x[order], y[order], dof[order]
        starts, total = [0], 0
        for i in range(len(x) - 1):
            total += dof[i]
            if total >= target_dof:
                starts.append(i + 1)
                total = 0
        grouped_dof = np.add.reduceat(dof, starts)
        x = np.add.reduceat(x * dof, starts) / grouped_dof
        y = np.add.reduceat(y * dof, starts) / grouped_dof
        dof = grouped_dof
    # Centre and scale I to keep the two-parameter regression well conditioned.
    intensity_scale = high - low or 1.0
    x = (x - low) / intensity_scale
    # A median start prevents a noisy bin from setting the residual scale.
    intercept, slope = np.median(y), 0.0
    tolerance = np.finfo(float).eps * max(np.max(y), np.finfo(float).tiny)
    for _ in range(25):
        residual = (y - intercept - slope * x) * np.sqrt(dof)
        mad = np.median(np.abs(residual - np.median(residual)))
        cutoff = max(1.345 * 1.4826 * mad, tolerance, np.finfo(float).tiny)
        weights = dof * np.minimum(1.0, cutoff / np.maximum(np.abs(residual), cutoff))
        centre = np.average(x, weights=weights)
        delta = x - centre
        spread = weights @ (delta * delta)
        new_slope = (weights * delta) @ y / spread if spread > 0 else 0.0
        new_intercept = np.average(y, weights=weights) - new_slope * centre
        if max(abs(new_intercept - intercept), abs(new_slope - slope)) <= tolerance:
            break
        intercept, slope = new_intercept, new_slope
    # Avoid extrapolating the variance through a region sampled only by singleton bins.
    target = np.clip((means - low) / intensity_scale, 0.0, (high - low) / intensity_scale)
    predicted = new_intercept + new_slope * target
    # Texture or an intensity gradient can bend the variance curve. If the line
    # crosses zero, estimate its floor from measured scatter on that side.
    # Clipping to zero would discard these measured bins in weighted fits.
    tail = measured & (predicted <= 0.0)
    floor = np.average(sample_variance[tail], weights=pixels[tail] - 1) if tail.any() else 0.0
    return np.maximum(predicted, floor)


def scatter_sigma(sums, squares, counts, *, pooled=True):
    """Scatter standard error, using the fitted pixel variance below ten pixels by default."""
    sums, squares, counts = (np.asarray(v, dtype=np.float64) for v in (sums, squares, counts))
    means = sums / counts
    variance = np.maximum(squares - means * sums, 0.0) / np.maximum(counts - 1, 1)
    sparse = counts < 10
    if pooled and sparse.any():
        variance[sparse] = _pooled_variance(means, variance, counts)[sparse]
    elif (counts < 2).any():
        typical = np.median(variance[counts >= 2]) if (counts >= 2).any() else 0.0
        variance[counts < 2] = typical
    return np.sqrt(variance / counts)


def _smoothed_shot_variance(shot, scales, breaks):
    """Predict the shot variance from adjacent bins, excluding the observed point."""
    groups = np.cumsum(breaks)
    neighbour_shot = np.zeros_like(shot)
    neighbour_scales = np.zeros_like(scales)
    for offset in (1, 2):
        same = groups[offset:] == groups[:-offset]
        neighbour_shot[offset:] += shot[:-offset] * same
        neighbour_shot[:-offset] += shot[offset:] * same
        neighbour_scales[offset:] += scales[:-offset] * same
        neighbour_scales[:-offset] += scales[offset:] * same
    return scales * np.divide(neighbour_shot, neighbour_scales, out=shot / scales, where=neighbour_scales > 0)


def curve_sigma(errors, pixels, sums, squares, variances, scales, extra, breaks):
    """Standard error for each final bin. A true entry in breaks starts a new contiguous interval."""
    if errors.model == "scatter":
        return scatter_sigma(sums, squares, pixels, pooled=errors.pooled_scatter)
    if errors.model == "sigma":
        if variances is None:
            raise ValueError("A σ frame is required for the supplied error model.")
        return np.sqrt(variances + extra) / pixels
    shot = sums if variances is None else variances
    scales = pixels.astype(float) if scales is None else scales
    if errors.poisson_variance == "smoothed":
        shot = _smoothed_shot_variance(shot, scales, breaks)
    least = (scales / pixels) ** 2
    return np.sqrt(np.maximum(shot, least) + extra) / pixels
