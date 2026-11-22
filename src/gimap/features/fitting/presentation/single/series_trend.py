"""Values and uncertainty bars drawn for a series parameter, grouped by growth stage."""

import math

import numpy as np

from src.gimap.app.presentation.components import stage_color
from src.gimap.app.presentation.components.curve_plot import Marker
from src.gimap.app.presentation.i18n import tr, trf

CHI2 = "chi2"
PAIR = "R+D"
PAIR_NAME = "Radius and spacing (nm)"


def paired_choice(key):
    return isinstance(key, tuple) and key[1] == PAIR


def paired_label(model, owner, translate=True):
    if len(model.components) == 1:
        return tr(PAIR_NAME) if translate else PAIR_NAME
    label = "Particle {n} · radius and spacing (nm)"
    return trf(label, n=owner + 1) if translate else label.format(n=owner + 1)


def trend_series(key, name, groups, positions, *, show_least_squares, pair=False):
    """Each entry is a labelled curve and optional interval ends in the same x order as its points."""
    intervals = {f.index: f.selection.interval(key) for members in groups.values() for f in members
                 if key != CHI2 and f.selection is not None}
    posterior = {i: v for i, v in intervals.items() if len(v) >= 4 and all(
        n is not None and np.isfinite(n) for n in v[:4])}
    curves = []
    for stage, members in groups.items():
        staged = stage is not None and len(groups) > 1
        color = stage_color(stage) if staged else "#2563eb"
        label = trf("stage {n}", n=stage) if staged else name
        prefix = label + " · " if staged else ""
        if pair:
            color = "#2563eb" if key[1] == "R" else "#c16a17"
            prefix = name + " · " + prefix
        predicted = [f for f in members if f.index in posterior]
        if predicted:
            x = np.array([positions[f.index] for f in predicted])
            y, low, high = np.array([posterior[f.index][1:4] for f in predicted]).T
            curves.append((prefix + tr("Posterior median (68%)"), x, y, color, Marker("o", 6), (low, high)))
        other = [f for f in members if f.index not in posterior]
        if other:
            x = np.array([positions[f.index] for f in other])
            if key == CHI2:
                y = np.array([f.result.chi2_reduced for f in other])
                bounds = None
            else:
                y = np.array([f.result.model.get(key).value for f in other])
                error = np.array([f.result.errors.get(key, math.nan) for f in other])
                bounds = y - error, y + error
            legend = prefix + tr("Refined fit (±1σ)") if pair or posterior else label
            marker = Marker("d" if posterior or (pair and key[1] == "D") else "o", 6)
            curves.append((legend, x, y, color, marker, bounds))
    if posterior and show_least_squares:
        indices = sorted(posterior)
        x = np.array([positions[i] for i in indices])
        lsq = np.array([posterior[i][0] for i in indices])
        label = (name + " · " if pair else "") + tr("Predict least squares")
        curves.append((label, x, lsq, "#777777", Marker("s", 4, hollow=True), None))
    return curves, bool(posterior)
