"""Record the actual start of each diffraction peak in a batch, including starts from earlier frames."""

import numpy as np

from src.gimap.shared.reproduction import pack_column

from .batch_fit import FitStarts, fit_peak_inputs, peak_inputs, peak_row


def peak_tables(fits):
    tables = {}
    for target, fit in fits:
        prefix = "peak_" + target.key
        row = peak_row("", [(target, fit)])
        columns = {name.removeprefix(target.name + " "): np.array([value])
                   for name, value in row.items() if isinstance(value, float)}
        columns.update(ok=np.array([int(fit.ok)]), points=np.array([fit.points]),
                       b0=np.array([fit.background[0]]), b1=np.array([fit.background[1]]))
        tables[prefix + "_parameters"] = columns
        if fit.x.size:
            tables[prefix + "_curve"] = {"q": fit.x, "intensity": fit.y, "fit": fit.fitted}
    return tables


def fit_recorded_peaks(outcome, profile, starts):
    inputs = outcome.peak_inputs
    initial = {item.target.key: starts.for_key(item.target.key) if starts else None for item in inputs}
    fits = fit_peak_inputs(inputs, profile, starts)
    recipe = outcome.metadata.get("reproduction")
    if recipe and fits:
        recipe["peak_fits"] = {"profile": profile, "initial": initial}
        for name, columns in peak_tables(fits).items():
            recipe["reference"][name] = {key: pack_column(values) for key, values in columns.items()}
    return fits


def replay_peak_tables(recipe, analysis, maps):
    starts = FitStarts()
    for key, values in recipe["initial"].items():
        if values is not None:
            starts.remember(key, values)
    return peak_tables(fit_peak_inputs(peak_inputs(analysis, maps=maps), recipe["profile"], starts))
