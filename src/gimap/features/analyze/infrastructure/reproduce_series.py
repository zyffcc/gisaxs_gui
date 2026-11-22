"""Rebuild the grid shown in Series from each contributing raw reduction in its original order."""

from concurrent.futures import CancelledError

from ..application.series_reproduction import series_columns
from ..domain import q_from_two_theta, series_map
from .reproduce import replay_reduction


def replay_series(record, folder, *, should_cancel=None):
    plan = record["reproduction"]
    key = plan["curve"]
    curves = []
    for frame in plan["frames"]:
        if should_cancel and should_cancel():
            raise CancelledError()
        table = replay_reduction(frame, folder)[key]
        x = table["x"]
        if frame["native_x_label"].startswith("2θ"):
            x = q_from_two_theta(x, frame["reproduction"]["geometry"]["wavelength_angstrom"])
        curves.append((x, table["intensity"]))
    series = series_map(curves, [r["label"] for r in record["rows"]], x_label=record["x_label"],
                        curve=key, view=plan["view"])
    return series_columns(series, plan["row"], numbers=[r["frame"] for r in record["rows"]])
