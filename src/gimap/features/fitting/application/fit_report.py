"""The fit report of a manual fit: a page to read or attach (Markdown, HTML or plain text) made from the same record
a save writes (``single_fit.fit_record``): the data file and its SHA-256, the points and q range, the model with its
links and constraints, every parameter with value ± 1σ, its status and range, the goodness of fit, the strongly
correlated pairs, the values at a bound or not determined, and the settings.
"""

from __future__ import annotations

import html
import math
from datetime import datetime, timezone

from ..domain.fit_model import INFO, SIZE_LAWS
from .fit_uncertainty import (
    SHOWN_CORRELATION,
    describe_links,
    interference_label,
    parameter_label,
    parameter_name,
    strong_pairs,
    tail_flags,
    undetermined,
)
from .single_fit import FAMILIES, fit_record

FORMATS = {"markdown": ".md", "html": ".html", "text": ".txt"}
METHODS = {"local": "Refine (bounded least squares from the values in Model)",
           "global": "Search the ranges (differential evolution, then least squares)"}
SIDES = {"mean": "mean of both halves", "both": "both halves on |q|", "positive": "q > 0 half",
         "negative": "q < 0 half"}


def _number(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    value = float(value)
    return f"{value:.{digits}g}" if math.isfinite(value) else ("∞" if value > 0 else "—")


def _status(model, result, path, parameter) -> str:
    if parameter.tie:
        return f"linked: = {parameter.tie}"
    if not parameter.free:
        return "fixed"
    if result is None:
        return "free (not fitted)"
    if path in result.at_bounds:
        return "at a bound of its range"
    if path in undetermined(result):
        return "not determined by the curve"
    return "fitted"


def blocks(model, data, curve, *, side: str, q_range, result, excluded=()) -> list:
    """The report as blocks: ``("h1"|"h2"|"p", text)``, ``("ul", [items])``, ``("table", header, rows)``."""
    record = fit_record(model, data, curve, side=side, q_range=q_range, result=result, excluded=excluded)
    out: list = [("h1", "GIMaP fit report"),
                 ("p", f"{record['software_version']} · written {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")]
    source = (record.get("curve") or {}).get("source") or {}
    points = record["points"]
    items = []
    if record.get("curve"):
        items.append(f"File: {record['curve']['name']} ({record['curve']['path'] or 'not saved'})")
        if source.get("sha256"):
            items.append(f"SHA-256: {source['sha256']}")
        items.append(f"q unit of the file: {record['curve']['q_unit_of_file']} ({record['curve'].get('q_unit_from') or '—'})")
    low, high = (float(data.q.min()), float(data.q.max())) if data.q.size else (math.nan, math.nan)
    items.append(f"Points fitted: {points['count']}, |q| {_number(low)}–{_number(high)} nm⁻¹"
                 + (" (fitting range set)" if points["q_range_nm^-1"] else " (the whole curve)"))
    items.append(f"Made of: {SIDES.get(points.get('halves_used') or side, side)}; points left out by hand: "
                 f"{len(points['left_out_q_nm^-1'])}")
    weights = {"file": ("(I − model)/σ with the σ of the file" if data.sigma is not None else
                        "σ from the file selected, no usable σ. Residual = ln I − ln model"),
               "relative": "relative, (I − model)/I", "poisson": "Poisson, (I − model)/√I",
               "none": "none, ln I − ln model"}
    items.append("Weighting: " + weights[points["weighting_choice"]])
    out += [("h2", "Data"), ("ul", items)]

    items = [record["equation"]]
    resolution = record["resolution"]
    if resolution["mode"] == "file":
        items.append("Gaussian q resolution: dQ from the file at each fitted point (nm⁻¹)")
    else:
        items.append(f"Gaussian q resolution: constant σ_q = {_number(resolution['sigma_q_nm^-1'])} nm⁻¹")
    for index, component in enumerate(model.components, start=1):
        structure = interference_label(component)
        coupling = f", {component.coupling.upper()}" if component.structure else ""
        items.append(f"{index}: {FAMILIES[component.family][0]}; sizes: {SIZE_LAWS[component.law]}; "
                     f"interference: {structure}{coupling}")
        size = record["derived"][index - 1]
        if size["R_number_mean_nm"] is not None:
            items.append(f"Number mean radius {_number(size['R_number_mean_nm'])} nm, "
                         f"relative standard deviation {_number(size['sigma_R_number_rel'])}")
    items.append(f"Spheres: {record['sphere_model']}")
    items.append("Contact: " + ("particles do not overlap (D ≥ 2R, R_HS ≥ R)" if model.contact else "off"))
    links = describe_links(model)
    items.append("Links and constraints: " + ("; ".join(links) if links else "none"))
    out += [("h2", "Model"), ("ul", items)]

    if result is not None:
        chi = "χ²ᵣ" if result.weighting == "sigma" else "var ln I (χ²ᵣ of the relative residuals)"
        state = ("did not improve on the start values" if result.start_kept else
                 "converged" if result.converged else "stopped" if result.stopped else "not converged")
        out += [("h2", "Fit"), ("ul", [
            f"Method: {METHODS.get(result.method, result.method)}; {state} ({result.message})",
            f"{chi} = {_number(result.chi2_reduced)} over {result.points} points with {result.free} free values; "
            f"log RMSE {_number(result.log_rmse)}",
            f"{result.evaluations} evaluations, {result.seconds:.1f} s",
            "Errors: 1σ from the covariance (JᵀJ)⁻¹·s², s² = Σr²/(N − p); linked values by propagation"])]
    else:
        out += [("h2", "Fit"), ("p", "No fit of these values yet: the values are those in Model, without errors.")]

    rows = []
    for path, parameter in model.parameters():
        if path[0] == "globals" and not parameter.free and not parameter.tie and parameter.value == INFO[path[1]].value:
            continue  # a global term at its default and fixed: not part of this model
        error = None if result is None else result.errors.get(path)
        rows.append([parameter_label(model, path), parameter_name(path), _number(parameter.value, 6),
                     "" if error is None else _number(error, 3), _status(model, result, path, parameter),
                     f"{_number(parameter.lower)} – {_number(parameter.upper)}"])
    out += [("h2", "Parameters"), ("table", ["Parameter", "Name", "Value", "± 1σ", "Status", "Range"], rows)]

    if result is not None:
        pairs = strong_pairs(result)
        out += [("h2", f"Correlations (|r| > {SHOWN_CORRELATION:g})"),
                ("ul", [f"{parameter_name(a)} ↔ {parameter_name(b)}: r = {r:+.3f}" for a, b, r in pairs])
                if pairs else ("p", "None: no pair of free values is that strongly correlated.")]
        notes = [f"{parameter_name(path)} stopped at a bound of its range: its error means little" for path in result.at_bounds]
        notes += [f"{parameter_name(path)} is not determined by the curve (error not finite or above 10× the value)"
                  for path in undetermined(result)]
        notes += [f"Beam tail: {parameter_name(path)} " + ("at a bound of its range" if why == "bound" else "not determined")
                  for path, why in tail_flags(result)]
        if notes:
            out += [("h2", "Warnings"), ("ul", notes)]
    return out


def render(parts: list, fmt: str = "markdown") -> str:
    if fmt == "html":
        return _html(parts)
    if fmt == "text":
        return _text(parts)
    return _markdown(parts)


def report(model, data, curve, *, side: str, q_range, result, excluded=(), fmt: str = "markdown") -> str:
    """The report in ``fmt`` (``FORMATS``)."""
    return render(blocks(model, data, curve, side=side, q_range=q_range, result=result, excluded=excluded), fmt)


def _markdown(parts) -> str:
    lines = []
    for part in parts:
        kind = part[0]
        if kind == "h1":
            lines += [f"# {part[1]}", ""]
        elif kind == "h2":
            lines += [f"## {part[1]}", ""]
        elif kind == "p":
            lines += [part[1], ""]
        elif kind == "ul":
            lines += [f"- {item}" for item in part[1]] + [""]
        elif kind == "table":
            header, rows = part[1], part[2]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "---|" * len(header))
            lines += ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows]
            lines.append("")
    return "\n".join(lines)


def _html(parts) -> str:
    body = []
    for part in parts:
        kind = part[0]
        if kind in ("h1", "h2", "p"):
            body.append(f"<{kind}>{html.escape(part[1])}</{kind}>")
        elif kind == "ul":
            body.append("<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in part[1]) + "</ul>")
        elif kind == "table":
            head = "".join(f"<th>{html.escape(cell)}</th>" for cell in part[1])
            rows = "".join("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in part[2])
            body.append(f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>")
    style = ("body{font-family:'Segoe UI',Arial,Helvetica,sans-serif;margin:24px;color:#222}table{border-collapse:collapse}"
             "td,th{border:1px solid #bbb;padding:3px 8px;text-align:left}th{background:#eee}")
    return ("<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>GIMaP fit report</title>"
            f"<style>{style}</style></head><body>\n" + "\n".join(body) + "\n</body></html>\n")


def _text(parts) -> str:
    lines = []
    for part in parts:
        kind = part[0]
        if kind in ("h1", "h2"):
            lines += [part[1], ("=" if kind == "h1" else "-") * len(part[1]), ""]
        elif kind == "p":
            lines += [part[1], ""]
        elif kind == "ul":
            lines += [f"* {item}" for item in part[1]] + [""]
        elif kind == "table":
            table = [part[1]] + part[2]
            widths = [max(len(row[column]) for row in table) for column in range(len(part[1]))]
            for row in table:
                lines.append("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())
            lines.append("")
    return "\n".join(lines)


def suffix(fmt: str) -> str:
    return FORMATS.get(fmt, ".md")


def format_of(path, default: str = "markdown") -> str:
    """The format a file name asks for (.md, .html/.htm, .txt)."""
    name = str(path).lower()
    if name.endswith((".html", ".htm")):
        return "html"
    if name.endswith(".txt"):
        return "text"
    return "markdown" if name.endswith((".md", ".markdown")) else default


__all__ = ["FORMATS", "blocks", "format_of", "render", "report", "suffix"]

