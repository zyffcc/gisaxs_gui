"""The run report as HTML: the model's findings per requested result and the computed tables.

Tables are rendered from the tool results (``RunResults``), never from text the
model wrote, so every number in them is the one the tools computed.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Optional

from ..application import BILLING_SUBSCRIPTION, GOALS, RunOutcome

STATUS_COLORS = {"done": "#2e7d32", "partial": "#ef6c00", "not_available": "#c62828"}
ITEM_TITLES = {
    "peaks": "Peak table",
    "orientation": "Orientation (in-plane vs out-of-plane)",
    "ring_orientation": "Ring orientation distribution",
    "crystallite_size": "Crystallite size",
    "other": "Other",
}
CHINESE = {
    "Peak table": "峰位表",
    "Orientation (in-plane vs out-of-plane)": "取向（面内 vs 面外）",
    "Ring orientation distribution": "单个环的取向分布",
    "Crystallite size": "晶粒尺寸",
    "Other": "其他",
    "done": "完成",
    "partial": "部分",
    "not available": "无法得到",
    "Summary": "摘要",
    "No report": "没有报告",
    "Why:": "原因：",
    "Evidence:": "依据：",
    "Computed results": "计算结果（来自工具）",
    "Caveats": "注意事项",
    "Suggestions": "建议",
    "Missing capabilities noted": "记录的缺失功能",
    "Files written": "写出的文件",
}
"""Fixed headings of a report requested in 中文; tables keep their units and symbols."""


def _e(text) -> str:
    return html.escape(str(text if text is not None else ""))


def _n(value, digits: int = 4) -> str:
    if value is None:
        return "–"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return _e(value)
    if not math.isfinite(value):
        return "–"
    return f"{value:.{digits}g}"


def _pm(value, error, digits: int = 4) -> str:
    if error is None or (isinstance(error, float) and not math.isfinite(error)):
        return _n(value, digits)
    return f"{_n(value, digits)} ± {_n(error, 2)}"


def _table(headers, rows) -> str:
    head = "".join(f"<th align='left'>{_e(item)}</th>" for item in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table cellspacing='0' cellpadding='4' border='1'><tr>{head}</tr>{body}</table>"


def _peak_tables(results) -> str:
    parts = []
    for curve, search in results.peak_searches.items():
        title = f"<h4>Peaks of '{_e(curve)}' (q = {_n(search.x_range[0])}–{_n(search.x_range[1])} Å⁻¹)</h4>"
        if not search.peaks:
            parts.append(title + f"<p>{_e(search.reason)}</p>")
            continue
        rows = [
            [
                str(index + 1), _pm(peak.q, peak.q_err), _pm(peak.d, peak.d_err), _pm(peak.fwhm, peak.fwhm_err),
                _n(peak.height), _n(peak.area), _n(peak.snr, 3), _e(", ".join(peak.flags)),
            ]
            for index, peak in enumerate(search.peaks)
        ]
        hints = "".join(f"<li>{_e(hint)}</li>" for hint in search.series)
        parts.append(
            title
            + _table(["#", "q (Å⁻¹)", "d (Å)", "FWHM (Å⁻¹)", "height", "area", "SNR", "flags"], rows)
            + (f"<ul>{hints}</ul>" if hints else "")
        )
    return "".join(parts)


def _sector_table(results) -> str:
    if not results.sector_rows:
        return ""
    rows = [
        [
            _n(row.q), _pm(row.in_plane, row.in_plane_err, 3), _pm(row.out_of_plane, row.out_of_plane_err, 3),
            _pm(row.ratio, row.ratio_err, 3), _e(row.preference), _e(row.note),
        ]
        for row in results.sector_rows
    ]
    return "<h4>In-plane versus out-of-plane</h4>" + _table(
        ["q (Å⁻¹)", "in-plane (net)", "out-of-plane (net)", "ratio out/in", "preference", "note"], rows
    )


def _ring_tables(results) -> str:
    parts = []
    for ring in results.rings:
        low, high = ring.q_window
        title = f"<h4>Ring q = {_n(low)}–{_n(high)} Å⁻¹</h4>"
        if ring.reason:
            parts.append(title + f"<p>{_e(ring.reason)}</p>")
            continue
        maxima = "; ".join(
            f"|χ| = {_n(item['chi'], 3)}° (FWHM {_n(item['fwhm'], 3)}°)" for item in ring.maxima
        ) or "none above the noise"
        missing = ", ".join(f"{_n(a, 3)}–{_n(b, 3)}°" for a, b in ring.missing) or "none"
        parts.append(title + _table(
            ["texture", "Herman's f", "⟨cos²χ⟩", "maxima", "χ coverage", "unmeasured |χ|"],
            [[_e(ring.texture), _pm(ring.herman, ring.herman_err, 3), _n(ring.cos2, 3), _e(maxima),
              f"{ring.coverage:.0%}", _e(missing)]],
        ))
    return "".join(parts)


def _size_table(results) -> str:
    if not results.sizes:
        return ""
    rows = []
    for size in results.sizes:
        if size.size is None:
            rows.append([_n(size.q), _n(size.fwhm), "–", _e(size.reason)])
            continue
        value = _pm(size.size / 10.0, None if size.size_err is None else size.size_err / 10.0, 3)
        rows.append([_n(size.q), _n(size.fwhm), ("≥ " if size.lower_bound else "") + value, _e(" ".join(size.notes[1:]))])
    return "<h4>Crystallite size (Scherrer, K = 0.9)</h4>" + _table(["q (Å⁻¹)", "FWHM (Å⁻¹)", "L (nm)", "note"], rows)


def _geometry_section(results) -> str:
    parts = []
    used = results.geometry_used
    if used:
        rows = [[
            _e(used.get("source", "")), _n(used.get("distance_mm")),
            f"{_n(used.get('beam_center_x_px'))}, {_n(used.get('beam_center_y_px'))}",
            _n(used.get("wavelength_angstrom")), _n(used.get("incidence_deg")) if used.get("incidence_deg") is not None else "–",
        ]]
        parts.append("<h4>Geometry used for this frame</h4>" + _table(
            ["from", "distance (mm)", "beam centre (px)", "λ (Å)", "αi (°)"], rows,
        ))
    if results.calibrations:
        rows = [
            [
                str(index), _e(item.get("standard", "")), _e(Path(str(item.get("source_image", ""))).name),
                str(item.get("matched_rings", "–")), _n(item.get("rms_residual_px"), 3), _e(item.get("confidence", "")),
                _n(item.get("distance_mm")),
            ]
            for index, item in enumerate(results.calibrations)
        ]
        parts.append("<h4>Calibration fits</h4>" + _table(
            ["#", "standard", "image", "rings", "rms (px)", "confidence", "distance (mm)"], rows,
        ))
    return "".join(parts)


def report_html(
    outcome: RunOutcome,
    *,
    cost: Optional[float] = None,
    elapsed: Optional[float] = None,
    language: str = "English",
) -> str:
    words = CHINESE if language == "中文" else {}

    def t(text: str) -> str:
        return _e(words.get(text, text))

    results = outcome.results
    report = results.report
    parts = []
    if report is not None:
        parts.append(f"<h3>{t('Summary')}</h3><p>{_e(report.summary)}</p>")
        for item in report.items:
            color = STATUS_COLORS.get(item.status, "#616161")
            label = item.status.replace("_", " ")
            parts.append(
                f"<h4>{t(ITEM_TITLES.get(item.item, item.item))} "
                f"<span style='background:{color}; color:white'>&nbsp;{t(label)}&nbsp;</span></h4>"
                f"<p>{_e(item.findings)}</p>"
                + (f"<p><i>{t('Why:')}</i> {_e(item.reason)}</p>" if item.reason else "")
                + (f"<p><small>{t('Evidence:')} {_e(item.evidence)}</small></p>" if item.evidence else "")
            )
    else:
        parts.append(f"<h3>{t('No report')}</h3><p>{_e(outcome.message)}</p>")
    tables = (
        _geometry_section(results) + _peak_tables(results) + _sector_table(results)
        + _ring_tables(results) + _size_table(results)
    )
    if tables:
        parts.append(f"<h3>{t('Computed results')}</h3>" + tables)
    if report is not None and report.caveats:
        parts.append(f"<h3>{t('Caveats')}</h3><ul>" + "".join(f"<li>{_e(text)}</li>" for text in report.caveats) + "</ul>")
    if report is not None and report.suggestions:
        parts.append(
            f"<h3>{t('Suggestions')}</h3><ul>" + "".join(f"<li>{_e(text)}</li>" for text in report.suggestions) + "</ul>"
        )
    if results.feature_requests:
        parts.append(
            f"<h3>{t('Missing capabilities noted')}</h3><ul>"
            + "".join(f"<li>{_e(entry['capability'])}: {_e(entry['reason'])}</li>" for entry in results.feature_requests)
            + "</ul>"
        )
    if results.exports:
        parts.append(f"<h3>{t('Files written')}</h3><ul>" + "".join(f"<li>{_e(path)}</li>" for path in results.exports) + "</ul>")
    usage = outcome.usage
    footer = (
        f"{_e(outcome.model or 'model')} · {usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens:,} "
        f"input and {usage.output_tokens:,} output tokens"
    )
    if outcome.billing == BILLING_SUBSCRIPTION:
        footer += " · your Claude plan" + (f" (≈ ${cost:.2f} at API prices)" if cost is not None else "")
    elif cost is not None:
        footer += f" · ≈ ${cost:.2f}"
    if elapsed is not None:
        footer += f" · {elapsed:.0f} s"
    parts.append(f"<p><small>{footer} · {len(outcome.steps)} tool calls · {_e(outcome.message)}</small></p>")
    return "".join(parts)


def requested_titles(goals) -> list[str]:
    return [GOALS[goal] for goal in goals]


__all__ = ["CHINESE", "ITEM_TITLES", "report_html", "requested_titles"]
